"""Long-term person re-identification.

The tracker gives each continuous sighting a track id (P001, P002, ...).
When someone leaves the frame and comes back they get a new track id. This
module tries to reconnect the new track to the earlier person using
appearance embeddings, nudged by clothing colours, and never by clothing
alone.

Backends, best first:
  torchreid    OSNet trained for pedestrian ReID (needs the torchreid package
               and a weight file, see docs/MODEL_REQUIREMENTS.md)
  torchvision  ImageNet ResNet-50 features. Generic, not trained for ReID, so
               it uses stricter thresholds and is not used for authorisation.
  histogram    colour histograms of the person mask. Weakest; last resort.

A reconnection is a probabilistic guess with a score. The raw track id is
always kept; the reconnection is stored separately as an identity link.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import ReIDConfig, resolve_path

log = logging.getLogger(__name__)

PENDING = "pending"
NEW = "new"
RECONNECTED = "reconnected"
CANDIDATE = "candidate"  # a plausible earlier identity, still collecting evidence
UNCERTAIN = "uncertain"  # gave up: kept as a new profile but noted the closest earlier one


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class ReIDBackend:
    name = "base"
    person_specific = False
    description = ""

    def embed(self, crops: list[np.ndarray], masks: list[np.ndarray | None]) -> np.ndarray:
        raise NotImplementedError


class TorchreidBackend(ReIDBackend):
    name = "torchreid"
    person_specific = True

    def __init__(self, model_name: str, weights: Path, device: str):
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError:
            from torchreid.reid.utils import FeatureExtractor  # newer package layout
        if not weights.exists():
            raise FileNotFoundError(f"ReID weights not found: {weights}")
        self.extractor = FeatureExtractor(model_name=model_name, model_path=str(weights), device=device, verbose=False)
        self.description = f"{model_name} pedestrian ReID ({weights.name})"

    def embed(self, crops, masks):
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        feats = self.extractor(rgb).detach().cpu().numpy()
        return _l2norm(feats.astype(np.float32))


class TorchvisionBackend(ReIDBackend):
    name = "torchvision"
    person_specific = False

    def __init__(self, device: str):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
        if not cached.exists():
            raise FileNotFoundError(
                f"ImageNet ResNet-50 weights are not cached ({cached}). "
                "Run: python scripts/download_models.py --reid-generic"
            )
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(device)
        self.device = device
        self.torch = torch
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.description = "ImageNet ResNet-50 features (generic, not ReID-trained)"

    def embed(self, crops, masks):
        torch = self.torch
        batch = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (128, 256)) for c in crops])
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std
            feats = self.model(x).cpu().numpy()
        return _l2norm(feats.astype(np.float32))


class HistogramBackend(ReIDBackend):
    """Hue/saturation histograms over three horizontal stripes of the person mask."""

    name = "histogram"
    person_specific = False
    description = "HSV colour histograms (weak, appearance-only fallback)"

    def embed(self, crops, masks):
        out = []
        for crop, mask in zip(crops, masks):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h = hsv.shape[0]
            m = (mask.astype(np.uint8) if mask is not None else np.ones(hsv.shape[:2], np.uint8))
            parts = []
            for y0, y1 in ((0.15, 0.45), (0.45, 0.75), (0.75, 1.0)):
                sl = slice(int(y0 * h), max(int(y1 * h), int(y0 * h) + 1))
                hist = cv2.calcHist([hsv[sl]], [0, 1], m[sl], [16, 8], [0, 180, 0, 256]).flatten()
                parts.append(hist / max(hist.sum(), 1.0))
            out.append(np.sqrt(np.concatenate(parts)))  # Hellinger-style so cosine behaves
        return _l2norm(np.array(out, dtype=np.float32))


def build_reid_backend(cfg: ReIDConfig, device: str) -> tuple[ReIDBackend | None, str]:
    """Create the best available backend. Returns (backend or None, human-readable status)."""
    if not cfg.enabled:
        return None, "disabled in settings"
    order = ["torchreid", "torchvision", "histogram"] if cfg.backend == "auto" else [cfg.backend]
    problems = []
    for name in order:
        try:
            if name == "torchreid":
                backend = TorchreidBackend(cfg.torchreid_model, resolve_path(cfg.torchreid_weights), device)
            elif name == "torchvision":
                backend = TorchvisionBackend(device)
            else:
                backend = HistogramBackend()
            status = f"{backend.name}: {backend.description}"
            if problems:
                status += " | skipped: " + "; ".join(problems)
            return backend, status
        except Exception as exc:  # missing package / weights; try the next one
            problems.append(f"{name} unavailable ({exc.__class__.__name__}: {exc})")
    return None, "unavailable: " + "; ".join(problems)


@dataclass
class IdentityDecision:
    track_id: str
    profile_id: str | None
    status: str
    score: float | None
    ts: float
    frame: int
    candidate_id: str | None = None


@dataclass
class Profile:
    profile_id: str
    first_ts: float
    last_ts: float
    embeddings: deque
    track_ids: set[str] = field(default_factory=set)
    clothing: dict | None = None
    authorized: dict | None = None  # {"record_id", "label", "similarity"} from the registry


def clothing_similarity(a: dict | None, b: dict | None) -> float | None:
    """0..1 agreement of upper/lower colours, or None if either side is unknown."""
    if not a or not b:
        return None
    scores = []
    for part in ("upper", "lower"):
        ca, cb = a.get(f"{part}_color"), b.get(f"{part}_color")
        if ca and cb:
            neutral = {"black", "gray", "white"}
            scores.append(1.0 if ca == cb else 0.5 if ca in neutral and cb in neutral else 0.0)
    return float(np.mean(scores)) if scores else None


class PersonGallery:
    """Profiles of people seen so far and the logic to reconnect returning tracks.

    Thread-safe: the secondary worker adds samples while the main worker reads identities.
    """

    def __init__(self, cfg: ReIDConfig, backend_name: str):
        self.cfg = cfg
        self.backend_name = backend_name
        th = cfg.thresholds.get(backend_name, {"match": 0.9, "candidate": 0.8})
        self.match_th, self.candidate_th = th["match"], th["candidate"]
        self.profiles: dict[str, Profile] = {}
        self.track_to_profile: dict[str, str] = {}
        self.decisions: dict[str, IdentityDecision] = {}
        self._pending: dict[str, list[np.ndarray]] = {}
        self._pending_clothing: dict[str, dict] = {}
        self._last_seen: dict[str, float] = {}
        self._cooccur: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    def note_visible(self, track_ids: list[str], ts: float) -> None:
        """Record which person tracks are on screen together; they cannot be the same person."""
        with self._lock:
            visible = set(track_ids)
            for t in track_ids:
                self._last_seen[t] = ts
                self._cooccur.setdefault(t, set()).update(visible - {t})
                pid = self.track_to_profile.get(t)
                if pid:
                    self.profiles[pid].last_ts = ts

    def _compatible(self, track_id: str, prof: Profile, ts: float) -> bool:
        partners = self._cooccur.get(track_id, set())
        for t in prof.track_ids:
            if t in partners:
                return False  # seen at the same time as this track
            if ts - self._last_seen.get(t, -1e9) <= self.cfg.active_seconds:
                return False  # that profile is still on screen under another track
        return True

    def _score(self, query: np.ndarray, prof: Profile, clothing: dict | None) -> float:
        sims = np.sort(np.stack(prof.embeddings) @ query)[::-1]
        score = float(np.mean(sims[:3]))
        cs = clothing_similarity(clothing, prof.clothing)
        if cs is not None:
            quality = min((clothing or {}).get("quality", 0.0), (prof.clothing or {}).get("quality", 0.0))
            score += self.cfg.clothing_weight * (cs - 0.5) * 2 * quality
        return score

    def _new_profile(self, track_id: str, embs: list[np.ndarray], ts: float) -> Profile:
        prof = Profile(track_id, ts, ts, deque(embs, maxlen=self.cfg.max_embeddings_per_profile), {track_id},
                       self._pending_clothing.pop(track_id, None))
        self.profiles[track_id] = prof
        self.track_to_profile[track_id] = track_id
        return prof

    def add_sample(self, track_id: str, emb: np.ndarray, ts: float, frame: int) -> IdentityDecision | None:
        """Add one embedding. Returns a decision when the identity status of the track changes."""
        with self._lock:
            pid = self.track_to_profile.get(track_id)
            if pid is not None:
                self.profiles[pid].embeddings.append(emb)
                return None
            samples = self._pending.setdefault(track_id, [])
            samples.append(emb)
            if len(samples) < self.cfg.min_samples_for_decision:
                self.decisions[track_id] = IdentityDecision(track_id, None, PENDING, None, ts, frame)
                return None
            query = _l2norm(np.mean(samples, axis=0))
            clothing = self._pending_clothing.get(track_id)
            scored = sorted(
                ((self._score(query, p, clothing), p.profile_id) for p in self.profiles.values()
                 if self._compatible(track_id, p, ts)),
                reverse=True,
            )
            best, best_id = scored[0] if scored else (None, None)
            second = scored[1][0] if len(scored) > 1 else -1.0
            exhausted = len(samples) >= self.cfg.max_pending_samples

            if best is not None and best >= self.match_th and best - second >= self.cfg.decision_margin:
                prof = self.profiles[best_id]
                prof.track_ids.add(track_id)
                prof.embeddings.extend(samples)
                if clothing:
                    prof.clothing = clothing
                self.track_to_profile[track_id] = best_id
                self._pending.pop(track_id, None)
                self._pending_clothing.pop(track_id, None)
                dec = IdentityDecision(track_id, best_id, RECONNECTED, round(best, 3), ts, frame)
            elif best is not None and best >= self.candidate_th and not exhausted:
                dec = IdentityDecision(track_id, None, CANDIDATE, round(best, 3), ts, frame, candidate_id=best_id)
                prev = self.decisions.get(track_id)
                self.decisions[track_id] = dec
                return dec if prev is None or prev.status != CANDIDATE or prev.candidate_id != best_id else None
            else:
                self._new_profile(track_id, samples, ts)
                self._pending.pop(track_id, None)
                status = UNCERTAIN if best is not None and best >= self.candidate_th else NEW
                dec = IdentityDecision(track_id, track_id, status, None if best is None else round(best, 3), ts, frame,
                                       candidate_id=best_id if status == UNCERTAIN else None)
            self.decisions[track_id] = dec
            return dec

    def set_clothing(self, track_id: str, attrs: dict) -> None:
        with self._lock:
            pid = self.track_to_profile.get(track_id)
            if pid is not None:
                self.profiles[pid].clothing = attrs
            else:
                self._pending_clothing[track_id] = attrs

    def identity(self, track_id: str) -> IdentityDecision | None:
        with self._lock:
            return self.decisions.get(track_id)

    def profile_for(self, track_id: str) -> Profile | None:
        with self._lock:
            pid = self.track_to_profile.get(track_id)
            return self.profiles.get(pid) if pid else None

    def tracks_of_profile(self, profile_id: str) -> set[str]:
        with self._lock:
            prof = self.profiles.get(profile_id)
            return set(prof.track_ids) if prof else set()

    def export(self) -> dict[str, np.ndarray]:
        """Mean embedding per profile, for saving at the end of a session."""
        with self._lock:
            return {pid: _l2norm(np.mean(np.stack(p.embeddings), axis=0)) for pid, p in self.profiles.items() if p.embeddings}


class AuthorizationIndex:
    """Embeddings of registered authorised people, compared against profiles.

    Only used with a person-specific ReID backend unless explicitly allowed,
    because generic features are not reliable enough to grant authorisation.
    """

    def __init__(self, people: list[dict], backend: ReIDBackend | None, cfg: ReIDConfig):
        self.entries: list[tuple[int, str, np.ndarray]] = []
        self.status = "no authorised people registered"
        if backend is None:
            self.status = "ReID unavailable; authorisation matching off"
            return
        if not backend.person_specific and not cfg.allow_generic_backend_for_authorization:
            self.status = f"'{backend.name}' backend is not person-specific; authorisation matching off"
            return
        skipped = 0
        for p in people:
            path = p.get("embedding_path")
            if p.get("reid_backend") != backend.name or not path or not Path(path).exists():
                skipped += 1
                continue
            emb = np.load(path).astype(np.float32)
            self.entries.append((int(p["person_record_id"]), p["label"], _l2norm(emb.reshape(-1, emb.shape[-1]))))
        if self.entries:
            self.status = f"{len(self.entries)} authorised people loaded"
        if skipped:
            self.status += f" ({skipped} skipped: registered with another backend or file missing)"
        self.threshold = cfg.authorization_threshold

    def match(self, embeddings: list[np.ndarray]) -> dict | None:
        if not self.entries or not embeddings:
            return None
        q = _l2norm(np.mean(np.stack(embeddings), axis=0))
        best = None
        for rid, label, embs in self.entries:
            sim = float(np.max(embs @ q))
            if sim >= self.threshold and (best is None or sim > best["similarity"]):
                best = {"record_id": rid, "label": label, "similarity": round(sim, 3)}
        return best
