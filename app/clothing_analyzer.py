"""Clothing appearance: dominant upper/lower colours, garment type and a rough pattern hint.

Two backends:

* segformer: a SegFormer human-parsing model fine-tuned on the ATR dataset
  (Hugging Face: mattmdjaga/segformer_b2_clothes). It labels pixels as
  Upper-clothes, Pants, Skirt, Dress, etc., so we get real garment regions
  and garment categories. It does not separate outerwear from other tops.
* person_mask_bands: fallback when that model is not installed. It takes the
  person's *instance* mask and splits it into an upper-body and a lower-body
  band. That is NOT clothing segmentation: no garment types, lower confidence,
  and the output is labelled as an approximation.

Colours are coarse names (blue, black, red...), not exact shades or designs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import ClothingConfig, resolve_path

log = logging.getLogger(__name__)

# Labels expected in the ATR-style human-parsing model.
UPPER_LABELS = ("Upper-clothes",)
LOWER_LABELS = ("Pants", "Skirt")
DRESS_LABELS = ("Dress",)
REQUIRED_LABELS = {"Upper-clothes", "Pants"}


def color_name(bgr: np.ndarray) -> str:
    """Map a BGR colour to a coarse English colour name using HSV rules."""
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 190 else "gray"
    if (h < 20 or h >= 170) and v < 140 and s > 60:
        return "brown"
    if 8 <= h < 33 and s < 90 and v > 150:
        return "beige"
    if h < 8 or h >= 170:
        return "red"
    if h < 20:
        return "orange"
    if h < 33:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"


def dominant_color(pixels: np.ndarray, rng: np.random.Generator | None = None) -> tuple[str, float]:
    """K-means (k=3) over the region's pixels; returns (name of the biggest cluster, its share)."""
    rng = rng or np.random.default_rng(0)
    if len(pixels) > 3000:
        pixels = pixels[rng.choice(len(pixels), 3000, replace=False)]
    data = pixels.astype(np.float32)
    k = min(3, len(data))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    top = int(np.argmax(counts))
    return color_name(centers[top]), float(counts[top] / counts.sum())


def pattern_hint(gray: np.ndarray, region: np.ndarray, min_pixels: int) -> dict | None:
    """Edge density inside a garment region: 'plain', 'patterned' or 'unclear'. Approximate."""
    inner = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8))
    n = int(inner.sum())
    if n < min_pixels:
        return None
    edges = cv2.Canny(gray, 60, 150) > 0
    density = float(edges[inner > 0].mean())
    label = "plain" if density < 0.06 else "patterned" if density > 0.15 else "unclear"
    return {"label": label, "edge_density": round(density, 3)}


def crop_quality(crop: np.ndarray, mask: np.ndarray | None) -> tuple[float, list[str]]:
    """0..1 quality from size, brightness, sharpness and mask fill, plus human-readable notes."""
    notes = []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    size_q = min(1.0, crop.shape[0] / 256)
    if crop.shape[0] < 128:
        notes.append("small crop")
    brightness = float(gray.mean())
    light_q = 1.0 if 50 <= brightness <= 210 else 0.4
    if brightness < 50:
        notes.append("low light")
    elif brightness > 210:
        notes.append("over-exposed")
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_q = min(1.0, sharp / 150)
    if sharp < 40:
        notes.append("blurry")
    fill_q = 1.0
    if mask is not None:
        fill = float(mask.mean())
        fill_q = min(1.0, fill / 0.45)
        if fill < 0.3:
            notes.append("partly occluded")
    return round(size_q * light_q * (0.5 + 0.5 * sharp_q) * fill_q, 3), notes


class SegformerClothingBackend:
    name = "segformer"

    def __init__(self, model_dir: Path, device: str):
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} not found (run: python scripts/download_models.py --clothing)")
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForSemanticSegmentation.from_pretrained(str(model_dir)).to(device).eval()
        self.device = device
        labels = {int(k): v for k, v in self.model.config.id2label.items()}
        missing = REQUIRED_LABELS - set(labels.values())
        if missing:
            raise ValueError(f"model labels {sorted(labels.values())} lack {sorted(missing)}; not a clothing parser")
        self.labels = labels
        self.ids = {name: i for i, name in labels.items()}

    def parse(self, crop: np.ndarray) -> np.ndarray:
        """Per-pixel label ids at crop resolution."""
        torch = self.torch
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            up = torch.nn.functional.interpolate(logits, size=crop.shape[:2], mode="bilinear", align_corners=False)
        return up.argmax(dim=1)[0].cpu().numpy()

    def regions(self, crop: np.ndarray, person_mask: np.ndarray | None) -> dict[str, np.ndarray]:
        labels = self.parse(crop)
        keep = person_mask.astype(bool) if person_mask is not None else np.ones(labels.shape, bool)
        out = {}
        for name in UPPER_LABELS + LOWER_LABELS + DRESS_LABELS:
            if name in self.ids:
                out[name] = (labels == self.ids[name]) & keep
        return out


class ClothingAnalyzer:
    """Picks a backend at start-up and turns a person crop into clothing attributes."""

    def __init__(self, cfg: ClothingConfig, device: str):
        self.cfg = cfg
        self.backend: SegformerClothingBackend | None = None
        self.mode = "off"
        if not cfg.enabled or cfg.backend == "off":
            self.status = "disabled in settings"
            return
        problem = None
        if cfg.backend in ("auto", "segformer"):
            try:
                self.backend = SegformerClothingBackend(resolve_path(cfg.segformer_dir), device)
                self.mode = "segformer"
                self.status = "garment segmentation: SegFormer ATR human parsing"
                return
            except Exception as exc:
                problem = f"{exc.__class__.__name__}: {exc}"
                if cfg.backend == "segformer":
                    log.warning("Clothing segmentation model unavailable: %s", problem)
        self.mode = "person_mask_bands"
        self.status = ("garment segmentation UNAVAILABLE; using approximate colour bands of the person mask"
                       + (f" ({problem})" if problem else ""))

    @property
    def available(self) -> bool:
        return self.mode != "off"

    def analyze(self, crop: np.ndarray, person_mask: np.ndarray | None) -> dict:
        quality, notes = crop_quality(crop, person_mask)
        result = {"backend": self.mode, "quality": quality, "notes": notes, "garments": [],
                  "upper_color": None, "upper_conf": 0.0, "lower_color": None, "lower_conf": 0.0}
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if self.mode == "segformer":
            regions = self.regions_from_model(crop, person_mask, result)
        else:
            regions = self.regions_from_bands(crop, person_mask, result)
        for part in ("upper", "lower"):
            region = regions.get(part)
            if region is None or region.sum() < self.cfg.min_region_pixels:
                continue
            name, share = dominant_color(crop[region])
            result[f"{part}_color"] = name
            result[f"{part}_conf"] = round(share * result["quality"], 3)
        if regions.get("upper") is not None:
            result["pattern_upper"] = pattern_hint(gray, regions["upper"], self.cfg.pattern_min_pixels)
        return result

    def regions_from_model(self, crop, person_mask, result) -> dict[str, np.ndarray]:
        r = self.backend.regions(crop, person_mask)
        result["segmentation"] = "garment"
        sizes = {k: int(v.sum()) for k, v in r.items()}
        upper = r.get("Upper-clothes")
        lower_name = max(LOWER_LABELS, key=lambda n: sizes.get(n, 0))
        lower = r.get(lower_name)
        dress = r.get("Dress")
        if dress is not None and sizes.get("Dress", 0) > max(sizes.get("Upper-clothes", 0), sizes.get(lower_name, 0)):
            result["garments"].append("dress")
            h = crop.shape[0]
            top = np.zeros_like(dress)
            top[: h // 2] = True
            return {"upper": dress & top, "lower": dress & ~top}
        if upper is not None and sizes.get("Upper-clothes", 0) >= self.cfg.min_region_pixels:
            result["garments"].append("upper-clothes")
        if lower is not None and sizes.get(lower_name, 0) >= self.cfg.min_region_pixels:
            result["garments"].append(lower_name.lower())
        result["notes"].append("outerwear is not distinguished by this model")
        return {"upper": upper, "lower": lower}

    def regions_from_bands(self, crop, person_mask, result) -> dict[str, np.ndarray]:
        h, w = crop.shape[:2]
        base = person_mask.astype(bool) if person_mask is not None else np.zeros((h, w), bool)
        if person_mask is None:
            base[:, int(0.25 * w): int(0.75 * w)] = True
            result["notes"].append("no instance mask; used the centre of the box")
        upper = np.zeros((h, w), bool)
        lower = np.zeros((h, w), bool)
        upper[int(0.20 * h): int(0.50 * h)] = True
        lower[int(0.55 * h): int(0.90 * h)] = True
        result["segmentation"] = "approx_bands"
        result["notes"].append("approximate body bands, not garment segmentation")
        result["quality"] = round(result["quality"] * 0.6, 3)
        return {"upper": upper & base, "lower": lower & base}
