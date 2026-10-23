"""Vehicle-person association: which vehicle a person appears to have got out of.

The idea is simple. A person who gets out of a car *first appears* right
next to that car (they were hidden inside it), usually beside a door, while
the car is standing still, and then moves away from it. A person walking
past a car was already being tracked before they reached it, so their
track was not born there and they are never considered.

For every new person track that starts next to one or more vehicles we
watch it for ``observation_seconds`` and then score each candidate vehicle
on five pieces of evidence (weights in settings.yaml):

  birth_overlap       how much of the person's first box lies inside the vehicle's door-expanded box
  door_zone           whether the first foot point is in a side / lower door band of the vehicle box
  outward_motion      whether the person then moved away from the vehicle
  vehicle_stationary  whether the vehicle was standing still when the person appeared
  recent_arrival      whether we saw the vehicle arrive / come to rest not long before

Penalties are applied when the track was born touching the frame border, or
when another person track was lost close by moments earlier (more likely an
occlusion re-appearance than an exit). Births right after the analysis
started or right after a tracker reset are never treated as exits, because
we did not actually observe the person appearing.

If the two best vehicles score within ``ambiguity_margin`` of each other the
result is "ambiguous" and no vehicle is assigned. An association records an
*observation*; it is never treated as proof of ownership or authorisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import AssociationConfig
from app.datatypes import (
    GROUP_PERSON,
    GROUP_VEHICLE,
    box_diag,
    box_foot,
    distance_to_box,
    expand_box,
    overlap_fraction,
    point_in_box,
)
from app.tracker import TrackState, TrackStore

OBSERVED_EXIT = "observed_exit"
POSSIBLE_EXIT = "possible_exit"
AMBIGUOUS = "ambiguous"
NO_ASSOCIATION = "none"

STARTUP_GRACE_SECONDS = 0.5


@dataclass
class AssociationRecord:
    person_id: str
    vehicle_id: str | None
    status: str
    confidence: float
    birth_ts: float
    birth_frame: int
    decided_ts: float
    decided_frame: int
    components: dict[str, float] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)  # vehicle_id, score, box (pixels at birth)
    reasons: list[str] = field(default_factory=list)
    person_box: list[float] | None = None  # the person's first box, for matching against labelled footage

    @property
    def is_linked(self) -> bool:
        return self.vehicle_id is not None and self.status in (OBSERVED_EXIT, POSSIBLE_EXIT)


@dataclass
class _Candidate:
    vehicle_id: str
    vbox: np.ndarray
    birth_overlap: float
    door_zone: float
    vehicle_stationary: float
    recent_arrival: float
    arrival_known: bool


@dataclass
class _Pending:
    birth_ts: float
    birth_frame: int
    candidates: dict[str, _Candidate]
    border: bool
    reappeared_from: str | None
    person_box: list[float] | None = None


def door_zone_score(person_box: np.ndarray, vbox: np.ndarray, margin: float) -> float:
    """Rough image-space test for "standing where a door would be".

    Doors are on the sides of a car. Depending on the camera they show up near
    the left/right ends of the box or along its lower edge. This is an
    approximation; it does not know the real 3D geometry of the car.
    """
    fx, fy = box_foot(person_box)
    w, h = vbox[2] - vbox[0], vbox[3] - vbox[1]
    side = (vbox[0] - margin * w <= fx <= vbox[0] + 0.35 * w) or (vbox[2] - 0.35 * w <= fx <= vbox[2] + margin * w)
    lower_half = vbox[1] + 0.3 * h <= fy <= vbox[3] + 0.35 * h
    bottom_band = vbox[0] <= fx <= vbox[2] and vbox[3] - 0.35 * h <= fy <= vbox[3] + 0.35 * h
    if side and lower_half:
        return 1.0
    if bottom_band:
        return 0.8
    if point_in_box((fx, fy), expand_box(vbox, margin, margin / 2)):
        return 0.4
    return 0.0


class AssociationEngine:
    def __init__(self, cfg: AssociationConfig):
        self.cfg = cfg
        self.records: dict[str, AssociationRecord] = {}
        self._pending: dict[str, _Pending] = {}
        self._arrival_ts: dict[str, float] = {}  # vehicle id -> when we saw it come to rest
        self._was_moving: set[str] = set()
        self._start_ts: float | None = None
        self._reset_ts: float | None = None

    def notify_reset(self, ts: float) -> None:
        """Called after a tracker reset: new tracks right after it are re-births, not exits."""
        self._reset_ts = ts
        self._pending.clear()

    def _in_grace(self, ts: float) -> bool:
        if self._start_ts is not None and ts - self._start_ts <= STARTUP_GRACE_SECONDS:
            return True
        return self._reset_ts is not None and 0 <= ts - self._reset_ts <= STARTUP_GRACE_SECONDS

    def _update_arrivals(self, store: TrackStore) -> None:
        for v in store.visible(GROUP_VEHICLE):
            if store.is_stationary(v.track_id):
                if v.track_id in self._was_moving and v.track_id not in self._arrival_ts:
                    self._arrival_ts[v.track_id] = store.current_ts
            else:
                self._was_moving.add(v.track_id)

    def update(self, store: TrackStore, new_ids: list[str]) -> list[AssociationRecord]:
        """Process one frame. Returns association decisions made on this frame."""
        ts, frame = store.current_ts, store.current_frame
        if self._start_ts is None:
            self._start_ts = ts
        self._update_arrivals(store)

        for pid in new_ids:
            pst = store.tracks.get(pid)
            if pst is None or pst.group != GROUP_PERSON:
                continue
            self._start_pending(pid, pst, store)

        decided = []
        for pid, pend in list(self._pending.items()):
            pst = store.tracks.get(pid)
            elapsed = ts - pend.birth_ts
            if pst is not None and pst.active and elapsed < self.cfg.observation_seconds:
                continue
            del self._pending[pid]
            rec = self._decide(pid, pend, pst, ts, frame)
            self.records[pid] = rec
            decided.append(rec)
        return decided

    def _start_pending(self, pid: str, pst: TrackState, store: TrackStore) -> None:
        ts = store.current_ts
        if self._in_grace(ts):
            self.records[pid] = AssociationRecord(
                pid, None, NO_ASSOCIATION, 0.0, ts, store.current_frame, ts, store.current_frame,
                reasons=["already present when analysis started or tracking was reset; exit not observed"],
            )
            return
        cands: dict[str, _Candidate] = {}
        for v in store.visible(GROUP_VEHICLE):
            exp = expand_box(v.box, self.cfg.door_margin, self.cfg.door_margin / 2)
            ov = overlap_fraction(pst.box, exp)
            if ov < self.cfg.birth_overlap_min:
                continue
            arrival = self._arrival_ts.get(v.track_id)
            arrival_known = arrival is not None
            recent = 1.0 if arrival_known and ts - arrival <= self.cfg.recent_arrival_seconds else 0.5
            if arrival_known and ts - arrival > self.cfg.recent_arrival_seconds:
                recent = 0.3
            cands[v.track_id] = _Candidate(
                vehicle_id=v.track_id,
                vbox=v.box.copy(),
                birth_overlap=min(1.0, ov),
                door_zone=door_zone_score(pst.box, v.box, self.cfg.door_margin),
                vehicle_stationary=1.0 if store.is_stationary(v.track_id) else 0.2,
                recent_arrival=recent,
                arrival_known=arrival_known,
            )
        if not cands:
            self.records[pid] = AssociationRecord(
                pid, None, NO_ASSOCIATION, 0.0, ts, store.current_frame, ts, store.current_frame,
                reasons=["did not first appear next to a vehicle"],
            )
            return
        self._pending[pid] = _Pending(ts, store.current_frame, cands, pst.born_at_border, self._reappearance(pst, store),
                                      [round(float(v), 1) for v in pst.box])

    def _reappearance(self, pst: TrackState, store: TrackStore) -> str | None:
        """Another person lost nearby moments ago? Then this is probably them re-appearing."""
        foot = np.array(box_foot(pst.box))
        for other in store.recently_lost(GROUP_PERSON, self.cfg.reappear_seconds):
            if other.track_id == pst.track_id:
                continue
            if np.linalg.norm(foot - np.array(box_foot(other.box))) <= box_diag(pst.box):
                return other.track_id
        return None

    def _outward_motion(self, pst: TrackState | None, cand: _Candidate, birth_ts: float) -> tuple[float, int]:
        if pst is None:
            return 0.0, 0
        obs = [h for h in pst.history if birth_ts <= h[0] <= birth_ts + self.cfg.observation_seconds]
        if len(obs) < 2:
            return 0.0, len(obs)
        diag = max(box_diag(cand.vbox), 1.0)
        dists = [distance_to_box(box_foot(b), cand.vbox) / diag for _, _, b in obs]
        gain = dists[-1] - dists[0]
        steps = np.diff(dists)
        steady = float(np.mean(steps >= -0.01)) if len(steps) else 0.0
        return float(np.clip(gain / 0.15, 0.0, 1.0) * steady), len(obs)

    def _decide(self, pid: str, pend: _Pending, pst: TrackState | None, ts: float, frame: int) -> AssociationRecord:
        w = self.cfg.weights
        scored = []
        n_obs = 0
        for cand in pend.candidates.values():
            outward, n_obs = self._outward_motion(pst, cand, pend.birth_ts)
            comps = {
                "birth_overlap": cand.birth_overlap,
                "door_zone": cand.door_zone,
                "outward_motion": outward,
                "vehicle_stationary": cand.vehicle_stationary,
                "recent_arrival": cand.recent_arrival,
            }
            total = sum(w[k] * v for k, v in comps.items())
            if pend.border:
                total *= self.cfg.border_penalty
            if pend.reappeared_from:
                total *= 1.0 - self.cfg.reappear_penalty
            scored.append((total, cand, comps))
        scored.sort(key=lambda s: s[0], reverse=True)
        candidates = [{"vehicle_id": c.vehicle_id, "score": round(t, 3), "box": [round(float(v), 1) for v in c.vbox]}
                      for t, c, _ in scored]
        best_total, best, best_comps = scored[0]
        reasons = []
        if pend.border:
            reasons.append("track started at the frame border (penalised)")
        if pend.reappeared_from:
            reasons.append(f"{pend.reappeared_from} was lost nearby just before (possible re-appearance, penalised)")

        def record(vehicle_id, status, conf) -> AssociationRecord:
            return AssociationRecord(
                pid, vehicle_id, status, round(conf, 3), pend.birth_ts, pend.birth_frame, ts, frame,
                {k: round(v, 3) for k, v in best_comps.items()}, candidates, reasons, pend.person_box,
            )

        if n_obs < self.cfg.min_observations:
            reasons.append(f"only {n_obs} observations in the observation window")
            return record(None, NO_ASSOCIATION, best_total)
        second_total = scored[1][0] if len(scored) > 1 else 0.0
        if (second_total >= self.cfg.candidate_threshold
                and best_total - second_total < self.cfg.ambiguity_margin):
            reasons.append(
                f"{best.vehicle_id} ({best_total:.2f}) and {scored[1][1].vehicle_id} ({second_total:.2f}) "
                "are too close to call"
            )
            return record(None, AMBIGUOUS, best_total)
        if best_total >= self.cfg.confirm_threshold:
            reasons.append(f"first appeared beside {best.vehicle_id} and the evidence score is {best_total:.2f}")
            return record(best.vehicle_id, OBSERVED_EXIT, best_total)
        if pend.reappeared_from:
            # Weak evidence plus a nearby lost track is far more likely an occlusion than an exit.
            reasons.append(f"evidence score {best_total:.2f} too weak to outweigh a likely re-appearance")
            return record(None, NO_ASSOCIATION, best_total)
        if best_total >= self.cfg.candidate_threshold:
            reasons.append(f"weak evidence ({best_total:.2f}) of exiting {best.vehicle_id}")
            return record(best.vehicle_id, POSSIBLE_EXIT, best_total)
        reasons.append(f"evidence score {best_total:.2f} below threshold")
        return record(None, NO_ASSOCIATION, best_total)

    def get(self, person_id: str) -> AssociationRecord | None:
        return self.records.get(person_id)

    def persons_for_vehicle(self, vehicle_id: str) -> list[AssociationRecord]:
        return [r for r in self.records.values() if r.vehicle_id == vehicle_id and r.is_linked]

    def is_pending(self, person_id: str) -> bool:
        return person_id in self._pending
