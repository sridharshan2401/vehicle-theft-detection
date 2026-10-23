"""Parking-zone configuration and per-vehicle parking state.

Zones are polygons stored in a small JSON file with coordinates normalised to
0..1, so the same file works if the video is re-encoded at another
resolution (as long as the camera view is the same).

Zone kinds:
  parking     where vehicles are expected to park
  restricted  areas people should not be in (raises a LOW-weight signal)
  proximity   optional fixed "close to the vehicles" area (informational)
  exit        the way out of the car park; a moving vehicle entering it is a departure

Everything here works in image space. Distances in pixels are not metres:
objects far from the camera look smaller and move fewer pixels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.config import ZonesConfig
from app.datatypes import (
    GROUP_VEHICLE,
    Event,
    EventKind,
    box_area,
    box_center,
    box_diag,
    box_foot,
    box_iou,
    touches_border,
)
from app.tracker import TrackStore

log = logging.getLogger(__name__)

ZONE_KINDS = ("parking", "restricted", "proximity", "exit")


class ZoneError(ValueError):
    pass


@dataclass
class Zone:
    name: str
    kind: str
    points: list[list[float]]  # normalised [x, y] pairs
    vehicle_record: str | None = None  # optional link to a registered vehicle's spot

    def scaled(self, width: int, height: int) -> np.ndarray:
        return np.array(self.points, dtype=np.float32) * np.array([width, height], dtype=np.float32)

    def contains(self, pt: tuple[float, float], width: int, height: int) -> bool:
        return point_in_polygon(pt, self.scaled(width, height))


def point_in_polygon(pt: tuple[float, float], poly: np.ndarray) -> bool:
    """Even-odd ray casting test."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


@dataclass
class ZoneSet:
    zones: list[Zone] = field(default_factory=list)

    def validate(self) -> "ZoneSet":
        names = set()
        for z in self.zones:
            if z.kind not in ZONE_KINDS:
                raise ZoneError(f"Zone {z.name!r}: kind must be one of {ZONE_KINDS}")
            if len(z.points) < 3:
                raise ZoneError(f"Zone {z.name!r}: a polygon needs at least 3 points")
            for p in z.points:
                if len(p) != 2 or not all(0.0 <= c <= 1.0 for c in p):
                    raise ZoneError(f"Zone {z.name!r}: points must be [x, y] pairs in 0..1, got {p}")
            if z.name in names:
                raise ZoneError(f"Duplicate zone name {z.name!r}")
            names.add(z.name)
        return self

    def of_kind(self, kind: str) -> list[Zone]:
        return [z for z in self.zones if z.kind == kind]

    def find(self, pt: tuple[float, float], kind: str, width: int, height: int) -> Zone | None:
        for z in self.of_kind(kind):
            if z.contains(pt, width, height):
                return z
        return None

    @classmethod
    def load(cls, path: str | Path) -> "ZoneSet":
        p = Path(path)
        if not p.exists():
            log.info("No zone file at %s; the whole frame is treated as one parking area.", p)
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            zones = [Zone(**z) for z in data.get("zones", [])]
        except (json.JSONDecodeError, TypeError) as exc:
            raise ZoneError(f"Could not read zone file {p}: {exc}") from exc
        return cls(zones).validate()

    def save(self, path: str | Path) -> None:
        self.validate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "normalised-polygons-v1", "zones": [asdict(z) for z in self.zones]}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


IMPLICIT_ZONE = "frame"

# Vehicle states
ARRIVING = "arriving"
PARKED = "parked"
MOVED = "moved"
LEFT_ZONE = "left_zone"
DEPARTED = "departed"
RELINKED = "relinked"  # the same car came back under a newer track id


@dataclass
class VehicleZoneState:
    state: str = ARRIVING
    zone: str | None = None
    still_since: float | None = None
    parked_box: np.ndarray | None = None
    parked_zone: str | None = None
    parked_ts: float | None = None
    canonical_id: str | None = None
    move_since: float | None = None  # when a not-yet-confirmed movement started


class VehicleZoneTracker:
    """Turns vehicle tracks into parked / moved / left-zone / departed events.

    It also re-links a vehicle that disappears and comes back in the same
    parking spot under a new track id (after occlusion or a tracker reset),
    so risk history stays attached to the same car.
    """

    def __init__(self, cfg: ZonesConfig, zones: ZoneSet, frame_size: tuple[int, int]):
        self.cfg = cfg
        self.zones = zones
        self.width, self.height = frame_size
        self.states: dict[str, VehicleZoneState] = {}

    @property
    def has_parking_zones(self) -> bool:
        return bool(self.zones.of_kind("parking"))

    def canonical(self, vehicle_id: str) -> str:
        st = self.states.get(vehicle_id)
        return st.canonical_id if st and st.canonical_id else vehicle_id

    def parked_ids(self) -> set[str]:
        return {vid for vid, st in self.states.items() if st.state == PARKED}

    def state_of(self, vehicle_id: str) -> str | None:
        st = self.states.get(vehicle_id)
        return st.state if st else None

    def _parking_zone_at(self, pt: tuple[float, float]) -> str | None:
        if not self.has_parking_zones:
            return IMPLICIT_ZONE
        z = self.zones.find(pt, "parking", self.width, self.height)
        return z.name if z else None

    def _try_relink(self, vid: str, st: VehicleZoneState, box: np.ndarray, store: TrackStore) -> None:
        """If a parked car vanished and a new track appears on its spot, treat it as the same car."""
        visible = {t.track_id for t in store.visible(GROUP_VEHICLE)}
        for other_id, other in self.states.items():
            if other_id == vid or other_id in visible or other.state != PARKED or other.parked_box is None:
                continue
            if box_iou(box, other.parked_box) >= 0.6:
                st.canonical_id = self.canonical(other_id)
                st.state, st.parked_box, st.parked_zone, st.parked_ts = PARKED, other.parked_box, other.parked_zone, other.parked_ts
                other.state = RELINKED
                log.info("%s re-linked to parked vehicle %s", vid, st.canonical_id)
                return

    def update(self, store: TrackStore, lost_ids: list[str]) -> list[Event]:
        ts, frame = store.current_ts, store.current_frame
        events: list[Event] = []
        for v in store.visible(GROUP_VEHICLE):
            vid = v.track_id
            st = self.states.get(vid)
            if st is None:
                st = self.states[vid] = VehicleZoneState()
                self._try_relink(vid, st, v.box, store)
            foot = box_foot(v.box)
            zone = self._parking_zone_at(foot)
            st.zone = zone
            stationary = store.is_stationary(vid)
            cid = self.canonical(vid)

            def emit(kind: str, conf: float = 1.0, **details) -> None:
                events.append(Event(kind, ts, frame, vehicle_id=cid, confidence=conf,
                                    details={"track_id": vid, "zone": zone, **details}))

            if st.state in (ARRIVING, MOVED, LEFT_ZONE):
                if stationary and zone is not None:
                    st.still_since = st.still_since if st.still_since is not None else ts
                    if ts - st.still_since >= self.cfg.park_seconds:
                        st.state, st.parked_box, st.parked_zone, st.parked_ts = PARKED, v.box.copy(), zone, ts
                        emit(EventKind.VEHICLE_PARKED)
                        continue
                else:
                    st.still_since = None
            if st.state == PARKED:
                (x0, y0), (x1, y1) = box_center(st.parked_box), box_center(v.box)
                moved = np.hypot(x1 - x0, y1 - y0) / max(box_diag(st.parked_box), 1.0)
                ratio = box_area(v.box) / max(box_area(st.parked_box), 1.0)
                same_size = 1 - self.cfg.max_area_change <= ratio <= 1 / (1 - self.cfg.max_area_change)
                credible = same_size or moved >= 2 * self.cfg.move_threshold or touches_border(v.box, self.width, self.height)
                if moved >= self.cfg.move_threshold and credible:
                    st.move_since = st.move_since if st.move_since is not None else ts
                    if ts - st.move_since >= self.cfg.move_confirm_seconds:
                        st.state, st.still_since = MOVED, None
                        emit(EventKind.VEHICLE_MOVED, displacement=round(float(moved), 3), parked_zone=st.parked_zone,
                             moving_since=round(st.move_since, 2))
                else:
                    st.move_since = None
            if st.state == MOVED and self.has_parking_zones and zone != st.parked_zone:
                st.state = LEFT_ZONE
                emit(EventKind.VEHICLE_LEFT_ZONE, parked_zone=st.parked_zone)
            if st.state in (MOVED, LEFT_ZONE) and self.zones.find(foot, "exit", self.width, self.height):
                st.state = DEPARTED
                emit(EventKind.VEHICLE_DEPARTED, reason="entered exit zone")

        for vid in lost_ids:
            st = self.states.get(vid)
            if st and st.state == PARKED and st.move_since is not None:
                # It started moving and vanished (e.g. left the frame) before the move was confirmed.
                st.state = MOVED
                events.append(Event(EventKind.VEHICLE_MOVED, ts, frame, vehicle_id=self.canonical(vid), confidence=0.6,
                                    details={"track_id": vid, "reason": "lost while starting to move",
                                             "parked_zone": st.parked_zone, "moving_since": round(st.move_since, 2)}))
            if st and st.state in (MOVED, LEFT_ZONE):
                st.state = DEPARTED
                events.append(Event(EventKind.VEHICLE_DEPARTED, ts, frame, vehicle_id=self.canonical(vid), confidence=0.6,
                                    details={"track_id": vid, "reason": "track lost while moving"}))
        return events
