"""Person-vehicle interaction analysis.

Produces observations, not judgements:

  proximity             a person lingered (not just walked past) near a parked vehicle
  repeated_interaction  the same person came back to the same vehicle N times
  extended_presence     a person stayed near a vehicle for a long time
  apparent_entry        a person's track ended while they overlapped the vehicle and
                        nobody re-appeared there within a few seconds (or the vehicle
                        then moved); they may still just be hidden behind it
  restricted_area       a person stood inside a configured restricted zone

The risk engine decides how much each observation matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.config import BehaviorConfig
from app.datatypes import (
    GROUP_PERSON,
    GROUP_VEHICLE,
    Event,
    EventKind,
    box_foot,
    expand_box,
    overlap_fraction,
    point_in_box,
    touches_border,
)
from app.parking_zones import DEPARTED, LEFT_ZONE, MOVED, VehicleZoneTracker, ZoneSet
from app.tracker import TrackStore, TrackState


@dataclass
class Interaction:
    in_zone: bool = False
    episode_start: float | None = None
    last_leave: float | None = None
    approaches: int = 0
    total_seconds: float = 0.0
    last_ts: float | None = None
    loiter_reported: bool = False
    repeated_reported: bool = False
    last_contact_ts: float | None = None
    linger_seconds: float = 0.0  # slow-moving time in the zone during the current episode
    proximity_reported: bool = False


@dataclass
class Entrant:
    person_id: str
    ts: float
    confidence: float
    confirmed: bool = True


@dataclass
class _PendingEntry:
    event: Event  # the apparent_entry to emit once confirmed (timestamped when the person vanished)
    track_box: np.ndarray
    declared_ts: float  # when the track was declared lost and the wait started
    vehicle_track_id: str


class BehaviorAnalyzer:
    def __init__(self, cfg: BehaviorConfig, zones: ZoneSet, frame_size: tuple[int, int]):
        self.cfg = cfg
        self.zones = zones
        self.width, self.height = frame_size
        self.interactions: dict[tuple[str, str], Interaction] = {}
        self.entrants: dict[str, list[Entrant]] = {}  # canonical vehicle id -> people who appeared to get in
        self._pending_entries: list[_PendingEntry] = []
        self.retracted_entries: list[dict] = []  # entries withdrawn because the person re-appeared
        self._restricted_seen: set[tuple[str, str]] = set()

    def proximity_box(self, vbox):
        m = self.cfg.proximity_margin
        return expand_box(vbox, m, m / 2)

    def recent_entrants(self, vehicle_id: str, now: float) -> list[Entrant]:
        """Confirmed entrants, plus people whose entry is still waiting for confirmation."""
        out = [e for e in self.entrants.get(vehicle_id, []) if now - e.ts <= self.cfg.entrant_memory_seconds]
        out += [Entrant(p.event.person_id, p.event.ts, p.event.confidence, confirmed=False)
                for p in self._pending_entries if p.event.vehicle_id == vehicle_id]
        return out

    def update(self, store: TrackStore, vzt: VehicleZoneTracker, lost_ids: list[str],
               hold=None) -> list[Event]:
        """``hold(person_id) -> bool``: while True, proximity for that person is not reported yet
        (used while the association engine is still deciding whether they just got out of the car)."""
        ts, frame = store.current_ts, store.current_frame
        events: list[Event] = []
        parked = [v for v in store.visible(GROUP_VEHICLE) if v.track_id in vzt.parked_ids()]
        persons = store.visible(GROUP_PERSON)

        for p in persons:
            foot = box_foot(p.box)
            slow = p.speed() < self.cfg.linger_speed
            held = hold is not None and hold(p.track_id)
            for v in parked:
                vid = vzt.canonical(v.track_id)
                key = (p.track_id, vid)
                it = self.interactions.setdefault(key, Interaction())
                in_zone = point_in_box(foot, self.proximity_box(v.box))
                if overlap_fraction(p.box, v.box) >= self.cfg.contact_overlap:
                    it.last_contact_ts = ts
                if in_zone and not it.in_zone:
                    if it.last_leave is None or ts - it.last_leave >= self.cfg.reapproach_gap_seconds:
                        it.episode_start = ts  # a new visit, not a brief step out of the zone
                        it.linger_seconds = 0.0
                        it.proximity_reported = it.loiter_reported = False
                if in_zone:
                    dt = ts - it.last_ts if it.last_ts is not None and it.in_zone else 0.0
                    it.total_seconds += dt
                    if slow:
                        it.linger_seconds += dt
                    if (not it.proximity_reported and not held
                            and it.linger_seconds >= self.cfg.proximity_min_seconds):
                        it.proximity_reported = True
                        it.approaches += 1
                        events.append(Event(EventKind.PROXIMITY, ts, frame, vid, p.track_id, p.conf,
                                            {"approach": it.approaches, "linger_seconds": round(it.linger_seconds, 1)}))
                        if it.approaches >= self.cfg.repeated_count and not it.repeated_reported:
                            it.repeated_reported = True
                            events.append(Event(EventKind.REPEATED_INTERACTION, ts, frame, vid, p.track_id, p.conf,
                                                {"approaches": it.approaches}))
                    if (not it.loiter_reported and it.episode_start is not None
                            and ts - it.episode_start >= self.cfg.loiter_seconds):
                        it.loiter_reported = True
                        events.append(Event(EventKind.EXTENDED_PRESENCE, ts, frame, vid, p.track_id, p.conf,
                                            {"seconds": round(ts - it.episode_start, 1)}))
                elif it.in_zone:
                    it.last_leave = ts
                it.in_zone = in_zone
                it.last_ts = ts

            for z in self.zones.of_kind("restricted"):
                if (p.track_id, z.name) not in self._restricted_seen and z.contains(foot, self.width, self.height):
                    self._restricted_seen.add((p.track_id, z.name))
                    events.append(Event(EventKind.RESTRICTED_AREA, ts, frame, None, p.track_id, p.conf, {"zone": z.name}))

        self._start_entries(store, vzt, lost_ids)
        events.extend(self._resolve_entries(store, vzt))
        return events

    def _reappeared(self, pend: _PendingEntry, store: TrackStore) -> TrackState | None:
        """A person track that started after the vanish, close to where they vanished."""
        x0, y0 = box_foot(pend.track_box)
        radius = self.cfg.reappear_radius * float(pend.track_box[3] - pend.track_box[1])
        for t in store.tracks.values():
            if t.group != GROUP_PERSON or t.track_id == pend.event.person_id or t.first_ts < pend.event.ts:
                continue
            x1, y1 = box_foot(t.history[0][2])
            if np.hypot(x1 - x0, y1 - y0) <= radius:
                return t
        return None

    def _resolve_entries(self, store: TrackStore, vzt: VehicleZoneTracker) -> list[Event]:
        """Withdraw entries where someone re-appeared; emit the ones confirmed by time or vehicle movement."""
        events, keep = [], []
        for pend in self._pending_entries:
            e = pend.event
            back = self._reappeared(pend, store)
            if back is not None:
                self.retracted_entries.append({"person_id": e.person_id, "vehicle_id": e.vehicle_id, "ts": e.ts,
                                               "reappeared_as": back.track_id})
                continue
            moving = vzt.state_of(pend.vehicle_track_id) in (MOVED, LEFT_ZONE, DEPARTED)
            if moving or store.current_ts - pend.declared_ts >= self.cfg.entry_confirm_seconds:
                e.details["confirmed_by"] = "vehicle movement" if moving else "no re-appearance"
                self.entrants.setdefault(e.vehicle_id, []).append(Entrant(e.person_id, e.ts, e.confidence))
                events.append(e)
            else:
                keep.append(pend)
        self._pending_entries = keep
        return events

    def _start_entries(self, store: TrackStore, vzt: VehicleZoneTracker, lost_ids: list[str]) -> None:
        """A person whose track ends while overlapping a vehicle, away from the frame edge, becomes a pending entry."""
        vehicles = store.visible(GROUP_VEHICLE)
        for pid in lost_ids:
            p = store.tracks.get(pid)
            if p is None or p.group != GROUP_PERSON or touches_border(p.box, self.width, self.height):
                continue
            best = None
            for v in vehicles:
                ov = overlap_fraction(p.box, v.box)
                if ov >= self.cfg.entry_overlap and (best is None or ov > best[1]):
                    best = (v, ov)
            if best is None:
                continue
            v, ov = best
            vid = vzt.canonical(v.track_id)
            it = self.interactions.get((pid, vid))
            had_contact = it is not None and it.last_contact_ts is not None and p.last_ts - it.last_contact_ts <= 2.0
            # Disappearing behind a car looks the same as getting in, so confidence stays moderate.
            conf = round(min(0.8, 0.4 + 0.4 * ov) * (1.0 if had_contact else 0.75), 3)
            event = Event(EventKind.APPARENT_ENTRY, p.last_ts, p.last_frame, vid, pid, conf,
                          {"overlap": round(ov, 3), "vehicle_state": vzt.state_of(v.track_id)})
            self._pending_entries.append(_PendingEntry(event, p.box.copy(), store.current_ts, v.track_id))
