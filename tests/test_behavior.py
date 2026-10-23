"""Behaviour rules that exist to cut false positives (logic tests on scripted tracks)."""

import numpy as np
import pytest

from app.behavior import BehaviorAnalyzer
from app.config import BehaviorConfig, TrackerConfig, ZonesConfig
from app.datatypes import GROUP_PERSON, GROUP_VEHICLE, Detection, EventKind
from app.parking_zones import VehicleZoneTracker, ZoneSet
from app.tracker import TrackStore

FPS = 10
CAR = [400, 300, 700, 450]


class World:
    def __init__(self, cfg=None):
        self.store = TrackStore(TrackerConfig())
        self.vzt = VehicleZoneTracker(ZonesConfig(park_seconds=1.0), ZoneSet(), (1280, 720))
        self.beh = BehaviorAnalyzer(cfg or BehaviorConfig(), ZoneSet(), (1280, 720))
        self.frame = 0
        self.events = []

    def step(self, boxes: dict, hold=None, car=CAR):
        dets = [] if car is None else [Detection(2, "car", GROUP_VEHICLE, 0.9, np.array(car, np.float32), track_id="V001")]
        dets += [Detection(0, "person", GROUP_PERSON, 0.9, np.array(b, np.float32), track_id=pid) for pid, b in boxes.items()]
        _, lost = self.store.update(self.frame / FPS, self.frame, dets, (720, 1280))
        self.vzt.update(self.store, lost)
        new = self.beh.update(self.store, self.vzt, lost, hold=hold)
        self.events += new
        self.frame += 1
        return new

    def run(self, seconds, boxes_at, **kw):
        for _ in range(int(round(seconds * FPS))):
            self.step(boxes_at(self.frame / FPS), **kw)

    def kinds(self):
        return [e.kind for e in self.events]


def person_at(x, y=460):
    return [x - 25, y - 130, x + 25, y]


def parked_world(cfg=None):
    w = World(cfg)
    w.run(2, lambda t: {})
    assert w.vzt.state_of("V001") == "parked"
    return w


def test_walking_past_does_not_count_as_proximity():
    w = parked_world()
    w.run(8, lambda t: {"P001": person_at(160 * (t - 2))})  # ~1.2 box diagonals per second
    assert EventKind.PROXIMITY not in w.kinds()


def test_lingering_counts_after_min_seconds():
    w = parked_world()
    w.run(1.0, lambda t: {"P001": person_at(720)})
    assert EventKind.PROXIMITY not in w.kinds()
    w.run(1.5, lambda t: {"P001": person_at(720)})
    prox = [e for e in w.events if e.kind == EventKind.PROXIMITY]
    assert len(prox) == 1 and prox[0].details["linger_seconds"] >= 1.5


def test_proximity_is_held_while_association_is_pending():
    w = parked_world()
    pending = {"P001": True}
    w.run(3, lambda t: {"P001": person_at(720)}, hold=lambda pid: pending.get(pid, False))
    assert EventKind.PROXIMITY not in w.kinds()
    pending["P001"] = False
    w.run(0.2, lambda t: {"P001": person_at(720)}, hold=lambda pid: pending.get(pid, False))
    assert w.kinds().count(EventKind.PROXIMITY) == 1


def vanish_at_car(w):
    """Person stands at the car for 2 s then disappears while overlapping it."""
    w.run(2, lambda t: {"P001": person_at(600)})
    w.run(1.2, lambda t: {})  # lost after 1 s -> pending entry


def test_entry_withdrawn_when_person_reappears_nearby():
    w = parked_world()
    vanish_at_car(w)
    assert EventKind.APPARENT_ENTRY not in w.kinds()
    w.run(0.5, lambda t: {"P002": person_at(640)})  # someone steps out right where P001 vanished
    w.run(4, lambda t: {"P002": person_at(640 + 100 * (t - 4.2))})
    assert EventKind.APPARENT_ENTRY not in w.kinds()
    assert w.beh.retracted_entries[0]["reappeared_as"] == "P002"


def test_entry_confirmed_when_nobody_reappears():
    w = parked_world()
    vanish_at_car(w)
    assert [e.person_id for e in w.beh.recent_entrants("V001", w.frame / FPS)] == ["P001"]  # pending, still listed
    w.run(3.0, lambda t: {})
    entries = [e for e in w.events if e.kind == EventKind.APPARENT_ENTRY]
    assert len(entries) == 1
    assert entries[0].details["confirmed_by"] == "no re-appearance"
    assert entries[0].ts == pytest.approx(3.9)  # timestamped when the person vanished, not when confirmed


def test_entry_confirmed_immediately_by_vehicle_movement():
    w = parked_world()
    vanish_at_car(w)
    shifted = [CAR[0] + 120, CAR[1], CAR[2] + 120, CAR[3]]
    w.run(1.3, lambda t: {}, car=shifted)  # moved car is confirmed after 1 s, entry before the 3 s wait ends
    entries = [e for e in w.events if e.kind == EventKind.APPARENT_ENTRY]
    assert len(entries) == 1 and entries[0].details["confirmed_by"] == "vehicle movement"


def test_passer_by_from_elsewhere_does_not_withdraw_entry():
    w = parked_world()
    vanish_at_car(w)
    w.run(3.0, lambda t: {"P009": person_at(80 * (t - 5.2) + 1000)})  # appears far away, walks the other way
    assert EventKind.APPARENT_ENTRY in w.kinds()
