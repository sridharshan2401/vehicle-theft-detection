"""Association logic on hand-built synthetic tracks (logic tests, not accuracy measurements)."""

import numpy as np

from app.association import (
    AMBIGUOUS,
    NO_ASSOCIATION,
    OBSERVED_EXIT,
    POSSIBLE_EXIT,
    AssociationEngine,
    door_zone_score,
)
from app.config import AssociationConfig, TrackerConfig
from app.datatypes import GROUP_PERSON, GROUP_VEHICLE, Detection
from app.tracker import TrackStore

FPS = 10
SHAPE = (720, 1280)
CAR = np.array([400, 300, 700, 450], np.float32)


def det(track_id, box, group=None):
    group = group or (GROUP_VEHICLE if track_id.startswith("V") else GROUP_PERSON)
    return Detection(2 if group == GROUP_VEHICLE else 0, "car" if group == GROUP_VEHICLE else "person", group, 0.9,
                     np.array(box, np.float32), track_id=track_id)


class Sim:
    """Feeds per-frame detections to a TrackStore and the association engine."""

    def __init__(self, cfg=None):
        self.store = TrackStore(TrackerConfig())
        self.engine = AssociationEngine(cfg or AssociationConfig())
        self.frame = 0
        self.decided = []

    def step(self, dets):
        ts = self.frame / FPS
        new, _ = self.store.update(ts, self.frame, dets, SHAPE)
        self.decided += self.engine.update(self.store, new)
        self.frame += 1

    def run(self, seconds, fn):
        for _ in range(int(seconds * FPS)):
            self.step(fn(self.frame / FPS))


def walking(start, velocity, t0):
    def box(t):
        dx = velocity[0] * (t - t0)
        dy = velocity[1] * (t - t0)
        return [start[0] + dx, start[1] + dy, start[2] + dx, start[3] + dy]
    return box


def test_person_exiting_parked_car_is_associated():
    sim = Sim()
    sim.run(5, lambda t: [det("V001", CAR)])
    p = walking([680, 330, 730, 460], (60, 0), 5.0)  # appears at the right door, walks right
    sim.run(4, lambda t: [det("V001", CAR), det("P001", p(t))])
    rec = sim.engine.get("P001")
    assert rec.status == OBSERVED_EXIT
    assert rec.vehicle_id == "V001"
    assert rec.confidence >= 0.6
    assert rec.components["outward_motion"] > 0.5


def test_person_walking_past_is_not_associated():
    sim = Sim()
    sim.run(2, lambda t: [det("V001", CAR)])
    p = walking([50, 330, 100, 460], (150, 0), 2.0)  # enters far left, walks across in front of the car
    sim.run(8, lambda t: [det("V001", CAR), det("P001", p(t))])
    rec = sim.engine.get("P001")
    assert rec.status == NO_ASSOCIATION
    assert rec.vehicle_id is None
    assert "did not first appear next to a vehicle" in rec.reasons[0]


def test_three_people_from_one_car_keep_separate_ids():
    sim = Sim()
    sim.run(3, lambda t: [det("V001", CAR)])
    paths = {
        "P001": walking([680, 330, 730, 460], (60, 0), 3.0),
        "P002": walking([380, 330, 430, 460], (-60, 0), 4.0),
        "P003": walking([690, 340, 740, 465], (50, 20), 5.0),
    }
    starts = {"P001": 3.0, "P002": 4.0, "P003": 5.0}

    def frame(t):
        return [det("V001", CAR)] + [det(pid, f(t)) for pid, f in paths.items() if t >= starts[pid]]

    sim.run(6, frame)
    recs = {pid: sim.engine.get(pid) for pid in paths}
    assert all(r.vehicle_id == "V001" and r.status in (OBSERVED_EXIT, POSSIBLE_EXIT) for r in recs.values())
    assert {r.person_id for r in recs.values()} == {"P001", "P002", "P003"}
    assert len(sim.engine.persons_for_vehicle("V001")) == 3


def test_person_between_two_cars_is_ambiguous():
    left = np.array([300, 300, 500, 450], np.float32)
    right = np.array([520, 300, 720, 450], np.float32)
    sim = Sim()
    sim.run(3, lambda t: [det("V001", left), det("V002", right)])
    person = [485, 330, 535, 460]  # standing exactly between them, not moving
    sim.run(4, lambda t: [det("V001", left), det("V002", right), det("P001", person)])
    rec = sim.engine.get("P001")
    assert rec.status == AMBIGUOUS
    assert rec.vehicle_id is None
    assert {c["vehicle_id"] for c in rec.candidates} == {"V001", "V002"}


def test_people_present_at_start_or_after_reset_are_not_exits():
    sim = Sim()
    sim.run(3, lambda t: [det("V001", CAR), det("P001", [680, 330, 730, 460])])
    assert sim.engine.get("P001").status == NO_ASSOCIATION

    sim.engine.notify_reset(sim.frame / FPS)
    sim.run(3, lambda t: [det("V001", CAR), det("P002", [680, 330, 730, 460])])
    rec = sim.engine.get("P002")
    assert rec.status == NO_ASSOCIATION
    assert "reset" in rec.reasons[0]


def test_reappearance_after_occlusion_is_penalised():
    sim = Sim()
    sim.run(3, lambda t: [det("V001", CAR)])
    # P001 walks in from the left and disappears behind the car...
    p1 = walking([150, 330, 200, 460], (100, 0), 3.0)
    sim.run(3, lambda t: [det("V001", CAR), det("P001", p1(t))])
    sim.run(1.5, lambda t: [det("V001", CAR)])
    # ...then someone appears where P001 vanished.
    p2 = walking([450, 330, 500, 460], (0, 30), sim.frame / FPS)
    sim.run(3, lambda t: [det("V001", CAR), det("P002", p2(t))])
    rec = sim.engine.get("P002")
    assert any("re-appearance" in r for r in rec.reasons)
    assert rec.status != OBSERVED_EXIT


def test_door_zone_score_regions():
    assert door_zone_score(np.array([680, 330, 730, 460]), CAR, 0.25) == 1.0  # beside the right end
    assert door_zone_score(np.array([530, 380, 570, 470]), CAR, 0.25) == 0.8  # along the lower edge
    assert door_zone_score(np.array([1000, 100, 1050, 200]), CAR, 0.25) == 0.0  # far away
