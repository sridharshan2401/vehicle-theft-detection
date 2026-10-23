"""Scripted, synthetic car-park scenarios for the analysis logic (zones, association,
behaviour, risk).

Each scenario is a list of actors whose boxes are given as functions of time, so
the same situations can be replayed after every change. They measure how the
*rules* react to idealised tracks. They are not footage and say nothing about
real-world accuracy, detector misses or tracker id switches.

Used by tests/test_scenarios.py and scripts/run_scenarios.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.association import AssociationEngine
from app.behavior import BehaviorAnalyzer
from app.config import load_config
from app.datatypes import GROUP_PERSON, GROUP_VEHICLE, Detection, Event, EventKind
from app.parking_zones import VehicleZoneTracker, ZoneSet
from app.risk_engine import MOVEMENT_KINDS, OBSERVED_ASSOCIATED, UNRECOGNIZED, PersonContext, RiskEngine
from app.tracker import TrackStore

try:  # added in the false-positive work; absent in older versions of the code
    from app.risk_engine import POSSIBLE_ASSOCIATED
except ImportError:
    POSSIBLE_ASSOCIATED = UNRECOGNIZED

FPS = 10
FRAME = (720, 1280)
CAR = (400.0, 300.0, 700.0, 450.0)

Box = tuple[float, float, float, float]


@dataclass
class Actor:
    track_id: str
    box_at: Callable[[float], Box | None]

    @property
    def group(self) -> str:
        return GROUP_VEHICLE if self.track_id.startswith("V") else GROUP_PERSON


@dataclass
class Scenario:
    name: str
    description: str
    suspicious: bool  # should this raise an alert?
    duration: float
    actors: list[Actor]
    identity: dict[str, str] = field(default_factory=dict)  # person -> identity status the ReID layer would report
    expect_min_level: str = "MEDIUM"  # for suspicious scenarios
    expect_max_level: str | None = None  # for benign scenarios: highest acceptable alert (None = no alert at all)


def parked(box: Box = CAR, until: float | None = None) -> Callable[[float], Box | None]:
    return lambda t: box if until is None or t < until else None


def drive_off(box: Box, start: float, speed: float = 60.0) -> Callable[[float], Box | None]:
    """Parked until ``start``, then moves right at ``speed`` px/s until it leaves the frame."""
    def f(t):
        dx = max(0.0, t - start) * speed
        if box[0] + dx > FRAME[1]:
            return None
        return (box[0] + dx, box[1], min(box[2] + dx, FRAME[1] - 1), box[3])
    return f


def path(points: list[tuple[float, float, float]], w: float = 50, h: float = 130) -> Callable[[float], Box | None]:
    """Piecewise-linear foot-point path [(t, x, y), ...]; invisible outside its time range."""
    ts = [p[0] for p in points]

    def f(t):
        if t < ts[0] or t > ts[-1]:
            return None
        x = float(np.interp(t, ts, [p[1] for p in points]))
        y = float(np.interp(t, ts, [p[2] for p in points]))
        return (x - w / 2, y - h, x + w / 2, y)
    return f


def occluded_car(box: Box, start: float, end: float, cut: Box) -> Callable[[float], Box | None]:
    """Parked car whose detected box shrinks to ``cut`` between start and end (someone standing in front)."""
    return lambda t: cut if start <= t < end else box


SCENARIOS = [
    Scenario(
        "walk_past_x3", "three people walk across in front of a parked car without stopping",
        False, 40.0,
        [Actor("V001", parked()),
         Actor("P001", path([(5, 0, 470), (13, 1280, 470)])),
         Actor("P002", path([(15, 1280, 465), (23, 0, 465)])),
         Actor("P003", path([(25, 0, 475), (33, 1280, 475)]))],
    ),
    Scenario(
        "occlusion_behind_car", "a person walks behind the car, is hidden for 2 s, and comes out the other side",
        False, 30.0,
        [Actor("V001", parked()),
         Actor("P001", path([(5, 200, 420), (9, 520, 420)])),
         Actor("P002", path([(11, 600, 420), (16, 1100, 420)]))],
    ),
    Scenario(
        "parked_car_box_jitter", "someone stands in front of a parked car and its box shrinks for 0.6 s",
        False, 30.0,
        [Actor("V001", occluded_car(CAR, 10.0, 10.6, (400.0, 300.0, 520.0, 450.0)))],
    ),
    Scenario(
        "driver_exits_and_leaves", "two people get out of the car and walk away; the car stays",
        False, 30.0,
        [Actor("V001", parked()),
         Actor("P001", path([(6, 690, 450), (14, 1240, 470)])),
         Actor("P002", path([(7, 410, 450), (15, 30, 470)]))],
    ),
    Scenario(
        "owner_returns_reid_match", "the driver leaves, comes back (ReID reconnects them) and drives off",
        False, 70.0,
        [Actor("V001", drive_off(CAR, 52.0)),
         Actor("P001", path([(6, 690, 450), (14, 1270, 470)])),
         Actor("P002", path([(38, 1270, 470), (45, 720, 450), (46, 690, 445)]))],
        # Observed association is not authorisation, so a LOW note is acceptable; nothing higher.
        identity={"P002": OBSERVED_ASSOCIATED}, expect_max_level="LOW",
    ),
    Scenario(
        "owner_returns_reid_unsure", "same, but ReID only reports a possible match to the driver",
        False, 70.0,
        [Actor("V001", drive_off(CAR, 52.0)),
         Actor("P001", path([(6, 690, 450), (14, 1270, 470)])),
         Actor("P002", path([(38, 1270, 470), (45, 720, 450), (46, 690, 445)]))],
        # Genuinely uncertain: worth a look, but must rank below a clear theft.
        identity={"P002": POSSIBLE_ASSOCIATED}, expect_max_level="MEDIUM",
    ),
    Scenario(
        "theft", "an unknown person walks up, lingers at the door, gets in, and the car drives off",
        True, 60.0,
        [Actor("V001", drive_off(CAR, 30.0)),
         Actor("P001", path([(5, 0, 460), (12, 690, 445), (20, 695, 445), (22, 680, 440)]))],
        expect_min_level="MEDIUM",
    ),
    Scenario(
        "loitering", "an unknown person stands next to the parked car for 25 s, then leaves",
        True, 50.0,
        [Actor("V001", parked()),
         Actor("P001", path([(5, 0, 460), (10, 720, 455), (35, 725, 455), (40, 1280, 470)]))],
        expect_min_level="LOW",
    ),
]


@dataclass
class ScenarioResult:
    name: str
    suspicious: bool
    events: dict[str, int]
    alert_levels: list[str]
    max_level: str
    passed: bool
    expectation: str


def run(scenario: Scenario, settings=None) -> ScenarioResult:
    from app.config import RISK_LEVELS

    s = settings or load_config()
    store = TrackStore(s.tracker)
    zones = ZoneSet()
    vzt = VehicleZoneTracker(s.zones, zones, (FRAME[1], FRAME[0]))
    assoc = AssociationEngine(s.association)
    behavior = BehaviorAnalyzer(s.behavior, zones, (FRAME[1], FRAME[0]))
    risk = RiskEngine(s.risk)

    def resolve(pid, vid):
        status = scenario.identity.get(pid)
        if status:
            return PersonContext(status, pid)
        r = assoc.get(pid)
        if r and r.is_linked and vzt.canonical(r.vehicle_id) == vid:
            return PersonContext(OBSERVED_ASSOCIATED, pid)
        return PersonContext(UNRECOGNIZED, pid)

    counts: dict[str, int] = {}
    levels: list[str] = []
    for i in range(int(scenario.duration * FPS)):
        t = i / FPS
        dets = []
        for a in scenario.actors:
            b = a.box_at(t)
            if b is not None:
                cls = (2, "car") if a.group == GROUP_VEHICLE else (0, "person")
                dets.append(Detection(cls[0], cls[1], a.group, 0.9, np.array(b, np.float32), track_id=a.track_id))
        new, lost = store.update(t, i, dets, FRAME)
        events: list[Event] = list(vzt.update(store, lost))
        for rec in assoc.update(store, new):
            if rec.is_linked:
                events.append(Event(EventKind.PERSON_EXITED_VEHICLE, rec.decided_ts, rec.decided_frame,
                                    vzt.canonical(rec.vehicle_id), rec.person_id, rec.confidence))
        events += behavior.update(store, vzt, lost, hold=assoc.is_pending)
        for e in events:
            if e.kind in MOVEMENT_KINDS:
                e.details["recent_entrants"] = [en.person_id for en in behavior.recent_entrants(e.vehicle_id, t)]
            counts[e.kind] = counts.get(e.kind, 0) + 1
        levels += [a.level for a in risk.process(events, resolve)]
        risk.tick(t)

    max_level = max(levels, key=RISK_LEVELS.index, default="INFORMATIONAL")
    if scenario.suspicious:
        passed = RISK_LEVELS.index(max_level) >= RISK_LEVELS.index(scenario.expect_min_level)
        expectation = f"alert >= {scenario.expect_min_level}"
    elif scenario.expect_max_level:
        passed = RISK_LEVELS.index(max_level) <= RISK_LEVELS.index(scenario.expect_max_level)
        expectation = f"max {scenario.expect_max_level}"
    else:
        passed = not levels
        expectation = "no alert"
    return ScenarioResult(scenario.name, scenario.suspicious, counts, levels, max_level, passed, expectation)


def run_all(settings=None) -> list[ScenarioResult]:
    return [run(sc, settings) for sc in SCENARIOS]
