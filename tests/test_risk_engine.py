import pytest

from app.config import RiskConfig
from app.datatypes import Event, EventKind
from app.risk_engine import (
    AUTHORIZED,
    OBSERVED_ASSOCIATED,
    POSSIBLE_ASSOCIATED,
    UNRECOGNIZED,
    PersonContext,
    RiskEngine,
)


def resolver(status_by_person):
    def resolve(pid, vid):
        return PersonContext(status_by_person.get(pid, UNRECOGNIZED), pid, "Owner")
    return resolve


UNKNOWN = resolver({})


def ev(kind, ts, person="P001", vehicle="V001", conf=1.0, **details):
    return Event(kind, ts, int(ts * 10), vehicle, person, conf, details)


def test_levels_follow_thresholds():
    r = RiskEngine(RiskConfig())
    assert r.level_for(0) == "INFORMATIONAL"
    assert r.level_for(15) == "LOW"
    assert r.level_for(54.9) == "MEDIUM"
    assert r.level_for(55) == "HIGH"
    assert r.level_for(100) == "CRITICAL"


def test_identity_changes_the_points():
    cfg = RiskConfig()
    w = cfg.weights["proximity"]
    for status, factor in ((UNRECOGNIZED, 1.0), (POSSIBLE_ASSOCIATED, 0.6), (OBSERVED_ASSOCIATED, 0.3), (AUTHORIZED, 0.0)):
        r = RiskEngine(cfg)
        r.process([ev(EventKind.PROXIMITY, 1.0, conf=0.8)], resolver({"P001": status}))
        assert r.current("V001", 1.0)[0] == pytest.approx(w * factor * 0.8)


def test_score_decays_with_half_life():
    cfg = RiskConfig(half_life_seconds=10)
    r = RiskEngine(cfg)
    r.process([ev(EventKind.APPARENT_ENTRY, 0.0)], UNKNOWN)
    s0 = r.current("V001", 0.0)[0]
    assert r.current("V001", 10.0)[0] == pytest.approx(s0 / 2)
    assert r.current("V001", 20.0)[0] == pytest.approx(s0 / 4)


def test_cooldown_suppresses_duplicates():
    r = RiskEngine(RiskConfig(cooldown_seconds=20, half_life_seconds=1e9))
    r.process([ev(EventKind.PROXIMITY, 0.0)], UNKNOWN)
    r.process([ev(EventKind.PROXIMITY, 5.0)], UNKNOWN)
    assert r.suppressed == 1
    assert r.current("V001", 5.0)[0] == pytest.approx(10.0)
    r.process([ev(EventKind.PROXIMITY, 25.0)], UNKNOWN)
    assert r.current("V001", 25.0)[0] == pytest.approx(20.0)
    # a different person is not a duplicate
    r.process([ev(EventKind.PROXIMITY, 26.0, person="P002")], UNKNOWN)
    assert r.current("V001", 26.0)[0] == pytest.approx(30.0)


def test_alerts_on_escalation_only():
    r = RiskEngine(RiskConfig(half_life_seconds=1e9, realert_seconds=1e9))
    assert r.process([ev(EventKind.PROXIMITY, 0.0)], UNKNOWN) == []  # 10 points: INFORMATIONAL
    a = r.process([ev(EventKind.REPEATED_INTERACTION, 1.0)], UNKNOWN)  # 30 -> LOW
    assert [x.level for x in a] == ["LOW"]
    b = r.process([ev(EventKind.PROXIMITY, 2.0, person="P009")], UNKNOWN)  # 40 -> MEDIUM
    assert [x.level for x in b] == ["MEDIUM"]
    assert r.process([ev(EventKind.PROXIMITY, 3.0, person="P010")], UNKNOWN) == []  # 50, still MEDIUM
    assert r.current("V001", 3.0)[1] == "MEDIUM"


def test_escalation_alert_after_decay():
    r = RiskEngine(RiskConfig(half_life_seconds=10, cooldown_seconds=0, realert_seconds=1e9))
    first = r.process([ev(EventKind.APPARENT_ENTRY, 0.0)], UNKNOWN)
    assert [a.level for a in first] == ["LOW"]
    r.tick(100.0)  # fully decayed
    again = r.process([ev(EventKind.APPARENT_ENTRY, 100.0)], UNKNOWN)
    assert [a.level for a in again] == ["LOW"]


def test_movement_after_unrecognised_entry_is_high_and_cautiously_worded():
    r = RiskEngine(RiskConfig(half_life_seconds=1e9))
    r.process([ev(EventKind.PROXIMITY, 0.0), ev(EventKind.APPARENT_ENTRY, 3.0)], UNKNOWN)
    alerts = r.process([ev(EventKind.VEHICLE_MOVED, 10.0, person=None, recent_entrants=["P001"])], UNKNOWN)
    assert alerts and alerts[-1].level == "HIGH"
    msg = alerts[-1].message
    assert alerts[-1].category == "possible unauthorised vehicle movement"
    assert "unrecognised person (P001)" in msg
    assert "thief" not in msg.lower() and "stolen" not in msg.lower()


def test_movement_with_authorised_entrant_adds_nothing():
    r = RiskEngine(RiskConfig())
    r.process([ev(EventKind.VEHICLE_MOVED, 10.0, person=None, recent_entrants=["P001"])], resolver({"P001": AUTHORIZED}))
    assert r.current("V001", 10.0)[0] == 0.0


def test_movement_without_observed_driver_uses_reduced_factor():
    cfg = RiskConfig()
    r = RiskEngine(cfg)
    r.process([ev(EventKind.VEHICLE_MOVED, 10.0, person=None)], UNKNOWN)
    expected = cfg.weights["vehicle_moved"] * cfg.identity_factors["no_entrant_observed"]
    assert r.current("V001", 10.0)[0] == pytest.approx(expected)
    assert "driver not observed" in r.vehicles["V001"].reasons[-1].description


def test_informational_events_do_not_score():
    r = RiskEngine(RiskConfig())
    assert r.process([ev(EventKind.VEHICLE_PARKED, 1.0, person=None),
                      ev(EventKind.PERSON_EXITED_VEHICLE, 2.0)], UNKNOWN) == []
    assert r.current("V001", 2.0)[0] == 0.0


def test_score_is_capped():
    r = RiskEngine(RiskConfig(cooldown_seconds=0, half_life_seconds=1e9))
    for i in range(20):
        r.process([ev(EventKind.APPARENT_ENTRY, float(i), person=f"P{i}")], UNKNOWN)
    score, level = r.current("V001", 19.0)
    assert score == pytest.approx(100.0) and level == "CRITICAL"


def test_movement_uses_the_most_suspicious_entrant():
    cfg = RiskConfig()
    statuses = {"P001": OBSERVED_ASSOCIATED, "P002": POSSIBLE_ASSOCIATED, "P003": UNRECOGNIZED}
    for entrants, factor in ((["P001"], 0.3), (["P001", "P002"], 0.6), (["P002", "P003", "P001"], 1.0)):
        r = RiskEngine(cfg)
        r.process([ev(EventKind.VEHICLE_MOVED, 5.0, person=None, recent_entrants=entrants)], resolver(statuses))
        assert r.current("V001", 5.0)[0] == pytest.approx(cfg.weights["vehicle_moved"] * factor)


def test_possible_match_is_worded_as_unconfirmed():
    ctx = PersonContext(POSSIBLE_ASSOCIATED, "P009", related_id="P001")
    assert ctx.describe() == "P009, possibly P001 who got out of this vehicle (unconfirmed appearance match)"


def test_pipeline_resolver_gives_partial_credit_only_for_this_vehicle():
    """Pipeline._resolve_person with a real gallery and association engine, no video."""
    import numpy as np

    from app.association import AssociationEngine, AssociationRecord
    from app.config import AssociationConfig, ReIDConfig, ZonesConfig
    from app.parking_zones import VehicleZoneTracker, ZoneSet
    from app.person_reid import PersonGallery
    from app.pipeline import Pipeline

    p = Pipeline.__new__(Pipeline)
    p.registry = {}
    p.vzt = VehicleZoneTracker(ZonesConfig(), ZoneSet(), (1280, 720))
    p.assoc = AssociationEngine(AssociationConfig())
    p.assoc.records["P001"] = AssociationRecord("P001", "V001", "observed_exit", 0.8, 1.0, 10, 3.5, 35)
    p.gallery = PersonGallery(ReIDConfig(min_samples_for_decision=1, max_pending_samples=3), "torchreid")
    p.gallery.add_sample("P001", np.array([1.0, 0.0], np.float32), 1.0, 10)
    p.gallery.note_visible(["P009"], 60.0)
    dec = p.gallery.add_sample("P009", np.array([0.6, 0.8], np.float32), 60.0, 600)  # cosine 0.6: unsure
    assert dec.status == "candidate" and dec.candidate_id == "P001"

    ctx = p._resolve_person("P009", "V001")
    assert ctx.status == POSSIBLE_ASSOCIATED and ctx.related_id == "P001"
    assert p._resolve_person("P009", "V002").status == UNRECOGNIZED  # P001 got out of V001, not V002
    assert p._resolve_person("P001", "V001").status == OBSERVED_ASSOCIATED
