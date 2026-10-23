"""The evaluation harness itself: label parsing, matching rules and metric arithmetic.

These use hand-built session data with known answers. They test the scoring code,
not the detector.
"""

import json

import pytest

from app.evaluation import (
    LabelError,
    SessionData,
    Tolerances,
    aggregate,
    evaluate,
    load_labels,
    wilson,
)

W, H = 1000, 500
CAR_A = [100, 200, 300, 300]  # contains labelled point A = (0.2, 0.5)
CAR_B = [600, 200, 800, 300]  # contains labelled point B = (0.7, 0.5)


def labels(tmp_path, **extra):
    data = {"video": "x.mp4", "vehicles": {"A": [0.2, 0.5], "B": [0.7, 0.5]}, "duration_seconds": 1800}
    data.update(extra)
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(data))
    return load_labels(p)


def assoc(aid, ts, status, vehicle, cands, person="P1"):
    return {"association_id": aid, "birth_ts": ts, "status": status, "vehicle_id": vehicle,
            "person_track_id": person, "evidence": {"candidates": [{"vehicle_id": v, "box": b} for v, b in cands]}}


def event(eid, ts, kind, box=None, level="INFORMATIONAL", score=None):
    return {"event_id": eid, "video_ts": ts, "category": kind, "risk_level": level, "risk_score": score,
            "location": {"vehicle_box": box} if box else None}


def session(events=(), assocs=()):
    return SessionData("S1", W, H, 1800.0, list(events), list(assocs))


def test_wilson_interval_known_value():
    assert wilson(9, 10) == (0.596, 0.982)
    assert wilson(0, 0) is None
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0.15 < hi < 0.17


def test_label_validation(tmp_path):
    with pytest.raises(LabelError, match="0..1"):
        labels(tmp_path, vehicles={"A": [1.5, 0.2]})
    with pytest.raises(LabelError, match="vehicle"):
        labels(tmp_path, exits=[{"time": 3.0, "vehicle": "Z"}])
    with pytest.raises(LabelError, match="start <= end"):
        labels(tmp_path, incidents=[{"start": 10, "end": 5}])


def test_exit_outcomes(tmp_path):
    lab = labels(tmp_path, exits=[{"time": 10, "vehicle": "A"}, {"time": 50, "vehicle": "A"},
                                  {"time": 90, "vehicle": "B"}, {"time": 130, "vehicle": "B"}])
    s = session(assocs=[
        assoc(1, 10.5, "observed_exit", "V1", [("V1", CAR_A)]),                     # correct
        assoc(2, 51.0, "possible_exit", "V2", [("V2", CAR_B), ("V1", CAR_A)]),       # wrong vehicle
        assoc(3, 89.0, "ambiguous", None, [("V2", CAR_B), ("V1", CAR_A)]),           # ambiguous
        assoc(4, 300.0, "observed_exit", "V1", [("V1", CAR_A)]),                     # false link (nobody exited)
    ])                                                                               # t=130 exit is missed
    a = evaluate(lab, s)["association"]
    assert (a["correct"], a["wrong_vehicle"], a["ambiguous"], a["missed"]) == (1, 1, 1, 1)
    assert a["exit_recall"]["value"] == 0.25
    assert a["vehicle_accuracy_when_linked"]["value"] == 0.5
    assert a["false_links"] == 1 and a["false_links_per_hour"] == 2.0  # 1 in 0.5 h
    assert a["link_precision"] == {"k": 2, "n": 3, "value": 0.667, "ci95": wilson(2, 3)}


def test_event_matching_needs_time_and_place(tmp_path):
    lab = labels(tmp_path, movements=[{"time": 100, "vehicle": "A"}, {"time": 200, "vehicle": "B"}])
    s = session(events=[
        event("e1", 101.5, "vehicle_moved", CAR_A),   # TP
        event("e2", 201.0, "vehicle_moved", CAR_A),   # right time, wrong car -> FP, B missed
        event("e3", 400.0, "vehicle_moved", CAR_B),   # nothing labelled -> FP
    ])
    m = evaluate(lab, s)["events"]["vehicle_moved"]
    assert (m["true_positives"], m["false_positives"]) == (1, 2)
    assert m["recall"]["value"] == 0.5 and m["precision"]["value"] == 0.333
    assert m["false_positives_per_hour"] == 4.0


def test_incident_recall_latency_and_false_alerts(tmp_path):
    lab = labels(tmp_path, incidents=[{"id": "T1", "start": 100, "end": 150, "vehicle": "A"},
                                      {"id": "T2", "start": 500, "end": 550, "vehicle": "B"}])
    s = session(events=[
        event("a1", 112, "suspicious interaction", CAR_A, "LOW", 20),
        event("a2", 130, "possible unauthorised vehicle movement", CAR_A, "HIGH", 60),
        event("a3", 520, "suspicious interaction", CAR_B, "LOW", 18),      # T2 only reaches LOW
        event("a4", 900, "suspicious interaction", CAR_B, "MEDIUM", 40),   # false alert
        event("a5", 1000, "suspicious interaction", CAR_A, "LOW", 16),     # false alert
    ])
    rep = evaluate(lab, s, min_level="MEDIUM")
    inc = rep["incidents"]
    assert inc["recall"]["value"] == 0.5
    assert inc["details"][0] == {"id": "T1", "detected": True, "latency_seconds": 30.0, "first_level": "HIGH"}
    al = rep["alerts"]
    assert al["false_alerts_at_least_LOW"] == 2 and al["false_alerts_per_hour_at_least_LOW"] == 4.0
    assert al["false_alerts_at_least_MEDIUM"] == 1
    assert evaluate(lab, s, min_level="LOW")["incidents"]["recall"]["value"] == 1.0


def test_aggregate_pools_counts(tmp_path):
    lab = labels(tmp_path, exits=[{"time": 10, "vehicle": "A"}])
    good = evaluate(lab, session(assocs=[assoc(1, 10, "observed_exit", "V1", [("V1", CAR_A)])]))
    bad = evaluate(lab, session())
    pooled = aggregate([good, bad])
    assert pooled["videos"] == 2 and pooled["hours"] == 1.0
    assert pooled["association"]["exit_recall"]["k"] == 1 and pooled["association"]["exit_recall"]["n"] == 2


def test_tolerance_is_respected(tmp_path):
    lab = labels(tmp_path, exits=[{"time": 10, "vehicle": "A"}])
    s = session(assocs=[assoc(1, 14.5, "observed_exit", "V1", [("V1", CAR_A)])])
    assert evaluate(lab, s)["association"]["missed"] == 1
    assert evaluate(lab, s, Tolerances(exit_seconds=5))["association"]["correct"] == 1


def test_end_to_end_pipeline_then_evaluation(test_settings, tmp_path):
    """Synthetic theft video -> full pipeline -> SQLite -> evaluator, with hand-written labels.

    Checks that events carry locations and that the harness can match them. It is a
    plumbing test on coloured rectangles, not an accuracy measurement.
    """
    from conftest import ColorBlobDetector, iou_tracker_factory, make_theft_scene_video

    from app.evaluation import load_session
    from app.pipeline import Pipeline
    from app.storage import Storage
    from app.video_reader import probe_video

    video = make_theft_scene_video(tmp_path / "theft.mp4")
    s = test_settings(**{"app.mode": "complete", "zones.park_seconds": 2.0})
    summary = Pipeline(s, probe_video(video), "cpu", False, detector=ColorBlobDetector(),
                       tracker_factory=iou_tracker_factory, display=False).run()
    car = [150 / 320, 130 / 240]  # centre of the blue car in the 320x240 synthetic video
    lab_path = tmp_path / "theft_labels.json"
    lab_path.write_text(json.dumps({
        "video": "theft.mp4", "vehicles": {"A": car},
        "entries": [{"time": 8.4, "vehicle": "A"}],        # person vanishes into the car at frame 84
        "movements": [{"time": 10.0, "vehicle": "A"}],     # car starts moving at frame 100
        "departures": [{"time": 15.5, "vehicle": "A"}],    # car leaves the frame around frame 155
        "incidents": [{"id": "T1", "start": 4.0, "end": 16.0, "vehicle": "A"}],
    }))
    db = Storage(s.storage.db_path)
    rep = evaluate(load_labels(lab_path), load_session(db, summary["session_id"]))
    db.close()
    assert rep["warnings"] == []
    for kind in ("apparent_entry", "vehicle_moved", "vehicle_departed"):
        assert rep["events"][kind]["true_positives"] == 1, (kind, rep["events"][kind])
        assert rep["events"][kind]["false_positives"] == 0, (kind, rep["events"][kind])
    assert rep["incidents"]["recall"]["value"] == 1.0
    assert rep["alerts"]["false_alerts_at_least_LOW"] == 0
    assert rep["association"]["false_links"] == 0
