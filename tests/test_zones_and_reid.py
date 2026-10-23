"""Parking zones, vehicle state machine, ReID gallery decisions and clothing colour naming (logic only)."""

import numpy as np
import pytest

from app.clothing_analyzer import color_name, dominant_color
from app.config import ReIDConfig, TrackerConfig, ZonesConfig
from app.datatypes import GROUP_VEHICLE, Detection, EventKind
from app.parking_zones import PARKED, Zone, ZoneError, ZoneSet, VehicleZoneTracker, point_in_polygon
from app.person_reid import CANDIDATE, NEW, PENDING, RECONNECTED, PersonGallery
from app.tracker import TrackStore


def test_point_in_polygon():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)


def test_zone_file_round_trip_and_validation(tmp_path):
    zs = ZoneSet([Zone("A1", "parking", [[0.1, 0.5], [0.5, 0.5], [0.5, 0.9], [0.1, 0.9]]),
                  Zone("gate", "exit", [[0.9, 0.0], [1.0, 0.0], [1.0, 1.0], [0.9, 1.0]])])
    path = tmp_path / "zones.json"
    zs.save(path)
    loaded = ZoneSet.load(path)
    assert [z.name for z in loaded.zones] == ["A1", "gate"]
    assert loaded.find((0.3 * 1280, 0.7 * 720), "parking", 1280, 720).name == "A1"
    assert ZoneSet.load(tmp_path / "missing.json").zones == []
    with pytest.raises(ZoneError):
        ZoneSet([Zone("bad", "parking", [[0, 0], [1, 1]])]).validate()
    with pytest.raises(ZoneError):
        ZoneSet([Zone("bad", "garage", [[0, 0], [1, 0], [1, 1]])]).validate()


def drive(vzt, store, boxes, fps=10, start=0):
    events = []
    for i, box in enumerate(boxes):
        f = start + i
        dets = [] if box is None else [Detection(2, "car", GROUP_VEHICLE, 0.9, np.array(box, np.float32), track_id="V001")]
        _, lost = store.update(f / fps, f, dets, (720, 1280))
        events += vzt.update(store, lost)
    return events


def parked_tracker():
    store = TrackStore(TrackerConfig())
    vzt = VehicleZoneTracker(ZonesConfig(park_seconds=2.0), ZoneSet(), (1280, 720))
    drive(vzt, store, [[400, 300, 600, 420]] * 30)
    assert vzt.state_of("V001") == PARKED
    return store, vzt


def test_box_shrinking_from_occlusion_is_not_movement():
    store, vzt = parked_tracker()
    # someone stands in front: the detected box shrinks to its left part for 2 s
    events = drive(vzt, store, [[400, 300, 480, 420]] * 20 + [[400, 300, 600, 420]] * 10, start=30)
    assert EventKind.VEHICLE_MOVED not in [e.kind for e in events]
    assert vzt.state_of("V001") == PARKED


def test_brief_displacement_is_not_movement():
    store, vzt = parked_tracker()
    shifted = [[460, 300, 660, 420]] * 5  # same size, 0.5 s, then back
    events = drive(vzt, store, shifted + [[400, 300, 600, 420]] * 15, start=30)
    assert EventKind.VEHICLE_MOVED not in [e.kind for e in events]


def test_sustained_displacement_is_confirmed_after_one_second():
    store, vzt = parked_tracker()
    events = drive(vzt, store, [[460, 300, 660, 420]] * 15, start=30)
    moved = [e for e in events if e.kind == EventKind.VEHICLE_MOVED]
    assert len(moved) == 1 and moved[0].details["moving_since"] == pytest.approx(3.0)
    assert moved[0].ts == pytest.approx(4.0)


def test_vehicle_that_vanishes_while_starting_to_move_still_departs():
    store, vzt = parked_tracker()
    events = drive(vzt, store, [[460, 300, 660, 420]] * 5 + [None] * 15, start=30)
    kinds = [e.kind for e in events]
    assert kinds == [EventKind.VEHICLE_MOVED, EventKind.VEHICLE_DEPARTED]


def test_vehicle_parks_moves_and_leaves_zone():
    zones = ZoneSet([Zone("A1", "parking", [[0.2, 0.3], [0.6, 0.3], [0.6, 0.8], [0.2, 0.8]])])
    store = TrackStore(TrackerConfig())
    vzt = VehicleZoneTracker(ZonesConfig(park_seconds=2.0), zones, (1280, 720))
    still = [[400, 300, 600, 420]] * 40
    events = drive(vzt, store, still)
    assert [e.kind for e in events] == [EventKind.VEHICLE_PARKED]
    assert vzt.state_of("V001") == PARKED
    leaving = [[400 + 15 * i, 300, 600 + 15 * i, 420] for i in range(1, 40)]
    events = drive(vzt, store, leaving, start=40)
    kinds = [e.kind for e in events]
    assert kinds[:2] == [EventKind.VEHICLE_MOVED, EventKind.VEHICLE_LEFT_ZONE]


def test_relink_parked_vehicle_after_track_change():
    store = TrackStore(TrackerConfig())
    vzt = VehicleZoneTracker(ZonesConfig(park_seconds=1.0), ZoneSet(), (1280, 720))
    drive(vzt, store, [[400, 300, 600, 420]] * 20)
    # same car comes back as a new track id
    d = Detection(2, "car", GROUP_VEHICLE, 0.9, np.array([402, 301, 601, 421], np.float32), track_id="V007")
    store.update(5.0, 50, [d], (720, 1280))
    vzt.update(store, [])
    assert vzt.canonical("V007") == "V001"
    assert vzt.state_of("V007") == PARKED


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_gallery_reconnects_returning_person():
    g = PersonGallery(ReIDConfig(min_samples_for_decision=2), "torchreid")
    a = unit([1, 0, 0, 0])
    g.note_visible(["P001"], 0.0)
    assert g.add_sample("P001", a, 0.0, 0) is None
    assert g.identity("P001").status == PENDING
    assert g.add_sample("P001", a, 0.5, 5).status == NEW
    # P001 leaves; much later P005 appears looking the same
    g.note_visible(["P005"], 30.0)
    g.add_sample("P005", unit([0.98, 0.05, 0, 0]), 30.0, 300)
    d = g.add_sample("P005", unit([0.97, 0.1, 0, 0]), 30.5, 305)
    assert d.status == RECONNECTED and d.profile_id == "P001"
    assert g.tracks_of_profile("P001") == {"P001", "P005"}


def test_gallery_never_merges_people_seen_together():
    g = PersonGallery(ReIDConfig(min_samples_for_decision=1, max_pending_samples=1), "torchreid")
    same = unit([1, 0, 0])
    g.note_visible(["P001", "P002"], 0.0)  # on screen at the same time
    assert g.add_sample("P001", same, 0.0, 0).status == NEW
    g.note_visible(["P001", "P002"], 5.0)
    assert g.add_sample("P002", same, 5.0, 50).status == NEW  # identical look, still a different person


def test_gallery_reports_candidate_when_unsure():
    g = PersonGallery(ReIDConfig(min_samples_for_decision=1, max_pending_samples=3), "torchreid")
    g.add_sample("P001", unit([1, 0]), 0.0, 0)
    g.note_visible(["P009"], 60.0)
    d = g.add_sample("P009", unit([0.6, 0.8]), 60.0, 600)  # cosine 0.6: between candidate and match
    assert d.status == CANDIDATE and d.candidate_id == "P001" and d.profile_id is None


@pytest.mark.parametrize("bgr,name", [((0, 0, 0), "black"), ((255, 255, 255), "white"), ((128, 128, 128), "gray"),
                                      ((0, 0, 220), "red"), ((220, 0, 0), "blue"), ((0, 200, 0), "green"),
                                      ((0, 230, 230), "yellow")])
def test_color_names(bgr, name):
    assert color_name(np.array(bgr, np.float32)) == name


def test_dominant_color_picks_majority():
    pixels = np.vstack([np.tile([200, 30, 30], (700, 1)), np.tile([20, 20, 20], (300, 1))]).astype(np.uint8)
    name, share = dominant_color(pixels)
    assert name == "blue" and share == pytest.approx(0.7, abs=0.01)
