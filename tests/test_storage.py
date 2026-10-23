import threading

import pytest

from app.storage import Storage, StorageError


@pytest.fixture
def db(tmp_path):
    s = Storage(tmp_path / "sub" / "test.sqlite3")
    yield s
    s.close()


def test_event_round_trip(db):
    sid = db.create_session("C:/videos/lot.mp4", "complete", {"fps": 25}, {"x": 1})
    eid = db.add_event(session_id=sid, video_path="C:/videos/lot.mp4", frame=120, video_ts=4.8,
                       category="suspicious interaction", risk_level="HIGH", vehicle_id="V001",
                       person_track_id="P003", person_identity="new:P003:None", risk_score=61.5,
                       reasons=[{"kind": "apparent_entry", "points": 30}], confidences={"apparent_entry": 0.7},
                       message="High-priority alert", snapshot_path="snap.jpg", clip_path="clip.mp4",
                       processing_stats={"frames_skipped": 3})
    rows = db.list_events(sid)
    assert len(rows) == 1
    r = rows[0]
    assert r["event_id"] == eid and r["frame"] == 120 and r["video_ts"] == 4.8
    assert r["risk_level"] == "HIGH" and r["vehicle_id"] == "V001" and r["person_track_id"] == "P003"
    assert '"apparent_entry"' in r["reasons"] and '"frames_skipped": 3' in r["processing_stats"]
    db.update_event_clip(eid, None)
    assert db.list_events(sid)[0]["clip_path"] is None


def test_session_end_and_delete(db):
    sid = db.create_session("v.mp4", "responsive")
    db.add_event(session_id=sid, video_path="v.mp4", frame=1, video_ts=0.1, category="x", risk_level="LOW")
    db.end_session(sid, {"frames_read": 10})
    assert db.list_sessions()[0]["ended_at"] is not None
    db.delete_session(sid)
    assert db.list_sessions() == [] and db.list_events() == []


def test_association_review_is_separate_from_authorisation(db):
    sid = db.create_session("v.mp4", "complete")
    aid = db.add_association(sid, "P001", "V001", "observed_exit", 0.82, 50, 5.0, {"components": {}})
    assert db.list_associations(sid)[0]["review_status"] == "unreviewed"
    db.review_association(aid, "confirmed", "checked footage")
    row = db.list_associations(sid)[0]
    assert row["review_status"] == "confirmed" and row["review_note"] == "checked footage"
    # confirming an association never creates an authorisation
    assert db.list_people() == []
    with pytest.raises(StorageError):
        db.review_association(aid, "approved")
    with pytest.raises(StorageError):
        db.review_association(9999, "rejected")


def test_registry_and_authorisation(db):
    vid = db.register_vehicle("Blue hatchback", "KA01AB1234", "A1")
    pid = db.register_person("Owner", "data/registry/owner.npy", "torchreid")
    other = db.register_person("Visitor", None, None)
    db.authorize(vid, pid)
    db.authorize(vid, pid)  # idempotent
    assert db.vehicle_for_zone("A1")["vehicle_record_id"] == vid
    assert db.vehicle_for_zone("B7") is None
    assert [p["label"] for p in db.authorized_people_for_vehicle(vid)] == ["Owner"]
    db.revoke(vid, pid)
    assert db.authorized_people_for_vehicle(vid) == []
    assert other != pid
    with pytest.raises(StorageError):
        db.authorize(999, pid)


def test_concurrent_writes(db):
    sid = db.create_session("v.mp4", "complete")

    def writer(n):
        for i in range(25):
            db.add_event(session_id=sid, video_path="v.mp4", frame=n * 100 + i, video_ts=i, category="t", risk_level="LOW")

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(db.list_events(sid)) == 100


def test_unwritable_location_reports_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(StorageError):
        Storage(blocker / "db.sqlite3")
