"""Video input, queues, frame accounting, evidence paths and the full pipeline on synthetic video.

The pipeline tests use a colour-blob detector and an IoU tracker (see conftest),
so they check the plumbing, not detection quality.
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import ColorBlobDetector, iou_tracker_factory

from app.datatypes import FrameMeta, GROUP_PERSON, GROUP_VEHICLE, Detection
from app.evidence import (
    ClipRecorder,
    EvidenceError,
    SessionPaths,
    ensure_within,
    purge_old_sessions,
    safe_name,
    save_jpeg,
)
from app.performance import END_OF_STREAM, FrameAccounting, FrameQueue, QueueClosed
from app.pipeline import Pipeline, PipelineError
from app.storage import Storage
from app.tracker import IdRegistry, MultiClassTracker
from app.config import TrackerConfig
from app.video_reader import VideoError, VideoReader, probe_video


# ---- video input --------------------------------------------------------
def test_missing_and_invalid_video_paths(tmp_path):
    with pytest.raises(VideoError, match="No video path"):
        probe_video("")
    with pytest.raises(VideoError, match="not found"):
        probe_video(tmp_path / "missing.mp4")
    with pytest.raises(VideoError, match="folder"):
        probe_video(tmp_path)
    fake = tmp_path / "not_a_video.mp4"
    fake.write_bytes(b"this is not a video")
    with pytest.raises(VideoError):
        probe_video(fake)


def test_probe_reads_metadata(scene_video):
    info = probe_video(scene_video)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(10.0)
    assert info.frame_count == 60
    assert info.duration == pytest.approx(6.0)


def test_reader_complete_mode_keeps_every_frame_in_order(scene_video):
    info = probe_video(scene_video)
    q, acc = FrameQueue(2), FrameAccounting()
    stop, pause = threading.Event(), threading.Event()
    reader = VideoReader(info, q, "complete", acc, stop, pause)
    reader.start()
    seen = []
    while True:
        item = q.get()
        if item is END_OF_STREAM:
            break
        time.sleep(0.002)  # slow consumer: the reader must wait, not drop
        seen.append(item.meta.index)
        assert item.meta.timestamp == pytest.approx(item.meta.index / 10.0)
    reader.join()
    assert seen == list(range(60))
    assert acc.read == 60 and acc.skipped == 0


# ---- queues and accounting ----------------------------------------------
def test_drop_oldest_queue_and_end_marker():
    q = FrameQueue(2)
    assert q.put_drop_oldest(1) == []
    assert q.put_drop_oldest(2) == []
    assert q.put_drop_oldest(3) == [1]
    assert [q.get(), q.get()] == [2, 3]
    q.put_drop_oldest(5)
    q.put_end()  # may exceed maxsize by one so the end is always delivered
    assert len(q) == 2
    assert q.put_drop_oldest(6) == [5]  # the end marker itself is never dropped
    assert [q.get(), q.get()] == [END_OF_STREAM, 6]


def test_blocking_put_stops_on_event():
    q = FrameQueue(1)
    q.put("a")
    stop = threading.Event()
    threading.Timer(0.1, stop.set).start()
    assert q.put("b", stop) is False
    q.get()
    with pytest.raises(QueueClosed):
        q.get(stop)


def test_frame_accounting_balance():
    acc = FrameAccounting()
    acc.add("read", 10)
    for i in range(7):
        acc.mark_processed(i)
    assert not acc.is_balanced()
    acc.add("skipped", 3)
    assert acc.is_balanced() and acc.in_flight == 0


# ---- tracker id handling -------------------------------------------------
def test_id_namespaces_and_reset_on_large_gap():
    reg = IdRegistry()
    assert reg.get(GROUP_VEHICLE, 0, 1) == "V001"
    assert reg.get(GROUP_PERSON, 0, 1) == "P001"
    assert reg.get(GROUP_PERSON, 1, 1) == "P002"  # same raw id after a reset is a new app id

    trk = MultiClassTracker(TrackerConfig(max_gap_frames=5), 10, [GROUP_VEHICLE, GROUP_PERSON], iou_tracker_factory)
    car = lambda: Detection(2, "car", GROUP_VEHICLE, 0.9, np.array([10, 10, 50, 40], np.float32))  # noqa: E731
    out, reset = trk.update([car()], np.zeros((100, 100, 3), np.uint8), 0)
    assert out[0].track_id == "V001" and not reset
    out, reset = trk.update([car()], np.zeros((100, 100, 3), np.uint8), 3)
    assert out[0].track_id == "V001" and not reset
    out, reset = trk.update([car()], np.zeros((100, 100, 3), np.uint8), 20)
    assert reset and trk.resets == 1 and out[0].track_id == "V002"


def test_duplicate_tracks_removed():
    trk = MultiClassTracker(TrackerConfig(), 10, [GROUP_PERSON], iou_tracker_factory)
    a = Detection(0, "person", GROUP_PERSON, 0.9, np.array([10, 10, 30, 60], np.float32))
    b = Detection(0, "person", GROUP_PERSON, 0.8, np.array([11, 10, 31, 61], np.float32))
    a.track_id, b.track_id = "P001", "P002"
    assert [d.track_id for d in trk._remove_duplicates([b, a])] == ["P001"]


# ---- evidence -----------------------------------------------------------
def test_evidence_path_safety(tmp_path):
    assert safe_name("V001/../../etc") == "V001_.._.._etc"
    with pytest.raises(EvidenceError):
        ensure_within(tmp_path, tmp_path / ".." / "outside.jpg")
    paths = SessionPaths.create(tmp_path, "2026 session:1")
    assert paths.root.name == "2026_session_1" and paths.snapshots.is_dir() and paths.clips.is_dir()
    target = tmp_path / "ünïcode dir"
    target.mkdir()
    assert save_jpeg(target / "snap.jpg", np.zeros((20, 20, 3), np.uint8))
    assert (target / "snap.jpg").stat().st_size > 0
    assert not save_jpeg(tmp_path / "missing_dir" / "x.jpg", np.zeros((20, 20, 3), np.uint8))


def test_clip_includes_pre_event_buffer(tmp_path):
    rec = ClipRecorder(tmp_path, pre_seconds=1.0, post_seconds=1.0, max_width=64)
    frame = np.zeros((48, 64, 3), np.uint8)
    for i in range(30):
        rec.push(frame, FrameMeta(i, i / 10))
    path = rec.start("V001", "A0001", FrameMeta(29, 2.9), fps=10)
    for i in range(30, 45):
        rec.push(frame, FrameMeta(i, i / 10))
    rec.close_all()
    clip = rec.finished[0]
    assert clip["error"] is None
    assert clip["frames"] == 11 + 10  # 1 s before (inclusive) + 1 s after
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == clip["frames"]
    cap.release()


def test_retention_only_touches_session_folders(tmp_path):
    old = tmp_path / "old_session"
    old.mkdir()
    (old / "session.json").write_text("{}")
    unrelated = tmp_path / "my_photos"
    unrelated.mkdir()
    import os

    past = time.time() - 40 * 86400
    os.utime(old / "session.json", (past, past))
    os.utime(unrelated, (past, past))
    assert purge_old_sessions(tmp_path, 30) == ["old_session"]
    assert not old.exists() and unrelated.exists()
    assert purge_old_sessions(tmp_path, 0) == []


# ---- full pipeline on synthetic video --------------------------------------
def run_pipeline(settings, video, detector=None, **kw):
    info = probe_video(video)
    p = Pipeline(settings, info, "cpu", False, detector=detector or ColorBlobDetector(),
                 tracker_factory=iou_tracker_factory, display=False, **kw)
    return p, p.run()


def test_complete_mode_processes_every_frame(test_settings, scene_video):
    s = test_settings(**{"app.mode": "complete"})
    p, summary = run_pipeline(s, scene_video)
    st = summary["stats"]
    assert st["frames_read"] == 60 and st["frames_processed"] == 60 and st["frames_skipped"] == 0
    assert summary["frames_balanced"] and summary["reached_end_of_video"]
    av = summary["annotated_video"]
    assert av["reopened"] and av["frame_count"] == 60 and av["frames_written"] == 60
    # session marker, database rows and the observed exit made it to disk
    marker = json.loads((p.paths.root / "session.json").read_text())
    assert marker["status"] == "finished"
    db = Storage(s.storage.db_path)
    assocs = db.list_associations(summary["session_id"])
    db.close()
    assert len(assocs) == 1 and assocs[0]["vehicle_id"] == "V001" and assocs[0]["person_track_id"] == "P001"
    assert assocs[0]["status"] in ("observed_exit", "possible_exit")


def test_responsive_mode_counts_skipped_frames(test_settings, scene_video):
    s = test_settings(**{"app.mode": "responsive", "pipeline.input_queue_size": 1})
    _, summary = run_pipeline(s, scene_video, detector=ColorBlobDetector(delay=0.25))
    st = summary["stats"]
    assert st["frames_skipped"] > 0
    assert st["frames_processed"] + st["frames_skipped"] == st["frames_read"] == 60
    assert summary["frames_balanced"]


def test_max_frames_and_start_frame(test_settings, scene_video):
    s = test_settings(**{"app.mode": "complete", "evidence.save_annotated_video": False})
    _, summary = run_pipeline(s, scene_video, max_frames=10, start_frame=5)
    assert summary["stats"]["frames_processed"] == 10 and summary["frames_balanced"]


def test_suspicious_scene_produces_alert_with_evidence(test_settings, tmp_path):
    from conftest import make_theft_scene_video

    video = make_theft_scene_video(tmp_path / "theft.mp4")
    s = test_settings(**{"app.mode": "complete", "zones.park_seconds": 2.0})
    p, summary = run_pipeline(s, video)
    assert summary["frames_balanced"]
    db = Storage(s.storage.db_path)
    events = db.list_events(summary["session_id"])
    db.close()
    kinds = {e["category"] for e in events}
    assert {"vehicle_parked", "proximity", "apparent_entry", "vehicle_moved"} <= kinds
    alerts = [e for e in events if e["risk_level"] != "INFORMATIONAL"]
    assert alerts, "expected at least one risk alert"
    assert any(a["category"] == "possible unauthorised vehicle movement" for a in alerts)
    evidenced = [a for a in alerts if a["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")]
    assert evidenced
    for a in evidenced:
        assert a["snapshot_path"] and Path(a["snapshot_path"]).stat().st_size > 0
        assert a["clip_path"] and probe_video(a["clip_path"]).frame_count > 0
        assert a["video_ts"] > 0 and a["frame"] > 0 and "unrecognised person" in a["message"]
    jsonl = (p.paths.root / "events.jsonl").read_text().splitlines()
    assert len(jsonl) == len(alerts)


class ScriptedWindow:
    """Stands in for the OpenCV window and replays a list of key presses.

    Start-up status frames (show_status / poll_quit) are recorded separately and never
    consume the scripted keys, which are meant for the main display loop.
    """

    def __init__(self, keys, quit_during_startup=False):
        self.keys = list(keys)
        self.shown = 0
        self.status_frames = []
        self.quit_during_startup = quit_during_startup
        self.opened = True

    def show(self, frame):
        self.shown += 1

    def show_status(self, frame):
        self.status_frames.append(frame)

    def poll_quit(self, wait_ms=1):
        time.sleep(wait_ms / 1000)
        return self.quit_during_startup

    def poll_key(self, wait_ms=1):
        time.sleep(wait_ms / 1000)
        return self.keys.pop(0) if self.keys and self.shown >= 5 else -1

    def closed_by_user(self):
        return False

    def close(self):
        self.opened = False


def test_keyboard_controls_and_quit(test_settings, scene_video):
    s = test_settings(**{"app.mode": "complete", "app.display": True})
    info = probe_video(scene_video)
    window = ScriptedWindow([ord(" "), ord(" "), ord("s"), ord("r"), ord("q")])
    p = Pipeline(s, info, "cpu", False, detector=ColorBlobDetector(), tracker_factory=iou_tracker_factory,
                 display=True, window=window)
    summary = p.run()
    assert p.renderer.mode_name == "boxes"  # 'r' cycled the overlay once
    assert list(p.paths.snapshots.glob("manual_*.jpg"))  # 's' saved a snapshot
    st = summary["stats"]
    assert st["frames_read"] < 60  # 'q' stopped before the end
    assert summary["frames_balanced"] and not summary["reached_end_of_video"]
    assert not p.display.opened


def test_detector_failure_shuts_down_cleanly(test_settings, scene_video):
    s = test_settings(**{"app.mode": "complete"})
    info = probe_video(scene_video)
    p = Pipeline(s, info, "cpu", False, detector=ColorBlobDetector(fail_after=5),
                 tracker_factory=iou_tracker_factory, display=False)
    with pytest.raises(PipelineError, match="simulated detector failure"):
        p.run()
    marker = json.loads((p.paths.root / "session.json").read_text())
    assert marker["status"] == "error"
    assert marker["frames_balanced"]
    assert not p.reader.is_alive()
