"""Live camera mode, tested without a physical webcam.

FakeCapture stands in for cv2.VideoCapture and replays synthetic frames, so these
tests check the camera handling and the live pipeline wiring (opening, failures,
frame dropping, shutdown, alerts). They say nothing about real camera quality.
"""

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import ColorBlobDetector, iou_tracker_factory, make_theft_scene_video

from app import camera as camera_mod
from app.camera import CameraError, CameraReader, CameraSource
from app.config import CameraConfig, ConfigError, load_config
from app.datatypes import FrameMeta
from app.performance import END_OF_STREAM, FrameAccounting, FrameQueue
from app.pipeline import Pipeline, PipelineError
from app.storage import Storage


class FakeCapture:
    """Minimal cv2.VideoCapture double."""

    def __init__(self, frames=None, opened=True, fail_after=None, read_delay=0.0, fps=30.0):
        self.frames = frames if frames is not None else [np.full((480, 640, 3), 90, np.uint8)]
        self.opened = opened
        self.fail_after = fail_after  # reads succeed for this many frames, then fail forever
        self.read_delay = read_delay
        self.fps = fps
        self.i = 0
        self.read_calls = 0
        self.release_calls = 0
        self.props = {}

    def isOpened(self):
        return self.opened and self.release_calls == 0

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG")}.get(prop, 0)

    def read(self):
        self.read_calls += 1
        if self.read_delay:
            time.sleep(self.read_delay)
        if not self.opened or self.release_calls or (self.fail_after is not None and self.i >= self.fail_after):
            return False, None
        frame = self.frames[self.i % len(self.frames)].copy()
        self.i += 1
        return True, frame

    def release(self):
        self.release_calls += 1


def factory(*caps):
    """capture_factory returning the given fakes in order (one per backend tried)."""
    calls = []
    queue = list(caps)

    def make(index, backend):
        calls.append((index, backend))
        return queue.pop(0) if len(queue) > 1 else queue[0]

    make.calls = calls
    return make


@pytest.fixture(autouse=True)
def windows_backends(monkeypatch):
    monkeypatch.setattr(camera_mod.sys, "platform", "win32")  # same backend order on every OS


# ---- opening and failures ---------------------------------------------------
def test_open_reports_live_feed_and_requests_resolution():
    cap = FakeCapture()
    src = CameraSource(CameraConfig(index=0, width=1280, height=720, fps=30), factory(cap))
    info = src.open()
    assert info.live and info.path == "camera:0"
    assert (info.width, info.height) == (640, 480)  # what the camera actually delivered
    assert info.fps == 30.0 and info.fps_reported
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280 and cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert src.backend_used == "dshow" and src.first_frame.shape == (480, 640, 3)
    assert "live feed" in info.describe()


def test_camera_index_is_passed_to_opencv():
    make = factory(FakeCapture())
    CameraSource(CameraConfig(index=2), make).open()
    assert make.calls[0] == (2, cv2.CAP_DSHOW)


def test_unavailable_camera_gives_actionable_error():
    caps = [FakeCapture(opened=False) for _ in range(3)]
    make = factory(*caps)
    with pytest.raises(CameraError) as err:
        CameraSource(CameraConfig(index=0), make).open()
    msg = str(err.value)
    assert "Camera 0 could not be used" in msg
    assert "no other application is using it" in msg and "Privacy" in msg and "--camera 1" in msg
    assert [b for _, b in make.calls] == [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]


def test_busy_camera_that_opens_but_sends_nothing():
    caps = [FakeCapture(fail_after=0) for _ in range(3)]
    with pytest.raises(CameraError, match="sent no picture"):
        CameraSource(CameraConfig(open_timeout_seconds=0.3), factory(*caps)).open()
    assert all(c.release_calls == 1 for c in caps)  # every failed attempt was released


def test_black_startup_frames_are_skipped():
    frames = [np.zeros((480, 640, 3), np.uint8)] * 3 + [np.full((480, 640, 3), 120, np.uint8)]
    src = CameraSource(CameraConfig(), factory(FakeCapture(frames)))
    src.open()
    assert src.first_frame.mean() == 120


def test_falls_back_to_the_next_backend():
    good = FakeCapture()
    src = CameraSource(CameraConfig(), factory(FakeCapture(opened=False), good))
    src.open()
    assert src.backend_used == "msmf" and src.cap is good


def test_release_is_idempotent():
    cap = FakeCapture()
    src = CameraSource(CameraConfig(), factory(cap))
    src.open()
    src.release()
    src.release()
    assert cap.release_calls == 1 and src.released
    assert src.read() == (False, None)


def test_camera_settings_are_validated():
    with pytest.raises(ConfigError, match="camera.index"):
        load_config(None, {"camera.index": -1})
    with pytest.raises(ConfigError, match="camera.backend"):
        load_config(None, {"camera.backend": "v4l2"})
    with pytest.raises(ConfigError, match="app.source"):
        load_config(None, {"app.source": "rtsp"})
    assert load_config().app.source == "camera"


# ---- the reader ------------------------------------------------------------
def make_reader(cap, max_frames=None, queue_size=2, max_failed_reads=50, clock=time.perf_counter):
    src = CameraSource(CameraConfig(), factory(cap))
    info = src.open()
    q, acc = FrameQueue(queue_size), FrameAccounting()
    stop, pause = threading.Event(), threading.Event()
    reader = CameraReader(src, info, q, acc, stop, pause, max_frames, clock, max_failed_reads)
    return reader, q, acc, stop, pause


def test_reader_keeps_the_newest_frames_and_counts_drops():
    reader, q, acc, _, _ = make_reader(FakeCapture([np.full((48, 64, 3), v, np.uint8) for v in range(1, 255)]),
                                       max_frames=50)
    reader.start()
    reader.join(5)
    items = q.drain()
    assert items[-1] is END_OF_STREAM
    kept = [p.meta.index for p in items[:-1]]
    assert kept == [48, 49]  # the consumer never read, so only the newest two survive
    assert acc.read == 50 and acc.skipped == 48
    assert reader.finished_cleanly and reader.error is None


def test_reader_timestamps_come_from_the_clock():
    ticks = iter(np.arange(0, 100, 0.5))
    reader, q, _, _, _ = make_reader(FakeCapture(), max_frames=3, queue_size=5, clock=lambda: float(next(ticks)))
    reader.start()
    reader.join(5)
    assert [p.meta.timestamp for p in q.drain()[:-1]] == [0.5, 1.0, 1.5]


def test_reader_reports_disconnect():
    cap = FakeCapture()
    reader, q, acc, _, _ = make_reader(cap, max_failed_reads=3)
    cap.fail_after = cap.i + 5  # five more frames, then the camera "unplugs"
    reader.start()
    reader.join(5)
    assert isinstance(reader.error, CameraError) and "stopped delivering frames" in str(reader.error)
    assert reader.status == "DISCONNECTED"
    assert q.drain()[-1] is END_OF_STREAM and acc.read == 5


def test_reader_does_not_read_while_paused():
    cap = FakeCapture(read_delay=0.001)
    reader, _, _, stop, pause = make_reader(cap)
    pause.set()
    before = cap.read_calls
    reader.start()
    time.sleep(0.2)
    assert cap.read_calls == before and reader.status == "paused"
    pause.clear()
    time.sleep(0.1)
    assert cap.read_calls > before
    stop.set()
    reader.join(5)


# ---- the full live pipeline (synthetic camera, real analysis logic) ----------
def theft_frames(tmp_path) -> list[np.ndarray]:
    video = make_theft_scene_video(tmp_path / "theft.mp4")
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def live_pipeline(settings, cap, **kw):
    src = CameraSource(settings.camera, factory(cap))
    info = src.open()
    # Scene time: the synthetic scene was made at 10 FPS; frames are delivered faster than that.
    p = Pipeline(settings, info, "cpu", False, detector=kw.pop("detector", ColorBlobDetector()),
                 tracker_factory=iou_tracker_factory, display=False, camera=src,
                 clock=lambda: cap.i / 10.0, **kw)
    return p, src


def test_live_pipeline_raises_explainable_alerts_without_recording(test_settings, tmp_path):
    frames = theft_frames(tmp_path)
    cap = FakeCapture(frames, read_delay=0.008)
    s = test_settings(**{"zones.park_seconds": 2.0})
    p, src = live_pipeline(s, cap, max_frames=len(frames) - 1)
    summary = p.run()

    assert summary["mode"] == "live" and summary["source"] == "camera:0"
    assert summary["frames_balanced"]
    assert src.released and cap.release_calls == 1
    # no full recording and no clips in live mode by default
    assert "annotated_video" not in summary and not p.paths.annotated_video.exists()
    assert list(p.paths.clips.iterdir()) == []
    db = Storage(s.storage.db_path)
    events = db.list_events(summary["session_id"])
    session = db.get_session(summary["session_id"])
    db.close()
    assert session["mode"] == "live" and session["video_path"] == "camera:0"
    kinds = {e["category"] for e in events}
    assert {"vehicle_parked", "apparent_entry", "vehicle_moved"} <= kinds
    alerts = [e for e in events if e["risk_score"] is not None]
    assert any(a["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL") for a in alerts)
    for a in alerts:
        assert "unrecognised person" in a["message"]
        assert "stolen" not in a["message"].lower() and "thief" not in a["message"].lower()
        if a["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL"):
            assert a["snapshot_path"] and Path(a["snapshot_path"]).exists()
            assert a["clip_path"] is None

    # what the window would show at the end of the run
    ctx = p._render_context([], [], FrameMeta(0, frames and len(frames) / 10.0), 640)
    assert "not proof of theft" in ctx.banner
    assert any(line.startswith("stolen-vehicle check: NOT AVAILABLE") for line in ctx.hud)
    assert any(line.startswith("risk: ") for line in ctx.hud)
    assert any(line.startswith("LIVE camera:0") for line in ctx.hud)


def test_live_alert_clips_and_recording_are_opt_in(test_settings, tmp_path):
    frames = theft_frames(tmp_path)
    cap = FakeCapture(frames, read_delay=0.008)
    s = test_settings(**{"zones.park_seconds": 2.0, "camera.save_alert_clips": True, "camera.save_annotated_video": True})
    p, _ = live_pipeline(s, cap, max_frames=len(frames) - 1)
    summary = p.run()
    assert summary["annotated_video"]["reopened"]
    assert list(p.paths.clips.glob("*.mp4"))


def test_live_pipeline_reports_camera_disconnect_and_releases_it(test_settings):
    cap = FakeCapture(fail_after=30)
    s = test_settings(**{"camera.max_failed_reads": 5})
    p, src = live_pipeline(s, cap)
    with pytest.raises(PipelineError, match="Camera failed") as err:
        p.run()
    assert isinstance(err.value.__cause__, CameraError)
    assert src.released
    marker = json.loads((p.paths.root / "session.json").read_text())
    assert marker["status"] == "error" and marker["frames_balanced"]


def test_live_pipeline_releases_camera_when_analysis_fails(test_settings):
    cap = FakeCapture(read_delay=0.005)
    p, src = live_pipeline(test_settings(), cap, detector=ColorBlobDetector(fail_after=3))
    with pytest.raises(PipelineError, match="simulated detector failure"):
        p.run()
    assert src.released and not p.reader.is_alive()


def test_live_pipeline_releases_camera_when_setup_fails(test_settings, tmp_path):
    bad_zones = tmp_path / "zones.json"
    bad_zones.write_text("{not json")
    cap = FakeCapture()
    s = test_settings(**{"zones.file": str(bad_zones)})
    p, src = live_pipeline(s, cap)
    with pytest.raises(Exception):
        p.run()
    assert src.released


def test_live_mode_forces_responsive(test_settings):
    p, _ = live_pipeline(test_settings(**{"app.mode": "complete"}), FakeCapture())
    assert p.mode == "responsive" and p.in_queue.maxsize == 2


# ---- the entry point -----------------------------------------------------------
def test_default_launch_uses_the_camera_and_reports_failure(monkeypatch, capsys):
    from app import main as main_mod

    opened = []

    class NoCamera(CameraSource):
        def open(self):
            opened.append(self.cfg.index)
            raise CameraError("Camera 0 could not be used (test)")

    monkeypatch.setattr(main_mod, "CameraSource", NoCamera)
    assert main_mod.main(["--no-display", "--device", "cpu"]) == 5
    assert opened == [0]
    assert main_mod.main(["--no-display", "--device", "cpu", "--camera", "3"]) == 5
    assert opened == [0, 3]


def test_video_flag_still_selects_a_file(monkeypatch, tmp_path):
    from app import main as main_mod

    monkeypatch.setattr(main_mod, "CameraSource", lambda *a, **k: pytest.fail("camera must not be opened"))
    assert main_mod.main(["--no-display", "--device", "cpu", "--video", str(tmp_path / "missing.mp4")]) == 2
    assert main_mod.main(["--camera", "1", "--video", "x.mp4"]) == 2  # contradictory inputs


def test_resolution_argument():
    from app.cli import build_parser, overrides_from_args

    o = overrides_from_args(build_parser().parse_args(["--resolution", "640x480", "--camera", "1"]))
    assert (o["camera.width"], o["camera.height"], o["camera.index"]) == (640, 480, 1)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--resolution", "big"])


# ---- start-up window and launcher ------------------------------------------------
class SlowWarmupDetector(ColorBlobDetector):
    """Colour-blob detector whose GPU warm-up takes a moment, like the real one."""

    def warmup(self, size, frame_shape=None, runs=2):
        time.sleep(0.4)


def test_window_shows_live_camera_while_starting_up(test_settings):
    from test_video_pipeline import ScriptedWindow

    frame = np.full((240, 320, 3), 80, np.uint8)
    cap = FakeCapture([frame], read_delay=0.01)
    window = ScriptedWindow([ord("q")])
    s = test_settings(**{"app.display": True})
    src = CameraSource(s.camera, factory(cap))
    info = src.open()
    p = Pipeline(s, info, "cpu", False, detector=SlowWarmupDetector(), tracker_factory=iou_tracker_factory,
                 display=True, window=window, camera=src, clock=lambda: cap.i / 10.0)
    summary = p.run()
    # start-up frames were shown before any analysis, and they are the camera picture (darkened, with text)
    assert len(window.status_frames) >= 3
    status = window.status_frames[0]
    assert status.shape == frame.shape and int(np.median(status)) == int(80 * 0.55)
    # then analysed frames were displayed until 'q'
    assert window.shown >= 5 and summary["stats"]["frames_displayed"] == window.shown
    assert summary["mode"] == "live" and summary["frames_balanced"]
    assert src.released and not window.opened


def test_quit_during_startup_releases_camera_and_analyses_nothing(test_settings):
    from test_video_pipeline import ScriptedWindow

    cap = FakeCapture(read_delay=0.01)
    window = ScriptedWindow([], quit_during_startup=True)
    s = test_settings(**{"app.display": True})
    src = CameraSource(s.camera, factory(cap))
    info = src.open()
    p = Pipeline(s, info, "cpu", False, detector=SlowWarmupDetector(), tracker_factory=iou_tracker_factory,
                 display=True, window=window, camera=src)
    summary = p.run()
    assert window.status_frames and window.shown == 0
    assert summary["status"] == "finished" and summary["stats"]["frames_read"] == 0
    assert p.reader._thread.ident is None  # the camera reader never started
    assert src.released and not window.opened


def test_launcher_passes_options_through(monkeypatch):
    import run_webcam

    import app.main as main_mod

    seen = []
    monkeypatch.setattr(main_mod, "main", lambda argv: seen.append(argv) or 0)
    assert run_webcam.main(["--camera", "1"]) == 0
    assert run_webcam.main([]) == 0
    assert seen == [["--camera", "1"], []]


def test_launcher_refuses_video_and_wrong_python(monkeypatch, capsys):
    import run_webcam

    assert run_webcam.main(["--video", "clip.mp4"]) == 2
    assert run_webcam.main(["--pick-video"]) == 2
    real_find = run_webcam.importlib.util.find_spec
    monkeypatch.setattr(run_webcam.importlib.util, "find_spec", lambda name: None if name == "torch" else real_find(name))
    assert run_webcam.main([]) == 2
    out = capsys.readouterr().out
    assert "is missing: torch" in out and "virtual environment" in out


def test_cpu_fallback_is_announced(monkeypatch, capsys):
    from app import main as main_mod

    class NoCamera(CameraSource):
        def open(self):
            raise CameraError("Camera 0 could not be used (test)")

    monkeypatch.setattr(main_mod, "CameraSource", NoCamera)
    monkeypatch.setattr(main_mod, "select_device", lambda *a: ("cpu", False))
    assert main_mod.main(["--no-display"]) == 5
    out = capsys.readouterr().out
    assert "Python: " in out and "WARNING: no usable CUDA GPU" in out
