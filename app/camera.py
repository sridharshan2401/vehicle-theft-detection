"""Live camera input: the laptop webcam (index 0) or another connected camera.

CameraSource opens the device once and owns the cv2.VideoCapture; CameraReader
is the live counterpart of video_reader.VideoReader and feeds the same
FrameQueue, so the rest of the pipeline does not care where frames come from.

Differences from a video file:

* there is no end and no way to "catch up": the reader always keeps the newest
  frame and drops older ones the analysis has not reached (counted as skipped);
* timestamps are wall-clock seconds since capture started, so durations such as
  "lingered 1.5 s" or "moved for 1 s" are real time;
* a run of failed reads means the camera was unplugged, disabled or taken by
  another application, and is reported as a CameraError.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable

import cv2
import numpy as np

from app.config import CameraConfig
from app.datatypes import FrameMeta, FramePacket
from app.performance import FrameAccounting, FrameQueue, RateMeter
from app.video_reader import VideoInfo, _fourcc_to_str

log = logging.getLogger(__name__)

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
DEFAULT_CAMERA_FPS = 30.0


class CameraError(RuntimeError):
    """The camera is missing, busy, blocked by privacy settings, or stopped delivering frames."""


def backend_order(name: str) -> list[str]:
    if name != "auto":
        return [name]
    # DirectShow opens quickly on most Windows laptops; Media Foundation is the fallback.
    return ["dshow", "msmf", "any"] if sys.platform == "win32" else ["any"]


class CameraSource:
    """One camera, opened once and released once (release is safe to call repeatedly)."""

    def __init__(self, cfg: CameraConfig, capture_factory: Callable | None = None):
        self.cfg = cfg
        self._factory = capture_factory or cv2.VideoCapture
        self.cap = None
        self.backend_used: str | None = None
        self.first_frame: np.ndarray | None = None
        self.released = False

    @property
    def label(self) -> str:
        return f"camera:{self.cfg.index}"

    def open(self) -> VideoInfo:
        """Open the camera, wait for a real frame, and describe the feed. Raises CameraError."""
        problems = []
        for name in backend_order(self.cfg.backend):
            try:
                cap = self._factory(self.cfg.index, BACKENDS[name])
            except Exception as exc:  # some backends throw instead of returning a closed capture
                problems.append(f"{name}: {exc}")
                continue
            if cap is None or not cap.isOpened():
                problems.append(f"{name}: could not open the device")
                self._safe_release(cap)
                continue
            self._configure(cap, name)
            frame, still_reading = self._first_frame(cap)
            if frame is None:
                problems.append(f"{name}: opened but sent no picture within {self.cfg.open_timeout_seconds:g} s")
                if not still_reading:  # never release a capture another thread is blocked inside
                    self._safe_release(cap)
                continue
            self.cap, self.backend_used, self.first_frame = cap, name, frame
            h, w = frame.shape[:2]
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fps_ok = 1.0 <= fps <= 240.0
            fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            log.info("Opened %s via %s: %dx%d, driver FPS %s", self.label, name, w, h, f"{fps:.1f}" if fps_ok else "?")
            return VideoInfo(self.label, w, h, fps if fps_ok else (self.cfg.fps or DEFAULT_CAMERA_FPS), fps_ok,
                             None, None, f"{fourcc or '?'} via {name}", live=True)
        raise CameraError(
            f"Camera {self.cfg.index} could not be used ({'; '.join(problems)}).\n"
            "Check that:\n"
            "  - the camera is connected and enabled (some laptops have a privacy shutter or an F-key to turn it on);\n"
            "  - no other application is using it (Camera app, Teams, Zoom, a browser tab...);\n"
            "  - Windows Settings > Privacy & security > Camera allows desktop apps to use the camera;\n"
            "  - the index is right: try --camera 1 for an external camera."
        )

    def _configure(self, cap, backend: str) -> None:
        c = self.cfg
        if backend == "dshow":
            # MJPG lets most USB/laptop webcams deliver 720p at full frame rate over DirectShow.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if c.width and c.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        if c.fps:
            cap.set(cv2.CAP_PROP_FPS, c.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by some backends; keeps latency down where honoured

    def _first_frame(self, cap) -> tuple[np.ndarray | None, bool]:
        """Read until a real frame arrives or the timeout passes. Returns (frame or None, reader still blocked).

        A busy camera often opens fine but then blocks or returns nothing, so the reads
        happen on a helper thread and are abandoned after the timeout.
        """
        result: dict = {}
        deadline = time.monotonic() + self.cfg.open_timeout_seconds

        def worker():
            while time.monotonic() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None and frame.size and frame.any():
                    result["frame"] = frame
                    return
                time.sleep(0.05)

        t = threading.Thread(target=worker, name="camera-open", daemon=True)
        t.start()
        t.join(self.cfg.open_timeout_seconds + 1.0)
        return result.get("frame"), t.is_alive()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None or self.released:
            return False, None
        return self.cap.read()

    @staticmethod
    def _safe_release(cap) -> None:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

    def release(self) -> None:
        if not self.released:
            self._safe_release(self.cap)
            self.released = True
            if self.cap is not None:
                log.info("Released %s", self.label)


class CameraReader:
    """Reads the live camera in a background thread; same interface as VideoReader."""

    def __init__(self, source: CameraSource, info: VideoInfo, out_queue: FrameQueue, accounting: FrameAccounting,
                 stop_event: threading.Event, pause_event: threading.Event, max_frames: int | None = None,
                 clock: Callable[[], float] = time.perf_counter, max_failed_reads: int = 50):
        self.source = source
        self.info = info
        self.queue = out_queue
        self.accounting = accounting
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.max_frames = max_frames
        self.clock = clock
        self.max_failed_reads = max_failed_reads
        self.error: BaseException | None = None
        self.finished_cleanly = False
        self.fps_meter = RateMeter()
        self.last_frame_time: float | None = None
        self.failed_reads = 0
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)

    @property
    def status(self) -> str:
        if self.error is not None:
            return "DISCONNECTED"
        if self.pause_event.is_set():
            return "paused"
        if self.last_frame_time is None or time.monotonic() - self.last_frame_time > 1.0:
            return "waiting for frames"
        return "connected"

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread.ident is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            t0 = self.clock()
            index = 0
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    self.stop_event.wait(0.02)  # stop reading while paused; the window keeps the last frame
                    continue
                ok, frame = self.source.read()
                if not ok or frame is None:
                    self.failed_reads += 1
                    if self.failed_reads >= self.max_failed_reads:
                        raise CameraError(
                            f"{self.source.label} stopped delivering frames ({self.failed_reads} failed reads in a row). "
                            "It may have been unplugged, disabled, or taken over by another application."
                        )
                    self.stop_event.wait(0.02)
                    continue
                self.failed_reads = 0
                self.last_frame_time = time.monotonic()
                self.fps_meter.tick()
                packet = FramePacket(frame, FrameMeta(index, self.clock() - t0))
                self.accounting.add("read")
                dropped = self.queue.put_drop_oldest(packet)  # live: the newest frame always wins
                if dropped:
                    self.accounting.add("skipped", len(dropped))
                index += 1
                if self.max_frames is not None and index >= self.max_frames:
                    break
            self.finished_cleanly = not self.stop_event.is_set()
        except BaseException as exc:  # reported by the pipeline
            self.error = exc
            log.error("Camera reader stopped: %s", exc)
        finally:
            self.queue.put_end()
