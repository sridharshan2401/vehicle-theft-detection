"""Video file probing and the background frame reader."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from app.datatypes import FrameMeta, FramePacket
from app.performance import FrameAccounting, FrameQueue

log = logging.getLogger(__name__)

KNOWN_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}
DEFAULT_FPS = 30.0


class VideoError(RuntimeError):
    """The video is missing, unreadable or cannot be decoded."""


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    fps_reported: bool  # False when the container had no usable FPS and we assumed DEFAULT_FPS
    frame_count: int | None
    duration: float | None
    fourcc: str
    live: bool = False  # a camera feed: no end, timestamps are wall-clock seconds since capture started

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        if self.live:
            fps_note = "" if self.fps_reported else " (not reported by the driver)"
            return f"{self.path}: {self.width}x{self.height}, {self.fps:.1f} FPS{fps_note}, live feed, format {self.fourcc or '?'}"
        dur = f"{self.duration:.1f}s" if self.duration else "unknown"
        frames = self.frame_count if self.frame_count else "unknown"
        fps_note = "" if self.fps_reported else f" (not reported, assuming {DEFAULT_FPS:g})"
        return (
            f"{Path(self.path).name}: {self.width}x{self.height}, {self.fps:.2f} FPS{fps_note}, "
            f"{frames} frames, duration {dur}, codec {self.fourcc or '?'}"
        )


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    chars = [chr((code >> 8 * i) & 0xFF) for i in range(4)]
    return "".join(c for c in chars if c.isprintable()).strip()


def probe_video(path: str | Path) -> VideoInfo:
    """Open a video, decode one frame and return its metadata.

    Raises VideoError with a message a user can act on.
    """
    if path is None or not str(path).strip():
        raise VideoError("No video path was given.")
    p = Path(str(path).strip().strip('"')).expanduser()
    if not p.exists():
        raise VideoError(f"Video file not found: {p}")
    if p.is_dir():
        raise VideoError(f"Expected a video file but got a folder: {p}")
    if p.suffix.lower() not in KNOWN_EXTENSIONS:
        log.warning("Unusual video extension %r; trying to open it anyway.", p.suffix)

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError(
                f"OpenCV could not open {p.name}. The file may be corrupt or use a codec this "
                "OpenCV build cannot decode. Try re-encoding it to H.264 MP4."
            )
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError(f"{p.name} opened but no frame could be decoded (empty or unsupported codec).")
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_ok = fps is not None and not math.isnan(fps) and 1.0 <= fps <= 240.0
        if not fps_ok:
            fps = DEFAULT_FPS
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = count if count > 0 else None
        duration = frame_count / fps if frame_count else None
        fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        cap.release()
    return VideoInfo(str(p.resolve()), width, height, float(fps), fps_ok, frame_count, duration, fourcc)


class VideoReader:
    """Reads frames in a background thread and feeds a bounded FrameQueue.

    responsive mode: frames are released at the source frame rate. If the
    analysis worker falls behind, the oldest waiting frame is dropped and
    counted as skipped.

    complete mode: frames are read as fast as the queue accepts them and
    the reader blocks when the queue is full, so no frame is ever dropped.
    """

    def __init__(
        self,
        info: VideoInfo,
        out_queue: FrameQueue,
        mode: str,
        accounting: FrameAccounting,
        stop_event: threading.Event,
        pause_event: threading.Event,
        max_frames: int | None = None,
        start_frame: int = 0,
    ):
        self.info = info
        self.queue = out_queue
        self.mode = mode
        self.accounting = accounting
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.max_frames = max_frames
        self.start_frame = max(0, start_frame)
        self.error: BaseException | None = None
        self.finished_cleanly = False
        self._thread = threading.Thread(target=self._run, name="video-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread.ident is not None:  # joining a never-started thread raises
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.info.path)
        try:
            if not cap.isOpened():
                raise VideoError(f"Could not reopen {self.info.path}")
            if self.start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            index = self.start_frame
            fps = self.info.fps
            clock_start = time.perf_counter()
            paused_total = 0.0
            emitted = 0
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    t_pause = time.perf_counter()
                    while self.pause_event.is_set() and not self.stop_event.is_set():
                        time.sleep(0.02)
                    paused_total += time.perf_counter() - t_pause
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                packet = FramePacket(frame, FrameMeta(index, index / fps))
                self.accounting.add("read")
                if self.mode == "responsive":
                    due = clock_start + paused_total + (index - self.start_frame) / fps
                    delay = due - time.perf_counter()
                    if delay > 0:
                        self.stop_event.wait(delay)
                    dropped = self.queue.put_drop_oldest(packet)
                    if dropped:
                        self.accounting.add("skipped", len(dropped))
                        log.debug("analysis behind: dropped frame(s) %s", [p.meta.index for p in dropped])
                else:
                    if not self.queue.put(packet, self.stop_event):
                        # Stopped by the user before this frame could be queued.
                        self.accounting.add("skipped")
                        break
                index += 1
                emitted += 1
                if self.max_frames is not None and emitted >= self.max_frames:
                    break
            self.finished_cleanly = not self.stop_event.is_set()
        except BaseException as exc:  # reported by the pipeline
            self.error = exc
            log.exception("Video reader failed")
        finally:
            cap.release()
            self.queue.put_end()
