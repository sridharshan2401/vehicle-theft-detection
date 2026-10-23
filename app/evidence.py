"""Evidence on disk: session folders, snapshots, incident clips and the annotated video.

Layout of one session (session id = start time + random suffix):

    evidence/<session_id>/
        session.json        video metadata, settings, final statistics (marker file)
        events.jsonl        one JSON line per recorded event / alert
        annotated.mp4       full annotated output (the source video is never modified)
        snapshots/*.jpg     full-resolution annotated frame at the moment of an alert
        clips/*.mp4         pre-event buffer + post-event frames around an alert

Every write is checked; failures are logged and returned to the caller instead
of being silently ignored.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.datatypes import FrameMeta

log = logging.getLogger(__name__)
MARKER = "session.json"


class EvidenceError(RuntimeError):
    pass


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80] or "item"


def ensure_within(base: Path, path: Path) -> Path:
    """Refuse paths that would escape the evidence folder."""
    base_r, path_r = base.resolve(), path.resolve()
    if base_r != path_r and base_r not in path_r.parents:
        raise EvidenceError(f"{path} is outside the evidence folder {base}")
    return path_r


@dataclass
class SessionPaths:
    root: Path

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def clips(self) -> Path:
        return self.root / "clips"

    @property
    def annotated_video(self) -> Path:
        return self.root / "annotated.mp4"

    @property
    def events_log(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / MARKER

    @classmethod
    def create(cls, output_dir: Path, session_id: str) -> "SessionPaths":
        root = output_dir / safe_name(session_id)
        try:
            for d in (root, root / "snapshots", root / "clips"):
                d.mkdir(parents=True, exist_ok=True)
            probe = root / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise EvidenceError(f"Evidence folder {root} is not writable: {exc}") from exc
        return cls(root)


def write_json(path: Path, data: dict) -> bool:
    try:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return True
    except OSError as exc:
        log.error("Could not write %s: %s", path, exc)
        return False


def append_jsonl(path: Path, data: dict) -> bool:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, default=str) + "\n")
        return True
    except OSError as exc:
        log.error("Could not append to %s: %s", path, exc)
        return False


def save_jpeg(path: Path, frame: np.ndarray, quality: int = 90) -> bool:
    """imencode + write_bytes also works for paths with non-ASCII characters on Windows."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        log.error("JPEG encoding failed for %s", path)
        return False
    try:
        path.write_bytes(buf.tobytes())
        return True
    except OSError as exc:
        log.error("Could not write %s: %s", path, exc)
        return False


class SafeVideoWriter:
    """cv2.VideoWriter that opens lazily on the first frame and reports failures."""

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = max(1.0, float(fps))
        self._writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self.frames = 0
        self.error: str | None = None

    def write(self, frame: np.ndarray) -> bool:
        if self.error:
            return False
        h, w = frame.shape[:2]
        if self._writer is None:
            self.size = (w, h)
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self._writer.isOpened():
                self.error = f"could not open video writer for {self.path}"
                log.error(self.error)
                return False
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size)
        self._writer.write(frame)
        self.frames += 1
        return True

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


@dataclass
class _ActiveClip:
    key: str
    writer: SafeVideoWriter
    end_ts: float
    event_ids: list[str] = field(default_factory=list)


class ClipRecorder:
    """Keeps the last few seconds of annotated frames and writes incident clips around alerts."""

    def __init__(self, clips_dir: Path, pre_seconds: float, post_seconds: float, max_width: int, max_buffer: int = 400):
        self.dir = clips_dir
        self.pre, self.post = pre_seconds, post_seconds
        self.max_width = max_width
        self.buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=max_buffer)
        self.active: list[_ActiveClip] = []
        self.finished: list[dict] = []

    def push(self, frame: np.ndarray, meta: FrameMeta) -> None:
        small = resize_to_width(frame, self.max_width)
        self.buffer.append((meta.timestamp, small))
        while self.buffer and meta.timestamp - self.buffer[0][0] > self.pre:
            self.buffer.popleft()
        for clip in list(self.active):
            clip.writer.write(small)
            if meta.timestamp >= clip.end_ts:
                self._close(clip)

    def start(self, key: str, event_id: str, meta: FrameMeta, fps: float) -> Path:
        """Start (or extend) the clip for ``key``. The pre-event buffer already includes the current frame."""
        for clip in self.active:
            if clip.key == key:
                clip.end_ts = max(clip.end_ts, meta.timestamp + self.post)
                clip.event_ids.append(event_id)
                return clip.writer.path
        path = ensure_within(self.dir, self.dir / f"{safe_name(key)}_{meta.index:07d}_{safe_name(event_id)}.mp4")
        writer = SafeVideoWriter(path, fps)
        for _, f in self.buffer:
            writer.write(f)
        clip = _ActiveClip(key, writer, meta.timestamp + self.post, [event_id])
        self.active.append(clip)
        return path

    def _close(self, clip: _ActiveClip) -> None:
        clip.writer.release()
        self.active.remove(clip)
        self.finished.append({"path": str(clip.writer.path), "frames": clip.writer.frames,
                              "error": clip.writer.error, "event_ids": clip.event_ids})

    def close_all(self) -> None:
        for clip in list(self.active):
            self._close(clip)


def purge_old_sessions(output_dir: Path, retention_days: int, storage=None) -> list[str]:
    """Delete session folders older than ``retention_days`` (0 disables).

    Only folders that contain our session marker file are touched, so nothing
    else that happens to live in the evidence folder can be deleted.
    """
    if retention_days <= 0 or not output_dir.exists():
        return []
    cutoff = time.time() - retention_days * 86400
    removed = []
    for d in output_dir.iterdir():
        marker = d / MARKER
        if d.is_dir() and marker.exists() and marker.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=False)
            if storage is not None:
                storage.delete_session(d.name)
            removed.append(d.name)
    if removed:
        log.info("Retention: removed %d old evidence session(s)", len(removed))
    return removed
