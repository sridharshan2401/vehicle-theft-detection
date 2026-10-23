"""Shared test helpers.

Everything here is a test double for *software logic*: a colour-blob
detector for synthetic videos and a simple IoU tracker. They say nothing
about real model accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.datatypes import GROUP_PERSON, GROUP_VEHICLE, Detection, box_iou  # noqa: E402
from app.detector import ClassMapping  # noqa: E402

CAR_BGR = (255, 0, 0)
PERSON_BGR = (0, 255, 0)


class ColorBlobDetector:
    """Finds pure blue (car) and pure green (person) rectangles in synthetic frames."""

    def __init__(self, delay: float = 0.0, fail_after: int | None = None):
        self.mapping = ClassMapping(groups={0: GROUP_PERSON, 2: GROUP_VEHICLE}, names={0: "person", 2: "car"})
        self.delay = delay
        self.calls = 0
        self.fail_after = fail_after

    def detect(self, frame, imgsz):
        import time

        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated detector failure")
        if self.delay:
            time.sleep(self.delay)
        dets = []
        for color, cid, name, group in ((CAR_BGR, 2, "car", GROUP_VEHICLE), (PERSON_BGR, 0, "person", GROUP_PERSON)):
            lo = np.clip(np.array(color) - 80, 0, 255)
            hi = np.clip(np.array(color) + 80, 0, 255)
            mask = cv2.inRange(frame, lo, hi)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w * h < 60:
                    continue
                dets.append(Detection(cid, name, group, 0.9, np.array([x, y, x + w, y + h], np.float32),
                                      c.reshape(-1, 2).astype(np.float32)))
        return dets, 1.0


class IoUTracker:
    """Greedy IoU matcher with the same update() contract as the Ultralytics trackers."""

    def __init__(self, kind=None, frame_rate=30):
        self.tracks: dict[int, np.ndarray] = {}
        self.next_id = 1

    def update(self, inp, img=None):
        rows = []
        used = set()
        for i in range(len(inp)):
            box = inp.xyxy[i]
            best, best_iou = None, 0.3
            for tid, tb in self.tracks.items():
                iou = box_iou(box, tb)
                if tid not in used and iou > best_iou:
                    best, best_iou = tid, iou
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.tracks[best] = box
            rows.append([*box, best, inp.conf[i], inp.cls[i], i])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 8), np.float32)


def iou_tracker_factory(kind, frame_rate):
    return IoUTracker(kind, frame_rate)


def make_scene_video(path: Path, n_frames: int = 60, fps: float = 10.0, person_from: int = 20) -> Path:
    """A parked blue 'car'; from ``person_from`` a green 'person' appears at its right side and walks away."""
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.full((h, w, 3), 40, np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 160), CAR_BGR, -1)
        if i >= person_from:
            x = 195 + 3 * (i - person_from)
            if x + 20 < w - 5:
                cv2.rectangle(frame, (x, 110), (x + 20, 165), PERSON_BGR, -1)
        cv2.putText(frame, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        writer.write(frame)
    writer.release()
    return path


def make_theft_scene_video(path: Path, fps: float = 10.0) -> Path:
    """Parked car; a person walks up from the left, waits at the door, overlaps the car and vanishes;
    then the car drives off."""
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(170):
        frame = np.full((h, w, 3), 40, np.uint8)
        cx = 100 if i < 100 else 100 + 4 * (i - 100)
        if cx < w:
            cv2.rectangle(frame, (cx, 100), (min(w - 1, cx + 100), 160), CAR_BGR, -1)
        if 40 <= i < 85:
            x = min(20 + 3 * (i - 40), 95)  # walks up, then stands at the door for 2 s before vanishing
            cv2.rectangle(frame, (x, 105), (x + 20, 162), PERSON_BGR, -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def scene_video(tmp_path) -> Path:
    return make_scene_video(tmp_path / "scene.mp4")


@pytest.fixture
def test_settings(tmp_path):
    def make(**extra):
        overrides = {
            "storage.db_path": str(tmp_path / "db.sqlite3"),
            "evidence.output_dir": str(tmp_path / "evidence"),
            "zones.file": str(tmp_path / "zones.json"),
            "app.display": False,
            "device.device": "cpu",
            "reid.backend": "histogram",
            "reid.min_crop_height": 20,
            "reid.min_crop_width": 10,
            "reid.min_sharpness": 0.0,
            "clothing.backend": "person_mask_bands",
            "clothing.min_crop_height": 20,
        }
        overrides.update(extra)
        return load_config(None, overrides)

    return make
