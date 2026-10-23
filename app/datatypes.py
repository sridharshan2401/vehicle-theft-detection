"""Small shared data types and box geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

GROUP_VEHICLE = "vehicle"
GROUP_PERSON = "person"
GROUP_OTHER = "other"
GROUP_PREFIX = {GROUP_VEHICLE: "V", GROUP_PERSON: "P", GROUP_OTHER: "O"}


@dataclass
class FrameMeta:
    index: int  # 0-based frame number in the source video
    timestamp: float  # seconds from the start of the source video


@dataclass
class FramePacket:
    frame: np.ndarray
    meta: FrameMeta


@dataclass
class Detection:
    class_id: int
    class_name: str
    group: str
    conf: float
    box: np.ndarray  # x1, y1, x2, y2 in frame pixels
    polygon: np.ndarray | None = None  # instance mask outline, (N, 2) frame pixels
    track_id: str | None = None  # application id, e.g. "V001" / "P003"
    raw_track_id: int | None = None  # id from the underlying tracker

    @property
    def foot(self) -> tuple[float, float]:
        return box_foot(self.box)


class EventKind:
    PERSON_EXITED_VEHICLE = "person_exited_vehicle"
    PROXIMITY = "proximity"
    REPEATED_INTERACTION = "repeated_interaction"
    EXTENDED_PRESENCE = "extended_presence"
    APPARENT_ENTRY = "apparent_entry"
    RESTRICTED_AREA = "restricted_area"
    VEHICLE_PARKED = "vehicle_parked"
    VEHICLE_MOVED = "vehicle_moved"
    VEHICLE_LEFT_ZONE = "vehicle_left_zone"
    VEHICLE_DEPARTED = "vehicle_departed"
    MANUAL_SNAPSHOT = "manual_snapshot"


@dataclass
class Event:
    """Something the analysis noticed. Behaviour, zone and association modules emit these."""

    kind: str
    ts: float
    frame: int
    vehicle_id: str | None = None
    person_id: str | None = None
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


def box_area(b: np.ndarray) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def box_intersection(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(max(0.0, w) * max(0.0, h))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = box_intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box ``a`` covered by box ``b``."""
    area = box_area(a)
    return box_intersection(a, b) / area if area > 0 else 0.0


def expand_box(b: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Grow a box by fx * width on the left and right and fy * height on top and bottom."""
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] - fx * w, b[1] - fy * h, b[2] + fx * w, b[3] + fy * h], dtype=np.float32)


def box_center(b: np.ndarray) -> tuple[float, float]:
    return float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)


def box_foot(b: np.ndarray) -> tuple[float, float]:
    """Bottom-centre point: a rough ground-contact point for people and vehicles."""
    return float((b[0] + b[2]) / 2), float(b[3])


def box_diag(b: np.ndarray) -> float:
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def point_in_box(p: tuple[float, float], b: np.ndarray) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def distance_to_box(p: tuple[float, float], b: np.ndarray) -> float:
    """Euclidean distance from a point to the nearest edge of a box (0 if inside)."""
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return float(np.hypot(dx, dy))


def touches_border(b: np.ndarray, width: int, height: int, margin: float = 4.0) -> bool:
    return b[0] <= margin or b[1] <= margin or b[2] >= width - margin or b[3] >= height - margin
