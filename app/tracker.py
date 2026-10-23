"""Multi-object tracking with separate vehicle and person ID namespaces.

The tracking itself is Ultralytics' ByteTrack (default) or BoT-SORT. We run
one tracker instance per object group so a person box can never be matched
to a car track, then translate the tracker's raw integers into stable
application IDs such as V001 and P003.

A track ID only says "the tracker believes this is the same box as before".
It is not a person's identity. Long-term identity is handled separately by
person_reid.py.
"""

from __future__ import annotations

import inspect
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.config import TrackerConfig
from app.datatypes import (
    GROUP_PERSON,
    GROUP_PREFIX,
    Detection,
    box_center,
    box_diag,
    box_iou,
    touches_border,
)

log = logging.getLogger(__name__)


class _TrackerInput:
    """The minimal 'Boxes'-like object Ultralytics trackers read from."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.reshape(-1, 4).astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        out = self.xyxy.copy()
        out[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        out[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        out[:, 0] = self.xyxy[:, 0] + out[:, 2] / 2
        out[:, 1] = self.xyxy[:, 1] + out[:, 3] / 2
        return out

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_TrackerInput":
        return _TrackerInput(self.xyxy[idx], self.conf[idx], self.cls[idx])


def _make_ultralytics_tracker(kind: str, frame_rate: float):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    try:
        from ultralytics.utils import YAML

        load = YAML.load
    except ImportError:  # older Ultralytics releases
        from ultralytics.utils import yaml_load as load

    cfg_name = "bytetrack.yaml" if kind == "bytetrack" else "botsort.yaml"
    args = IterableSimpleNamespace(**load(check_yaml(cfg_name)))
    cls = BYTETracker if kind == "bytetrack" else BOTSORT
    if kind == "botsort":
        # Appearance ReID inside BoT-SORT would need its own model; long-term
        # identity is handled by person_reid.py instead.
        args.with_reid = False
    if "frame_rate" in inspect.signature(cls.__init__).parameters:
        return cls(args=args, frame_rate=int(round(frame_rate)))
    # Ultralytics >= 8.4 takes track_buffer in frames; the YAML value assumes 30 FPS, so scale it.
    args.track_buffer = max(1, int(round(args.track_buffer * frame_rate / 30.0)))
    return cls(args=args)


class IdRegistry:
    """Maps (group, tracker epoch, raw tracker id) to app ids like V001 / P001.

    The epoch increases whenever the trackers are reset, so a raw id reused
    after a reset still gets a fresh application id.
    """

    def __init__(self) -> None:
        self._ids: dict[tuple[str, int, int], str] = {}
        self._counters: dict[str, int] = {}

    def get(self, group: str, epoch: int, raw_id: int) -> str:
        key = (group, epoch, raw_id)
        if key not in self._ids:
            n = self._counters.get(group, 0) + 1
            self._counters[group] = n
            self._ids[key] = f"{GROUP_PREFIX[group]}{n:03d}"
        return self._ids[key]


class MultiClassTracker:
    """One Ultralytics tracker per object group, with a shared id registry."""

    def __init__(self, cfg: TrackerConfig, frame_rate: float, groups: list[str], factory=None):
        self.cfg = cfg
        self.frame_rate = frame_rate
        self.groups = groups
        self._factory = factory or _make_ultralytics_tracker
        self.registry = IdRegistry()
        self.epoch = 0
        self.resets = 0
        self.duplicates_removed = 0
        self.last_index: int | None = None
        self._trackers = {g: self._factory(cfg.type, frame_rate) for g in groups}

    def reset(self) -> None:
        self._trackers = {g: self._factory(self.cfg.type, self.frame_rate) for g in self.groups}
        self.epoch += 1
        self.resets += 1

    def update(self, detections: list[Detection], frame: np.ndarray, frame_index: int) -> tuple[list[Detection], bool]:
        """Assign track ids. Returns (tracked detections, whether a reset happened).

        If more than ``max_gap_frames`` source frames were skipped since the last
        update, motion between the two frames is too large for the tracker's
        motion model, so we reset and let re-identification reconnect people.
        """
        did_reset = False
        if self.last_index is not None and frame_index - self.last_index > self.cfg.max_gap_frames:
            log.info("Frame gap %d > %d: resetting trackers", frame_index - self.last_index, self.cfg.max_gap_frames)
            self.reset()
            did_reset = True
        self.last_index = frame_index

        tracked: list[Detection] = []
        for group, tracker in self._trackers.items():
            dets = [d for d in detections if d.group == group]
            if dets:
                inp = _TrackerInput(
                    np.stack([d.box for d in dets]),
                    np.array([d.conf for d in dets]),
                    np.array([d.class_id for d in dets]),
                )
            else:
                inp = _TrackerInput(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
            out = np.asarray(tracker.update(inp, frame))
            if out.size == 0:
                continue
            # Rows are x1, y1, x2, y2, track_id, score, cls, det_index.
            for row in out.reshape(len(out), -1):
                det = dets[int(row[-1])]
                det.raw_track_id = int(row[4])
                det.track_id = self.registry.get(group, self.epoch, det.raw_track_id)
                tracked.append(det)
        return self._remove_duplicates(tracked), did_reset

    def _remove_duplicates(self, dets: list[Detection]) -> list[Detection]:
        """Drop the newer of two same-group tracks that sit on top of each other."""
        keep = sorted(dets, key=lambda d: int(d.track_id[1:]))  # lower number = older track
        result: list[Detection] = []
        for d in keep:
            if any(o.group == d.group and box_iou(o.box, d.box) >= self.cfg.duplicate_iou for o in result):
                self.duplicates_removed += 1
                continue
            result.append(d)
        return result


@dataclass
class TrackState:
    """What we know about one application track over time."""

    track_id: str
    group: str
    class_name: str
    first_ts: float
    first_frame: int
    last_ts: float
    last_frame: int
    box: np.ndarray
    conf: float
    born_at_border: bool
    polygon: np.ndarray | None = None
    active: bool = True
    lost_ts: float | None = None
    hits: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=900))  # (ts, frame, box)

    @property
    def age(self) -> float:
        return self.last_ts - self.first_ts

    def speed(self, window: float = 1.0) -> float:
        """Centre speed in box-diagonals per second over the last ``window`` seconds."""
        if len(self.history) < 2:
            return 0.0
        t_end, _, b_end = self.history[-1]
        older = [h for h in self.history if t_end - h[0] <= window]
        t0, _, b0 = older[0]
        if t_end - t0 <= 0:
            return 0.0
        (x0, y0), (x1, y1) = box_center(b0), box_center(b_end)
        diag = max(box_diag(b_end), 1.0)
        return float(np.hypot(x1 - x0, y1 - y0) / diag / (t_end - t0))


class TrackStore:
    """Keeps per-track history and works out which tracks are visible, lost or forgotten."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: dict[str, TrackState] = {}
        self.current_frame = -1
        self.current_ts = 0.0

    def update(self, ts: float, frame_index: int, dets: list[Detection], frame_shape: tuple[int, int]) -> tuple[list[str], list[str]]:
        """Fold this frame's tracked detections in. Returns (new track ids, newly lost track ids)."""
        self.current_frame, self.current_ts = frame_index, ts
        h, w = frame_shape[:2]
        new_ids = []
        for d in dets:
            if d.track_id is None:
                continue
            st = self.tracks.get(d.track_id)
            if st is None:
                st = TrackState(
                    track_id=d.track_id, group=d.group, class_name=d.class_name,
                    first_ts=ts, first_frame=frame_index, last_ts=ts, last_frame=frame_index,
                    box=d.box.copy(), conf=d.conf, born_at_border=touches_border(d.box, w, h),
                    polygon=d.polygon,
                )
                self.tracks[d.track_id] = st
                new_ids.append(d.track_id)
            else:
                st.hits += 1
                st.last_ts, st.last_frame = ts, frame_index
                st.box, st.conf, st.polygon = d.box.copy(), d.conf, d.polygon
                st.active, st.lost_ts = True, None
                st.class_name = d.class_name
            st.history.append((ts, frame_index, d.box.copy()))

        lost = []
        for tid, st in list(self.tracks.items()):
            if st.active and ts - st.last_ts > self.cfg.lost_after_seconds:
                st.active, st.lost_ts = False, ts
                lost.append(tid)
            elif not st.active and ts - st.last_ts > self.cfg.forget_after_seconds:
                del self.tracks[tid]
        return new_ids, lost

    def visible(self, group: str | None = None) -> list[TrackState]:
        """Tracks seen in the current frame."""
        return [t for t in self.tracks.values() if t.last_frame == self.current_frame and (group is None or t.group == group)]

    def active(self, group: str | None = None) -> list[TrackState]:
        return [t for t in self.tracks.values() if t.active and (group is None or t.group == group)]

    def recently_lost(self, group: str, within: float) -> list[TrackState]:
        return [t for t in self.tracks.values()
                if t.group == group and not t.active and t.lost_ts is not None and self.current_ts - t.last_ts <= within]

    def is_stationary(self, track_id: str) -> bool:
        st = self.tracks.get(track_id)
        return st is not None and st.speed() < self.cfg.stationary_speed

    def persons(self) -> list[TrackState]:
        return self.visible(GROUP_PERSON)
