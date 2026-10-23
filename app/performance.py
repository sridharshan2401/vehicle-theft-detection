"""Bounded queues, frame accounting and timing helpers."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

END_OF_STREAM = object()


class QueueClosed(Exception):
    """Raised by FrameQueue.get when the stop event fires while waiting."""


class FrameQueue:
    """A bounded FIFO with two put styles.

    ``put`` blocks while the queue is full (complete-evidence mode never loses
    frames). ``put_drop_oldest`` evicts the oldest item instead and returns it, so
    the caller can count it as skipped (responsive mode).
    """

    def __init__(self, maxsize: int):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._items: deque[Any] = deque()
        self._cond = threading.Condition()
        self.max_depth = 0

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)

    def _append(self, item: Any) -> None:
        self._items.append(item)
        self.max_depth = max(self.max_depth, len(self._items))
        self._cond.notify_all()

    def put(self, item: Any, stop_event: threading.Event | None = None, poll: float = 0.05) -> bool:
        """Block until there is room. Returns False if stop_event fired first."""
        with self._cond:
            while len(self._items) >= self.maxsize:
                if stop_event is not None and stop_event.is_set():
                    return False
                self._cond.wait(poll)
            self._append(item)
            return True

    def put_drop_oldest(self, item: Any) -> list[Any]:
        dropped = []
        with self._cond:
            while len(self._items) >= self.maxsize:
                old = self._items.popleft()
                if old is END_OF_STREAM:  # never drop the end marker
                    self._items.appendleft(old)
                    break
                dropped.append(old)
            self._append(item)
        return dropped

    def put_end(self) -> None:
        """Always enqueue the end marker, even if that briefly exceeds maxsize."""
        with self._cond:
            self._append(END_OF_STREAM)

    def get(self, stop_event: threading.Event | None = None, poll: float = 0.05) -> Any:
        with self._cond:
            while not self._items:
                if stop_event is not None and stop_event.is_set():
                    raise QueueClosed
                self._cond.wait(poll)
            item = self._items.popleft()
            self._cond.notify_all()
            return item

    def get_nowait(self) -> Any | None:
        with self._cond:
            if not self._items:
                return None
            item = self._items.popleft()
            self._cond.notify_all()
            return item

    def drain(self) -> list[Any]:
        with self._cond:
            items = list(self._items)
            self._items.clear()
            self._cond.notify_all()
            return items


class FrameAccounting:
    """Thread-safe counters that let us prove what happened to every source frame.

    Invariant once the pipeline has drained: read == processed + skipped.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.read = 0
        self.processed = 0
        self.skipped = 0
        self.displayed = 0
        self.display_dropped = 0
        self.tracker_resets = 0
        self.last_processed_index = -1

    def add(self, field_name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + n)

    def mark_processed(self, index: int) -> None:
        with self._lock:
            self.processed += 1
            self.last_processed_index = index

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self.read - self.processed - self.skipped

    def is_balanced(self) -> bool:
        return self.in_flight == 0

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "frames_read": self.read,
                "frames_processed": self.processed,
                "frames_skipped": self.skipped,
                "frames_displayed": self.displayed,
                "display_frames_dropped": self.display_dropped,
                "tracker_resets": self.tracker_resets,
            }


class RateMeter:
    """Events per second over a sliding time window."""

    def __init__(self, window: float = 2.0):
        self.window = window
        self._stamps: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self._stamps.append(now)
        while self._stamps and now - self._stamps[0] > self.window:
            self._stamps.popleft()

    @property
    def rate(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


@dataclass
class LatencyStat:
    """Exponential moving average plus min/max/total for a latency in milliseconds."""

    alpha: float = 0.1
    ema: float = 0.0
    count: int = 0
    total: float = 0.0
    max: float = 0.0

    def add(self, ms: float) -> None:
        self.ema = ms if self.count == 0 else (1 - self.alpha) * self.ema + self.alpha * ms
        self.count += 1
        self.total += ms
        self.max = max(self.max, ms)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"mean_ms": round(self.mean, 2), "ema_ms": round(self.ema, 2), "max_ms": round(self.max, 2), "count": self.count}


class AdaptiveResolution:
    """Steps the detector input size down when it is too slow and back up when there is headroom.

    Only used in responsive mode. Sizes stay multiples of 32 within [min_size, max_size].
    """

    def __init__(self, max_size: int, min_size: int, target_ms: float, step: int = 64, patience: int = 15):
        self.max_size, self.min_size = max_size, min_size
        self.size = max_size
        self.target_ms = target_ms
        self.step = step
        self.patience = patience
        self._slow = 0
        self._fast = 0

    def update(self, latency_ms: float) -> int:
        if latency_ms > self.target_ms * 1.1:
            self._slow, self._fast = self._slow + 1, 0
        elif latency_ms < self.target_ms * 0.6:
            self._fast, self._slow = self._fast + 1, 0
        else:
            self._slow = self._fast = 0
        if self._slow >= self.patience and self.size > self.min_size:
            self.size = max(self.min_size, self.size - self.step)
            self._slow = 0
        elif self._fast >= self.patience * 4 and self.size < self.max_size:
            self.size = min(self.max_size, self.size + self.step)
            self._fast = 0
        return self.size


def cuda_sync(device: str) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def gpu_memory_mb(device: str) -> dict[str, float] | None:
    if not device.startswith("cuda"):
        return None
    import torch

    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
