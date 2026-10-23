"""Secondary analysis (ReID embeddings and clothing) on a separate thread.

The main worker only picks good person crops and drops them into a small
bounded queue. This thread batches them through the ReID and clothing models
and updates the person gallery. If the queue is full the crop is dropped and
counted: these results are useful but never worth stalling the main
detector for.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from app.clothing_analyzer import ClothingAnalyzer
from app.config import ClothingConfig, ReIDConfig
from app.datatypes import GROUP_PERSON, box_intersection, box_area, touches_border
from app.performance import FrameQueue, LatencyStat, QueueClosed, cuda_sync
from app.person_reid import AuthorizationIndex, IdentityDecision, PersonGallery, ReIDBackend
from app.segmenter import polygon_mask_in_box
from app.tracker import TrackStore

log = logging.getLogger(__name__)


@dataclass
class CropJob:
    track_id: str
    crop: np.ndarray
    mask: np.ndarray | None
    ts: float
    frame: int
    want_reid: bool
    want_clothing: bool


class CropSelector:
    """Decides which visible people are worth a ReID / clothing crop this frame."""

    def __init__(self, reid_cfg: ReIDConfig, clothing_cfg: ClothingConfig, reid_on: bool, clothing_on: bool):
        self.r, self.c = reid_cfg, clothing_cfg
        self.reid_on, self.clothing_on = reid_on, clothing_on
        self._last_reid: dict[str, int] = {}
        self._last_clothing: dict[str, int] = {}
        self._tick = 0

    def select(self, frame: np.ndarray, store: TrackStore, gallery: PersonGallery | None) -> list[CropJob]:
        self._tick += 1
        if not (self.reid_on or self.clothing_on):
            return []
        h, w = frame.shape[:2]
        persons = store.visible(GROUP_PERSON)
        gray = None
        candidates = []
        for p in persons:
            bw, bh = p.box[2] - p.box[0], p.box[3] - p.box[1]
            if bw < self.r.min_crop_width or bh < self.r.min_crop_height or touches_border(p.box, w, h):
                continue
            # Skip crops where another person covers a big part of this one.
            if any(o is not p and box_intersection(p.box, o.box) > 0.3 * box_area(p.box) for o in persons):
                continue
            tid = p.track_id
            need_reid = self.reid_on and self._tick - self._last_reid.get(tid, -10**9) >= self.r.sample_interval_frames
            need_cloth = (self.clothing_on and bh >= self.c.min_crop_height
                          and self._tick - self._last_clothing.get(tid, -10**9) >= self.c.interval_frames)
            if not (need_reid or need_cloth):
                continue
            undecided = gallery is not None and gallery.identity(tid) is None
            candidates.append((0 if undecided else 1, self._last_reid.get(tid, -1), p, need_reid, need_cloth))
        candidates.sort(key=lambda c: (c[0], c[1]))

        jobs = []
        for _, _, p, need_reid, need_cloth in candidates[: self.r.max_crops_per_frame]:
            mask, (x1, y1, x2, y2) = polygon_mask_in_box(p.polygon, p.box, frame.shape)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_64F).var() < self.r.min_sharpness:
                continue  # too blurry to be useful
            if need_reid:
                self._last_reid[p.track_id] = self._tick
            if need_cloth:
                self._last_clothing[p.track_id] = self._tick
            jobs.append(CropJob(p.track_id, crop.copy(), mask, store.current_ts, store.current_frame, need_reid, need_cloth))
        return jobs


class SecondaryAnalyzer:
    def __init__(self, backend: ReIDBackend | None, gallery: PersonGallery | None, clothing: ClothingAnalyzer | None,
                 auth: AuthorizationIndex | None, device: str, queue_size: int, batch_size: int = 4):
        self.backend = backend
        self.gallery = gallery
        self.clothing = clothing if clothing is not None and clothing.available else None
        self.auth = auth
        self.device = device
        self.batch_size = batch_size
        self.queue = FrameQueue(queue_size)
        self.stop_event = threading.Event()
        self.reid_latency = LatencyStat()
        self.clothing_latency = LatencyStat()
        self.jobs_done = 0
        self.jobs_dropped = 0
        self.errors = 0
        self.latest_clothing: dict[str, dict] = {}
        self.outbox: deque = deque()  # (kind, payload) for the main worker: identity decisions and clothing results
        self.warmup = None  # optional callable run on this thread before the first job
        self._thread = threading.Thread(target=self._run, name="secondary-analysis", daemon=True)

    @property
    def enabled(self) -> bool:
        return self.backend is not None or self.clothing is not None

    def start(self) -> None:
        if self.enabled:
            self._thread.start()

    def submit(self, job: CropJob) -> None:
        if not self.enabled:
            return
        self.jobs_dropped += len(self.queue.put_drop_oldest(job))

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
        self.jobs_dropped += len(self.queue.drain())

    def _run(self) -> None:
        if self.warmup is not None:
            self.warmup()
        while not self.stop_event.is_set():
            try:
                first = self.queue.get(self.stop_event)
            except QueueClosed:
                break
            batch = [first]
            while len(batch) < self.batch_size:
                nxt = self.queue.get_nowait()
                if nxt is None:
                    break
                batch.append(nxt)
            try:
                self._process(batch)
            except Exception:
                self.errors += 1
                log.exception("Secondary analysis failed on a batch; continuing")

    def _process(self, batch: list[CropJob]) -> None:
        if self.clothing is not None:
            for job in (j for j in batch if j.want_clothing):
                t0 = time.perf_counter()
                attrs = self.clothing.analyze(job.crop, job.mask)
                cuda_sync(self.device)
                self.clothing_latency.add((time.perf_counter() - t0) * 1000)
                attrs.update(ts=round(job.ts, 2), frame=job.frame)
                self.latest_clothing[job.track_id] = attrs
                if self.gallery is not None:
                    self.gallery.set_clothing(job.track_id, attrs)
                self.outbox.append(("clothing", (job.track_id, attrs)))
        reid_jobs = [j for j in batch if j.want_reid]
        if self.backend is not None and self.gallery is not None and reid_jobs:
            t0 = time.perf_counter()
            embs = self.backend.embed([j.crop for j in reid_jobs], [j.mask for j in reid_jobs])
            cuda_sync(self.device)
            self.reid_latency.add((time.perf_counter() - t0) * 1000 / len(reid_jobs))
            for job, emb in zip(reid_jobs, embs):
                decision: IdentityDecision | None = self.gallery.add_sample(job.track_id, emb, job.ts, job.frame)
                if decision is not None:
                    self.outbox.append(("identity", decision))
                prof = self.gallery.profile_for(job.track_id)
                if prof is not None and self.auth is not None and self.auth.entries:
                    prof.authorized = self.auth.match(list(prof.embeddings))
        self.jobs_done += len(batch)
