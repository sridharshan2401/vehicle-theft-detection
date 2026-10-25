"""The processing pipeline: reader thread -> analysis worker thread -> display (main thread).

    CameraReader (live webcam, default)
      or VideoReader (file)  --(bounded FrameQueue)--> analysis worker --(bounded queue)--> window / headless
                                                             |
                                                             +--> SecondaryAnalyzer thread (ReID + clothing)

The live camera and a video file go through exactly the same analysis code;
only the reader, the clock and the output defaults differ (live mode keeps the
newest frame, never records the full feed unless asked, and saves snapshots
rather than clips for alerts).

The analysis worker owns all per-frame state (tracker, association, zones,
behaviour, risk, evidence), so only the queues and the person gallery are
shared between threads.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from app.association import AMBIGUOUS, OBSERVED_EXIT, POSSIBLE_EXIT, AssociationEngine
from app.behavior import BehaviorAnalyzer
from app.clothing_analyzer import ClothingAnalyzer
from app.config import RISK_LEVELS, Settings, resolve_path
from app.datatypes import GROUP_PERSON, GROUP_VEHICLE, Event, EventKind, FrameMeta, FramePacket
from app.display import DisplayWindow, RenderContext, Renderer, render_status
from app.evidence import (
    ClipRecorder,
    SafeVideoWriter,
    SessionPaths,
    append_jsonl,
    purge_old_sessions,
    save_jpeg,
    write_json,
)
from app.parking_zones import VehicleZoneTracker, ZoneSet
from app.performance import (
    END_OF_STREAM,
    AdaptiveResolution,
    FrameAccounting,
    FrameQueue,
    LatencyStat,
    QueueClosed,
    RateMeter,
    gpu_memory_mb,
)
from app.person_reid import (
    CANDIDATE,
    NEW,
    PENDING,
    RECONNECTED,
    UNCERTAIN,
    AuthorizationIndex,
    PersonGallery,
    build_reid_backend,
)
from app.risk_engine import (
    AUTHORIZED,
    MOVEMENT_KINDS,
    OBSERVED_ASSOCIATED,
    POSSIBLE_ASSOCIATED,
    UNRECOGNIZED,
    PersonContext,
    RiskEngine,
    level_rank,
)
from app.secondary import CropSelector, SecondaryAnalyzer
from app.storage import Storage
from app.tracker import MultiClassTracker, TrackStore
from app.camera import CameraReader, CameraSource
from app.video_reader import VideoInfo, VideoReader, probe_video

log = logging.getLogger(__name__)

KEY_QUIT = {ord("q"), ord("Q"), 27}
KEY_PAUSE = {ord(" ")}
KEY_SNAPSHOT = {ord("s"), ord("S")}
KEY_OVERLAY = {ord("r"), ord("R")}


class PipelineError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, settings: Settings, info: VideoInfo, device: str, half: bool, *,
                 detector=None, tracker_factory=None, display: bool | None = None,
                 max_frames: int | None = None, start_frame: int = 0, hardware: dict | None = None,
                 camera: CameraSource | None = None, clock=None, window=None):
        self.s = settings
        self._window = window  # a ready-made window object (tests); otherwise an OpenCV DisplayWindow
        self.display = None
        self._quit_requested = False
        self._status_frame: np.ndarray | None = None
        self._startup_frames = 0  # start-up screens shown before analysis began
        self.info = info
        self.device = device
        self.half = half
        self.live = info.live
        self.camera = camera
        self._clock = clock or time.perf_counter
        if self.live and camera is None:
            raise ValueError("a live VideoInfo needs the opened CameraSource")
        self.mode = settings.app.mode
        if self.live and self.mode != "responsive":
            log.warning("A live camera cannot wait for slow analysis; using responsive mode (newest frame wins).")
            self.mode = "responsive"
        self.display_enabled = settings.app.display if display is None else display
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.hardware = hardware or {}
        self._detector = detector
        self._tracker_factory = tracker_factory

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self._worker_ready = threading.Event()
        self.accounting = FrameAccounting()
        self.in_queue = FrameQueue(settings.camera.queue_size if self.live else settings.pipeline.input_queue_size)
        self.out_queue = FrameQueue(settings.pipeline.output_queue_size)
        self.det_latency = LatencyStat()
        self.frame_latency = LatencyStat()
        self.proc_rate = RateMeter()
        self.display_rate = RateMeter()
        self.event_log: list[tuple[str, str]] = []
        self.alert_counts = {lv: 0 for lv in RISK_LEVELS}
        self.association_counts: dict[str, int] = {}
        self.identity_counts: dict[str, int] = {}
        self.worker_error: BaseException | None = None
        self.write_errors: list[str] = []
        self._event_ids: dict[str, str] = {}
        self._last_displayed: tuple[np.ndarray, FrameMeta] | None = None
        self._display_fps = 0.0
        self._started = False

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        s = self.s
        frame_size = (self.info.width, self.info.height)
        # Things that can fail on bad input (zones, model) come before any session is created on disk.
        self.zones = ZoneSet.load(resolve_path(s.zones.file))
        if self._detector is None:
            from app.detector import Detector

            self._detector = Detector(s.detector, self.device, self.half)
            self.detector = self._detector
            self._verify_half()
        self.detector = self._detector

        self.storage = Storage(resolve_path(s.storage.db_path))
        evidence_root = resolve_path(s.evidence.output_dir)
        purge_old_sessions(evidence_root, s.evidence.retention_days, self.storage)
        self.session_id = self.storage.create_session(self.info.path, "live" if self.live else self.mode,
                                                      self.info.as_dict(), s.to_dict())
        self.paths = SessionPaths.create(evidence_root, self.session_id)
        mapping = getattr(self.detector, "mapping", None)
        groups = sorted(set(mapping.groups.values())) if mapping else [GROUP_VEHICLE, GROUP_PERSON]
        self.tracker = MultiClassTracker(s.tracker, self.info.fps, groups, self._tracker_factory)
        self.store = TrackStore(s.tracker)
        self.assoc = AssociationEngine(s.association)
        self.vzt = VehicleZoneTracker(s.zones, self.zones, frame_size)
        self.behavior = BehaviorAnalyzer(s.behavior, self.zones, frame_size)
        self.risk = RiskEngine(s.risk)
        self.adaptive = None
        if self.mode == "responsive" and s.detector.adaptive_imgsz:
            self.adaptive = AdaptiveResolution(s.detector.imgsz, s.detector.min_imgsz, 1000.0 / s.pipeline.target_fps)

        reid_backend, self.reid_status = build_reid_backend(s.reid, self.device)
        self.gallery = PersonGallery(s.reid, reid_backend.name) if reid_backend else None
        self.clothing = ClothingAnalyzer(s.clothing, self.device)
        self.auth = AuthorizationIndex(self.storage.list_people(), reid_backend, s.reid)
        self._load_registry()
        self.secondary = SecondaryAnalyzer(reid_backend, self.gallery, self.clothing, self.auth, self.device,
                                           s.pipeline.secondary_queue_size)
        self.selector = CropSelector(s.reid, s.clothing, reid_backend is not None, self.clothing.available)
        self.reid_backend = reid_backend
        self.secondary.warmup = self._warmup_secondary
        log.info("ReID: %s", self.reid_status)
        log.info("Clothing: %s", self.clothing.status)
        log.info("Authorisation: %s", self.auth.status)

        self.renderer = Renderer(frame_size)
        clip_fps = self.info.fps if self.mode == "complete" else min(self.info.fps, s.pipeline.target_fps)
        self.clip_fps = clip_fps
        self.clips = ClipRecorder(self.paths.clips, s.evidence.pre_event_seconds, s.evidence.post_event_seconds,
                                  s.evidence.clip_max_width)
        # Live mode is a preview: no full recording and no alert clips unless explicitly enabled.
        # Alert snapshots, the event log and the database work the same in both modes.
        record = s.camera.save_annotated_video if self.live else s.evidence.save_annotated_video
        self.clips_enabled = s.camera.save_alert_clips if self.live else True
        self.annotated = SafeVideoWriter(self.paths.annotated_video, clip_fps) if record else None
        if self.live:
            self.reader = CameraReader(self.camera, self.info, self.in_queue, self.accounting, self.stop_event,
                                       self.pause_event, self.max_frames, self._clock, s.camera.max_failed_reads)
        else:
            self.reader = VideoReader(self.info, self.in_queue, self.mode, self.accounting, self.stop_event,
                                      self.pause_event, self.max_frames, self.start_frame)
        write_json(self.paths.summary, {"session_id": self.session_id, "status": "running",
                                        "video": self.info.as_dict(), "mode": self.mode})

    def _warmup_detector(self) -> None:
        """Warm up on the worker thread with the real frame shape.

        cuDNN state is per thread and per input shape: warming up on the main thread,
        or on a square dummy image, still left the first real frame taking ~1.8 s.
        """
        warm = getattr(self.detector, "warmup", None)
        if warm is None:
            return
        shape = (self.info.height, self.info.width)
        warm(self.s.detector.imgsz, shape)
        if self.adaptive is not None and self.adaptive.min_size < self.adaptive.max_size:
            for size in range(self.adaptive.max_size - self.adaptive.step, self.adaptive.min_size - 1, -self.adaptive.step):
                warm(size, shape, runs=1)

    def _warmup_secondary(self) -> None:
        """First calls into cuDNN are slow; pay that cost (on the secondary thread) before work arrives."""
        crop = np.full((192, 96, 3), 127, np.uint8)
        try:
            if self.reid_backend is not None:
                self.reid_backend.embed([crop], [None])
            if self.clothing.mode == "segformer":
                self.clothing.analyze(crop, None)
        except Exception as exc:
            log.warning("Secondary model warm-up failed: %s", exc)

    def _verify_half(self) -> None:
        if not self.half:
            return
        try:
            from ultralytics.utils import ASSETS

            img = cv2.imread(str(ASSETS / "bus.jpg"))
        except Exception:
            img = None
        if img is not None:
            self.half = self.detector.verify_half_precision(img, self.s.detector.imgsz)

    def _load_registry(self) -> None:
        """Registered vehicles by parking zone, with the authorised person ids for each."""
        self.registry: dict[str, dict] = {}
        for v in self.storage.list_vehicles():
            if v.get("parking_zone"):
                people = self.storage.authorized_people_for_vehicle(v["vehicle_record_id"])
                self.registry[v["parking_zone"]] = {**v, "authorized_ids": {p["person_record_id"] for p in people}}

    # --------------------------------------------------------- identity glue
    def _vehicle_record(self, vehicle_id: str | None) -> dict | None:
        if vehicle_id is None:
            return None
        for vid, st in self.vzt.states.items():
            if self.vzt.canonical(vid) == vehicle_id and st.parked_zone in self.registry:
                return self.registry[st.parked_zone]
        return None

    def _resolve_person(self, person_id: str, vehicle_id: str | None) -> PersonContext:
        prof = self.gallery.profile_for(person_id) if self.gallery else None
        rec = self._vehicle_record(vehicle_id)
        if prof and prof.authorized and rec and prof.authorized["record_id"] in rec["authorized_ids"]:
            return PersonContext(AUTHORIZED, person_id, prof.authorized["label"])
        tracks = self.gallery.tracks_of_profile(prof.profile_id) if prof else set()
        tracks.add(person_id)
        if self._exited_vehicle(tracks, vehicle_id):
            return PersonContext(OBSERVED_ASSOCIATED, person_id)
        # ReID is unsure but its best guess is someone who got out of this vehicle: partial credit.
        dec = self.gallery.identity(person_id) if self.gallery else None
        if dec is not None and dec.status in (CANDIDATE, UNCERTAIN) and dec.candidate_id:
            if self._exited_vehicle(self.gallery.tracks_of_profile(dec.candidate_id) | {dec.candidate_id}, vehicle_id):
                return PersonContext(POSSIBLE_ASSOCIATED, person_id, related_id=dec.candidate_id)
        return PersonContext(UNRECOGNIZED, person_id)

    def _exited_vehicle(self, track_ids, vehicle_id: str | None) -> bool:
        """Was any of these person tracks linked to this (canonical) vehicle by an observed exit?"""
        if vehicle_id is None:
            return False
        return any((r := self.assoc.get(t)) is not None and r.is_linked and self.vzt.canonical(r.vehicle_id) == vehicle_id
                   for t in track_ids)

    # ------------------------------------------------------------ the worker
    def _log_event(self, level: str, text: str, ts: float) -> None:
        self.event_log.append((level, f"[{ts:7.1f}s] {text}"))
        del self.event_log[:-50]

    def _store_event(self, e: Event, category: str, level: str = "INFORMATIONAL", message: str | None = None) -> None:
        self.storage.add_event(
            session_id=self.session_id, video_path=self.info.path, frame=e.frame, video_ts=e.ts, category=category,
            risk_level=level, vehicle_id=e.vehicle_id, person_track_id=e.person_id,
            person_identity=self._identity_text(e.person_id), confidences={"event": e.confidence},
            reasons=[e.details] if e.details else None, message=message,
            location=self._location(e.vehicle_id, e.person_id),
        )

    def _vehicle_box(self, vehicle_id: str | None):
        """Where a (canonical) vehicle is: its parked box if known, else its latest tracked box."""
        if vehicle_id is None:
            return None
        for vid, st in self.vzt.states.items():
            if self.vzt.canonical(vid) == vehicle_id and st.parked_box is not None:
                return st.parked_box
        tracks = [t for t in self.store.tracks.values() if t.group == GROUP_VEHICLE
                  and self.vzt.canonical(t.track_id) == vehicle_id]
        return max(tracks, key=lambda t: t.last_ts).box if tracks else None

    def _location(self, vehicle_id: str | None, person_ids) -> dict | None:
        """Pixel boxes stored with events so they can be matched against labelled footage."""
        loc = {}
        vbox = self._vehicle_box(vehicle_id)
        if vbox is not None:
            loc["vehicle_box"] = [round(float(v), 1) for v in vbox]
        for pid in ([person_ids] if isinstance(person_ids, str) else person_ids or []):
            st = self.store.tracks.get(pid)
            if st is not None:
                loc.setdefault("person_boxes", {})[pid] = [round(float(v), 1) for v in st.box]
        return loc or None

    def _identity_text(self, person_id: str | None) -> str | None:
        if not person_id or not self.gallery:
            return None
        dec = self.gallery.identity(person_id)
        if dec is None:
            return "pending"
        return f"{dec.status}:{dec.profile_id or dec.candidate_id}:{dec.score}"

    def _drain_secondary(self) -> None:
        while self.secondary.outbox:
            kind, payload = self.secondary.outbox.popleft()
            if kind == "identity":
                d = payload
                self.identity_counts[d.status] = self.identity_counts.get(d.status, 0) + 1
                self.storage.add_identity_link(self.session_id, d.track_id, d.profile_id or d.candidate_id or "",
                                               d.status, d.score, self.reid_backend.name, d.ts, d.frame)
                if d.status == RECONNECTED:
                    self._log_event("INFORMATIONAL", f"{d.track_id} looks like returning person {d.profile_id} ({d.score:.2f})", d.ts)
            elif kind == "clothing":
                track_id, attrs = payload
                prof = self.gallery.profile_for(track_id) if self.gallery else None
                self.storage.add_clothing_observation(self.session_id, track_id, prof.profile_id if prof else None,
                                                      attrs.get("ts", 0.0), attrs.get("frame", 0), attrs)

    def process_frame(self, packet: FramePacket) -> np.ndarray:
        t_start = time.perf_counter()
        frame, meta = packet.frame, packet.meta
        ts, idx = meta.timestamp, meta.index
        imgsz = self.adaptive.size if self.adaptive else self.s.detector.imgsz
        dets, det_ms = self.detector.detect(frame, imgsz)
        self.det_latency.add(det_ms)

        tracked, did_reset = self.tracker.update(dets, frame, idx)
        if did_reset:
            self.accounting.add("tracker_resets")
            self.assoc.notify_reset(ts)
            self._log_event("INFORMATIONAL", "tracker reset after skipped frames; ids restart, ReID may reconnect people", ts)
        tracked_ids = {id(d) for d in tracked}
        untracked = [d for d in dets if id(d) not in tracked_ids]
        new_ids, lost_ids = self.store.update(ts, idx, tracked, frame.shape)

        if self.gallery:
            self.gallery.note_visible([d.track_id for d in tracked if d.group == GROUP_PERSON], ts)
        for job in self.selector.select(frame, self.store, self.gallery):
            self.secondary.submit(job)
        self._drain_secondary()

        events: list[Event] = []
        zone_events = self.vzt.update(self.store, lost_ids)
        for rec in self.assoc.update(self.store, new_ids):
            if rec.status == "none":
                continue
            self.association_counts[rec.status] = self.association_counts.get(rec.status, 0) + 1
            self.storage.add_association(self.session_id, rec.person_id, rec.vehicle_id, rec.status, rec.confidence,
                                         rec.birth_frame, rec.birth_ts,
                                         {"components": rec.components, "candidates": rec.candidates,
                                          "reasons": rec.reasons, "person_box": rec.person_box})
            if rec.is_linked:
                vid = self.vzt.canonical(rec.vehicle_id)
                e = Event(EventKind.PERSON_EXITED_VEHICLE, rec.decided_ts, rec.decided_frame, vid, rec.person_id,
                          rec.confidence, {"status": rec.status, "components": rec.components})
                events.append(e)
                word = "got out of" if rec.status == OBSERVED_EXIT else "may have got out of"
                self._log_event("INFORMATIONAL", f"{rec.person_id} {word} {vid} ({rec.confidence:.2f})", ts)
            elif rec.status == AMBIGUOUS:
                self._log_event("INFORMATIONAL", f"{rec.person_id} exit ambiguous: {rec.reasons[-1]}", ts)
        events.extend(zone_events)
        events.extend(self.behavior.update(self.store, self.vzt, lost_ids, hold=self.assoc.is_pending))
        for e in events:
            if e.kind in MOVEMENT_KINDS:
                e.details["recent_entrants"] = [en.person_id for en in self.behavior.recent_entrants(e.vehicle_id, ts)]
            if e.kind in (EventKind.VEHICLE_PARKED, EventKind.APPARENT_ENTRY) or e.kind in MOVEMENT_KINDS:
                self._log_event("INFORMATIONAL", f"{e.kind.replace('_', ' ')}: {e.vehicle_id or ''} {e.person_id or ''}", e.ts)
            self._store_event(e, e.kind)
        alerts = self.risk.process(events, self._resolve_person)
        self.risk.tick(ts)

        annotated = self.renderer.render(frame, self._render_context(tracked, untracked, meta, imgsz))
        if self.clips_enabled:
            self.clips.push(annotated, meta)
        for alert in alerts:
            self._record_alert(alert, annotated, meta)
        if self.annotated is not None and not self.annotated.write(annotated):
            self._note_write_error(self.annotated.error)
        self.proc_rate.tick()
        frame_ms = (time.perf_counter() - t_start) * 1000
        self.frame_latency.add(frame_ms)
        if self.adaptive:
            # Whole-frame time is what limits throughput, not just the detector.
            self.adaptive.update(frame_ms)
        return annotated

    def _note_write_error(self, msg: str | None) -> None:
        if msg and msg not in self.write_errors:
            self.write_errors.append(msg)

    def _record_alert(self, alert, annotated: np.ndarray, meta: FrameMeta) -> None:
        self.alert_counts[alert.level] += 1
        snapshot = clip = None
        if level_rank(alert.level) >= level_rank(self.s.evidence.min_level):
            snap_path = self.paths.snapshots / f"{alert.alert_id}_{alert.vehicle_id}_{meta.index:07d}.jpg"
            if save_jpeg(snap_path, annotated, self.s.evidence.snapshot_jpeg_quality):
                snapshot = str(snap_path)
            else:
                self._note_write_error(f"snapshot failed: {snap_path}")
            if self.clips_enabled:
                clip = str(self.clips.start(alert.vehicle_id, alert.alert_id, meta, self.clip_fps))
        event_id = f"{self.session_id}-{alert.alert_id}"
        self._event_ids[alert.alert_id] = event_id
        reasons = [asdict(r) for r in alert.reasons]
        stats = self.stats_snapshot(short=True)
        self.storage.add_event(
            session_id=self.session_id, video_path=self.info.path, frame=meta.index, video_ts=meta.timestamp,
            category=alert.category, risk_level=alert.level, vehicle_id=alert.vehicle_id,
            person_track_id=",".join(alert.person_ids) or None,
            person_identity=";".join(filter(None, (self._identity_text(p) for p in alert.person_ids))) or None,
            risk_score=alert.score, reasons=reasons, confidences={r.kind: r.confidence for r in alert.reasons},
            message=alert.message, snapshot_path=snapshot, clip_path=clip, processing_stats=stats, event_id=event_id,
            location=self._location(alert.vehicle_id, alert.person_ids),
        )
        append_jsonl(self.paths.events_log, {"event_id": event_id, "frame": meta.index, "video_ts": round(meta.timestamp, 3),
                                             "level": alert.level, "score": alert.score, "vehicle_id": alert.vehicle_id,
                                             "person_ids": alert.person_ids, "category": alert.category,
                                             "message": alert.message, "reasons": reasons, "snapshot": snapshot,
                                             "clip": clip, "processing": stats})
        self._log_event(alert.level, alert.message, meta.timestamp)
        log.warning("%s", alert.message)

    def _render_context(self, tracked, untracked, meta: FrameMeta, imgsz: int) -> RenderContext:
        labels: dict[str, list[str]] = {}
        levels: dict[str, str] = {}
        ts = meta.timestamp
        for d in tracked:
            if d.group == GROUP_VEHICLE:
                cid = self.vzt.canonical(d.track_id)
                score, level = self.risk.current(cid, ts)
                levels[d.track_id] = level
                name = d.track_id if cid == d.track_id else f"{d.track_id}={cid}"
                short = "INFO" if level == "INFORMATIONAL" else level
                lines = [f"{name} {d.class_name} {d.conf:.2f}",
                         f"{self.vzt.state_of(d.track_id) or '?'} | risk {short} {score:.0f}"]
                exited = [r.person_id for r in self.assoc.records.values()
                          if r.is_linked and self.vzt.canonical(r.vehicle_id) == cid]
                if exited:
                    lines.append("exited: " + ",".join(exited[:4]) + ("..." if len(exited) > 4 else ""))
                labels[d.track_id] = lines
            elif d.group == GROUP_PERSON:
                line1 = f"{d.track_id} {d.conf:.2f}"
                dec = self.gallery.identity(d.track_id) if self.gallery else None
                if dec is not None:
                    if dec.status == RECONNECTED and dec.profile_id != d.track_id:
                        line1 += f" = {dec.profile_id} ({dec.score:.2f})"
                    elif dec.status == CANDIDATE:
                        line1 += f" ?{dec.candidate_id} ({dec.score:.2f})"
                    elif dec.status == UNCERTAIN:
                        line1 += f" new (near {dec.candidate_id}?)"
                    elif dec.status == PENDING:
                        line1 += " id:?"
                elif self.gallery:
                    line1 += " id:?"
                lines = [line1]
                rec = self.assoc.get(d.track_id)
                if self.assoc.is_pending(d.track_id):
                    lines.append("exit: observing")
                elif rec is not None and rec.is_linked:
                    lines.append(f"{'exited' if rec.status == OBSERVED_EXIT else 'exited?'} "
                                 f"{self.vzt.canonical(rec.vehicle_id)} ({rec.confidence:.2f})")
                elif rec is not None and rec.status == AMBIGUOUS:
                    lines.append("exit: ambiguous")
                attrs = self.secondary.latest_clothing.get(d.track_id)
                if attrs and (attrs.get("upper_color") or attrs.get("lower_color")):
                    approx = "~" if attrs.get("segmentation") != "garment" else ""
                    lines.append(f"{approx}top:{attrs.get('upper_color') or '?'} bottom:{attrs.get('lower_color') or '?'}")
                prof = self.gallery.profile_for(d.track_id) if self.gallery else None
                if prof and prof.authorized:
                    lines.append(f"registered: {prof.authorized['label']}")
                labels[d.track_id] = lines

        acc = self.accounting.snapshot()
        disp = f"{self._display_fps:.1f}" if self.display is not None else "-"
        if self.live:
            hud = [
                f"LIVE {self.info.path} [{self.reader.status}] | {self.device}{' FP16' if self.half else ''}"
                f" | overlay: {self.renderer.mode_name}",
                f"camera {self.info.width}x{self.info.height} @ {self.reader.fps_meter.rate:.1f} FPS"
                f" | analysed {self.proc_rate.rate:.1f} | disp {disp}",
                f"t={ts:7.1f}s  frame {meta.index}",
            ]
        else:
            hud = [
                f"{self.mode.upper()} | {self.device}{' FP16' if self.half else ''} | overlay: {self.renderer.mode_name}",
                f"t={ts:7.2f}s  frame {meta.index}/{self.info.frame_count or '?'}",
                f"src {self.info.fps:.1f} FPS | proc {self.proc_rate.rate:.1f} | disp {disp}",
            ]
        people = sum(d.group == GROUP_PERSON for d in tracked)
        vehicles = [d for d in tracked if d.group == GROUP_VEHICLE]
        parked_ids = self.vzt.parked_ids()
        parked = sum(d.track_id in parked_ids for d in vehicles)
        hud += [
            f"detector {self.det_latency.ema:.0f} ms @ {imgsz} | ReID {self.secondary.reid_latency.ema:.0f} ms"
            f" | cloth {self.secondary.clothing_latency.ema:.0f} ms",
            f"in view: {people} people, {len(vehicles)} vehicles ({parked} parked)"
            + (f" | untracked {len(untracked)}" if untracked else ""),
            f"processed {acc['frames_processed']} | skipped {acc['frames_skipped']} | resets {acc['tracker_resets']}",
            f"ReID: {self.reid_backend.name if self.reid_backend else 'OFF'} | clothing: {self.clothing.mode}",
        ]
        hud += self._risk_lines(ts)
        if self.live:
            hud.append("stolen-vehicle check: NOT AVAILABLE (no plate reader / registry)")
            banner = "LIVE CAMERA - suspicious-behaviour risk only, not proof of theft"
        elif self.mode == "responsive":
            banner = "RESPONSIVE PREVIEW - frames may be skipped"
        else:
            banner = "COMPLETE EVIDENCE - every frame analysed"
        return RenderContext(tracked + untracked, labels, levels, hud, list(self.event_log), banner, self.zones)

    def _risk_lines(self, ts: float) -> list[str]:
        """Current highest risk and the strongest recent reason behind it, for the status panel."""
        top = max(((v, *self.risk.current(v, ts)) for v in self.risk.vehicles), key=lambda x: x[1], default=None)
        if top is None or top[1] < 1:
            return ["risk: none (no suspicious behaviour scored)"]
        vid, score, level = top
        lines = [f"risk: {level} on {vid} (score {score:.0f}/100)"]
        recent = [r for r in self.risk.vehicles[vid].reasons if r.points > 0]
        if recent:
            why = recent[-1].description
            lines.append("why: " + (why if len(why) <= 70 else why[:67] + "..."))
        return lines

    def _worker(self) -> None:
        current = None
        try:
            self._warmup_detector()
            self._worker_ready.set()
            while not self.stop_event.is_set():
                try:
                    item = self.in_queue.get(self.stop_event)
                except QueueClosed:
                    break
                if item is END_OF_STREAM:
                    break
                current = item
                annotated = self.process_frame(item)
                self.accounting.mark_processed(item.meta.index)
                current = None
                if self.display is None:
                    continue
                out = (annotated, item.meta)
                if self.mode == "responsive":
                    self.accounting.add("display_dropped", len(self.out_queue.put_drop_oldest(out)))
                elif not self.out_queue.put(out, self.stop_event):
                    break
        except BaseException as exc:
            self.worker_error = exc
            if current is not None:
                self.accounting.add("skipped")  # the frame that failed was not analysed
            log.exception("Analysis worker failed")
            self.stop_event.set()
        finally:
            self._worker_ready.set()
            self.out_queue.put_end()

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        if self.display_enabled:
            # Open the window first so something (the live camera, in live mode) is visible
            # while the models load, instead of a silent 10-20 s wait.
            self.display = self._window or DisplayWindow(self.s.app.window_name, self.s.app.display_max_width)
        try:
            self._setup_showing_status()
        except Exception as exc:
            if hasattr(self, "storage"):
                if hasattr(self, "session_id"):
                    self.storage.end_session(self.session_id, {"status": "setup_failed", "error": str(exc)})
                self.storage.close()
            if self.camera is not None:
                self.camera.release()
            if self.display is not None:
                self.display.close()
            raise
        self._t0 = time.perf_counter()
        self._started = True
        worker = threading.Thread(target=self._worker, name="analysis-worker", daemon=True)
        try:
            if self._quit_requested:
                log.info("Quit requested during start-up; nothing was analysed")
                worker = None
            else:
                self.secondary.start()
                worker.start()
                # Start the clock only once the detector is warm; keep the window alive meanwhile.
                while not self._worker_ready.wait(0.03):
                    if not self._show_status("Warming up the GPU..."):
                        self._quit_requested = True
                        self.stop_event.set()
                if not self._quit_requested:
                    self._t0 = time.perf_counter()
                    self.reader.start()
                    if self.display is not None:
                        self._display_loop()
                    else:
                        self._headless_wait(worker)
        except KeyboardInterrupt:
            log.info("Interrupted; shutting down")
        finally:
            summary = self.shutdown(worker)
        if self.worker_error is not None:
            raise PipelineError(f"Analysis failed: {self.worker_error}") from self.worker_error
        if self.reader.error is not None:
            what = "Camera failed" if self.live else "Video reading failed"
            raise PipelineError(f"{what}: {self.reader.error}") from self.reader.error
        return summary

    def _setup_showing_status(self) -> None:
        """Run setup (model loading) on a helper thread while the window shows a start-up screen."""
        if self.display is None:
            self.setup()
            return
        result: dict = {}

        def target():
            try:
                self.setup()
            except BaseException as exc:
                result["error"] = exc

        t = threading.Thread(target=target, name="setup", daemon=True)
        t.start()
        while t.is_alive():
            if not self._show_status("Loading AI models (the first start takes 10-20 s)..."):
                self._quit_requested = True  # honoured once loading finishes; it cannot be interrupted
            t.join(0.03)
        if "error" in result:
            raise result["error"]

    def _show_status(self, text: str) -> bool:
        """Draw one start-up frame. Returns False if the user asked to quit.

        In live mode this reads the camera directly (the reader thread has not started yet),
        so the real camera picture is visible from the first second. These preview frames are
        not analysed and not counted as source frames.
        """
        if self.display is None:
            return True
        if self.live and self.camera is not None and not self.camera.released:
            ok, frame = self.camera.read()
            if ok and frame is not None:
                self._status_frame = frame
        base = self._status_frame
        if base is None and self.live and self.camera is not None:
            base = self.camera.first_frame
        if base is None:
            base = np.zeros((self.info.height, self.info.width, 3), np.uint8)
        if self.live:
            lines = [f"LIVE CAMERA {self.info.path} - starting", text,
                     "Raw camera picture: boxes, IDs and risk appear when analysis starts.",
                     "Q / Esc / close window: quit"]
        else:
            lines = [f"VIDEO FILE {Path(self.info.path).name} - starting", text, "Q / Esc / close window: quit"]
        if self._quit_requested:
            lines[1] = "Quitting as soon as loading finishes..."
        self.display.show_status(render_status(base, lines))
        if self._startup_frames == 0:
            log.info("Window open: showing the %s while the models load", "live camera" if self.live else "start-up screen")
        self._startup_frames += 1
        return not self.display.poll_quit(1)

    def _display_loop(self) -> None:
        while True:
            paused = self.pause_event.is_set()
            item = self.out_queue.get_nowait()
            if item is END_OF_STREAM:
                break
            if item is not None:
                annotated, meta = item
                self.display.show(annotated)
                self._last_displayed = (annotated, meta)
                self.accounting.add("displayed")
                self.display_rate.tick()
                self._display_fps = self.display_rate.rate
            key = self.display.poll_key(30 if paused or item is None else 1)
            if key in KEY_QUIT or self.display.closed_by_user():
                log.info("Quit requested")
                break
            if key in KEY_PAUSE:
                if paused:
                    self.pause_event.clear()
                else:
                    self.pause_event.set()
                log.info("Paused" if not paused else "Resumed")
            elif key in KEY_SNAPSHOT:
                self.manual_snapshot()
            elif key in KEY_OVERLAY:
                log.info("Overlay mode: %s", self.renderer.cycle_overlay())

    def _headless_wait(self, worker: threading.Thread) -> None:
        last = time.perf_counter()
        while worker.is_alive():
            worker.join(0.5)
            if time.perf_counter() - last >= 5:
                last = time.perf_counter()
                log.info(self.progress_text())

    def progress_text(self) -> str:
        acc = self.accounting.snapshot()
        done = acc["frames_processed"] + acc["frames_skipped"]
        total = self.max_frames or self.info.frame_count
        rate = self.proc_rate.rate
        if total and rate > 0 and self.mode == "complete":
            remaining = max(0, total - done) / rate
            return f"progress {done}/{total} frames ({100 * done / total:.1f}%), {rate:.1f} FPS, ~{remaining:.0f}s remaining"
        return f"processed {acc['frames_processed']}, skipped {acc['frames_skipped']}, {rate:.1f} FPS"

    def manual_snapshot(self) -> str | None:
        if self._last_displayed is None:
            return None
        frame, meta = self._last_displayed
        path = self.paths.snapshots / f"manual_{meta.index:07d}.jpg"
        if not save_jpeg(path, frame, self.s.evidence.snapshot_jpeg_quality):
            self._note_write_error(f"snapshot failed: {path}")
            return None
        self.storage.add_event(session_id=self.session_id, video_path=self.info.path, frame=meta.index,
                               video_ts=meta.timestamp, category=EventKind.MANUAL_SNAPSHOT, risk_level="INFORMATIONAL",
                               message="snapshot saved by user", snapshot_path=str(path))
        log.info("Snapshot saved: %s", path)
        return str(path)

    # -------------------------------------------------------------- shutdown
    def stats_snapshot(self, short: bool = False) -> dict:
        acc = self.accounting.snapshot()
        out = {**acc, "processing_fps": round(self.proc_rate.rate, 2), "detector_latency_ms": round(self.det_latency.ema, 1)}
        if short:
            return out
        out.update(
            detector_latency=self.det_latency.as_dict(),
            frame_latency=self.frame_latency.as_dict(),
            reid_latency_per_crop=self.secondary.reid_latency.as_dict(),
            clothing_latency=self.secondary.clothing_latency.as_dict(),
            secondary_jobs_done=self.secondary.jobs_done,
            secondary_jobs_dropped=self.secondary.jobs_dropped,
            secondary_errors=self.secondary.errors,
            input_queue_max_depth=self.in_queue.max_depth,
            output_queue_max_depth=self.out_queue.max_depth,
            duplicate_tracks_removed=self.tracker.duplicates_removed,
            risk_events_suppressed_by_cooldown=self.risk.suppressed,
            apparent_entries_withdrawn=len(self.behavior.retracted_entries),
            startup_frames_shown=self._startup_frames,
            gpu_memory=gpu_memory_mb(self.device),
            final_imgsz=self.adaptive.size if self.adaptive else self.s.detector.imgsz,
        )
        return out

    def shutdown(self, worker: threading.Thread | None = None) -> dict:
        if not self._started:
            return {}
        self.stop_event.set()
        self.pause_event.clear()
        self.reader.join(5)
        if self.camera is not None:
            self.camera.release()  # free the webcam as soon as nothing reads from it
        if worker is not None:
            worker.join(15)
        leftover = [p for p in self.in_queue.drain() if p is not END_OF_STREAM]
        if leftover:
            self.accounting.add("skipped", len(leftover))
        self.out_queue.drain()
        self.secondary.stop()
        self._drain_secondary()
        self.clips.close_all()
        for clip in self.clips.finished:
            if clip["error"]:
                self._note_write_error(clip["error"])
                for aid in clip["event_ids"]:
                    if aid in self._event_ids:
                        self.storage.update_event_clip(self._event_ids[aid], None)
        if self.annotated is not None:
            self.annotated.release()
        if self.display is not None:
            self.display.close()

        wall = time.perf_counter() - self._t0
        stats = self.stats_snapshot()
        acc = self.accounting
        summary = {
            "session_id": self.session_id,
            "status": "error" if self.worker_error or self.reader.error else "finished",
            "video": self.info.as_dict(),
            "source": self.info.path,
            "mode": "live" if self.live else self.mode,
            "device": self.device,
            "fp16": self.half,
            "hardware": self.hardware,
            "wall_time_seconds": round(wall, 2),
            "average_processing_fps": round(acc.processed / wall, 2) if wall > 0 else 0.0,
            "frames_balanced": acc.is_balanced(),
            "reached_end_of_video": self.reader.finished_cleanly,
            "stats": stats,
            "alerts_by_level": self.alert_counts,
            "associations": self.association_counts,
            "identity_decisions": self.identity_counts,
            "models": {
                "detector": self.s.detector.weights,
                "classes": self.detector.mapping.describe() if hasattr(self.detector, "mapping") else None,
                "reid": self.reid_status,
                "clothing": self.clothing.status,
                "authorization": self.auth.status,
            },
            "zones": [z.name for z in self.zones.zones] or "none configured (whole frame)",
            "evidence_dir": str(self.paths.root),
            "clips": self.clips.finished,
            "write_errors": self.write_errors,
            "errors": [str(e) for e in (self.worker_error, self.reader.error) if e],
        }
        if self.live:
            summary["camera"] = {"backend": self.camera.backend_used, "released": self.camera.released,
                                 "failed_reads_at_end": self.reader.failed_reads,
                                 "annotated_video_saved": self.annotated is not None,
                                 "alert_clips_saved": self.clips_enabled}
        if self.annotated is not None:
            summary["annotated_video"] = self._verify_video(self.paths.annotated_video, self.annotated.frames)
            if self.mode == "responsive":
                summary["annotated_video"]["note"] = (
                    "responsive mode: contains processed frames only; the burned-in frame numbers show the source position")
        if self.gallery and self.s.privacy.save_profile_embeddings:
            embs = self.gallery.export()
            if embs:
                path = self.paths.root / f"profiles_{self.reid_backend.name}.npz"
                np.savez(path, **embs)
                summary["profiles_file"] = str(path)
        write_json(self.paths.summary, summary)
        self.storage.end_session(self.session_id, summary)
        self.storage.close()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return summary

    @staticmethod
    def _verify_video(path: Path, frames_written: int) -> dict:
        info = {"path": str(path), "frames_written": frames_written, "reopened": False}
        if frames_written == 0:
            return info
        try:
            vi = probe_video(path)
            info.update(reopened=True, frame_count=vi.frame_count, width=vi.width, height=vi.height)
        except Exception as exc:
            info["error"] = str(exc)
        return info
