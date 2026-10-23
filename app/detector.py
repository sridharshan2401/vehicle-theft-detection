"""YOLO instance-segmentation detector for people and vehicles."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import DetectorConfig, resolve_path
from app.datatypes import GROUP_OTHER, GROUP_PERSON, GROUP_VEHICLE, Detection, box_iou
from app.performance import cuda_sync

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Model weights are missing, unreadable or not the right kind of model."""


@dataclass
class ClassMapping:
    """Which model class ids we use, and for what. Built from the model's own names table."""

    groups: dict[int, str] = field(default_factory=dict)  # class id -> group
    names: dict[int, str] = field(default_factory=dict)  # class id -> class name
    missing: list[str] = field(default_factory=list)  # configured names the model does not have

    @property
    def ids(self) -> list[int]:
        return sorted(self.groups)

    def describe(self) -> str:
        parts = [f"{g}: " + ", ".join(f"{self.names[i]}({i})" for i in self.ids if self.groups[i] == g)
                 for g in (GROUP_VEHICLE, GROUP_PERSON, GROUP_OTHER)]
        text = "; ".join(parts)
        if self.missing:
            text += f"; NOT in model: {', '.join(self.missing)}"
        return text


def build_class_mapping(model_names: dict[int, str], cfg: DetectorConfig) -> ClassMapping:
    by_name = {str(n).lower(): int(i) for i, n in model_names.items()}
    mapping = ClassMapping()
    for group, wanted in (
        (GROUP_VEHICLE, cfg.vehicle_classes),
        (GROUP_PERSON, cfg.person_classes),
        (GROUP_OTHER, cfg.other_classes),
    ):
        for name in wanted:
            cid = by_name.get(name.lower())
            if cid is None:
                mapping.missing.append(name)
            else:
                mapping.groups[cid] = group
                mapping.names[cid] = str(model_names[cid])
    return mapping


class Detector:
    """Loads a YOLO segmentation model once and runs it on frames.

    Boxes, classes, confidences and mask outlines all come from a single
    forward pass; no second segmentation model is run.
    """

    def __init__(self, cfg: DetectorConfig, device: str, half: bool):
        self.cfg = cfg
        self.device = device
        self.half = half
        self._quantize_api: bool | None = None
        self.model = self._load(cfg)
        if getattr(self.model, "task", None) != "segment":
            raise ModelUnavailableError(
                f"{cfg.weights} is a '{self.model.task}' model, not an instance-segmentation model. "
                "Use a *-seg.pt checkpoint such as yolo11s-seg.pt."
            )
        self.mapping = build_class_mapping(self.model.names, cfg)
        if not any(g == GROUP_VEHICLE for g in self.mapping.groups.values()):
            raise ModelUnavailableError(f"Model has none of the vehicle classes {cfg.vehicle_classes}")
        if not any(g == GROUP_PERSON for g in self.mapping.groups.values()):
            raise ModelUnavailableError(f"Model has none of the person classes {cfg.person_classes}")
        if self.mapping.missing:
            log.warning("Classes not supported by this model: %s", ", ".join(self.mapping.missing))
        self.half_verified: bool | None = None

    def _load(self, cfg: DetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailableError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
        path = resolve_path(cfg.weights)
        if not path.exists():
            if not cfg.auto_download:
                raise ModelUnavailableError(
                    f"Detector weights not found at {path}.\n"
                    "Download them with: python scripts/download_models.py --detector\n"
                    "(or set detector.auto_download: true to let Ultralytics fetch official weights)."
                )
            from ultralytics.utils.downloads import attempt_download_asset

            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading official Ultralytics weights %s", path.name)
            attempt_download_asset(str(path))
            if not path.exists():
                raise ModelUnavailableError(f"Automatic download of {path.name} failed.")
        try:
            return YOLO(str(path))
        except Exception as exc:
            raise ModelUnavailableError(f"Could not load detector weights {path}: {exc}") from exc

    def _precision_kwargs(self) -> dict:
        # Ultralytics 8.4 replaced `half=True` with `quantize=16`; support both.
        if self._quantize_api is None:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            self._quantize_api = "quantize" in DEFAULT_CFG_DICT
        if self._quantize_api:
            return {"quantize": 16 if self.half else 32}
        return {"half": self.half}

    def warmup(self, size: int, frame_shape: tuple[int, int] | None = None, runs: int = 2) -> None:
        """Run dummy inferences so CUDA kernels are initialised before the video clock starts.

        Use the real frame shape: a different aspect ratio letterboxes to a different
        input shape, and the first call at a new shape is slow (over a second here).
        """
        h, w = frame_shape or (size, size)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(runs):
            self.detect(dummy, size)

    def verify_half_precision(self, image: np.ndarray, imgsz: int) -> bool:
        """Compare FP16 against FP32 on one image. Falls back to FP32 if they disagree.

        Agreement means the same number of detections and every FP16 box matching an
        FP32 box of the same class with IoU >= 0.9.
        """
        if not self.half:
            return False
        fp16, _ = self.detect(image, imgsz)
        self.half = False
        fp32, _ = self.detect(image, imgsz)
        ok = len(fp16) == len(fp32) and all(
            any(a.class_id == b.class_id and box_iou(a.box, b.box) >= 0.9 for b in fp32) for a in fp16
        )
        self.half = ok
        self.half_verified = ok
        if not ok:
            log.warning("FP16 output differed from FP32 (%d vs %d detections); using FP32.", len(fp16), len(fp32))
        return ok

    def detect(self, frame: np.ndarray, imgsz: int) -> tuple[list[Detection], float]:
        """Return detections for one BGR frame and the inference latency in ms."""
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            classes=self.mapping.ids,
            max_det=self.cfg.max_det,
            device=self.device,
            verbose=False,
            **self._precision_kwargs(),
        )
        cuda_sync(self.device)
        latency_ms = (time.perf_counter() - t0) * 1000
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return [], latency_ms
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        # masks.xy holds one outline per detection in original-frame pixels.
        polygons = list(r.masks.xy) if r.masks is not None else [None] * len(boxes)
        dets = []
        for box, conf, cid, poly in zip(boxes, confs, classes, polygons):
            group = self.mapping.groups.get(int(cid))
            if group is None:
                continue
            if poly is not None and len(poly) < 3:
                poly = None  # tiny or degenerate mask; leave it out rather than invent one
            dets.append(
                Detection(
                    class_id=int(cid),
                    class_name=self.mapping.names[int(cid)],
                    group=group,
                    conf=float(conf),
                    box=box,
                    polygon=None if poly is None else np.asarray(poly, dtype=np.float32),
                )
            )
        return dets, latency_ms
