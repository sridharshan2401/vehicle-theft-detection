"""Configuration loading and validation.

All settings have defaults here, so configs/settings.yaml only needs the values
you want to change. Unknown keys are rejected so a typo does not silently fall
back to a default.
"""

from __future__ import annotations

import copy
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"

RISK_LEVELS = ["INFORMATIONAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
MODES = ["responsive", "complete"]
SOURCES = ["camera", "video"]
CAMERA_BACKENDS = ["auto", "dshow", "msmf", "any"]


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed or invalid."""


@dataclass
class AppConfig:
    source: str = "camera"  # camera (live webcam, the default) | video (prerecorded file via --video)
    mode: str = "responsive"  # video files only; the live camera always runs responsive
    display: bool = True
    window_name: str = "Vehicle Theft Detection"
    display_max_width: int = 1280
    log_level: str = "INFO"


@dataclass
class CameraConfig:
    index: int = 0  # 0 = default laptop webcam; 1, 2... for other connected cameras
    width: int = 1280  # requested resolution; 0 keeps the driver default
    height: int = 720
    fps: float = 30.0  # requested frame rate; 0 keeps the driver default
    backend: str = "auto"  # auto | dshow | msmf | any  (auto tries DirectShow, then Media Foundation on Windows)
    open_timeout_seconds: float = 10.0  # how long to wait for the first frame
    max_failed_reads: int = 50  # consecutive failed reads before the camera counts as disconnected
    queue_size: int = 2  # frames waiting for analysis; small keeps latency low (older frames are dropped)
    save_annotated_video: bool = False  # live mode never records the full feed unless this is turned on
    save_alert_clips: bool = False  # short pre/post clips for MEDIUM+ alerts (snapshots are always saved)


@dataclass
class DeviceConfig:
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    half: str = "auto"  # auto | on | off  (FP16 is only used on CUDA)


@dataclass
class DetectorConfig:
    weights: str = "models/yolo11s-seg.pt"
    auto_download: bool = False
    imgsz: int = 640
    min_imgsz: int = 416
    adaptive_imgsz: bool = True
    conf: float = 0.35
    iou: float = 0.5
    max_det: int = 100
    vehicle_classes: list[str] = field(default_factory=lambda: ["car", "motorcycle", "bus", "truck"])
    person_classes: list[str] = field(default_factory=lambda: ["person"])
    other_classes: list[str] = field(default_factory=lambda: ["bicycle"])


@dataclass
class TrackerConfig:
    type: str = "bytetrack"  # bytetrack | botsort
    max_gap_frames: int = 15
    lost_after_seconds: float = 1.0
    forget_after_seconds: float = 30.0
    duplicate_iou: float = 0.85
    stationary_speed: float = 0.08  # box diagonals per second


@dataclass
class PipelineConfig:
    input_queue_size: int = 6
    output_queue_size: int = 3
    target_fps: float = 20.0
    secondary_queue_size: int = 16


@dataclass
class ReIDConfig:
    enabled: bool = True
    backend: str = "auto"  # auto | torchreid | torchvision | histogram
    torchreid_model: str = "osnet_x0_25"
    torchreid_weights: str = "models/osnet_x0_25_msmt17.pth"
    min_crop_width: int = 32
    min_crop_height: int = 80
    min_sharpness: float = 25.0
    sample_interval_frames: int = 10
    max_crops_per_frame: int = 4
    max_embeddings_per_profile: int = 30
    min_samples_for_decision: int = 2
    max_pending_samples: int = 6
    decision_margin: float = 0.05
    clothing_weight: float = 0.10
    active_seconds: float = 1.0
    thresholds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "torchreid": {"match": 0.70, "candidate": 0.55},
            "torchvision": {"match": 0.88, "candidate": 0.80},
            "histogram": {"match": 0.92, "candidate": 0.85},
        }
    )
    authorization_threshold: float = 0.75
    allow_generic_backend_for_authorization: bool = False


@dataclass
class ClothingConfig:
    enabled: bool = True
    backend: str = "auto"  # auto | segformer | person_mask_bands | off
    segformer_dir: str = "models/segformer_b2_clothes"
    interval_frames: int = 30
    min_crop_height: int = 96
    min_region_pixels: int = 150
    pattern_min_pixels: int = 2500


@dataclass
class AssociationConfig:
    observation_seconds: float = 2.5
    min_observations: int = 4
    birth_overlap_min: float = 0.35
    door_margin: float = 0.25
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "birth_overlap": 0.30,
            "door_zone": 0.15,
            "outward_motion": 0.25,
            "vehicle_stationary": 0.15,
            "recent_arrival": 0.15,
        }
    )
    confirm_threshold: float = 0.60
    candidate_threshold: float = 0.40
    ambiguity_margin: float = 0.12
    recent_arrival_seconds: float = 120.0
    reappear_seconds: float = 2.0
    reappear_penalty: float = 0.4
    border_penalty: float = 0.5


@dataclass
class ZonesConfig:
    file: str = "configs/zones.json"
    park_seconds: float = 3.0
    move_threshold: float = 0.25  # fraction of the vehicle's box diagonal
    # A parked vehicle only counts as moved when the displacement lasts this long, so a
    # one-off box change (someone standing in front, a partial detection) is ignored.
    move_confirm_seconds: float = 1.0
    # Frames where the box area changed by more than this fraction are treated as occlusion,
    # not movement (unless the displacement is very large or the box is at the frame edge).
    max_area_change: float = 0.4


@dataclass
class BehaviorConfig:
    proximity_margin: float = 0.6  # fraction of vehicle width added on each side
    contact_overlap: float = 0.25
    entry_overlap: float = 0.35
    loiter_seconds: float = 20.0
    repeated_count: int = 3
    reapproach_gap_seconds: float = 3.0
    entrant_memory_seconds: float = 120.0
    # Proximity only counts once a person has lingered (moved slower than linger_speed,
    # in box diagonals per second) inside the zone for proximity_min_seconds. Someone
    # walking past at normal pace (~1 diagonal/s) never triggers it.
    proximity_min_seconds: float = 1.5
    linger_speed: float = 0.35
    # An apparent entry is only reported if nobody re-appears near that spot within
    # entry_confirm_seconds (otherwise they were just hidden behind the vehicle). Vehicle
    # movement during that wait confirms it immediately.
    entry_confirm_seconds: float = 3.0
    reappear_radius: float = 1.5  # in person-box heights


@dataclass
class RiskConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "proximity": 10.0,
            "repeated_interaction": 20.0,
            "extended_presence": 15.0,
            "apparent_entry": 30.0,
            "restricted_area": 10.0,
            "vehicle_moved": 20.0,
            "vehicle_left_zone": 25.0,
            "vehicle_departed": 20.0,
        }
    )
    identity_factors: dict[str, float] = field(
        default_factory=lambda: {
            "authorized": 0.0,
            "observed_associated": 0.3,
            "possible_associated": 0.6,
            "unrecognized": 1.0,
            "no_entrant_observed": 0.5,
        }
    )
    levels: dict[str, float] = field(
        default_factory=lambda: {"LOW": 15.0, "MEDIUM": 35.0, "HIGH": 55.0, "CRITICAL": 80.0}
    )
    half_life_seconds: float = 120.0
    cooldown_seconds: float = 20.0
    alert_min_level: str = "LOW"
    realert_seconds: float = 60.0
    max_score: float = 100.0


@dataclass
class EvidenceConfig:
    output_dir: str = "evidence"
    save_annotated_video: bool = True
    min_level: str = "MEDIUM"
    pre_event_seconds: float = 3.0
    post_event_seconds: float = 5.0
    clip_max_width: int = 960
    snapshot_jpeg_quality: int = 90
    retention_days: int = 0  # 0 = keep forever; checked at startup


@dataclass
class StorageConfig:
    db_path: str = "evidence/vtd.sqlite3"


@dataclass
class PrivacyConfig:
    face_processing: bool = False
    save_profile_embeddings: bool = True


@dataclass
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    clothing: ClothingConfig = field(default_factory=ClothingConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected true/false, got {value!r}")
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if isinstance(value, bool):  # YAML turns on/off into booleans
            return "on" if value else "off"
        if not isinstance(value, (str, int, float)):
            raise ConfigError(f"{where}: expected text, got {value!r}")
        return str(value)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        (item_hint,) = typing.get_args(hint)
        return [_coerce(v, item_hint, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        _, val_hint = typing.get_args(hint)
        return {str(k): _coerce(v, val_hint, f"{where}.{k}") for k, v in value.items()}
    return value


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'settings'}: expected a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where or 'settings'}: unknown key(s) {sorted(unknown)}")
    obj = cls()
    for name, value in data.items():
        hint = hints[name]
        key = f"{where}.{name}" if where else name
        if is_dataclass(hint):
            setattr(obj, name, _build(hint, value or {}, key))
        elif typing.get_origin(hint) is dict and isinstance(getattr(obj, name), dict):
            # Dict settings merge with the defaults so partial overrides work.
            merged = dict(getattr(obj, name))
            merged.update(_coerce(value, hint, key))
            setattr(obj, name, merged)
        else:
            setattr(obj, name, _coerce(value, hint, key))
    return obj


def validate(s: Settings) -> Settings:
    """Check ranges and enums. Raises ConfigError with a readable message."""
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(s.app.source in SOURCES, f"app.source must be one of {SOURCES}")
    check(s.app.mode in MODES, f"app.mode must be one of {MODES}")
    check(s.app.display_max_width >= 320, "app.display_max_width must be >= 320")
    c = s.camera
    check(c.index >= 0, "camera.index must be >= 0")
    check(c.width == 0 or 160 <= c.width <= 7680, "camera.width must be 0 (driver default) or 160..7680")
    check(c.height == 0 or 120 <= c.height <= 4320, "camera.height must be 0 (driver default) or 120..4320")
    check(c.fps == 0 or 1 <= c.fps <= 240, "camera.fps must be 0 (driver default) or 1..240")
    check(c.backend in CAMERA_BACKENDS, f"camera.backend must be one of {CAMERA_BACKENDS}")
    check(c.open_timeout_seconds > 0, "camera.open_timeout_seconds must be > 0")
    check(c.max_failed_reads >= 1, "camera.max_failed_reads must be >= 1")
    check(c.queue_size >= 1, "camera.queue_size must be >= 1")
    check(s.device.half in ("auto", "on", "off"), "device.half must be auto, on or off")
    check(
        s.device.device == "auto" or s.device.device == "cpu" or s.device.device.startswith("cuda"),
        "device.device must be auto, cpu, cuda or cuda:N",
    )
    d = s.detector
    check(d.imgsz % 32 == 0 and 160 <= d.imgsz <= 1920, "detector.imgsz must be a multiple of 32 in [160, 1920]")
    check(d.min_imgsz % 32 == 0 and 160 <= d.min_imgsz <= d.imgsz, "detector.min_imgsz must be a multiple of 32 and <= imgsz")
    check(0.0 < d.conf < 1.0, "detector.conf must be in (0, 1)")
    check(0.0 < d.iou < 1.0, "detector.iou must be in (0, 1)")
    check(bool(d.vehicle_classes), "detector.vehicle_classes must not be empty")
    check(bool(d.person_classes), "detector.person_classes must not be empty")
    check(s.tracker.type in ("bytetrack", "botsort"), "tracker.type must be bytetrack or botsort")
    check(s.tracker.max_gap_frames >= 1, "tracker.max_gap_frames must be >= 1")
    check(s.tracker.lost_after_seconds > 0, "tracker.lost_after_seconds must be > 0")
    p = s.pipeline
    check(p.input_queue_size >= 1 and p.output_queue_size >= 1, "pipeline queue sizes must be >= 1")
    check(p.target_fps > 0, "pipeline.target_fps must be > 0")
    r = s.reid
    check(r.backend in ("auto", "torchreid", "torchvision", "histogram"), "reid.backend is invalid")
    for name, th in r.thresholds.items():
        check(
            {"match", "candidate"} <= set(th) and 0 < th["candidate"] <= th["match"] <= 1,
            f"reid.thresholds.{name} needs 0 < candidate <= match <= 1",
        )
    check(r.min_samples_for_decision >= 1, "reid.min_samples_for_decision must be >= 1")
    check(r.max_pending_samples >= r.min_samples_for_decision, "reid.max_pending_samples must be >= min_samples_for_decision")
    check(s.clothing.backend in ("auto", "segformer", "person_mask_bands", "off"), "clothing.backend is invalid")
    a = s.association
    check(0 < a.candidate_threshold <= a.confirm_threshold <= 1, "association needs 0 < candidate_threshold <= confirm_threshold <= 1")
    check(abs(sum(a.weights.values()) - 1.0) < 1e-6, "association.weights must sum to 1.0")
    check(a.observation_seconds > 0, "association.observation_seconds must be > 0")
    k = s.risk
    check(set(k.levels) == set(RISK_LEVELS[1:]), f"risk.levels must define {RISK_LEVELS[1:]}")
    if set(k.levels) == set(RISK_LEVELS[1:]):
        vals = [k.levels[lv] for lv in RISK_LEVELS[1:]]
        check(all(a_ < b_ for a_, b_ in zip(vals, vals[1:])) and vals[0] > 0, "risk.levels must be positive and strictly increasing")
    check(all(w >= 0 for w in k.weights.values()), "risk.weights must be >= 0")
    check(all(0 <= f <= 1 for f in k.identity_factors.values()), "risk.identity_factors must be in [0, 1]")
    check(k.half_life_seconds > 0, "risk.half_life_seconds must be > 0")
    check(k.cooldown_seconds >= 0, "risk.cooldown_seconds must be >= 0")
    check(k.alert_min_level in RISK_LEVELS, "risk.alert_min_level must be a risk level")
    e = s.evidence
    check(e.min_level in RISK_LEVELS, "evidence.min_level must be a risk level")
    check(e.pre_event_seconds >= 0 and e.post_event_seconds >= 0, "evidence pre/post seconds must be >= 0")
    check(1 <= e.snapshot_jpeg_quality <= 100, "evidence.snapshot_jpeg_quality must be in [1, 100]")
    check(e.retention_days >= 0, "evidence.retention_days must be >= 0")
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    return s


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Load settings from YAML (defaults to configs/settings.yaml) and validate them.

    ``overrides`` uses dotted keys, e.g. {"app.mode": "complete"}.
    """
    data: dict[str, Any] = {}
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if path is not None and not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    if cfg_path.exists():
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {cfg_path}: {exc}") from exc
        data = loaded or {}
    data = copy.deepcopy(data)
    for dotted, value in (overrides or {}).items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return validate(_build(Settings, data, ""))
