"""What models the project uses, where they come from, and whether they are present.

scripts/download_models.py writes models/model_status.json after downloading
and verifying a model; this module reads it back for status reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from app.config import PROJECT_ROOT

STATUS_FILE = PROJECT_ROOT / "models" / "model_status.json"


@dataclass
class ModelSpec:
    key: str
    purpose: str
    source: str
    path: str
    outputs: str
    license: str
    setup: str
    required: bool


MODELS = [
    ModelSpec(
        key="detector",
        purpose="Person/vehicle detection with instance masks (one forward pass)",
        source="Ultralytics YOLO11 segmentation, official release assets (github.com/ultralytics/assets)",
        path="models/yolo11s-seg.pt",
        outputs="boxes, classes, confidences and instance masks for the 80 COCO classes",
        license="AGPL-3.0 (Ultralytics); commercial use needs an Ultralytics Enterprise licence",
        setup="python scripts/download_models.py --detector",
        required=True,
    ),
    ModelSpec(
        key="detector_nano",
        purpose="Faster, less accurate alternative detector",
        source="Ultralytics YOLO11 segmentation, official release assets",
        path="models/yolo11n-seg.pt",
        outputs="same as detector",
        license="AGPL-3.0 (Ultralytics)",
        setup="python scripts/download_models.py --detector --detector-name yolo11n-seg.pt",
        required=False,
    ),
    ModelSpec(
        key="clothing",
        purpose="Garment segmentation (upper clothes, pants, skirt, dress...)",
        source="Hugging Face mattmdjaga/segformer_b2_clothes (SegFormer-B2 fine-tuned on the ATR human-parsing dataset)",
        path="models/segformer_b2_clothes/",
        outputs="per-pixel labels: Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt, shoes, Face, legs, arms, Bag, Scarf",
        license="Marked 'other' on Hugging Face; read models/segformer_b2_clothes/README.md. The ATR dataset has its own terms.",
        setup="python scripts/download_models.py --clothing",
        required=False,
    ),
    ModelSpec(
        key="reid_generic",
        purpose="Fallback appearance embeddings for re-identification (generic, not ReID-trained)",
        source="torchvision ResNet-50 IMAGENET1K_V2 weights from download.pytorch.org",
        path="<torch hub cache>/checkpoints/resnet50-11ad3fa6.pth",
        outputs="2048-d global image feature",
        license="torchvision BSD-3-Clause; ImageNet weights are for non-commercial research use per ImageNet terms",
        setup="python scripts/download_models.py --reid-generic",
        required=False,
    ),
    ModelSpec(
        key="reid_osnet",
        purpose="Pedestrian re-identification embeddings (preferred ReID backend)",
        source="deep-person-reid / torchreid by Kaiyang Zhou (github.com/KaiyangZhou/deep-person-reid), model zoo weight "
               "osnet_x0_25 trained on MSMT17",
        path="models/osnet_x0_25_msmt17.pth",
        outputs="512-d person embedding",
        license="MIT (code); check the model zoo page for dataset terms (MSMT17 is research-only)",
        setup="manual: install torchreid from the official GitHub repo and place the weight file; see docs/MODEL_REQUIREMENTS.md",
        required=False,
    ),
]


def load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")


def as_table() -> list[dict]:
    status = load_status()
    return [{**asdict(m), **status.get(m.key, {"downloaded": False, "verified": False})} for m in MODELS]
