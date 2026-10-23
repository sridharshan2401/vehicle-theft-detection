"""Download and verify the models from their official sources.

    python scripts/download_models.py --all
    python scripts/download_models.py --detector [--detector-name yolo11n-seg.pt]
    python scripts/download_models.py --clothing
    python scripts/download_models.py --reid-generic
    python scripts/download_models.py --verify          (re-check what is already on disk)

Results are written to models/model_status.json. Only these fixed, documented
sources are used; the script never downloads from an arbitrary URL.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from app.model_registry import load_status, save_status  # noqa: E402

YOLO_ALLOWED = {"yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt"}
SEGFORMER_REPO = "mattmdjaga/segformer_b2_clothes"
SEGFORMER_FILES = ["README.md", "config.json", "preprocessor_config.json", "model.safetensors"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def detector(name: str, status: dict) -> None:
    if name not in YOLO_ALLOWED:
        raise SystemExit(f"--detector-name must be one of {sorted(YOLO_ALLOWED)}")
    from ultralytics.utils.downloads import attempt_download_asset

    path = PROJECT_ROOT / "models" / name
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        print(f"Downloading {name} from the official Ultralytics release assets...")
        attempt_download_asset(str(path))
    key = "detector" if name == "yolo11s-seg.pt" else "detector_nano" if name == "yolo11n-seg.pt" else name
    status[key] = {"downloaded": path.exists(), "path": str(path.relative_to(PROJECT_ROOT)), "at": now()}
    verify_detector(path, status[key])


def verify_detector(path: Path, entry: dict) -> None:
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    model = YOLO(str(path))
    res = model.predict(str(ASSETS / "bus.jpg"), verbose=False)[0]
    names = model.names
    wanted = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    entry.update(
        task=model.task,
        num_classes=len(names),
        class_ids={n: i for i, n in names.items() if n in wanted},
        missing_classes=[n for n in wanted if n not in names.values()],
        test_image="ultralytics/assets/bus.jpg",
        test_detections=int(len(res.boxes)),
        test_masks=int(0 if res.masks is None else len(res.masks)),
        test_classes=sorted({names[int(c)] for c in res.boxes.cls.cpu().numpy().astype(int)}),
        mask_shape=list(res.masks.data.shape) if res.masks is not None else None,
        verified=bool(model.task == "segment" and res.masks is not None and len(res.masks) == len(res.boxes)
                      and np.all([len(p) >= 3 for p in res.masks.xy])),
        verified_at=now(),
    )
    print(f"  detector {path.name}: task={entry['task']}, {entry['test_detections']} detections / "
          f"{entry['test_masks']} masks on bus.jpg ({', '.join(entry['test_classes'])}) -> verified={entry['verified']}")


def clothing(status: dict) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    target = resolve_path(load_config().clothing.segformer_dir)
    target.mkdir(parents=True, exist_ok=True)
    sha = HfApi().model_info(SEGFORMER_REPO).sha
    print(f"Downloading {SEGFORMER_REPO} @ {sha[:10]} (weights, config, README only)...")
    for f in SEGFORMER_FILES:
        hf_hub_download(SEGFORMER_REPO, f, revision=sha, local_dir=str(target))
    status["clothing"] = {"downloaded": True, "path": str(target.relative_to(PROJECT_ROOT)), "revision": sha, "at": now()}
    verify_clothing(target, status["clothing"])


def verify_clothing(target: Path, entry: dict) -> None:
    import cv2
    from ultralytics.utils import ASSETS

    from app.clothing_analyzer import ClothingAnalyzer
    from app.config import ClothingConfig

    cfg = ClothingConfig(segformer_dir=str(target))
    an = ClothingAnalyzer(cfg, "cuda:0" if _cuda() else "cpu")
    entry["labels"] = sorted(an.backend.labels.values()) if an.backend else None
    img = cv2.imread(str(ASSETS / "zidane.jpg"))
    crop = img[100:720, 120:620]  # a person in the Ultralytics sample image
    attrs = an.analyze(crop, None) if an.mode == "segformer" else None
    entry.update(mode=an.mode, test_attributes=attrs, verified=an.mode == "segformer" and attrs is not None,
                 verified_at=now())
    print(f"  clothing: mode={an.mode}; test crop -> {attrs and {k: attrs[k] for k in ('upper_color', 'lower_color', 'garments')}}")


def reid_generic(status: dict) -> None:
    import torch
    from torchvision.models import ResNet50_Weights

    w = ResNet50_Weights.IMAGENET1K_V2
    print("Fetching torchvision ResNet-50 ImageNet weights into the torch hub cache...")
    w.get_state_dict(progress=True)
    path = Path(torch.hub.get_dir()) / "checkpoints" / w.url.rsplit("/", 1)[-1]
    status["reid_generic"] = {"downloaded": path.exists(), "path": str(path), "at": now()}
    verify_reid(status["reid_generic"], "torchvision")


def verify_reid(entry: dict, backend: str) -> None:
    import numpy as np

    from app.config import ReIDConfig
    from app.person_reid import build_reid_backend

    be, msg = build_reid_backend(ReIDConfig(backend=backend), "cuda:0" if _cuda() else "cpu")
    if be is None:
        entry.update(verified=False, error=msg)
        print(f"  reid {backend}: {msg}")
        return
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (256, 128, 3), dtype=np.uint8) for _ in range(2)]
    emb = be.embed(crops, [None, None])
    entry.update(embedding_dim=int(emb.shape[1]), unit_norm=bool(np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-3)),
                 verified=True, verified_at=now(), note="shape/normalisation check only, not an accuracy test")
    print(f"  reid {backend}: embedding dim {emb.shape[1]}")


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--detector", action="store_true")
    ap.add_argument("--detector-name", default="yolo11s-seg.pt")
    ap.add_argument("--clothing", action="store_true")
    ap.add_argument("--reid-generic", action="store_true")
    ap.add_argument("--verify", action="store_true", help="re-verify models already on disk")
    args = ap.parse_args()
    if not any((args.all, args.detector, args.clothing, args.reid_generic, args.verify)):
        ap.print_help()
        return 1
    status = load_status()
    failures = 0
    steps = []
    if args.all or args.detector:
        steps.append(("detector", lambda: detector(args.detector_name, status)))
    if args.all or args.clothing:
        steps.append(("clothing", lambda: clothing(status)))
    if args.all or args.reid_generic:
        steps.append(("reid_generic", lambda: reid_generic(status)))
    if args.verify:
        cfg = load_config()
        det = resolve_path(cfg.detector.weights)
        if det.exists():
            steps.append(("detector", lambda: verify_detector(det, status.setdefault("detector", {"downloaded": True}))))
        seg = resolve_path(cfg.clothing.segformer_dir)
        if seg.exists():
            steps.append(("clothing", lambda: verify_clothing(seg, status.setdefault("clothing", {"downloaded": True}))))
        steps.append(("reid_generic", lambda: verify_reid(status.setdefault("reid_generic", {}), "torchvision")))
        if resolve_path(cfg.reid.torchreid_weights).exists():
            steps.append(("reid_osnet", lambda: verify_reid(status.setdefault("reid_osnet", {"downloaded": True}), "torchreid")))
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:
            failures += 1
            status.setdefault(name, {})["error"] = f"{exc.__class__.__name__}: {exc}"
            print(f"FAILED {name}: {exc}")
            traceback.print_exc()
        save_status(status)
    print(f"\nStatus written to {Path('models') / 'model_status.json'}")
    print("OSNet (torchreid) is not downloaded automatically; see docs/MODEL_REQUIREMENTS.md.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
