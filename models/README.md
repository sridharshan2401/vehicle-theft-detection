# models/

Model weights live here. They are not committed to git.

| File / folder | What it is | How to get it |
|---|---|---|
| `yolo11s-seg.pt` | YOLO11-S instance segmentation (required) | `python scripts/download_models.py --detector` |
| `yolo11n-seg.pt` | YOLO11-N, faster and less accurate (optional) | `python scripts/download_models.py --detector --detector-name yolo11n-seg.pt` |
| `segformer_b2_clothes/` | SegFormer clothing parser (optional) | `python scripts/download_models.py --clothing` |
| `osnet_x0_25_msmt17.pth` | OSNet person ReID (optional, manual) | see [docs/MODEL_REQUIREMENTS.md](../docs/MODEL_REQUIREMENTS.md) |
| `model_status.json` | written by the download script: what was downloaded and verified | generated |

The torchvision ResNet-50 fallback for ReID goes to the PyTorch cache
(`%USERPROFILE%\.cache\torch\hub\checkpoints`), not this folder:
`python scripts/download_models.py --reid-generic`.

Run `python scripts/check_environment.py` to see which models are present.
