# Models

Pretrained weights only; nothing in this project was trained or fine-tuned.
`scripts/download_models.py` fetches models from fixed, documented sources and
writes what it downloaded and verified to `models/model_status.json`.
`python scripts/check_environment.py` prints the current state.

## Summary

| Purpose | Model | Source | Status on the dev machine |
|---|---|---|---|
| Detection + instance masks (required) | YOLO11-S seg (`yolo11s-seg.pt`) | Ultralytics release assets via `ultralytics.utils.downloads.attempt_download_asset` | downloaded, verified |
| Faster alternative | YOLO11-N seg (`yolo11n-seg.pt`) | same | not downloaded |
| Clothing / garment segmentation | SegFormer-B2 `mattmdjaga/segformer_b2_clothes` (revision `584abc1e1d26…`) | Hugging Face | downloaded, verified |
| ReID fallback (generic) | torchvision ResNet-50 `IMAGENET1K_V2` | download.pytorch.org via torchvision | downloaded, verified (shape only) |
| ReID (preferred) | OSNet via torchreid | KaiyangZhou/deep-person-reid model zoo | **not installed** (manual) |

"Verified" means a functional check, not an accuracy test. Results of those checks
on 2026-10-23 (from `model_status.json`):

* **YOLO11-S seg**: `task=segment`, 80 COCO classes. Ids: person 0, bicycle 1, car 2,
  motorcycle 3, bus 5, truck 7. No requested class missing. On Ultralytics' `bus.jpg`:
  5 detections, 5 masks (bus + 4 people), mask tensor `5×640×480`.
* **SegFormer clothes**: labels Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt,
  Pants, Dress, Belt, Left/Right-shoe, Face, Left/Right-leg, Left/Right-arm, Bag, Scarf.
  On a person crop from `zidane.jpg` it returned garment `upper-clothes`, colour black.
* **ResNet-50**: 2048-d unit-norm embeddings.

## Detector: YOLO11 instance segmentation

* Why: one forward pass gives boxes, classes and per-instance masks, so we never run a
  second segmentation model. The COCO classes cover person, bicycle, car, motorcycle,
  bus and truck. The `s` size is a reasonable speed/accuracy point for a laptop GPU;
  `n` is the fallback for 4 GB cards.
* Outputs used: `boxes.xyxy`, `boxes.conf`, `boxes.cls`, `masks.xy` (outline polygons in
  original-frame pixels).
* The app checks at start-up that `model.task == "segment"` and maps class *names* to ids
  from the model itself; missing classes are listed in the log and in `session.json`.
* FP16: on CUDA the app runs FP16 and FP32 on the same test image at start-up and only
  keeps FP16 if the detections match (same count, IoU ≥ 0.9). It matched on the RTX 5050.
* Licence: AGPL-3.0 (Ultralytics). Commercial use needs an Ultralytics licence.

## Clothing: SegFormer human parsing

* Model card: <https://huggingface.co/mattmdjaga/segformer_b2_clothes>. SegFormer-B2
  fine-tuned on the ATR human parsing dataset (18 labels above).
* The download script pins the exact revision and fetches only `config.json`,
  `preprocessor_config.json`, `model.safetensors` and `README.md` (no pickled files).
* What it can tell us: upper clothes vs pants vs skirt vs dress regions. What it cannot:
  outerwear vs shirt, brands, exact patterns. The app says "outerwear is not distinguished
  by this model" in the attributes it stores.
* Licence: the Hugging Face card says `license: other` without further text. Treat it
  as research/prototype use until you have confirmed terms with the author; the ATR
  dataset has its own academic terms.
* Without this model the app uses **person-mask bands** (upper/lower body strips of the
  YOLO instance mask). Those results are tagged `segmentation: approx_bands`, have no
  garment types, get a 0.6× quality factor and show with a `~` prefix on screen.

## Re-identification

The app picks the first backend that works, in this order:

1. **torchreid / OSNet** (person-specific). Not installed on the dev machine.
2. **torchvision ResNet-50** (generic ImageNet features). This is what ran in all tests.
   It uses stricter thresholds (match ≥ 0.88) and is **not** used for authorisation.
3. **HSV histograms** (no model). Weakest, last resort.

### Installing OSNet (manual)

The `torchreid` package on PyPI (0.2.5) is a third-party repackaging, not the official
project, so the scripts do not install it. Use the official repository:

```powershell
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -r requirements.txt
python setup.py develop
```

On Windows the last step compiles a Cython extension and needs the
"Desktop development with C++" workload from Visual Studio Build Tools.

Then download an OSNet person-ReID checkpoint from the project's model zoo
(<https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO>), put it in `models/`, and
point the settings at it:

```yaml
reid:
  torchreid_model: osnet_x0_25        # must match the checkpoint's architecture
  torchreid_weights: models/<checkpoint file>.pth
```

Check it with `python scripts/download_models.py --verify`. The torchreid thresholds
(`match 0.70`, `candidate 0.55`) are starting points and have **not** been tuned on
real data. Model-zoo weights trained on MSMT17 / Market-1501 inherit those datasets'
research-only terms.

## Model registry in code

`app/model_registry.py` lists every model with purpose, source, expected path, outputs,
licence notes and setup command. `scripts/check_environment.py` prints it with live
"present / verified" status.
