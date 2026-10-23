# Datasets

The prototype runs on pretrained weights. No dataset was downloaded or used for
training in this build, and no training results exist. This guide is about what
you would need if you decide to fine-tune, and how to check a dataset before
trusting it.

## Do we need to train at all?

Probably not for detection: COCO-pretrained YOLO11-seg already finds people and cars
well in daylight footage (see the real-run screenshots). Fine-tuning becomes worth it when
you have measured a problem on *your* cameras, for example:

* top-down or fisheye views where COCO-style cars are rare,
* night / IR footage,
* many missed people next to cars (occlusion at doors).

Do the measurement first (docs/TESTING.md, "staged test set"), then train.

## Matching the dataset to the task

| Task in this project | Annotation you need | Not good enough |
|---|---|---|
| Detector + instance masks | per-object polygons/masks with class labels | boxes only (that is detection, not segmentation) |
| Person ReID | the same person id across several cameras or times | images grouped by class only |
| Clothing segmentation | per-pixel garment labels (human parsing) | image-level tags like "red shirt" |
| Association / risk evaluation | your own staged videos with labelled events | any of the above |

## Candidate public datasets

These are well-known datasets I know of, **not checked in this build**. Before
using any of them, open the official page and confirm the items in the checklist below;
licences and availability change (some ReID datasets have been withdrawn by their authors).

| Dataset | Useful for | Annotation type | Things to check |
|---|---|---|---|
| COCO 2017 | detector baseline (already what YOLO11 was trained on) | instance polygons, 80 classes | annotation licence vs image (Flickr) licences |
| Cityscapes | street-level instance masks | instance masks for person/car/etc. | registration required, non-commercial terms; street not car-park |
| LIP / ATR | clothing parsing (ATR is what the SegFormer model used) | per-pixel human parsing labels | academic-use terms |
| DeepFashion2 | garment categories | garment masks, mostly product/consumer photos | non-commercial terms; very different viewpoint from CCTV |
| Market-1501, MSMT17 | person ReID | person ids across cameras | research-only; check whether still distributed and under what terms |
| PKLot, CNRPark+EXT | parking-space occupancy | per-space occupied/empty labels | classification labels, **not** instance masks; useful for zone ideas only |

### Roboflow Universe

Roboflow Universe hosts many user-uploaded car-park datasets. I did not evaluate
specific ones for this build. Many are box-only detection sets even when the title
says "segmentation", and licences vary per dataset. Use the checklist.

## Checklist before using a dataset

1. **Classes**: list them. Do they include person and the vehicle types you care about?
2. **Annotation type**: polygons/masks, or only boxes? Identity ids? Garment pixels?
3. **Licence**: commercial use? Redistribution? Attribution? Write it down.
4. **Consent / privacy**: were the people filmed in a way that allows this use?
5. **Domain**: camera height and angle, day/night, resolution, compression vs your footage.
6. **Split**: is there a test split you will never train on? Are the same scenes in train and test?
7. **Quality**: open 50 random samples and look at the masks yourself.

## Fine-tuning workflow (if you need it)

A reproducible outline with the tools already installed (Ultralytics):

1. Convert to YOLO segmentation format (`images/`, `labels/` with polygons, one
   `data.yaml` listing class names). Ultralytics' own converters handle COCO JSON.
2. Validate: count images per split, instances per class, and render 50 random labels
   to images and check them by eye.
3. Split by **scene / camera / day**, not by frame, so near-identical frames do not leak
   between train and test. Keep the test split frozen.
4. Train, starting from the pretrained weights:

   ```powershell
   yolo segment train model=models/yolo11s-seg.pt data=path\to\data.yaml epochs=100 imgsz=640 batch=8 seed=0 deterministic=True
   ```

   Ultralytics applies mosaic, HSV and flip augmentation by default; add low-light /
   blur augmentation if night footage matters.
5. Evaluate on the frozen test split: `yolo segment val model=runs/segment/train/weights/best.pt data=...`
   and record box mAP50-95 and mask mAP50-95 per class.
6. Pick the checkpoint by validation mask mAP, then run it once on test. Compare with
   the pretrained model on the same test set before switching `detector.weights`.
7. Keep the exact command, data version, seed and resulting numbers in a notes file.
