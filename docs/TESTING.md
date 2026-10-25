# Testing

## How to run

```powershell
python -m pytest                       # everything (about 20 s)
python -m pytest tests\test_risk_engine.py -v
```

## What the tests are and are not

* **Logic tests** use hand-made tracks, synthetic videos of coloured rectangles, a
  colour-blob "detector" and a simple IoU tracker (all in `tests/conftest.py`). They
  check that the software does the right thing with given inputs. They say nothing about
  how well the real models see real people.
* **Model smoke tests** (`test_model_smoke.py`) load the real downloaded models and run
  them on the two sample images that ship with Ultralytics. They check output shapes,
  classes and that masks exist. They skip themselves if the weights are missing.
* **Real-video runs** use a public car-park clip and are described below with their
  actual numbers.

## Automated test results

Run on 2026-10-25 from `E:\vehicle-theft-detection`, Windows 11, Python 3.12.10, RTX 5050 Laptop GPU:
**127 passed, 0 failed, 0 skipped.** (70 in the first round; 31 added with the evaluation harness
and false-positive work; 21 added with live camera mode; 5 added with the start-up window and
`run_webcam.py` launcher, see "Live start-up fixes" below.)

| File | Tests | Covers |
|---|---|---|
| `test_config.py` | 7 | default file loads, overrides, unknown keys, wrong types, invalid ranges, broken YAML |
| `test_association.py` | 7 | exit from parked car, walk-past ignored, 3 people from one car keep separate ids, ambiguous between two cars, no exits at start-up / after tracker reset, occlusion re-appearance penalty, door-zone geometry |
| `test_risk_engine.py` | 11 | level thresholds, identity factors (authorised 0, associated 0.3, unrecognised 1), half-life decay, cooldown and duplicates, escalation-only alerts, re-alert after decay, cautious wording, informational events score 0, score cap |
| `test_storage.py` | 6 | event round trip, session end / delete, association review never creates authorisation, registry and authorisation, 4-thread concurrent writes, unwritable DB location |
| `test_video_pipeline.py` | 17 | missing / folder / corrupt video errors, metadata, complete-mode reader keeps every frame in order, drop-oldest queue and end marker, blocking put stops on quit, frame accounting, id namespaces and reset on gaps, duplicate removal, evidence path safety (incl. non-ASCII folder), clip with pre-event buffer, retention only deletes session folders, **full pipeline** in complete mode (60/60 frames, annotated video reopened, association stored), responsive mode counts skips, `--max-frames/--start-frame`, **synthetic theft scene → LOW/MEDIUM/HIGH alerts with real snapshot and clip files**, **keyboard pause/snapshot/overlay/quit**, detector crash → clean shutdown with balanced counts |
| `test_zones_and_reid.py` | 15 | point-in-polygon, zone file round trip and validation, park → move → leave zone, re-linking a car under a new track id, ReID reconnect, never merging people seen together, candidate status when unsure, colour names, dominant colour |
| `test_model_smoke.py` | 7 | YOLO class mapping (all 6 classes present), every detection has a mask (≥ 3 people on `bus.jpg`), FP16 check runs, ByteTrack and BoT-SORT give stable ids, SegFormer returns garment segmentation, fallback is labelled approximate |
| `test_evaluation.py` | 8 | Wilson intervals, label validation, exit outcomes (correct / wrong vehicle / ambiguous / missed / false link), event matching needs time **and** place, incident recall + latency + false alerts per hour, pooling, tolerance; **end-to-end: synthetic theft video → pipeline → SQLite → evaluator** |
| `test_behavior.py` | 7 | walking past ≠ proximity, lingering ≥ 1.5 s = proximity, proximity held while exit association pending, apparent entry withdrawn on re-appearance / confirmed after 3 s / confirmed at once by vehicle movement, passer-by from elsewhere does not withdraw it |
| `test_scenarios.py` | 9 | the 8 scripted scenarios meet their pre-written expectations; unsure-owner scenario ranks below theft |
| `test_camera.py` | 21 | with a fake `VideoCapture` (no webcam needed): open reports a live feed and requests the configured resolution, camera index passed through, unavailable camera → actionable error after trying DirectShow / Media Foundation / any, busy camera that opens but sends nothing → timeout error and every attempt released, black start-up frames skipped, backend fallback, idempotent release, camera settings validated; reader keeps the newest frames and counts drops, clock-based timestamps, disconnect after N failed reads, no reading while paused; **full live pipeline on the synthetic theft scene → MEDIUM+ alert with snapshot, no clip, no recording, cautious wording, camera released**, clips/recording opt-in, camera disconnect → `Camera failed` + release, analysis crash → release, setup failure → release, live forces responsive; default launch opens camera 0 (`--camera 3` → index 3) and returns exit code 5 when unavailable, `--video` still selects a file, `--resolution` parsing |
| (additions to existing files) | 7 | 4 movement-confirmation tests (occlusion shrink, brief shift, 1 s confirmation, vanish while starting to move); 3 identity tests (possible-match factor, worst entrant decides, pipeline resolver with a real gallery) |

The 2-second pause at the car door was added to the synthetic theft video when lingering
became a requirement for proximity: the test still asserts a proximity event, so it was the
scene that changed, not the assertion.

### Bugs the tests and runs found (and fixed)

* Ultralytics 8.4 removed `frame_rate` from the tracker constructors → now detected and
  `track_buffer` scaled manually.
* Ultralytics 8.4 deprecated `half=` → `quantize=16` used when available.
* `_verify_half` ran before the detector attribute existed (crashed on first real run).
* First real frame took ~1.8 s because warm-up ran on the main thread with a square image,
  causing 15–16 skipped frames at the start of every responsive run. Warm-up now runs on
  the worker thread with the real frame shape: 0 skipped frames on the same clip.
* HUD covered labels of people near the top edge; label placement now avoids it.

## Live start-up fixes (2026-10-25)

Problem reported: "the project still shows the old prerecorded-video results". Findings from the
session database: the live code was present and the default, but the user's runs were started in
**video mode** (`app.source = video`, the sample clip, consistent with the VS Code "file picker"
launch entry). Contributing causes found: `python -m app.main` fails outside the project folder
("No module named app"); plain `python` on the laptop is system Python 3.14 with a CPU-only PyTorch;
`.vscode/settings.json` pointed at a non-existent `.venv`; the window only appeared after 10–20 s of
model loading; the README's screenshots were all from video mode.

Changes: the window now opens immediately and shows the live camera while the models load (Q /
close works during start-up); `run_webcam.py` launches live mode from any folder and refuses a
Python without the packages; `main.py` prints the interpreter and warns when it falls back to CPU;
VS Code launch entries use the project environment and are labelled LIVE WEBCAM / PRERECORDED VIDEO.

New tests (in `test_camera.py`, with a fake camera and a scripted window): start-up frames are the
camera picture and come before analysed frames; quitting during start-up releases the camera and
analyses nothing; the launcher passes options through, refuses `--video`, and explains a missing
package; the CPU fallback prints a warning and the interpreter path. The keyboard-controls test now
passes its scripted window in through the new `window=` argument (same assertions).

Real webcam on 2026-10-25 (power state not recorded; no camera images viewed or saved):

| Check | Result |
|---|---|
| Windows camera permission | camera access Allow (system and user), "desktop apps" Allow; one device, "Integrated Camera", status OK |
| Launch from `C:\Users\Dharshan` with `...\vtd\Scripts\python.exe E:\vehicle-theft-detection\run_webcam.py` | worked (this folder used to fail with "No module named app") |
| Window appears | 6.1 s after launch, 0.1 s after the camera opened, showing the live picture; 247 start-up frames shown during loading/warm-up |
| Analysis | run 1: 400 read / 390 analysed / 10 dropped, 23.6 FPS, detector 14.8 ms. Run 2: 300 / 300 / 0, camera delivered ≈14.6 FPS (likely low light), detector 16.1 ms |
| People | tracked (1 ReID identity in run 2) |
| Vehicles / alerts | none in view, so no vehicle events and `risk: none`: live alerts remain verified only with the simulated camera |
| Recording | none: session folder holds `session.json` only |
| Exit | camera released, exit code 0 |

## Live webcam manual test (2026-10-24)

The built-in webcam of the development laptop was available and used. No camera images were
viewed or saved during these checks; results come from the session statistics. The laptop was
**on battery** (55 %, Balanced plan, GPU at 600 of 3,090 MHz), so speeds are lower than plugged in.

| Check | Result |
|---|---|
| Default launch opens the webcam | `python -m app.main` opened `camera:0` via DirectShow, 1280×720, YUY2, driver 30 FPS; no `--video` needed |
| Headless run, 300 frames | read 300, analysed 231, dropped 69 (newest-frame policy), balanced; 17.4 FPS average analysed (≈20 FPS after warm-up); detector 31 ms, whole frame 52 ms |
| People detected and tracked live | 2 person tracks got clothing results, 1 ReID identity decision |
| Window run, 450 frames | camera delivered 450 frames in 15.5 s (≈29 FPS); analysed 364 at 23.5 FPS average, all 364 shown, 0 display drops; whole frame 41 ms |
| No recording | session folders contain only `session.json` and empty `snapshots/` and `clips/`; no `annotated.mp4` |
| Clean exit | camera released at the end of both runs (`released: True`), exit code 0 |
| Risk alerts live | **not exercised**: no vehicle was in view, so no vehicle events and risk stayed at none. Alerts in live mode are covered by the synthetic-camera test only. |
| Pause / snapshot / overlay keys in the live window | not pressed during the manual run; the same key handling is covered by `test_keyboard_controls_and_quit` (video path) |

Regression checks of the video tools after this change: `--video` complete run 647/647 frames
balanced, `scripts/benchmark_video.py` ran, `scripts/run_scenarios.py` unchanged (1/6 benign
with any alert, 0/6 ≥ MEDIUM, 2/2 suspicious detected), evaluation tests pass.

## Real-video smoke test

Footage: `person-bicycle-car-detection.mp4` from Intel's sample-videos repo (CC BY 4.0),
768×432, 12 FPS, 647 frames, 53.9 s, an empty car park with a pedestrian, two cars passing
and a cyclist.

```powershell
python -m app.main --video data\samples\person-bicycle-car-detection.mp4 --mode complete --no-display
```

Verified on that run:

| Check | Result |
|---|---|
| Model class mapping | person 0, bicycle 1, car 2, motorcycle 3, bus 5, truck 7; nothing missing |
| Masks produced | yes, every detection in the inspected frames has a mask (see docs/images) |
| Tracking ids | `V001`, `P001`, `P003`… generated (ID stability over the whole clip was not checked frame by frame) |
| Frame accounting | read 647, processed 647, skipped 0, balanced |
| Annotated video | 647 frames written and reopened with OpenCV |
| FP16 | agreed with FP32 on the check image, used |
| Clothing | SegFormer ran; e.g. pedestrian "top: blue, bottom: blue" |
| ReID | ResNet-50 fallback ran; 1 identity decision (new) |
| Events / alerts | none: nobody exits or enters a car in this clip, so nothing to associate |
| Live window | separate 150-frame run with the window: 150 read / processed / displayed, 0 dropped |

## Acceptance targets

| Target | Status |
|---|---|
| ≥ 90 % correct associations on a labelled staged set | **Not evaluated.** No staged footage of people getting in/out of cars was available. The logic is unit-tested on synthetic tracks only. |
| ≥ 90 % recall on high-priority staged events, false alerts per hour reported | **Not evaluated** for the same reason. The synthetic scene raises LOW → MEDIUM → HIGH as designed. |
| 20 displayed FPS at 1080p in responsive mode | **Met headless**: 26.9 FPS processed on a 1080p copy re-timed to 30 FPS (see PERFORMANCE.md). With the window open it was only measured at 768×432 (12/12 FPS, source-limited). |
| Complete mode accounts for every frame | **Met**: 647/647 at 768×432 and 1080p, `frames_balanced: true`, also asserted in tests. |
| 30-minute stress test without crashes or memory growth | **Met on the conditions below.** 30.5 min of video (the 54 s sample clip looped to 21,960 frames, 768×432 @ 12 FPS), complete mode, headless, `--no-save-video`, RTX 5050: 21,960 read / 21,960 processed / 0 skipped, balanced, no errors, 384.7 s wall time (57.1 FPS, detector 15.2 ms mean). Process memory sampled every 15 s (27 samples): working set 2,315 MB after start-up → 2,282 MB at the end (max 2,318 MB); private bytes 4,387 → 4,363 MB (max 4,390 MB). No growth. Caveats: a looped clip with 0–3 objects at a time is a light load, no alerts were raised, and the annotated-video writer was off. |

### How to run the missing evaluations

1. Record or collect footage of your car park with staged events: people exiting cars
   (1–4 per car), people walking past, people standing between cars, someone returning to a
   car, a car leaving with/without its driver seen. Note the frame of each event.
2. Write the ground truth as CSV: `video, person_track_or_time, true_vehicle_or_none, event_type, frame`.
3. Run each video in complete mode, then compare with `python -m app.manage associations --session <id>`
   and `events --session <id>`.
4. Report: number of persons / events, accuracy, recall, false alerts per hour of video,
   and the cases that failed.
5. For the stress test, run a ≥ 30-minute video in complete mode with
   `python scripts/benchmark_video.py --video long.mp4 --mode complete` and watch memory in
   Task Manager; the input queue depth is recorded in the session summary.
