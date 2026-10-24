# Project overview

## The problem

Car-park CCTV is mostly recorded and rarely watched. When a car goes missing,
someone has to scrub through hours of footage to find the few seconds that
matter. This project is a first pass at doing that automatically for a
prerecorded video: find the vehicles and people, follow them, notice the
patterns that usually come before a theft, and pull out the evidence.

The goal is to **save a reviewer time**, not to decide who is guilty. Every
output is phrased as an observation with a confidence ("an unrecognised
person appeared to get into V001"), never as a conclusion.

## Objectives

1. Process a video file on a Windows laptop GPU and show the analysis live.
2. Detect people and road vehicles with **instance masks**, not just boxes.
3. Give vehicles and people stable, separate tracking IDs.
4. Link people to the vehicle they got out of, without linking passers-by.
5. Re-identify people who leave and come back, using appearance and clothing.
6. Know when a parked vehicle moves or leaves its parking zone.
7. Turn all of that into explainable risk levels and alerts.
8. Save evidence (snapshot, clip, database row) for every alert worth checking.
9. Be honest about what is measured, what is guessed and what is missing.

## Requirements

### Functional

| ID | Requirement | Where |
|---|---|---|
| F1 | Video by `--video` path or file picker; metadata report | `app/main.py`, `app/cli.py`, `app/video_reader.py` |
| F2 | Person / vehicle detection with instance masks, class mapping checked at start-up | `app/detector.py`, `app/segmenter.py` |
| F3 | Tracking with separate `V###` / `P###` namespaces, ByteTrack default, BoT-SORT optional | `app/tracker.py` |
| F4 | Vehicle-person association with confidence and evidence | `app/association.py` |
| F5 | Person re-identification with explicit uncertainty | `app/person_reid.py`, `app/secondary.py` |
| F6 | Clothing colours / garments, with a clearly labelled fallback | `app/clothing_analyzer.py` |
| F7 | Polygon zones, interactive editor, vehicle parking state | `app/parking_zones.py`, `app/zone_editor.py` |
| F8 | Rule-based risk with decay, cooldown and explanations | `app/behavior.py`, `app/risk_engine.py` |
| F9 | Registry of vehicles and authorised people, review of associations | `app/storage.py`, `app/manage.py` |
| F10 | Evidence: snapshots, pre/post clips, annotated video, JSON + SQLite | `app/evidence.py`, `app/pipeline.py` |
| F11 | Live window with HUD, events, keyboard controls | `app/display.py`, `app/pipeline.py` |

### Non-functional

* Runs on a single consumer GPU; one heavy model on the main path, secondary models rate-limited.
* Bounded memory: every queue has a fixed size, clip buffers are time-limited.
* Every source frame is accounted for (processed or skipped, never silently lost).
* Clean shutdown on quit, end of video or error.
* Optional features degrade gracefully and say so on screen and in the logs.
* Local only: no network access while analysing.

## Status

Legend: **Tested** = implemented and covered by automated tests and/or a real
run; **Implemented, not tested** = code exists but has not been exercised end
to end; **Needs model/data** = works only after extra setup; **Planned** = not built.

| Feature | Status | Evidence |
|---|---|---|
| Live webcam input (default launch, camera index, failure messages, release) | Tested with a fake camera; manually tested on the built-in webcam (detection + tracking of people, no recording, clean release) | `test_camera.py`; TESTING.md "Live webcam manual test" |
| Live risk alerts | Tested on a synthetic camera feed only (no vehicle was in front of the real webcam) | `test_live_pipeline_raises_explainable_alerts_without_recording` |
| Stolen-vehicle verification (plate reading + authorised database) | **Not implemented**; the window states it | — |
| Video probing, missing/invalid file errors | Tested | `test_video_pipeline.py` |
| File picker with `--pick-video` | Implemented, not tested | needs a person to click |
| YOLO11-seg detection + instance masks on GPU | Tested | smoke tests + 647-frame real run |
| FP16 with FP32 cross-check | Tested | `verify_half_precision`, smoke test |
| ByteTrack tracking, separate namespaces, reset on large gaps | Tested | unit tests + real run |
| BoT-SORT | Tested (ID assignment only) | `test_real_tracker_assigns_ids[botsort]` |
| Vehicle-person association | Tested on synthetic tracks | `test_association.py`, synthetic pipeline test |
| Association accuracy on real footage | **Not measured** | no staged footage available |
| Evaluation harness (labels → recall / precision / false alerts per hour with CIs) | Tested on hand-built data and a synthetic end-to-end run | `test_evaluation.py`; see VALIDATION.md |
| False-positive rules (lingering, entry confirmation, movement confirmation, association hold) | Tested on scripted scenarios | `test_behavior.py`, `test_scenarios.py`; real-footage effect not measured |
| Person ReID gallery logic | Tested on synthetic embeddings | `test_zones_and_reid.py` |
| ReID with OSNet (torchreid) | Needs model/data, not tested | manual install, see MODEL_REQUIREMENTS |
| ReID with generic ResNet-50 | Tested (runs, embeddings valid) | real run; accuracy not measured |
| Clothing garment segmentation (SegFormer) | Tested (runs, labels verified) | smoke test, real run |
| Clothing fallback (mask bands) | Tested | `test_clothing_fallback_is_labelled_as_approximate` |
| Zones: file, validation, park/move/leave | Tested | `test_zones_and_reid.py` |
| Zone editor GUI | Implemented, not tested | needs a person to click |
| Risk engine: levels, decay, cooldown, wording | Tested | `test_risk_engine.py` |
| Alerts with snapshot + clip + DB row | Tested on synthetic scene | `test_suspicious_scene_produces_alert_with_evidence` |
| Registry and manual review | Tested (storage layer) | `test_storage.py` |
| `app.manage add-person --images` | Implemented, not tested | |
| Authorisation matching at runtime | Implemented, not tested | needs OSNet + registered people |
| Live window, keyboard controls | Tested | real 150-frame window run + scripted key test |
| Evidence retention purge | Tested | `test_retention_only_touches_session_folders` |
| Face recognition | Planned (deliberately off) | `privacy.face_processing: false` |
| Number-plate reading | Planned | |

## Future work

* Record and hand-label a staged car-park test set (exits, entries, walk-pasts,
  departures) and run `scripts/evaluate_session.py` on it (protocol in VALIDATION.md).
* Install and evaluate OSNet ReID; tune thresholds on real returning-person cases.
* Use the vehicle's own mask for door positions instead of box geometry.
* Ground-plane calibration (homography) so distances are in metres.
* Number-plate recognition to link vehicles across sessions without zones; for India this
  needs a plate detector/OCR trained on Indian plates (including two-wheeler and two-line
  plates). Checking a plate against a stolen-vehicle list would additionally need lawful
  access to an authorised data source; nothing like that is part of this project.
* Validate on Indian street and parking footage (two-wheelers, dense scenes, night, rain)
  and fine-tune if the measured recall is poor.
* Multi-camera support with camera-to-camera ReID.
* A small review UI for confirming/rejecting associations and alerts.
* TensorRT export of the detector, after checking mask output matches PyTorch.
