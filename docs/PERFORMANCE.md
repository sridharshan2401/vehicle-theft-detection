# Performance

## Design

* **One heavy model on the main path.** YOLO11-S-seg gives boxes and masks in one pass.
  ReID and clothing run on a separate thread, only on good person crops, at most 4 per
  frame, every 10 (ReID) / 30 (clothing) processed frames per person.
* **Bounded everything.** Input queue 6 frames, display queue 3, secondary queue 16 crops,
  pre-event buffer 3 s at ≤ 960 px wide. Nothing grows with video length except the
  database and the evidence files.
* **Two modes.**
  * *Responsive*: the reader releases frames at the source rate; if the worker is behind,
    the oldest waiting frame is dropped and counted. The display always shows the newest
    processed frame.
  * *Complete*: the reader blocks when the queue is full. Every frame is processed in order.
    If analysis is slower than the video, the run just takes longer.
* **Adaptive input size** (responsive only): if whole-frame processing time stays above
  the `pipeline.target_fps` budget (50 ms at 20 FPS) for 15 frames, `imgsz` drops by 64
  (not below `min_imgsz`, 416); it climbs back after a long run of fast frames.
* **FP16** on CUDA, only after a start-up check that FP16 and FP32 agree on a test image.
* **Warm-up on the right thread.** cuDNN state is per thread and per input shape. Warming
  up on the main thread with a square dummy image still left the first real frame at
  ~1.8 s (measured), which caused 15–16 dropped frames at the start of every responsive run.
  The detector now warms up on the worker thread with the real frame shape (and every
  adaptive size) before the video clock starts; the secondary models warm up on their thread.
* **Honest timing.** Detector and secondary-model timings call `torch.cuda.synchronize()`
  before reading the clock.
* **Cheap rendering.** Masks are blended only inside the bounding region of all masks,
  and the frame is rendered once and reused for the window, the annotated video and evidence.

## Measured results

Machine: RTX 5050 Laptop GPU 8 GB (driver 595.71), Windows 11, Python 3.12.10,
PyTorch 2.9.1+cu128, Ultralytics 8.4.184, YOLO11-S-seg @ 640, FP16 (verified),
ReID = ResNet-50 fallback, clothing = SegFormer-B2. Headless (`--no-display`), measured
with `scripts/benchmark_video.py` on 2026-10-23. Single runs, not averaged.

Clip: Intel `person-bicycle-car-detection.mp4`, 647 frames, light scene (0–3 objects at a time).
The 1080p rows use a copy upscaled to 1920×1080 and re-timed to 30 FPS (same frames,
played faster). That is a throughput test, not real 1080p/30 footage.

| Video | Mode | Annotated video | Processed / read | Skipped | Processed FPS | Detector ms (mean) | Whole frame ms (mean) |
|---|---|---|---|---|---|---|---|
| 768×432 @ 12 | complete | on | 647 / 647 | 0 | **63.7** | 12.5 | 15.6 |
| 768×432 @ 12 | responsive | on | 647 / 647 | 0 | 12.0 (= source) | 16.9 | 20.1 |
| 1080p @ 30 | complete | on | 647 / 647 | 0 | **26.7** | 16.4 | 36.8 |
| 1080p @ 30 | responsive | on | 586 / 647 | 61 | **26.9** | 15.1 | 36.6 |
| 1080p @ 30 | complete | off | 647 / 647 | 0 | **34.7** | 19.5 | 28.4 |
| 1080p @ 30 | responsive | off | 635 / 647 | 12 | **29.4** | 18.0 | 27.5 |

Other numbers from the same runs:

* ReID embedding (ResNet-50): 10–14 ms per crop. SegFormer clothing: 41–48 ms per crop
  (one run averaged 207 ms because of a few slow calls; it runs on the secondary thread and did not delay frames).
* Secondary jobs dropped: 0 in all runs (few people in this clip).
* GPU memory: peak allocated 0.78 GB (complete) to 1.06 GB (responsive, extra warm-up sizes);
  reserved ≤ 1.12 GB.
* With the window open (768×432, responsive, 150 frames): 150 displayed, 0 dropped, 12 FPS (source rate).

What this means:

* At 1080p the detector is only ~16 ms of a ~37 ms frame. Writing the full 1080p annotated
  MP4 costs about 9 ms per frame; turning it off (`--no-save-video`) raised complete-mode
  throughput from 26.7 to 34.7 FPS. The rest is decoding, rendering and clip buffering on the CPU.
* The 20 FPS @ 1080p responsive target is met headless on this GPU. It was not measured with
  the window open at 1080p; expect a few ms more per frame for `imshow`.
* The RTX 3050 laptop GPU the project was specified for was not available. It has fewer
  CUDA cores and 4 GB VRAM; memory use above fits, but expect lower FPS. Run the benchmark
  script on it rather than trusting these numbers.

## Tuning for slower GPUs

In rough order of impact:

1. `--no-save-video` (if you do not need the full annotated copy; evidence clips are still saved)
2. `--weights models\yolo11n-seg.pt`
3. `--no-clothing` (frees GPU time on the secondary thread)
4. `--imgsz 512` (smaller objects get missed sooner)
5. Lower `pipeline.target_fps` so adaptive resolution does not kick in too early

## Live camera

The live reader never waits for the analysis: the camera is read continuously, at most
`camera.queue_size` (2) frames wait, and older ones are dropped and counted as skipped, so
what you see is never more than a couple of frames behind. Only one copy of each model is
loaded (the detector on the analysis thread, ReID and clothing on the secondary thread).

Measured on 2026-10-24 with the built-in webcam (1280×720 YUY2 over DirectShow, driver 30 FPS),
YOLO11-S-seg @ 640 FP16, ResNet-50 ReID + SegFormer clothing, **laptop on battery**:

| Run | Camera delivered | Analysed | Dropped | Detector | Whole frame |
|---|---|---|---|---|---|
| headless, 300 frames | ≈22.6 FPS (300 in 13.3 s, incl. start) | 17.4 FPS avg (≈20 after warm-up) | 69 | 31 ms | 52 ms |
| window, 450 frames | ≈29 FPS (450 in 15.5 s) | 23.5 FPS avg, all shown | 86 | 30 ms | 41 ms |

Same-day video-file runs were also slow (29 FPS on the 768×432 sample vs 64 FPS earlier on a
cool machine), so expect noticeably better live numbers when plugged in. Not measured: a
plugged-in live run and long live sessions. Start-up (camera open + model load + warm-up) took
about 14 s.

If the live view lags: plug in, use `--weights models\yolo11n-seg.pt`, `--no-clothing`, or
`--resolution 960x540` (smaller frames cut decoding and drawing time; the detector input stays 640).

## Long run and run-to-run variation

* **30-minute stress test** (new behaviour rules included): 21,960 frames at 57.1 FPS average,
  detector 15.2 ms, no skipped frames, no memory growth. Details in TESTING.md.
* **Variation on this laptop is large.** Right after the stress test, the same 768×432 clip ran
  at only 36–39 FPS (detector 21–23 ms) instead of the 63.7 FPS measured earlier, for both
  code versions. An A/B run back to back in that state gave 39.2 FPS for the code before the
  false-positive changes and 36.4 FPS after. The detector code did not change, so most of the
  drop is the machine's state (power/thermal). Whether the new rules cost the remaining few
  percent was not settled. Benchmark on a cool, plugged-in machine and repeat runs before
  drawing conclusions.

## Not done

* TensorRT / ONNX export: possible with Ultralytics, but mask outputs would need checking
  against PyTorch before use; not attempted.
