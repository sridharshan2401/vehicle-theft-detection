# User guide

Assumes setup from the README is done and the virtual environment is active
(`.\.venv\Scripts\Activate.ps1`).

## 1. Watch the live camera (default)

```powershell
python -m app.main                                   # from the project folder, environment active
C:\path\to\venv\Scripts\python.exe run_webcam.py     # from anywhere, no activation needed
```

The console prints which Python is running, the GPU it found, opens camera 0 and shows what the
camera delivers (e.g. `camera:0: 1280x720, 30.0 FPS, live feed, format YUY2 via dshow`). The window
opens straight away with the raw camera picture and "Loading AI models...", and switches to the
analysed view once the models are loaded and warmed up (10–20 s). No video file is involved.
If it prints `WARNING: no usable CUDA GPU`, you started it with the wrong Python.

* Another camera: `python -m app.main --camera 1` (or set `camera.index` in `configs/settings.yaml`).
  Indexes follow the order Windows lists the devices; 0 is normally the built-in webcam.
* Resolution: `--resolution 640x480` (lighter) or `1280x720` (default request).
* Quit with `Q` or `Esc` (or close the window). `Space` pauses: the camera is not read while
  paused and the last frame stays on screen. `S` saves a snapshot, `R` cycles overlays.
* The status panel shows `camera:0 [connected]`. `waiting for frames` means nothing arrived
  for a second; `DISCONNECTED` means the camera stopped and the program is shutting down.
* Live mode always analyses the newest frame: if analysis is slower than the camera, older
  frames are dropped and counted as "skipped". That keeps the picture current.
* Nothing is recorded except alert snapshots and the database. Add `--alert-clips` for short
  clips around MEDIUM+ alerts or `--record` for the whole annotated feed.

What the risk line means: `risk: MEDIUM on V001 (score 41/100)` with `why: ...` naming the
strongest recent observation. It is a behaviour score with reasons, not a finding of theft, and
the panel always states that stolen-vehicle verification is not available.

## 2. Analyse a video file

```powershell
python -m app.main --video "C:\videos\carpark_evening.mp4"
```

Or `python -m app.main --pick-video` and pick the file in the dialog. The console prints the video
details, the GPU it found and which optional models are active, then the window opens.

Pick the mode for the job:

* `--mode responsive` (default): plays at roughly the video's speed. If the GPU falls
  behind, frames are skipped, counted on the HUD ("skipped N") and in the summary.
* `--mode complete`: analyses every frame in order. Slower than real time on big videos,
  but nothing is skipped. Use this when you need evidence. Progress and an estimated time
  remaining are logged every 5 seconds when running with `--no-display`.

### Reading the window

* **Top left (HUD)**: mode, device, video time and frame, source / processing / display
  FPS, detector and secondary-model latency, processed and skipped frames, tracker resets,
  which ReID and clothing backends are active, and the highest current risk.
* **Top right**: banner saying which mode you are in.
* **Vehicles**: box coloured by risk level (gray info, yellow low, orange medium, red
  high, magenta critical), mask, `V001 car 0.93`, parking state and risk score, and
  `exited: P001,P002` once people are linked to it.
* **People**: mask, `P003 0.88`, then identity status:
  `= P001 (0.91)` reconnected to an earlier person, `?P001 (0.62)` possible match still being
  checked, `id:?` not decided yet. Second line: `exited V001 (0.74)`, `exit: observing` or
  `exit: ambiguous`. Third line: clothing, with a `~` prefix when it is the approximate fallback.
* **Bottom left**: the last six events and alerts.

Keys: `Space` pause, `S` snapshot, `R` change overlays, `Q`/`Esc` quit.

## 3. Draw zones (optional but recommended)

```powershell
python -m app.zone_editor --camera 0
python -m app.zone_editor --video "C:\videos\carpark_evening.mp4" --frame 300
```

With `--camera` one frame is captured and the camera is released at once; mount the camera
first, because zones only fit the view they were drawn on.

Click the corners of a parking area, press `Enter`, then type the kind (`p` parking,
`r` restricted, `x` exit, `z` proximity) and a name in the terminal. Repeat, then press
`s` to save to `configs/zones.json` (the default `zones.file`). `u` undoes a point, `d`
deletes the last zone.

Zones are stored with coordinates from 0 to 1, so they keep working if the same camera
view is exported at another resolution. Make one zone file per camera view and pass it with
`--zones`.

Without zones, the whole frame is one parking area, and departures are detected only when
a moving car's track is lost.

## 4. Register vehicles and authorised people (optional)

```powershell
python -m app.manage add-vehicle --name "Blue hatchback" --plate KA01AB1234 --zone A1
python -m app.manage add-person --label "Owner" --images owner1.jpg owner2.jpg
python -m app.manage authorize --vehicle 1 --person 1
python -m app.manage registry
```

* `--zone` must match a parking zone name from your zone file; that is how the app knows
  which registered vehicle is parked there (there is no plate reading).
* Person images should be cropped full-body photos. You can also take a profile from a
  past session: `add-person --label "Owner" --session <session-id> --profile P002`.
* Authorisation matching only works with a person-specific ReID model (OSNet); with the
  generic fallback it stays off and the HUD/session summary says so.

## 5. Review results

```powershell
python -m app.manage sessions
python -m app.manage events --session 20261023-010637-50eac4
python -m app.manage associations --session 20261023-010637-50eac4
python -m app.manage review-association 12 confirm --note "checked the clip"
```

Or open the session folder: `evidence\<session-id>\` has `session.json` (all counts and
settings), `events.jsonl`, `snapshots\`, `clips\` and `annotated.mp4`. The SQLite file
`evidence\vtd.sqlite3` can be opened with any SQLite browser.

### Measure accuracy against your own labels

1. Process the video with `--mode complete` and note the session id.
2. Copy `configs/labels_template.json` next to your video and fill it in while watching it:
   one point per parked car you care about (`[x, y]` as fractions of width and height), and
   the times in seconds of exits, entries, movements, departures and incident windows.
3. Run `python scripts/evaluate_session.py --session <id> --labels my_labels.json`.

The report lists each labelled exit as correct / wrong vehicle / ambiguous / missed, recall and
precision for entries, movements and departures, incident recall, and false alerts per hour
(at LOW, MEDIUM and HIGH), each with a 95 % confidence interval. The JSON report is saved in
`evidence\evaluations\`. For several videos, list them in a manifest
(`{"items": [{"session": "...", "labels": "..."}]}`) and pass `--manifest`. Change thresholds
on one set of videos and judge them on another; see docs/VALIDATION.md.

## 6. Tune settings

Edit `configs/settings.yaml`. Common changes:

| Want | Setting |
|---|---|
| more speed | `detector.weights: models/yolo11n-seg.pt`, `detector.imgsz: 512`, `clothing.enabled: false` |
| fewer weak alerts | raise the `risk.levels` thresholds or `risk.cooldown_seconds` |
| alerts only for serious cases | `risk.alert_min_level: MEDIUM`, `evidence.min_level: HIGH` |
| longer clips | `evidence.pre_event_seconds`, `evidence.post_event_seconds` |
| stricter association | `association.confirm_threshold: 0.7` |
| auto-delete old evidence | `evidence.retention_days: 30` |

The app refuses to start if a setting is misspelled or out of range and tells you which one.

## 7. Benchmark your machine

```powershell
python scripts/benchmark_video.py --video "C:\videos\clip.mp4" --mode both
```

Writes `evidence\benchmarks\benchmark_<time>.json` with FPS, latencies, skipped frames and
GPU memory.
