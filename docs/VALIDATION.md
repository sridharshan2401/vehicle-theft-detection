# Validation: audit, criteria and what has (not) been measured

Status on 2026-10-23. **No real-world accuracy figures exist for this project yet.**
There is no labelled footage of people getting into or out of cars, so every accuracy
statement below is either a target or a result on synthetic, scripted data, and says so.

## 1. Audit against the master specification

What the original spec asked for, what exists, and what is still missing, ordered by how much
each gap matters for real theft detection.

| # | Area | State before this round | Gap / risk | Priority | Now |
|---|---|---|---|---|---|
| 1 | Accuracy measurement | Only a CSV idea in TESTING.md; events stored without positions | Impossible to measure association accuracy, alert recall or false alerts per hour on any footage | **Critical** | **Fixed**: label format, evaluator with Wilson intervals, locations stored on every event (§3) |
| 2 | Labelled staged footage | none | Every accuracy target in the spec is unevaluated | **Critical** | **Still missing**: needs footage you record (§5) |
| 3 | False positives: passers-by | any foot point inside a car's zone = proximity event | 3 people walking past one parked car → LOW alert (synthetic) | High | **Fixed**: lingering required |
| 4 | False positives: occlusion | vanishing behind a car = instant "apparent entry"; re-appearing = weak "exit" | person walking behind a car → MEDIUM alert and a false exit link (synthetic) | High | **Fixed**: entry waits for confirmation; re-appearance cancels it and blocks weak exit links |
| 5 | False positives: box jitter | one frame with a shifted/shrunk box = "vehicle moved" | someone standing in front of a parked car → false movement | High | **Fixed**: movement must persist 1 s with a plausible box size |
| 6 | Association race | proximity scored while the exit decision (2.5 s) was still pending | driver getting out → LOW alert against their own car (synthetic) | High | **Fixed**: proximity held until the decision |
| 7 | Returning owner | without a confident ReID match the owner is "unrecognised" | owner coming back and driving off → HIGH alert (synthetic) | High | **Partly**: unsure ReID match to the driver now scores 0.6× (→ LOW in the scenario). Real fix needs a person-ReID model (OSNet) and tuning on real returns |
| 8 | ReID quality | generic ImageNet ResNet-50 only | reconnections rare and untested on people | High | **Still missing**: OSNet needs a manual install; no ReID accuracy measured |
| 9 | Tracker ID switches on real video | not measured | an ID switch next to a car looks like a new person "born" there → possible false exit | Medium | Partly mitigated by the re-appearance rule; unmeasured |
| 10 | Geometry | image-space boxes, approximate door bands, no calibration | distances depend on camera angle; side vs top-down views behave differently | Medium | Unchanged (documented) |
| 11 | 30-minute stress test | not run | memory growth / crashes unknown | Medium | **Run** (see TESTING.md) |
| 12 | Number plates | none | registered vehicles tied to zones, not plates | Low (spec did not require) | Unchanged |
| 13 | RTX 3050 target hardware | not available | FPS on the specified GPU unknown | Low | Unchanged |

Everything else in the spec (segmentation masks, separate ID namespaces, two processing modes,
evidence recording, registry separate from observations, privacy defaults, docs) was already
implemented and tested in the previous round; see PROJECT_OVERVIEW.md.

## 2. Measurable validation criteria

To be computed with `scripts/evaluate_session.py` on **held-out** labelled footage processed in
`--mode complete`. A system item matches a label when it is within the time tolerance (3 s for
exits/entries/movements, 6 s for departures) and the labelled vehicle point lies inside the
system's vehicle box. Every proportion is reported with a 95 % Wilson interval.

| Metric | Definition | Target | Minimum sample |
|---|---|---|---|
| Exit recall | labelled exits linked to the right vehicle ÷ labelled exits | ≥ 0.90 (spec) | 50 exits |
| Vehicle accuracy when linked | right vehicle ÷ (right + wrong vehicle) | ≥ 0.95 | 50 exits |
| Link precision | system exit links that match a labelled exit ÷ all system links | ≥ 0.90 | — |
| False exit links per hour | unmatched system links ÷ hours of footage | ≤ 2 | 5 h |
| Apparent-entry recall / precision | per labelled entry | ≥ 0.80 / ≥ 0.70 (inherently ambiguous from one camera) | 30 entries |
| Vehicle moved / departed recall / precision | per labelled movement / departure | ≥ 0.95 / ≥ 0.90 | 30 each |
| Incident recall | labelled staged incidents with an alert ≥ MEDIUM on the right vehicle within [start, end + 10 s] | ≥ 0.90 (spec) | 20 incidents |
| Alert latency | first qualifying alert − incident start, median | report (aim ≤ 10 s after the car moves) | — |
| False alerts per hour ≥ MEDIUM | alerts outside every incident window ÷ hours | ≤ 0.5 | 5 h benign |
| False alerts per hour ≥ LOW | same at LOW | report (aim ≤ 3) | 5 h benign |

Why the sample sizes matter: 18 detected out of 20 incidents is 0.90, but the 95 % interval is
0.70–0.97, so a 20-incident test cannot prove the 90 % target, only fail to rule it out. Zero
false alerts in 5 hours still leaves an upper bound of about 0.74 per hour. Report the interval,
not just the point value.

Rules for a fair evaluation:

* Tune thresholds on one set of videos and report on a **different** set.
* Include ordinary activity (people parking, walking past, loading boots, waiting) in the benign
  hours. Without it, false alerts per hour are meaningless.
* Report the camera position, lighting, resolution and frame rate with the numbers.
* Use complete mode; the evaluator warns if frames were skipped.

## 3. What was implemented this round

| Change | Files | Verified by |
|---|---|---|
| Label format, evaluator (exits, entries, movements, departures, incidents, false alerts/hour, Wilson CIs, pooling) | `app/evaluation.py`, `scripts/evaluate_session.py`, `configs/labels_template.json` | `tests/test_evaluation.py` (7 metric tests + 1 end-to-end pipeline→DB→evaluator test) |
| Pixel locations stored on events, alerts and associations; DB migration for old databases | `app/storage.py`, `app/pipeline.py`, `app/association.py` | end-to-end test; existing DB migrated in place |
| Proximity needs ≥ 1.5 s of lingering (< 0.35 box diagonals/s) | `app/behavior.py`, `behavior.proximity_min_seconds`, `linger_speed` | `tests/test_behavior.py` |
| Proximity held while the person's exit association is pending | `app/behavior.py`, `app/pipeline.py` | `tests/test_behavior.py`, scenario `driver_exits_and_leaves` |
| Vehicle movement must last 1 s and keep a plausible box size; a car that vanishes while starting to move still yields moved + departed | `app/parking_zones.py`, `zones.move_confirm_seconds`, `max_area_change` | 4 new tests in `tests/test_zones_and_reid.py` |
| Apparent entry waits 3 s; withdrawn if a new person track starts near the spot; confirmed at once by vehicle movement | `app/behavior.py`, `behavior.entry_confirm_seconds`, `reappear_radius` | `tests/test_behavior.py` |
| Re-appearance after occlusion can no longer produce a weak "possible exit" link | `app/association.py` | scenario `occlusion_behind_car` |
| `possible_associated` identity (0.6×) when ReID's unsure best match got out of this vehicle | `app/risk_engine.py`, `app/pipeline.py`, `risk.identity_factors` | `tests/test_risk_engine.py` (resolver test with a real gallery) |
| Synthetic scenario suite and runner | `tests/scenarios.py`, `tests/test_scenarios.py`, `scripts/run_scenarios.py` | itself; results below |

## 4. Synthetic scenario results (before vs after)

Eight scripted situations replayed through the real zone, association, behaviour and risk code
(no detector, no tracker: tracks are perfect). Expectations were written **before** the fixes.
Raw results: [`docs/results/baseline_before_changes.json`](results/baseline_before_changes.json) and
[`after_changes.json`](results/after_changes.json) (regenerate with `python scripts/run_scenarios.py --json <file>`).

| Scenario | Expected | Before | After |
|---|---|---|---|
| walk_past_x3 | no alert | LOW | none |
| occlusion_behind_car | no alert | LOW, MEDIUM (+ false entry and false exit link) | none |
| parked_car_box_jitter | no alert | none, but a false `vehicle_moved` | none, no false event |
| driver_exits_and_leaves | no alert | LOW | none |
| owner_returns_reid_match | max LOW | LOW | none |
| owner_returns_reid_unsure | max MEDIUM, below theft | LOW → HIGH | LOW |
| theft | ≥ MEDIUM | LOW → MEDIUM → HIGH | LOW → MEDIUM → HIGH |
| loitering | ≥ LOW | LOW | LOW |
| **benign with any alert** | | **5 / 6** | **1 / 6** |
| **benign with alert ≥ MEDIUM** | | **2 / 6** | **0 / 6** |
| **suspicious detected** | | **2 / 2** | **2 / 2** |

What this shows: the rules now behave as designed on idealised versions of the most common
false-positive situations, without losing the two suspicious cases. What it does **not** show:
anything about real footage, where detector misses, tracker ID switches, partial occlusion and
camera geometry dominate. Eight hand-written scenarios are also easy to over-fit, which is why
§2 insists on held-out real footage.

Trade-offs introduced (worth checking on real footage):

* A thief who walks straight to a door and gets in **without pausing** no longer gets a
  proximity event; the apparent entry and the movement still score.
* Apparent entries are reported up to ~4 s after the person vanishes (1 s to declare the track
  lost + 3 s confirmation) unless the car moves first.
* A real car movement is reported 1 s later than before.
* If a genuine entrant's spot is immediately re-entered by a *newly appearing* person, the
  entry is withdrawn (a false negative).

## 5. How to produce the missing real-world numbers

1. Get permission to film a car park (or use your own cars and friends) and record staged
   sessions from the camera position you will actually use: drivers getting out (1–4 people),
   people walking past and between cars, someone loading a boot, the owner returning and driving
   off, and staged "stranger gets in and drives off" incidents. Add a few hours of normal activity.
2. Process each video: `python -m app.main --video clip.mp4 --mode complete --no-display`.
3. Label each video by watching it (any player with a time display): copy
   `configs/labels_template.json`, mark a point on each relevant parked car, then list exits,
   entries, movements, departures and incident windows with times in seconds.
4. Evaluate: `python scripts/evaluate_session.py --session <id> --labels labels.json`, or
   list many in a manifest and use `--manifest` for pooled numbers with confidence intervals.
5. Put the pooled table, sample sizes, camera details and the failure cases into TESTING.md.
   Only then can the targets in §2 be called met or not met.
