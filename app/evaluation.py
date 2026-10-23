"""Score a processed session against hand-made labels for the same video.

This is how real-world accuracy gets measured. Nothing here invents numbers:
with no labelled footage there is nothing to report.

Label file (JSON, one per video; template in configs/labels_template.json):

    {
      "video": "cam1_morning.mp4",
      "vehicles":   {"A": [0.42, 0.61], "B": [0.70, 0.60]},   # a point on each labelled car, 0..1 image coords
      "exits":      [{"time": 12.4, "vehicle": "A"}],         # a person appears getting out of A
      "entries":    [{"time": 95.0, "vehicle": "A"}],         # a person gets into A
      "movements":  [{"time": 101.0, "vehicle": "A"}],        # a parked A starts to move
      "departures": [{"time": 110.0, "vehicle": "A"}],        # A leaves the area / frame
      "incidents":  [{"id": "T1", "start": 80.0, "end": 130.0, "vehicle": "A"}]  # windows that SHOULD alert
    }

Times are seconds from the start of the video. A system item matches a label
when it is within the time tolerance and the labelled vehicle point lies
inside the system's vehicle box. The system never sees the labels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import RISK_LEVELS

LINKED = ("observed_exit", "possible_exit")
EVENT_LABELS = {"entries": "apparent_entry", "movements": "vehicle_moved", "departures": "vehicle_departed"}


class LabelError(ValueError):
    pass


@dataclass
class Tolerances:
    exit_seconds: float = 3.0  # association decisions are made after a 2.5 s observation window
    event_seconds: float = 3.0
    departure_seconds: float = 6.0
    incident_grace_seconds: float = 10.0  # alerts shortly after an incident window still count for it
    box_margin: float = 0.05  # grow system boxes by this fraction before the point test


@dataclass
class Labels:
    video: str
    vehicles: dict[str, tuple[float, float]]
    exits: list[dict] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)
    movements: list[dict] = field(default_factory=list)
    departures: list[dict] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    duration_seconds: float | None = None


def load_labels(path: str | Path) -> Labels:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelError(f"Cannot read label file {path}: {exc}") from exc
    vehicles = {}
    for name, pt in (data.get("vehicles") or {}).items():
        if not (isinstance(pt, list) and len(pt) == 2 and all(0 <= c <= 1 for c in pt)):
            raise LabelError(f"vehicle {name!r}: point must be [x, y] in 0..1")
        vehicles[name] = (float(pt[0]), float(pt[1]))
    lab = Labels(str(data.get("video", "")), vehicles, duration_seconds=data.get("duration_seconds"))
    for key in ("exits", "entries", "movements", "departures"):
        items = data.get(key) or []
        for it in items:
            if "time" not in it or it.get("vehicle") not in vehicles:
                raise LabelError(f"{key}: each item needs 'time' and a 'vehicle' from 'vehicles', got {it}")
        setattr(lab, key, sorted(items, key=lambda i: i["time"]))
    for inc in data.get("incidents") or []:
        if "start" not in inc or "end" not in inc or inc["end"] < inc["start"]:
            raise LabelError(f"incident needs start <= end: {inc}")
        if inc.get("vehicle") is not None and inc["vehicle"] not in vehicles:
            raise LabelError(f"incident vehicle {inc['vehicle']!r} is not in 'vehicles'")
        lab.incidents.append(inc)
    return lab


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval for k successes out of n (None when n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "value": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


@dataclass
class SessionData:
    session_id: str
    width: int
    height: int
    duration: float
    events: list[dict]
    associations: list[dict]
    warnings: list[str] = field(default_factory=list)


def load_session(storage, session_id: str) -> SessionData:
    row = storage.get_session(session_id)
    if row is None:
        raise LabelError(f"No session {session_id} in {storage.path}")
    info = json.loads(row["video_info"] or "{}")
    stats = json.loads(row["stats"] or "{}")
    fps = info.get("fps") or 30.0
    st = stats.get("stats", {})
    frames = st.get("frames_read") or info.get("frame_count") or 0
    warnings = []
    if not stats:
        warnings.append("session has no final statistics (still running or crashed)")
    if st.get("frames_skipped"):
        warnings.append(f"{st['frames_skipped']} frames were skipped (responsive mode); use --mode complete for evaluation")
    events = []
    for e in storage.list_events(session_id):
        e = dict(e)
        e["location"] = json.loads(e["location"]) if e.get("location") else None
        events.append(e)
    assocs = []
    for a in storage.list_associations(session_id):
        a = dict(a)
        a["evidence"] = json.loads(a["evidence"]) if a.get("evidence") else {}
        assocs.append(a)
    return SessionData(session_id, int(info.get("width", 0)), int(info.get("height", 0)), frames / fps,
                       events, assocs, warnings)


def _contains(box, pt, w, h, margin) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    px, py = pt[0] * w, pt[1] * h
    return x1 - mx <= px <= x2 + mx and y1 - my <= py <= y2 + my


def evaluate(labels: Labels, s: SessionData, tol: Tolerances | None = None, min_level: str = "MEDIUM") -> dict:
    tol = tol or Tolerances()
    w, h = s.width, s.height
    inside = lambda box, vname: _contains(box, labels.vehicles[vname], w, h, tol.box_margin)  # noqa: E731
    report: dict[str, Any] = {"session": s.session_id, "video": labels.video, "warnings": list(s.warnings)}
    duration = labels.duration_seconds or s.duration
    hours = duration / 3600 if duration else 0.0
    report["duration_seconds"] = round(duration, 1)

    # ---- person -> vehicle exits ------------------------------------------
    used: set[int] = set()
    outcomes = {"correct": 0, "wrong_vehicle": 0, "ambiguous": 0, "missed": 0}
    details = []
    for ex in labels.exits:
        near = [a for a in s.associations if a["association_id"] not in used
                and abs(a["birth_ts"] - ex["time"]) <= tol.exit_seconds]
        near.sort(key=lambda a: abs(a["birth_ts"] - ex["time"]))

        def cand_box(a, vid=None):
            for c in a["evidence"].get("candidates", []):
                if vid is None or c.get("vehicle_id") == vid:
                    yield c.get("box")

        outcome, hit = "missed", None
        for a in near:  # linked to the labelled vehicle
            if a["status"] in LINKED and any(inside(b, ex["vehicle"]) for b in cand_box(a, a["vehicle_id"])):
                outcome, hit = "correct", a
                break
        if hit is None:
            for a in near:  # the labelled vehicle was a candidate but the system chose differently
                if any(inside(b, ex["vehicle"]) for b in cand_box(a)):
                    outcome, hit = ("ambiguous" if a["status"] == "ambiguous" else "wrong_vehicle"), a
                    break
        if hit is not None:
            used.add(hit["association_id"])
        outcomes[outcome] += 1
        details.append({"label_time": ex["time"], "vehicle": ex["vehicle"], "outcome": outcome,
                        "person_track_id": hit["person_track_id"] if hit else None})
    linked = [a for a in s.associations if a["status"] in LINKED]
    false_links = [a for a in linked if a["association_id"] not in used]
    n_exits = len(labels.exits)
    report["association"] = {
        "labelled_exits": n_exits,
        **outcomes,
        "exit_recall": rate(outcomes["correct"], n_exits),
        "vehicle_accuracy_when_linked": rate(outcomes["correct"], outcomes["correct"] + outcomes["wrong_vehicle"]),
        "link_precision": rate(len(linked) - len(false_links), len(linked)),
        "false_links": len(false_links),
        "false_links_per_hour": round(len(false_links) / hours, 2) if hours else None,
        "details": details,
        "unmatched_links": [{"person_track_id": a["person_track_id"], "vehicle_id": a["vehicle_id"],
                             "birth_ts": a["birth_ts"], "status": a["status"]} for a in false_links],
    }

    # ---- behaviour / vehicle events -------------------------------------------
    report["events"] = {}
    for key, kind in EVENT_LABELS.items():
        gt = getattr(labels, key)
        sys_ev = [e for e in s.events if e["category"] == kind and e.get("risk_score") is None]
        window = tol.departure_seconds if key == "departures" else tol.event_seconds
        taken: set[str] = set()
        tp = 0
        for g in gt:
            for e in sorted(sys_ev, key=lambda e: abs(e["video_ts"] - g["time"])):
                if e["event_id"] in taken or abs(e["video_ts"] - g["time"]) > window:
                    continue
                vbox = (e["location"] or {}).get("vehicle_box")
                if inside(vbox, g["vehicle"]):
                    taken.add(e["event_id"])
                    tp += 1
                    break
        fp = len(sys_ev) - len(taken)
        report["events"][kind] = {"labelled": len(gt), "detected": len(sys_ev), "true_positives": tp,
                                  "recall": rate(tp, len(gt)), "precision": rate(tp, len(sys_ev)),
                                  "false_positives": fp,
                                  "false_positives_per_hour": round(fp / hours, 2) if hours else None}

    # ---- alerts vs incident windows ---------------------------------------
    alerts = [e for e in s.events if e.get("risk_score") is not None]
    report["alerts"] = {"total": len(alerts), "by_level": {}}
    for lv in RISK_LEVELS[1:]:
        report["alerts"]["by_level"][lv] = sum(a["risk_level"] == lv for a in alerts)

    def at_least(a, level):
        return RISK_LEVELS.index(a["risk_level"]) >= RISK_LEVELS.index(level)

    def matches(a, inc):
        if not inc["start"] <= a["video_ts"] <= inc["end"] + tol.incident_grace_seconds:
            return False
        if inc.get("vehicle") is None:
            return True
        return inside((a["location"] or {}).get("vehicle_box"), inc["vehicle"])

    inc_rows = []
    detected = 0
    for inc in labels.incidents:
        hits = sorted((a for a in alerts if at_least(a, min_level) and matches(a, inc)), key=lambda a: a["video_ts"])
        if hits:
            detected += 1
        inc_rows.append({"id": inc.get("id"), "detected": bool(hits),
                         "latency_seconds": round(hits[0]["video_ts"] - inc["start"], 2) if hits else None,
                         "first_level": hits[0]["risk_level"] if hits else None})
    latencies = sorted(r["latency_seconds"] for r in inc_rows if r["latency_seconds"] is not None)
    report["incidents"] = {
        "labelled": len(labels.incidents), "min_level": min_level,
        "recall": rate(detected, len(labels.incidents)),
        "median_latency_seconds": latencies[len(latencies) // 2] if latencies else None,
        "details": inc_rows,
    }
    for lv in ("LOW", "MEDIUM", "HIGH"):
        false = [a for a in alerts if at_least(a, lv) and not any(matches(a, i) for i in labels.incidents)]
        report["alerts"][f"false_alerts_at_least_{lv}"] = len(false)
        report["alerts"][f"false_alerts_per_hour_at_least_{lv}"] = round(len(false) / hours, 2) if hours else None
    return report


def aggregate(reports: list[dict]) -> dict:
    """Pool counts over several labelled videos and recompute the rates and intervals."""
    if not reports:
        return {}
    hours = sum(r["duration_seconds"] for r in reports) / 3600
    a = [r["association"] for r in reports]
    sums = {k: sum(x[k] for x in a) for k in ("labelled_exits", "correct", "wrong_vehicle", "ambiguous", "missed", "false_links")}
    n_linked = sum(x["link_precision"]["n"] for x in a)
    out: dict[str, Any] = {
        "videos": len(reports),
        "hours": round(hours, 3),
        "association": {
            **sums,
            "exit_recall": rate(sums["correct"], sums["labelled_exits"]),
            "vehicle_accuracy_when_linked": rate(sums["correct"], sums["correct"] + sums["wrong_vehicle"]),
            "link_precision": rate(n_linked - sums["false_links"], n_linked),
            "false_links_per_hour": round(sums["false_links"] / hours, 2) if hours else None,
        },
        "events": {},
    }
    for kind in EVENT_LABELS.values():
        ev = [r["events"][kind] for r in reports]
        lab, det, tp = (sum(e[k] for e in ev) for k in ("labelled", "detected", "true_positives"))
        out["events"][kind] = {"recall": rate(tp, lab), "precision": rate(tp, det),
                               "false_positives_per_hour": round((det - tp) / hours, 2) if hours else None}
    inc = [r["incidents"] for r in reports]
    out["incidents"] = {"recall": rate(sum(i["recall"]["k"] for i in inc), sum(i["labelled"] for i in inc)),
                        "min_level": inc[0]["min_level"]}
    for lv in ("LOW", "MEDIUM", "HIGH"):
        n = sum(r["alerts"][f"false_alerts_at_least_{lv}"] for r in reports)
        out[f"false_alerts_per_hour_at_least_{lv}"] = round(n / hours, 2) if hours else None
    out["warnings"] = [w for r in reports for w in r.get("warnings", [])]
    return out
