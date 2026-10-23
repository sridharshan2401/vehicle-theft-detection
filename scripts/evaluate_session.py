"""Evaluate processed sessions against hand labels (see app/evaluation.py for the label format).

    python scripts/evaluate_session.py --session <session-id> --labels data\\labels\\cam1.json
    python scripts/evaluate_session.py --manifest data\\labels\\eval_set.json

A manifest is {"items": [{"session": "...", "labels": "..."}, ...]} and gives pooled
results over all listed videos. Process the videos with --mode complete first.
Reports are written to evidence/evaluations/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_config, resolve_path  # noqa: E402
from app.evaluation import LabelError, Tolerances, aggregate, evaluate, load_labels, load_session  # noqa: E402
from app.storage import Storage  # noqa: E402


def fmt(r: dict | None) -> str:
    if not r or r["value"] is None:
        return "n/a (no cases)"
    lo, hi = r["ci95"]
    return f"{r['value']:.2f}  ({r['k']}/{r['n']}, 95% CI {lo:.2f}-{hi:.2f})"


def print_report(rep: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    for w in rep.get("warnings", []):
        print(f"WARNING: {w}")
    a = rep["association"]
    print(f"Exits: labelled {a['labelled_exits']}; correct {a['correct']}, wrong vehicle {a['wrong_vehicle']}, "
          f"ambiguous {a['ambiguous']}, missed {a['missed']}")
    print(f"  exit recall                 {fmt(a['exit_recall'])}")
    print(f"  vehicle accuracy if linked  {fmt(a['vehicle_accuracy_when_linked'])}")
    print(f"  link precision              {fmt(a['link_precision'])}")
    print(f"  false links per hour        {a['false_links_per_hour']}")
    for kind, e in rep["events"].items():
        print(f"{kind:<18} recall {fmt(e['recall'])} | precision {fmt(e['precision'])} | FP/h {e['false_positives_per_hour']}")
    inc = rep["incidents"]
    print(f"Incidents (alert >= {inc['min_level']}): recall {fmt(inc['recall'])}"
          + (f", median latency {inc['median_latency_seconds']} s" if inc.get("median_latency_seconds") is not None else ""))
    src = rep.get("alerts", rep)
    for lv in ("LOW", "MEDIUM", "HIGH"):
        print(f"False alerts per hour >= {lv:<6} {src.get(f'false_alerts_per_hour_at_least_{lv}')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session")
    ap.add_argument("--labels")
    ap.add_argument("--manifest")
    ap.add_argument("--min-level", default="MEDIUM", help="alert level that counts as detecting an incident")
    ap.add_argument("--tolerance", type=float, help="time tolerance in seconds for exits and events")
    args = ap.parse_args()
    if args.manifest:
        items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["items"]
    elif args.session and args.labels:
        items = [{"session": args.session, "labels": args.labels}]
    else:
        ap.error("give --session and --labels, or --manifest")
    tol = Tolerances()
    if args.tolerance:
        tol.exit_seconds = tol.event_seconds = args.tolerance
    db = Storage(resolve_path(load_config().storage.db_path))
    reports = []
    try:
        for it in items:
            rep = evaluate(load_labels(it["labels"]), load_session(db, it["session"]), tol, args.min_level)
            reports.append(rep)
            print_report(rep, f"{it['session']} vs {Path(it['labels']).name}")
    except LabelError as exc:
        print(f"error: {exc}")
        return 2
    finally:
        db.close()
    out = {"when": datetime.now().isoformat(timespec="seconds"), "reports": reports}
    if len(reports) > 1:
        out["pooled"] = agg = aggregate(reports)
        print(f"\n=== Pooled over {agg['videos']} videos, {agg['hours']} h ===")
        print(f"exit recall {fmt(agg['association']['exit_recall'])}")
        print(f"vehicle accuracy if linked {fmt(agg['association']['vehicle_accuracy_when_linked'])}")
        print(f"incident recall {fmt(agg['incidents']['recall'])}")
        for lv in ("LOW", "MEDIUM", "HIGH"):
            print(f"false alerts per hour >= {lv}: {agg[f'false_alerts_per_hour_at_least_{lv}']}")
    out_dir = resolve_path("evidence/evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"evaluation_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
