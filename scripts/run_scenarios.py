"""Replay the synthetic scenario suite (tests/scenarios.py) and print how the rules react.

    python scripts/run_scenarios.py [--json out.json]

These are scripted, idealised tracks: a regression check for the rule logic,
not a measurement of real-world accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scenarios import SCENARIOS, run_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write results to this file")
    args = ap.parse_args()
    results = run_all()
    print(f"{'scenario':<28} {'expect':<16} {'alerts':<28} {'pass':<5} events")
    for r in results:
        alerts = ",".join(r.alert_levels) or "-"
        ev = ", ".join(f"{k}={v}" for k, v in sorted(r.events.items()))
        print(f"{r.name:<28} {r.expectation:<16} {alerts:<28} {'yes' if r.passed else 'NO':<5} {ev}")
    benign = [r for r in results if not r.suspicious]
    positive = [r for r in results if r.suspicious]
    print(f"\nbenign scenarios with any alert: {sum(bool(r.alert_levels) for r in benign)}/{len(benign)}"
          f" | with an alert >= MEDIUM: {sum(any(l in ('MEDIUM', 'HIGH', 'CRITICAL') for l in r.alert_levels) for r in benign)}/{len(benign)}")
    print(f"suspicious scenarios meeting their expected level: {sum(r.passed for r in positive)}/{len(positive)}")
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
