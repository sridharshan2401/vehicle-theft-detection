"""Regression test over the synthetic scenario suite (tests/scenarios.py).

Each scenario has an expectation written down before the false-positive fixes were
made (see docs/VALIDATION.md). These are scripted tracks, not footage.
"""

import pytest
from scenarios import SCENARIOS, run

from app.config import RISK_LEVELS


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_scenario_meets_expectation(scenario):
    r = run(scenario)
    assert r.passed, f"{r.name}: expected {r.expectation}, got alerts {r.alert_levels}, events {r.events}"


def test_unsure_owner_ranks_below_theft():
    by_name = {s.name: run(s) for s in SCENARIOS if s.name in ("theft", "owner_returns_reid_unsure")}
    rank = RISK_LEVELS.index
    assert rank(by_name["owner_returns_reid_unsure"].max_level) < rank(by_name["theft"].max_level)
