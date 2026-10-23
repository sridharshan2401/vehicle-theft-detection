"""Explainable, rule-based risk scoring per vehicle.

Each observation (event) adds points to the vehicle it concerns:

    points = weight[kind] * identity_factor * event_confidence

identity_factor depends on who was involved: 0 for a registered authorised
person, a small factor for someone we earlier saw getting out of this same
vehicle (an observation, not proof of authorisation), and 1 for an
unrecognised person. Scores decay exponentially (half-life in settings), the
same (vehicle, kind, person) cannot add points again within the cooldown, and
an alert is raised when the level escalates (or re-raised after
``realert_seconds`` if new evidence keeps arriving at the same level).

Wording is deliberately cautious: "suspicious interaction", "possible
unauthorised vehicle movement". Nothing here proves theft.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from app.config import RISK_LEVELS, RiskConfig
from app.datatypes import Event, EventKind

AUTHORIZED = "authorized"
OBSERVED_ASSOCIATED = "observed_associated"
POSSIBLE_ASSOCIATED = "possible_associated"  # unconfirmed ReID match to someone seen getting out of this vehicle
UNRECOGNIZED = "unrecognized"
# Most to least suspicious; for a vehicle movement the most suspicious entrant decides the factor.
IDENTITY_ORDER = (UNRECOGNIZED, POSSIBLE_ASSOCIATED, OBSERVED_ASSOCIATED, AUTHORIZED)

MOVEMENT_KINDS = {EventKind.VEHICLE_MOVED, EventKind.VEHICLE_LEFT_ZONE, EventKind.VEHICLE_DEPARTED}
SITE_KEY = "SITE"  # risk not tied to one vehicle, e.g. restricted-area entries


@dataclass
class PersonContext:
    status: str  # one of IDENTITY_ORDER
    person_id: str
    label: str | None = None  # registered name, when authorised
    related_id: str | None = None  # for possible_associated: the earlier person they may be

    def describe(self) -> str:
        if self.status == AUTHORIZED:
            return f"authorised person '{self.label}' ({self.person_id})"
        if self.status == OBSERVED_ASSOCIATED:
            return f"{self.person_id}, who was earlier seen getting out of this vehicle"
        if self.status == POSSIBLE_ASSOCIATED:
            return (f"{self.person_id}, possibly {self.related_id or 'someone'} who got out of this vehicle "
                    "(unconfirmed appearance match)")
        return f"an unrecognised person ({self.person_id})"


Resolver = Callable[[str, str | None], PersonContext]


@dataclass
class RiskReason:
    kind: str
    person_id: str | None
    points: float
    confidence: float
    description: str
    ts: float
    frame: int


@dataclass
class Alert:
    alert_id: str
    vehicle_id: str
    level: str
    score: float
    ts: float
    frame: int
    category: str
    message: str
    reasons: list[RiskReason]
    person_ids: list[str]


@dataclass
class VehicleRisk:
    score: float = 0.0
    last_ts: float = 0.0
    reasons: deque = field(default_factory=lambda: deque(maxlen=30))
    last_alert_ts: float | None = None
    last_alert_rank: int = 0
    reasons_since_alert: int = 0


def level_rank(level: str) -> int:
    return RISK_LEVELS.index(level)


class RiskEngine:
    def __init__(self, cfg: RiskConfig, id_prefix: str = "A"):
        self.cfg = cfg
        self.vehicles: dict[str, VehicleRisk] = {}
        self._cooldown: dict[tuple[str, str, str], float] = {}
        self._ids = itertools.count(1)
        self._prefix = id_prefix
        self.suppressed = 0

    def level_for(self, score: float) -> str:
        level = RISK_LEVELS[0]
        for name in RISK_LEVELS[1:]:
            if score >= self.cfg.levels[name]:
                level = name
        return level

    def _decayed(self, vr: VehicleRisk, ts: float) -> float:
        dt = max(0.0, ts - vr.last_ts)
        return vr.score * 0.5 ** (dt / self.cfg.half_life_seconds)

    def current(self, vehicle_id: str, ts: float) -> tuple[float, str]:
        vr = self.vehicles.get(vehicle_id)
        if vr is None:
            return 0.0, RISK_LEVELS[0]
        score = self._decayed(vr, ts)
        return score, self.level_for(score)

    def _factor_and_text(self, e: Event, resolve: Resolver) -> tuple[float, str, list[str]]:
        f = self.cfg.identity_factors
        if e.kind in MOVEMENT_KINDS:
            base = {
                EventKind.VEHICLE_MOVED: "the vehicle moved from its parking position",
                EventKind.VEHICLE_LEFT_ZONE: f"the vehicle left parking zone '{e.details.get('parked_zone', '?')}'",
                EventKind.VEHICLE_DEPARTED: "the vehicle appears to have left the area",
            }[e.kind]
            entrants = e.details.get("recent_entrants") or []
            if not entrants:
                return f["no_entrant_observed"], base + " with no one seen getting in (driver not observed)", []
            ctxs = [resolve(pid, e.vehicle_id) for pid in entrants]
            ids = [c.person_id for c in ctxs]
            worst = min(ctxs, key=lambda c: IDENTITY_ORDER.index(c.status))
            verb = "got in" if worst.status == AUTHORIZED else "appeared to get in"
            return f[worst.status], base + f" after {worst.describe()} {verb}", ids

        if e.person_id is None:
            return 1.0, e.kind.replace("_", " "), []
        ctx = resolve(e.person_id, e.vehicle_id)
        who = ctx.describe()
        text = {
            EventKind.PROXIMITY: f"{who} came close to the vehicle",
            EventKind.REPEATED_INTERACTION: f"{who} returned to the vehicle {e.details.get('approaches', '')} times",
            EventKind.EXTENDED_PRESENCE: f"{who} stayed near the vehicle for {e.details.get('seconds', '?')}s",
            EventKind.APPARENT_ENTRY: f"{who} appeared to get into the vehicle",
            EventKind.RESTRICTED_AREA: f"{who} entered restricted zone '{e.details.get('zone', '?')}'",
        }.get(e.kind, f"{who}: {e.kind}")
        return f[ctx.status], text, [ctx.person_id]

    def process(self, events: list[Event], resolve: Resolver) -> list[Alert]:
        """Score a batch of events (normally one frame's worth) and return any new alerts."""
        touched: dict[str, tuple[float, int]] = {}
        persons: dict[str, set[str]] = {}
        for e in events:
            weight = self.cfg.weights.get(e.kind)
            if weight is None:
                continue  # informational only (e.g. vehicle parked, person exited)
            key = e.vehicle_id or SITE_KEY
            cd_key = (key, e.kind, e.person_id or "")
            last = self._cooldown.get(cd_key)
            if last is not None and e.ts - last < self.cfg.cooldown_seconds:
                self.suppressed += 1
                continue
            self._cooldown[cd_key] = e.ts
            factor, text, pids = self._factor_and_text(e, resolve)
            points = weight * factor * e.confidence
            vr = self.vehicles.setdefault(key, VehicleRisk(last_ts=e.ts))
            vr.score = min(self.cfg.max_score, self._decayed(vr, e.ts) + points)
            vr.last_ts = e.ts
            vr.reasons.append(RiskReason(e.kind, e.person_id, round(points, 2), e.confidence, text, e.ts, e.frame))
            if points > 0:
                vr.reasons_since_alert += 1
            touched[key] = (e.ts, e.frame)
            persons.setdefault(key, set()).update(pids)

        alerts = []
        min_rank = level_rank(self.cfg.alert_min_level)
        for key, (ts, frame) in touched.items():
            vr = self.vehicles[key]
            level = self.level_for(vr.score)
            rank = level_rank(level)
            if rank < max(min_rank, 1):
                continue
            escalated = rank > vr.last_alert_rank
            repeat = (vr.last_alert_ts is not None and ts - vr.last_alert_ts >= self.cfg.realert_seconds
                      and vr.reasons_since_alert > 0)
            if not (escalated or repeat):
                continue
            alerts.append(self._make_alert(key, vr, level, ts, frame, sorted(persons.get(key, set()))))
            vr.last_alert_ts, vr.last_alert_rank, vr.reasons_since_alert = ts, rank, 0
        return alerts

    def tick(self, ts: float) -> None:
        """Let alert levels fall back as scores decay, so a later escalation alerts again."""
        for vr in self.vehicles.values():
            rank = level_rank(self.level_for(self._decayed(vr, ts)))
            if rank < vr.last_alert_rank:
                vr.last_alert_rank = rank

    def _make_alert(self, key: str, vr: VehicleRisk, level: str, ts: float, frame: int, pids: list[str]) -> Alert:
        recent = [r for r in vr.reasons if r.points > 0 and ts - r.ts <= self.cfg.half_life_seconds * 2]
        top = sorted(recent, key=lambda r: (r.points, r.ts), reverse=True)[:3]
        top.sort(key=lambda r: r.ts)
        category = ("possible unauthorised vehicle movement" if any(r.kind in MOVEMENT_KINDS for r in top)
                    else "suspicious interaction")
        subject = "the site" if key == SITE_KEY else key
        message = f"{level.title()}-priority alert for {subject} ({category}): " + "; ".join(r.description for r in top) + "."
        return Alert(f"{self._prefix}{next(self._ids):04d}", key, level, round(vr.score, 1), ts, frame,
                     category, message, list(top), pids)
