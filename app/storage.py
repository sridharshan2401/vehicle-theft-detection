"""SQLite storage for sessions, events, associations and the local registry.

Two kinds of data are kept deliberately apart:

* Observations made by the system (events, observed person-vehicle
  associations, identity re-connections). These are what the camera
  suggested, with confidences.
* The registry, entered by a person: registered vehicles, authorised people
  and which people are authorised for which vehicle. Only this is used to
  decide that someone is authorised. Confirming an observed association
  records a human review decision; it does not authorise anyone.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    video_path TEXT NOT NULL,
    mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    video_info TEXT,
    settings TEXT,
    stats TEXT
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    video_path TEXT NOT NULL,
    frame INTEGER NOT NULL,
    video_ts REAL NOT NULL,
    vehicle_id TEXT,
    person_track_id TEXT,
    person_identity TEXT,
    category TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    risk_score REAL,
    reasons TEXT,
    confidences TEXT,
    message TEXT,
    snapshot_path TEXT,
    clip_path TEXT,
    processing_stats TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE TABLE IF NOT EXISTS associations (
    association_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    person_track_id TEXT NOT NULL,
    vehicle_id TEXT,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    birth_frame INTEGER,
    birth_ts REAL,
    evidence TEXT,
    review_status TEXT NOT NULL DEFAULT 'unreviewed',
    reviewed_at TEXT,
    review_note TEXT
);
CREATE TABLE IF NOT EXISTS identity_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    backend TEXT,
    video_ts REAL,
    frame INTEGER
);
CREATE TABLE IF NOT EXISTS clothing_observations (
    obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    profile_id TEXT,
    video_ts REAL,
    frame INTEGER,
    attributes TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS registered_vehicles (
    vehicle_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    registration_number TEXT,
    parking_zone TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS authorized_people (
    person_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    embedding_path TEXT,
    reid_backend TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vehicle_authorizations (
    vehicle_record_id INTEGER NOT NULL REFERENCES registered_vehicles(vehicle_record_id),
    person_record_id INTEGER NOT NULL REFERENCES authorized_people(person_record_id),
    created_at TEXT NOT NULL,
    PRIMARY KEY (vehicle_record_id, person_record_id)
);
"""

REVIEW_STATES = ("unreviewed", "confirmed", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, default=str)


class StorageError(RuntimeError):
    pass


class Storage:
    """Thin wrapper over one SQLite connection, safe to share between threads."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"Cannot open database {self.path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the first version, so older databases keep working."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(events)")}
        if "location" not in cols:  # pixel boxes, used to match events against labelled footage
            self._conn.execute("ALTER TABLE events ADD COLUMN location TEXT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # ---- sessions -------------------------------------------------------
    def create_session(self, video_path: str, mode: str, video_info: dict | None = None,
                       settings: dict | None = None, session_id: str | None = None) -> str:
        sid = session_id or datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self._exec(
            "INSERT INTO sessions (session_id, video_path, mode, started_at, video_info, settings) VALUES (?,?,?,?,?,?)",
            (sid, video_path, mode, _now(), _json(video_info), _json(settings)),
        )
        return sid

    def end_session(self, session_id: str, stats: dict) -> None:
        self._exec("UPDATE sessions SET ended_at=?, stats=? WHERE session_id=?", (_now(), _json(stats), session_id))

    def get_session(self, session_id: str) -> dict | None:
        rows = self._query("SELECT * FROM sessions WHERE session_id=?", (session_id,))
        return rows[0] if rows else None

    def list_sessions(self) -> list[dict]:
        return self._query("SELECT session_id, video_path, mode, started_at, ended_at FROM sessions ORDER BY started_at DESC")

    def delete_session(self, session_id: str) -> None:
        for table in ("events", "associations", "identity_links", "clothing_observations", "sessions"):
            self._exec(f"DELETE FROM {table} WHERE session_id=?", (session_id,))

    # ---- events ---------------------------------------------------------
    def add_event(self, *, session_id: str, video_path: str, frame: int, video_ts: float, category: str,
                  risk_level: str, vehicle_id: str | None = None, person_track_id: str | None = None,
                  person_identity: str | None = None, risk_score: float | None = None,
                  reasons: list | None = None, confidences: dict | None = None, message: str | None = None,
                  snapshot_path: str | None = None, clip_path: str | None = None,
                  processing_stats: dict | None = None, event_id: str | None = None,
                  location: dict | None = None) -> str:
        """``location`` holds pixel boxes ({"vehicle_box": [...], "person_box": [...]}) for evaluation."""
        eid = event_id or uuid.uuid4().hex[:12]
        self._exec(
            """INSERT INTO events (event_id, session_id, video_path, frame, video_ts, vehicle_id, person_track_id,
               person_identity, category, risk_level, risk_score, reasons, confidences, message, snapshot_path,
               clip_path, processing_stats, created_at, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (eid, session_id, video_path, frame, video_ts, vehicle_id, person_track_id, person_identity, category,
             risk_level, risk_score, _json(reasons), _json(confidences), message, snapshot_path, clip_path,
             _json(processing_stats), _now(), _json(location)),
        )
        return eid

    def update_event_clip(self, event_id: str, clip_path: str | None) -> None:
        self._exec("UPDATE events SET clip_path=? WHERE event_id=?", (clip_path, event_id))

    def list_events(self, session_id: str | None = None) -> list[dict]:
        if session_id:
            return self._query("SELECT * FROM events WHERE session_id=? ORDER BY video_ts", (session_id,))
        return self._query("SELECT * FROM events ORDER BY created_at")

    # ---- observations ---------------------------------------------------
    def add_association(self, session_id: str, person_track_id: str, vehicle_id: str | None, status: str,
                        confidence: float, birth_frame: int, birth_ts: float, evidence: dict) -> int:
        cur = self._exec(
            """INSERT INTO associations (session_id, person_track_id, vehicle_id, status, confidence, birth_frame,
               birth_ts, evidence) VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, person_track_id, vehicle_id, status, confidence, birth_frame, birth_ts, _json(evidence)),
        )
        return int(cur.lastrowid)

    def review_association(self, association_id: int, decision: str, note: str | None = None) -> None:
        if decision not in REVIEW_STATES[1:]:
            raise StorageError(f"decision must be one of {REVIEW_STATES[1:]}")
        cur = self._exec(
            "UPDATE associations SET review_status=?, reviewed_at=?, review_note=? WHERE association_id=?",
            (decision, _now(), note, association_id),
        )
        if cur.rowcount == 0:
            raise StorageError(f"No association with id {association_id}")

    def list_associations(self, session_id: str | None = None) -> list[dict]:
        if session_id:
            return self._query("SELECT * FROM associations WHERE session_id=? ORDER BY birth_ts", (session_id,))
        return self._query("SELECT * FROM associations ORDER BY association_id")

    def add_identity_link(self, session_id: str, track_id: str, profile_id: str, status: str, score: float | None,
                          backend: str, video_ts: float, frame: int) -> None:
        self._exec(
            """INSERT INTO identity_links (session_id, track_id, profile_id, status, score, backend, video_ts, frame)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, track_id, profile_id, status, score, backend, video_ts, frame),
        )

    def add_clothing_observation(self, session_id: str, track_id: str, profile_id: str | None, video_ts: float,
                                 frame: int, attributes: dict) -> None:
        self._exec(
            "INSERT INTO clothing_observations (session_id, track_id, profile_id, video_ts, frame, attributes) VALUES (?,?,?,?,?,?)",
            (session_id, track_id, profile_id, video_ts, frame, _json(attributes)),
        )

    # ---- registry -------------------------------------------------------
    def register_vehicle(self, name: str, registration_number: str | None = None, parking_zone: str | None = None,
                         notes: str | None = None) -> int:
        cur = self._exec(
            "INSERT INTO registered_vehicles (name, registration_number, parking_zone, notes, created_at) VALUES (?,?,?,?,?)",
            (name, registration_number, parking_zone, notes, _now()),
        )
        return int(cur.lastrowid)

    def register_person(self, label: str, embedding_path: str | None, reid_backend: str | None,
                        notes: str | None = None) -> int:
        cur = self._exec(
            "INSERT INTO authorized_people (label, embedding_path, reid_backend, notes, created_at) VALUES (?,?,?,?,?)",
            (label, embedding_path, reid_backend, notes, _now()),
        )
        return int(cur.lastrowid)

    def authorize(self, vehicle_record_id: int, person_record_id: int) -> None:
        try:
            self._exec(
                "INSERT OR IGNORE INTO vehicle_authorizations (vehicle_record_id, person_record_id, created_at) VALUES (?,?,?)",
                (vehicle_record_id, person_record_id, _now()),
            )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"Unknown vehicle {vehicle_record_id} or person {person_record_id}") from exc

    def revoke(self, vehicle_record_id: int, person_record_id: int) -> None:
        self._exec("DELETE FROM vehicle_authorizations WHERE vehicle_record_id=? AND person_record_id=?",
                   (vehicle_record_id, person_record_id))

    def list_vehicles(self) -> list[dict]:
        return self._query("SELECT * FROM registered_vehicles ORDER BY vehicle_record_id")

    def list_people(self) -> list[dict]:
        return self._query("SELECT * FROM authorized_people ORDER BY person_record_id")

    def vehicle_for_zone(self, zone_name: str) -> dict | None:
        rows = self._query("SELECT * FROM registered_vehicles WHERE parking_zone=? ORDER BY vehicle_record_id", (zone_name,))
        return rows[0] if rows else None

    def authorized_people_for_vehicle(self, vehicle_record_id: int) -> list[dict]:
        return self._query(
            """SELECT p.* FROM authorized_people p JOIN vehicle_authorizations a
               ON a.person_record_id = p.person_record_id WHERE a.vehicle_record_id=?""",
            (vehicle_record_id,),
        )
