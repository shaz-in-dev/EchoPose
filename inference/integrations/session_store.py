"""SQLite-backed session store — persists analytics history across server restarts.

Stores:
  - events      : fall, presence change, inactivity, health alerts (from webhook manager)
  - vitals       : HR/RR/SpO2/stress snapshots every 5 minutes
  - system_state : key/value pairs (e.g. last_activity_ts so webhooks survive restarts)

Configure:
  SESSION_DB_PATH=data/sessions.db   (default)
  SESSION_VITALS_INTERVAL_S=300      (how often to snapshot vitals, default 5 min)

Endpoints added by server.py:
  GET /history/events?hours=24        recent alert events
  GET /history/vitals?hours=24        vitals timeline
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rf_inference.session_store")

_DB_PATH         = Path(os.getenv("SESSION_DB_PATH", "data/sessions.db"))
_VITALS_INTERVAL = float(os.getenv("SESSION_VITALS_INTERVAL_S", "300"))  # 5 min


class SessionStore:
    """Thread-safe SQLite session store using connection-per-call pattern."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_vitals_ts: float = 0.0
        self._init_db()
        logger.info("Session store ready at %s", self.db_path)

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         REAL    NOT NULL,
                    event_type TEXT    NOT NULL,
                    severity   TEXT    NOT NULL DEFAULT 'info',
                    message    TEXT,
                    details    TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

                CREATE TABLE IF NOT EXISTS vitals_snapshots (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts             REAL    NOT NULL,
                    heart_rate     REAL,
                    respiratory_rate REAL,
                    spo2           REAL,
                    stress_score   REAL,
                    activity       TEXT,
                    person_count   INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_vitals_ts ON vitals_snapshots(ts);

                CREATE TABLE IF NOT EXISTS system_state (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        message:    str,
        severity:   str = "info",
        details:    Any = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events (ts, event_type, severity, message, details) VALUES (?,?,?,?,?)",
                (time.time(), event_type, severity, message,
                 json.dumps(details) if details is not None else None),
            )

    def maybe_log_vitals(self, analytics: dict, person_count: int) -> None:
        """Log a vitals snapshot at most every SESSION_VITALS_INTERVAL_S seconds."""
        now = time.time()
        if now - self._last_vitals_ts < _VITALS_INTERVAL:
            return
        self._last_vitals_ts = now

        vitals = analytics.get("vitals") or {}
        hr     = (vitals.get("heart_rate")       or {}).get("heart_rate")
        rr     = (vitals.get("respiratory_rate") or {}).get("respiratory_rate")
        spo2   = (vitals.get("spo2")             or {}).get("spo2")
        stress = (analytics.get("emotion")       or {}).get("stress_score")
        act    = (analytics.get("activity")      or {}).get("activity")

        if any(v is not None for v in [hr, rr, spo2, stress, act]):
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO vitals_snapshots "
                    "(ts, heart_rate, respiratory_rate, spo2, stress_score, activity, person_count) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (now, hr, rr, spo2, stress, act, person_count),
                )

    def set_state(self, key: str, value: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?,?,?)",
                (key, json.dumps(value), time.time()),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()
            return json.loads(row["value"]) if row else default

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_events(self, hours: float = 24.0, limit: int = 200) -> list[dict]:
        since = time.time() - hours * 3600
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, event_type, severity, message, details "
                "FROM events WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        return [
            {
                "ts":         r["ts"],
                "event_type": r["event_type"],
                "severity":   r["severity"],
                "message":    r["message"],
                "details":    json.loads(r["details"]) if r["details"] else None,
            }
            for r in rows
        ]

    def get_vitals(self, hours: float = 24.0, limit: int = 500) -> list[dict]:
        since = time.time() - hours * 3600
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, heart_rate, respiratory_rate, spo2, stress_score, activity, person_count "
                "FROM vitals_snapshots WHERE ts >= ? ORDER BY ts ASC LIMIT ?",
                (since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        with self._conn() as conn:
            total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            total_vitals = conn.execute("SELECT COUNT(*) FROM vitals_snapshots").fetchone()[0]
            last_fall    = conn.execute(
                "SELECT ts FROM events WHERE event_type='fall_detected' ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return {
            "total_events":     total_events,
            "total_vitals_pts": total_vitals,
            "last_fall_ts":     last_fall[0] if last_fall else None,
            "db_path":          str(self.db_path),
        }
