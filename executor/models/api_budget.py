"""Atomic API attempt budget shared by clients, processes and restarts."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class ApiBudgetExhausted(ValueError):
    """Execution needs an additional authorized attempt; this is no semantic verdict."""


class ApiBudget:
    def __init__(self, path: str, max_calls: int):
        if max_calls < 1:
            raise ValueError("Persistent API budget requires a positive max_calls")
        self.path, self.max_calls = str(path), max_calls
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS budget (singleton INTEGER PRIMARY KEY CHECK(singleton=1), max_calls INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, started REAL, prompt_sha256 TEXT, model TEXT, status TEXT, result TEXT)")
            db.execute("INSERT OR IGNORE INTO budget VALUES (1, ?)", (max_calls,))
            stored = db.execute("SELECT max_calls FROM budget WHERE singleton=1").fetchone()[0]
            if stored != max_calls:
                raise ValueError("Existing API budget limit differs; choose a new run ledger instead of resetting it")

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, prompt_sha256: str, model: str) -> str:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            if count >= self.max_calls:
                raise ApiBudgetExhausted("Persistent API call budget exhausted")
            identifier = uuid.uuid4().hex
            db.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, 'reserved', NULL)", (identifier, time.time(), prompt_sha256, model))
            return identifier

    def finish(self, identifier: str, status: str, result: dict):
        with self._connection() as db:
            db.execute("UPDATE attempts SET status=?, result=? WHERE id=?", (status, json.dumps(result), identifier))

    def snapshot(self) -> dict:
        with self._connection() as db:
            rows = db.execute("SELECT id, status, prompt_sha256, model, result FROM attempts ORDER BY started").fetchall()
        return {"max_calls": self.max_calls, "attempts_reserved": len(rows), "remaining": self.max_calls - len(rows),
            "attempts": [{"id": r[0], "status": r[1], "prompt_sha256": r[2], "model": r[3], "result": json.loads(r[4]) if r[4] else None} for r in rows]}
