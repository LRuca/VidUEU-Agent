"""Persistent asynchronous jobs used by the iteration smoke mode.

The smoke runtime deliberately simulates wall-clock work without loading model
weights.  Jobs are file-backed, so a controller restart can observe a RUNNING
job and wait for it instead of submitting a duplicate job.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .state import read_json, write_json


class SmokeJobError(RuntimeError):
    """Raised when a smoke job fails or exceeds its timeout."""


def submit_smoke_job(
    path: Path,
    *,
    job_id: str,
    kind: str,
    iteration: int,
    duration_s: float,
    input_digest: str,
) -> dict[str, Any]:
    """Create or reuse a persistent smoke job.

    A completed job is immutable and reused.  A failed job can be submitted
    again with an incremented attempt number; a running job is returned as-is.
    """
    duration = max(0.0, float(duration_s))
    if path.exists():
        existing = read_json(path)
        if not isinstance(existing, dict):
            raise SmokeJobError(f"invalid_smoke_job:{path}")
        status = str(existing.get("status", ""))
        if status in {"RUNNING", "SUCCEEDED"}:
            return existing
        attempt = int(existing.get("attempt", 1) or 1) + 1
    else:
        attempt = 1
    now = time.time()
    job = {
        "schema_version": 1,
        "job_id": job_id,
        "kind": kind,
        "iteration": int(iteration),
        "attempt": attempt,
        "status": "RUNNING",
        "simulated": True,
        "input_digest": input_digest,
        "created_at": now,
        "started_at": now,
        "ready_at": now + duration,
        "duration_s": duration,
        "poll_count": 0,
        "waited_s": 0.0,
    }
    write_json(path, job)
    return job


def wait_for_smoke_job(
    path: Path,
    *,
    timeout_s: float,
    poll_interval_s: float,
    on_poll: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Wait for a file-backed job and mark it successful when ready."""
    timeout = max(0.0, float(timeout_s))
    poll_interval = max(0.001, float(poll_interval_s))
    wait_started = time.time()
    while True:
        job = read_json(path)
        if not isinstance(job, dict):
            raise SmokeJobError(f"invalid_smoke_job:{path}")
        status = str(job.get("status", ""))
        if status == "SUCCEEDED":
            return job
        if status == "FAILED":
            raise SmokeJobError(f"smoke_job_failed:{job.get('error', 'unknown')}")
        if status != "RUNNING":
            raise SmokeJobError(f"invalid_smoke_job_status:{status}")

        now = time.time()
        elapsed = now - wait_started
        if now >= float(job.get("ready_at", now)):
            # The completion check is itself a poll.  Counting it makes the
            # persisted lifecycle truthful even when a short job completes
            # between submission and the first sleep.
            job.update({
                "status": "SUCCEEDED",
                "finished_at": now,
                "waited_s": elapsed,
                "poll_count": int(job.get("poll_count", 0) or 0) + 1,
                "last_poll_at": now,
            })
            write_json(path, job)
            return job
        if elapsed >= timeout:
            job.update({
                "status": "FAILED",
                "finished_at": now,
                "waited_s": elapsed,
                "error": f"timeout_after_{timeout:g}s",
            })
            write_json(path, job)
            raise SmokeJobError(f"smoke_job_timeout:{path}")

        job["poll_count"] = int(job.get("poll_count", 0) or 0) + 1
        job["last_poll_at"] = now
        job["waited_s"] = elapsed
        write_json(path, job)
        if on_poll is not None:
            on_poll(dict(job))
        time.sleep(min(poll_interval, max(0.001, timeout - elapsed)))


__all__ = ["SmokeJobError", "submit_smoke_job", "wait_for_smoke_job"]
