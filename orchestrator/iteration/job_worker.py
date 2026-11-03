"""Detached supervisor for one persistent job attempt."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .jobs import file_lock, process_identity, validate_outputs
from .state import read_json, write_json


def run(job_dir: Path, attempt: int) -> None:
    try:
        # On Linux the command inherits the locked open file description.
        # Closing the supervisor's FD (including SIGKILL) cannot release the
        # child's lease. An explicit LOCK_UN here would release both owners.
        with file_lock(job_dir / "execution.lock", blocking=False, release_by_close=True) as lease:
            _execute(job_dir, attempt, lease.fileno())
    except BlockingIOError:
        # Another supervisor already owns this job; never run a duplicate.
        return


def _execute(job_dir: Path, attempt: int, lease_fd: int) -> None:
    path = job_dir / "status.json"
    spec = read_json(job_dir / "spec.json")
    with file_lock(job_dir / "state.lock"):
        state = read_json(path)
        if state["attempt"] != attempt or state["status"] != "queued":
            return
        state.update(status="running", process=process_identity(os.getpid()), started_at=time.time(),
                     command_launch="starting")
        write_json(path, state)
    command = None
    try:
        with (job_dir / f"command_{attempt:03d}.log").open("ab") as log:
            options = {"pass_fds": (lease_fd,)} if os.name != "nt" else {}
            command = subprocess.Popen(spec["argv"], cwd=spec["cwd"], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, **options)
            with file_lock(job_dir / "state.lock"):
                state["command_process"] = process_identity(command.pid)
                state["command_launch"] = "started"
                write_json(path, state)
            exit_code = command.wait()
        state["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"command_exit:{exit_code}")
        _, artifacts = validate_outputs(spec)
        state.update(status="succeeded", artifacts=artifacts)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}:{exc}")
        if command is None:
            state["command_launch"] = "not_started"
        elif command.poll() is None:
            state["status"] = "running"
            state["command_process"] = process_identity(command.pid)
    state["finished_at"] = time.time()
    with file_lock(job_dir / "state.lock"):
        write_json(path, state)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), int(sys.argv[2]))
