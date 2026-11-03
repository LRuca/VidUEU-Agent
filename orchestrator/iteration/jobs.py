"""Persistent subprocess jobs, independent of the observing controller.

Commands use argv (never shell strings). Credentials belong in the inherited
environment, not the persisted spec. A timeout only stops observation.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .state import digest, read_json, write_json


class JobFailed(RuntimeError):
    pass


class JobObservationTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True, release_by_close: bool = False) -> Iterator[BinaryIO]:
    """OS locks release on process death; the file's existence is immaterial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if not blocking:
                        raise BlockingIOError(str(path))
                    time.sleep(0.02)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield stream
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            elif not release_by_close:
                fcntl.flock(stream, fcntl.LOCK_UN)


def execution_is_locked(job_dir: Path) -> bool:
    try:
        with file_lock(job_dir / "execution.lock", blocking=False):
            return False
    except BlockingIOError:
        return True


def process_identity(pid: int) -> dict[str, Any] | None:
    """PID plus kernel start time prevents mistaken attachment after PID reuse."""
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                return None
            return {"pid": pid, "started": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}
        finally:
            kernel.CloseHandle(handle)
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": pid, "started": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_matches(identity: dict[str, Any] | None) -> bool:
    return bool(identity and process_identity(int(identity["pid"])) == identity)


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"artifact_missing_or_empty:{path}")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": value.hexdigest()}


def _validated_spec(spec: dict[str, Any]) -> dict[str, Any]:
    required = {"argv", "cwd", "result_path", "input_digests"}
    if not isinstance(spec, dict) or not required <= set(spec):
        raise ValueError("job_spec_missing_fields")
    allowed = required | {"required_artifacts", "result_equals", "kind"}
    if set(spec) - allowed:
        raise ValueError("job_spec_unknown_fields")
    argv = spec["argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv):
        raise ValueError("job_requires_nonempty_argv")
    if not isinstance(spec["input_digests"], dict) or not spec["input_digests"]:
        raise ValueError("job_input_digests_required")
    result = dict(spec)
    for key in ("cwd", "result_path"):
        if not Path(spec[key]).is_absolute():
            raise ValueError(f"job_requires_absolute_path:{key}")
        result[key] = str(Path(spec[key]).resolve())
    if not Path(result["cwd"]).is_dir():
        raise ValueError("job_cwd_missing")
    artifacts = spec.get("required_artifacts", [])
    if not isinstance(artifacts, list) or any(not Path(p).is_absolute() for p in artifacts):
        raise ValueError("job_artifacts_require_absolute_paths")
    result["required_artifacts"] = [str(Path(p).resolve()) for p in artifacts]
    if not isinstance(spec.get("result_equals", {}), dict):
        raise ValueError("job_result_equals_must_be_object")
    result["result_equals"] = spec.get("result_equals", {})
    return result


def validate_outputs(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = Path(spec["result_path"])
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("job_result_must_be_object")
    for key, expected in spec["result_equals"].items():
        if key not in result or result[key] != expected:
            raise ValueError(f"job_result_contract_mismatch:{key}")
    paths = list(dict.fromkeys([spec["result_path"], *spec["required_artifacts"]]))
    return result, [file_fingerprint(Path(p)) for p in paths]


def _check_completed(spec: dict[str, Any], state: dict[str, Any]) -> None:
    _, artifacts = validate_outputs(spec)
    if artifacts != state.get("artifacts"):
        raise ValueError("completed_job_artifacts_changed")


def submit_job(job_dir: Path, spec: dict[str, Any], *, retry_failed: bool = False) -> dict[str, Any]:
    """Idempotent submission. Failed attempts require an explicit retry.

    A queued job with no identity is left pending: killing the submitter in the
    narrow spawn/persist window must never cause a second training launch.
    Workers also hold an execution lock for the entire command.
    """
    job_dir = job_dir.resolve()
    spec = _validated_spec(spec)
    job_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(job_dir / "state.lock"):
        spec_path, state_path = job_dir / "spec.json", job_dir / "status.json"
        attempt = 1
        if spec_path.exists():
            if read_json(spec_path) != spec:
                raise ValueError("job_spec_changed_use_new_job_directory")
            state = read_json(state_path)
            if state["status"] == "succeeded":
                _check_completed(spec, state)
                return state
            if state["status"] != "failed" or not retry_failed:
                return state
            # A supervisor can still be in its final state write after it has
            # published ``failed``.  The execution lease is the authoritative
            # indicator for that short window; checking the supervisor PID
            # first made explicit retries timing-dependent.  A command PID is
            # kept as an additional guard because a failed state must never
            # launch over a still-running payload command.
            if execution_is_locked(job_dir):
                raise ValueError("cannot_retry_locked_job")
            if process_matches(state.get("command_process")):
                raise ValueError("cannot_retry_live_job")
            if state.get("command_launch") == "starting" and not state.get("command_process"):
                raise ValueError("cannot_retry_ambiguous_job")
            write_json(job_dir / "attempts" / f'{state["attempt"]:03d}.json', state)
            attempt = int(state["attempt"]) + 1
        # Preserve old outputs on explicit retry so they cannot count as new.
        for index, name in enumerate(dict.fromkeys([spec["result_path"], *spec["required_artifacts"]])):
            path = Path(name)
            if path.exists():
                if not retry_failed:
                    raise ValueError(f"job_output_already_exists:{path}")
                archive = job_dir / "attempts" / f"{attempt - 1:03d}_artifact_{index}_{path.name}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, archive)
        write_json(spec_path, spec)
        state = {"schema_version": 1, "spec_digest": digest(spec), "attempt": attempt,
                 "status": "queued", "submitted_at": time.time(), "process": None}
        write_json(state_path, state)
        options: dict[str, Any] = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
        # Import the worker from this checkout, not from the command's cwd.
        repo_root = Path(__file__).resolve().parents[2]
        try:
            with (job_dir / f"worker_{attempt:03d}.log").open("ab") as log:
                child = subprocess.Popen(
                    [sys.executable, "-m", "orchestrator.iteration.job_worker", str(job_dir), str(attempt)],
                    cwd=repo_root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options,
                )
            state["process"] = process_identity(child.pid)
            write_json(state_path, state)
        except Exception as exc:
            state.update(status="failed", error=f"worker_launch:{type(exc).__name__}", finished_at=time.time())
            write_json(state_path, state)
            raise
        return state


def job_status(job_dir: Path) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    with file_lock(job_dir / "state.lock"):
        state = read_json(job_dir / "status.json")
        if state["status"] == "succeeded":
            _check_completed(read_json(job_dir / "spec.json"), state)
        elif state["status"] in ("queued", "running") and state.get("process"):
            if not process_matches(state["process"]):
                # A live orphaned child must finish before any retry.
                if process_matches(state.get("command_process")) or execution_is_locked(job_dir):
                    return {**state, "observer_note": "supervisor_lost_command_still_running"}
                if state.get("command_launch") == "starting" and not state.get("command_process"):
                    return {**state, "observer_note": "ambiguous_command_launch_requires_reconciliation"}
                state.update(status="failed", error="worker_lost", finished_at=time.time())
                write_json(job_dir / "status.json", state)
        return state


def wait_for_job(job_dir: Path, *, timeout_s: float = 3600, poll_interval_s: float = 1) -> dict[str, Any]:
    if timeout_s < 0 or poll_interval_s <= 0:
        raise ValueError("invalid_job_observation_limits")
    deadline = time.monotonic() + timeout_s
    while True:
        state = job_status(job_dir)
        if state["status"] == "succeeded":
            return read_json(Path(read_json(job_dir / "spec.json")["result_path"]))
        if state["status"] == "failed":
            raise JobFailed(f'{job_dir}:{state.get("error", "unknown")}')
        if time.monotonic() >= deadline:
            raise JobObservationTimeout(f"job_still_pending:{job_dir}")
        time.sleep(min(poll_interval_s, max(0, deadline - time.monotonic())))
