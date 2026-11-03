"""Exercise actual detached subprocesses; no GPU or network is needed."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orchestrator.iteration.jobs import (
    JobFailed, JobObservationTimeout, file_fingerprint, job_status,
    process_identity, process_matches, submit_job, wait_for_job,
)
from orchestrator.iteration.state import read_json, write_json

REPO = Path(__file__).resolve().parents[1]


def make_spec(tmp_path, body=None, delay=0):
    script = tmp_path / "task.py"
    script.write_text(body or (
        "import json,time\nfrom pathlib import Path\n"
        f"time.sleep({delay})\n"
        "counter=Path('counter.txt')\n"
        "counter.write_text(counter.read_text()+'x' if counter.exists() else 'x')\n"
        "Path('weights.bin').write_bytes(b'trained-weights')\n"
        "Path('result.json').write_text(json.dumps({'backend':'real','steps':2}))\n"
    ))
    return {"argv": [sys.executable, str(script)], "cwd": str(tmp_path),
            "result_path": str(tmp_path / "result.json"),
            "input_digests": {"script": file_fingerprint(script)["sha256"]},
            "required_artifacts": [str(tmp_path / "weights.bin")],
            "result_equals": {"backend": "real", "steps": 2}}


def test_real_subprocess_result_and_idempotent_submission(tmp_path):
    spec = make_spec(tmp_path, delay=0.2)
    folder = tmp_path / "job"
    first = submit_job(folder, spec)
    assert submit_job(folder, spec)["attempt"] == first["attempt"]
    assert wait_for_job(folder, timeout_s=15, poll_interval_s=0.02)["steps"] == 2
    assert submit_job(folder, spec)["status"] == "succeeded"
    assert (tmp_path / "counter.txt").read_text() == "x"
    state = job_status(folder)
    assert state["exit_code"] == 0
    assert len(state["artifacts"]) == 2
    (tmp_path / "weights.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="artifacts_changed"):
        job_status(folder)


def test_timeout_reattaches_same_live_job_without_relaunch(tmp_path):
    spec = make_spec(tmp_path, delay=1)
    folder = tmp_path / "job"
    submit_job(folder, spec)
    with pytest.raises(JobObservationTimeout):
        wait_for_job(folder, timeout_s=0.01, poll_interval_s=0.01)
    assert job_status(folder)["status"] in ("queued", "running")
    assert submit_job(folder, spec)["attempt"] == 1
    wait_for_job(folder, timeout_s=15, poll_interval_s=0.02)
    assert (tmp_path / "counter.txt").read_text() == "x"


def test_worker_survives_submitting_controller_exit(tmp_path):
    spec = make_spec(tmp_path, delay=0.6)
    folder = tmp_path / "job"
    spec_path = tmp_path / "input.json"
    write_json(spec_path, spec)
    launcher = (
        "from pathlib import Path\n"
        "from orchestrator.iteration.jobs import submit_job\n"
        "from orchestrator.iteration.state import read_json\n"
        f"submit_job(Path({str(folder)!r}), read_json(Path({str(spec_path)!r})))\n"
        "import os; os._exit(0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", launcher], cwd=REPO, timeout=10)
    assert completed.returncode == 0
    assert wait_for_job(folder, timeout_s=15, poll_interval_s=0.02)["steps"] == 2
    assert (tmp_path / "counter.txt").read_text() == "x"


@pytest.mark.parametrize("body,error", [
    ("raise SystemExit(7)\n", "command_exit:7"),
    ("pass\n", "FileNotFoundError"),
    ("from pathlib import Path\nPath('result.json').write_text('{}')\n", "contract_mismatch"),
])
def test_failed_command_or_incomplete_outputs_never_succeed(tmp_path, body, error):
    folder = tmp_path / "job"
    spec = make_spec(tmp_path, body)
    submit_job(folder, spec)
    with pytest.raises(JobFailed, match=error):
        wait_for_job(folder, timeout_s=15, poll_interval_s=0.02)
    assert submit_job(folder, spec)["attempt"] == 1


def test_job_inputs_are_immutable(tmp_path):
    spec = make_spec(tmp_path)
    folder = tmp_path / "job"
    submit_job(folder, spec)
    wait_for_job(folder, timeout_s=15, poll_interval_s=0.02)
    changed = {**spec, "input_digests": {"dataset": "different"}}
    with pytest.raises(ValueError, match="spec_changed"):
        submit_job(folder, changed)


def test_explicit_retry_preserves_previous_attempt(tmp_path):
    script = (
        "from pathlib import Path\n"
        "if not Path('once').exists():\n"
        " Path('once').write_text('1')\n"
        " Path('result.json').write_text('{}')\n"
        " raise SystemExit(5)\n"
        "Path('weights.bin').write_bytes(b'ok')\n"
        "Path('result.json').write_text('{\"backend\":\"real\",\"steps\":2}')\n"
    )
    spec = make_spec(tmp_path, script)
    folder = tmp_path / "job"
    submit_job(folder, spec)
    with pytest.raises(JobFailed):
        wait_for_job(folder, timeout_s=15, poll_interval_s=0.02)
    assert submit_job(folder, spec, retry_failed=True)["attempt"] == 2
    wait_for_job(folder, timeout_s=15, poll_interval_s=0.02)
    assert read_json(folder / "attempts" / "001.json")["exit_code"] == 5
    assert list((folder / "attempts").glob("001_artifact*"))


def test_pid_reuse_and_dead_worker_are_not_treated_as_live(tmp_path):
    current = process_identity(os.getpid())
    assert current is not None and process_matches(current)
    wrong = {**current, "started": "different"}
    assert not process_matches(wrong)
    folder = tmp_path / "job"
    write_json(folder / "status.json", {"status": "running", "process": wrong, "attempt": 1})
    assert job_status(folder)["error"] == "worker_lost"
