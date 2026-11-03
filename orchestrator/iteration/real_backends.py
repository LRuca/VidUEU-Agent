"""Command-backed BAGEL training/export adapters.

The controller owns state and recovery; this module only builds immutable job
specifications and validates the result contracts.  The actual GPU work runs in
the repository scripts through the configured BAGEL Python environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .jobs import JobObservationTimeout, submit_job, wait_for_job
from .jobs import file_fingerprint
from .state import digest, read_json


class RealBackendError(RuntimeError):
    """A real backend completed without satisfying its artifact contract."""


def _bagel_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("bagel")
    if not isinstance(value, dict):
        value = config.get("trainer", {}).get("bagel")
    if not isinstance(value, dict):
        raise ValueError("bagel_config_missing")
    return value


def run_bagel_train(*, train_manifest: Path, round_dir: Path, iteration: int,
                    config: dict[str, Any]) -> dict[str, Any]:
    bagel = _bagel_config(config)
    vendor = Path(bagel["vendor_root"]).resolve()
    model = Path(bagel["model_path"]).resolve()
    python = str(bagel.get("python_executable", "python"))
    run_root = (round_dir / "train").resolve()
    # Keep the immutable job result separate from the controller's enriched
    # train_result.json.  The controller adds job metadata after the worker
    # succeeds; using the same path would invalidate the persisted artifact
    # fingerprint and make a completed job appear corrupted.
    job_result_path = (round_dir / "train_job_result.json").resolve()
    audit_path = (round_dir / "train_audit.json").resolve()
    job_dir = (round_dir / "jobs" / "train").resolve()
    episodes = (round_dir / "bagel_episodes.jsonl").resolve()
    if not episodes.is_file():
        raise RealBackendError(f"bagel_episodes_missing:{episodes}")
    steps = int(bagel.get("total_steps", bagel.get("smoke_steps", 2)))
    nproc = int(bagel.get("nproc_per_node", 1))
    argv = [python, str(Path(__file__).resolve().parents[2] / "scripts" / "run_bagel_train.py"),
            "--vendor-root", str(vendor), "--model-path", str(model),
            "--episodes", str(episodes), "--run-root", str(run_root),
            "--result-path", str(job_result_path), "--steps", str(steps),
            "--nproc-per-node", str(nproc), "--seed", str(config.get("loop", {}).get("seed", 0)),
            "--expected-num-tokens", str(int(bagel.get("expected_num_tokens", 4096))),
            "--max-num-tokens", str(int(bagel.get("max_num_tokens", 8192))),
            "--max-num-tokens-per-sample", str(int(bagel.get("max_num_tokens_per_sample", 8192))),
            "--master-port", str(int(bagel.get("master_port", 29601))),
            "--gradient-accumulation-steps", str(int(bagel.get("gradient_accumulation_steps", 1))),
            "--audit-output", str(audit_path)]
    if bagel.get("visible_devices"):
        argv.extend(["--visible-devices", str(bagel["visible_devices"])])
    if bool(bagel.get("require_nine_tasks", False)):
        argv.append("--require-nine-tasks")
        plan_path = (round_dir / "optimizer_coverage_plan.json").resolve()
        plan_argv = [python, str(Path(__file__).resolve().parents[2] / "scripts" / "plan_bagel_coverage.py"),
                     "--vendor-root", str(vendor), "--model-path", str(model),
                     "--episodes", str(episodes), "--report", str(plan_path),
                     "--nproc-per-node", str(nproc), "--steps", str(steps),
                     "--gradient-accumulation-steps", str(int(bagel.get("gradient_accumulation_steps", 1))),
                     "--expected-num-tokens", str(int(bagel.get("expected_num_tokens", 4096))),
                     "--max-num-tokens", str(int(bagel.get("max_num_tokens", 8192))),
                     "--max-num-tokens-per-sample", str(int(bagel.get("max_num_tokens_per_sample", 8192)))]
        plan_job = (round_dir / "jobs" / "train_coverage_plan").resolve()
        submit_job(plan_job, {"kind": "native_optimizer_coverage_plan", "argv": plan_argv,
                              "cwd": str(Path(__file__).resolve().parents[2]), "result_path": str(plan_path),
                              "input_digests": {"episodes": file_fingerprint(episodes)},
                              "required_artifacts": [str(plan_path)], "result_equals": {"status": "ready"}})
        wait_for_job(plan_job, timeout_s=float(bagel.get("timeout_s", 3600)),
                     poll_interval_s=float(bagel.get("poll_interval_s", 2)))
    spec = {
        "kind": "bagel_train",
        "argv": argv,
        "cwd": str(Path(__file__).resolve().parents[2]),
        "result_path": str(job_result_path),
        "input_digests": {
            "train_manifest": digest(read_json(round_dir / "train_manifest.json")),
            "episodes": file_fingerprint(episodes),
            "model_path": str(model),
        },
        "required_artifacts": [str((run_root / "checkpoint_complete.json").resolve())],
        "result_equals": {"backend": "bagel", "simulated": False},
    }
    timeout = float(bagel.get("timeout_s", config.get("smoke", {}).get("timeout_s", 3600)))
    poll = float(bagel.get("poll_interval_s", config.get("smoke", {}).get("poll_interval_s", 2)))
    submit_job(job_dir, spec)
    try:
        result = wait_for_job(job_dir, timeout_s=timeout, poll_interval_s=poll)
    except JobObservationTimeout:
        raise
    if int(result.get("real_optimizer_steps", 0)) <= 0:
        raise RealBackendError("bagel_training_reported_zero_optimizer_steps")
    status_path = job_dir / "status.json"
    status = read_json(status_path) if status_path.exists() else {"status": "observed_succeeded"}
    return {**result, "job_id": str(job_dir), "job_status": status}


def run_bagel_export(*, train_result: dict[str, Any], round_dir: Path,
                     config: dict[str, Any]) -> dict[str, Any]:
    bagel = _bagel_config(config)
    python = str(bagel.get("python_executable", "python"))
    vendor = Path(bagel["vendor_root"]).resolve()
    base = Path(bagel["model_path"]).resolve()
    checkpoint = Path(str(train_result["checkpoint_dir"])).resolve()
    output = (round_dir / "checkpoint" / "official_layout").resolve()
    result_path = (round_dir / "export_result.json").resolve()
    job_dir = (round_dir / "jobs" / "export").resolve()
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_bagel_export.py"
    spec = {
        "kind": "bagel_export",
        "argv": [python, str(script), "--vendor-root", str(vendor),
                 "--base-model-dir", str(base), "--checkpoint-dir", str(checkpoint),
                 "--output-dir", str(output), "--result-path", str(result_path),
                 "--python-executable", python],
        "cwd": str(Path(__file__).resolve().parents[2]),
        "result_path": str(result_path),
        "input_digests": {"train_result": digest(train_result), "checkpoint": train_result["checkpoint_digest"]},
        "required_artifacts": [str((output / "ema.safetensors").resolve())],
        "result_equals": {"backend": "bagel_export", "simulated": False},
    }
    submit_job(job_dir, spec)
    result = wait_for_job(job_dir, timeout_s=float(bagel.get("export_timeout_s", 3600)),
                          poll_interval_s=float(bagel.get("poll_interval_s", 2)))
    status_path = job_dir / "status.json"
    status = read_json(status_path) if status_path.exists() else {"status": "observed_succeeded"}
    return {**result, "job_id": str(job_dir), "job_status": status}


__all__ = ["RealBackendError", "run_bagel_train", "run_bagel_export"]
