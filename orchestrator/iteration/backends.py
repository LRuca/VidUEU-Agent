"""Training/evaluation adapters for dry-run and smoke loop deployments."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from .feedback import METRIC_UNITS, fixture_feedback
from .smoke_runtime import submit_smoke_job, wait_for_smoke_job
from .state import digest, read_json, write_json


def dry_run_train(train_manifest: Path, *, iteration: int, config: dict[str, Any]) -> dict[str, Any]:
    target = int(config["loop"]["target_train_samples"])
    actual = len(train_manifest.read_text(encoding="utf-8").splitlines()) if train_manifest.exists() else 0
    return {
        "iteration": iteration,
        "backend": "dry_run",
        "model_name": config["trainer"]["model_name"],
        "init_checkpoint": config["trainer"]["init_checkpoint"],
        "epochs": int(config["trainer"].get("epochs", 1)),
        "target_samples": target,
        "actual_samples": actual,
        "optimizer_steps": 0,
        "checkpoint_loaded": False,
        "output_checkpoint": None,
    }


def smoke_train(
    train_manifest: Path,
    *,
    iteration: int,
    round_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Simulate asynchronous Bagel training and produce a fake checkpoint."""
    smoke = config.get("smoke", {}) if isinstance(config.get("smoke"), dict) else {}
    input_digest = hashlib.sha256(train_manifest.read_bytes()).hexdigest()
    job_path = round_dir / "train_job.json"
    job = submit_smoke_job(
        job_path,
        job_id=f"train_iter_{iteration:02d}",
        kind="train",
        iteration=iteration,
        duration_s=float(smoke.get("train_duration_s", 0.2)),
        input_digest=input_digest,
    )
    completed = wait_for_smoke_job(
        job_path,
        timeout_s=float(smoke.get("timeout_s", 120.0)),
        poll_interval_s=float(smoke.get("poll_interval_s", 0.05)),
    )
    checkpoint_path = round_dir / "simulated_checkpoint.json"
    checkpoint = {
        "schema_version": 1,
        "simulated": True,
        "iteration": iteration,
        "init_checkpoint": config.get("trainer", {}).get("init_checkpoint"),
        "train_manifest_digest": input_digest,
        "job_id": completed["job_id"],
    }
    checkpoint["checkpoint_digest"] = digest(checkpoint)
    write_json(checkpoint_path, checkpoint)
    return {
        "iteration": iteration,
        "backend": "smoke_simulator",
        "simulated": True,
        "model_name": config["trainer"]["model_name"],
        "init_checkpoint": config["trainer"]["init_checkpoint"],
        "epochs": int(config["trainer"].get("epochs", 1)),
        "target_samples": int(config["loop"]["target_train_samples"]),
        "actual_samples": len(train_manifest.read_text(encoding="utf-8").splitlines()),
        "optimizer_steps": 0,
        "optimizer_steps_simulated": 1,
        "real_optimizer_steps": 0,
        "checkpoint_loaded": False,
        "output_checkpoint": str(checkpoint_path),
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "train_job_id": completed["job_id"],
        "train_job_poll_count": int(completed.get("poll_count", 0) or 0),
        "train_manifest_digest": input_digest,
    }


def evaluate_fixture(*, iteration: int, distribution: dict[str, Any]) -> dict[str, Any]:
    return fixture_feedback(iteration, distribution)


def smoke_evaluate(
    *,
    iteration: int,
    distribution: dict[str, Any],
    train_result: dict[str, Any],
    policy: dict[str, Any] | None,
    round_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Simulate an evaluator that waits on and consumes the smoke checkpoint."""
    checkpoint_digest = train_result.get("checkpoint_digest")
    if not checkpoint_digest or not bool(train_result.get("simulated")):
        raise ValueError("smoke_evaluator_requires_simulated_checkpoint")
    smoke = config.get("smoke", {}) if isinstance(config.get("smoke"), dict) else {}
    input_digest = digest({
        "checkpoint_digest": checkpoint_digest,
        "train_manifest_digest": train_result.get("train_manifest_digest"),
        "iteration_policy_digest": digest(policy or {}),
        "distribution": distribution,
    })
    job_path = round_dir / "eval_job.json"
    job = submit_smoke_job(
        job_path,
        job_id=f"eval_iter_{iteration:02d}",
        kind="evaluate",
        iteration=iteration,
        duration_s=float(smoke.get("eval_duration_s", 0.1)),
        input_digest=input_digest,
    )
    completed = wait_for_smoke_job(
        job_path,
        timeout_s=float(smoke.get("timeout_s", 120.0)),
        poll_interval_s=float(smoke.get("poll_interval_s", 0.05)),
    )
    seed = int(smoke.get("deterministic_seed", 20261018)) + int(iteration)
    rng = random.Random(seed)
    result = {
        "iteration": iteration,
        "backend": "smoke_simulator",
        "simulated": True,
        "feedback_metrics": {
            "MMDU": round(28.0 + rng.random() * 6.0, 3),
            "GEdit-Bench-En": round(6.4 + rng.random() * 0.5, 3),
            "WEAVEbench": round(0.42 + rng.random() * 0.10, 3),
        },
        "held_out_metrics": {"MMBench": None, "MMStar": None, "VSI-Bench": None},
        "metric_units": dict(METRIC_UNITS),
        "distribution": distribution,
        "loss_curve_summary": {"available": False},
        "checkpoint_digest": checkpoint_digest,
        "iteration_policy_digest": digest(policy or {}),
        "consumed_train_result_digest": digest(train_result),
        "evaluator_input_digest": input_digest,
        "metric_seed": seed,
        "eval_job_id": completed["job_id"],
        "eval_job_poll_count": int(completed.get("poll_count", 0) or 0),
    }
    return result
