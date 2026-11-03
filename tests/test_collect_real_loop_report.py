import json
from pathlib import Path

from scripts.collect_real_loop_report import TASK_TYPES, collect


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _base_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    _write_json(
        root / "loop_manifest.json",
        {
            "target_train_samples": 10000,
            "actual_sample_cap": 32,
            "backends": {"trainer": "bagel", "evaluator": "benchmark"},
            "paper_claim": "smoke_chain_only",
        },
    )
    return root


def _report(root: Path) -> dict:
    return json.loads((root / "report.json").read_text(encoding="utf-8"))


def test_missing_artifacts_are_not_reported_as_passed(tmp_path):
    root = _base_run(tmp_path)
    collect(root)
    report = _report(root)

    assert report["gates"]["loop_lifecycle"]["status"] == "not_run"
    assert report["gates"]["real_training"]["status"] == "not_run"
    assert report["gates"]["real_benchmarks"]["status"] == "not_run"
    assert report["gates"]["nine_task_real_optimizer_coverage"]["status"] == "failed"
    assert report["task_coverage"]["run_union_missing"] == sorted(TASK_TYPES)


def test_simulated_training_and_eval_cannot_pass_real_gates(tmp_path):
    root = _base_run(tmp_path)
    for iteration in (1, 2):
        directory = root / f"iteration_{iteration:02d}"
        _write_json(directory / "train_result.json", {
            "simulated": True,
            "real_optimizer_steps": 2,
            "episodes": 3,
            "training_audit": {"consumed_task_types": list(TASK_TYPES)},
        })
        _write_json(directory / "eval_result.json", {
            "simulated": True,
            "benchmark_cases": [{
                "simulated": True, "expected": 1, "inferred": 1,
                "scored": 1, "failed": 0,
            }],
        })
    _write_json(root / "final_summary.json", {
        "status": "COMPLETED", "completed": True,
        "same_init_checkpoint_recorded": True,
        "rounds": [{}, {}],
    })

    collect(root)
    report = _report(root)
    assert report["gates"]["loop_lifecycle"]["status"] == "passed"
    assert report["gates"]["real_training"]["status"] == "not_run"
    assert report["gates"]["real_benchmarks"]["status"] == "not_run"
    assert report["gates"]["nine_task_real_optimizer_coverage"]["status"] == "failed"


def test_nine_task_gate_uses_actual_optimizer_consumption(tmp_path):
    root = _base_run(tmp_path)
    consumed = list(TASK_TYPES[:-1])
    for iteration in (1, 2):
        directory = root / f"iteration_{iteration:02d}"
        _write_json(directory / "train_result.json", {
            "simulated": False,
            "real_optimizer_steps": 2,
            "episodes": len(consumed),
            "training_audit": {
                "consumed_task_types": consumed,
                "parameter_updates_verified": True,
            },
        })
    _write_json(root / "final_summary.json", {
        "status": "COMPLETED", "completed": True,
        "same_init_checkpoint_recorded": True, "rounds": [{}, {}],
    })

    collect(root)
    report = _report(root)
    gate = report["gates"]["nine_task_real_optimizer_coverage"]
    assert gate["status"] == "failed"
    assert report["task_coverage"]["run_union_missing"] == [TASK_TYPES[-1]]
    assert report["task_coverage"]["per_round"][0]["missing"] == [TASK_TYPES[-1]]


def test_feedback_consumption_and_no_change_reflection_are_distinguished(tmp_path):
    root = _base_run(tmp_path)
    for iteration in (1, 2):
        directory = root / f"iteration_{iteration:02d}"
        _write_json(directory / "reflection.json", {
            "backend": "deepseek",
            "model_identity": "deepseek-flash",
            "policy_diff": {},
            "input_feedback_digest": f"feedback-{iteration}",
        })
        _write_json(directory / "feedback.json", {
            "simulated": False,
            "feedback_metrics": {"MMDU": 1.0},
        })
        _write_json(directory / "distribution_report.json", {
            "total": 1, "selection_status": "ok", "selection_unfilled_count": 0,
        })
        _write_json(directory / "bagel_data_report.json", {
            "task_counts": {}, "missing_task_types": list(TASK_TYPES),
        })
    _write_json(root / "iteration_02" / "round_inputs.json", {
        "previous_feedback_consumed": True,
    })
    _write_json(root / "iteration_02" / "policy.json", {"iteration": 2})
    _write_json(root / "iteration_02" / "config_overlay.json", {})
    _write_json(root / "final_summary.json", {
        "status": "COMPLETED", "completed": True,
        "same_init_checkpoint_recorded": True,
        "rounds": [
            {"previous_feedback_consumed": False},
            {"previous_feedback_consumed": True,
             "policy_diff_from_previous_count": 0},
        ],
    })

    collect(root)
    report = _report(root)
    feedback = json.loads((root / "feedback_consumption_report.json").read_text(encoding="utf-8"))
    assert feedback["iteration_1_to_2"]["previous_feedback_consumed"] is True
    assert feedback["iteration_1_to_2"]["reflection1_policy_changed"] is False
    assert "no-change" in feedback["iteration_1_to_2"]["interpretation"]
    assert report["rounds"][1]["reflection"]["status"] == "passed"


def test_nine_tasks_need_complete_episodes_beyond_prefixes(tmp_path):
    root = _base_run(tmp_path)
    for i in (1, 2):
        _write_json(root / f"iteration_{i:02d}/train_result.json", {
            "simulated": False, "real_optimizer_steps": 3,
            "training_audit": {"consumed_task_types": list(TASK_TYPES),
                               "complete_task_types": list(TASK_TYPES[:-1]), "parameter_updates_verified": True}})
    collect(root)
    assert _report(root)["gates"]["nine_task_real_optimizer_coverage"]["status"] == "failed"
    for i in (1, 2):
        path = root / f"iteration_{i:02d}/train_result.json"
        row = json.loads(path.read_text())
        row["training_audit"]["complete_task_types"] = list(TASK_TYPES)
        _write_json(path, row)
    collect(root)
    assert _report(root)["gates"]["nine_task_real_optimizer_coverage"]["status"] == "passed"


def test_recovery_uses_actual_attempts_and_requires_matching_job_identity(tmp_path):
    root = _base_run(tmp_path)
    _write_json(root / "jobs/full_loop/attempts/001.json", {
        "attempt": 1, "status": "failed", "spec_digest": "job-A"})
    _write_json(root / "jobs/full_loop/status.json", {
        "attempt": 2, "status": "succeeded", "spec_digest": "job-B"})
    collect(root)
    assert _report(root)["gates"]["persistent_job_recovery"]["status"] == "not_run"
    _write_json(root / "jobs/full_loop/status.json", {
        "attempt": 2, "status": "succeeded", "spec_digest": "job-A"})
    collect(root)
    assert _report(root)["gates"]["persistent_job_recovery"]["status"] == "passed"


def test_recovery_probe_is_accepted_without_full_loop_attempts(tmp_path):
    root = _base_run(tmp_path)
    _write_json(root / "recovery_checks/persistent_job_recovery.json", {
        "status": "passed",
        "same_spec_digest": True,
        "retry_failed_explicit": True,
        "duplicate_successful_payload_runs": False,
        "failed_attempt": {"attempt": 1, "status": "failed", "spec_digest": "probe-A"},
        "successful_attempt": {"attempt": 2, "status": "succeeded", "spec_digest": "probe-A"},
    })

    collect(root)
    report = _report(root)
    assert report["gates"]["persistent_job_recovery"]["status"] == "passed"
    assert report["recovery"]["successful_attempt"]["attempt"] == 2
    assert any(
        item["path"].replace("\\", "/") == "recovery_checks/persistent_job_recovery.json"
        for item in report["recovery"]["evidence"]
    )
