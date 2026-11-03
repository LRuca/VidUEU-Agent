"""Collect machine-checkable acceptance reports for a real iteration run.

The collector is deliberately read-only with respect to training artifacts. It
only reads the run directory and writes report files beside it (or to an
explicit output directory). Missing evidence is reported as ``not_run`` rather
than inferred from a neighboring artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_TYPES = (
    "fine_grained_perception",
    "single_image_reasoning",
    "multi_image_logic",
    "localized_editing",
    "grounded_edit_to_select",
    "understand_recall_generate_select",
    "stepwise_sequence",
    "middle_step_gap",
    "instructional_future_sequence",
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def evidence(path: Path, root: Path) -> dict[str, Any]:
    return {"path": rel(path, root), "exists": path.is_file(), "sha256": sha256(path)}


def status(passed: bool, *, reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "passed" if passed else "not_run"}
    if reason and not passed:
        result["reason"] = reason
    return result


def json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def benchmark_gate(result: dict[str, Any] | None) -> dict[str, Any]:
    cases = (result or {}).get("benchmark_cases", [])
    complete = bool(cases) and all(
        not case.get("simulated", True)
        and case.get("expected") == case.get("inferred") == case.get("scored")
        and case.get("failed") == 0
        for case in cases
    )
    return {
        **status(complete, reason="missing_or_incomplete_real_cases" if not complete else None),
        "simulated": None if result is None else result.get("simulated"),
        "expected": None if result is None else result.get("expected"),
        "inferred": None if result is None else result.get("inferred"),
        "scored": None if result is None else result.get("scored"),
        "failed": None if result is None else result.get("failed"),
        "checkpoint_digest": None if result is None else result.get("checkpoint_digest"),
    }


def collect(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    output_dir = (output_dir or run_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_root / "final_summary.json"
    if read_json(final_path) is None:
        final_path = run_root / "summary.json"
    final = read_json(final_path)
    manifest = read_json(run_root / "loop_manifest.json")
    rounds: list[dict[str, Any]] = []
    for iteration in (1, 2):
        directory = run_root / f"iteration_{iteration:02d}"
        train = read_json(directory / "train_result.json")
        export = read_json(directory / "export_result.json")
        evaluation = read_json(directory / "eval_result.json")
        reflection = read_json(directory / "reflection.json")
        feedback = read_json(directory / "feedback.json")
        distribution = read_json(directory / "distribution_report.json")
        data_report = read_json(directory / "bagel_data_report.json")
        state = read_json(directory / "state.json")
        consumed = set((train or {}).get("training_audit", {}).get("consumed_task_types", []))
        complete_tasks = set((train or {}).get("training_audit", {}).get("complete_task_types", []))
        missing = sorted(set(TASK_TYPES) - consumed)
        rounds.append(
            {
                "iteration": iteration,
                "state": state,
                "actual_samples": (train or {}).get("episodes"),
                "train": {
                    "status": "passed"
                    if train and not train.get("simulated", True) and train.get("real_optimizer_steps", 0) > 0
                    else "not_run",
                    "simulated": None if train is None else train.get("simulated"),
                    "real_optimizer_steps": None if train is None else train.get("real_optimizer_steps"),
                    "consumed_task_types": sorted(consumed),
                    "missing_task_types": missing,
                    "complete_task_types": sorted(complete_tasks),
                    "missing_complete_task_types": sorted(set(TASK_TYPES) - complete_tasks),
                    "parameter_updates_verified": (train or {}).get("training_audit", {}).get(
                        "parameter_updates_verified"
                    ),
                },
                "export": {
                    "status": "passed"
                    if export and not export.get("simulated", True) and export.get("export_digest")
                    else "not_run",
                    "simulated": None if export is None else export.get("simulated"),
                    "checkpoint_digest": None if export is None else export.get("checkpoint_digest"),
                    "export_digest": None if export is None else export.get("export_digest"),
                    "required_files": [] if export is None else export.get("required_files", []),
                },
                "evaluation": benchmark_gate(evaluation),
                "reflection": {
                    "status": "passed" if reflection and reflection.get("backend") == "deepseek" else "not_run",
                    "backend": None if reflection is None else reflection.get("backend"),
                    "model_identity": None if reflection is None else reflection.get("model_identity"),
                    "policy_diff": {} if reflection is None else reflection.get("policy_diff", {}),
                    "input_feedback_digest": None if reflection is None else reflection.get("input_feedback_digest"),
                },
                "feedback": {
                    "status": "passed" if feedback and not feedback.get("simulated", True) else "not_run",
                    "checkpoint_digest": None if feedback is None else feedback.get("checkpoint_digest"),
                    "metrics": None if feedback is None else feedback.get("feedback_metrics"),
                },
                "distribution": {
                    "selected": None if distribution is None else distribution.get("total"),
                    "selection_status": None if distribution is None else distribution.get("selection_status"),
                    "unfilled_count": None if distribution is None else distribution.get("selection_unfilled_count"),
                    "task_counts": None if data_report is None else data_report.get("task_counts", {}),
                    "missing_task_types": None if data_report is None else data_report.get("missing_task_types", []),
                },
                "evidence": [
                    evidence(directory / name, run_root)
                    for name in (
                        "state.json",
                        "train_result.json",
                        "export_result.json",
                        "eval_result.json",
                        "feedback.json",
                        "reflection.json",
                        "distribution_report.json",
                        "bagel_data_report.json",
                    )
                ],
            }
        )

    all_consumed = sorted({task for item in rounds for task in item["train"]["consumed_task_types"]})
    all_evaluations = [benchmark_gate(read_json(run_root / "baseline" / "eval_result.json"))] + [
        item["evaluation"] for item in rounds
    ]
    same_init = bool(final and final.get("same_init_checkpoint_recorded"))
    loop_completed = bool(final and final.get("completed") and final.get("status") == "COMPLETED")
    real_training = all(item["train"]["status"] == "passed" for item in rounds)
    exports = all(item["export"]["status"] == "passed" for item in rounds)
    evaluations = all(item["evaluation"]["status"] == "passed" for item in rounds) and all_evaluations[0]["status"] == "passed"
    # A task name in a simulated or format-only artifact is not evidence that
    # the native optimizer consumed that task.  Coverage is gated on the same
    # real-training contract as the optimizer-step gate.
    nine_per_round = all(
        item["train"]["status"] == "passed"
        and not item["train"]["missing_task_types"]
        and not item["train"]["missing_complete_task_types"]
        and item["train"]["parameter_updates_verified"] is True
        for item in rounds
    )
    job_status = read_json(run_root / "jobs" / "full_loop" / "status.json")
    # A recovery probe is a self-contained persistent-job contract record.  It
    # is preferred when present because the probe may intentionally use a
    # small CPU job outside the full-loop attempts directory.
    recovery_probe_path = run_root / "recovery_checks" / "persistent_job_recovery.json"
    recovery_probe = read_json(recovery_probe_path)
    recovery_path = None
    recovery_attempt = None
    for path in sorted((run_root / "jobs" / "full_loop" / "attempts").glob("*.json")):
        attempt = read_json(path)
        if (attempt and job_status and attempt.get("status") == "failed"
                and isinstance(attempt.get("attempt"), int)
                and attempt["attempt"] < job_status.get("attempt", 0)
                and attempt.get("spec_digest") and attempt["spec_digest"] == job_status.get("spec_digest")):
            recovery_path, recovery_attempt = path, attempt
    legacy_recovery_passed = bool(
        job_status
        and job_status.get("status") == "succeeded"
        and recovery_attempt
        and recovery_attempt.get("status") == "failed"
        and recovery_attempt.get("spec_digest") == job_status.get("spec_digest")
    )
    probe_failed = (recovery_probe or {}).get("failed_attempt")
    probe_success = (recovery_probe or {}).get("successful_attempt")
    probe_passed = bool(
        recovery_probe
        and recovery_probe.get("status") == "passed"
        and recovery_probe.get("same_spec_digest") is True
        and recovery_probe.get("retry_failed_explicit") is True
        and isinstance(probe_failed, dict)
        and probe_failed.get("status") == "failed"
        and isinstance(probe_success, dict)
        and probe_success.get("status") == "succeeded"
        and probe_failed.get("spec_digest") == probe_success.get("spec_digest")
        and probe_failed.get("attempt", 0) < probe_success.get("attempt", 0)
        and recovery_probe.get("duplicate_successful_payload_runs") is False
    )
    recovery_passed = probe_passed or legacy_recovery_passed
    recovery_failed_attempt = probe_failed if probe_passed else recovery_attempt
    recovery_successful_attempt = probe_success if probe_passed else job_status

    report = {
        "schema_version": 1,
        "run_root": str(run_root),
        "generated_by": "scripts/collect_real_loop_report.py",
        "manifest": {
            "present": manifest is not None,
            "target_train_samples": None if manifest is None else manifest.get("target_train_samples"),
            "actual_sample_cap": None if manifest is None else manifest.get("actual_sample_cap"),
            "backends": None if manifest is None else manifest.get("backends"),
            "paper_claim": None if manifest is None else manifest.get("paper_claim"),
        },
        "gates": {
            "loop_lifecycle": {**status(loop_completed), "evidence": evidence(final_path, run_root)},
            "same_initial_checkpoint": status(same_init),
            "real_training": status(real_training, reason="one_or_more_rounds_not_real_or_zero_step"),
            "export_reload_artifacts": status(exports, reason="one_or_more_rounds_missing_real_export"),
            "real_benchmarks": status(evaluations, reason="one_or_more_benchmark_nodes_incomplete"),
            "persistent_job_recovery": status(recovery_passed, reason="missing_same_spec_failed_then_successful_job_evidence"),
            "nine_task_real_optimizer_coverage": (
                {"status": "passed"}
                if nine_per_round
                else {
                    "status": "failed",
                    "reason": "nine_task_coverage_missing_in_real_optimizer_consumption",
                }
            ),
            "paper_scale_and_effect": {"status": "not_run", "reason": "smoke_configuration"},
        },
        "task_coverage": {
            "required": list(TASK_TYPES),
            "run_union_consumed": all_consumed,
            "run_union_missing": sorted(set(TASK_TYPES) - set(all_consumed)),
            "per_round": [
                {
                    "iteration": item["iteration"],
                    "consumed": item["train"]["consumed_task_types"],
                    "missing": item["train"]["missing_task_types"],
                    "complete_tasks": item["train"]["complete_task_types"],
                    "missing_complete_tasks": item["train"]["missing_complete_task_types"],
                }
                for item in rounds
            ],
        },
        "baseline": {"evaluation": all_evaluations[0]},
        "rounds": rounds,
        "recovery": {
            "status": "passed" if recovery_passed else "not_run",
            "failed_attempt": recovery_failed_attempt,
            "successful_attempt": recovery_successful_attempt,
            "evidence": [
                *([evidence(recovery_probe_path, run_root)] if probe_passed else []),
                *([evidence(recovery_path, run_root)] if recovery_path else []),
                evidence(run_root / "jobs" / "full_loop" / "status.json", run_root),
            ],
        },
    }

    feedback = {
        "schema_version": 1,
        "run_root": str(run_root),
        "reflection_backends": [item["reflection"]["backend"] for item in rounds],
        "iteration_1_to_2": {
            "previous_feedback_consumed": bool((final or {}).get("rounds", [{} , {}])[1].get("previous_feedback_consumed"))
            if final and len(final.get("rounds", [])) > 1
            else False,
            "reflection1_policy_diff": rounds[0]["reflection"]["policy_diff"],
            "reflection1_policy_changed": bool(rounds[0]["reflection"]["policy_diff"]),
            "iteration2_policy_diff_from_previous": (final or {}).get("rounds", [{}, {}])[1].get(
                "policy_diff_from_previous_count"
            )
            if final and len(final.get("rounds", [])) > 1
            else None,
            "interpretation": "feedback was consumed; both reflections returned valid no-change policies"
            if not rounds[0]["reflection"]["policy_diff"] and not rounds[1]["reflection"]["policy_diff"]
            else "feedback was consumed and at least one policy field changed",
        },
        "control_execution_evidence": [
            {
                "iteration": item["iteration"],
                "policy_fields": ["capability", "domain", "task_type", "video_type_weights", "sampling_density"],
                "selection_report": f"iteration_{item['iteration']:02d}/distribution_report.json",
                "source_schedule": f"iteration_{item['iteration']:02d}/source_scheduler_report.json",
                "observed_selection_status": item["distribution"]["selection_status"],
                "observed_task_counts": item["distribution"]["task_counts"],
            }
            for item in rounds
        ],
        "evidence": [
            "iteration_01/reflection.json",
            "iteration_02/reflection.json",
            "iteration_02/round_inputs.json",
            "iteration_02/policy.json",
            "iteration_02/config_overlay.json",
        ],
    }

    recovery = report["recovery"]
    report_path = output_dir / "report.json"
    feedback_path = output_dir / "feedback_consumption_report.json"
    recovery_path = output_dir / "recovery_checks.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    feedback_path.write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    recovery_path.write_text(json.dumps(recovery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Real Loop Acceptance Report",
        "",
        f"Run: `{run_root}`",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    for name, gate in report["gates"].items():
        lines.append(f"| {name} | {gate['status']} |")
    lines += [
        "",
        f"Real optimizer task union: {', '.join(all_consumed) or 'none'}",
        f"Missing from union: {', '.join(report['task_coverage']['run_union_missing']) or 'none'}",
        "",
        "Only gates explicitly marked passed have supporting evidence. Task IDs alone do not establish complete episode consumption. Paper-scale training and performance improvement are outside this smoke run.",
        "",
        "## Evidence",
        "",
        "- `final_summary.json`",
        "- `loop_manifest.json`",
        "- `baseline/eval_result.json`",
        "- `iteration_01/` and `iteration_02/` stage artifacts",
        "- `jobs/full_loop/attempts/` and `jobs/full_loop/status.json`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(report_path), "feedback": str(feedback_path), "recovery": str(recovery_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.run_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
