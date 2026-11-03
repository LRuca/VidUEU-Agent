"""Command-backed benchmark evaluation for the real iteration loop.

The benchmark projects remain external, read-only inputs.  This adapter only
starts explicitly configured commands and validates the JSON result they
publish.  It never fabricates a score when a command, case, or metric is
missing.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

from .artifacts import content_identity, publish_immutable_json
from .feedback import FEEDBACK_BENCHMARKS, HELD_OUT_BENCHMARKS, METRIC_UNITS, validate_feedback
from .jobs import JobFailed, JobObservationTimeout, submit_job, wait_for_job
from .state import digest, read_json, write_json


class BenchmarkBackendError(RuntimeError):
    """A benchmark command completed without satisfying its result contract."""


def _sha256_path(path: Path) -> str:
    if path.is_file():
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    if path.is_dir():
        entries = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            value = hashlib.sha256()
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            entries.append({"path": child.relative_to(path).as_posix(), "sha256": value.hexdigest()})
        if not entries:
            raise BenchmarkBackendError(f"checkpoint_empty:{path}")
        return digest(entries)
    raise FileNotFoundError(path)


def _benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("benchmark")
    if not isinstance(value, dict):
        evaluator = config.get("evaluator")
        if isinstance(evaluator, dict):
            value = evaluator.get("benchmark")
    if not isinstance(value, dict):
        raise ValueError("benchmark_config_missing")
    return value


def _as_path(value: Any, *, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(os.path.expandvars(str(value)))
    return path if path.is_absolute() else (base / path).resolve()


def _safe_template(value: str, fields: dict[str, str]) -> str:
    # Formatting is deliberately restricted to known placeholders.  A command
    # containing a secret-like literal is rejected before it is persisted.
    if "sk-" in value or "api_key=" in value.lower() or "authorization:" in value.lower():
        raise ValueError("benchmark_command_must_not_contain_secret")
    try:
        result = value.format_map(fields)
    except KeyError as exc:
        raise ValueError(f"benchmark_unknown_command_placeholder:{exc.args[0]}") from None
    if "{" in result or "}" in result:
        raise ValueError("benchmark_unexpanded_command_placeholder")
    return result


def _command_for(
    spec: dict[str, Any],
    *,
    benchmark: str,
    checkpoint: Path,
    checkpoint_digest: str,
    subset: Path | None,
    output: Path,
    result: Path,
    iteration: int,
    root: Path,
) -> tuple[list[str], Path]:
    raw = spec.get("argv", spec.get("command"))
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) or not item for item in raw):
        raise ValueError(f"benchmark_command_missing:{benchmark}")
    fields = {
        "checkpoint": str(checkpoint),
        "checkpoint_digest": checkpoint_digest,
        "subset": str(subset) if subset else "",
        "output": str(output),
        "output_dir": str(output.parent),
        "result": str(result),
        "iteration": str(iteration),
        "benchmark": benchmark,
    }
    command = [_safe_template(item, fields) for item in raw]
    cwd = _as_path(spec.get("cwd"), base=root) or root
    if not cwd.is_dir():
        raise FileNotFoundError(f"benchmark_cwd_missing:{cwd}")
    return command, cwd


def _read_metric(value: dict[str, Any], benchmark: str) -> tuple[float, str]:
    expected_unit = METRIC_UNITS[benchmark]
    metric = value.get("metric")
    expected_name = {"MMDU": "aggregate", "GEdit-Bench-En": "G_O", "WEAVEbench": "Overall"}[benchmark]
    if not isinstance(metric, dict) or metric.get("name") != expected_name:
        raise BenchmarkBackendError(f"benchmark_metric_definition_mismatch:{benchmark}")
    found, unit = metric.get("value"), metric.get("unit")
    if isinstance(found, bool) or not isinstance(found, (int, float)):
        raise BenchmarkBackendError(f"benchmark_metric_missing:{benchmark}")
    if unit != expected_unit:
        raise BenchmarkBackendError(f"benchmark_metric_unit_mismatch:{benchmark}:{unit}")
    if not math.isfinite(found):
        raise BenchmarkBackendError(f"benchmark_metric_nonfinite:{benchmark}")
    maximum = {"MMDU": 100.0, "GEdit-Bench-En": 10.0, "WEAVEbench": 1.0}[benchmark]
    if float(found) < 0 or float(found) > maximum:
        raise BenchmarkBackendError(f"benchmark_metric_out_of_range:{benchmark}")
    return float(found), expected_unit


def _validate_case_counts(value: dict[str, Any], benchmark: str) -> dict[str, int]:
    counts = {}
    for key in ("expected", "inferred", "scored", "failed"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise BenchmarkBackendError(f"benchmark_count_missing:{benchmark}:{key}")
        counts[key] = number
    if counts["inferred"] + counts["failed"] != counts["expected"]:
        raise BenchmarkBackendError(f"benchmark_count_inconsistent:{benchmark}:inferred_failed")
    if counts["scored"] > counts["inferred"]:
        raise BenchmarkBackendError(f"benchmark_count_inconsistent:{benchmark}:scored")
    if counts["expected"] <= 0:
        raise BenchmarkBackendError(f"benchmark_empty_subset:{benchmark}")
    if counts["failed"] or counts["scored"] != counts["expected"]:
        raise BenchmarkBackendError(f"benchmark_subset_incomplete:{benchmark}")
    return counts


def benchmark_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze subset, prompts, decoding, judge, and adapter code identities."""
    value = _benchmark_config(config)
    commands = value.get("commands")
    if not isinstance(commands, dict) or set(commands) != set(FEEDBACK_BENCHMARKS):
        raise ValueError("benchmark_requires_three_commands")
    identities = {}
    for name, spec in commands.items():
        if not isinstance(spec, dict):
            raise ValueError(f"benchmark_command_missing:{name}")
        if not isinstance(spec.get("argv"), list) or not spec["argv"]:
            raise ValueError(f"benchmark_command_missing:{name}")
        subset = _as_path(spec.get("subset"), base=Path.cwd())
        if subset is None or not subset.is_file():
            raise FileNotFoundError(f"benchmark_subset_missing:{name}:{subset}")
        case_ids = spec.get("case_ids")
        if (not isinstance(case_ids, list) or not case_ids
                or any(not isinstance(key, str) or not key for key in case_ids)
                or len(set(case_ids)) != len(case_ids)):
            raise ValueError(f"benchmark_case_ids_required:{name}")
        source_files = spec.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise ValueError(f"benchmark_source_files_required:{name}")
        sources = [content_identity(Path(path)) for path in source_files]
        media_files = spec.get("media_files")
        if not isinstance(media_files, list) or not media_files:
            raise ValueError(f"benchmark_media_files_required:{name}")
        media = [content_identity(Path(path)) for path in media_files]
        # Judge secrets are inherited via env/key files; only identity and
        # prompt/decode definitions belong in this public protocol.
        for key in ("judge", "prompt_version", "decoding"):
            if not spec.get(key):
                raise ValueError(f"benchmark_protocol_field_required:{name}:{key}")
        identities[name] = {
            "config": spec, "subset": content_identity(subset), "sources": sources, "media": media,
            "case_ids": case_ids,
        }
    return {"schema_version": 1, "benchmarks": identities,
            "metric_units": dict(METRIC_UNITS)}


def _run_one(
    *,
    benchmark: str,
    spec: dict[str, Any],
    checkpoint: Path,
    checkpoint_digest: str,
    iteration: int,
    eval_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    subset = _as_path(spec.get("subset"), base=Path.cwd())
    if subset is not None and not subset.exists():
        raise FileNotFoundError(f"benchmark_subset_missing:{benchmark}:{subset}")
    out_dir = eval_root / benchmark.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "raw_output.json"
    result_path = out_dir / "result.json"
    command, cwd = _command_for(
        spec,
        benchmark=benchmark,
        checkpoint=checkpoint,
        checkpoint_digest=checkpoint_digest,
        subset=subset,
        output=output,
        result=result_path,
        iteration=iteration,
        root=eval_root,
    )
    job_dir = eval_root / "jobs" / benchmark.replace("/", "_")
    job_spec = {
        "kind": f"benchmark_{benchmark}",
        "argv": command,
        "cwd": str(cwd),
        "result_path": str(result_path),
        "input_digests": {
            "checkpoint": checkpoint_digest,
            "subset": content_identity(subset) if subset else None,
            "config": digest({"benchmark": benchmark, "spec": spec, "iteration": iteration}),
        },
        "required_artifacts": [str(output)] if spec.get("raw_output_required", False) else [],
        "result_equals": {"benchmark": benchmark, "simulated": False,
                          "checkpoint_digest": checkpoint_digest},
    }
    # A controller restart or transient evaluator failure must be recoverable
    # at the adapter boundary.  Observation timeouts reattach to the same
    # running job; a terminal failed job is explicitly retried, which creates a
    # new persisted attempt without duplicating a successful result.
    max_attempts = spec.get("max_attempts", config.get("evaluator", {}).get("max_attempts", 2))
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError(f"benchmark_invalid_max_attempts:{benchmark}")
    result = None
    for attempt_index in range(max_attempts):
        submit_job(job_dir, job_spec, retry_failed=attempt_index > 0)
        try:
            result = wait_for_job(
                job_dir,
                timeout_s=float(spec.get("timeout_s", config.get("evaluator", {}).get("timeout_s", 3600))),
                poll_interval_s=float(spec.get("poll_interval_s", config.get("evaluator", {}).get("poll_interval_s", 2))),
            )
            break
        except JobObservationTimeout:
            if attempt_index + 1 >= max_attempts:
                raise
            # The next submit is idempotent while the original worker remains
            # running, so this path never launches a duplicate evaluator.
        except JobFailed:
            if attempt_index + 1 >= max_attempts:
                raise
            # The next iteration passes retry_failed=True and archives any
            # failed attempt's outputs before launching the replacement.
    if not isinstance(result, dict):
        raise BenchmarkBackendError(f"benchmark_result_not_object:{benchmark}")
    counts = _validate_case_counts(result, benchmark)
    if result.get("simulated") is not False or result.get("checkpoint_digest") != checkpoint_digest:
        raise BenchmarkBackendError(f"benchmark_checkpoint_or_backend_mismatch:{benchmark}")
    expected_ids = spec["case_ids"]
    if sorted(result.get("case_ids", [])) != sorted(expected_ids) or counts["expected"] != len(expected_ids):
        raise BenchmarkBackendError(f"benchmark_case_ids_mismatch:{benchmark}")
    score, unit = _read_metric(result, benchmark)
    return {
        "benchmark": benchmark,
        "simulated": False,
        "checkpoint_digest": checkpoint_digest,
        "expected": counts["expected"],
        "inferred": counts["inferred"],
        "scored": counts["scored"],
        "failed": counts["failed"],
        "metric": score,
        "metric_unit": unit,
        "raw_result_path": str(result_path),
        "raw_output_path": str(output) if output.exists() else None,
        "job_id": str(job_dir),
        "job_attempt": int((read_json(job_dir / "status.json") or {}).get("attempt", 0)),
        "model_identity": result.get("model_identity"),
        "subcategory_scores": result.get("subcategory_scores", {}),
        "judge_identity": result.get("judge_identity"),
    }


def run_benchmark_evaluation(
    *,
    checkpoint: Path,
    round_dir: Path,
    iteration: int,
    config: dict[str, Any],
    checkpoint_digest: str | None = None,
) -> dict[str, Any]:
    """Run the fixed feedback subset for all three real benchmarks."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"benchmark_checkpoint_missing:{checkpoint}")
    actual_digest = content_identity(checkpoint, exclude=("export_manifest.json",))["sha256"]
    if checkpoint_digest is not None and checkpoint_digest != actual_digest:
        raise BenchmarkBackendError("benchmark_checkpoint_digest_mismatch")
    checkpoint_digest = actual_digest
    benchmark_cfg = _benchmark_config(config)
    protocol = benchmark_protocol(config)
    commands = benchmark_cfg.get("commands")
    if not isinstance(commands, dict):
        raise ValueError("benchmark_commands_missing")
    eval_root = (round_dir / "eval").resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    publish_immutable_json(eval_root / "protocol.json", protocol)
    rows = []
    for benchmark in FEEDBACK_BENCHMARKS:
        spec = commands.get(benchmark)
        if not isinstance(spec, dict):
            raise ValueError(f"benchmark_command_missing:{benchmark}")
        rows.append(_run_one(
            benchmark=benchmark,
            spec=spec,
            checkpoint=checkpoint,
            checkpoint_digest=checkpoint_digest,
            iteration=iteration,
            eval_root=eval_root,
            config=config,
        ))
    metrics = {row["benchmark"]: row["metric"] for row in rows}
    units = {row["benchmark"]: row["metric_unit"] for row in rows}
    result = {
        "schema_version": 1,
        "iteration": iteration,
        "backend": "benchmark",
        "simulated": False,
        "checkpoint_digest": checkpoint_digest,
        "protocol_digest": digest(protocol),
        "feedback_metrics": metrics,
        "metric_units": units,
        "subcategory_scores": {row["benchmark"]: row["subcategory_scores"] for row in rows},
        "held_out_metrics": {name: None for name in HELD_OUT_BENCHMARKS},
        "benchmark_cases": rows,
        "expected": sum(row["expected"] for row in rows),
        "inferred": sum(row["inferred"] for row in rows),
        "scored": sum(row["scored"] for row in rows),
        "failed": sum(row["failed"] for row in rows),
        "loss_curve_summary": {"available": False},
    }
    validate_feedback(result)
    write_json(round_dir / "eval_result.json", result)
    write_json(round_dir / "feedback.json", result)
    return result


__all__ = ["BenchmarkBackendError", "run_benchmark_evaluation"]
