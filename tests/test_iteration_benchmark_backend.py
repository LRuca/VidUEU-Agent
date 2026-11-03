"""CPU contract tests; these workers do not count as real benchmark evidence."""
import json
import sys
from pathlib import Path

import pytest

from orchestrator.iteration.artifacts import content_identity
from orchestrator.iteration.benchmark_backends import (
    BenchmarkBackendError,
    benchmark_protocol,
    run_benchmark_evaluation,
)
from orchestrator.iteration.controller import IterationController


def _config(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "name, checksum, subset, output, result = sys.argv[1:]\n"
        "row = json.loads(Path(subset).read_text())\n"
        "scores = {'MMDU': ('aggregate', 30.0, 'percent'), "
        "'GEdit-Bench-En': ('G_O', 6.5, 'score_0_10'), "
        "'WEAVEbench': ('Overall', 0.4, 'fraction')}\n"
        "key, value, unit = scores[name]\n"
        "data = {'benchmark': name, 'simulated': False, 'checkpoint_digest': checksum, "
        "'case_ids': row['ids'], 'expected': len(row['ids']), 'inferred': len(row['ids']), "
        "'scored': len(row['ids']), 'failed': 0, "
        "'metric': {'name': key, 'value': value, 'unit': unit}}\n"
        "data.update(row.get('overrides', {}))\n"
        "Path(output).write_text(json.dumps({'cpu_contract_test': True}))\n"
        "Path(result).write_text(json.dumps(data))\n"
        "with open(Path(subset).with_suffix('.calls'), 'a') as f: f.write('call\\n')\n",
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.txt").write_text("test-only-weight", encoding="utf-8")
    commands = {}
    media = tmp_path / "media.bin"
    media.write_bytes(b"cpu-contract-test-only")
    for name in ("MMDU", "GEdit-Bench-En", "WEAVEbench"):
        subset = tmp_path / f"{name}.json"
        ids = [f"{name}:0", f"{name}:1"] if name == "GEdit-Bench-En" else [f"{name}:0"]
        subset.write_text(json.dumps({"ids": ids}), encoding="utf-8")
        commands[name] = {
            "argv": [sys.executable, str(worker), name, "{checkpoint_digest}", "{subset}", "{output}", "{result}"],
            "cwd": str(tmp_path), "subset": str(subset), "case_ids": ids,
            "source_files": [str(worker)], "judge": {"model": "cpu-contract-test"},
            "media_files": [str(media)],
            "prompt_version": "test-only", "decoding": {"seed": 1},
            "timeout_s": 10, "poll_interval_s": 0.02, "raw_output_required": True,
        }
    return {
        "loop": {"rounds": 1, "target_train_samples": 1, "actual_sample_cap": 1},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": str(model)},
        "bagel": {"model_path": str(model)},
        "benchmark": {"commands": commands},
    }


def test_benchmark_jobs_reuse_validated_results_without_repeating_commands(tmp_path):
    config = _config(tmp_path)
    model = Path(config["bagel"]["model_path"])
    result = run_benchmark_evaluation(checkpoint=model, round_dir=tmp_path / "run", iteration=0, config=config)
    assert result["expected"] == result["inferred"] == result["scored"] == 4
    assert result["failed"] == 0
    assert result["feedback_metrics"]["GEdit-Bench-En"] == 6.5
    assert result["checkpoint_digest"] == content_identity(model)["sha256"]
    repeated = run_benchmark_evaluation(checkpoint=model, round_dir=tmp_path / "run", iteration=0, config=config)
    assert repeated == result
    for spec in config["benchmark"]["commands"].values():
        assert Path(spec["subset"]).with_suffix(".calls").read_text().splitlines() == ["call"]


def test_failed_evaluator_attempt_is_retried_without_duplicate_success(tmp_path):
    config = _config(tmp_path)
    worker = tmp_path / "retry_worker.py"
    worker.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "name, checksum, subset, output, result = sys.argv[1:]\n"
        "marker = Path(subset).with_suffix('.failed_once')\n"
        "if name == 'MMDU' and not marker.exists():\n"
        "    marker.write_text('failed')\n"
        "    raise SystemExit(7)\n"
        "row = json.loads(Path(subset).read_text())\n"
        "scores = {'MMDU': ('aggregate', 30.0, 'percent'), 'GEdit-Bench-En': ('G_O', 6.5, 'score_0_10'), 'WEAVEbench': ('Overall', 0.4, 'fraction')}\n"
        "key, value, unit = scores[name]\n"
        "data = {'benchmark': name, 'simulated': False, 'checkpoint_digest': checksum, 'case_ids': row['ids'], 'expected': len(row['ids']), 'inferred': len(row['ids']), 'scored': len(row['ids']), 'failed': 0, 'metric': {'name': key, 'value': value, 'unit': unit}}\n"
        "Path(output).write_text(json.dumps({'retry_contract_test': True}))\n"
        "Path(result).write_text(json.dumps(data))\n",
        encoding="utf-8",
    )
    for spec in config["benchmark"]["commands"].values():
        spec["argv"][1] = str(worker)

    result = run_benchmark_evaluation(
        checkpoint=Path(config["bagel"]["model_path"]),
        round_dir=tmp_path / "retry_run",
        iteration=0,
        config=config,
    )
    assert result["failed"] == 0
    mmdu_job = tmp_path / "retry_run" / "eval" / "jobs" / "MMDU"
    state = json.loads((mmdu_job / "status.json").read_text(encoding="utf-8"))
    assert state["status"] == "succeeded"
    assert state["attempt"] == 2
    assert result["benchmark_cases"][0]["job_attempt"] == 2


@pytest.mark.parametrize("overrides,reason", [
    ({"expected": 2, "failed": 1}, "incomplete"),
    ({"expected": 2}, "inconsistent"),
    ({"scored": 0}, "incomplete"),
    ({"case_ids": ["wrong"]}, "case_ids_mismatch"),
    ({"metric": {"name": "Average", "value": 6.5, "unit": "score_0_10"}}, "definition_mismatch"),
    ({"metric": {"name": "aggregate", "value": 0.3, "unit": "fraction"}}, "unit_mismatch"),
])
def test_benchmark_rejects_incomplete_or_semantically_wrong_results(tmp_path, overrides, reason):
    config = _config(tmp_path)
    subset = Path(config["benchmark"]["commands"]["MMDU"]["subset"])
    row = json.loads(subset.read_text())
    row["overrides"] = overrides
    subset.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(BenchmarkBackendError, match=reason):
        run_benchmark_evaluation(checkpoint=Path(config["bagel"]["model_path"]),
                                 round_dir=tmp_path / "run", iteration=0, config=config)
    assert not (tmp_path / "run" / "feedback.json").exists()


def test_benchmark_invalidates_changed_subset_and_model_content(tmp_path):
    config = _config(tmp_path)
    protocol = benchmark_protocol(config)
    subset = Path(config["benchmark"]["commands"]["MMDU"]["subset"])
    subset.write_text('{"ids": ["different-id"]}', encoding="utf-8")
    assert benchmark_protocol(config) != protocol
    model = Path(config["bagel"]["model_path"])
    old_digest = content_identity(model)["sha256"]
    (model / "weights.txt").write_text("changed-weights", encoding="utf-8")
    with pytest.raises(BenchmarkBackendError, match="checkpoint_digest_mismatch"):
        run_benchmark_evaluation(checkpoint=model, checkpoint_digest=old_digest,
                                 round_dir=tmp_path / "run", iteration=0, config=config)


def test_model_identity_excludes_huggingface_lock_cache_but_keeps_weights(tmp_path):
    config = _config(tmp_path)
    model = Path(config["bagel"]["model_path"])
    before = content_identity(model)
    lock = model / ".cache" / "huggingface" / "download" / "README.md.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    assert content_identity(model) == before
    (model / "weights.txt").write_text("new-weights", encoding="utf-8")
    assert content_identity(model) != before


def test_benchmark_protocol_changes_when_media_content_changes(tmp_path):
    config = _config(tmp_path)
    before = benchmark_protocol(config)
    media = Path(config["benchmark"]["commands"]["MMDU"]["media_files"][0])
    media.write_bytes(b"changed-image-with-the-same-path")
    assert benchmark_protocol(config) != before


@pytest.mark.parametrize("field,bad", [
    ("generation_backend", "candidate_pool"), ("trainer_backend", "dry_run"),
    ("evaluator_backend", "smoke_simulator"), ("reflection_backend", "fixture"),
])
def test_real_smoke_prohibits_partial_simulation_before_launch(tmp_path, field, bad):
    config = _config(tmp_path)
    config["loop"].update(mode="real_smoke", generation_backend="run_pipeline",
                          trainer_backend="bagel", evaluator_backend="benchmark", reflection_backend="deepseek")
    config["loop"][field] = bad
    with pytest.raises(ValueError, match="real_smoke_requires_real_backend"):
        IterationController(config, candidate_pool=tmp_path / "pool.jsonl", output_root=tmp_path / "run")


def test_baseline_precedes_rounds_and_is_reused_by_job_identity(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["loop"].update(evaluator_backend="benchmark")
    pool = tmp_path / "pool.jsonl"
    pool.write_text("{}\n", encoding="utf-8")
    controller = IterationController(config, candidate_pool=pool, output_root=tmp_path / "run")
    calls = []

    def run_round(iteration):
        baseline = tmp_path / "run" / "baseline" / "eval_result.json"
        assert baseline.is_file()
        assert json.loads(baseline.read_text())["iteration"] == 0
        calls.append(iteration)
        return {"iteration": iteration, "status": "COMPLETED"}

    monkeypatch.setattr(controller, "run_round", run_round)
    controller.run()
    controller.run()
    assert calls == [1, 1]
    for spec in config["benchmark"]["commands"].values():
        assert Path(spec["subset"]).with_suffix(".calls").read_text().splitlines() == ["call"]


def test_failed_baseline_does_not_start_training_or_generation(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["loop"].update(evaluator_backend="benchmark")
    config["benchmark"]["commands"].pop("WEAVEbench")
    controller = IterationController(config, candidate_pool=tmp_path / "pool", output_root=tmp_path / "run")
    monkeypatch.setattr(controller, "run_round", lambda i: pytest.fail("round before valid baseline"))
    with pytest.raises(ValueError, match="requires_three_commands"):
        controller.run()
