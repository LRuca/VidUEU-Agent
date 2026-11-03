import json
import pytest

from orchestrator.iteration.controller import IterationController


def _make_pool(path):
    rows = [
        {"record_key": "one", "task_type": "fine_grained_perception", "video_type": "tutorial"},
        {"record_key": "two", "task_type": "localized_editing", "video_type": "sports"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_completed_round_is_resumable_without_repeating_stages(tmp_path):
    pool = tmp_path / "pool.jsonl"
    _make_pool(pool)
    config = {
        "loop": {"rounds": 1, "target_train_samples": 10, "actual_sample_cap": 2},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1},
    }
    output = tmp_path / "run"
    first = IterationController(config, candidate_pool=pool, output_root=output).run()
    state_path = output / "iteration_01" / "state.json"
    before = json.loads(state_path.read_text(encoding="utf-8"))["stage_history"]
    second = IterationController(config, candidate_pool=pool, output_root=output).run()
    after = json.loads(state_path.read_text(encoding="utf-8"))["stage_history"]
    assert first["status"] == second["status"] == "COMPLETED"
    assert before == after


def test_failed_stage_is_persisted_and_can_resume(tmp_path, monkeypatch):
    import orchestrator.iteration.controller as controller_module
    pool = tmp_path / "pool.jsonl"
    _make_pool(pool)
    config = {
        "loop": {"rounds": 1, "target_train_samples": 10, "actual_sample_cap": 2},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1},
    }
    output = tmp_path / "run"
    original = controller_module.dry_run_train
    def unavailable(*args, **kwargs):
        raise RuntimeError("transient_worker_failure")
    monkeypatch.setattr(controller_module, "dry_run_train", unavailable)
    broken = IterationController(config, candidate_pool=pool, output_root=output)
    with pytest.raises(RuntimeError, match="transient_worker_failure"):
        broken.run()
    failed = json.loads((output / "iteration_01" / "state.json").read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED"
    assert failed["failed_stage"] == "TRAIN_DRY_RUN"
    monkeypatch.setattr(controller_module, "dry_run_train", original)
    resumed = IterationController(config, candidate_pool=pool, output_root=output).run()
    assert resumed["status"] == "COMPLETED"


def test_deployment_only_pipeline_change_is_resumable(tmp_path):
    pool = tmp_path / "pool.jsonl"
    _make_pool(pool)
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        "runtime:\n"
        "  output_root: /old/output\n"
        "  single_gpu_scheduler:\n"
        "    devices: [cuda:0]\n"
        "    reserved_margin_mb: 30720\n"
        "tasks:\n"
        "  T1:\n"
        "    GLOBAL_MAX_FRAMES: 16\n",
        encoding="utf-8",
    )
    config = {
        "loop": {"rounds": 1, "target_train_samples": 10, "actual_sample_cap": 2},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1},
    }
    output = tmp_path / "run"
    IterationController(config, candidate_pool=pool, output_root=output, pipeline_config=pipeline)
    original = json.loads((output / "loop_manifest.json").read_text(encoding="utf-8"))

    pipeline.write_text(
        "runtime:\n"
        "  output_root: /new/output\n"
        "  single_gpu_scheduler:\n"
        "    devices: [cuda:6, cuda:7]\n"
        "    reserved_margin_mb: 4096\n"
        "tasks:\n"
        "  T1:\n"
        "    GLOBAL_MAX_FRAMES: 16\n",
        encoding="utf-8",
    )
    IterationController(config, candidate_pool=pool, output_root=output, pipeline_config=pipeline)
    current = json.loads((output / "loop_manifest.json").read_text(encoding="utf-8"))
    assert current["schema_version"] == 3
    assert current["semantic"] == original["semantic"]
    revisions = json.loads((output / "deployment_revisions.json").read_text(encoding="utf-8"))
    assert revisions and revisions[-1]["from"] != revisions[-1]["to"]


def test_semantic_pipeline_change_requires_new_output_root(tmp_path):
    pool = tmp_path / "pool.jsonl"
    _make_pool(pool)
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("tasks:\n  T1:\n    GLOBAL_MAX_FRAMES: 16\n", encoding="utf-8")
    config = {"loop": {"rounds": 1}, "trainer": {"init_checkpoint": "/checkpoint"}}
    output = tmp_path / "run"
    IterationController(config, candidate_pool=pool, output_root=output, pipeline_config=pipeline)
    pipeline.write_text("tasks:\n  T1:\n    GLOBAL_MAX_FRAMES: 32\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run_manifest_semantics_changed_use_new_output_root"):
        IterationController(config, candidate_pool=pool, output_root=output, pipeline_config=pipeline)


def test_legacy_generated_failure_migrates_only_after_semantic_check(tmp_path):
    pool = tmp_path / "pool.jsonl"
    _make_pool(pool)
    pipeline = tmp_path / "pipeline.yaml"
    old_pipeline = (
        "runtime:\n"
        "  output_root: /old/output\n"
        "  single_gpu_scheduler:\n"
        "    devices: [cuda:0]\n"
        "    reserved_margin_mb: 30720\n"
        "tasks:\n"
        "  T1:\n"
        "    GLOBAL_MAX_FRAMES: 16\n"
    )
    pipeline.write_text(old_pipeline, encoding="utf-8")
    config = {
        "loop": {"rounds": 1, "target_train_samples": 10, "actual_sample_cap": 2},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1},
    }
    output = tmp_path / "run"
    round_dir = output / "iteration_01"
    round_dir.mkdir(parents=True)
    (round_dir / "pipeline_config.yaml").write_text(old_pipeline, encoding="utf-8")
    (round_dir / "state.json").write_text(
        json.dumps({"status": "FAILED", "failed_stage": "GENERATED", "stage_history": [{"stage": "INVENTORIED"}]}),
        encoding="utf-8",
    )
    existing = {
        "schema_version": 2,
        "candidate_pool": str(pool.resolve()),
        "candidate_pool_sha256": __import__("hashlib").sha256(pool.read_bytes()).hexdigest(),
        "pipeline_input_identity": None,
        "rounds": 1,
        "target_train_samples": 10,
        "actual_sample_cap": 2,
        "seed": 20261016,
        "backends": {"reflection": "fixture", "trainer": "dry_run", "evaluator": "fixture", "generation": "candidate_pool"},
        "config_digest": "legacy-full-digest",
    }
    output.mkdir(exist_ok=True)
    (output / "loop_manifest.json").write_text(json.dumps(existing), encoding="utf-8")
    pipeline.write_text(
        old_pipeline.replace("/old/output", "/new/output").replace("[cuda:0]", "[cuda:6, cuda:7]").replace("30720", "4096"),
        encoding="utf-8",
    )
    IterationController(config, candidate_pool=pool, output_root=output, pipeline_config=pipeline)
    migration = json.loads((output / "manifest_migration.json").read_text(encoding="utf-8"))
    assert migration["from_schema"] == 2
    assert migration["to_schema"] == 3
    (round_dir / "state.json").write_text(
        json.dumps({"status": "FAILED", "failed_stage": "TRAIN_MANIFEST_READY", "stage_history": []}),
        encoding="utf-8",
    )
    # Once the semantic migration has been recorded, a later-stage failure
    # can resume without repeating the GENERATED-only legacy check.
    IterationController(config, candidate_pool=pool, output_root=output, pipeline_config=pipeline)
