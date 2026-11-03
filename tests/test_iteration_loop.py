import json
import subprocess

import pytest

from orchestrator.iteration.controller import IterationController, _expand_config_text
from orchestrator.iteration.scheduler import source_key


def test_run_pipeline_preflight_reports_unexpanded_environment_variables(tmp_path, monkeypatch):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "models:\n  api_key: ${ITERATION_TEST_MISSING_KEY}\n"
        "  base_url: ${ITERATION_TEST_MISSING_URL}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ITERATION_TEST_MISSING_KEY", raising=False)
    monkeypatch.delenv("ITERATION_TEST_MISSING_URL", raising=False)

    with pytest.raises(ValueError, match="unexpanded_environment_variables") as exc_info:
        _expand_config_text(config_path.read_text(encoding="utf-8"), source=config_path)

    message = str(exc_info.value)
    assert "ITERATION_TEST_MISSING_KEY" in message
    assert "ITERATION_TEST_MISSING_URL" in message
    assert str(config_path) in message


def test_two_round_smoke_chain_consumes_feedback_and_changes_policy(tmp_path):
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"record_key": "a", "task_type": "fine_grained_perception", "video_type": "tutorial"},
        {"record_key": "b", "task_type": "localized_editing", "video_type": "sports"},
        {"record_key": "c", "task_type": "stepwise_sequence", "video_type": "gaming"},
    ]
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = {
        "loop": {"rounds": 2, "target_train_samples": 10000, "actual_sample_cap": 32, "seed": 3},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1},
    }
    output = tmp_path / "run"
    summary = IterationController(config, candidate_pool=pool, output_root=output).run()
    assert summary["status"] == "COMPLETED"
    assert summary["same_init_checkpoint_recorded"] is True
    assert all(item["actual_samples"] == 3 for item in summary["rounds"])
    assert summary["rounds"][1]["previous_feedback_consumed"] is True
    assert summary["rounds"][1]["policy_diff_from_previous_count"] > 0
    first = json.loads((output / "iteration_01" / "policy.json").read_text(encoding="utf-8"))
    second = json.loads((output / "iteration_02" / "policy.json").read_text(encoding="utf-8"))
    assert first["capability"] != second["capability"]
    train = json.loads((output / "iteration_01" / "train_result.json").read_text(encoding="utf-8"))
    assert train["optimizer_steps"] == 0
    assert train["checkpoint_loaded"] is False
    assert json.loads((output / "iteration_01" / "train_manifest.json").read_text(encoding="utf-8"))["shortage"] is True
    manifest_row = json.loads((output / "iteration_01" / "train_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert manifest_row["source_key"] == source_key(manifest_row["candidate"])


def test_run_pipeline_generation_backend_is_explicit_and_collects_output(tmp_path, monkeypatch):
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text("runtime: {}\n", encoding="utf-8")
    output = tmp_path / "run"

    def fake_run(command, **kwargs):
        root = Path(command[command.index("--output-root") + 1])
        root.mkdir(parents=True, exist_ok=True)
        (root / "data.jsonl").write_text(
            json.dumps({"record_key": "generated", "task_type": "fine_grained_perception"}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    from pathlib import Path

    monkeypatch.setattr("orchestrator.iteration.controller.subprocess.run", fake_run)
    config = {
        "loop": {"rounds": 1, "target_train_samples": 2, "actual_sample_cap": 1,
                 "generation_backend": "run_pipeline"},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1},
    }
    controller = IterationController(
        config,
        candidate_pool=tmp_path / "missing.jsonl",
        output_root=output,
        pipeline_config=pipeline_config,
        pipeline_input="video.mp4",
    )
    summary = controller.run()
    assert summary["status"] == "COMPLETED"
    assert json.loads((output / "iteration_01" / "generation_report.json").read_text(encoding="utf-8"))["backend"] == "run_pipeline"


def test_run_pipeline_generation_explicitly_forwards_audited_source_pool(tmp_path, monkeypatch):
    from pathlib import Path
    from PIL import Image
    from orchestrator.iteration.source_pool import prepare_source_pool
    from orchestrator.iteration.policy import default_policy

    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text("runtime: {}\n", encoding="utf-8")
    history = tmp_path / "history"
    history.mkdir()
    Image.new("RGB", (32, 32), "red").save(history / "frame.png")
    (history / "train.jsonl").write_text(json.dumps({"benchmark": "mmt_bench", "image": "frame.png",
        "question": "Color?", "A": "red", "B": "blue", "C": "green", "D": "yellow", "_answer": "A"}) + "\n")
    prepare_source_pool([{"split": "train", "root": str(history), "media_root": str(history),
                          "files": ["train.jsonl"]}], tmp_path / "historical_pool")
    source_pool = tmp_path / "historical_pool" / "candidate_pool.jsonl"
    output = tmp_path / "run"

    def fake_run(command, **kwargs):
        root = Path(command[command.index("--output-root") + 1])
        root.mkdir(parents=True, exist_ok=True)
        (root / "data.jsonl").write_text(
            json.dumps({"record_key": "generated", "task_type": "fine_grained_perception"}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("orchestrator.iteration.controller.subprocess.run", fake_run)
    config = {"loop": {"rounds": 1, "target_train_samples": 2, "actual_sample_cap": 1,
                        "generation_backend": "run_pipeline", "audited_source_pool": str(source_pool)},
              "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1}}
    controller = IterationController(config, candidate_pool=tmp_path / "missing.jsonl", output_root=output,
                                     pipeline_config=pipeline_config, pipeline_input="video.mp4")
    rd = controller._round_dir(1)
    generated = controller._run_pipeline_generation(rd, default_policy(1), iteration=1)
    assert len(generated) == 1
    rows = controller._supplement_candidates(generated, rd)
    assert len(rows) == 2
    assert sum(row.get("generation_origin") == "audited_historical_training_source" for row in rows) == 1
    report = json.loads((output / "iteration_01" / "generation_sources_report.json").read_text())
    assert report["pipeline_generated"] == 1
    assert report["audited_source_pool_forwarded"] == 1
    assert report["historical_sources_are_explicit"] is True
    source_pool.write_text(source_pool.read_text() + "\n")
    with pytest.raises(ValueError, match="changed_during_run"):
        controller._supplement_candidates(generated, rd)


def test_second_generation_round_consumes_pre_generation_source_schedule(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"record_key": "seed", "task_type": "fine_grained_perception"}) + "\n", encoding="utf-8")
    config = {
        "loop": {
            "rounds": 2,
            "target_train_samples": 2,
            "actual_sample_cap": 1,
            "generation_backend": "run_pipeline",
            "generation_source_cap": 1,
        },
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1},
    }
    controller = IterationController(
        config,
        candidate_pool=pool,
        output_root=tmp_path / "run",
        pipeline_config=tmp_path / "pipeline.yaml",
        pipeline_input="/videos",
    )
    controller.pipeline_config.write_text("runtime: {}\n", encoding="utf-8")
    prior = controller._round_dir(1) / "source_inventory.jsonl"
    prior.write_text(
        "\n".join([
            json.dumps({"record_key": "a", "video_path": "/videos/a.mp4", "video_type": "tutorial", "policy_domain": "Technology&Tutorial"}),
            json.dumps({"record_key": "b", "video_path": "/videos/b.mp4", "video_type": "gaming", "policy_domain": "Gaming&ACG"}),
        ]) + "\n",
        encoding="utf-8",
    )
    source_list = controller._prepare_generation_source_schedule(
        controller._round_dir(2), 2, controller._policy_for_round(2)[0]
    )
    assert source_list is not None and source_list.exists()
    assert len(source_list.read_text(encoding="utf-8").splitlines()) == 1
    schedule = (controller._round_dir(2) / "pre_generation_source_schedule.jsonl").read_text(encoding="utf-8")
    assert sum(json.loads(line)["selected"] for line in schedule.splitlines()) == 1


def test_zero_generation_source_cap_is_audit_only(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps({"record_key": "seed", "video_path": "/videos/seed.mp4", "task_type": "fine_grained_perception"}) + "\n",
        encoding="utf-8",
    )
    config = {
        "loop": {"rounds": 2, "target_train_samples": 2, "actual_sample_cap": 1, "generation_source_cap": 0},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/checkpoint", "epochs": 1},
    }
    controller = IterationController(config, candidate_pool=pool, output_root=tmp_path / "run")
    prior = controller._round_dir(1) / "source_inventory.jsonl"
    prior.write_text(pool.read_text(encoding="utf-8"), encoding="utf-8")
    assert controller._prepare_generation_source_schedule(
        controller._round_dir(2), 2, controller._policy_for_round(2)[0]
    ) is None
    assert (controller._round_dir(2) / "pre_generation_source_schedule.jsonl").exists()
    assert not (controller._round_dir(2) / "pre_generation_video_list.txt").exists()
