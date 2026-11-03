import json
from pathlib import Path

import pytest

from PIL import Image

from orchestrator.iteration.controller import IterationController
from orchestrator.iteration.real_backends import run_bagel_export, run_bagel_train


def _real_config(tmp_path):
    return {
        "loop": {"rounds": 1, "target_train_samples": 1, "actual_sample_cap": 1},
        "trainer": {"model_name": "BAGEL-7B-MoT", "init_checkpoint": "/base"},
        "bagel": {
            "vendor_root": str(tmp_path / "vendor"),
            "model_path": str(tmp_path / "model"),
            "python_executable": "python",
            "asset_root": str(tmp_path),
            "total_steps": 2,
            "nproc_per_node": 1,
        },
    }


def test_real_backend_builds_persistent_train_job_and_requires_real_result(tmp_path, monkeypatch):
    round_dir = tmp_path / "iteration_01"
    round_dir.mkdir()
    episodes = round_dir / "bagel_episodes.jsonl"
    episodes.write_text(json.dumps({"id": "e1", "turns": []}) + "\n", encoding="utf-8")
    (round_dir / "train_manifest.json").write_text(json.dumps({"actual_samples": 1}), encoding="utf-8")
    calls = {}

    def submit(job_dir, spec):
        calls["job_dir"] = job_dir
        calls["spec"] = spec
        return {"status": "queued"}

    def wait(job_dir, **kwargs):
        calls["wait"] = kwargs
        return {
            "backend": "bagel", "simulated": False, "real_optimizer_steps": 2,
            "optimizer_steps": 2, "checkpoint_dir": str(tmp_path / "ckpt"),
            "checkpoint_digest": "d" * 64,
        }

    monkeypatch.setattr("orchestrator.iteration.real_backends.submit_job", submit)
    monkeypatch.setattr("orchestrator.iteration.real_backends.wait_for_job", wait)
    result = run_bagel_train(
        train_manifest=round_dir / "train_manifest.jsonl",
        round_dir=round_dir,
        iteration=1,
        config=_real_config(tmp_path),
    )
    assert result["real_optimizer_steps"] == 2
    assert calls["spec"]["kind"] == "bagel_train"
    assert calls["spec"]["result_equals"] == {"backend": "bagel", "simulated": False}
    assert "--episodes" in calls["spec"]["argv"]


def test_real_backend_rejects_zero_optimizer_steps(tmp_path, monkeypatch):
    round_dir = tmp_path / "iteration_01"
    round_dir.mkdir()
    (round_dir / "bagel_episodes.jsonl").write_text("{}\n", encoding="utf-8")
    (round_dir / "train_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("orchestrator.iteration.real_backends.submit_job", lambda *a, **k: {})
    monkeypatch.setattr(
        "orchestrator.iteration.real_backends.wait_for_job",
        lambda *a, **k: {"backend": "bagel", "simulated": False, "real_optimizer_steps": 0},
    )
    try:
        run_bagel_train(
            train_manifest=round_dir / "train_manifest.jsonl",
            round_dir=round_dir,
            iteration=1,
            config=_real_config(tmp_path),
        )
    except RuntimeError as exc:
        assert "zero_optimizer_steps" in str(exc)
    else:
        raise AssertionError("zero optimizer steps must fail the real backend")


def test_coverage_plan_failure_prevents_gpu_training_submission(tmp_path, monkeypatch):
    rd = tmp_path / "iteration_01"
    rd.mkdir()
    (rd / "bagel_episodes.jsonl").write_text("{}\n", encoding="utf-8")
    config = _real_config(tmp_path)
    config["bagel"]["require_nine_tasks"] = True
    calls = []
    monkeypatch.setattr("orchestrator.iteration.real_backends.submit_job",
                        lambda path, spec: calls.append(spec))

    def failed_plan(*args, **kwargs):
        raise RuntimeError("native_coverage_plan_shortage")

    monkeypatch.setattr("orchestrator.iteration.real_backends.wait_for_job", failed_plan)
    with pytest.raises(RuntimeError, match="coverage_plan_shortage"):
        run_bagel_train(train_manifest=rd / "train_manifest.jsonl", round_dir=rd, iteration=1, config=config)
    assert len(calls) == 1
    assert calls[0]["kind"] == "native_optimizer_coverage_plan"


def test_controller_real_trainer_creates_bagel_data_and_export_stage(tmp_path, monkeypatch):
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), (20, 40, 60)).save(image)
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({
        "record_key": "r1", "task_type": "fine_grained_perception",
        "image": str(image), "question": "What color is the image?",
        "A": "blue", "B": "red", "C": "green", "D": "yellow", "_answer": "A",
    }) + "\n", encoding="utf-8")
    config = _real_config(tmp_path)
    config["loop"]["trainer_backend"] = "bagel"
    config["loop"]["evaluator_backend"] = "fixture"
    config["loop"]["reflection_backend"] = "fixture"
    monkeypatch.setattr(
        "orchestrator.iteration.controller.run_bagel_train",
        lambda **kwargs: {"backend": "bagel", "simulated": False, "real_optimizer_steps": 2,
                           "optimizer_steps": 2, "checkpoint_dir": str(tmp_path / "ckpt"),
                           "checkpoint_digest": "a" * 64},
    )
    monkeypatch.setattr(
        "orchestrator.iteration.controller.run_bagel_export",
        lambda **kwargs: {"backend": "bagel_export", "simulated": False,
                           "export_digest": "b" * 64},
    )
    summary = IterationController(config, candidate_pool=pool, output_root=tmp_path / "run").run()
    assert summary["status"] == "COMPLETED"
    round_dir = tmp_path / "run" / "iteration_01"
    report = json.loads((round_dir / "bagel_data_report.json").read_text(encoding="utf-8"))
    assert report["episode_count"] == 1
    assert json.loads((round_dir / "export_result.json").read_text(encoding="utf-8"))["backend"] == "bagel_export"
    state = json.loads((round_dir / "state.json").read_text(encoding="utf-8"))
    assert "EXPORT_DRY_RUN" in [entry["stage"] for entry in state["stage_history"]]
