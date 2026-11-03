import json
import time

import pytest

from orchestrator.iteration.backends import smoke_evaluate, smoke_train
from orchestrator.iteration.controller import IterationController
from orchestrator.iteration.smoke_runtime import SmokeJobError, submit_smoke_job, wait_for_smoke_job
from orchestrator.iteration.state import digest, write_json


def _config():
    return {
        "loop": {"rounds": 2, "target_train_samples": 10, "actual_sample_cap": 2, "seed": 7},
        "smoke": {
            "train_duration_s": 0.06,
            "eval_duration_s": 0.04,
            "poll_interval_s": 0.01,
            "timeout_s": 1,
            "deterministic_seed": 11,
        },
        "trainer": {
            "model_name": "BAGEL-7B-MoT",
            "init_checkpoint": "/checkpoint",
            "epochs": 1,
        },
        "evaluator": {"backend": "smoke_simulator"},
    }


def test_smoke_two_round_loop_waits_and_consumes_checkpoint(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "\n".join([
            json.dumps({"record_key": "one", "task_type": "fine_grained_perception", "video_type": "tutorial"}),
            json.dumps({"record_key": "two", "task_type": "localized_editing", "video_type": "sports"}),
        ]) + "\n",
        encoding="utf-8",
    )
    config = _config()
    config["loop"].update({
        "trainer_backend": "smoke_simulator",
        "evaluator_backend": "smoke_simulator",
        "reflection_backend": "fixture",
    })
    output = tmp_path / "run"
    summary = IterationController(config, candidate_pool=pool, output_root=output).run()

    assert summary["status"] == "COMPLETED"
    for iteration in (1, 2):
        round_dir = output / f"iteration_{iteration:02d}"
        train_job = json.loads((round_dir / "train_job.json").read_text(encoding="utf-8"))
        eval_job = json.loads((round_dir / "eval_job.json").read_text(encoding="utf-8"))
        train_result = json.loads((round_dir / "train_result.json").read_text(encoding="utf-8"))
        eval_result = json.loads((round_dir / "eval_result.json").read_text(encoding="utf-8"))
        assert train_job["status"] == eval_job["status"] == "SUCCEEDED"
        assert train_job["poll_count"] > 0
        assert eval_job["poll_count"] > 0
        assert train_result["simulated"] is True
        assert train_result["real_optimizer_steps"] == 0
        assert eval_result["simulated"] is True
        assert eval_result["checkpoint_digest"] == train_result["checkpoint_digest"]
        assert eval_result["iteration_policy_digest"]
        state = json.loads((round_dir / "state.json").read_text(encoding="utf-8"))
        assert state["lifecycle_events"] == [
            "TRAIN_SUBMITTED", "TRAIN_WAITING", "TRAIN_COMPLETED",
            "EVAL_SUBMITTED", "EVAL_WAITING", "EVAL_COMPLETED",
        ]
    assert summary["rounds"][1]["previous_feedback_consumed"] is True


def test_smoke_job_is_persistent_and_reused_after_completion(tmp_path):
    job_path = tmp_path / "job.json"
    first = submit_smoke_job(
        job_path,
        job_id="train_iter_01",
        kind="train",
        iteration=1,
        duration_s=0,
        input_digest="abc",
    )
    completed = wait_for_smoke_job(job_path, timeout_s=1, poll_interval_s=0.01)
    reused = submit_smoke_job(
        job_path,
        job_id="train_iter_01",
        kind="train",
        iteration=1,
        duration_s=99,
        input_digest="changed",
    )
    assert first["job_id"] == completed["job_id"] == reused["job_id"]
    assert reused["status"] == "SUCCEEDED"
    assert reused["input_digest"] == "abc"


def test_running_smoke_job_is_reused_without_duplicate_submission(tmp_path):
    job_path = tmp_path / "running.json"
    first = submit_smoke_job(
        job_path,
        job_id="train_iter_01",
        kind="train",
        iteration=1,
        duration_s=0.2,
        input_digest="abc",
    )
    reused = submit_smoke_job(
        job_path,
        job_id="train_iter_01",
        kind="train",
        iteration=1,
        duration_s=0,
        input_digest="changed",
    )
    assert first["status"] == reused["status"] == "RUNNING"
    assert reused["attempt"] == first["attempt"] == 1
    assert reused["created_at"] == first["created_at"]
    assert reused["input_digest"] == "abc"


def test_failed_smoke_job_can_be_resubmitted_and_completes(tmp_path):
    job_path = tmp_path / "retry.json"
    submit_smoke_job(
        job_path,
        job_id="train_retry",
        kind="train",
        iteration=1,
        duration_s=1,
        input_digest="abc",
    )
    with pytest.raises(SmokeJobError, match="timeout"):
        wait_for_smoke_job(job_path, timeout_s=0.01, poll_interval_s=0.005)
    retried = submit_smoke_job(
        job_path,
        job_id="train_retry",
        kind="train",
        iteration=1,
        duration_s=0,
        input_digest="abc",
    )
    assert retried["attempt"] == 2
    completed = wait_for_smoke_job(job_path, timeout_s=1, poll_interval_s=0.005)
    assert completed["status"] == "SUCCEEDED"
    assert completed["attempt"] == 2


def test_smoke_evaluator_requires_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="requires_simulated_checkpoint"):
        smoke_evaluate(
            iteration=1,
            distribution={"pool_size": 0},
            train_result={"simulated": False},
            policy={},
            round_dir=tmp_path,
            config=_config(),
        )
    assert not (tmp_path / "eval_job.json").exists()


def test_smoke_job_timeout_is_persisted(tmp_path):
    job_path = tmp_path / "job.json"
    submit_smoke_job(
        job_path,
        job_id="slow",
        kind="train",
        iteration=1,
        duration_s=1,
        input_digest="abc",
    )
    with pytest.raises(SmokeJobError, match="timeout"):
        wait_for_smoke_job(job_path, timeout_s=0.01, poll_interval_s=0.005)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["status"] == "FAILED"
    assert "timeout_after" in job["error"]


def test_reflection_reuses_qwen_llm_config_from_pipeline_overlay(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"record_key": "one", "task_type": "fine_grained_perception"}) + "\n",
                    encoding="utf-8")
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text(
        "models:\n"
        "  qwen_llm:\n"
        "    mode: api\n"
        "    api:\n"
        "      model: deepseek-v4-flash\n"
        "      base_url: https://example.invalid\n"
        "      api_key_file: /run/secrets/deepseek.key\n",
        encoding="utf-8",
    )
    controller = IterationController(
        _config(),
        candidate_pool=pool,
        output_root=tmp_path / "run",
        pipeline_config=pipeline_config,
    )
    reflection_config = controller._reflection_config()
    assert reflection_config["models"]["qwen_llm"]["api"]["model"] == "deepseek-v4-flash"
    assert reflection_config["models"]["qwen_llm"]["api"]["api_key_file"] == "/run/secrets/deepseek.key"
