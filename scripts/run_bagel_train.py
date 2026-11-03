#!/usr/bin/env python3
"""Run a small, real BAGEL training job without modifying the vendor snapshot.

The wrapper patches the in-memory dataset registry before launching BAGEL's
native trainer.  It never copies model weights into this repository and writes
only run-local metadata beside the requested checkpoint directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
    elif path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            value.update(str(child.relative_to(path)).encode("utf-8"))
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
    else:
        raise FileNotFoundError(path)
    return value.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    from orchestrator.iteration.state import write_json
    write_json(path, value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--dataset-name", default="iteration_real_smoke")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--expected-num-tokens", type=int, default=4096)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=8192)
    parser.add_argument("--master-port", type=int, default=29601)
    parser.add_argument("--visible-devices", default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--require-nine-tasks", action="store_true",
                        help="Require a complete real optimizer-consumed episode for every paper task")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Validate an existing audited checkpoint without launching training")
    return parser.parse_args()


def _episode_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _prepare_dataset_registry(vendor_root: Path, dataset_name: str, episodes: Path, count: int) -> Path:
    # Native DATASET_INFO is patched only inside each torchrun worker.
    config_path = episodes.parent / f"{episodes.stem}.bagel_dataset.yaml"
    config = {
        "multiturn_jsonl": {
            "dataset_names": [dataset_name],
            "image_transform_args": {
                "image_stride": 16,
                "max_image_size": 256,
                "min_image_size": 256,
            },
            "vit_image_transform_args": {
                "image_stride": 14,
                "max_image_size": 224,
                "min_image_size": 224,
                "max_pixels": 50176,
            },
            "is_mandatory": True,
            "shuffle_lines": False,
            "shuffle_seed": 0,
            "only_current_turn_images": False,
            "num_used_data": [count],
            "weight": 1.0,
        }
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _build_trainer_argv(args: argparse.Namespace, dataset_config: Path, checkpoint_dir: Path) -> list[str]:
    results_dir = args.run_root
    return [
        str(args.vendor_root / "train" / "pretrain_unified_navit.py"),
        "--model_path", str(args.model_path.resolve()),
        "--dataset_config_file", str(dataset_config.resolve()),
        "--results_dir", str(results_dir.resolve()),
        "--checkpoint_dir", str(checkpoint_dir.resolve()),
        "--total_steps", str(args.steps),
        "--save_every", str(args.steps),
        "--log_every", "1",
        "--num_workers", "0",
        "--prefetch_factor", "1",
        "--expected_num_tokens", str(args.expected_num_tokens),
        "--max_num_tokens", str(args.max_num_tokens),
        "--max_num_tokens_per_sample", str(args.max_num_tokens_per_sample),
        "--global_seed", str(args.seed),
        "--data_seed", str(args.seed),
        "--visual_gen", "True",
        "--visual_und", "True",
        "--use_lora", "True",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.05",
        "--freeze_vit", "True",
        "--freeze_vae", "True",
        "--finetune_from_hf", "True",
        "--finetune_from_ema", "True",
        "--resume_model_only", "True",
        "--auto_resume", "False",
        "--resume_from", str(args.model_path.resolve()),
        "--disable_ema", "True",
        "--export_official_layout", "False",
        "--save_optimizer_state", "True",
        "--lr", "5e-5",
        "--lr_scheduler", "cosine",
        "--warmup_steps", "0",
        "--sharding_strategy", "FULL_SHARD",
        "--num_shard", str(max(1, args.nproc_per_node)),
        "--num_replicate", "1",
        "--wandb_offline", "True",
        "--wandb_project", "vidueu_real_smoke",
        "--wandb_name", f"iteration_{args.run_root.name}",
        "--debug_forward_text", "False",
        "--grad_diag_enable", "False",
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
    ]


def _find_checkpoint(checkpoint_dir: Path, expected_steps: int, nproc: int = 1) -> Path:
    checkpoint = checkpoint_dir / f"{expected_steps:07d}"
    for name in ("model.safetensors", "scheduler.pt", "data_status.pt"):
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise FileNotFoundError(f"bagel_checkpoint_incomplete:{checkpoint / name}")
    for rank in range(nproc):
        optimizer = checkpoint / f"optimizer.{rank:05d}-of-{nproc:05d}.pt"
        if not optimizer.is_file() or not optimizer.stat().st_size:
            raise FileNotFoundError(f"bagel_optimizer_state_missing:{optimizer}")
    return checkpoint


def _parse_loss_summary(log_path: Path) -> dict[str, float | int | None]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    import math
    points = []
    for line in text.splitlines():
        match = re.search(r"\(step=(\d+)\)", line)
        if not match:
            continue
        terms = {key: float(value) for key, value in re.findall(r"Train Loss (ce|mse):\s*([^, ]+)", line)}
        if set(terms) != {"ce", "mse"} or any(not math.isfinite(v) for v in terms.values()):
            raise ValueError("invalid_or_incomplete_native_loss_curve")
        points.append({"step": int(match.group(1)), **terms, "total": terms["ce"] + terms["mse"]})
    losses = [point["total"] for point in points]
    return {
        "available": bool(losses),
        "points": points,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
    }


def main() -> int:
    args = _parse_args()
    if args.steps <= 0 or args.nproc_per_node <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("steps_and_nproc_must_be_positive")
    vendor_root = args.vendor_root.resolve()
    model_path = args.model_path.resolve()
    episodes = args.episodes.resolve()
    run_root = args.run_root.resolve()
    result_path = args.result_path.resolve()
    args.vendor_root, args.model_path, args.episodes, args.run_root = vendor_root, model_path, episodes, run_root
    if not vendor_root.is_dir() or not model_path.is_dir() or not episodes.is_file():
        raise FileNotFoundError("bagel_train_input_missing")
    count = _episode_count(episodes)
    if count <= 0:
        raise ValueError("bagel_train_episodes_empty")
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_root / "checkpoints"
    dataset_config = episodes.parent / f"{episodes.stem}.bagel_dataset.yaml"
    trainer_argv = _build_trainer_argv(args, dataset_config, checkpoint_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(vendor_root) + os.pathsep + env.get("PYTHONPATH", "")
    if args.visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = str(run_root)
    audit_output = args.audit_output.resolve()
    if "LOCAL_RANK" in env:
        sys.path.insert(0, str(vendor_root))
        os.environ.update(env)
        from data.dataset_info import DATASET_INFO
        DATASET_INFO.setdefault("multiturn_jsonl", {})[args.dataset_name] = {
            "data_dir": "/", "jsonl_path": str(episodes), "num_total_samples": count,
        }
        sys.argv = [str(vendor_root / "train" / "pretrain_unified_navit.py"), *trainer_argv[1:]]
        from orchestrator.iteration.bagel_audit import TrainingAudit
        rank = os.environ.get("LOCAL_RANK", "0")
        rank_output = audit_output.with_name(f"{audit_output.stem}_rank{rank}{audit_output.suffix}")
        audit = TrainingAudit(rank_output)
        audit.install()
        try:
            runpy.run_path(str(vendor_root / "train" / "pretrain_unified_navit.py"), run_name="__main__")
        finally:
            audit.close()
        return 0

    from orchestrator.iteration.artifacts import content_identity, publish_immutable_json
    from orchestrator.iteration.state import read_json
    from orchestrator.iteration.jobs import file_fingerprint, process_matches
    if args.finalize_only:
        # The failed wrapper can leave a complete native checkpoint. Reconcile
        # its immutable launch inputs before recovering the post-training work.
        job_dir = run_root.parent / "jobs" / "train"
        spec = read_json(job_dir / "spec.json")
        state = read_json(job_dir / "status.json")
        if state["status"] != "failed" or any(process_matches(state.get(k)) for k in ("process", "command_process")):
            raise ValueError("cannot_finalize_live_or_unfailed_training_job")
        expected_argv = [sys.executable, str(Path(__file__).resolve()),
                         *[arg for arg in sys.argv[1:] if arg != "--finalize-only"]]
        if spec["argv"] != expected_argv or spec["input_digests"]["episodes"] != file_fingerprint(episodes):
            raise ValueError("training_finalization_inputs_changed")
    elif (checkpoint_dir / f"{args.steps:07d}").exists():
        raise ValueError("existing_checkpoint_requires_explicit_finalization")
    dataset_config = _prepare_dataset_registry(vendor_root, args.dataset_name, episodes, count)
    initial_identity = content_identity(model_path)
    publish_immutable_json(run_root / "initial_model_identity.json", initial_identity)
    publish_immutable_json(run_root / "trainer_argv.json", trainer_argv)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", str(args.nproc_per_node),
               "--master_port", str(args.master_port), str(Path(__file__).resolve()), *sys.argv[1:]]
    if not args.finalize_only:
        completed = subprocess.run(command, env=env, cwd=str(REPO_ROOT), check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    checkpoint = _find_checkpoint(checkpoint_dir, args.steps, args.nproc_per_node)
    from safetensors import safe_open
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as weights:
        if not weights.keys() or not any("lora_B" in key for key in weights.keys()):
            raise ValueError("native_checkpoint_has_no_lora_weights")
    log_path = run_root / "log.txt"
    if content_identity(model_path) != initial_identity:
        raise ValueError("initial_model_changed_during_training")
    from orchestrator.iteration.bagel_audit import summarize_training_audits
    episode_rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_rows = [read_json(audit_output.with_name(f"{audit_output.stem}_rank{rank}{audit_output.suffix}"))
                  for rank in range(args.nproc_per_node)]
    audit_summary = summarize_training_audits(audit_rows, episode_rows, args.steps)
    if args.require_nine_tasks and audit_summary["missing_complete_task_types"]:
        _write_json(run_root / "coverage_shortage.json", audit_summary)
        raise ValueError("incomplete_optimizer_task_coverage:" + ",".join(audit_summary["missing_complete_task_types"]))
    result = {
        "schema_version": 1,
        "backend": "bagel",
        "simulated": False,
        "episodes": count,
        "requested_optimizer_steps": args.steps,
        "real_optimizer_steps": audit_summary["real_optimizer_steps"],
        "optimizer_steps": audit_summary["real_optimizer_steps"],
        "training_audit": audit_summary,
        "loss_curve_summary": _parse_loss_summary(log_path),
        "model_path": str(model_path),
        "init_checkpoint": str(model_path),
        "init_checkpoint_sha256": initial_identity["sha256"],
        "checkpoint_loaded": True,
        "recovered_finalization": args.finalize_only,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_digest": _sha256(checkpoint),
        "dataset_config": str(dataset_config),
        "episodes_path": str(episodes),
    }
    marker = run_root / "checkpoint_complete.json"
    _write_json(marker, result)
    _write_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
