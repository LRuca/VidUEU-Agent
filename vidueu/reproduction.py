"""Offline paper-recipe planning; execution always requires an explicit adapter."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from vidueu_config import ConfigLoader, expand_environment, read_environment, redact
import yaml

PAPER_TRAINING = {
    "internvl3_understanding": {"model": "InternVL3-8B", "epochs": 2, "learning_rate": 5e-5,
        "optimizer": "AdamW", "scheduler": "cosine", "global_batch_size": 32, "warmup_ratio": 0.03,
        "freeze": ["vision_encoder"], "lora": True, "precision": "bf16", "gradient_checkpointing": True,
        "deepspeed_stage": 2, "reference_gpus": 2},
    "bagel_understanding": {"model": "BAGEL-7B-MoT", "epochs": 1, "learning_rate": 5e-5,
        "optimizer": "AdamW", "scheduler": "cosine", "freeze": ["vae_encoder", "vae_decoder", "vit_encoder"],
        "lora_r": 16, "lora_alpha": 32, "fully_train": [], "losses": ["text_cross_entropy"], "reference_gpus": 8},
    "bagel_editing": {"model": "BAGEL-7B-MoT", "epochs": 1, "learning_rate": 5e-5,
        "optimizer": "AdamW", "scheduler": "cosine", "freeze": ["vae_encoder", "vae_decoder", "vit_encoder"],
        "lora_r": 16, "lora_alpha": 32, "fully_train": ["llm2vae", "vae2llm", "time_embedder"],
        "losses": ["text_cross_entropy", "image_mse"], "reference_gpus": 8},
}
PAPER_TRAINING["bagel_unified"] = dict(PAPER_TRAINING["bagel_editing"])
JUDGES = {"LLaVA-Bench": "Qwen3.5-35B-A3B", "MMDU": "Qwen3-VL-235B-A22B-Instruct",
          "GEdit-Bench-En": "gpt-4.1", "WEAVEbench": "gpt-4o"}


def run_recipe(path, operation="plan", *, execute=False):
    path = Path(path).resolve(strict=True)
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("invalid_recipe_version")
    recipe = config.get("recipe")
    if recipe not in {*PAPER_TRAINING, "paper_evaluation"}:
        raise ValueError("unknown_paper_recipe")
    if recipe in PAPER_TRAINING:
        kind = "train"
        paper = PAPER_TRAINING[recipe]
        supplied = config.get("paper", {})
        if supplied != paper:
            raise ValueError("paper_training_protocol_mismatch")
    else:
        kind = "eval"
        paper = config.get("paper", {})
        if paper.get("judges") != JUDGES or paper.get("generation_passes") != 1 or paper.get("judge_repetitions") not in {1, 3}:
            raise ValueError("paper_evaluation_protocol_mismatch")
        if paper.get("fixed_outputs_for_judge_repeats") is not True:
            raise ValueError("judge_repeats_require_fixed_outputs")
    if operation in {"train", "eval"} and operation != kind:
        raise ValueError("recipe_operation_mismatch")
    deployment = config.get("deployment", {})
    if not isinstance(deployment, dict):
        raise ValueError("recipe_deployment_must_be_object")
    missing = []
    required = ("python", "adapter", "source_revision", "model_path", "input_manifest", "output", "gpus")
    for name in required:
        if deployment.get(name) is None or deployment.get(name) == "":
            missing.append(name)
    values = {}
    environment = read_environment(path)
    for key, value in deployment.items():
        if isinstance(value, str):
            values[key] = expand_environment(value, environment)
        else:
            values[key] = value
    gpus = values.get("gpus", [])
    if not isinstance(gpus, list) or any(type(v) is not int or v < 0 for v in gpus) or len(gpus) != len(set(gpus)):
        raise ValueError("invalid_recipe_gpu_list")
    if not gpus:
        missing.append("nonempty_explicit_gpu_list")
    revision = values.get("source_revision")
    if revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("recipe_requires_full_source_commit")
    if kind == "train":
        # BAGEL packs tokens. Epoch length must be measured by the deployment,
        # never inferred from GPU count times gradient accumulation alone.
        extra = ("gradient_accumulation", "micro_batch_size") if recipe.startswith("internvl") else ("steps_per_epoch", "image_preprocessing", "token_packing")
        for key in extra:
            if not values.get(key):
                missing.append(key)
        if recipe.startswith("internvl") and values.get("micro_batch_size") and values.get("gradient_accumulation") and gpus:
            if len(gpus) * values["micro_batch_size"] * values["gradient_accumulation"] != 32:
                raise ValueError("internvl_effective_batch_must_equal_32")
    else:
        for key in ("frozen_case_ids", "prompt_file", "decoding", "judge_configuration"):
            if not values.get(key):
                missing.append(key)
    arguments = values.get("adapter_arguments", [])
    if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
        raise ValueError("recipe_adapter_arguments_must_be_strings")
    command = [values.get("python") or "<python>", values.get("adapter") or "<adapter>",
               "--recipe", str(path), *arguments]
    result = {"ok": True, "operation": operation, "kind": kind, "recipe": recipe,
              "paper": paper, "deployment": redact(values), "ready_for_execution": not missing,
              "missing_deployment_fields": missing, "command": command,
              "executed": False, "simulated": False,
              "note": "This validates the recipe contract, not optimizer behavior or benchmark correctness."}
    if execute:
        if operation not in {"train", "eval"}:
            raise ValueError("execute_requires_train_or_eval")
        if missing:
            raise ValueError("recipe_deployment_incomplete")
        def local(value):
            p = Path(value).expanduser()
            return (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
        adapter, model, manifest, output = (local(values[name]) for name in ("adapter", "model_path", "input_manifest", "output"))
        if not adapter.is_file() or not model.exists() or not manifest.is_file():
            raise ValueError("recipe_input_missing")
        if output.exists():
            raise ValueError("recipe_output_exists")
        try:
            checkout = Path(subprocess.check_output(["git", "-C", str(adapter.parent), "rev-parse", "--show-toplevel"],
                                                    text=True, stderr=subprocess.DEVNULL).strip())
            actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
            relative = str(adapter.relative_to(checkout))
            subprocess.run(["git", "-C", str(checkout), "ls-files", "--error-unmatch", "--", relative],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "--", relative], check=True)
            if actual != revision:
                raise ValueError("recipe_adapter_revision_mismatch")
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("recipe_adapter_revision_unverified") from exc
        # The deployment adapter accepts this public recipe and is responsible
        # for converting official datasets, recording consumption and metrics.
        command[1] = str(adapter)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
        completed = subprocess.run(command, cwd=path.parent, env=env, check=True)
        result.update(executed=True, returncode=completed.returncode)
    return result
