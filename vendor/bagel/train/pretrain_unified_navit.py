# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import gc
import json
import os
import subprocess
import sys
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Optional

from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from data.citation_dialogue_utils import (
    CE_GROUP_ANSWER_TEXT,
    CE_GROUP_CITE,
    CE_GROUP_CLAIM,
    MSE_GROUP_ANSWER_IMAGE,
)
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


def _is_training_checkpoint_dir(path: Optional[str]) -> bool:
    if path is None:
        return False
    if not os.path.isdir(path):
        return False
    has_model = (
        os.path.isfile(os.path.join(path, "model.safetensors"))
        or os.path.isfile(os.path.join(path, "trainable_model.safetensors"))
    )
    has_optimizer_or_scheduler = False
    for name in os.listdir(path):
        if name.startswith("optimizer.") and name.endswith(".pt"):
            has_optimizer_or_scheduler = True
            break
    if os.path.isfile(os.path.join(path, "scheduler.pt")):
        has_optimizer_or_scheduler = True
    return has_model and has_optimizer_or_scheduler


def _is_trainable_only_checkpoint_dir(path: Optional[str]) -> bool:
    if path is None:
        return False
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "trainable_model.safetensors")):
        return False
    for name in os.listdir(path):
        if name.startswith("optimizer.") and name.endswith(".pt"):
            return True
    return os.path.isfile(os.path.join(path, "scheduler.pt"))


def _is_official_bagel_dir(path: Optional[str]) -> bool:
    if path is None:
        return False
    if not os.path.isdir(path):
        return False
    required = [
        "ema.safetensors",
        "llm_config.json",
        "vit_config.json",
        "ae.safetensors",
    ]
    return all(os.path.isfile(os.path.join(path, n)) for n in required)


def _resolve_finetune_init_source(
    training_args,
    model_args,
    resume_from,
    resume_model_only: bool,
    finetune_from_ema: bool,
    logger,
):
    """
    For finetune_from_hf=True, prevent accidental random-init training:
    - if resume_from is empty, force it to model_args.model_path
    - when loading from a model directory (not a train checkpoint), require model-only resume
    - prefer ema.safetensors by default
    """
    if not training_args.finetune_from_hf:
        return resume_from, resume_model_only, finetune_from_ema

    model_path = os.path.abspath(model_args.model_path)
    if not _is_official_bagel_dir(model_path):
        raise RuntimeError(
            "finetune_from_hf=True requires model_path to be a BAGEL official-like directory "
            f"with ema.safetensors/llm_config.json/vit_config.json/ae.safetensors, got: {model_path}"
        )

    resume_from_val = resume_from
    if resume_from_val is None or str(resume_from_val).strip() == "":
        resume_from_val = model_path
        resume_model_only = True
        finetune_from_ema = True
        if dist.get_rank() == 0:
            logger.warning(
                "[InitGuard] finetune_from_hf=True but resume_from is empty; "
                f"forcing resume_from=model_path ({model_path}), resume_model_only=True, finetune_from_ema=True."
            )

    resume_from_abs = os.path.abspath(str(resume_from_val))
    if not os.path.exists(resume_from_abs):
        raise FileNotFoundError(f"resume_from path does not exist: {resume_from_abs}")

    # Continue-training checkpoint (contains optimizer/scheduler) is allowed to keep full-state resume.
    if _is_training_checkpoint_dir(resume_from_abs):
        if dist.get_rank() == 0:
            logger.info(
                f"[InitGuard] resume_from is a training checkpoint dir: {resume_from_abs}; "
                f"resume_model_only={resume_model_only}, finetune_from_ema={finetune_from_ema}"
            )
        return resume_from_abs, resume_model_only, finetune_from_ema

    # Base-model initialization path: must be model-only load to avoid accidental scratch training.
    if not resume_model_only:
        raise RuntimeError(
            "finetune_from_hf=True with a base model directory must use resume_model_only=True "
            f"(resume_from={resume_from_abs})."
        )

    ema_path = os.path.join(resume_from_abs, "ema.safetensors")
    model_path_single = os.path.join(resume_from_abs, "model.safetensors")
    if finetune_from_ema and (not os.path.isfile(ema_path)) and os.path.isfile(model_path_single):
        finetune_from_ema = False
        if dist.get_rank() == 0:
            logger.warning(
                f"[InitGuard] finetune_from_ema=True but {ema_path} not found; "
                "falling back to model.safetensors."
            )
    if (not finetune_from_ema) and os.path.isfile(ema_path):
        finetune_from_ema = True
        if dist.get_rank() == 0:
            logger.warning(
                f"[InitGuard] ema.safetensors exists at {ema_path}; forcing finetune_from_ema=True."
            )
    if (not os.path.isfile(ema_path)) and (not os.path.isfile(model_path_single)):
        raise FileNotFoundError(
            f"Neither ema.safetensors nor model.safetensors found under resume_from={resume_from_abs}"
        )

    if dist.get_rank() == 0:
        logger.info(
            "[InitGuard] base-model initialization confirmed: "
            f"resume_from={resume_from_abs}, resume_model_only={resume_model_only}, finetune_from_ema={finetune_from_ema}"
        )
    return resume_from_abs, resume_model_only, finetune_from_ema


def _export_official_layout_from_ckpt(
    *,
    ckpt_dir: str,
    base_model_dir: str,
    out_dir: str,
    lora_r: int,
    lora_alpha: float,
    out_dtype: str,
    logger,
):
    export_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tools", "export_ckpt_to_official_layout_hardcoded.py")
    )
    if not os.path.isfile(export_script):
        raise FileNotFoundError(f"Export script not found: {export_script}")

    cmd = [
        sys.executable,
        export_script,
        "--ckpt-dir",
        ckpt_dir,
        "--base-model-dir",
        base_model_dir,
        "--out-dir",
        out_dir,
        "--lora-r",
        str(lora_r),
        "--lora-alpha",
        str(lora_alpha),
        "--out-dtype",
        str(out_dtype),
    ]
    logger.info(f"[OfficialExport] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _pick_safe_decode_id(tokenizer) -> int:
    candidates = [
        getattr(tokenizer, "unk_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        0,
    ]
    for c in candidates:
        if c is None:
            continue
        try:
            ci = int(c)
        except Exception:
            continue
        if ci < 0 or ci >= len(tokenizer):
            continue
        try:
            tok = tokenizer._convert_id_to_token(ci)
        except Exception:
            tok = None
        if tok is not None:
            return ci

    limit = min(len(tokenizer), 2048)
    for ci in range(limit):
        try:
            tok = tokenizer._convert_id_to_token(ci)
        except Exception:
            tok = None
        if tok is not None:
            return ci
    return 0


def _safe_decode_ids(tokenizer, token_ids):
    """
    Decode ids robustly for debug logging:
    replace ids out of vocab / unresolved-to-token with a known safe id.
    """
    ids = token_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if not isinstance(ids, list):
        ids = [int(ids)]

    safe_id = _pick_safe_decode_id(tokenizer)
    fixed = []
    replaced = 0
    for x in ids:
        try:
            xi = int(x)
        except Exception:
            xi = safe_id
            replaced += 1
            fixed.append(xi)
            continue
        if xi < 0 or xi >= len(tokenizer):
            xi = safe_id
            replaced += 1
            fixed.append(xi)
            continue
        try:
            tok = tokenizer._convert_id_to_token(xi)
        except Exception:
            tok = None
        if tok is None:
            xi = safe_id
            replaced += 1
        fixed.append(xi)

    text = tokenizer.decode(fixed)
    return text, replaced


def _take_first_label_segment(label_ids, eos_token_id: Optional[int], max_tokens: int) -> list[int]:
    """
    For CE debug readability, take one contiguous target segment:
    stop at the first EOS (inclusive) or max_tokens.
    """
    out = []
    if max_tokens <= 0:
        return out
    eos = None if eos_token_id is None else int(eos_token_id)
    for x in label_ids:
        out.append(int(x))
        if eos is not None and int(x) == eos:
            break
        if len(out) >= max_tokens:
            break
    return out


def _unpatchify_latent_patches(
    patches_2d: torch.Tensor, h: int, w: int, patch_size: int, latent_channel: int
) -> torch.Tensor:
    expected = int(h) * int(w)
    if patches_2d.dim() != 2 or patches_2d.shape[0] != expected:
        raise ValueError(
            f"Invalid patch tensor shape={tuple(patches_2d.shape)}, expected ({expected}, D)"
        )
    x = patches_2d.reshape(h, w, patch_size, patch_size, latent_channel)
    x = x.permute(4, 0, 2, 1, 3).contiguous().reshape(
        latent_channel, h * patch_size, w * patch_size
    )
    return x.unsqueeze(0)


def _to_pil_image(x_chw: torch.Tensor) -> Image.Image:
    x = x_chw.detach().float().cpu()
    if x.numel() == 0:
        raise ValueError("Empty tensor cannot be converted to image.")
    # Heuristic: VAE decoder outputs are usually in [-1, 1].
    if x.min().item() < -0.05 or x.max().item() > 1.05:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    arr = (x * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def _build_sample_spans(sample_lens, sample_count: Optional[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    if sample_count is None:
        sample_count = len(sample_lens)
    n = min(int(sample_count), len(sample_lens))
    for i in range(n):
        seg = int(sample_lens[i])
        end = start + seg
        spans.append((start, end))
        start = end
    return spans


def _find_sample_idx(seq_pos: int, sample_spans: list[tuple[int, int]]) -> int:
    for i, (l, r) in enumerate(sample_spans):
        if l <= seq_pos < r:
            return i
    return -1


def _summarize_sample_meta(meta) -> str:
    if not isinstance(meta, dict):
        return str(meta)
    ds = str(meta.get("dataset_name", "unknown"))
    di = meta.get("data_indexes", None)
    if isinstance(di, list) and len(di) >= 3:
        return f"{ds}|ep={di[1]}|task={di[2]}"
    return f"{ds}|{di}"


def _save_ce_debug_records(
    *,
    curr_step: int,
    data: dict,
    sample_spans: list[tuple[int, int]],
    sample_metas: list,
    debug_pred_ids: Optional[torch.Tensor],
    debug_label_ids: Optional[torch.Tensor],
    tokenizer,
    max_tokens: int,
    out_dir: str,
    logger,
) -> Optional[str]:
    if (
        not isinstance(debug_pred_ids, torch.Tensor)
        or not isinstance(debug_label_ids, torch.Tensor)
        or "ce_loss_indexes" not in data
    ):
        return None
    pred_all = debug_pred_ids.detach().cpu().tolist()
    label_all = debug_label_ids.detach().cpu().tolist()
    pair_len = min(len(pred_all), len(label_all))
    if pair_len <= 0:
        return None
    pred_all = pred_all[:pair_len]
    label_all = label_all[:pair_len]

    ce_seq = [int(x) for x in data["ce_loss_indexes"].detach().cpu().tolist()][:pair_len]
    mse_seq = (
        [int(x) for x in data["mse_loss_indexes"].detach().cpu().tolist()]
        if "mse_loss_indexes" in data
        else []
    )
    seq_to_stream = {sid: i for i, sid in enumerate(ce_seq)}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"step{curr_step:07d}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (l, r) in enumerate(sample_spans):
            meta = sample_metas[i] if i < len(sample_metas) else None
            ce_stream_positions = [seq_to_stream[s] for s in ce_seq if l <= s < r]
            ce_count = len(ce_stream_positions)
            mse_count = sum(1 for s in mse_seq if l <= s < r)
            if ce_count > 0 and mse_count > 0:
                modality = "both"
            elif ce_count > 0:
                modality = "text"
            elif mse_count > 0:
                modality = "image"
            else:
                modality = "none"

            record = {
                "step": int(curr_step),
                "sample_idx": int(i),
                "sample_meta": meta,
                "modality": modality,
                "ce_tokens": int(ce_count),
                "mse_tokens": int(mse_count),
            }
            if ce_count > 0:
                sample_pred = [pred_all[p] for p in ce_stream_positions]
                sample_label = [label_all[p] for p in ce_stream_positions]
                label_seg = _take_first_label_segment(
                    sample_label, getattr(tokenizer, "eos_token_id", None), max_tokens
                )
                seg_len = len(label_seg)
                pred_seg = sample_pred[:seg_len]
                pred_text, pred_replaced = _safe_decode_ids(tokenizer, pred_seg)
                label_text, label_replaced = _safe_decode_ids(tokenizer, label_seg)
                token_acc = 0.0
                if seg_len > 0:
                    token_acc = sum(int(a == b) for a, b in zip(pred_seg, label_seg)) / float(seg_len)
                record.update(
                    {
                        "segment_tokens": int(seg_len),
                        "token_acc": float(token_acc),
                        "pred_text": pred_text,
                        "label_text": label_text,
                        "safe_decode_replaced_pred": int(pred_replaced),
                        "safe_decode_replaced_label": int(label_replaced),
                    }
                )

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                f"[StepDbg] step={curr_step:07d} sample={i:02d} modality={modality} "
                f"ce_tokens={ce_count} mse_tokens={mse_count} meta={_summarize_sample_meta(meta)}"
            )
            if ce_count > 0:
                logger.info(
                    f"[StepDbgText] step={curr_step:07d} sample={i:02d} "
                    f"token_acc={record['token_acc']:.3f} pred='{record['pred_text']}' | label='{record['label_text']}'"
                )
    return out_path


def _save_mse_debug_previews(
    *,
    curr_step: int,
    data: dict,
    sample_spans: list[tuple[int, int]],
    debug_mse_pred_patches: Optional[torch.Tensor],
    debug_mse_clean_patches: Optional[torch.Tensor],
    debug_mse_target_patches: Optional[torch.Tensor],
    vae_model,
    device,
    latent_patch_size: int,
    latent_channel: int,
    timestep_shift: float,
    out_dir: str,
    logger,
) -> list[dict]:
    records = []
    if (
        not isinstance(debug_mse_pred_patches, torch.Tensor)
        or not isinstance(debug_mse_clean_patches, torch.Tensor)
        or not isinstance(debug_mse_target_patches, torch.Tensor)
        or debug_mse_pred_patches.numel() == 0
    ):
        return records
    if (
        "packed_vae_token_indexes" not in data
        or "patchified_vae_latent_shapes" not in data
        or "mse_loss_indexes" not in data
        or "packed_timesteps" not in data
    ):
        return records

    packed_vae_token_indexes = [int(x) for x in data["packed_vae_token_indexes"].detach().cpu().tolist()]
    mse_loss_indexes = [int(x) for x in data["mse_loss_indexes"].detach().cpu().tolist()]
    packed_timesteps = [float(x) for x in data["packed_timesteps"].detach().cpu().tolist()]
    latent_shapes = data["patchified_vae_latent_shapes"]
    if len(mse_loss_indexes) == 0 or len(latent_shapes) == 0:
        return records
    if len(packed_timesteps) != len(packed_vae_token_indexes):
        logger.warning(
            f"[FwdImgDbg] step={curr_step:07d} skip previews: packed_timesteps length mismatch "
            f"({len(packed_timesteps)} vs {len(packed_vae_token_indexes)})."
        )
        return records

    seq_to_mse_pos = {sid: pos for pos, sid in enumerate(mse_loss_indexes)}
    seq_to_t_raw = {sid: t for sid, t in zip(packed_vae_token_indexes, packed_timesteps)}
    offset = 0
    os.makedirs(out_dir, exist_ok=True)
    vae_dtype = next(vae_model.parameters()).dtype

    for img_idx, shape in enumerate(latent_shapes):
        h, w = int(shape[0]), int(shape[1])
        n = h * w
        seq_slice = packed_vae_token_indexes[offset : offset + n]
        offset += n

        positions = [seq_to_mse_pos[s] for s in seq_slice if s in seq_to_mse_pos]
        if len(positions) != n:
            continue

        sample_idx = _find_sample_idx(seq_slice[0], sample_spans)
        pred = debug_mse_pred_patches.detach().float().cpu()[positions]
        clean = debug_mse_clean_patches.detach().float().cpu()[positions]
        target = debug_mse_target_patches.detach().float().cpu()[positions]
        t_raw = torch.tensor([seq_to_t_raw[s] for s in seq_slice], dtype=torch.float32)
        t = torch.sigmoid(t_raw)
        t = timestep_shift * t / (1 + (timestep_shift - 1) * t)
        pred_x0 = clean + t[:, None] * (target - pred)
        gt_x0 = clean

        pred_latent = _unpatchify_latent_patches(pred_x0, h, w, latent_patch_size, latent_channel)
        gt_latent = _unpatchify_latent_patches(gt_x0, h, w, latent_patch_size, latent_channel)
        with torch.no_grad():
            pred_img = vae_model.decode(pred_latent.to(device=device, dtype=vae_dtype))[0]
            gt_img = vae_model.decode(gt_latent.to(device=device, dtype=vae_dtype))[0]

        sample_tag = f"s{sample_idx:02d}" if sample_idx >= 0 else "sXX"
        stem = f"step{curr_step:07d}_{sample_tag}_img{img_idx:02d}"
        pred_path = os.path.join(out_dir, f"{stem}_pred.png")
        gt_path = os.path.join(out_dir, f"{stem}_gt.png")
        _to_pil_image(pred_img).save(pred_path)
        _to_pil_image(gt_img).save(gt_path)
        logger.info(
            f"[FwdImgDbg] step={curr_step:07d} sample={sample_idx:02d} img={img_idx:02d} "
            f"saved: pred={pred_path}, gt={gt_path}"
        )
        records.append(
            {
                "step": int(curr_step),
                "sample_idx": int(sample_idx),
                "img_idx": int(img_idx),
                "pred_path": pred_path,
                "gt_path": gt_path,
            }
        )
    return records


def _build_chat_prompt(user_text: str) -> str:
    return f"<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"


def _move_tensor_dict_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _default_preflight_text_prompts() -> list[str]:
    return [
        "What is 2 + 3? Answer with only the number.",
        "Describe this in one short sentence: a blue circle on white background.",
        "Output exactly this JSON: {\"ok\": true}",
    ]


def _default_preflight_image_prompts() -> list[str]:
    return [
        "Generate an image of a red apple on a white table.",
        "Generate an image of a small yellow bird on a branch.",
    ]


@torch.no_grad()
def _run_preflight_inference(
    *,
    training_args,
    model_args,
    model,
    vae_model,
    vae_config,
    tokenizer,
    new_token_ids,
    device: int,
    logger,
):
    if not getattr(training_args, "preflight_infer", False):
        return

    rank = dist.get_rank()
    device_obj = torch.device(f"cuda:{device}")
    preflight_root = os.path.join(training_args.results_dir, "preflight_infer")
    preflight_text_path = os.path.join(preflight_root, "text_outputs.jsonl")
    preflight_image_root = os.path.join(preflight_root, "images")
    reasoning_mode = str(getattr(training_args, "preflight_reasoning_mode", "none")).strip().lower()
    if reasoning_mode not in {"none", "think", "citation"}:
        raise ValueError(
            f"Invalid preflight_reasoning_mode='{reasoning_mode}'. "
            "Expected one of: none, think, citation."
        )
    use_think = reasoning_mode == "think"
    use_citation = reasoning_mode == "citation"

    if rank == 0:
        os.makedirs(preflight_root, exist_ok=True)
        os.makedirs(preflight_image_root, exist_ok=True)
        logger.info(f"[Preflight] root dir: {preflight_root}")
        logger.info(
            f"[Preflight] enabled with text_n={training_args.preflight_text_n}, "
            f"img_n={training_args.preflight_img_n}, img_steps={training_args.preflight_img_steps}, "
            f"reasoning_mode={reasoning_mode}"
        )

    failed = torch.zeros(1, device=device_obj, dtype=torch.int32)
    error_msg = ""

    try:
        model.eval()
        model.to(device_obj)
        if training_args.visual_gen and vae_model is not None:
            vae_model.to(device_obj)
            vae_model.eval()
        preflight_inferencer = None
        if reasoning_mode != "none":
            try:
                from inferencer import InterleaveInferencer
            except Exception:
                from BAGEL.inferencer import InterleaveInferencer
            preflight_inferencer = InterleaveInferencer(
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                vae_transform=None,
                vit_transform=None,
                new_token_ids=deepcopy(new_token_ids),
            )

        # 1) text-only sanity checks via BAGEL chat()
        text_prompts = _default_preflight_text_prompts()
        text_n = max(0, int(training_args.preflight_text_n))
        text_max_len = max(1, int(training_args.preflight_text_max_length))
        text_records = []
        for idx in range(text_n):
            user_prompt = text_prompts[idx % len(text_prompts)]
            prompt = _build_chat_prompt(user_prompt)
            if preflight_inferencer is None:
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    answer = model.chat(
                        tokenizer=tokenizer,
                        new_token_ids=deepcopy(new_token_ids),
                        image_transform=None,
                        images=[],
                        prompt=prompt,
                        max_length=text_max_len,
                        do_sample=False,
                        temperature=1.0,
                    )
            else:
                out = preflight_inferencer.interleave_inference(
                    input_lists=[prompt],
                    think=use_think,
                    understanding_output=True,
                    max_think_token_n=text_max_len,
                    do_sample=False,
                    text_temperature=1.0,
                    citation=use_citation,
                )
                answer = ""
                for item in out:
                    if isinstance(item, str):
                        answer = item
                        break
            text_records.append(
                {
                    "idx": int(idx),
                    "prompt": user_prompt,
                    "answer": answer,
                }
            )
            if rank == 0:
                logger.info(f"[Preflight/Text] idx={idx:02d} prompt='{user_prompt}' | answer='{answer}'")

        if rank == 0 and len(text_records) > 0:
            with open(preflight_text_path, "w", encoding="utf-8") as f:
                for rec in text_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"[Preflight/Text] saved: {preflight_text_path}")

        # 2) text-to-image sanity checks via generate_image() + VAE decode
        img_n = max(0, int(training_args.preflight_img_n))
        img_steps = max(2, int(training_args.preflight_img_steps))
        if img_n > 0:
            if (not training_args.visual_gen) or vae_model is None or vae_config is None:
                if rank == 0:
                    logger.warning("[Preflight/Img] skipped: visual_gen is disabled.")
            else:
                img_prompts = _default_preflight_image_prompts()
                max_side = int(model.max_latent_size * model.latent_downsample)
                side = int(training_args.preflight_img_size)
                side = max(model.latent_downsample, min(side, max_side))
                side = max(model.latent_downsample, (side // model.latent_downsample) * model.latent_downsample)
                latent_h = side // model.latent_downsample
                latent_w = side // model.latent_downsample
                vae_dtype = next(vae_model.parameters()).dtype

                for idx in range(img_n):
                    user_prompt = img_prompts[idx % len(img_prompts)]
                    prompt = _build_chat_prompt(user_prompt)
                    reasoning_text = None
                    if preflight_inferencer is None:
                        past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
                        text_input, newlens, new_rope = model.prepare_prompts(
                            curr_kvlens=[0],
                            curr_rope=[0],
                            prompts=[prompt],
                            tokenizer=tokenizer,
                            new_token_ids=deepcopy(new_token_ids),
                        )
                        text_input = _move_tensor_dict_to_device(text_input, device_obj)

                        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            past_key_values = model.forward_cache_update_text(past_key_values, **text_input)
                            image_input = model.prepare_vae_latent(
                                curr_kvlens=newlens,
                                curr_rope=new_rope,
                                image_sizes=[(side, side)],
                                new_token_ids=deepcopy(new_token_ids),
                            )
                            image_input = _move_tensor_dict_to_device(image_input, device_obj)
                            unpacked_latent = model.generate_image(
                                past_key_values=past_key_values,
                                num_timesteps=img_steps,
                                timestep_shift=training_args.timestep_shift,
                                show_progress=(rank == 0),
                                **image_input,
                            )

                        pred_patches = unpacked_latent[0].detach().float().cpu()
                        pred_latent = _unpatchify_latent_patches(
                            pred_patches,
                            latent_h,
                            latent_w,
                            model_args.latent_patch_size,
                            vae_config.z_channels,
                        )
                        with torch.no_grad():
                            pred_img = vae_model.decode(pred_latent.to(device=device_obj, dtype=vae_dtype))[0]
                        pred_img_pil = _to_pil_image(pred_img)
                    else:
                        out = preflight_inferencer.interleave_inference(
                            input_lists=[prompt],
                            think=use_think,
                            understanding_output=False,
                            max_think_token_n=text_max_len,
                            do_sample=False,
                            text_temperature=1.0,
                            cfg_text_scale=1.0,
                            cfg_img_scale=1.0,
                            timestep_shift=training_args.timestep_shift,
                            num_timesteps=img_steps,
                            image_shapes=(side, side),
                            enable_taylorseer=False,
                            citation=use_citation,
                        )
                        pred_img_pil = None
                        for item in out:
                            if isinstance(item, str):
                                reasoning_text = item
                            elif isinstance(item, Image.Image):
                                pred_img_pil = item
                        if pred_img_pil is None:
                            raise RuntimeError("[Preflight/Img] no image returned from interleave_inference.")

                    if rank == 0:
                        img_path = os.path.join(preflight_image_root, f"img_{idx:02d}.png")
                        meta_path = os.path.join(preflight_image_root, f"img_{idx:02d}.json")
                        pred_img_pil.save(img_path)
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "idx": int(idx),
                                    "prompt": user_prompt,
                                    "image_path": img_path,
                                    "num_timesteps": int(img_steps),
                                    "image_size": [int(side), int(side)],
                                    "reasoning_text": reasoning_text,
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                        logger.info(
                            f"[Preflight/Img] idx={idx:02d} steps={img_steps} size={side}x{side} "
                            f"saved: {img_path}"
                        )
    except Exception as e:
        failed.fill_(1)
        error_msg = str(e)
        logger.exception(f"[Preflight] failed on rank={rank}: {e}")

    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if int(failed.item()) > 0:
        raise RuntimeError(f"Preflight inference failed (rank{rank} error='{error_msg}').")

    # Keep original FSDP memory behavior: wrap from CPU model.
    model.cpu()
    if training_args.visual_gen and vae_model is not None:
        vae_model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        logger.info("[Preflight] done.")


def maybe_apply_lora(model: torch.nn.Module, training_args: TrainingArguments, logger, tokenizer=None, num_new_tokens=0):
    if not getattr(training_args, "use_lora", False):
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "LoRA requested but peft is not installed. Please run: pip install peft"
        ) from e

    # Freeze all parameters first; PEFT will mark LoRA params trainable.
    for p in model.parameters():
        p.requires_grad = False

    target_modules = [s.strip() for s in training_args.lora_target_modules.split(",") if s.strip()]

    # Exact validation: ensure each target module name exists in this model.
    module_names = [name for name, _ in model.named_modules()]
    missing = []
    for t in target_modules:
        found = any(name.endswith(f".{t}") or name == t for name in module_names)
        if not found:
            missing.append(t)

    if missing:
        suffixes = set()
        for name, mod in model.named_modules():
            if hasattr(mod, "weight") and hasattr(mod, "bias") and mod.__class__.__name__.lower().endswith("linear"):
                suffixes.add(name.split(".")[-1])
        suffixes = sorted(suffixes)

        raise RuntimeError(
            "LoRA target_modules contains names that do not exist in the current LLM module.\n"
            f"Missing: {missing}\n"
            f"Available leaf module suffix candidates (from Linear-like modules): {suffixes}\n"
            "Please set --lora_target_modules to a comma-separated list that matches the actual module names."
        )

    lora_cfg = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=target_modules,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # LatentCOT: new special-token embeddings need full fine-tuning while
    # old vocab rows stay frozen. Use a gradient hook to zero out gradients
    # for old vocabulary positions.
    if num_new_tokens > 0:
        _old_vocab_size = len(tokenizer) - num_new_tokens

        def _mask_old_vocab_grad(grad):
            grad[:_old_vocab_size] = 0.0
            return grad

        for _n, _p in model.named_parameters():
            if "embed_tokens.weight" in _n or "lm_head.weight" in _n:
                _p.requires_grad = True
                _p.register_hook(_mask_old_vocab_grad)
        if dist.get_rank() == 0:
            _emb_trainable = sum(
                _p.numel() for _n, _p in model.named_parameters()
                if _p.requires_grad and ("embed_tokens" in _n or "lm_head" in _n)
            )
            logger.info(
                f"[LatentCOT] Unfrozen embed+lm_head (old={_old_vocab_size} frozen, "
                f"new={num_new_tokens} trainable → {_emb_trainable} params)"
            )

    if dist.get_rank() == 0:
        try:
            model.print_trainable_parameters()
        except Exception:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(f"[LoRA] trainable params: {trainable} / {total} ({100*trainable/total:.4f}%)")

    return model

def log_trainable_params(tag: str, module: torch.nn.Module, logger, max_lines: int = 80):
    """
    Deterministically verify what's being trained.
    Prints trainable parameter tensor names on rank0 and checks whether LoRA params exist.
    """
    if dist.get_rank() != 0:
        return

    trainable = [(n, p.numel()) for n, p in module.named_parameters() if p.requires_grad]
    total_trainable = sum(n for _, n in trainable)
    logger.info(f"[{tag}] trainable tensors: {len(trainable)} | trainable params: {total_trainable:,}")

    # print a few names for sanity
    for n, numel in trainable[:max_lines]:
        logger.info(f"[{tag}] trainable: {n} ({numel:,})")

    has_lora = any("lora_" in n for n, _ in trainable)
    if has_lora:
        logger.info(f"[{tag}] ✅ LoRA params detected (contains 'lora_').")
    else:
        logger.warning(f"[{tag}] ❌ No LoRA params detected (no 'lora_' in trainable names).")


def cast_module_floats_to_bf16(module: torch.nn.Module):
    """
    Cast floating-point params/buffers to bf16 in-place.
    Helps reduce FSDP init-time peak memory on 40GB cards.
    """
    return module.to(dtype=torch.bfloat16)

@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    grad_diag_enable: bool = field(
        default=False,
        metadata={"help": "Enable gradient diagnostics: print CE/MSE gradient norms and ratios."}
    )
    grad_diag_cosine_enable: bool = field(
        default=False,
        metadata={"help": "When grad_diag_enable=True, also compute CE-vs-MSE gradient cosine similarity on trainable params."}
    )
    grad_diag_every: int = field(
        default=100,
        metadata={"help": "Run gradient diagnostics every N steps when enabled."}
    )
    latent_context_noise_std: float = field(
        default=0.0,
        metadata={"help": "Std of Gaussian noise injected into VAE latents during training (LatentCOT)."}
    )
    latent_mse_weight: float = field(
        default=1.0,
        metadata={"help": "MSE weight for intermediate latent_step VAE outputs (LatentCOT)."}
    )
    mse_window_enable: bool = field(
        default=False,
        metadata={"help": "Track window stats of MSE supervision coverage (active-step ratio and token distribution)."}
    )
    mse_window_size: int = field(
        default=100,
        metadata={"help": "Window size (in optimizer steps) for mse_window statistics."}
    )
    mse_window_log_every: int = field(
        default=20,
        metadata={"help": "Log mse_window stats every N optimizer steps when enabled."}
    )
    overfit_single_batch: bool = field(
        default=False,
        metadata={"help": "Debug-only: repeatedly train on the first observed batch to test overfit behavior."}
    )
    overfit_single_batch_steps: int = field(
        default=200,
        metadata={"help": "Number of optimizer steps to keep using the fixed batch when overfit_single_batch=True."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    save_optimizer_state: bool = field(
        default=True,
        metadata={"help": "Save optimizer/scheduler state in checkpoints. Disable for inference-only overfit checkpoints."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    gen_lr_scale: float = field(
        default=1.0,
        metadata={"help": "Learning-rate multiplier for MoT generation-branch LoRA params (names containing 'moe_gen')."}
    )
    cosine_min_lr_ratio: float = field(
        default=0.2,
        metadata={"help": "Minimum LR ratio for cosine schedule. 0.2 means min_lr = 20 percent of each param-group base LR."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    disable_ema: bool = field(
        default=False,
        metadata={"help": "Disable EMA model to save host RAM (skip deepcopy + ema updates)."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    training_mode: str = field(
        default="default",
        metadata={"help": "Loss/data mode: default | citation_multiturn."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    cite_ce_weight: float = field(
        default=1.0,
        metadata={"help": "Citation-mode only: weight for cite CE loss."}
    )
    claim_ce_weight: float = field(
        default=1.0,
        metadata={"help": "Citation-mode only: weight for claim CE loss."}
    )
    answer_text_ce_weight: float = field(
        default=1.0,
        metadata={"help": "Citation-mode only: weight for final text answer CE loss."}
    )
    answer_image_mse_weight: float = field(
        default=1.0,
        metadata={"help": "Citation-mode only: weight for final image answer MSE loss."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

    # --- LoRA (minimal) ---
    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA finetuning (train adapters only)."}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias mode: 'none' | 'all' | 'lora_only'."}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen",
        metadata={"help": "Comma-separated module name list to apply LoRA to."}
    )

    # --- official-layout export ---
    export_official_layout: bool = field(
        default=True,
        metadata={"help": "After each checkpoint save (rank0), export official-like 13-file inference package."}
    )
    official_export_dir: str = field(
        default="",
        metadata={"help": "Optional output root for official-layout exports; default is results_dir/official_layout."}
    )
    official_out_dtype: str = field(
        default="bf16",
        metadata={"help": "Output dtype for exported official-layout weights: bf16 | fp16 | fp32 | keep."}
    )
    checkpoint_weight_dtype: str = field(
        default="bf16",
        metadata={"help": "Raw checkpoint model/ema weight dtype on disk: bf16 | fp16 | fp32 | keep."}
    )

    # --- forward text debug (teacher-forcing) ---
    debug_forward_text: bool = field(
        default=False,
        metadata={"help": "Print teacher-forcing predicted text snippets from CE branch for sanity checking."}
    )
    debug_forward_interval: int = field(
        default=100,
        metadata={"help": "Debug period in steps. e.g. 100 means every 100-step window."}
    )
    debug_forward_first_n: int = field(
        default=10,
        metadata={"help": "Within each debug window, print for the first N steps."}
    )
    debug_forward_max_tokens: int = field(
        default=48,
        metadata={"help": "Maximum decoded token count for teacher-forcing debug snippets."}
    )

    # --- preflight inference (before training) ---
    preflight_infer: bool = field(
        default=False,
        metadata={"help": "Run pre-training inference sanity checks on all ranks; only rank0 saves outputs."}
    )
    preflight_text_n: int = field(
        default=2,
        metadata={"help": "Number of text QA preflight cases to run with Bagel.chat()."}
    )
    preflight_text_max_length: int = field(
        default=96,
        metadata={"help": "Max new tokens for each preflight text case."}
    )
    preflight_reasoning_mode: str = field(
        default="none",
        metadata={"help": "Reasoning mode for preflight inference: none, think, or citation."}
    )
    preflight_img_n: int = field(
        default=1,
        metadata={"help": "Number of preflight text-to-image cases to run with generate_image()."}
    )
    preflight_img_steps: int = field(
        default=8,
        metadata={"help": "Diffusion steps for each preflight image generation case."}
    )
    preflight_img_size: int = field(
        default=256,
        metadata={"help": "Square output size for preflight image generation (will be adjusted to latent stride)."}
    )


def main():
    # Parse --help and invalid arguments before touching CUDA or distributed state.
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    if training_args.training_mode not in {"default", "citation_multiturn"}:
        raise ValueError(
            f"Unknown training_mode={training_args.training_mode}. "
            "Expected one of: default, citation_multiturn."
        )
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    resume_from, resume_model_only, finetune_from_ema = _resolve_finetune_init_source(
        training_args=training_args,
        model_args=model_args,
        resume_from=resume_from,
        resume_model_only=resume_model_only,
        finetune_from_ema=finetune_from_ema,
        logger=logger,
    )

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_source_path = (
            os.path.join(model_args.model_path, "ae.safetensors")
            if training_args.finetune_from_hf else model_args.vae_path
        )
        vae_model, vae_config = load_ae(
            local_path=vae_source_path
        )
    else:
        vae_source_path = None

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        latent_context_noise_std=training_args.latent_context_noise_std,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        tokenizer_vocab_size = len(tokenizer)
        checkpoint_vocab_size = int(getattr(model.language_model.config, "vocab_size", tokenizer_vocab_size))
        target_vocab_size = max(tokenizer_vocab_size, checkpoint_vocab_size)
        if target_vocab_size != checkpoint_vocab_size or target_vocab_size != tokenizer_vocab_size:
            logger.info(
                f"[LatentCOT] keep padded vocab for checkpoint load: "
                f"tokenizer={tokenizer_vocab_size}, checkpoint/model={checkpoint_vocab_size}, target={target_vocab_size}"
            )
        model.language_model.resize_token_embeddings(target_vocab_size)
        model.config.llm_config.vocab_size = target_vocab_size
        model.language_model.config.vocab_size = target_vocab_size

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
    if training_args.freeze_und and training_args.visual_und:
        model.connector.eval()
        for param in model.connector.parameters():
            param.requires_grad = False

    if training_args.use_lora:
        # LoRA finetuning: freeze all non-LoRA parameters globally.
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # Keep visual generation flow heads trainable so MSE can backprop through
        # noised_latent -> vae2llm -> LM -> llm2vae.
        if training_args.visual_gen:
            flow_heads = ["llm2vae", "vae2llm", "time_embedder"]
            for head_name in flow_heads:
                head = getattr(model, head_name, None)
                if head is None:
                    continue
                for param in head.parameters():
                    param.requires_grad = True

        # If understanding branch is frozen, only keep moe_gen LoRA trainable.
        if training_args.freeze_und:
            for name, param in model.named_parameters():
                if "lora_" in name and "moe_gen" not in name:
                    param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )

    use_orig_params = bool(getattr(training_args, "use_lora", False))
    use_ema = (not getattr(training_args, "disable_ema", False))
    trainable_only_checkpoint = (
        bool(training_args.finetune_from_hf)
        and (not use_ema)
        and int(num_new_tokens) == 0
    )
    resume_from_trainable_only_ckpt = _is_trainable_only_checkpoint_dir(resume_from)
    defer_lora_until_after_base_load = (
        bool(training_args.use_lora)
        and bool(training_args.finetune_from_hf)
        and bool(resume_model_only)
        and (resume_from is not None)
        and (not _is_training_checkpoint_dir(resume_from))
    )
    if resume_from_trainable_only_ckpt:
        defer_lora_until_after_base_load = False
    if dist.get_rank() == 0 and training_args.use_lora:
        logger.info(
            f"[LoRAInit] defer_lora_until_after_base_load={defer_lora_until_after_base_load} "
            f"(resume_from={resume_from}, resume_model_only={resume_model_only})"
        )
    if dist.get_rank() == 0:
        logger.info(
            f"[CheckpointMode] trainable_only_checkpoint={trainable_only_checkpoint} "
            f"(finetune_from_hf={training_args.finetune_from_hf}, use_ema={use_ema}, num_new_tokens={num_new_tokens})"
        )
    if dist.get_rank() == 0 and resume_from_trainable_only_ckpt:
        logger.info(
            f"[InitGuard] resume_from is a trainable-only training checkpoint: {resume_from}; "
            "will bootstrap from base model_path and then overlay trainable checkpoint weights."
        )

    if resume_from_trainable_only_ckpt:
        model, _ = FSDPCheckpoint.try_load_ckpt(
            model_args.model_path, logger, model, None, resume_from_ema=finetune_from_ema
        )
        if training_args.use_lora:
            model.language_model = maybe_apply_lora(model.language_model, training_args, logger, tokenizer, num_new_tokens)
            log_trainable_params("LLM_AFTER_LORA", model.language_model, logger)
        ema_model = deepcopy(model) if use_ema else None
        model, ema_model = FSDPCheckpoint.try_load_ckpt(
            resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
        )
    elif defer_lora_until_after_base_load:
        # Critical fix: load official/base checkpoint into plain model key-space first,
        # then inject LoRA adapters to avoid key-prefix mismatch.
        model, _ = FSDPCheckpoint.try_load_ckpt(
            resume_from, logger, model, None, resume_from_ema=finetune_from_ema
        )
        model.language_model = maybe_apply_lora(model.language_model, training_args, logger)
        log_trainable_params("LLM_AFTER_LORA", model.language_model, logger)
        ema_model = deepcopy(model) if use_ema else None
    else:
        if training_args.use_lora:
            model.language_model = maybe_apply_lora(model.language_model, training_args, logger, tokenizer, num_new_tokens)
            log_trainable_params("LLM_AFTER_LORA", model.language_model, logger)
        ema_model = deepcopy(model) if use_ema else None
        model, ema_model = FSDPCheckpoint.try_load_ckpt(
            resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
        )

    # Reduce FSDP wrap-time peak memory by casting model weights to bf16 first.
    # This matches the intended bf16 training setup.
    model = cast_module_floats_to_bf16(model)
    if ema_model is not None:
        ema_model = cast_module_floats_to_bf16(ema_model)
    if dist.get_rank() == 0:
        logger.info("[DType] Cast model (and EMA if enabled) floating weights to bf16 before FSDP wrap.")

    _run_preflight_inference(
        training_args=training_args,
        model_args=model_args,
        model=model,
        vae_model=vae_model if training_args.visual_gen else None,
        vae_config=vae_config if training_args.visual_gen else None,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        device=device,
        logger=logger,
    )

    if use_ema:
        ema_model = fsdp_ema_setup(ema_model, fsdp_config, use_orig_params=use_orig_params)

    fsdp_model = fsdp_wrapper(model, fsdp_config, use_orig_params=use_orig_params)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        # check what is trainable AFTER FSDP wrapping
        trainable_names = [n for n, p in fsdp_model.named_parameters() if p.requires_grad]
        logger.info(f"[FSDP_AFTER_WRAP] trainable tensors: {len(trainable_names)}")
        for n in trainable_names[:120]:
            logger.info(f"[FSDP_AFTER_WRAP] trainable: {n}")

    # Setup optimizer and scheduler (ONLY trainable parameters, e.g. LoRA adapters)
    trainable_named = [(n, p) for n, p in fsdp_model.named_parameters() if p.requires_grad]
    if training_args.use_lora:
        gen_params = [p for n, p in trainable_named if "moe_gen" in n]
        main_params = [p for n, p in trainable_named if "moe_gen" not in n]
        param_groups = []
        if len(main_params) > 0:
            param_groups.append(
                {
                    "params": main_params,
                    "lr": training_args.lr,
                    "group_name": "main",
                }
            )
        if len(gen_params) > 0:
            param_groups.append(
                {
                    "params": gen_params,
                    "lr": training_args.lr * training_args.gen_lr_scale,
                    "group_name": "moe_gen",
                }
            )
        if dist.get_rank() == 0:
            logger.info(
                f"[OPTIM] param groups: main={len(main_params)} tensors @ {training_args.lr:g}, "
                f"moe_gen={len(gen_params)} tensors @ {(training_args.lr * training_args.gen_lr_scale):g}"
            )
    else:
        gen_params = []
        param_groups = [
            {
                "params": [p for _, p in trainable_named],
                "lr": training_args.lr,
                "group_name": "main",
            }
        ]
        main_params = param_groups[0]["params"]

    grad_diag_all_params = [p for _, p in trainable_named]
    grad_diag_main_params = list(main_params)
    grad_diag_gen_params = list(gen_params)

    def _global_grad_l2(params):
        grad_sq = torch.zeros((), device=device, dtype=torch.float64)
        for p in params:
            if p.grad is not None:
                g = p.grad.detach()
                grad_sq = grad_sq + g.float().pow(2).sum().to(torch.float64)
        dist.all_reduce(grad_sq, op=dist.ReduceOp.SUM)
        return grad_sq.sqrt().item()

    def _clone_batch_value(v):
        if torch.is_tensor(v):
            return v.clone()
        if isinstance(v, list):
            return [_clone_batch_value(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_clone_batch_value(x) for x in v)
        if isinstance(v, dict):
            return {k: _clone_batch_value(x) for k, x in v.items()}
        return deepcopy(v)

    def _clone_batch_dict(d: dict) -> dict:
        return {k: _clone_batch_value(v) for k, v in d.items()}

    if (
        training_args.grad_diag_enable
        and training_args.gradient_accumulation_steps != 1
        and dist.get_rank() == 0
    ):
        logger.warning(
            "[GradDiag] gradient_accumulation_steps != 1. "
            "Diagnostics are skipped to avoid perturbing accumulation behavior."
        )
    if (
        training_args.grad_diag_enable
        and training_args.gradient_accumulation_steps == 1
        and training_args.grad_diag_every != 1
        and dist.get_rank() == 0
    ):
        logger.warning(
            "[GradDiag] Diagnostics run extra forward/backward passes. "
            "With FSDP, this must be consistent every step to avoid exec-order/collective mismatch. "
            "Overriding grad_diag_every to 1."
        )
    if training_args.grad_diag_cosine_enable and (not training_args.grad_diag_enable) and dist.get_rank() == 0:
        logger.warning(
            "[GradDiag] grad_diag_cosine_enable=True but grad_diag_enable=False; cosine probe will be skipped."
        )
    grad_diag_mode_supported = training_args.training_mode == "default"
    if training_args.grad_diag_enable and (not grad_diag_mode_supported) and dist.get_rank() == 0:
        logger.warning(
            f"[GradDiag] training_mode={training_args.training_mode} is not supported for grad diagnostics; "
            "grad_diag will be skipped."
        )
    if training_args.overfit_single_batch and dist.get_rank() == 0:
        logger.warning(
            f"[OverfitProbe] enabled for first {max(1, int(training_args.overfit_single_batch_steps))} optimizer steps; "
            "all steps will reuse one fixed batch."
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        min_lr_ratio = float(training_args.cosine_min_lr_ratio)
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(
                f"cosine_min_lr_ratio must be in [0, 1], got {training_args.cosine_min_lr_ratio}"
            )

        # Prefer min_lr_rate so every optimizer param-group (e.g. main + moe_gen)
        # uses the same floor ratio to its own base LR; fall back for older transformers.
        try:
            scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=training_args.warmup_steps,
                num_training_steps=training_args.total_steps,
                min_lr_rate=min_lr_ratio,
            )
            cosine_floor_mode = "min_lr_rate"
        except TypeError as e:
            if "min_lr_rate" not in str(e):
                raise
            scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=training_args.warmup_steps,
                num_training_steps=training_args.total_steps,
                min_lr=float(training_args.lr) * min_lr_ratio,
            )
            cosine_floor_mode = "min_lr"
        if dist.get_rank() == 0:
            logger.info(
                f"[LRScheduler] cosine floor mode={cosine_floor_mode}, "
                f"cosine_min_lr_ratio={min_lr_ratio:g}"
            )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, logger=logger,
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=None if data_args.num_workers == 0 else data_args.prefetch_factor,
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    if training_args.official_export_dir and str(training_args.official_export_dir).strip() != "":
        official_export_root = training_args.official_export_dir
    else:
        official_export_root = os.path.join(training_args.results_dir, "official_layout")
    if dist.get_rank() == 0 and training_args.export_official_layout and training_args.finetune_from_hf:
        os.makedirs(official_export_root, exist_ok=True)
        logger.info(f"[OfficialExport] root dir: {official_export_root}")
    debug_image_root = os.path.join(training_args.results_dir, "debug_forward_images")
    debug_text_root = os.path.join(training_args.results_dir, "debug_forward_text")
    if dist.get_rank() == 0 and training_args.debug_forward_text and training_args.visual_gen:
        os.makedirs(debug_image_root, exist_ok=True)
        logger.info(f"[FwdImgDbg] root dir: {debug_image_root}")
    if dist.get_rank() == 0 and training_args.debug_forward_text:
        os.makedirs(debug_text_root, exist_ok=True)
        logger.info(f"[StepDbg] text root dir: {debug_text_root}")

    def _need_forward_text_debug(step: int) -> bool:
        if not training_args.debug_forward_text:
            return False
        if training_args.debug_forward_interval <= 0:
            return False
        mod = step % training_args.debug_forward_interval
        return mod < max(0, training_args.debug_forward_first_n)

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)

    # tqdm: show progress only on rank0 to avoid multiple progress bars
    pbar = None
    if dist.get_rank() == 0:
        pbar = tqdm(
            total=training_args.total_steps,
            initial=train_step,
            dynamic_ncols=True,
            desc="train",
        )

    overfit_batch_cache = None
    overfit_probe_stop_step = train_step + max(1, int(training_args.overfit_single_batch_steps))
    overfit_probe_finish_logged = False
    mse_window_tokens = []
    mse_window_stats = None

    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        grad_diag_stats = None
        do_grad_diag = (
            training_args.grad_diag_enable
            and grad_diag_mode_supported
            and training_args.gradient_accumulation_steps == 1
            and (curr_step % 1 == 0)
        )

        if pbar is not None:
            # keep tqdm in sync with curr_step (handles any non-1 accumulation cleanly)
            pbar.n = curr_step
            pbar.refresh()

        data = data.cuda(device).to_dict()
        if training_args.overfit_single_batch and curr_step < overfit_probe_stop_step:
            if overfit_batch_cache is None:
                overfit_batch_cache = _clone_batch_dict(data)
                if dist.get_rank() == 0:
                    logger.info(
                        f"[OverfitProbe] captured fixed batch at step={curr_step:07d}, "
                        f"sequence_length={int(overfit_batch_cache.get('sequence_length', 0))}"
                    )
            data = _clone_batch_dict(overfit_batch_cache)
        elif training_args.overfit_single_batch and (not overfit_probe_finish_logged):
            overfit_probe_finish_logged = True
            if dist.get_rank() == 0:
                logger.info(
                    f"[OverfitProbe] finished at step={curr_step:07d}; "
                    "switching back to normal streaming batches."
                )
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        ce_loss_group_ids = data.pop('ce_loss_group_ids', None)
        mse_loss_group_ids = data.pop('mse_loss_group_ids', None)
        if _need_forward_text_debug(curr_step) and dist.get_rank() == 0:
            local_ce_dbg = int(data['ce_loss_indexes'].numel()) if 'ce_loss_indexes' in data else 0
            local_mse_dbg = int(data['mse_loss_indexes'].numel()) if 'mse_loss_indexes' in data else 0
            logger.info(
                f"[BatchDbg] step={curr_step:07d} sequence_length={int(data['sequence_length'])} "
                f"samples={len(data.get('sample_lens', []))} local_ce_tokens={local_ce_dbg} local_mse_tokens={local_mse_dbg}"
            )
            latent_shapes_dbg = data.get("patchified_vae_latent_shapes", None)
            if isinstance(latent_shapes_dbg, list) and len(latent_shapes_dbg) > 0:
                latent_tokens_dbg = [int(h) * int(w) for h, w in latent_shapes_dbg]
                total_latent_tokens_dbg = int(sum(latent_tokens_dbg))
                supervised_ratio = (float(local_mse_dbg) / float(max(total_latent_tokens_dbg, 1)))
                logger.info(
                    f"[BatchDbgImg] step={curr_step:07d} vae_images={len(latent_shapes_dbg)} "
                    f"latent_tokens[min/mean/max]={min(latent_tokens_dbg)}/"
                    f"{(total_latent_tokens_dbg / max(len(latent_tokens_dbg), 1)):.1f}/{max(latent_tokens_dbg)} "
                    f"supervised_ratio={supervised_ratio:.4f}"
                )
            if isinstance(data_indexes, list):
                for i, item in enumerate(data_indexes):
                    logger.info(
                        f"[BatchDbgSample] step={curr_step:07d} sample={i:02d} meta={_summarize_sample_meta(item)}"
                    )
        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
            dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
            seqlen_square_window += sample_square.item()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            has_gen_batch = (
                training_args.visual_gen
                and ('padded_images' in data)
                and ('mse_loss_indexes' in data)
                and (len(data.get('mse_loss_indexes', [])) > 0)
            )
            if has_gen_batch:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            try:
                debug_return = _need_forward_text_debug(curr_step)
                loss_dict = fsdp_model(
                    **data,
                    debug_return_ce_pred_ids=debug_return,
                    debug_return_mse_pred_latent=debug_return,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e

        if _need_forward_text_debug(curr_step):
            debug_pred_ids = loss_dict.pop("debug_ce_pred_ids", None)
            debug_label_ids = loss_dict.pop("debug_label_ids", None)
            debug_mse_pred_patches = loss_dict.pop("debug_mse_pred_patches", None)
            debug_mse_clean_patches = loss_dict.pop("debug_mse_clean_patches", None)
            debug_mse_target_patches = loss_dict.pop("debug_mse_target_patches", None)
            debug_mse_align_exact = loss_dict.pop("debug_mse_align_exact", None)
            debug_mse_align_len_loss = loss_dict.pop("debug_mse_align_len_loss", None)
            debug_mse_align_len_t = loss_dict.pop("debug_mse_align_len_t", None)
            debug_mse_align_first_mismatch = loss_dict.pop("debug_mse_align_first_mismatch", None)
            debug_mse_pred_abs_mean = loss_dict.pop("debug_mse_pred_abs_mean", None)
            debug_mse_target_abs_mean = loss_dict.pop("debug_mse_target_abs_mean", None)
            debug_mse_t_min = loss_dict.pop("debug_mse_t_min", None)
            debug_mse_t_max = loss_dict.pop("debug_mse_t_max", None)
            debug_mse_t_mean = loss_dict.pop("debug_mse_t_mean", None)
            if dist.get_rank() == 0:
                sample_count = len(data_indexes) if isinstance(data_indexes, list) else None
                sample_spans = _build_sample_spans(data.get("sample_lens", []), sample_count)
                sample_metas = data_indexes if isinstance(data_indexes, list) else [None] * len(sample_spans)
                def _scalar_or_default(v, default=0.0):
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 0:
                            return default
                        return float(v.detach().cpu().reshape(-1)[0].item())
                    if v is None:
                        return default
                    try:
                        return float(v)
                    except Exception:
                        return default

                align_exact_i = int(round(_scalar_or_default(debug_mse_align_exact, 1.0)))
                align_len_loss_i = int(round(_scalar_or_default(debug_mse_align_len_loss, 0.0)))
                align_len_t_i = int(round(_scalar_or_default(debug_mse_align_len_t, 0.0)))
                align_first_mismatch_i = int(round(_scalar_or_default(debug_mse_align_first_mismatch, -1.0)))
                pred_abs_mean_f = _scalar_or_default(debug_mse_pred_abs_mean, 0.0)
                target_abs_mean_f = _scalar_or_default(debug_mse_target_abs_mean, 0.0)
                t_min_f = _scalar_or_default(debug_mse_t_min, 0.0)
                t_max_f = _scalar_or_default(debug_mse_t_max, 0.0)
                t_mean_f = _scalar_or_default(debug_mse_t_mean, 0.0)
                logger.info(
                    f"[MSEDbgAlign] step={curr_step:07d} exact={align_exact_i} "
                    f"len_loss={align_len_loss_i} len_t={align_len_t_i} first_mismatch={align_first_mismatch_i}"
                )
                logger.info(
                    f"[MSEDbgScale] step={curr_step:07d} pred_abs_mean={pred_abs_mean_f:.6e} "
                    f"target_abs_mean={target_abs_mean_f:.6e} t[min/mean/max]={t_min_f:.6f}/{t_mean_f:.6f}/{t_max_f:.6f}"
                )
                if align_exact_i == 0:
                    logger.warning(
                        f"[MSEDbgAlign][WARN] step={curr_step:07d} MSE index alignment mismatch detected "
                        f"(len_loss={align_len_loss_i}, len_t={align_len_t_i}, first_mismatch={align_first_mismatch_i})."
                    )

                try:
                    ce_report_path = _save_ce_debug_records(
                        curr_step=curr_step,
                        data=data,
                        sample_spans=sample_spans,
                        sample_metas=sample_metas,
                        debug_pred_ids=debug_pred_ids,
                        debug_label_ids=debug_label_ids,
                        tokenizer=tokenizer,
                        max_tokens=max(1, int(training_args.debug_forward_max_tokens)),
                        out_dir=debug_text_root,
                        logger=logger,
                    )
                    if ce_report_path is not None:
                        logger.info(f"[StepDbg] step={curr_step:07d} ce records: {ce_report_path}")
                except Exception as e:
                    logger.warning(f"[StepDbg] step={curr_step:07d} ce_save_failed: {e}")

                if training_args.visual_gen:
                    try:
                        _save_mse_debug_previews(
                            curr_step=curr_step,
                            data=data,
                            sample_spans=sample_spans,
                            debug_mse_pred_patches=debug_mse_pred_patches,
                            debug_mse_clean_patches=debug_mse_clean_patches,
                            debug_mse_target_patches=debug_mse_target_patches,
                            vae_model=vae_model,
                            device=device,
                            latent_patch_size=model_args.latent_patch_size,
                            latent_channel=vae_config.z_channels,
                            timestep_shift=training_args.timestep_shift,
                            out_dir=debug_image_root,
                            logger=logger,
                        )
                    except Exception as e:
                        logger.warning(f"[FwdImgDbg] step={curr_step:07d} save_failed: {e}")
        
        loss = 0
        ce_vec = loss_dict["ce"]
        mse_vec = loss_dict["mse"]

        def _reduce_ce_with_mask(mask: torch.Tensor):
            local_tokens = int(mask.sum().item()) if mask.numel() > 0 else 0
            total_tokens = torch.tensor(local_tokens, device=device)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

            if total_tokens.item() <= 0:
                return ce_vec.sum() * 0.0, total_tokens

            if training_args.ce_loss_reweighting and ce_loss_weights is not None and ce_loss_weights.numel() > 0:
                masked_ce = ce_vec[mask] * ce_loss_weights[mask]
                total_weights = ce_loss_weights[mask].sum()
                dist.all_reduce(total_weights, op=dist.ReduceOp.SUM)
                if total_weights.item() > 0:
                    ce_out = masked_ce.sum() * dist.get_world_size() / total_weights
                else:
                    ce_out = ce_vec[mask].sum() * dist.get_world_size() / total_tokens
            else:
                ce_out = ce_vec[mask].sum() * dist.get_world_size() / total_tokens
            return ce_out, total_tokens

        def _reduce_mse_with_mask(mask: torch.Tensor):
            local_tokens = int(mask.sum().item()) if mask.numel() > 0 else 0
            total_tokens = torch.tensor(local_tokens, device=device)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

            if total_tokens.item() <= 0:
                return mse_vec.sum() * 0.0, total_tokens

            mse_token = mse_vec.mean(dim=-1) if mse_vec.dim() >= 2 else mse_vec
            mse_out = mse_token[mask].sum() * dist.get_world_size() / total_tokens
            return mse_out, total_tokens

        if training_args.training_mode == "citation_multiturn":
            if ce_loss_group_ids is None or ce_loss_group_ids.numel() == 0:
                raise RuntimeError(
                    "training_mode=citation_multiturn requires ce_loss_group_ids from dataset, but none found."
                )
            if ce_loss_group_ids.numel() != ce_vec.numel():
                raise RuntimeError(
                    "ce_loss_group_ids length mismatch: "
                    f"group_ids={int(ce_loss_group_ids.numel())}, ce={int(ce_vec.numel())}"
                )

            cite_mask = ce_loss_group_ids == CE_GROUP_CITE
            claim_mask = ce_loss_group_ids == CE_GROUP_CLAIM
            answer_text_mask = ce_loss_group_ids == CE_GROUP_ANSWER_TEXT

            cite_ce, total_cite_tokens = _reduce_ce_with_mask(cite_mask)
            claim_ce, total_claim_tokens = _reduce_ce_with_mask(claim_mask)
            answer_text_ce, total_answer_text_tokens = _reduce_ce_with_mask(answer_text_mask)

            mse_token_len = int(mse_vec.shape[0]) if mse_vec.dim() >= 1 else 0
            if mse_token_len > 0:
                if mse_loss_group_ids is None or mse_loss_group_ids.numel() == 0:
                    raise RuntimeError(
                        "training_mode=citation_multiturn requires mse_loss_group_ids for image-answer tokens."
                    )
                if mse_loss_group_ids.numel() != mse_token_len:
                    raise RuntimeError(
                        "mse_loss_group_ids length mismatch: "
                        f"group_ids={int(mse_loss_group_ids.numel())}, mse_tokens={mse_token_len}"
                    )
                answer_image_mask = mse_loss_group_ids == MSE_GROUP_ANSWER_IMAGE
            else:
                answer_image_mask = torch.zeros((0,), dtype=torch.bool, device=device)
            answer_image_mse, total_answer_image_tokens = _reduce_mse_with_mask(answer_image_mask)

            total_ce_tokens = total_cite_tokens + total_claim_tokens + total_answer_text_tokens
            total_mse_tokens = total_answer_image_tokens

            ce = cite_ce + claim_ce + answer_text_ce
            mse = answer_image_mse
            weighted_total = (
                cite_ce * training_args.cite_ce_weight
                + claim_ce * training_args.claim_ce_weight
                + answer_text_ce * training_args.answer_text_ce_weight
                + answer_image_mse * training_args.answer_image_mse_weight
            )

            loss_dict["ce_cite"] = cite_ce.detach()
            loss_dict["ce_claim"] = claim_ce.detach()
            loss_dict["ce_answer_text"] = answer_text_ce.detach()
            loss_dict["mse_answer_image"] = answer_image_mse.detach()
            loss_dict["ce"] = ce.detach()
            loss_dict["mse"] = mse.detach()
            loss_dict["loss_total_weighted"] = weighted_total.detach()
            loss = loss + weighted_total
        else:
            ce = ce_vec
            local_ce_tokens = int(data['ce_loss_indexes'].numel()) if 'ce_loss_indexes' in data else 0
            total_ce_tokens = torch.tensor(local_ce_tokens, device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                if ce_loss_weights is not None and ce_loss_weights.numel() > 0:
                    ce = ce * ce_loss_weights
                    total_ce_loss_weights = ce_loss_weights.sum()
                else:
                    total_ce_loss_weights = torch.tensor(0.0, device=device)
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
            if total_ce_tokens.item() > 0:
                if training_args.ce_loss_reweighting and total_ce_loss_weights.item() > 0:
                    ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
                else:
                    ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            else:
                ce = ce.sum() * 0.0
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight

            local_mse_tokens = int(data['mse_loss_indexes'].numel()) if 'mse_loss_indexes' in data else 0
            total_mse_tokens = torch.tensor(local_mse_tokens, device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse_vec
            if total_mse_tokens.item() > 0:
                mse_num = mse.mean(dim=-1).sum() if mse.dim() >= 2 else mse.sum()
                mse = mse_num * dist.get_world_size() / total_mse_tokens
            else:
                mse = mse.sum() * 0.0
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight

        step_total_ce_tokens_i = int(total_ce_tokens.item())
        step_total_mse_tokens_i = int(total_mse_tokens.item())

        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()
        if _need_forward_text_debug(curr_step):
            llm2vae_grad_sq = torch.zeros((), device=device, dtype=torch.float64)
            vae2llm_grad_sq = torch.zeros((), device=device, dtype=torch.float64)
            lm_head_grad_sq = torch.zeros((), device=device, dtype=torch.float64)
            for n, p in fsdp_model.named_parameters():
                if p.grad is None:
                    continue
                g2 = p.grad.detach().float().pow(2).sum().to(torch.float64)
                if "llm2vae" in n:
                    llm2vae_grad_sq = llm2vae_grad_sq + g2
                elif "vae2llm" in n:
                    vae2llm_grad_sq = vae2llm_grad_sq + g2
                elif "lm_head" in n:
                    lm_head_grad_sq = lm_head_grad_sq + g2
            dist.all_reduce(llm2vae_grad_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(vae2llm_grad_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(lm_head_grad_sq, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0:
                llm2vae_grad_l2 = llm2vae_grad_sq.sqrt().item()
                vae2llm_grad_l2 = vae2llm_grad_sq.sqrt().item()
                lm_head_grad_l2 = lm_head_grad_sq.sqrt().item()
                logger.info(
                    f"[MSEDbgGrad] step={curr_step:07d} llm2vae_grad_l2={llm2vae_grad_l2:.4e} "
                    f"vae2llm_grad_l2={vae2llm_grad_l2:.4e} lm_head_grad_l2={lm_head_grad_l2:.4e}"
                )

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            if training_args.mse_window_enable:
                mse_window_tokens.append(step_total_mse_tokens_i)
                max_win = max(1, int(training_args.mse_window_size))
                if len(mse_window_tokens) > max_win:
                    mse_window_tokens = mse_window_tokens[-max_win:]
                if len(mse_window_tokens) > 0 and (curr_step % max(1, int(training_args.mse_window_log_every)) == 0):
                    sorted_tokens = sorted(mse_window_tokens)
                    n = len(sorted_tokens)
                    p50 = float(sorted_tokens[n // 2])
                    p90 = float(sorted_tokens[min(n - 1, int(0.9 * (n - 1)))])
                    active_steps = sum(1 for x in mse_window_tokens if x > 0)
                    active_ratio = float(active_steps) / float(max(n, 1))
                    mean_tokens = float(sum(mse_window_tokens)) / float(max(n, 1))
                    mse_window_stats = {
                        "window": n,
                        "active_ratio": active_ratio,
                        "min": float(sorted_tokens[0]),
                        "mean": mean_tokens,
                        "p50": p50,
                        "p90": p90,
                        "max": float(sorted_tokens[-1]),
                    }
                    if dist.get_rank() == 0:
                        logger.info(
                            f"[MSEWindow] step={curr_step:07d} window={n} active_ratio={active_ratio:.3f} "
                            f"tokens[min/mean/p50/p90/max]={mse_window_stats['min']:.1f}/{mean_tokens:.1f}/"
                            f"{p50:.1f}/{p90:.1f}/{mse_window_stats['max']:.1f}"
                        )
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()

            if do_grad_diag:
                def _diag_one_term(term_name: str, capture_grads: bool = False):
                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        diag_loss_dict = fsdp_model(**data)

                    diag_ce = diag_loss_dict["ce"]
                    diag_local_ce_tokens = int(data['ce_loss_indexes'].numel()) if 'ce_loss_indexes' in data else 0
                    diag_total_ce_tokens = torch.tensor(diag_local_ce_tokens, device=device)
                    dist.all_reduce(diag_total_ce_tokens, op=dist.ReduceOp.SUM)
                    if training_args.ce_loss_reweighting:
                        if ce_loss_weights is not None and ce_loss_weights.numel() > 0:
                            diag_ce = diag_ce * ce_loss_weights
                            diag_total_ce_loss_weights = ce_loss_weights.sum()
                        else:
                            diag_total_ce_loss_weights = torch.tensor(0.0, device=device)
                        dist.all_reduce(diag_total_ce_loss_weights, op=dist.ReduceOp.SUM)
                    if diag_total_ce_tokens.item() > 0:
                        if training_args.ce_loss_reweighting and diag_total_ce_loss_weights.item() > 0:
                            diag_ce = diag_ce.sum() * dist.get_world_size() / diag_total_ce_loss_weights
                        else:
                            diag_ce = diag_ce.sum() * dist.get_world_size() / diag_total_ce_tokens
                    else:
                        diag_ce = diag_ce.sum() * 0.0

                    diag_mse = diag_loss_dict["mse"]
                    diag_local_mse_tokens = int(data['mse_loss_indexes'].numel()) if 'mse_loss_indexes' in data else 0
                    diag_total_mse_tokens = torch.tensor(diag_local_mse_tokens, device=device)
                    dist.all_reduce(diag_total_mse_tokens, op=dist.ReduceOp.SUM)
                    if diag_total_mse_tokens.item() > 0:
                        diag_mse_num = diag_mse.mean(dim=-1).sum() if diag_mse.dim() >= 2 else diag_mse.sum()
                        diag_mse = diag_mse_num * dist.get_world_size() / diag_total_mse_tokens
                    else:
                        diag_mse = diag_mse.sum() * 0.0

                    if term_name == "ce":
                        diag_term = diag_ce * training_args.ce_weight
                    elif term_name == "mse":
                        diag_term = diag_mse * training_args.mse_weight
                    else:
                        raise ValueError(f"Unknown term_name={term_name}")

                    diag_term = diag_term / training_args.gradient_accumulation_steps
                    diag_term.backward()

                    diag_all = _global_grad_l2(grad_diag_all_params)
                    diag_main = _global_grad_l2(grad_diag_main_params) if len(grad_diag_main_params) > 0 else 0.0
                    diag_moe_gen = _global_grad_l2(grad_diag_gen_params) if len(grad_diag_gen_params) > 0 else 0.0

                    captured_grads = None
                    if capture_grads:
                        captured_grads = []
                        for p in grad_diag_all_params:
                            if p.grad is None:
                                captured_grads.append(None)
                            else:
                                captured_grads.append(p.grad.detach().float().clone())

                    optimizer.zero_grad()
                    return (
                        diag_all,
                        diag_main,
                        diag_moe_gen,
                        int(diag_total_ce_tokens.item()),
                        int(diag_total_mse_tokens.item()),
                        captured_grads,
                    )

                need_grad_cosine = bool(training_args.grad_diag_cosine_enable)
                ce_grad_l2, ce_grad_main_l2, ce_grad_moe_gen_l2, diag_ce_tokens, diag_mse_tokens, ce_grads = _diag_one_term(
                    "ce", capture_grads=need_grad_cosine
                )
                mse_grad_l2, mse_grad_main_l2, mse_grad_moe_gen_l2, _, _, mse_grads = _diag_one_term(
                    "mse", capture_grads=need_grad_cosine
                )

                grad_cosine = None
                if need_grad_cosine:
                    dot = torch.zeros((), device=device, dtype=torch.float64)
                    ce_sq = torch.zeros((), device=device, dtype=torch.float64)
                    mse_sq = torch.zeros((), device=device, dtype=torch.float64)
                    for g_ce, g_mse in zip(ce_grads, mse_grads):
                        if g_ce is None or g_mse is None:
                            continue
                        dot = dot + (g_ce * g_mse).sum().to(torch.float64)
                        ce_sq = ce_sq + g_ce.pow(2).sum().to(torch.float64)
                        mse_sq = mse_sq + g_mse.pow(2).sum().to(torch.float64)
                    dist.all_reduce(dot, op=dist.ReduceOp.SUM)
                    dist.all_reduce(ce_sq, op=dist.ReduceOp.SUM)
                    dist.all_reduce(mse_sq, op=dist.ReduceOp.SUM)
                    denom = (ce_sq.sqrt() * mse_sq.sqrt()).item()
                    grad_cosine = dot.item() / max(denom, 1e-12)
                    del ce_grads
                    del mse_grads

                eps = 1e-12
                grad_diag_stats = {
                    "step": curr_step,
                    "ce_grad_l2": ce_grad_l2,
                    "mse_grad_l2": mse_grad_l2,
                    "ratio_mse_over_ce": mse_grad_l2 / max(ce_grad_l2, eps),
                    "ratio_ce_over_mse": ce_grad_l2 / max(mse_grad_l2, eps),
                    "ce_grad_main_l2": ce_grad_main_l2,
                    "ce_grad_moe_gen_l2": ce_grad_moe_gen_l2,
                    "mse_grad_main_l2": mse_grad_main_l2,
                    "mse_grad_moe_gen_l2": mse_grad_moe_gen_l2,
                    "total_ce_tokens": diag_ce_tokens,
                    "total_mse_tokens": diag_mse_tokens,
                }
                if grad_cosine is not None:
                    grad_diag_stats["grad_cosine_ce_mse"] = grad_cosine

                if dist.get_rank() == 0:
                    diag_msg = (
                        f"[GradDiag] step={curr_step:07d}, "
                        f"ce_grad_l2={ce_grad_l2:.4e}, mse_grad_l2={mse_grad_l2:.4e}, "
                        f"mse/ce={grad_diag_stats['ratio_mse_over_ce']:.4e}, ce/mse={grad_diag_stats['ratio_ce_over_mse']:.4e}, "
                        f"ce_main={ce_grad_main_l2:.4e}, ce_moe_gen={ce_grad_moe_gen_l2:.4e}, "
                        f"mse_main={mse_grad_main_l2:.4e}, mse_moe_gen={mse_grad_moe_gen_l2:.4e}, "
                        f"ce_tokens={diag_ce_tokens}, mse_tokens={diag_mse_tokens}"
                    )
                    if grad_cosine is not None:
                        diag_msg += f", cos(ce,mse)={grad_cosine:.4f}"
                    logger.info(diag_msg)
                    print(diag_msg, flush=True)
        
        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            if len(optimizer.param_groups) > 1:
                for i, pg in enumerate(optimizer.param_groups):
                    group_name = pg.get("group_name", f"group{i}")
                    wandb_log[f"lr_{group_name}"] = pg["lr"]
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value
            if grad_diag_stats is not None:
                wandb_log['ce_grad_l2'] = grad_diag_stats['ce_grad_l2']
                wandb_log['mse_grad_l2'] = grad_diag_stats['mse_grad_l2']
                wandb_log['ratio_mse_over_ce'] = grad_diag_stats['ratio_mse_over_ce']
                wandb_log['ratio_ce_over_mse'] = grad_diag_stats['ratio_ce_over_mse']
                wandb_log['ce_grad_main_l2'] = grad_diag_stats['ce_grad_main_l2']
                wandb_log['ce_grad_moe_gen_l2'] = grad_diag_stats['ce_grad_moe_gen_l2']
                wandb_log['mse_grad_main_l2'] = grad_diag_stats['mse_grad_main_l2']
                wandb_log['mse_grad_moe_gen_l2'] = grad_diag_stats['mse_grad_moe_gen_l2']
                if 'grad_cosine_ce_mse' in grad_diag_stats:
                    wandb_log['grad_cosine_ce_mse'] = grad_diag_stats['grad_cosine_ce_mse']
            if mse_window_stats is not None:
                wandb_log['mse_window_active_ratio'] = mse_window_stats['active_ratio']
                wandb_log['mse_window_tokens_min'] = mse_window_stats['min']
                wandb_log['mse_window_tokens_mean'] = mse_window_stats['mean']
                wandb_log['mse_window_tokens_p50'] = mse_window_stats['p50']
                wandb_log['mse_window_tokens_p90'] = mse_window_stats['p90']
                wandb_log['mse_window_tokens_max'] = mse_window_stats['max']

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            # Clear caches and ensure all CUDA operations complete before checkpoint
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
            except RuntimeError as e:
                logger.error(f"Error during gather_object at step {curr_step}: {e}")
                gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list,
                tokenizer=tokenizer,
                bagel_config=config,
                llm_config=llm_config,
                vit_config=vit_config if training_args.visual_und else None,
                vae_config=vae_config if training_args.visual_gen else None,
                vae_source_path=vae_source_path,
                checkpoint_weight_dtype=training_args.checkpoint_weight_dtype,
                save_trainable_only=trainable_only_checkpoint,
                save_optimizer_state=training_args.save_optimizer_state,
            )
            if training_args.export_official_layout and training_args.finetune_from_hf:
                ckpt_step_dir = os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}")
                out_dir = os.path.join(official_export_root, f"{curr_step:07d}")
                if dist.get_rank() == 0:
                    _export_official_layout_from_ckpt(
                        ckpt_dir=ckpt_step_dir,
                        base_model_dir=model_args.model_path,
                        out_dir=out_dir,
                        lora_r=training_args.lora_r,
                        lora_alpha=training_args.lora_alpha,
                        out_dtype=training_args.official_out_dtype,
                        logger=logger,
                    )
            # Clear CUDA cache and force garbage collection after checkpoint to free memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # comment out as an alternative to save the ema model in pt format
            # ema_state_dict = {}
            # for name, param in ema_model.named_parameters():
            #     ema_state_dict[name] = param.detach().cpu()
            
            # torch.save(
            #     ema_state_dict, 
            #     os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}", "ema_standard.pt")
            # )
    
    # Save final checkpoint if not already saved
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        # Clear caches and ensure all CUDA operations complete before final checkpoint
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during final gather_object: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()
        
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, 
            train_steps=curr_step, 
            model=fsdp_model, 
            ema_model=ema_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list,
            tokenizer=tokenizer,
            bagel_config=config,
            llm_config=llm_config,
            vit_config=vit_config if training_args.visual_und else None,
            vae_config=vae_config if training_args.visual_gen else None,
            vae_source_path=vae_source_path,
            checkpoint_weight_dtype=training_args.checkpoint_weight_dtype,
            save_trainable_only=trainable_only_checkpoint,
            save_optimizer_state=training_args.save_optimizer_state,
        )
        if training_args.export_official_layout and training_args.finetune_from_hf:
            ckpt_step_dir = os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}")
            out_dir = os.path.join(official_export_root, f"{curr_step:07d}")
            if dist.get_rank() == 0:
                _export_official_layout_from_ckpt(
                    ckpt_dir=ckpt_step_dir,
                    base_model_dir=model_args.model_path,
                    out_dir=out_dir,
                    lora_r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    out_dtype=training_args.official_out_dtype,
                    logger=logger,
                )
        # Clear CUDA cache and force garbage collection after final checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
