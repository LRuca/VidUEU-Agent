# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import json
import shutil
import gc
from collections import Counter
from dataclasses import asdict, is_dataclass

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file
try:
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
        StateDictOptions,
    )
except Exception:
    get_model_state_dict = None
    set_model_state_dict = None
    StateDictOptions = None

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[], use_orig_params: bool = False):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=use_orig_params,
    )


class FSDPCheckpoint:
    @staticmethod
    def _has_dcp_state_dict_api() -> bool:
        return (
            get_model_state_dict is not None
            and set_model_state_dict is not None
            and StateDictOptions is not None
        )

    @staticmethod
    def _log_load_compatibility(tag, load_msg, model_keys, ckpt_keys, logger):
        missing = list(getattr(load_msg, "missing_keys", []) or [])
        unexpected = list(getattr(load_msg, "unexpected_keys", []) or [])
        model_total = len(model_keys)
        ckpt_total = len(ckpt_keys)
        matched = max(model_total - len(missing), 0)

        logger.info(
            f"[{tag}] model_keys={model_total} ckpt_keys={ckpt_total} "
            f"matched={matched} missing={len(missing)} unexpected={len(unexpected)}"
        )

        if len(missing) > 0:
            miss_counter = Counter(k.split(".", 1)[0] for k in missing)
            logger.warning(f"[{tag}] missing top prefixes: {miss_counter.most_common(8)}")
            for k in missing[:12]:
                logger.warning(f"[{tag}] missing sample: {k}")

        if len(unexpected) > 0:
            unexp_counter = Counter(k.split(".", 1)[0] for k in unexpected)
            logger.warning(f"[{tag}] unexpected top prefixes: {unexp_counter.most_common(8)}")
            for k in unexpected[:12]:
                logger.warning(f"[{tag}] unexpected sample: {k}")

        # Heuristic: LoRA-wrapped model expects `language_model.base_model.model.*`,
        # while official base ckpt may provide `language_model.model.*`.
        missing_peft_pref = sum(k.startswith("language_model.base_model.model.") for k in missing)
        unexpected_plain_pref = sum(k.startswith("language_model.model.") for k in unexpected)
        if missing_peft_pref > 0 and unexpected_plain_pref > 0:
            logger.error(
                f"[{tag}] key-space mismatch detected: missing language_model.base_model.model.*={missing_peft_pref}, "
                f"unexpected language_model.model.*={unexpected_plain_pref}. "
                "This usually means base checkpoint is being loaded into a LoRA-wrapped model with incompatible key prefixes."
            )

    @staticmethod
    def _parse_float_dtype(dtype_name):
        if dtype_name is None:
            return None
        n = str(dtype_name).strip().lower()
        if n in ("keep", "same", "same_as_input", ""):
            return None
        if n in ("bf16", "bfloat16"):
            return torch.bfloat16
        if n in ("fp16", "float16"):
            return torch.float16
        if n in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported checkpoint_weight_dtype: {dtype_name}")

    @staticmethod
    def _cast_state_dict_floats_inplace(state_dict, out_dtype):
        if out_dtype is None:
            return
        for k, v in list(state_dict.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != out_dtype:
                state_dict[k] = v.to(out_dtype)

    @staticmethod
    def _config_to_dict(cfg):
        if cfg is None:
            return None
        if hasattr(cfg, "to_dict"):
            try:
                return cfg.to_dict()
            except Exception:
                pass
        if is_dataclass(cfg):
            try:
                return asdict(cfg)
            except Exception:
                pass
        if isinstance(cfg, dict):
            return cfg
        try:
            return dict(cfg.__dict__)
        except Exception:
            return {"repr": repr(cfg)}

    @staticmethod
    def _jsonify(obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): FSDPCheckpoint._jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [FSDPCheckpoint._jsonify(v) for v in obj]
        if hasattr(obj, "to_dict"):
            try:
                return FSDPCheckpoint._jsonify(obj.to_dict())
            except Exception:
                pass
        if is_dataclass(obj):
            try:
                return FSDPCheckpoint._jsonify(asdict(obj))
            except Exception:
                pass
        if isinstance(obj, torch.dtype):
            return str(obj).replace("torch.", "")
        return str(obj)

    @staticmethod
    def _write_json_atomic(path: str, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                FSDPCheckpoint._jsonify(payload),
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp_path, path)

    @staticmethod
    def _link_or_copy(src_path: str, dst_path: str):
        if not src_path or (not os.path.exists(src_path)) or os.path.exists(dst_path):
            return
        try:
            os.link(src_path, dst_path)
            return
        except Exception:
            pass
        shutil.copy2(src_path, dst_path)

    @staticmethod
    def _export_inference_assets(
        save_path: str,
        logger,
        tokenizer=None,
        bagel_config=None,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        vae_source_path: str | None = None,
    ):
        try:
            if tokenizer is not None:
                tokenizer.save_pretrained(save_path)
        except Exception as e:
            logger.warning(f"[Checkpoint] Failed to save tokenizer at {save_path}: {e}")

        config_map = {
            "bagel_config.json": bagel_config,
            "config.json": bagel_config,
            "llm_config.json": llm_config,
            "vit_config.json": vit_config,
            "vae_config.json": vae_config,
        }
        for filename, cfg in config_map.items():
            if cfg is None:
                continue
            out_path = os.path.join(save_path, filename)
            try:
                FSDPCheckpoint._write_json_atomic(
                    out_path,
                    FSDPCheckpoint._config_to_dict(cfg),
                )
            except Exception as e:
                logger.warning(f"[Checkpoint] Failed to save {filename}: {e}")

        if vae_source_path is not None:
            try:
                FSDPCheckpoint._link_or_copy(
                    vae_source_path,
                    os.path.join(save_path, "ae.safetensors")
                )
            except Exception as e:
                logger.warning(f"[Checkpoint] Failed to place ae.safetensors from {vae_source_path}: {e}")

    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
        tokenizer=None,
        bagel_config=None,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        vae_source_path=None,
        checkpoint_weight_dtype="bf16",
        save_trainable_only=False,
        save_optimizer_state=True,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")
        out_dtype = FSDPCheckpoint._parse_float_dtype(checkpoint_weight_dtype)
        logger.info(f"[Checkpoint] checkpoint_weight_dtype={checkpoint_weight_dtype} parsed={out_dtype}")

        if save_trainable_only:
            if ema_model is not None:
                raise RuntimeError(
                    "save_trainable_only=True does not support EMA checkpoints. "
                    "Please disable EMA or use full checkpoint save."
                )
            if not FSDPCheckpoint._has_dcp_state_dict_api():
                raise RuntimeError(
                    "save_trainable_only=True requires torch.distributed.checkpoint.state_dict "
                    "(get_model_state_dict/set_model_state_dict/StateDictOptions), but it is unavailable."
                )

            logger.info("[Checkpoint] saving trainable-only model state (low-memory mode).")
            # PyTorch DCP + FSDP full_state_dict + cpu_offload currently returns a
            # rank0-only consolidated state dict in our setup, but the internal
            # `ignore_frozen_params=True` filtering still runs on non-zero ranks.
            # On those ranks the returned dict may not contain frozen FQNs such as
            # `language_model.base_model.model.model.embed_tokens.weight`, which
            # triggers KeyError during the post-filtering pop pass.
            #
            # Rank0 is the only rank that needs the filtered trainable-only state
            # for serialization, so keep the filtering on rank0 and let other
            # ranks participate in the same collective path without frozen-param
            # filtering. They discard the returned dict immediately afterwards.
            ignore_frozen_params = (dist.get_rank() == 0)
            model_state_dict = get_model_state_dict(
                model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                    ignore_frozen_params=ignore_frozen_params,
                ),
            )
            FSDPCheckpoint._cast_state_dict_floats_inplace(model_state_dict, out_dtype)
            if dist.get_rank() == 0:
                logger.info("[Checkpoint] writing trainable_model.safetensors.")
                save_file(model_state_dict, os.path.join(save_path, "trainable_model.safetensors"))
                logger.info("[Checkpoint] wrote trainable_model.safetensors.")
            del model_state_dict
            gc.collect()
            dist.barrier()
        else:
            if ema_model is not None:
                logger.info("[Checkpoint] collecting full EMA state dict on rank0.")
                with FSDP.state_dict_type(
                    ema_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    ema_state_dict = ema_model.state_dict()
                    FSDPCheckpoint._cast_state_dict_floats_inplace(ema_state_dict, out_dtype)
                    if dist.get_rank() == 0:
                        logger.info("[Checkpoint] writing ema.safetensors.")
                        save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))
                        logger.info("[Checkpoint] wrote ema.safetensors.")
                    del ema_state_dict
                    gc.collect()
            dist.barrier()

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                logger.info("[Checkpoint] collecting full model state dict on rank0.")
                model_state_dict = model.state_dict()
                FSDPCheckpoint._cast_state_dict_floats_inplace(model_state_dict, out_dtype)
                if dist.get_rank() == 0:
                    logger.info("[Checkpoint] writing model.safetensors.")
                    save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))
                    logger.info("[Checkpoint] wrote model.safetensors.")
                del model_state_dict
                gc.collect()
            # Keep non-zero ranks from writing optimizer shards when rank0 dies during model save.
            dist.barrier()

        if save_optimizer_state:
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                if fsdp_config.sharding_strategy == "FULL_SHARD":
                    shard_index = dist.get_rank()
                    total_shards = dist.get_world_size()
                elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                    shard_index = dist.get_rank() % fsdp_config.num_shard
                    total_shards = fsdp_config.num_shard
                else:
                    raise NotImplementedError

                optimizer_save_path = os.path.join(
                    save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
                )
                if fsdp_config.sharding_strategy == "FULL_SHARD":
                    logger.info(f"[Checkpoint] writing optimizer shard: {os.path.basename(optimizer_save_path)}")
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                    if dist.get_rank() < fsdp_config.num_shard:
                        logger.info(f"[Checkpoint] writing optimizer shard: {os.path.basename(optimizer_save_path)}")
                        torch.save(optimizer.state_dict(), optimizer_save_path)
                else:
                    raise NotImplementedError
            dist.barrier()

            if dist.get_rank() == 0 and scheduler is not None:
                logger.info("[Checkpoint] writing scheduler.pt.")
                torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        else:
            if dist.get_rank() == 0:
                logger.info("[Checkpoint] skipping optimizer/scheduler state save.")
            dist.barrier()

        if dist.get_rank() == 0 and data_status is not None:
            logger.info("[Checkpoint] writing data_status.pt.")
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        if dist.get_rank() == 0:
            FSDPCheckpoint._export_inference_assets(
                save_path=save_path,
                logger=logger,
                tokenizer=tokenizer,
                bagel_config=bagel_config,
                llm_config=llm_config,
                vit_config=vit_config,
                vae_config=vae_config,
                vae_source_path=vae_source_path,
            )
            logger.info("[Checkpoint] inference assets exported.")

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            trainable_model_state_dict_path = os.path.join(resume_from, "trainable_model.safetensors")
            if os.path.exists(trainable_model_state_dict_path):
                if not FSDPCheckpoint._has_dcp_state_dict_api():
                    raise RuntimeError(
                        "trainable_model.safetensors requires torch.distributed.checkpoint.state_dict "
                        "(set_model_state_dict), but it is unavailable."
                    )
                model_state_dict = load_file(trainable_model_state_dict_path, device="cpu")
                model_keys = list(model.state_dict().keys())
                ckpt_keys = list(model_state_dict.keys())
                msg = set_model_state_dict(
                    model,
                    model_state_dict,
                    options=StateDictOptions(strict=False),
                )
                logger.info(msg)
                if dist.get_rank() == 0:
                    FSDPCheckpoint._log_load_compatibility(
                        "CkptLoad/trainable_only",
                        msg,
                        model_keys=model_keys,
                        ckpt_keys=ckpt_keys,
                        logger=logger,
                    )
                del model_state_dict
                if ema_model is not None:
                    raise RuntimeError(
                        "trainable_model.safetensors checkpoint does not contain EMA weights. "
                        "Please resume with EMA disabled."
                    )
                return model, ema_model

            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed')
            model_state_dict.pop('vit_pos_embed.pos_embed')
            # LatentCOT: handle embedding size mismatch from special-token additions
            _model_sd = model.state_dict()
            for _k in list(model_state_dict.keys()):
                if _k in _model_sd and model_state_dict[_k].shape != _model_sd[_k].shape:
                    logger.info(
                        f'[CkptLoad] size mismatch, using model shape: '
                        f'{_k} {model_state_dict[_k].shape} -> {_model_sd[_k].shape}'
                    )
                    model_state_dict.pop(_k)
            model_keys = list(_model_sd.keys())
            ckpt_keys = list(model_state_dict.keys())
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            if dist.get_rank() == 0:
                FSDPCheckpoint._log_load_compatibility(
                    "CkptLoad/model",
                    msg,
                    model_keys=model_keys,
                    ckpt_keys=ckpt_keys,
                    logger=logger,
                )
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed')
                ema_state_dict.pop('vit_pos_embed.pos_embed')
                ema_model_keys = list(ema_model.state_dict().keys())
                ema_ckpt_keys = list(ema_state_dict.keys())
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                if dist.get_rank() == 0:
                    FSDPCheckpoint._log_load_compatibility(
                        "CkptLoad/ema",
                        msg,
                        model_keys=ema_model_keys,
                        ckpt_keys=ema_ckpt_keys,
                        logger=logger,
                    )
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config, logger=None):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1

            def _log(msg):
                if logger is not None:
                    logger.warning(msg)
                elif dist.get_rank() == 0:
                    print(msg, flush=True)

            optimizer_loaded = False
            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if os.path.exists(optimizer_state_dict_path):
                optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
                try:
                    optimizer.load_state_dict(optimizer_state_dict)
                    optimizer_loaded = True
                except ValueError as e:
                    if "different number of parameter groups" in str(e):
                        _log(
                            f"[Resume] Optimizer param-group mismatch at {optimizer_state_dict_path}; "
                            "skipping optimizer state load and continuing with fresh optimizer state."
                        )
                    else:
                        raise e
                del optimizer_state_dict
            else:
                _log(f"[Resume] Optimizer state not found: {optimizer_state_dict_path}")

            scheduler_loaded = False
            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            if os.path.exists(scheduler_state_dict_path):
                scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
                try:
                    scheduler.load_state_dict(scheduler_state_dict)
                    scheduler_loaded = True
                except Exception as e:
                    _log(f"[Resume] Failed to load scheduler state from {scheduler_state_dict_path}: {e}")
                del scheduler_state_dict
            else:
                _log(f"[Resume] Scheduler state not found: {scheduler_state_dict_path}")

            if (not scheduler_loaded) and train_steps > 0:
                # Keep LR phase approximately aligned even if scheduler state cannot be restored.
                try:
                    scheduler.step(train_steps)
                except Exception as e:
                    _log(f"[Resume] Failed to fast-forward scheduler to step {train_steps}: {e}")
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[], use_orig_params: bool = False):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules, use_orig_params=use_orig_params)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
