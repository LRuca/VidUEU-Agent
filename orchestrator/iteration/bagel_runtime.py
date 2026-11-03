"""Independent, full-precision BAGEL inference from an exported checkpoint."""
from __future__ import annotations

import json
import random
import re
import sys
from functools import wraps
from pathlib import Path
from typing import Any

from PIL import Image


def build_inferencer(vendor_root: Path, checkpoint: Path, *, image_size: int = 256,
                     device: str = "cuda:0"):
    """Read architecture from config/weight keys; never reuse legacy size guesses."""
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from safetensors import safe_open

    vendor_root, checkpoint = vendor_root.resolve(), checkpoint.resolve()
    if not vendor_root.is_dir() or not checkpoint.is_dir():
        raise FileNotFoundError("bagel_runtime_input_missing")
    if not torch.cuda.is_available():
        raise RuntimeError("bagel_runtime_requires_cuda")
    sys.path.insert(0, str(vendor_root))
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    # Reject accidentally imported modules from the old checkout.
    for name in ("data", "modeling", "inferencer"):
        module = sys.modules[name]
        if not Path(module.__file__).resolve().is_relative_to(vendor_root):
            raise RuntimeError(f"bagel_runtime_wrong_import:{name}")
    architecture = json.loads((checkpoint / "config.json").read_text())
    llm_config = Qwen2Config.from_json_file(str(checkpoint / "llm_config.json"))
    if llm_config.pad_token_id is None:
        llm_config.pad_token_id = llm_config.eos_token_id
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(str(checkpoint / "vit_config.json"))
    vit_config.rope = bool(architecture.get("vit_config", {}).get("rope", False))
    weight_path = checkpoint / "ema.safetensors"
    with safe_open(str(weight_path), framework="pt", device="cpu") as weights:
        weight_keys = set(weights.keys())
        latent_pos_shape = (
            weights.get_slice("latent_pos_embed.pos_embed").get_shape()
            if "latent_pos_embed.pos_embed" in weight_keys else None
        )
    vit_layers = {int(match.group(1)) for key in weight_keys
                  if (match := re.match(r"vit_model\.vision_model\.encoder\.layers\.(\d+)\.", key))}
    if not vit_layers or vit_layers != set(range(max(vit_layers) + 1)):
        raise ValueError("invalid_bagel_vision_layer_keys")
    vit_config.num_hidden_layers = len(vit_layers)
    vae, vae_config = load_ae(local_path=str(checkpoint / "ae.safetensors"))
    # Some released BAGEL checkpoints contain a stale config value here.  The
    # positional embedding tensor is the authoritative architecture contract:
    # infer the square side from its first dimension while retaining the
    # config value only as a fallback for older layouts.
    configured_latent_size = int(architecture.get("max_latent_size", 32))
    if latent_pos_shape and len(latent_pos_shape) == 2:
        latent_tokens = int(latent_pos_shape[0])
        inferred_latent_size = int(latent_tokens ** 0.5)
        if inferred_latent_size * inferred_latent_size != latent_tokens:
            raise ValueError("invalid_bagel_latent_position_shape")
        max_latent_size = inferred_latent_size
    else:
        max_latent_size = configured_latent_size
    bagel_config = BagelConfig(
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        visual_gen=True, visual_und=True,
        max_latent_size=max_latent_size,
        latent_patch_size=int(architecture.get("latent_patch_size", 2)),
        vit_max_num_patch_per_side=int(architecture.get("vit_max_num_patch_per_side", 70)),
        connector_act=architecture.get("connector_act", "gelu_pytorch_tanh"),
        interpolate_pos=bool(architecture.get("interpolate_pos", False)),
        timestep_shift=float(architecture.get("timestep_shift", 1.0)),
    )
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    expected_keys = set(model.state_dict())
    if expected_keys != weight_keys:
        raise ValueError("bagel_runtime_weight_keys_mismatch:"
                         f"missing={sorted(expected_keys - weight_keys)[:6]}:"
                         f"extra={sorted(weight_keys - expected_keys)[:6]}")
    # This smoke uses one otherwise-free 40GB card. Full residency avoids
    # accelerate hooks leaving custom RoPE position tensors on the CPU.
    model = load_checkpoint_and_dispatch(model, checkpoint=str(weight_path), device_map={"": device},
                                         dtype=torch.bfloat16, force_hooks=False).eval()
    model = model.to(device)
    # BAGEL's packed inference helpers construct index tensors on CPU. The
    # trainer's distributed path moves them through its batch wrapper; the
    # standalone runtime must make the same contract explicit.
    target_device = torch.device(device)

    # ``load_checkpoint_and_dispatch`` may leave BAGEL's non-transformer
    # generation heads on their construction device when loading a merged
    # checkpoint.  They are small enough to pin explicitly and are required
    # by the image-generation flow (the text path alone does not exercise
    # them).
    for _module_name in (
        "time_embedder", "vae2llm", "llm2vae", "latent_pos_embed",
        "connector", "vit_pos_embed",
    ):
        _module = getattr(model, _module_name, None)
        if _module is not None:
            _module.to(target_device)

    def _move_tensors(value):
        """Move packed inference inputs made by BAGEL's CPU helpers.

        The native training dataloader moves these tensors as part of its
        batch wrapper.  ``InterleaveInferencer`` builds the same packed
        structures directly on CPU, so an independent reload must enforce
        the model-device contract at the module boundary.
        """
        if isinstance(value, torch.Tensor):
            return value.to(target_device)
        if isinstance(value, list):
            return [_move_tensors(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_move_tensors(item) for item in value)
        if isinstance(value, dict):
            return {key: _move_tensors(item) for key, item in value.items()}
        return value

    original_embed = model._embed_tokens
    def _embed_tokens_on_device(input_ids):
        return original_embed(input_ids.to(target_device))
    model._embed_tokens = _embed_tokens_on_device

    # ``prepare_vit_images`` creates packed pixel values, position ids and
    # sequence metadata on CPU.  SigLIP's patch projection is on the model's
    # device, so move the complete call boundary together.
    original_vit_forward = model.vit_model.forward
    @wraps(original_vit_forward)
    def _vit_forward_on_device(*args, **kwargs):
        return original_vit_forward(
            *tuple(_move_tensors(arg) for arg in args),
            **_move_tensors(kwargs),
        )
    model.vit_model.forward = _vit_forward_on_device

    # The high-level cache-update methods receive additional packed tensors
    # (timesteps, positions, indexes) that are not passed through the VIT or
    # VAE module wrappers.  Normalize those arguments before BAGEL performs
    # embeddings, indexing, or writes into the CUDA sequence buffer.
    def _wrap_model_tensor_boundary(method_name):
        original = getattr(model, method_name)
        @wraps(original)
        def wrapped(*args, **kwargs):
            return original(
                *tuple(_move_tensors(arg) for arg in args),
                **_move_tensors(kwargs),
            )
        setattr(model, method_name, wrapped)

    for _method_name in (
        "forward_cache_update_vit",
        "forward_cache_update_vae",
        "forward_cache_update_vae_latents",
        "generate_image",
    ):
        _wrap_model_tensor_boundary(_method_name)

    language_model = model._get_base_language_model()
    original_forward_inference = language_model.forward_inference
    def _forward_inference_on_device(*args, **kwargs):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(target_device)
            if isinstance(value, (list, tuple)):
                return type(value)(move(item) for item in value)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value
        return original_forward_inference(*(move(arg) for arg in args), **move(kwargs))
    language_model.forward_inference = _forward_inference_on_device
    vae = vae.to(device=device, dtype=torch.float32).eval()
    # The packed VAE image helper also materializes its batch on CPU.  Keep
    # both directions explicit so understanding (VAE encode) and generation
    # (VAE decode) share the same independent-runtime contract.
    original_vae_encode = vae.encode
    @wraps(original_vae_encode)
    def _vae_encode_on_device(*args, **kwargs):
        return original_vae_encode(
            *tuple(_move_tensors(arg) for arg in args),
            **_move_tensors(kwargs),
        )
    vae.encode = _vae_encode_on_device
    original_vae_decode = vae.decode
    @wraps(original_vae_decode)
    def _vae_decode_on_device(*args, **kwargs):
        return original_vae_decode(
            *tuple(_move_tensors(arg) for arg in args),
            **_move_tensors(kwargs),
        )
    vae.decode = _vae_decode_on_device
    tokenizer = Qwen2Tokenizer.from_pretrained(str(checkpoint))
    tokenizer, special_ids, _ = add_special_tokens(tokenizer)
    # Match the native trainer: encode raw fragments without implicit BOS/EOS.
    tokenizer.encode = lambda text, *args, **kwargs: tokenizer(
        text, add_special_tokens=False, return_attention_mask=False,
        return_token_type_ids=False)["input_ids"]
    inferencer = InterleaveInferencer(
        model=model, vae_model=vae, tokenizer=tokenizer,
        vae_transform=ImageTransform(image_size, image_size, 16, max_pixels=image_size ** 2),
        vit_transform=ImageTransform(224, 224, 14, max_pixels=224 ** 2), new_token_ids=special_ids,
    )
    inferencer.runtime_identity = {
        "checkpoint": str(checkpoint), "vendor_root": str(vendor_root), "device": device,
        "image_size": image_size, "max_latent_size": bagel_config.max_latent_size,
        "vit_layers": len(vit_layers), "weight_key_count": len(weight_keys),
        "dtype": "bfloat16", "quantized": False,
    }
    return inferencer


def seed_inference(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dialogue_inputs(messages: list[dict[str, Any]]) -> list[Any]:
    """Replay complete history once, including generated images at their turn."""
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("inference_dialogue_must_end_with_user")
    terms: list[Any] = []
    for message in messages:
        if message["role"] not in {"user", "assistant"}:
            raise ValueError("invalid_inference_role")
        terms.append(f'<|im_start|>{message["role"]}\n')
        for image in message.get("images", []):
            with Image.open(image) as source:
                terms.append(source.convert("RGB"))
        terms.append(str(message.get("text", "")) + "<|im_end|>\n")
    terms.append("<|im_start|>assistant\n")
    return terms


def infer_dialogue(inferencer, messages: list[dict[str, Any]], *, output_type: str,
                   decoding: dict[str, Any], seed: int) -> str | Image.Image:
    if output_type not in {"text", "image"}:
        raise ValueError("invalid_inference_output_type")
    seed_inference(seed)
    result = inferencer.interleave_inference(
        input_lists=dialogue_inputs(messages), understanding_output=output_type == "text",
        max_think_token_n=int(decoding.get("max_new_tokens", 256)),
        do_sample=bool(decoding.get("do_sample", False)),
        text_temperature=float(decoding.get("temperature", 0.3)),
        num_timesteps=int(decoding.get("num_timesteps", 8)),
        image_shapes=(int(decoding.get("image_size", 256)),) * 2,
        cfg_text_scale=float(decoding.get("cfg_text_scale", 4)),
        cfg_img_scale=float(decoding.get("cfg_img_scale", 2)),
        cfg_interval=[0.0, 1.0], timestep_shift=float(decoding.get("timestep_shift", 3)),
        cfg_renorm_type="text_channel",
    )
    values = [item for item in result if isinstance(item, str if output_type == "text" else Image.Image)]
    if len(values) != 1 or (output_type == "text" and not values[0].strip()):
        raise ValueError("bagel_missing_or_ambiguous_output")
    return values[0].strip() if output_type == "text" else values[0]
