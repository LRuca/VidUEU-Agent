#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Must be set before importing torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

from pathlib import Path
import random
import sys
import argparse
from typing import Optional, Tuple

import torch
from PIL import Image

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# Make local project importable
PROJECT_ROOT = "/home/zecongt/Yichen_Zhu/BAGEL"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from BAGEL.data.data_utils import add_special_tokens
from BAGEL.data.transforms import ImageTransform
from BAGEL.inferencer import InterleaveInferencer
from BAGEL.modeling.autoencoder import load_ae
from BAGEL.modeling.bagel import (
    BagelConfig, Bagel,
    Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from BAGEL.modeling.qwen2 import Qwen2Tokenizer


MODEL_PATH = "/home/zecongt/Yichen_Zhu/BAGEL/models/BAGEL-7B-MoT"
OUTPUT_DIR = "/home/zecongt/Yichen_Zhu/output"
OFFLOAD_FOLDER = str(Path(OUTPUT_DIR) / "offload")

MAX_GPU_MEM = "80GiB"
DTYPE_PREFERRED = torch.bfloat16
SEED = 12345

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


def pick_dtype(preferred: torch.dtype) -> torch.dtype:
    """
    Select compute dtype for loading/inference.

    Input:
      preferred: expected dtype (e.g., torch.bfloat16)

    Output:
      dtype to use on current device (bf16 if supported on CUDA; else fp16; CPU -> fp32)
    """
    if not torch.cuda.is_available():
        return torch.float32
    if preferred == torch.bfloat16:
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return preferred


def build_chat_prompt(user_text: str, has_image: bool) -> str:
    """
    Build a single-turn chat prompt in Qwen-style message format.

    Inputs:
      user_text: raw user prompt
      has_image: whether the prompt should include an image placeholder

    Output:
      serialized prompt string that ends with assistant start token
    """
    if has_image:
        user_block = (
            f"<|im_start|>{USER_ROLE}\n"
            f"<img><|image_1|></img>\n"
            f"{user_text}\n"
            f"<|im_end|>\n"
        )
    else:
        user_block = (
            f"<|im_start|>{USER_ROLE}\n"
            f"{user_text}\n"
            f"<|im_end|>\n"
        )
    return user_block + f"<|im_start|>{ASSISTANT_ROLE}\n"


def extract_cot(text: str) -> Tuple[Optional[str], str]:
    """
    Extract Chain-of-Thought content from a model text output.

    Input:
      text: model output text that may contain <think>...</think>

    Output:
      (cot, rest)
        - cot: content inside <think>...</think> if present, else None
        - rest: original text with the <think>...</think> segment removed if present;
                if tags not found, rest equals original text
    """
    start_tag = "<think>"
    end_tag = "</think>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start == -1 or end == -1 or end < start:
        return None, text
    cot = text[start + len(start_tag): end].strip()
    rest = (text[:start] + text[end + len(end_tag):]).strip()
    return cot, rest


def load_pil_rgb(path: str) -> Image.Image:
    """
    Load an image from disk as a PIL.Image.

    Input:
      path: filesystem path

    Output:
      PIL.Image (kept as-is; downstream transform handles resizing/color conversion)
    """
    return Image.open(path)


def build_bagel_and_inferencer(model_path: str) -> InterleaveInferencer:
    """
    Construct BAGEL model, tokenizer, VAE, and InterleaveInferencer.

    Data flow:
      - load configs (llm/vit/vae)
      - create empty-weight skeleton (init_empty_weights)
      - compute device_map and load weights via accelerate dispatch
      - build tokenizer and special tokens
      - return InterleaveInferencer that calls model.prepare_* and model.generate_*
    """
    # -------- 1) LLM config --------
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))

    if getattr(llm_config, "pad_token_id", None) is None:
        llm_config.pad_token_id = llm_config.eos_token_id

    from BAGEL.modeling.qwen2.modeling_qwen2 import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["linear"]

    if getattr(llm_config, "rope_scaling", None) is None:
        llm_config.rope_scaling = {"type": "linear", "factor": 1.0}
    else:
        llm_config.rope_scaling.setdefault("type", "linear")
        llm_config.rope_scaling.setdefault("factor", 1.0)

    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # -------- 2) ViT config --------
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    # -------- 3) VAE --------
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # -------- 4) BagelConfig --------
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    # -------- 5) build skeleton (empty weights) --------
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)

    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # -------- 6) tokenizer + special tokens --------
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    _orig_encode = tokenizer.encode

    def _patched_encode(text, *args, **kwargs):
        enc = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc.get("input_ids", None)
        if input_ids is None:
            return _orig_encode(text, *args, **kwargs)
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
            return input_ids[0]
        return input_ids

    tokenizer.encode = _patched_encode

    # -------- 7) transforms --------
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # -------- 8) device_map + load weights --------
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("No CUDA devices found.")

    device_map = infer_auto_device_map(
        model,
        max_memory={i: MAX_GPU_MEM for i in range(gpu_count)},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    dtype = pick_dtype(DTYPE_PREFERRED)
    Path(OFFLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder=OFFLOAD_FOLDER,
        dtype=dtype,
        force_hooks=True,
    ).eval()

    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )


def set_seed(seed: int) -> None:
    """
    Set Python and Torch random seeds.

    Input:
      seed: integer seed

    Output:
      None (global RNG state updated)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_input_lists(prompt_text: str, input_image: Optional[Image.Image]) -> list:
    """
    Construct the interleave input stream.

    Inputs:
      prompt_text: user instruction text
      input_image: optional PIL image

    Output:
      List[Union[PIL.Image, str]] to feed into inferencer.interleave_inference
      - if image is provided: [image, prompt_string_with_<img><|image_1|></img>]
      - else: [prompt_string]
    """
    has_image = input_image is not None
    prompt = build_chat_prompt(prompt_text, has_image=has_image)
    if has_image:
        return [input_image, prompt]
    return [prompt]


def run_inference(
    inferencer: InterleaveInferencer,
    prompt: str,
    input_image_path: Optional[str],
    mode: str,
    enable_cot: bool,
    out_image_path: str,
    out_text_path: str,
) -> None:
    """
    Run inference with optional image conditioning and selectable output modality.

    Inputs:
      inferencer: initialized InterleaveInferencer
      prompt: user instruction text
      input_image_path: optional path to an input image
      mode: "image" or "text"
      enable_cot: enable think mode and print CoT when returned
      out_image_path: path to save image when mode="image"
      out_text_path: path to save text when mode="text"

    Output:
      Saves output (image or text) to disk; prints CoT segments to stdout when available.
    """
    input_image = load_pil_rgb(input_image_path) if input_image_path else None
    input_lists = build_input_lists(prompt, input_image)

    if mode == "image":
        inferencer.model.config.visual_gen = True
        inferencer.model.config.visual_und = False
        output = inferencer.interleave_inference(
            input_lists=input_lists,
            think=enable_cot,
            understanding_output=False,
            image_shapes=(1024, 1024),
            do_sample=True,
            text_temperature=0.7,
            num_timesteps=50,
            cfg_text_scale=3.0,
            cfg_img_scale=1.5,
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            max_think_token_n=512,
        )

        if enable_cot and isinstance(output, list) and output and isinstance(output[0], str):
            cot, rest = extract_cot(output[0])
            print("\n=== CoT (image) ===")
            print(cot if cot is not None else output[0])
            if rest and cot is not None:
                print("\n=== Non-CoT text (image) ===")
                print(rest)

        if not (isinstance(output, list) and output and isinstance(output[-1], Image.Image)):
            raise RuntimeError(f"Unexpected image output: {type(output)} -> {output}")

        img: Image.Image = output[-1]
        Path(Path(out_image_path).parent).mkdir(parents=True, exist_ok=True)
        img.save(out_image_path)
        print(f"[✓] Saved image: {out_image_path}")
        return

    if mode == "text":
        inferencer.model.config.visual_gen = False
        inferencer.model.config.visual_und = True
        output = inferencer.interleave_inference(
            input_lists=input_lists,
            think=enable_cot,
            understanding_output=True,
            do_sample=True,
            text_temperature=0.7,
            max_think_token_n=512,
        )

        if not (isinstance(output, list) and output and isinstance(output[0], str)):
            raise RuntimeError(f"Unexpected text output: {type(output)} -> {output}")

        full_text: str = output[0]
        if enable_cot:
            cot, rest = extract_cot(full_text)
            print("\n=== CoT (text) ===")
            print(cot if cot is not None else full_text)
            if rest and cot is not None:
                print("\n=== Non-CoT text (text) ===")
                print(rest)

        Path(Path(out_text_path).parent).mkdir(parents=True, exist_ok=True)
        with open(out_text_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"[✓] Saved text: {out_text_path}")
        print("\n--- Model Output (text) ---\n")
        print(full_text)
        return

    raise ValueError(f"Unsupported mode: {mode}. Use 'image' or 'text'.")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.

    Output:
      Namespace with:
        cot: bool
        seed: int
        mode: str ("image" or "text")
        input_image: Optional[str]
        prompt: str
        out_image: str
        out_text: str
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", action="store_true", help="Enable CoT generation and print <think>...</think> to stdout.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--mode", choices=["image", "text"], default="image", help="Output modality.")
    parser.add_argument("--input_image", type=str, default=None, help="Optional input image path for multimodal inference.")
    parser.add_argument("--prompt", type=str, default="A clean vector-style diagram, white background, conference-paper style.",
                        help="User instruction text.")
    parser.add_argument("--out_image", type=str, default=str(Path(OUTPUT_DIR) / "bagel_image.png"),
                        help="Output image path (mode=image).")
    parser.add_argument("--out_text", type=str, default=str(Path(OUTPUT_DIR) / "bagel_text.txt"),
                        help="Output text path (mode=text).")
    return parser.parse_args()


def main():
    """
    Entry point.

    Data flow:
      args -> seed -> build inferencer -> run_inference
    """
    args = parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    print("MODEL_PATH =", MODEL_PATH)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    print("MODE =", args.mode)
    print("COT =", bool(args.cot))
    print("SEED =", int(args.seed))
    print("INPUT_IMAGE =", args.input_image if args.input_image else "(none)")

    inferencer = build_bagel_and_inferencer(MODEL_PATH)

    run_inference(
        inferencer=inferencer,
        prompt=args.prompt,
        input_image_path=args.input_image,
        mode=args.mode,
        enable_cot=bool(args.cot),
        out_image_path=args.out_image,
        out_text_path=args.out_text,
    )


if __name__ == "__main__":
    main()
