from __future__ import annotations

import gc
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    from transformers import (
        AutoModel,
        AutoProcessor,
        AutoTokenizer,
        Qwen3VLForConditionalGeneration,
    )
except Exception:
    AutoModel = None
    AutoProcessor = None
    AutoTokenizer = None
    Qwen3VLForConditionalGeneration = None

try:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except Exception:
    T = None
    InterpolationMode = None

try:
    from qwen_vlm_api import QwenVLMApi
except Exception:
    QwenVLMApi = None

try:
    from executor.models.vllm_vision_api import VLLMVisionApi
except Exception:
    VLLMVisionApi = None


ImageLike = Union[str, Image.Image]


class QwenVLM:
    def __init__(
        self,
        model_config: Union[str, Dict[str, Any]],
        device: Optional[Union[str, int]] = None,
    ):
        self.mode = "local"
        self.local_backend = "qwen3_vl"
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.api_client = None
        # Propagated from the API adapter after every successful request so
        # metadata/domain audits can persist the exact requested/response
        # model identity alongside their prompt hash.
        self.last_attestation: Dict[str, Any] = {}
        self.api_backend = ""
        self.device = torch.device("cpu")
        self.assigned_device: Optional[str] = None
        self._chat_state = threading.local()

        self.mem_debug = bool(int(os.getenv("QWEN_VLM_MEM_DEBUG", "0") or "0"))
        self.empty_cache_on_return = bool(int(os.getenv("QWEN_VLM_EMPTY_CACHE", "0") or "0"))
        self.gc_on_return = bool(int(os.getenv("QWEN_VLM_GC", "1") or "1"))

        cfg: Dict[str, Any]
        if isinstance(model_config, str):
            cfg = {"mode": "local", "path": model_config}
        elif isinstance(model_config, dict):
            cfg = dict(model_config)
        else:
            raise TypeError(f"QwenVLM expects str or dict, got {type(model_config)}")

        if device is not None:
            self.assigned_device = self._normalize_device_arg(device)
            cfg["device"] = self.assigned_device

        self.mode = str(cfg.get("mode") or "local").strip().lower()
        if self.mode == "api":
            self._init_api(cfg.get("api") or cfg)
        else:
            self._init_local(cfg)

    def _normalize_device_arg(self, device: Union[str, int]) -> str:
        if isinstance(device, int):
            return f"cuda:{device}"
        s = str(device).strip()
        if s.isdigit():
            return f"cuda:{s}"
        return s or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def _init_api(self, api_cfg: Dict[str, Any]) -> None:
        provider = str(api_cfg.get("provider") or api_cfg.get("backend") or "").strip().lower()
        if provider in ("vllm", "vllm_openai", "openai_compatible_vllm"):
            if VLLMVisionApi is None:
                raise RuntimeError("QwenVLM vLLM api mode requires executor.models.vllm_vision_api")
            self.api_backend = "vllm_openai"
            self.api_client = VLLMVisionApi(api_cfg)
            print(
                f"[QwenVLM] using vLLM api mode: model={getattr(self.api_client, 'model', 'unknown')} "
                f"base_url={getattr(self.api_client, 'base_url', '')}"
            )
            return

        if QwenVLMApi is None:
            raise RuntimeError("QwenVLM api mode requires qwen_vlm_api.py to be importable")
        api_key = str(api_cfg.get("api_key") or "").strip()
        api_key_env = str(api_cfg.get("api_key_env") or "OPENROUTER_API_KEY").strip()
        if api_key:
            os.environ[api_key_env] = api_key
        self.api_backend = "legacy_api"
        self.api_client = QwenVLMApi({"api": api_cfg})
        print(f"[QwenVLM] using api mode: {getattr(self.api_client, 'model', 'unknown')}")

    def _init_local(self, cfg: Dict[str, Any]) -> None:
        model_path = str(cfg.get("path") or cfg.get("model_path") or cfg.get("local_model_dir") or "").strip()
        if not model_path:
            raise ValueError("QwenVLM local mode requires path/model_path/local_model_dir")

        device_pref = str(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")).strip()
        self.device = torch.device(device_pref if torch.cuda.is_available() else "cpu")
        self.assigned_device = str(self.device)

        backend = str(cfg.get("backend") or cfg.get("model_family") or "auto").strip().lower()
        if backend in ("", "auto"):
            backend = "internvl" if "internvl" in model_path.lower() else "qwen3_vl"
        self.local_backend = backend

        if self.local_backend in ("internvl", "internvl3", "internvl3.5", "internvl_3_5"):
            self._init_local_internvl(model_path, cfg)
            return

        self._init_local_qwen3_vl(model_path, cfg)

    def _init_local_qwen3_vl(self, model_path: str, cfg: Dict[str, Any]) -> None:
        if Qwen3VLForConditionalGeneration is None or AutoProcessor is None:
            raise RuntimeError("QwenVLM local Qwen3-VL mode requires transformers with Qwen3-VL support")

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "attn_implementation": str(cfg.get("attn_implementation") or "sdpa"),
        }
        if self.device.type == "cuda":
            if self.assigned_device:
                model_kwargs["device_map"] = {"": str(self.device)}
            else:
                model_kwargs["device_map"] = cfg.get("device_map") or "auto"

        print(f"[QwenVLM] loading local Qwen3-VL model from {model_path} on {self.device}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path)
        if self.device.type != "cuda":
            self.model.to(self.device)

    def _init_local_internvl(self, model_path: str, cfg: Dict[str, Any]) -> None:
        if AutoModel is None or AutoTokenizer is None or T is None or InterpolationMode is None:
            raise RuntimeError("QwenVLM local InternVL mode requires transformers and torchvision")

        use_fast = bool(cfg.get("tokenizer_use_fast", False))
        self.internvl_input_size = int(cfg.get("internvl_input_size") or 448)
        self.internvl_max_num_patches = int(cfg.get("internvl_max_num_patches") or 12)
        self.internvl_use_thumbnail = bool(cfg.get("internvl_use_thumbnail", True))

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=use_fast,
                fix_mistral_regex=True,
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=use_fast,
            )

        torch_dtype_name = str(
            cfg.get("torch_dtype") or ("bfloat16" if self.device.type == "cuda" else "float32")
        ).lower()
        torch_dtype = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }.get(torch_dtype_name, torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.model_dtype = torch_dtype

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            # 官方 quick start 是 True；这里改成 True 更稳
            "low_cpu_mem_usage": bool(cfg.get("low_cpu_mem_usage", True)),
        }
        if "use_flash_attn" in cfg:
            model_kwargs["use_flash_attn"] = bool(cfg.get("use_flash_attn"))
        if self.device.type == "cuda":
            if self.assigned_device:
                model_kwargs["device_map"] = {"": str(self.device)}
            elif cfg.get("device_map") is not None:
                model_kwargs["device_map"] = cfg.get("device_map")

        print(f"[QwenVLM] loading local InternVL model from {model_path} on {self.device}")
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
        if self.device.type != "cuda" or "device_map" not in model_kwargs:
            self.model.to(self.device)

    def _build_internvl_transform(self, input_size: int) -> Any:
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: List[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess_internvl(
        self,
        image: Image.Image,
        *,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = True,
    ) -> List[Image.Image]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = sorted(
            {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if min_num <= i * j <= max_num
            },
            key=lambda x: x[0] * x[1],
        )
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))

        processed_images: List[Image.Image] = []
        tiles_per_row = target_width // image_size
        for i in range(blocks):
            box = (
                (i % tiles_per_row) * image_size,
                (i // tiles_per_row) * image_size,
                ((i % tiles_per_row) + 1) * image_size,
                ((i // tiles_per_row) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def _prepare_internvl_inputs(
        self,
        images: List[ImageLike],
        max_side: Optional[int],
    ) -> tuple[torch.Tensor, List[int]]:
        transform = self._build_internvl_transform(self.internvl_input_size)
        pixel_tensors: List[torch.Tensor] = []
        num_patches_list: List[int] = []

        for img in images:
            pil = self._resize_image_for_attempt(img, max_side, self._resize_pil)
            processed = self._dynamic_preprocess_internvl(
                pil,
                image_size=self.internvl_input_size,
                max_num=self.internvl_max_num_patches,
                use_thumbnail=self.internvl_use_thumbnail,
            )
            patches = torch.stack([transform(x) for x in processed])
            pixel_tensors.append(patches)
            num_patches_list.append(int(patches.shape[0]))

        if not pixel_tensors:
            raise ValueError("InternVL chat requires at least one image")

        pixel_values = torch.cat(pixel_tensors, dim=0)
        return pixel_values, num_patches_list

    def _strip_internvl_placeholders(self, prompt: str) -> str:
        text = str(prompt or "")
        # 去掉已经存在的编号占位，避免重复
        text = re.sub(r"(?im)^\s*(Image-\d+|Frame\d+)\s*:\s*<image>\s*\n?", "", text)
        # 去掉零散的 <image>
        text = re.sub(r"\s*<image>\s*", " ", text)
        # 清理多余空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _build_internvl_question(self, prompt: str, image_count: int) -> str:
        prompt = str(prompt or "")
        if image_count <= 0:
            return prompt

        placeholder_count = prompt.count("<image>")

        # 已经完全匹配，就直接用
        if placeholder_count == image_count:
            return prompt

        # 没有占位，则自动补官方格式
        if placeholder_count == 0:
            if image_count == 1:
                return f"<image>\n{prompt}"
            prefix = "".join([f"Image-{i+1}: <image>\n" for i in range(image_count)])
            return prefix + prompt

        # 有占位但数量不匹配：统一重建，避免 token / patch 对不上
        cleaned = self._strip_internvl_placeholders(prompt)
        if image_count == 1:
            return f"<image>\n{cleaned}"
        prefix = "".join([f"Image-{i+1}: <image>\n" for i in range(image_count)])
        return prefix + cleaned

    def _load_image(self, img: ImageLike) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(f"[QwenVLM] image not found: {img}")
            return Image.open(img).convert("RGB")
        raise TypeError(f"[QwenVLM] unsupported image type: {type(img)}")

    def _local_model_ready(self) -> bool:
        if self.model is None:
            return False
        if self.local_backend.startswith("internvl"):
            return self.tokenizer is not None
        return self.processor is not None

    def get_image_embedding(self, images: List[ImageLike]) -> np.ndarray:
        if self.mode == "api":
            raise RuntimeError("QwenVLM api mode does not support get_image_embedding")
        if self.local_backend.startswith("internvl"):
            raise RuntimeError("QwenVLM local InternVL mode does not support get_image_embedding")
        if not self._local_model_ready():
            raise RuntimeError("QwenVLM local model is not initialized")

        pil_images = [self._load_image(img) for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        pixel_values = inputs["pixel_values"].to(model_device, dtype=self.model.dtype)

        with torch.no_grad():
            vision_tower = self.model.vision_tower[0]
            vision_outputs = vision_tower(pixel_values)
            if isinstance(vision_outputs, tuple):
                feat = vision_outputs[0]
            elif hasattr(vision_outputs, "last_hidden_state"):
                feat = vision_outputs.last_hidden_state
            else:
                feat = list(vision_outputs.values())[0]
            emb = feat.mean(dim=1)
        return emb.detach().cpu().numpy()

    def chat(
        self,
        images: List[ImageLike],
        prompt: str,
        max_new_tokens: int = 2048,
    ) -> str:
        request = {
            "prompt": str(prompt or ""),
            "max_new_tokens": int(max_new_tokens),
            "image_count": int(len(images or [])),
            "image_refs": self._summarize_images(images),
        }
        try:
            out = self._chat_impl(images, prompt, max_new_tokens)
            self._emit_chat_record(request=request, response=out)
            return out
        except Exception as e:
            self._emit_chat_record(request=request, error=e)
            raise

    def _chat_impl(
        self,
        images: List[ImageLike],
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        if self.mode == "api":
            if self.api_client is None:
                raise RuntimeError("QwenVLM api client is not initialized")
            if self.api_backend == "vllm_openai":
                out = str(self.api_client.chat(images, prompt, max_new_tokens=max_new_tokens)).strip()
                self.last_attestation = dict(getattr(self.api_client, "last_attestation", {}) or {})
                if not self.last_attestation.get("requested_model"):
                    self.last_attestation["requested_model"] = getattr(self.api_client, "model", None)
                return out
            pil_images = [self._load_image(img) for img in images]
            out = str(self.api_client.chat(pil_images, prompt, max_new_tokens=max_new_tokens)).strip()
            self.last_attestation = dict(getattr(self.api_client, "last_attestation", {}) or {})
            if not self.last_attestation.get("requested_model"):
                self.last_attestation["requested_model"] = getattr(self.api_client, "model", None)
            return out

        if not self._local_model_ready():
            raise RuntimeError("QwenVLM local model is not initialized")

        def _get_bool_attr(name: str, env: str, default: bool) -> bool:
            if hasattr(self, name):
                return bool(getattr(self, name))
            v = str(os.getenv(env, "")).lower()
            if v == "":
                return default
            return v not in ("0", "false", "no", "off")

        def _get_float_attr(name: str, env: str, default: float) -> float:
            if hasattr(self, name):
                try:
                    return float(getattr(self, name))
                except Exception:
                    return default
            try:
                return float(os.getenv(env, str(default)))
            except Exception:
                return default

        def _get_int_attr(name: str, env: str, default: int) -> int:
            if hasattr(self, name):
                try:
                    return int(getattr(self, name))
                except Exception:
                    return default
            try:
                return int(os.getenv(env, str(default)))
            except Exception:
                return default

        mem_debug = _get_bool_attr("mem_debug", "QWEN_VLM_MEM_DEBUG", False)
        empty_cache_on_return = _get_bool_attr("empty_cache_on_return", "QWEN_VLM_EMPTY_CACHE", False)
        gc_on_return = _get_bool_attr("gc_on_return", "QWEN_VLM_GC", True)
        oom_retry_enable = _get_bool_attr("oom_retry_enable", "QWEN_VLM_OOM_RETRY", True)
        oom_retry_sleep_s = _get_float_attr("oom_retry_sleep_s", "QWEN_VLM_OOM_RETRY_SLEEP", 0.2)

        max_new_tokens_cap = None
        if hasattr(self, "max_new_tokens_cap"):
            try:
                cap = int(getattr(self, "max_new_tokens_cap") or 0)
                max_new_tokens_cap = cap if cap > 0 else None
            except Exception:
                max_new_tokens_cap = None
        else:
            cap = _get_int_attr("unused", "QWEN_VLM_MAX_NEW_TOKENS_CAP", 0)
            max_new_tokens_cap = cap if cap > 0 else None

        oom_retry_max_sides = getattr(self, "oom_retry_max_sides", None)
        if not isinstance(oom_retry_max_sides, (list, tuple)) or len(oom_retry_max_sides) == 0:
            oom_retry_max_sides = [1024, 768, 512]

        oom_retry_token_scales = getattr(self, "oom_retry_token_scales", None)
        if not isinstance(oom_retry_token_scales, (list, tuple)) or len(oom_retry_token_scales) == 0:
            oom_retry_token_scales = [1.0, 0.75, 0.5]

        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = self.device

        def _mem(tag: str) -> None:
            if (not mem_debug) or (not torch.cuda.is_available()) or (model_device.type != "cuda"):
                return
            try:
                dev_idx = int(model_device.index) if model_device.index is not None else torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info(dev_idx)
                alloc = torch.cuda.memory_allocated(dev_idx)
                reserv = torch.cuda.memory_reserved(dev_idx)
                peak = torch.cuda.max_memory_allocated(dev_idx)
                print(
                    f"[QwenVLM][mem]{tag} dev={dev_idx} "
                    f"free={free/1024**3:.2f}GiB total={total/1024**3:.2f}GiB "
                    f"alloc={alloc/1024**3:.2f}GiB reserved={reserv/1024**3:.2f}GiB peak={peak/1024**3:.2f}GiB"
                )
            except Exception as e:
                print(f"[QwenVLM][mem]{tag} <mem_get_failed>: {repr(e)}")

        def _is_oom(e: Exception) -> bool:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                return True
            msg = str(e).lower()
            return "cuda out of memory" in msg or ("cublas" in msg and "alloc" in msg)

        def _resize_pil(im: Image.Image, max_side: Optional[int]) -> Image.Image:
            return self._resize_pil(im, max_side)

        attempts = []
        if len(images) == 0:
            for ts in oom_retry_token_scales:
                attempts.append((None, float(ts)))
        else:
            attempts.append((None, float(oom_retry_token_scales[0])))
            for ms in oom_retry_max_sides:
                attempts.append((int(ms), float(oom_retry_token_scales[0])))
            last_ms = int(oom_retry_max_sides[-1]) if len(oom_retry_max_sides) > 0 else None
            for ts in oom_retry_token_scales[1:]:
                attempts.append((last_ms, float(ts)))

        uniq = []
        seen = set()
        for ms, ts in attempts:
            key = (ms, round(ts, 4))
            if key not in seen:
                seen.add(key)
                uniq.append((ms, ts))
        attempts = uniq

        last_err = None
        for max_side, token_scale in attempts:
            contents = None
            messages = None
            inputs = None
            generated_ids = None
            generated_ids_trimmed = None
            pixel_values = None
            num_patches_list = None

            eff_max_new_tokens = max(1, int(max_new_tokens * token_scale))
            if max_new_tokens_cap is not None:
                eff_max_new_tokens = min(eff_max_new_tokens, int(max_new_tokens_cap))

            try:
                if self.local_backend.startswith("internvl"):
                    generation_config = {
                        "max_new_tokens": eff_max_new_tokens,
                        "do_sample": False,
                        "temperature": 0.0,
                    }

                    if len(images) == 0:
                        question = str(prompt or "").strip()
                        _mem(" before internvl text-only chat")
                        out = self.model.chat(
                            tokenizer=self.tokenizer,
                            pixel_values=None,
                            question=question,
                            generation_config=generation_config,
                            history=None,
                            return_history=False,
                            verbose=False,
                        )
                    else:
                        pixel_values, num_patches_list = self._prepare_internvl_inputs(images, max_side)
                        pixel_values = pixel_values.to(device=model_device, dtype=self.model_dtype)

                        if len(images) == 1:
                            question = str(prompt or "").strip()
                            _mem(" before internvl single-image chat")
                            out = self.model.chat(
                                tokenizer=self.tokenizer,
                                pixel_values=pixel_values,
                                question=question,
                                generation_config=generation_config,
                                history=None,
                                return_history=False,
                                num_patches_list=num_patches_list,
                                verbose=False,
                            )
                        else:
                            question = self._build_internvl_question(prompt, len(images))
                            _mem(" before internvl multi-image chat")
                            out = self.model.chat(
                                tokenizer=self.tokenizer,
                                pixel_values=pixel_values,
                                question=question,
                                generation_config=generation_config,
                                history=None,
                                return_history=False,
                                num_patches_list=num_patches_list,
                                verbose=False,
                            )

                    if isinstance(out, tuple):
                        out = out[0]
                else:
                    pil_images = [self._resize_image_for_attempt(x, max_side, _resize_pil) for x in images]
                    contents = [{"type": "text", "text": prompt}]
                    for im in pil_images:
                        contents.append({"type": "image", "image": im})
                    messages = [{"role": "user", "content": contents}]

                    _mem(" before processor")
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}

                    _mem(" before generate")
                    with torch.no_grad():
                        generated_ids = self.model.generate(**inputs, max_new_tokens=eff_max_new_tokens)
                    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]
                    out = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                return str(out).strip()

            except Exception as e:
                last_err = e
                if not oom_retry_enable or not _is_oom(e):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if gc_on_return:
                    gc.collect()
                time.sleep(max(0.0, oom_retry_sleep_s))
            finally:
                del contents, messages, inputs, generated_ids, generated_ids_trimmed, pixel_values, num_patches_list
                if gc_on_return:
                    gc.collect()
                if empty_cache_on_return and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if last_err is not None:
            raise last_err
        return ""

    def set_chat_logger(self, logger: Any) -> None:
        self._chat_state.logger = logger

    def set_chat_context(self, context: Dict[str, Any] | None) -> None:
        self._chat_state.context = dict(context or {})

    def _emit_chat_record(
        self,
        *,
        request: Dict[str, Any],
        response: Any = None,
        error: BaseException | None = None,
    ) -> None:
        logger = getattr(self._chat_state, "logger", None)
        if logger is None:
            return
        try:
            logger.log_interaction(
                model_kind="vlm",
                model_name=self._current_model_name(),
                mode=self.mode,
                context=dict(getattr(self._chat_state, "context", {}) or {}),
                request=request,
                response=response,
                error=error,
            )
        except Exception:
            pass

    def _current_model_name(self) -> str:
        if self.mode == "api":
            if self.api_client is not None:
                return str(getattr(self.api_client, "model", "") or "api")
            return "api"
        return str(getattr(self, "local_backend", "") or "local")

    def _summarize_images(self, images: List[ImageLike]) -> List[str]:
        refs: List[str] = []
        for img in images or []:
            if isinstance(img, str):
                refs.append(img)
            elif hasattr(img, "filename") and getattr(img, "filename", None):
                refs.append(str(getattr(img, "filename")))
            else:
                refs.append(f"<{type(img).__name__}>")
        return refs

    def _resize_pil(self, im: Image.Image, max_side: Optional[int]) -> Image.Image:
        if max_side is None or int(max_side) <= 0:
            return im
        w, h = im.size
        m = max(w, h)
        if m <= max_side:
            return im
        scale = float(max_side) / float(m)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return im.resize((nw, nh), resample=Image.BICUBIC)

    def _resize_image_for_attempt(
        self,
        img: ImageLike,
        max_side: Optional[int],
        resize_fn: Any,
    ) -> Image.Image:
        return resize_fn(self._load_image(img), max_side)
