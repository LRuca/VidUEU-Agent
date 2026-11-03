from __future__ import annotations

import base64
import io
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from PIL import Image
from executor.models.api_budget import ApiBudget
import hashlib
from pathlib import Path

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


ImageLike = Union[str, Image.Image]


class VLLMVisionApi:
    """OpenAI-compatible vLLM client for multimodal chat."""

    def __init__(self, cfg: Dict[str, Any]):
        if OpenAI is None:
            raise RuntimeError("VLLMVisionApi requires the openai package")

        self.base_url = str(cfg.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise ValueError("VLLMVisionApi requires api.base_url")

        self.model = str(cfg.get("model") or "").strip()
        if not self.model:
            raise ValueError("VLLMVisionApi requires api.model")

        api_key = str(cfg.get("api_key") or "").strip()
        if not api_key and cfg.get("api_key_file"):
            api_key = Path(os.path.expandvars(str(cfg["api_key_file"]))).read_text(encoding="utf-8").strip()
        api_key_env = str(cfg.get("api_key_env") or "VLLM_API_KEY").strip()
        if not api_key:
            api_key = str(os.getenv(api_key_env, "") or "").strip()
        self.api_key = api_key or "EMPTY"
        self.wire_profile = str(cfg.get("wire_profile", "vllm"))
        if self.wire_profile not in {"vllm", "openai"}:
            raise ValueError("Unknown vision API wire profile")
        self.reasoning_effort = str(cfg.get("reasoning_effort", "")).strip()

        self.timeout_s = float(cfg.get("timeout_s", 120.0) or 120.0)
        self.max_retries = int(cfg.get("max_retries", 0) or 0)
        self.enable_thinking = bool(cfg.get("enable_thinking", False))
        self.temperature = float(cfg.get("temperature", 0.0) or 0.0)
        self.top_p = float(cfg.get("top_p", 1.0) or 1.0)
        self.presence_penalty = float(cfg.get("presence_penalty", 0.0) or 0.0)
        self.top_k = int(cfg.get("top_k", 20) or 20)
        self.allow_local_file_urls = bool(cfg.get("allow_local_file_urls", True))
        self.jpeg_quality = int(cfg.get("jpeg_quality", 90) or 90)
        self.data_uri_max_side = int(cfg.get("data_uri_max_side", 0) or 0)
        self.request_retry_attempts = max(1, int(cfg.get("request_retry_attempts", 3) or 1))
        self.request_retry_backoff_s = max(0.05, float(cfg.get("request_retry_backoff_s", 0.5) or 0.5))
        self.request_retry_backoff_max_s = max(
            self.request_retry_backoff_s,
            float(cfg.get("request_retry_backoff_max_s", 3.0) or 3.0),
        )
        self.extra_body = dict(cfg.get("extra_body", {}) or {})
        self.api_budget = ApiBudget(str(cfg["budget_path"]), int(cfg["max_api_calls"])) if cfg.get("budget_path") else None
        self.retry_until_budget = bool(cfg.get("retry_until_budget", False))
        self.retry_until_success = bool(cfg.get("retry_until_success", False))
        if self.retry_until_budget and self.api_budget is None:
            raise ValueError("retry_until_budget requires a persistent API budget")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
            max_retries=0 if self.api_budget else self.max_retries,
        )

    def chat(
        self,
        images: List[ImageLike],
        prompt: str,
        max_new_tokens: int = 2048,
    ) -> str:
        content: List[Dict[str, Any]] = []
        for image in list(images or []):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_to_url(image),
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": str(prompt or ""),
            }
        )

        extra_body = dict(self.extra_body)
        if self.wire_profile == "vllm" and self.top_k > 0:
            extra_body.setdefault("top_k", int(self.top_k))
        if self.wire_profile == "vllm" and not self.enable_thinking:
            chat_template_kwargs = extra_body.get("chat_template_kwargs")
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            chat_template_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(max_new_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "presence_penalty": float(self.presence_penalty),
        }
        if self.wire_profile == "openai":
            for key in ["max_tokens", "temperature", "top_p", "presence_penalty"]:
                request_kwargs.pop(key)
            request_kwargs["max_completion_tokens"] = int(max_new_tokens)
            if self.reasoning_effort:
                request_kwargs["reasoning_effort"] = self.reasoning_effort
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        attempt = 0
        while True:
            attempt += 1
            budget_id = self.api_budget.reserve(hashlib.sha256(prompt.encode()).hexdigest(), self.model) if self.api_budget else None
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                usage = getattr(response, "usage", None)
                self.last_attestation = {"requested_model": self.model,
                    "response_model": getattr(response, "model", None), "response_id": getattr(response, "id", None),
                    "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
                    "reasoning_effort": self.reasoning_effort or None, "wire_profile": self.wire_profile,
                    "endpoint": self.base_url}
                if budget_id:
                    self.api_budget.finish(budget_id, "response", self.last_attestation)
                text = self._extract_content(response).strip()
                if not text:
                    raise RuntimeError("empty_content")
                return text
            except Exception as exc:
                if budget_id:
                    self.api_budget.finish(budget_id, "api_error", {"error_type": type(exc).__name__,
                        "http_status": getattr(exc, "status_code", None), "semantic_decision": None})
                if not (self.retry_until_budget or self.retry_until_success) and attempt >= self.request_retry_attempts:
                    raise
                sleep_s = min(
                    self.request_retry_backoff_max_s,
                    self.request_retry_backoff_s * (1.6 ** min(30, max(0, attempt - 1))),
                )
                print(
                    f"⚠️ [VLLMVisionApi] attempt={attempt} failed base_url={self.base_url} "
                    f"reason={type(exc).__name__}: {str(exc).replace(self.api_key, '[REDACTED]')}\n    -> sleep {sleep_s:.2f}s then retry"
                )
                time.sleep(sleep_s)

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    def _image_to_url(self, image: ImageLike) -> str:
        if isinstance(image, str):
            raw = str(image or "").strip()
            if raw.startswith(("http://", "https://", "data:")):
                return raw
            if raw.startswith("file://"):
                return self._path_to_data_uri(self._file_uri_to_path(raw))
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(f"VLLMVisionApi image not found: {raw}")
            return self._path_to_data_uri(path)
        if isinstance(image, Image.Image):
            return self._pil_to_data_uri(image.convert("RGB"))
        raise TypeError(f"Unsupported image type for VLLMVisionApi: {type(image)}")

    def _file_uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(str(uri or "").strip())
        if parsed.scheme.lower() != "file":
            raise ValueError(f"Unsupported file URI for VLLMVisionApi: {uri}")
        path_text = url2pathname(unquote(parsed.path or ""))
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            path_text = f"//{parsed.netloc}{path_text}"
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"VLLMVisionApi image not found: {uri}")
        return path

    def _path_to_data_uri(self, path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _pil_to_data_uri(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        if self.data_uri_max_side > 0:
            width, height = image.size
            max_side = max(width, height)
            if max_side > self.data_uri_max_side:
                scale = float(self.data_uri_max_side) / float(max_side)
                new_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                image = image.resize(new_size, resample=Image.BICUBIC)

        quality = max(30, min(int(self.jpeg_quality), 95))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _extract_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if message is None:
            return ""

        message_dict: Dict[str, Any] = {}
        if hasattr(message, "model_dump"):
            try:
                dumped = message.model_dump() or {}
                if isinstance(dumped, dict):
                    message_dict = dumped
            except Exception:
                message_dict = {}

        content = getattr(message, "content", None)
        if content is None:
            content = message_dict.get("content")

        answer_text = self._extract_answer_text(content)
        if answer_text:
            return answer_text

        # Thinking-enabled responses may carry reasoning separately; we intentionally
        # do not return reasoning as the final answer, but we do inspect common
        # payload shapes before concluding there is no answer text.
        for key in ("output_text", "text", "final_text", "answer"):
            value = message_dict.get(key)
            extracted = self._extract_answer_text(value)
            if extracted:
                return extracted

        return ""

    def _extract_answer_text(self, payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            part_type = str(payload.get("type", "") or "").strip().lower()
            if part_type in {"reasoning", "thinking", "reasoning_content"}:
                return ""
            for key in ("text", "content", "output_text", "value"):
                if key in payload:
                    extracted = self._extract_answer_text(payload.get(key))
                    if extracted:
                        return extracted
            return ""
        if isinstance(payload, list):
            texts: List[str] = []
            for item in payload:
                if isinstance(item, dict):
                    part_type = str(item.get("type", "") or "").strip().lower()
                    if part_type in {"reasoning", "thinking", "reasoning_content"}:
                        continue
                    for key in ("text", "content", "output_text", "value"):
                        if key in item:
                            extracted = self._extract_answer_text(item.get(key))
                            if extracted:
                                texts.append(extracted)
                                break
                else:
                    extracted = self._extract_answer_text(item)
                    if extracted:
                        texts.append(extracted)
            return "\n".join(texts).strip()
        return str(payload).strip()
