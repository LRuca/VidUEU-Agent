# executor/tasks/whisper_asr.py
"""
Whisper ASR executor (offline, local HF weights).

Compatibility:
- Model path:
  - Prefer config["models"]["whisper_model"]
  - Then config["models"]["whisper"]
  - If both missing, default to "models/whisper-large-v2"
    (in a fully offline env this will just fail to load gracefully)

- Task config (optional):
  - Prefer config["tasks"]["T2_asr"]
  - Fallback to config["tasks"]["T1_asr"]
  - Then config["tasks"]["T0_asr"]
  - If both missing, default:
      enabled = True
      language = "zh"
      task = "transcribe"
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


class WhisperASR:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 1. resolve model path
        models_conf = config.get("models", {}) if isinstance(config, dict) else {}
        model_path = models_conf.get("whisper_model") or models_conf.get("whisper")
        if not model_path:
            model_path = "models/whisper-large-v2"
        self.model_path = model_path

        # 2. resolve ASR task config
        tasks_conf = config.get("tasks", {}) if isinstance(config, dict) else {}
        t2_asr_conf = tasks_conf.get("T2_asr") or {}
        t1_asr_conf = tasks_conf.get("T1_asr") or {}
        t0_asr_conf = tasks_conf.get("T0_asr") or {}

        if t2_asr_conf:
            asr_conf = t2_asr_conf
        elif t1_asr_conf:
            asr_conf = t1_asr_conf
        elif t0_asr_conf:
            asr_conf = t0_asr_conf
        else:
            asr_conf = {}

        self.enabled: bool = asr_conf.get("enabled", True)
        self.device_preference: str = str(
            asr_conf.get("device") or config.get("device") or ""
        ).strip().lower()
        # For transformers Whisper, language/task are mainly controlled via generate_kwargs;
        # we keep these fields for compatibility.
        self.language: str = asr_conf.get("language", "zh")
        self.task: str = asr_conf.get("task", "transcribe")  # "transcribe" / "translate"

        if not self.enabled:
            print("🔇 [WhisperASR] Disabled by config (enabled = false), ASR will be skipped.")

        # 3. initialize local HF pipeline (fully offline)
        self.pipe = None
        self.init_error: str = ""
        self._init_pipeline()
        if self.enabled and self.pipe is None:
            raise RuntimeError(
                f"[WhisperASR] Pipeline initialization failed: {self.init_error or 'unknown error'}"
            )

    # ------------------------------------------------------------------
    # Internal: initialize transformers ASR pipeline
    # ------------------------------------------------------------------
    def _init_pipeline(self) -> None:
        if not self.enabled:
            return

        if pipeline is None:
            self.init_error = (
                "transformers.pipeline is not available. "
                "Please make sure 'transformers' is installed in this env."
            )
            print("❌ [WhisperASR] transformers.pipeline is not available. "
                  "Please make sure 'transformers' is installed in this env.")
            return

        if not os.path.exists(self.model_path):
            # We still allow HF hub ids here, but in offline env this will just fail.
            print(
                f"⚠️ [WhisperASR] Model path does not exist locally: {self.model_path}. "
                f"If you are offline, please place a Whisper HF model under this path."
            )

        print(f"🤖 [WhisperASR] Initializing HF ASR pipeline with model: {self.model_path}")

        # NOTE: To avoid the long-form error:
        #   "more than 3000 mel input features (> 30 seconds) ..."
        # we explicitly:
        #   - set chunk_length_s=30 so the pipeline chunks long audio
        #   - set return_timestamps=True so long-form generation is enabled properly
        #
        # We also set generate_kwargs for language/task if needed.
        generate_kwargs: Dict[str, Any] = {}

        # Only set task if it is one of the expected ones
        if self.task in {"transcribe", "translate"}:
            generate_kwargs["task"] = self.task

        # Language is optional; transformers will try to detect if not provided
        if self.language:
            # Whisper expects language codes like "en", "zh", etc.
            generate_kwargs["language"] = self.language

        if torch is None:
            device = -1
        else:
            if self.device_preference.startswith("cpu"):
                device = -1
            elif self.device_preference.startswith("cuda:"):
                try:
                    device = int(self.device_preference.split(":", 1)[1])
                except Exception:
                    device = 0 if torch.cuda.is_available() else -1
            elif self.device_preference == "cuda":
                device = 0 if torch.cuda.is_available() else -1
            else:
                device = 0 if torch.cuda.is_available() else -1

        try:
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.model_path,
                # Important for long audio:
                chunk_length_s=30,            # chunk into 30s segments
                return_timestamps=True,       # enable long-form mode safely
                generate_kwargs=generate_kwargs,
                device=device,
            )
            self.init_error = ""
        except Exception as e:
            self.init_error = str(e)
            print(f"❌ [WhisperASR] Failed to init HF ASR pipeline: {e}")
            self.pipe = None

    # ------------------------------------------------------------------
    # Public: run ASR
    # ------------------------------------------------------------------
    def run(self, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run ASR on the given audio file.

        Returns:
            text:    full transcription string
            segments: list of segment dicts (if available), typically result["chunks"]
        """
        if not self.enabled:
            print("🔇 [WhisperASR] Disabled, returning empty ASR result.")
            return "", []

        if self.pipe is None:
            print("⚠️ [WhisperASR] Pipeline not initialized, run() will return empty result.")
            return "", []

        if not audio_path or not os.path.exists(audio_path):
            print(f"⚠️ [WhisperASR] Audio file not found: {audio_path}")
            return "", []

        print(f"🎧 [WhisperASR] Start transcribing audio: {audio_path}")

        try:
            # Always ask for timestamps to be compatible with long-form mode in newer transformers.
            result = self.pipe(audio_path, return_timestamps=True)
        except Exception as e:
            print(f"❌ [WhisperASR] Transcription failed: {e}")
            return "", []

        # Normalize different pipeline outputs
        # Case 1: dict with "text" and "chunks"
        if isinstance(result, dict):
            text = result.get("text", "")
            segments = result.get("chunks", [])
        else:
            # Case 2: list[dict], some older transformers versions
            try:
                text = " ".join(r.get("text", "") for r in result)
            except Exception:
                text = ""
            segments = result  # keep raw for logging/debug

        return text, segments
    
    def transcribe(self, audio_path: str) -> str:
        """
        兼容接口：
        - 旧逻辑使用 run(audio_path) -> (text, segments)
        - 现在的 PipelinePlanner 使用 transcribe(audio_path) -> text
        这里做一个简单封装，内部复用 run()，不改原有行为。
        """
        text, _ = self.run(audio_path)
        return text
