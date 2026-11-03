from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict


class ChatTranscriptLogger:
    """Append per-call LLM/VLM records to a JSONL file."""

    def __init__(self, output_path: str):
        self.output_path = str(output_path)
        self._lock = threading.Lock()
        self._turn_id = 0
        self._stage_turn_ids: Dict[str, int] = {}
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self,
        *,
        model_kind: str,
        model_name: str,
        mode: str,
        request: Dict[str, Any],
        response: Any = None,
        error: BaseException | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        context = self._jsonable(context or {})
        stage_key = self._stage_key(context)
        payload = {
            "call_id": uuid.uuid4().hex,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "unix_ts": time.time(),
            "model_kind": str(model_kind or ""),
            "model_name": str(model_name or ""),
            "mode": str(mode or ""),
            "context": context,
            "request": self._jsonable(request or {}),
            "response": self._jsonable(response),
        }
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        with self._lock:
            self._turn_id += 1
            payload["turn_id"] = int(self._turn_id)
            self._stage_turn_ids[stage_key] = int(self._stage_turn_ids.get(stage_key, 0)) + 1
            payload["stage_turn_id"] = int(self._stage_turn_ids[stage_key])
            line = json.dumps(payload, ensure_ascii=False)
            with open(self.output_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _jsonable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._jsonable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, os.PathLike):
            return os.fspath(obj)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _stage_key(self, context: Dict[str, Any]) -> str:
        stage = str(context.get("stage", "") or "")
        segment_index = context.get("segment_index", "")
        return f"{stage}#{segment_index}"
