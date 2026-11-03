from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ToolContext:
    shot: Any
    candidates: List[Dict[str, Any]]
    pre_gate_candidates: List[Dict[str, Any]]
    video_path: str
    t2_workdir: str
    model_manager: Any
    config: Dict[str, Any]
    tool_spec: Dict[str, Any]


class BaseToolAdapter:
    tool_name: str = ""

    def run_shot(self, ctx: ToolContext) -> Dict[str, Any]:
        raise NotImplementedError

    def frame_key(self, shot_id: str, frame_index: Any) -> str:
        return f"{str(shot_id)}:{int(frame_index)}"
