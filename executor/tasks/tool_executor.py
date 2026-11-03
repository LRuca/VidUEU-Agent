from __future__ import annotations

import importlib
import time
from typing import Any, Dict, List

from executor.tasks.tool_registry import ToolRegistry
from executor.tools.base import ToolContext


class ToolExecutor:
    """
    Execute planned T2 tools through registry-driven adapters and attach
    normalized per-frame tool signals.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.registry = ToolRegistry(enabled=self.config.get("runtime", {}).get("enabled_tools"))
        self.adapters = self._load_adapters()
        runtime_cfg = (self.config.get("runtime", {}) or {}) if isinstance(self.config, dict) else {}
        tool_executor_cfg = (
            (runtime_cfg.get("tool_executor", {}) or {})
            if isinstance(runtime_cfg.get("tool_executor", {}), dict)
            else {}
        )
        self.retry_attempts = max(1, int(tool_executor_cfg.get("retry_attempts", 2) or 1))
        self.retry_backoff_s = max(0.0, float(tool_executor_cfg.get("retry_backoff_s", 0.5) or 0.0))

    def execute(
        self,
        *,
        shots: List[Any],
        candidates_by_shot: Dict[str, List[Dict[str, Any]]],
        pre_gate_candidates_by_shot: Dict[str, List[Dict[str, Any]]] | None = None,
        tool_plan: Dict[str, Any],
        video_path: str,
        t2_workdir: str,
        model_manager: Any,
    ) -> Dict[str, Any]:
        shot_map = {str(x.shot_id): x for x in shots}
        shot_plans = (tool_plan.get("shot_plans", {}) or {}) if isinstance(tool_plan, dict) else {}

        raw_by_shot: Dict[str, Dict[str, Any]] = {}
        semantic_by_shot: Dict[str, Dict[str, Any]] = {}
        enriched_candidates_by_shot: Dict[str, List[Dict[str, Any]]] = {}

        for shot_id, candidates in (candidates_by_shot or {}).items():
            shot_id = str(shot_id)
            shot = shot_map.get(shot_id)
            shot_plan = shot_plans.get(shot_id, {}) if isinstance(shot_plans, dict) else {}
            tool_actions = list(shot_plan.get("tool_actions", []) or [])
            selected_tools = self._tool_names_from_actions(tool_actions)
            pre_gate_candidates = list((pre_gate_candidates_by_shot or {}).get(shot_id, []) or candidates)

            tool_raw: Dict[str, Any] = {}
            tool_semantic: Dict[str, Any] = {}
            for tool_name in selected_tools:
                adapter = self.adapters.get(tool_name)
                if adapter is None:
                    raise KeyError(f"Tool adapter not found for {tool_name}")
                result = self._run_tool_with_retry(
                    adapter=adapter,
                    tool_name=tool_name,
                    shot_id=shot_id,
                    ctx=ToolContext(
                        shot=shot,
                        candidates=list(candidates),
                        pre_gate_candidates=pre_gate_candidates,
                        video_path=video_path,
                        t2_workdir=t2_workdir,
                        model_manager=model_manager,
                        config=self.config,
                        tool_spec=self.registry.get(tool_name),
                    ),
                )
                tool_raw[tool_name] = dict(result.get("raw", {}) or {})
                tool_semantic[tool_name] = dict(result.get("semantic", {}) or {})

            raw_by_shot[shot_id] = tool_raw
            semantic_by_shot[shot_id] = tool_semantic
            enriched_candidates_by_shot[shot_id] = self._attach_semantic_signals(
                candidates=candidates,
                tool_actions=tool_actions,
                selected_tools=selected_tools,
                shot_semantic=tool_semantic,
                shot_raw=tool_raw,
            )

        return {
            "tool_outputs_by_shot": {
                shot_id: {
                    tool_name: {
                        "raw": raw_by_shot.get(shot_id, {}).get(tool_name, {}),
                        "semantic": semantic_by_shot.get(shot_id, {}).get(tool_name, {}),
                    }
                    for tool_name in semantic_by_shot.get(shot_id, {}).keys()
                }
                for shot_id in semantic_by_shot.keys()
            },
            "raw_by_shot": raw_by_shot,
            "semantic_by_shot": semantic_by_shot,
            "enriched_candidates_by_shot": enriched_candidates_by_shot,
        }

    def group_by_shot(self, frames: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for frame in frames or []:
            out.setdefault(str(frame.get("shot_id", "")), []).append(frame)
        return out

    def _load_adapters(self) -> Dict[str, Any]:
        adapters: Dict[str, Any] = {}
        for tool_name, spec in self.registry.enabled_tools().items():
            class_path = str((((spec.get("adapter", {}) or {}).get("class_path")) or "")).strip()
            module_name, symbol = class_path.split(":", 1)
            module = importlib.import_module(module_name)
            adapters[tool_name] = getattr(module, symbol)()
        return adapters

    def _run_tool_with_retry(
        self,
        *,
        adapter: Any,
        tool_name: str,
        shot_id: str,
        ctx: ToolContext,
    ) -> Dict[str, Any]:
        attempt = 0
        last_error: Exception | None = None
        while attempt < self.retry_attempts:
            attempt += 1
            try:
                result = adapter.run_shot(ctx)
                if not isinstance(result, dict):
                    raise RuntimeError(
                        f"tool adapter returned invalid payload type={type(result)} for tool={tool_name}"
                    )
                return result
            except Exception as exc:
                last_error = exc
                print(
                    f"⚠️ [ToolExecutor] tool={tool_name} shot={shot_id} "
                    f"attempt={attempt}/{self.retry_attempts} failed: {type(exc).__name__}: {exc}"
                )
                if attempt >= self.retry_attempts:
                    break
                if self.retry_backoff_s > 0:
                    sleep_s = self.retry_backoff_s * (1.5 ** max(0, attempt - 1))
                    time.sleep(sleep_s)
        raise RuntimeError(
            f"Tool execution failed after {self.retry_attempts} attempts for tool={tool_name} shot={shot_id}: "
            f"{last_error}"
        ) from last_error

    def _attach_semantic_signals(
        self,
        *,
        candidates: List[Dict[str, Any]],
        tool_actions: List[Dict[str, Any]],
        selected_tools: List[str],
        shot_semantic: Dict[str, Any],
        shot_raw: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        hints_by_tool: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for tool_name in selected_tools:
            semantic = dict(shot_semantic.get(tool_name, {}) or {})
            indexed: Dict[str, Dict[str, Any]] = {}
            for hint in list(semantic.get("frame_hints", []) or []):
                if not isinstance(hint, dict):
                    continue
                frame_key = str(hint.get("frame_key", "") or "")
                if frame_key:
                    indexed[frame_key] = hint
            hints_by_tool[tool_name] = indexed

        out: List[Dict[str, Any]] = []
        for candidate in candidates:
            frame_key = self._build_frame_key(candidate)
            semantic_text_entries: List[Dict[str, Any]] = []
            tools_payload: Dict[str, Dict[str, Any]] = {}
            signal_scores: Dict[str, float] = {}

            for tool_name in selected_tools:
                hint = dict((hints_by_tool.get(tool_name, {}) or {}).get(frame_key, {}) or {})
                text = str(hint.get("text", "") or "").strip()
                score = float(hint.get("score", 0.0) or 0.0)
                attributes = dict(hint.get("attributes", {}) or {})
                evidence = dict(hint.get("evidence", {}) or {})
                raw_hint = {
                    str(k): v
                    for k, v in hint.items()
                    if str(k) not in {"frame_key", "img", "text", "score", "attributes", "evidence"}
                }
                tools_payload[tool_name] = {
                    "text": text,
                    "score": score,
                    "attributes": attributes,
                    "evidence": evidence,
                    "raw_hint": raw_hint,
                }
                signal_scores[tool_name] = score
                if text:
                    semantic_text_entries.append(
                        {
                            "tool_name": tool_name,
                            "text": text,
                            "weight": float((((self.registry.get(tool_name).get("semanticizer", {}) or {}).get("default_weight", 1.0)) or 1.0)),
                        }
                    )

            out.append(
                {
                    **candidate,
                    "tool_signals": {
                        "tool_actions": list(tool_actions),
                        "tools": tools_payload,
                        "semantic_texts": semantic_text_entries,
                        "semantic_summary": " | ".join(x["text"] for x in semantic_text_entries if str(x.get("text", "") or "").strip()),
                        "signal_scores": signal_scores,
                    },
                }
            )
        return out

    def _tool_names_from_actions(self, tool_actions: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for action in tool_actions or []:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "")
            if tool_name and tool_name not in out:
                out.append(tool_name)
        return out

    def _build_frame_key(self, candidate: Dict[str, Any]) -> str:
        return f"{str(candidate.get('shot_id', ''))}:{int(candidate.get('frame_index', -1))}"
