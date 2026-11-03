from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from executor.tasks.prompt_loader import load_prompt_text
from executor.tasks.tool_registry import ToolRegistry


class FrameSignalSemanticizer:
    """
    Convert per-frame tool evidence into frame-specific semantic descriptions.

    Output field:
    - semantic_profile: {tool_descs, semantic_text}
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.registry = ToolRegistry(enabled=self.config.get("runtime", {}).get("enabled_tools"))
        self.system_prompt = load_prompt_text("tool_signal_semanticizer")
        cfg = ((self.config.get("tasks", {}) or {}).get("T2_frame_signal_semanticizer", {}) or {})
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1400) or 1400)

    def build(
        self,
        enriched_frames: List[Dict[str, Any]],
        shots: List[Any],
        chapters: List[Any],
        video_profile: Any,
        tool_plan: Dict[str, Any],
        llm: Any = None,
    ) -> List[Dict[str, Any]]:
        if not enriched_frames:
            return []

        shot_map = {str(getattr(x, "shot_id", "")): x for x in shots}
        chapter_map = {str(getattr(x, "chapter_id", "")): x for x in chapters}
        shot_plans = (tool_plan.get("shot_plans", {}) or {}) if isinstance(tool_plan, dict) else {}

        by_shot: Dict[str, List[Dict[str, Any]]] = {}
        for frame in enriched_frames:
            by_shot.setdefault(str(frame.get("shot_id", "") or ""), []).append(frame)

        out: List[Dict[str, Any]] = []
        for shot_id, frames in by_shot.items():
            frames = sorted(frames, key=lambda x: (float(x.get("sec", 0.0) or 0.0), int(x.get("frame_index", -1) or -1)))
            shot = shot_map.get(shot_id)
            chapter = chapter_map.get(str(frames[0].get("chapter_id", "") or "")) if frames else None
            shot_plan = shot_plans.get(shot_id, {}) if isinstance(shot_plans, dict) else {}

            llm_result = self._semanticize_shot(
                frames=frames,
                shot=shot,
                chapter=chapter,
                video_profile=video_profile,
                tool_actions=list(shot_plan.get("tool_actions", []) or []),
                score_focus=list(shot_plan.get("score_focus", []) or []),
                llm=llm,
            )

            for frame in frames:
                frame_id = self._frame_id(frame)
                semantic_profile = llm_result.get(frame_id) or self._fallback_semantic_profile(frame)
                out.append(
                    {
                        **frame,
                        "semantic_profile": {
                            "tool_descs": dict(semantic_profile.get("tool_descs", {}) or {}),
                            "semantic_text": str(semantic_profile.get("semantic_text", "") or ""),
                        },
                    }
                )
        return out

    def _semanticize_shot(
        self,
        frames: List[Dict[str, Any]],
        shot: Any,
        chapter: Any,
        video_profile: Any,
        tool_actions: List[Dict[str, Any]],
        score_focus: List[str],
        llm: Any,
    ) -> Dict[str, Dict[str, Any]]:
        active_tools = self._resolve_active_tools(frames=frames, tool_actions=tool_actions)
        if llm is None or not active_tools:
            return {}

        payload = {
            "selection_objective": str(getattr(video_profile, "selection_objective", "") or "balanced_chapter_coverage"),
            "shot_context": {
                "shot_id": str(getattr(shot, "shot_id", "") or ""),
                "shot_role": str(getattr(shot, "shot_role", "") or "generic"),
                "scene_type": str(getattr(shot, "scene_type", "") or "generic"),
                "chapter_id": str(getattr(chapter, "chapter_id", "") or ""),
                "chapter_role": str(getattr(chapter, "chapter_role", "") or "generic"),
                "active_tools": list(active_tools),
                "score_focus": list(score_focus or []),
            },
            "tool_semanticization_specs": self._build_tool_semanticization_specs(active_tools),
            "frames": [self._build_frame_payload(x, active_tools=active_tools) for x in frames],
        }
        prompt = (
            self.system_prompt.strip()
            + "\n\n"
            + "Tools actually executed for this shot:\n"
            + self._build_tool_purpose_block(active_tools)
            + "\n\n"
            + "Tool-specific semanticization rules:\n"
            + self._build_tool_rule_block(active_tools)
            + "\n\n"
            + "Interpretation requirements for this run:\n"
            + "- For each tool, prefer the custom rules and structured evidence fields provided in Input JSON.\n"
            + "- Treat tool text as optional legacy hints; do not require it when structured fields already provide enough evidence.\n"
            + "- If a tool exposes raw_hint fields, use them as additional structured evidence instead of ignoring them.\n"
            + "- Keep each tool description grounded in the provided tool evidence for that exact frame.\n"
            + "\n\n"
            + "Return JSON with exactly this structure:\n"
            + "{\n"
            + '  "frames": [\n'
            + "    {\n"
            + '      "frame_id": "...",\n'
            + '      "tool_descs": {"tool_name": "description"},\n'
            + '      "semantic_text": "..."\n'
            + "    }\n"
            + "  ]\n"
            + "}\n\n"
            + "Input JSON:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        try:
            raw = llm.chat(prompt, max_new_tokens=self.max_new_tokens)
        except Exception:
            return {}
        obj = self._parse_json_obj(raw)
        if not isinstance(obj, dict):
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for row in list(obj.get("frames", []) or []):
            if not isinstance(row, dict):
                continue
            frame_id = str(row.get("frame_id", "") or "")
            if not frame_id:
                continue
            tool_descs = {
                str(k): str(v or "")
                for k, v in dict(row.get("tool_descs", {}) or {}).items()
                if str(k or "").strip()
            }
            out[frame_id] = {
                "tool_descs": tool_descs,
                "semantic_text": str(row.get("semantic_text", "") or ""),
            }
        return out

    def _build_frame_payload(self, frame: Dict[str, Any], active_tools: List[str]) -> Dict[str, Any]:
        tool_signals = dict(frame.get("tool_signals", {}) or {})
        tools = dict(tool_signals.get("tools", {}) or {})
        tool_evidence: Dict[str, Any] = {}
        for tool_name in active_tools:
            payload = dict(tools.get(tool_name, {}) or {})
            tool_evidence[tool_name] = {
                "text": str(payload.get("text", "") or ""),
                "score": float(payload.get("score", 0.0) or 0.0),
                "attributes": dict(payload.get("attributes", {}) or {}),
                "evidence": dict(payload.get("evidence", {}) or {}),
                "raw_hint": dict(payload.get("raw_hint", {}) or {}),
            }
        return {
            "frame_id": self._frame_id(frame),
            "sec": float(frame.get("sec", 0.0) or 0.0),
            "candidate_type": str(frame.get("candidate_type", "") or ""),
            "tool_evidence": tool_evidence,
        }

    def _fallback_semantic_profile(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        tool_signals = dict(frame.get("tool_signals", {}) or {})
        tools = dict(tool_signals.get("tools", {}) or {})
        tool_descs: Dict[str, str] = {}
        for tool_name, payload in tools.items():
            payload = dict(payload or {})
            desc = self._fallback_tool_desc(tool_name, payload)
            if desc:
                tool_descs[tool_name] = desc
        semantic_text = " | ".join(
            tool_descs[tool_name]
            for tool_name in sorted(tool_descs.keys())
            if str(tool_descs.get(tool_name, "") or "").strip()
        )
        return {"tool_descs": tool_descs, "semantic_text": semantic_text}

    def _resolve_active_tools(self, frames: List[Dict[str, Any]], tool_actions: List[Dict[str, Any]]) -> List[str]:
        observed: List[str] = []
        for action in tool_actions or []:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "").strip()
            if tool_name and tool_name in self.registry.semanticizer_enabled_tools() and tool_name not in observed:
                observed.append(tool_name)
        for frame in frames or []:
            tools = dict((frame.get("tool_signals", {}) or {}).get("tools", {}) or {})
            for tool_name, payload in tools.items():
                if tool_name not in self.registry.semanticizer_enabled_tools():
                    continue
                payload = dict(payload or {})
                if self._has_tool_evidence(payload) and tool_name not in observed:
                    observed.append(tool_name)
        return observed

    def _build_tool_purpose_block(self, active_tools: List[str]) -> str:
        lines: List[str] = []
        for tool_name in active_tools:
            spec = self.registry.get(tool_name)
            lines.append(f"- {tool_name}: {str(spec.get('purpose', '') or '').strip()}")
        return "\n".join(lines)

    def _build_tool_rule_block(self, active_tools: List[str]) -> str:
        lines: List[str] = []
        for tool_name in active_tools:
            spec = self.registry.get(tool_name)
            semanticizer = dict(spec.get("semanticizer", {}) or {})
            prompt_rules = [str(x or "").strip() for x in list(semanticizer.get("prompt_rules", []) or []) if str(x or "").strip()]
            evidence_fields = self._tool_evidence_field_names(tool_name)
            lines.append(f"- {tool_name}:")
            lines.append(f"  purpose: {str(spec.get('purpose', '') or '').strip()}")
            lines.append(f"  expected_fields: {', '.join(evidence_fields) if evidence_fields else 'text, score, attributes, evidence, raw_hint'}")
            if prompt_rules:
                for rule in prompt_rules:
                    lines.append(f"  rule: {rule}")
            else:
                lines.append("  rule: Use the structured evidence fields conservatively and describe only what is directly supported.")
        return "\n".join(lines)

    def _build_tool_semanticization_specs(self, active_tools: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for tool_name in active_tools:
            spec = self.registry.get(tool_name)
            semanticizer = dict(spec.get("semanticizer", {}) or {})
            prompt_rules = [str(x or "").strip() for x in list(semanticizer.get("prompt_rules", []) or []) if str(x or "").strip()]
            out[tool_name] = {
                "purpose": str(spec.get("purpose", "") or ""),
                "prompt_rules": prompt_rules,
                "expected_fields": self._tool_evidence_field_names(tool_name),
            }
        return out

    def _tool_evidence_field_names(self, tool_name: str) -> List[str]:
        return ["text", "score", "attributes", "evidence", "raw_hint"]

    def _has_tool_evidence(self, payload: Dict[str, Any]) -> bool:
        if str(payload.get("text", "") or "").strip():
            return True
        if dict(payload.get("attributes", {}) or {}):
            return True
        if dict(payload.get("evidence", {}) or {}):
            return True
        if dict(payload.get("raw_hint", {}) or {}):
            return True
        try:
            return float(payload.get("score", 0.0) or 0.0) > 0.0
        except Exception:
            return False

    def _fallback_tool_desc(self, tool_name: str, payload: Dict[str, Any]) -> str:
        text = str(payload.get("text", "") or "").strip()
        if text:
            return text
        has_structured_payload = any(
            bool(dict(payload.get(field, {}) or {}))
            for field in ("attributes", "evidence", "raw_hint")
        )
        if has_structured_payload:
            return f"{tool_name}: semantic summary unavailable"
        if self._has_tool_evidence(payload):
            try:
                score = float(payload.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            return f"{tool_name}: score={score:.3f}" if score > 0.0 else f"{tool_name}: signal detected"
        try:
            score = float(payload.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        return f"{tool_name}: score={score:.3f}" if score > 0.0 else ""

    def _frame_id(self, frame: Dict[str, Any]) -> str:
        return f"{str(frame.get('shot_id', ''))}:{int(frame.get('frame_index', -1))}"

    def _parse_json_obj(self, raw: Any) -> Dict[str, Any] | None:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if not text:
            return None
        for candidate in [text, self._extract_json_block(text)]:
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    def _extract_json_block(self, text: str) -> str | None:
        m = re.search(r"```json\s*(.*?)\s*```", text.strip(), flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return text[s : e + 1]
        return None
