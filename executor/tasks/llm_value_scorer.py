from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from executor.tasks.prompt_loader import load_prompt_text
from executor.tasks.tool_registry import ToolRegistry


class LLMValueScorer:
    """
    Shot-batched absolute value scorer.

    Output fields:
    - value_profile: {value_score, value_reason, focus_used, semantic_text, caption}
    - value_score
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.registry = ToolRegistry(enabled=self.config.get("runtime", {}).get("enabled_tools"))
        self.system_prompt = load_prompt_text("llm_value_analyse")
        cfg = ((self.config.get("tasks", {}) or {}).get("T2_llm_value_scoring", {}) or {})
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1200) or 1200)

    def analyze(
        self,
        semanticized_frames: List[Dict[str, Any]],
        shots: List[Any],
        chapters: List[Any],
        video_profile: Any,
        tool_plan: Dict[str, Any],
        llm: Any = None,
    ) -> List[Dict[str, Any]]:
        if not semanticized_frames:
            return []

        shot_map = {str(getattr(x, "shot_id", "")): x for x in shots}
        chapter_map = {str(getattr(x, "chapter_id", "")): x for x in chapters}
        shot_plans = (tool_plan.get("shot_plans", {}) or {}) if isinstance(tool_plan, dict) else {}

        by_shot: Dict[str, List[Dict[str, Any]]] = {}
        for frame in semanticized_frames:
            by_shot.setdefault(str(frame.get("shot_id", "") or ""), []).append(frame)

        out: List[Dict[str, Any]] = []
        for shot_id, frames in by_shot.items():
            frames = sorted(frames, key=lambda x: (float(x.get("sec", 0.0) or 0.0), int(x.get("frame_index", -1) or -1)))
            shot = shot_map.get(shot_id)
            chapter = chapter_map.get(str(frames[0].get("chapter_id", "") or "")) if frames else None
            shot_plan = shot_plans.get(shot_id, {}) if isinstance(shot_plans, dict) else {}
            score_focus = [tool_name for tool_name in list(shot_plan.get("score_focus", []) or []) if tool_name in self.registry.scorer_enabled_tools()]

            score_map = self._score_shot(
                frames=frames,
                shot=shot,
                chapter=chapter,
                video_profile=video_profile,
                tool_actions=list(shot_plan.get("tool_actions", []) or []),
                score_focus=score_focus,
                llm=llm,
            )

            for frame in frames:
                frame_id = self._frame_id(frame)
                row = score_map.get(frame_id) or self._fallback_score(frame, score_focus=score_focus)
                semantic_text = str(((frame.get("semantic_profile", {}) or {}).get("semantic_text", "") or ""))
                gateb_caption = str((((frame.get("quality_profile", {}) or {}).get("caption", "")) or ""))
                value_profile = {
                    "value_score": float(row["value_score"]),
                    "value_reason": str(row["value_reason"]),
                    "focus_used": list(score_focus),
                    "semantic_text": semantic_text,
                    "caption": gateb_caption,
                }
                out.append({**frame, "value_profile": value_profile, "value_score": float(value_profile["value_score"])})
        return out

    def _score_shot(
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
            "video_context": {
                "selection_objective": str(getattr(video_profile, "selection_objective", "") or "balanced_chapter_coverage"),
                "video_type": str(getattr(video_profile, "video_type", "") or "generic"),
                "video_summary": str(getattr(video_profile, "video_summary", "") or ""),
            },
            "shot_context": {
                "shot_id": str(getattr(shot, "shot_id", "") or ""),
                "shot_role": str(getattr(shot, "shot_role", "") or "generic"),
                "scene_type": str(getattr(shot, "scene_type", "") or "generic"),
                "shot_summary": str(getattr(shot, "vlm_summary", "") or ""),
                "chapter_id": str(getattr(chapter, "chapter_id", "") or ""),
                "chapter_role": str(getattr(chapter, "chapter_role", "") or "generic"),
                "chapter_summary": str(getattr(chapter, "chapter_summary", "") or ""),
                "active_tools": list(active_tools),
                "score_focus": list(score_focus or []),
            },
            "frames": [self._build_frame_payload(x, active_tools=active_tools) for x in frames],
        }
        prompt = (
            self.system_prompt.strip()
            + "\n\n"
            + "Tools selected for this shot:\n"
            + self._build_tool_purpose_block(active_tools)
            + "\n\n"
            + "Additional instruction for this run:\n"
            + "- If a frame includes a non-empty caption field, treat it as an auxiliary factual visual summary produced during GateB quality inspection.\n"
            + "- Use the caption as supporting evidence, but do not let it override semantic_text or tool_semantics when they are more specific.\n"
            + "\n\n"
            + "Return JSON with exactly this structure:\n"
            + "{\n"
            + '  "frames": [\n'
            + "    {\n"
            + '      "frame_id": "...",\n'
            + '      "value_score": 0.0,\n'
            + '      "value_reason": "..."\n'
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
            out[frame_id] = {
                "value_score": self._clip01(row.get("value_score", 0.0)),
                "value_reason": str(row.get("value_reason", "") or ""),
            }
        return out

    def _build_frame_payload(self, frame: Dict[str, Any], active_tools: List[str]) -> Dict[str, Any]:
        sp = dict(frame.get("semantic_profile", {}) or {})
        tool_descs = dict(sp.get("tool_descs", {}) or {})
        tool_semantics = {
            tool_name: str(tool_descs.get(tool_name, "") or "")
            for tool_name in active_tools
            if str(tool_descs.get(tool_name, "") or "").strip()
        }
        return {
            "frame_id": self._frame_id(frame),
            "sec": float(frame.get("sec", 0.0) or 0.0),
            "candidate_type": str(frame.get("candidate_type", "") or ""),
            "tool_semantics": tool_semantics,
            "semantic_text": str(sp.get("semantic_text", "") or ""),
            "caption": str((((frame.get("quality_profile", {}) or {}).get("caption", "")) or "")),
        }

    def _resolve_active_tools(self, frames: List[Dict[str, Any]], tool_actions: List[Dict[str, Any]]) -> List[str]:
        observed: List[str] = []
        for action in tool_actions or []:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "").strip()
            if tool_name and tool_name in self.registry.scorer_enabled_tools() and tool_name not in observed:
                observed.append(tool_name)
        for frame in frames or []:
            tool_descs = dict(((frame.get("semantic_profile", {}) or {}).get("tool_descs", {}) or {}))
            for tool_name, text in tool_descs.items():
                if tool_name in self.registry.scorer_enabled_tools() and str(text or "").strip() and tool_name not in observed:
                    observed.append(tool_name)
        return observed

    def _build_tool_purpose_block(self, active_tools: List[str]) -> str:
        return "\n".join(
            f"- {tool_name}: {str(self.registry.get(tool_name).get('purpose', '') or '').strip()}"
            for tool_name in active_tools
        )

    def _fallback_score(self, frame: Dict[str, Any], score_focus: List[str]) -> Dict[str, Any]:
        semantic_text = str(((frame.get("semantic_profile", {}) or {}).get("semantic_text", "") or "")).strip()
        tool_descs = dict(((frame.get("semantic_profile", {}) or {}).get("tool_descs", {}) or {}))
        tool_hits = sum(1 for tool_name in score_focus if str(tool_descs.get(tool_name, "") or "").strip())
        base = 0.35
        if semantic_text:
            base += 0.2
        base += min(0.35, 0.12 * tool_hits)
        return {
            "value_score": self._clip01(base),
            "value_reason": f"semantic_text={bool(semantic_text)}; focused_tools={tool_hits}",
        }

    def _frame_id(self, frame: Dict[str, Any]) -> str:
        return f"{str(frame.get('shot_id', ''))}:{int(frame.get('frame_index', -1))}"

    def _clip01(self, value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))

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
