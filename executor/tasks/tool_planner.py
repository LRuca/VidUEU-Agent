from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from orchestrator.structures import ChapterRecord, ShotRecord, VideoProfile
from executor.tasks.prompt_loader import load_prompt_text
from executor.tasks.tool_registry import ToolRegistry


class ToolPlanner:
    """
    LLM-based T2 tool planner.

    Output fields are strictly limited to:
    - global: selection_objective, allowed_tools, shot_count, chapter_count
    - chapter_plans[*]: chapter_id, tool_actions, plan_reason, score_focus (kept empty for compatibility)
    - shot_plans[*]: shot_id, tool_actions, score_focus, plan_reason
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.registry = ToolRegistry(enabled=self.config.get("runtime", {}).get("enabled_tools"))
        self.tool_catalog = self.registry.enabled_tools()
        self.chapter_system_prompt = load_prompt_text("toolplan_quality", "chapter_level")
        self.shot_system_prompt = load_prompt_text("toolplan_quality", "shot_level")
        self.allowed_tools = tuple(self.tool_catalog.keys())
        self.default_tool_by_role = self._build_default_tool_by_role()
        llm_cfg = ((self.config.get("tasks", {}) or {}).get("T2_tool_planning", {}) or {})
        self.max_new_tokens = int(llm_cfg.get("max_new_tokens", 768) or 768)
        self.max_chapter_tools = int(llm_cfg.get("max_chapter_tools", 2) or 2)
        self.max_shot_tools = int(llm_cfg.get("max_shot_tools", 4) or 4)

    def plan(
        self,
        video_profile: VideoProfile,
        chapters: List[ChapterRecord],
        shots: List[ShotRecord],
        kept_shot_ids: List[str] | None = None,
        llm: Any = None,
    ) -> Dict[str, Dict[str, Any]]:
        kept = set(kept_shot_ids or [x.shot_id for x in shots])
        kept_shots = [x for x in shots if x.shot_id in kept]

        chapter_plans = self._plan_all_chapters(
            video_profile=video_profile,
            chapters=chapters,
            llm=llm,
        )

        shot_plans: Dict[str, Dict[str, Any]] = {}
        for chapter in chapters:
            chapter_shots = [x for x in kept_shots if x.shot_id in set(chapter.shot_ids or [])]
            if not chapter_shots:
                continue
            chapter_plan = chapter_plans.get(chapter.chapter_id, {})
            shot_plans.update(
                self._plan_all_shots_in_chapter(
                    chapter=chapter,
                    chapter_shots=chapter_shots,
                    chapter_plan=chapter_plan,
                    video_profile=video_profile,
                    llm=llm,
                )
            )

        return {
            "global": {
                "selection_objective": str(getattr(video_profile, "selection_objective", "") or "balanced_chapter_coverage"),
                "allowed_tools": list(self.allowed_tools),
                "shot_count": int(len(kept)),
                "chapter_count": int(len(chapters)),
            },
            "chapter_plans": chapter_plans,
            "shot_plans": shot_plans,
        }

    def _plan_all_chapters(
        self,
        video_profile: VideoProfile,
        chapters: List[ChapterRecord],
        llm: Any,
    ) -> Dict[str, Dict[str, Any]]:
        if llm is None:
            return {
                str(chapter.chapter_id): self._fallback_chapter_plan(chapter)
                for chapter in chapters
            }
        payload = self._build_chapter_batch_payload(video_profile, chapters)
        raw = self._call_llm_chapter_batch(payload=payload, llm=llm)
        try:
            return self._validate_chapter_batch(raw=raw, chapters=chapters)
        except Exception:
            return {
                str(chapter.chapter_id): self._fallback_chapter_plan(chapter)
                for chapter in chapters
            }

    def _plan_all_shots_in_chapter(
        self,
        chapter: ChapterRecord,
        chapter_shots: List[ShotRecord],
        chapter_plan: Dict[str, Any],
        video_profile: VideoProfile,
        llm: Any,
    ) -> Dict[str, Dict[str, Any]]:
        if llm is None:
            return {
                str(shot.shot_id): self._fallback_shot_plan(shot, chapter, chapter_plan)
                for shot in chapter_shots
            }
        payload = self._build_shot_batch_payload(chapter, chapter_shots, chapter_plan, video_profile)
        raw = self._call_llm_shot_batch(payload=payload, llm=llm)
        try:
            return self._validate_shot_batch(
                raw=raw,
                chapter=chapter,
                chapter_shots=chapter_shots,
                chapter_plan=chapter_plan,
            )
        except Exception:
            return {
                str(shot.shot_id): self._fallback_shot_plan(shot, chapter, chapter_plan)
                for shot in chapter_shots
            }

    def _build_chapter_batch_payload(
        self,
        video_profile: VideoProfile,
        chapters: List[ChapterRecord],
    ) -> Dict[str, Any]:
        return {
            "video": {
                "selection_objective": str(getattr(video_profile, "selection_objective", "") or "balanced_chapter_coverage"),
                "video_type": str(getattr(video_profile, "video_type", "") or "generic"),
                "video_summary": str(getattr(video_profile, "video_summary", "") or ""),
            },
            "chapters": [
                {
                    "chapter_id": str(chapter.chapter_id),
                    "chapter_role": str(getattr(chapter, "chapter_role", "") or "generic"),
                    "chapter_summary": str(getattr(chapter, "chapter_summary", "") or ""),
                    "importance_hint": str(getattr(chapter, "importance_hint", "") or ""),
                    "coverage_need": str(getattr(chapter, "coverage_need", "") or ""),
                    "num_shots": int(len(chapter.shot_ids or [])),
                }
                for chapter in chapters
            ],
        }

    def _build_shot_batch_payload(
        self,
        chapter: ChapterRecord,
        chapter_shots: List[ShotRecord],
        chapter_plan: Dict[str, Any],
        video_profile: VideoProfile,
    ) -> Dict[str, Any]:
        return {
            "video_context": {
                "selection_objective": str(getattr(video_profile, "selection_objective", "") or "balanced_chapter_coverage"),
            },
            "chapter": {
                "chapter_id": str(chapter.chapter_id),
                "chapter_role": str(getattr(chapter, "chapter_role", "") or "generic"),
                "chapter_summary": str(getattr(chapter, "chapter_summary", "") or ""),
                "chapter_tools": [
                    str(x.get("tool_name", "") or "")
                    for x in list(chapter_plan.get("tool_actions", []) or [])
                    if isinstance(x, dict)
                ],
            },
            "shots": [
                {
                    "shot_id": str(shot.shot_id),
                    "shot_summary": str(getattr(shot, "vlm_summary", "") or ""),
                    "shot_role": str(getattr(shot, "shot_role", "") or "generic"),
                    "scene_type": str(getattr(shot, "scene_type", "") or "generic"),
                    "duration_sec": float(getattr(shot, "duration", 0.0) or 0.0),
                    "text_density": float(getattr(shot, "text_density", 0.0) or 0.0),
                    "speech_density": float(getattr(shot, "speech_density", 0.0) or 0.0),
                    "informativeness_hint": str(getattr(shot, "informativeness_hint", "") or ""),
                    "visual_quality_hint": str(getattr(shot, "visual_quality_hint", "") or ""),
                }
                for shot in chapter_shots
            ],
        }

    def _call_llm_chapter_batch(self, payload: Dict[str, Any], llm: Any) -> Dict[str, Any] | None:
        if llm is None:
            return None
        tool_guide = self._build_tool_guide_text()
        prompt = (
            self.chapter_system_prompt.strip()
            + "\n\n"
            + "Available tools for chapter planning:\n"
            + tool_guide
            + "\n\n"
            + "Task: plan tools for every chapter in this video.\n"
            + "Use the video summary for global context, then evaluate each chapter independently based on its own chapter summary.\n"
            + "For each chapter, choose exactly 1 or 2 most important tools only.\n"
            + "Choose the smallest sufficient tool set for each chapter.\n\n"
            + "Return JSON with exactly this structure:\n"
            + "{\n"
            + '  "chapter_plans": [\n'
            + "    {\n"
            + '      "chapter_id": "...",\n'
            + '      "tool_actions": [{"tool_name": "...", "reason": "..."}],\n'
            + '      "plan_reason": "..."\n'
            + "    }\n"
            + "  ]\n"
            + "}\n\n"
            + "Hard requirements:\n"
            + "- Return exactly one item for every chapter_id in the input.\n"
            + "- Do not invent chapter_id values.\n"
            + "- tool_actions must contain 1 or 2 tools.\n"
            + "- Do not return score_focus for chapter_plans.\n"
            + "- No text outside JSON.\n\n"
            + "Input JSON:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        raw = llm.chat(prompt, max_new_tokens=self.max_new_tokens)
        return self._parse_json_obj(raw)

    def _call_llm_shot_batch(self, payload: Dict[str, Any], llm: Any) -> Dict[str, Any] | None:
        if llm is None:
            return None
        chapter = dict(payload.get("chapter", {}) or {})
        recommended_tools = [
            str(x)
            for x in list(chapter.get("chapter_tools", []) or [])
            if str(x or "").strip()
        ]
        extra_tools = [x for x in self.allowed_tools if x not in recommended_tools]
        recommended_tool_guide = self._build_tool_guide_text(recommended_tools)
        extra_tool_guide = self._build_tool_guide_text(extra_tools)
        prompt = (
            self.shot_system_prompt.strip()
            + "\n\n"
            + "Chapter-level recommended tools for this chapter:\n"
            + (recommended_tool_guide if recommended_tool_guide else "- none\n")
            + "\n"
            + "These chapter-level tools are recommendations for this chapter.\n"
            + "When planning a shot, you may keep useful recommended tools and you may add a small number of extra tools when necessary.\n"
            + "If you keep a recommended tool for a shot, include it explicitly in that shot's final tool_actions.\n\n"
            + "Additional optional tools you may add only when clearly necessary for a specific shot:\n"
            + (extra_tool_guide if extra_tool_guide else "- none\n")
            + "\n"
            + "Task: plan tools for every shot inside this chapter.\n"
            + "Use the chapter summary as shared context, but decide each shot independently based on that shot's own summary.\n"
            + "Keep the plan minimal but sufficient for understanding each specific shot.\n"
            + "The returned tool_actions for each shot should be the final tool set for that shot.\n\n"
            + "Return JSON with exactly this structure:\n"
            + "{\n"
            + '  "shot_plans": [\n'
            + "    {\n"
            + '      "shot_id": "...",\n'
            + '      "tool_actions": [{"tool_name": "...", "reason": "..."}],\n'
            + '      "score_focus": ["..."],\n'
            + '      "plan_reason": "..."\n'
            + "    }\n"
            + "  ]\n"
            + "}\n\n"
            + "Hard requirements:\n"
            + "- Return exactly one item for every shot_id in the input.\n"
            + "- Do not invent shot_id values.\n"
            + "- Do not remove a recommended chapter-level tool unless it is clearly unnecessary for this shot.\n"
            + "- A shot may add extra tools only from the additional optional tools list above.\n"
            + "- If a recommended chapter-level tool is kept for a shot, it must appear explicitly in that shot's tool_actions.\n"
            + "- Decide each shot independently.\n"
            + "- tool_actions should usually contain 1 or 2 tools; use more only when clearly necessary.\n"
            + "- score_focus should contain 1 or 2 relatively more important tools chosen from that shot's final tool_actions.\n"
            + "- No text outside JSON.\n\n"
            + "Input JSON:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        raw = llm.chat(prompt, max_new_tokens=self.max_new_tokens)
        return self._parse_json_obj(raw)

    def _validate_chapter_batch(
        self,
        raw: Dict[str, Any] | None,
        chapters: List[ChapterRecord],
    ) -> Dict[str, Dict[str, Any]]:
        raw_rows = self._extract_plan_rows(raw, list_key="chapter_plans", id_key="chapter_id")
        raw_by_id = {
            str(row.get("chapter_id", "") or ""): row
            for row in raw_rows
            if str(row.get("chapter_id", "") or "")
        }
        if raw is None:
            raise ValueError("ToolPlanner chapter batch returned no JSON object")
        out: Dict[str, Dict[str, Any]] = {}
        for chapter in chapters:
            chapter_id = str(chapter.chapter_id)
            row = raw_by_id.get(chapter_id)
            if row is None:
                raise ValueError(f"Missing chapter plan for {chapter_id}")
            out[chapter_id] = self._validate_plan(
                raw=row,
                target_level="chapter",
                chapter_id=chapter_id,
                shot_id="",
                max_tools=self.max_chapter_tools,
                allow_score_focus=False,
            )
        return out

    def _validate_shot_batch(
        self,
        raw: Dict[str, Any] | None,
        chapter: ChapterRecord,
        chapter_shots: List[ShotRecord],
        chapter_plan: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        raw_rows = self._extract_plan_rows(raw, list_key="shot_plans", id_key="shot_id")
        raw_by_id = {
            str(row.get("shot_id", "") or ""): row
            for row in raw_rows
            if str(row.get("shot_id", "") or "")
        }
        if raw is None:
            raise ValueError("ToolPlanner shot batch returned no JSON object")
        out: Dict[str, Dict[str, Any]] = {}
        for shot in chapter_shots:
            shot_id = str(shot.shot_id)
            row = raw_by_id.get(shot_id)
            if row is None:
                raise ValueError(f"Missing shot plan for {shot_id}")
            out[shot_id] = self._validate_plan(
                raw=row,
                target_level="shot",
                chapter_id=str(chapter.chapter_id),
                shot_id=shot_id,
                max_tools=self.max_shot_tools,
            )
        return out

    def _extract_plan_rows(self, raw: Dict[str, Any] | None, list_key: str, id_key: str) -> List[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return []
        rows = raw.get(list_key, None)
        if isinstance(rows, list):
            return [x for x in rows if isinstance(x, dict)]
        if isinstance(rows, dict):
            out: List[Dict[str, Any]] = []
            for key, value in rows.items():
                if not isinstance(value, dict):
                    continue
                out.append({id_key: str(key), **value})
            return out
        return []

    def _build_default_tool_by_role(self) -> Dict[str, List[str]]:
        role_map: Dict[str, List[tuple[float, str]]] = {}
        for tool_name, spec in self.tool_catalog.items():
            planner_spec = dict(spec.get("planner", {}) or {})
            priority = float(planner_spec.get("priority", 0.0) or 0.0)
            for role in list(planner_spec.get("default_roles", []) or []):
                role_name = str(role or "").strip().lower()
                if not role_name:
                    continue
                role_map.setdefault(role_name, []).append((priority, tool_name))
        out: Dict[str, List[str]] = {}
        for role_name, items in role_map.items():
            items = sorted(items, key=lambda x: (-x[0], x[1]))
            out[role_name] = [tool_name for _, tool_name in items]
        return out

    def _build_tool_guide_text(self, tool_names: List[str] | tuple[str, ...] | None = None) -> str:
        names = [str(x) for x in (tool_names or self.allowed_tools) if str(x or "").strip()]
        lines: List[str] = []
        for tool_name in names:
            spec = dict(self.tool_catalog.get(tool_name, {}) or {})
            purpose = str(spec.get("purpose", "") or "")
            returns = ", ".join(spec.get("returns", []) or []) or "not specified"
            use_for = ", ".join(spec.get("use_for", []) or []) or "not specified"
            scenes = ", ".join(spec.get("scenes", []) or []) or "not specified"
            lines.append(f"- {tool_name}")
            if purpose:
                lines.append(f"  purpose: {purpose}")
            lines.append(f"  returns: {returns}")
            lines.append(f"  useful_for: {use_for}")
            lines.append(f"  recommended_scenes: {scenes}")
        return "\n".join(lines)

    def _fallback_chapter_plan(self, chapter: ChapterRecord) -> Dict[str, Any]:
        role = str(getattr(chapter, "chapter_role", "") or "").lower()
        tools = list(self.default_tool_by_role.get(role, list(self.allowed_tools)))[: self.max_chapter_tools]
        return {
            "tool_actions": [{"tool_name": t, "reason": f"chapter_role={role or 'generic'}"} for t in tools],
            "score_focus": [],
            "plan_reason": f"fallback chapter plan for role={role or 'generic'}",
        }

    def _fallback_shot_plan(
        self,
        shot: ShotRecord,
        chapter: ChapterRecord | None,
        chapter_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        role = str(getattr(shot, "shot_role", "") or "").lower()
        scene_type = str(getattr(shot, "scene_type", "") or "").lower()

        base_tools = [
            str(x.get("tool_name", "") or "")
            for x in list(chapter_plan.get("tool_actions", []) or [])
            if isinstance(x, dict)
        ]
        if not base_tools:
            base_tools = list(self.default_tool_by_role.get(str(getattr(chapter, "chapter_role", "") or "").lower(), list(self.allowed_tools)))

        tools = self._recommend_tools_for_shot(shot=shot, base_tools=base_tools, scene_type=scene_type, role=role)
        tools = self._dedup_tools(tools)

        focus: List[str] = [tool_name for tool_name in tools if self._tool_priority_matches_shot(tool_name, shot, role, scene_type)]
        tools = tools[: self.max_shot_tools]
        focus = self._dedup_tools(focus)[:2] or tools[:2]

        return {
            "tool_actions": [{"tool_name": t, "reason": f"fallback shot plan role={role or 'generic'}"} for t in tools],
            "score_focus": focus,
            "plan_reason": f"fallback shot plan for role={role or 'generic'}",
        }

    def _validate_plan(
        self,
        raw: Dict[str, Any],
        target_level: str,
        chapter_id: str,
        shot_id: str,
        max_tools: int | None = None,
        allow_score_focus: bool = True,
    ) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raw = {}

        clean_actions: List[Dict[str, Any]] = []
        for action in list(raw.get("tool_actions", []) or []):
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "").strip()
            if tool_name not in self.allowed_tools:
                continue
            clean_actions.append(
                {
                    "tool_name": tool_name,
                    "target_level": target_level,
                    "shot_id": shot_id if target_level == "shot" else "",
                    "chapter_id": chapter_id,
                    "reason": str(action.get("reason", "") or str(raw.get("plan_reason", "") or "")),
                }
            )

        seen_tools = set()
        dedup_actions: List[Dict[str, Any]] = []
        for action in clean_actions:
            tool_name = action["tool_name"]
            if tool_name in seen_tools:
                continue
            seen_tools.add(tool_name)
            dedup_actions.append(action)
            if isinstance(max_tools, int) and max_tools > 0 and len(dedup_actions) >= max_tools:
                break

        if not dedup_actions:
            raise ValueError(
                f"Validated plan has no allowed tools for target_level={target_level} "
                f"chapter_id={chapter_id} shot_id={shot_id}"
            )

        valid_tools = [x["tool_name"] for x in dedup_actions]
        score_focus: List[str] = []
        if allow_score_focus:
            for item in list(raw.get("score_focus", []) or []):
                tool_name = str(item or "").strip()
                if tool_name in valid_tools and tool_name not in score_focus:
                    score_focus.append(tool_name)
            if not score_focus:
                score_focus = valid_tools[: min(2, len(valid_tools))]

        out: Dict[str, Any] = {
            "tool_actions": dedup_actions,
            "score_focus": score_focus,
            "plan_reason": str(raw.get("plan_reason", "") or ""),
        }
        if target_level == "chapter":
            out["chapter_id"] = chapter_id
        else:
            out["shot_id"] = shot_id
        return out

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
        text = text.strip()
        m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return text[s : e + 1]
        return None

    def _dedup_tools(self, tools: List[str]) -> List[str]:
        out: List[str] = []
        for tool_name in tools:
            if tool_name not in self.allowed_tools:
                continue
            if tool_name not in out:
                out.append(tool_name)
        return out

    def _recommend_tools_for_shot(
        self,
        *,
        shot: ShotRecord,
        base_tools: List[str],
        scene_type: str,
        role: str,
    ) -> List[str]:
        tools = list(base_tools)
        for tool_name, spec in self.tool_catalog.items():
            if self._tool_priority_matches_shot(tool_name, shot, role, scene_type):
                tools.append(tool_name)
        return tools

    def _tool_priority_matches_shot(self, tool_name: str, shot: ShotRecord, role: str, scene_type: str) -> bool:
        planner_spec = dict((self.tool_catalog.get(tool_name, {}) or {}).get("planner", {}) or {})
        default_roles = {str(x or "").strip().lower() for x in list(planner_spec.get("default_roles", []) or []) if str(x or "").strip()}
        conditions = dict(planner_spec.get("default_conditions", {}) or {})
        if role in default_roles:
            return True
        text_thr = conditions.get("text_density_gte")
        if text_thr is not None and float(getattr(shot, "text_density", 0.0) or 0.0) >= float(text_thr):
            return True
        speech_thr = conditions.get("speech_density_gte")
        if speech_thr is not None and float(getattr(shot, "speech_density", 0.0) or 0.0) >= float(speech_thr):
            return True
        scene_types = {str(x or "").strip().lower() for x in list(conditions.get("scene_types", []) or []) if str(x or "").strip()}
        return scene_type in scene_types
