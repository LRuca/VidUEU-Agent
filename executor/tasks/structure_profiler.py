from __future__ import annotations

from executor.tasks.prompt_loader import load_prompt_text, configured_prompt_profile
from orchestrator.prompt_audit import audited_text
from typing import Any, Dict, List

from orchestrator.structures import ChapterRecord, ShotRecord, VideoProfile


class StructureProfiler:
    """
    Build chapter/video summaries from VLM-only shot profiles.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        profile = configured_prompt_profile(self.config)
        self.chapter_summary_prompt = load_prompt_text("chapter_summary_generation", profile=profile)
        self.video_summary_prompt = load_prompt_text("video_summary_generation", profile=profile)
        cfg = ((self.config.get("tasks", {}) or {}).get("T1_structure_summary", {}) or {})
        self.chapter_max_new_tokens = int(cfg.get("chapter_max_new_tokens", 220) or 220)
        self.video_max_new_tokens = int(cfg.get("video_max_new_tokens", 260) or 260)

    def profile_video_and_chapters(
        self,
        shots: List[ShotRecord],
        chapters: List[ChapterRecord],
        model_manager: Any,
        video_id: str,
    ) -> Dict[str, Any]:
        llm = self._get_llm(model_manager)
        enriched: List[ChapterRecord] = []
        for chapter in chapters:
            evidence = self._build_chapter_evidence(chapter, shots)
            enriched.append(self._profile_single_chapter(chapter, evidence, llm))
        video_profile = self._profile_video(enriched, shots, video_id, llm)
        return {"video_profile": video_profile, "chapters": enriched}

    def _build_chapter_evidence(self, chapter: ChapterRecord, shots: List[ShotRecord]) -> Dict[str, Any]:
        shot_map = {x.shot_id: x for x in shots}
        members = [shot_map[x] for x in chapter.shot_ids if x in shot_map]
        return {
            "vlm_summaries": [x.vlm_summary for x in members if x.vlm_summary],
            "roles": [x.shot_role for x in members if x.shot_role],
            "scene_types": [x.scene_type for x in members if x.scene_type],
            "quality_hints": [x.visual_quality_hint for x in members if x.visual_quality_hint],
            "informativeness_hints": [x.informativeness_hint for x in members if x.informativeness_hint],
            "sampling_hints": [x.candidate_sampling_hint for x in members if x.candidate_sampling_hint],
        }

    def _profile_single_chapter(self, chapter: ChapterRecord, evidence: Dict[str, Any], llm: Any) -> ChapterRecord:
        role = self._majority(evidence.get("roles", [])) or chapter.chapter_role or "generic"
        vlm_summaries = list(evidence.get("vlm_summaries", []) or [])
        fallback_summary = self._fallback_chapter_summary(vlm_summaries)
        if len(vlm_summaries) == 1:
            summary = fallback_summary
            topic_text = summary
        else:
            summary = self._summarize_chapter_with_llm(chapter, evidence, llm=llm) or fallback_summary
            topic_text = summary or fallback_summary
        chapter.chapter_role = role
        chapter.chapter_summary = summary
        chapter.chapter_title_hint = summary.split(".")[0][:80] if summary else role.replace("_", " ")
        chapter.key_topics = self._top_terms(topic_text, topk=6)
        chapter.key_entities = []
        chapter.importance_hint = self._chapter_importance_from_hints(evidence)
        chapter.coverage_need = "required" if role not in {"transition_bridge"} else "low"
        chapter.t2_candidate_tool_hints = self._infer_tool_hints(role, evidence)
        return chapter

    def _profile_video(
        self,
        chapters: List[ChapterRecord],
        shots: List[ShotRecord],
        video_id: str,
        llm: Any,
    ) -> VideoProfile:
        chapter_summaries = [str(x.chapter_summary or "").strip() for x in chapters if str(x.chapter_summary or "").strip()]
        if len(chapter_summaries) == 1:
            joined = chapter_summaries[0]
        else:
            joined = self._summarize_video_with_llm(chapters, llm=llm) or self._fallback_video_summary(chapter_summaries)
        structure_type = self._infer_structure_type(chapters)
        objective = "balanced_chapter_coverage"
        if structure_type in {"tutorial", "screen_demo"}:
            objective = "clarity_and_step_coverage"
        elif structure_type in {"action"}:
            objective = "phase_and_change_coverage"
        sampling_hint = self._majority([x.candidate_sampling_hint for x in shots if x.candidate_sampling_hint]) or "medium"
        return VideoProfile(
            video_id=str(video_id or "unknown"),
            video_type=structure_type,
            video_summary=joined[:600],
            global_topics=self._top_terms(joined, topk=10),
            structure_type=structure_type,
            selection_objective=objective,
            global_sampling_hint=sampling_hint,
            global_tool_hints=self._infer_global_tool_hints(chapters),
            global_quota_hints={"min_per_chapter": 1},
            metadata={"chapter_count": len(chapters), "shot_count": len(shots)},
        )

    def _get_llm(self, model_manager: Any) -> Any:
        if model_manager is None:
            return None
        try:
            return model_manager.get_resource("llm")
        except Exception:
            return None

    def _fallback_chapter_summary(self, vlm_summaries: List[str]) -> str:
        topic_text = " | ".join((vlm_summaries or [])[:8])
        return topic_text[:320].strip()

    def _fallback_video_summary(self, chapter_summaries: List[str]) -> str:
        return " ".join(chapter_summaries or [])[:600].strip()

    def _summarize_chapter_with_llm(self, chapter: ChapterRecord, evidence: Dict[str, Any], llm: Any) -> str:
        if llm is None:
            return ""
        shot_summaries = [str(x or "").strip() for x in list(evidence.get("vlm_summaries", []) or []) if str(x or "").strip()]
        if len(shot_summaries) <= 1:
            return shot_summaries[0] if shot_summaries else ""
        prompt = (
            self.chapter_summary_prompt.strip()
            + "\n\n"
            + "Input:\n"
            + f"- chapter_id: {str(chapter.chapter_id)}\n"
            + f"- chapter_role_hint: {str(getattr(chapter, 'chapter_role', '') or 'generic')}\n"
            + f"- num_shots: {len(shot_summaries)}\n"
            + "- shot_summaries:\n"
            + "\n".join(f"  - {x}" for x in shot_summaries)
            + "\n\nReturn the chapter summary only."
        )
        return self._call_summary_llm(llm, prompt, max_new_tokens=self.chapter_max_new_tokens)

    def _summarize_video_with_llm(self, chapters: List[ChapterRecord], llm: Any) -> str:
        if llm is None:
            return ""
        chapter_summaries = [str(getattr(x, "chapter_summary", "") or "").strip() for x in chapters if str(getattr(x, "chapter_summary", "") or "").strip()]
        if len(chapter_summaries) <= 1:
            return chapter_summaries[0] if chapter_summaries else ""
        prompt = (
            self.video_summary_prompt.strip()
            + "\n\n"
            + "Input:\n"
            + f"- num_chapters: {len(chapter_summaries)}\n"
            + "- chapter_summaries:\n"
            + "\n".join(f"  - {x}" for x in chapter_summaries)
            + "\n\nReturn the video summary only."
        )
        return self._call_summary_llm(llm, prompt, max_new_tokens=self.video_max_new_tokens)

    def _call_summary_llm(self, llm: Any, prompt: str, max_new_tokens: int) -> str:
        try:
            raw = audited_text(llm, prompt, max_new_tokens=max_new_tokens)
        except Exception:
            return ""
        return self._clean_summary_text(raw)

    def _clean_summary_text(self, raw: Any) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        if text.startswith("```"):
            lines = [x for x in text.splitlines() if not x.strip().startswith("```")]
            text = "\n".join(lines).strip()
        for prefix in ("chapter summary:", "video summary:", "summary:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break
        return " ".join(text.split())[:600].strip()

    def _majority(self, xs: List[str]) -> str:
        counts: Dict[str, int] = {}
        for x in xs:
            counts[x] = counts.get(x, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""

    def _top_terms(self, text: str, topk: int = 6) -> List[str]:
        stop = {"the", "and", "with", "from", "this", "that", "shot", "chapter", "scene", "video"}
        counts: Dict[str, int] = {}
        for tok in (text or "").lower().replace("|", " ").split():
            tok = tok.strip(".,:;!?()[]{}\"'")
            if len(tok) <= 2 or tok in stop:
                continue
            counts[tok] = counts.get(tok, 0) + 1
        return [x for x, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:topk]]

    def _chapter_importance_from_hints(self, evidence: Dict[str, Any]) -> str:
        vals = list(evidence.get("informativeness_hints", []) or [])
        if not vals:
            return "medium"
        if vals.count("high") >= max(1, len(vals) // 2):
            return "high"
        if vals.count("low") >= max(1, len(vals) // 2):
            return "low"
        return "medium"

    def _infer_tool_hints(self, role: str, evidence: Dict[str, Any]) -> List[str]:
        hints = set()
        if role in {"screen_demo", "slide", "title_card"}:
            hints.add("layout")
            hints.add("representative")
        if role in {"action", "broll"}:
            hints.add("motion")
        if any(x == "high" for x in (evidence.get("sampling_hints", []) or [])):
            hints.add("coverage")
        return sorted(hints)

    def _infer_structure_type(self, chapters: List[ChapterRecord]) -> str:
        roles = [x.chapter_role for x in chapters if x.chapter_role]
        if not roles:
            return "generic"
        if any(r in {"screen_demo", "slide"} for r in roles):
            return "tutorial"
        if any(r in {"action"} for r in roles):
            return "action"
        if any(r in {"talking_head", "broll"} for r in roles):
            return "documentary"
        return "generic"

    def _infer_global_tool_hints(self, chapters: List[ChapterRecord]) -> List[str]:
        out = set()
        for chapter in chapters:
            out.update(chapter.t2_candidate_tool_hints)
        return sorted(out)
