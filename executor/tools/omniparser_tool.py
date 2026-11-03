from __future__ import annotations

from typing import Any, Dict, List

from executor.tools.base import BaseToolAdapter, ToolContext


class OmniParserToolAdapter(BaseToolAdapter):
    tool_name = "omniparser"

    def run_shot(self, ctx: ToolContext) -> Dict[str, Any]:
        frame_paths = [str(x.get("img", "") or "") for x in ctx.candidates if x.get("img")]
        raw = ctx.model_manager.run_with_resource(
            "omniparser",
            lambda omniparser: omniparser.run(frame_paths) or {},
        )
        frame_hints: List[Dict[str, Any]] = []
        for candidate in ctx.candidates:
            img = str(candidate.get("img", "") or "")
            payload = dict(raw.get(img, {}) or {})
            elements = list(payload.get("elements", []) or [])
            text_items = self._extract_text_items(elements)
            interactive_items = self._extract_interactive_items(elements)
            captions = self._extract_captions(elements)
            summary = str(payload.get("summary", "") or "").strip()
            semantic_text = self._build_semantic_text(
                summary=summary,
                interactive_items=interactive_items,
                text_items=text_items,
                captions=captions,
            )
            score = self._score_frame(
                element_count=len(elements),
                interactive_count=len(interactive_items),
                text_count=len(text_items),
            )
            frame_hints.append(
                {
                    "frame_key": self.frame_key(str(candidate.get("shot_id", "")), candidate.get("frame_index", -1)),
                    "img": img,
                    "text": semantic_text,
                    "score": score,
                    "attributes": {
                        "element_count": len(elements),
                        "interactive_count": len(interactive_items),
                        "text_count": len(text_items),
                        "captions": captions[:8],
                        "text_lines": text_items[:8],
                    },
                    "evidence": {
                        "elements": elements,
                        "label_coordinates": dict(payload.get("label_coordinates", {}) or {}),
                        "parsed_content_list": list(payload.get("parsed_content_list", []) or []),
                    },
                }
            )
        return {
            "raw": raw,
            "semantic": {
                "summary": " | ".join(x["text"] for x in frame_hints if x.get("text")),
                "frame_hints": frame_hints,
            },
            "diagnostics": {},
        }

    def _extract_text_items(self, elements: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for element in elements:
            content = str(element.get("content", "") or "").strip()
            if not content:
                continue
            if str(element.get("type", "") or "").strip().lower() == "text" and content not in out:
                out.append(content)
        return out

    def _extract_interactive_items(self, elements: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for element in elements:
            if element.get("interactable") is not True:
                continue
            content = str(element.get("content", "") or "").strip()
            if content and content not in out:
                out.append(content)
        return out

    def _extract_captions(self, elements: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for element in elements:
            content = str(element.get("content", "") or "").strip()
            if not content or content in out:
                continue
            out.append(content)
        return out

    def _build_semantic_text(
        self,
        *,
        summary: str,
        interactive_items: List[str],
        text_items: List[str],
        captions: List[str],
    ) -> str:
        if interactive_items:
            return f"Interactive UI elements visible: {', '.join(interactive_items[:4])}"
        if text_items:
            return f"UI text and elements visible: {' | '.join(text_items[:4])}"
        if summary:
            return summary
        if captions:
            return f"UI elements visible: {', '.join(captions[:4])}"
        return ""

    def _score_frame(self, *, element_count: int, interactive_count: int, text_count: int) -> float:
        base = min(1.0, 0.05 * float(element_count))
        base += min(0.4, 0.1 * float(interactive_count))
        base += min(0.3, 0.06 * float(text_count))
        return min(1.0, round(base, 4))
