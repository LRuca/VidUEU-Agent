from __future__ import annotations

from typing import Any, Dict, List

from executor.tools.base import BaseToolAdapter, ToolContext


class OCRToolAdapter(BaseToolAdapter):
    tool_name = "ocr"

    def run_shot(self, ctx: ToolContext) -> Dict[str, Any]:
        frame_paths = [str(x.get("img", "") or "") for x in ctx.candidates if x.get("img")]
        raw = ctx.model_manager.run_with_resource(
            "ocr",
            lambda ocr: ocr.run(frame_paths) or {},
        )
        frame_hints: List[Dict[str, Any]] = []
        for candidate in ctx.candidates:
            img = str(candidate.get("img", "") or "")
            img_id = img.split("\\")[-1].split("/")[-1]
            lines = raw.get(img)
            if lines is None:
                lines = raw.get(img_id, [])
            texts = self._clean_ocr_lines(lines)
            top_lines = texts[:4]
            merged = " | ".join(top_lines).strip()
            frame_hints.append(
                {
                    "frame_key": self.frame_key(str(candidate.get("shot_id", "")), candidate.get("frame_index", -1)),
                    "img": img,
                    "text": f"Visible text present: {top_lines[0]}" if top_lines else "",
                    "score": min(1.0, 0.25 * len(top_lines)),
                    "attributes": {
                        "text_lines": top_lines,
                        "line_count": len(texts),
                        "has_text": bool(top_lines),
                    },
                    "evidence": {
                        "ocr_lines": list(lines or []),
                        "merged_preview": merged,
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

    def _clean_ocr_lines(self, lines: Any) -> List[str]:
        texts: List[str] = []
        seen = set()
        for item in (lines or []):
            txt = item.get("text", "") if isinstance(item, dict) else item
            txt = " ".join(str(txt or "").strip().split())
            key = txt.lower().replace(" ", "")
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            texts.append(txt)
        return texts
