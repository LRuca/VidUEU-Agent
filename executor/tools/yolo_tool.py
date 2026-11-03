from __future__ import annotations

from typing import Any, Dict, List

from executor.tools.base import BaseToolAdapter, ToolContext


class YoloToolAdapter(BaseToolAdapter):
    tool_name = "yolo"

    def run_shot(self, ctx: ToolContext) -> Dict[str, Any]:
        frame_paths = [str(x.get("img", "") or "") for x in ctx.candidates if x.get("img")]
        raw = ctx.model_manager.run_with_resource(
            "yolo",
            lambda yolo: yolo.run(frame_paths) or {},
        )
        frame_hints: List[Dict[str, Any]] = []
        for candidate in ctx.candidates:
            img = str(candidate.get("img", "") or "")
            dets = raw.get(img, []) if isinstance(raw, dict) else []
            labels: List[str] = []
            top_dets = sorted(dets, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            for det in top_dets:
                label = str(det.get("label", det.get("cls", "object")) or "").strip()
                if label and label not in labels:
                    labels.append(label)
                if len(labels) >= 5:
                    break
            top_detections = top_dets[:5]
            text = f"Visible entities: {', '.join(labels)}" if labels else ""
            frame_hints.append(
                {
                    "frame_key": self.frame_key(str(candidate.get("shot_id", "")), candidate.get("frame_index", -1)),
                    "img": img,
                    "text": text,
                    "score": min(1.0, 0.2 * len(labels)),
                    "attributes": {
                        "labels": labels,
                        "label_count": len(labels),
                        "top_scores": [float(x.get("score", 0.0) or 0.0) for x in top_detections],
                    },
                    "evidence": {"detections": top_detections},
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
