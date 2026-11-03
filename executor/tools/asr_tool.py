from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List

from executor.tools.base import BaseToolAdapter, ToolContext


class ASRToolAdapter(BaseToolAdapter):
    tool_name = "asr"

    def run_shot(self, ctx: ToolContext) -> Dict[str, Any]:
        shot = ctx.shot
        frame_secs = [float(x.get("sec", 0.0) or 0.0) for x in ctx.candidates]
        if shot is None or not ctx.video_path or not frame_secs:
            return self._empty_result(ctx)
        start_sec = float(getattr(shot, "start_sec", 0.0) or 0.0)
        end_sec = float(getattr(shot, "end_sec", start_sec) or start_sec)
        duration = max(0.0, end_sec - start_sec)
        if duration <= 0.05:
            return self._empty_result(ctx)

        audio_dir = os.path.join(ctx.t2_workdir, "tool_audio")
        os.makedirs(audio_dir, exist_ok=True)
        shot_name = str(getattr(shot, "shot_id", "shot") or "shot").replace("/", "_")
        audio_path = os.path.join(audio_dir, f"{shot_name}.wav")
        if not self._valid_audio_file(audio_path):
            has_audio_stream = self._probe_has_audio_stream(ctx.video_path)
            if has_audio_stream is False:
                print(
                    f"[ASRTool] skip shot={shot_name}: no audio stream in video {ctx.video_path}"
                )
                return self._empty_result(
                    ctx,
                    diagnostics={"audio_stream_present": False, "reason": "no_audio_stream"},
                )

            ok, extract_error = self._extract_audio_clip(
                video_path=ctx.video_path,
                audio_path=audio_path,
                start_sec=start_sec,
                duration=duration,
            )
            if not ok:
                if extract_error:
                    print(
                        f"[ASRTool] skip shot={shot_name}: ffmpeg audio extraction failed for "
                        f"{ctx.video_path}: {extract_error}"
                    )
                return self._empty_result(
                    ctx,
                    diagnostics={
                        "audio_stream_present": has_audio_stream,
                        "reason": "audio_extract_failed",
                        "extract_error": extract_error,
                    },
                )
        text, segments = ctx.model_manager.run_with_resource(
            "asr",
            lambda asr: asr.run(audio_path),
        )
        segments = self._clean_segments(list(segments or []))
        frame_hints: List[Dict[str, Any]] = []
        snippets: List[str] = []
        for candidate in ctx.candidates:
            sec = float(candidate.get("sec", 0.0) or 0.0)
            hits: List[Dict[str, Any]] = []
            merged_bits: List[str] = []
            for seg in segments:
                ts = seg["timestamp"]
                seg_start = float(ts[0]) + start_sec
                seg_end = float(ts[1]) + start_sec
                if seg_end >= sec - 1.5 and seg_start <= sec + 1.5:
                    hits.append(seg)
                    if seg["text"] not in merged_bits:
                        merged_bits.append(seg["text"])
            merged = " ".join(merged_bits[:3]).strip()
            if merged:
                snippets.append(merged)
            top_hits = hits[:3]
            frame_hints.append(
                {
                    "frame_key": self.frame_key(str(candidate.get("shot_id", "")), candidate.get("frame_index", -1)),
                    "img": str(candidate.get("img", "") or ""),
                    "text": f"Nearby speech: {merged[:160]}" if merged else "",
                    "score": min(1.0, 0.3 * len(hits)),
                    "attributes": {
                        "segments": top_hits,
                        "segment_count": len(hits),
                        "has_speech": bool(top_hits),
                    },
                    "evidence": {
                        "window_seconds": 1.5,
                        "full_transcript": str(text or ""),
                    },
                }
            )
        summary_bits: List[str] = []
        for snippet in snippets:
            if snippet not in summary_bits:
                summary_bits.append(snippet)
        return {
            "raw": {"text": str(text or ""), "segments": segments},
            "semantic": {
                "summary": ("Speech hints near this shot: " + " | ".join(summary_bits[:3])) if summary_bits else "",
                "frame_hints": frame_hints,
            },
            "diagnostics": {},
        }

    def _empty_result(self, ctx: ToolContext, diagnostics: Dict[str, Any] | None = None) -> Dict[str, Any]:
        frame_hints = [
            {
                "frame_key": self.frame_key(str(candidate.get("shot_id", "")), candidate.get("frame_index", -1)),
                "img": str(candidate.get("img", "") or ""),
                "text": "",
                "score": 0.0,
                "attributes": {"segments": [], "segment_count": 0, "has_speech": False},
                "evidence": {"window_seconds": 1.5, "full_transcript": ""},
            }
            for candidate in ctx.candidates
        ]
        return {
            "raw": {"text": "", "segments": []},
            "semantic": {"summary": "", "frame_hints": frame_hints},
            "diagnostics": dict(diagnostics or {}),
        }

    def _valid_audio_file(self, audio_path: str) -> bool:
        return os.path.exists(audio_path) and os.path.getsize(audio_path) > 0

    def _probe_has_audio_stream(self, video_path: str) -> bool | None:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "json",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        except Exception:
            return None
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout or "{}")
        except Exception:
            return None
        return bool((payload.get("streams") or []))

    def _extract_audio_clip(
        self,
        *,
        video_path: str,
        audio_path: str,
        start_sec: float,
        duration: float,
    ) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_sec:.3f}",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    audio_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, str(exc)
        if result.returncode == 0 and self._valid_audio_file(audio_path):
            return True, ""
        error = (result.stderr or result.stdout or "").strip()
        if self._valid_audio_file(audio_path):
            return True, ""
        return False, error[:400]

    def _clean_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        seen = set()
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            text = " ".join(str(seg.get("text", "") or "").strip().split())
            key = text.lower().replace(" ", "")
            ts = seg.get("timestamp") or seg.get("ts") or seg.get("time")
            if len(key) < 3 or key in seen or not isinstance(ts, (list, tuple)) or len(ts) < 2:
                continue
            normalized_ts = self._normalize_timestamp_pair(ts[0], ts[1])
            if normalized_ts is None:
                continue
            seen.add(key)
            cleaned.append({"text": text, "timestamp": normalized_ts})
        return cleaned

    def _normalize_timestamp_pair(self, start: Any, end: Any) -> List[float] | None:
        start_value = self._safe_float(start)
        end_value = self._safe_float(end)
        if start_value is None and end_value is None:
            return None
        if start_value is None:
            start_value = end_value
        if end_value is None:
            end_value = start_value
        if start_value is None or end_value is None:
            return None
        if end_value < start_value:
            start_value, end_value = end_value, start_value
        return [start_value, end_value]

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
