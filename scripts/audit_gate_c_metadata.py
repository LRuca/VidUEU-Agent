#!/usr/bin/env python3
"""Post-hoc Gate C audit over T1/T2/T3 segment logs.

The script reconstructs the minimal metadata package used by the paper's Gate C
from stage logs, samples a randomized audit set, and sends metadata with optional
reference keyframes to an OpenAI-compatible vLLM endpoint.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import orjson
except ImportError:  # Optional acceleration for large stage logs.
    orjson = None

try:
    from PIL import Image
except ImportError:
    Image = None


DOMAINS = (
    "Technology&Tutorial",
    "Sports&Action",
    "DailyLife&HumanActivities",
    "Nature&Scenery",
    "Gaming&ACG",
)

ADMISSION_PROMPT = """You are a strict metadata-level admission judge for a video-to-dataset generation pipeline.
Decide whether the provided metadata package is suitable for downstream dataset instance generation. Judge the metadata as a whole, using video-level summaries, chapter summaries, shot summaries, frame captions, semanticized tool signals, OCR, ASR, object, motion, UI, transition, and pairwise-change evidence when available.
Do NOT judge raw image quality directly unless quality problems are explicitly reflected in the metadata. Judge whether the metadata contains enough grounded, coherent, specific, and useful information to support reliable dataset generation.

ADMISSION GOAL:
Accept metadata only if it can support meaningful grounded dataset items, such as visual QA, multi-choice QA, reasoning QA, dialogue, temporal/change reasoning, or edit instructions. Reject metadata that is too weak, generic, noisy, contradictory, redundant, or poorly grounded.

DECISION LABELS:
- accept: suitable for downstream dataset generation.
- reject: unsuitable for downstream dataset generation.

JUDGMENT CRITERIA:
1. Grounded visual content: keep metadata with concrete visible entities, actions, states, layouts, text, UI elements, scene structure, or temporal changes; reject generic, empty, vague, or repetitive metadata.
2. Cross-level coherence: keep metadata whose video-, chapter-, shot-, and frame-level descriptions are broadly consistent; reject conflicting evidence that weakens grounding.
3. Dataset usefulness: keep metadata that supports at least one useful task type; reject metadata that only yields trivial, obvious, or hallucination-prone items.
4. Evidence density: keep metadata with enough distinct evidence points across frames or tools; reject sparse, duplicated, or semantically flat evidence.
5. Tool-signal reliability: treat tool signals as auxiliary evidence; keep them when they reinforce captions or summaries; reject incoherent, failed, irrelevant, or contradictory tool evidence.
6. Temporal and structural value: keep meaningful progression, state changes, interactions, procedures, spatial layouts, readable text, UI structure, or object relations; reject filler, static background, near-duplicates, or low-information motion.
7. Hallucination safety: reject metadata that would require inventing unseen details, intentions, hidden causes, identities, exact text, or precise attributes.

OUTPUT FORMAT (strict JSON only, no extra text):
{
  "domain": "Technology&Tutorial" | "Sports&Action" | "DailyLife&HumanActivities" | "Nature&Scenery" | "Gaming&ACG",
  "decision": "accept" | "reject",
  "primary_reasons": ["reason 1", "reason 2"]
}

FIELD RULES:
- domain must be exactly one label. Classify the primary visible content or activity: Technology&Tutorial includes demonstrations, how-to procedures, repairs, tools, products, and technical workflows; Sports&Action includes organized sports, exercise, racing, and physical performance; DailyLife&HumanActivities includes ordinary people, home, social, food, fashion, and routine activities; Nature&Scenery includes landscapes, wildlife, plants, weather, and natural environments; Gaming&ACG includes gameplay, game UI, animation, anime, cartoons, and virtual characters.
- primary_reasons must contain 2-4 concise reasons for the decision.
"""


def _clean_text(value: Any, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _safe_float(value: Any) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _stage_output(row: Dict[str, Any], stage: str) -> Optional[Dict[str, Any]]:
    payload = row.get(stage)
    if not isinstance(payload, dict):
        return None
    output = payload.get("output")
    return output if isinstance(output, dict) else None


def _segment_key(row: Dict[str, Any]) -> str:
    return str(row.get("segment_path") or "").strip()


def _parse_transition(text: Any) -> Dict[str, Any]:
    raw = _clean_text(text, 700)
    if not raw:
        return {
            "atomic_changes": [],
            "same_shot": None,
            "viewpoint_change": "unknown",
            "subject_consistent": "unknown",
            "reliability": "unknown",
        }

    parts = [part.strip() for part in raw.split("|")]
    edits = parts[0]
    shot = next((part.split(":", 1)[1].strip().lower() for part in parts if part.lower().startswith("shot:")), "unknown")
    view = next((part.split(":", 1)[1].strip().lower() for part in parts if part.lower().startswith("view:")), "unknown")
    subject = next((part.split(":", 1)[1].strip().lower() for part in parts if part.lower().startswith("subject:")), "unknown")
    uncertainty = next((part.split(":", 1)[1].strip().lower() for part in parts if part.lower().startswith("uncertainty:")), "unknown")
    atomic_changes = re.sub(r"^edits?\s*\([^)]*\)\s*:\s*", "", edits, flags=re.IGNORECASE)
    atomic_changes = [item.strip() for item in re.split(r"\s*;\s*|\s*\d+\)\s*", atomic_changes) if item.strip() and item.strip().lower() != "none"]
    return {
        "atomic_changes": atomic_changes[:4],
        "same_shot": not any(token in shot for token in ("cut", "scene_cut", "different")),
        "viewpoint_change": view,
        "subject_consistent": subject,
        "reliability": uncertainty,
    }


def _tool_semantics(frame: Dict[str, Any]) -> Dict[str, Any]:
    detail = frame.get("score_detail") if isinstance(frame.get("score_detail"), dict) else {}
    profile = frame.get("quality_profile") if isinstance(frame.get("quality_profile"), dict) else {}
    return {
        "focused_tools": list(detail.get("focus_used") or [])[:6],
        "semantic_text": _clean_text(detail.get("semantic_text"), 480),
        "quality_profile_pass": profile.get("final_pass"),
        "quality_profile_reason": _clean_text(profile.get("final_reason") or profile.get("reason"), 160),
    }


def _extract_metadata(segment_path: str, stages: Dict[str, Dict[str, Any]], source_log: Path) -> Optional[Dict[str, Any]]:
    t1 = stages.get("T1")
    t2 = stages.get("T2")
    t3 = stages.get("T3")
    if not (isinstance(t1, dict) and isinstance(t2, dict) and isinstance(t3, dict)):
        return None

    frame_scores = t2.get("keyframes") if isinstance(t2.get("keyframes"), list) else []
    score_by_origin: Dict[Any, Dict[str, Any]] = {}
    score_by_second: Dict[float, Dict[str, Any]] = {}
    for frame in frame_scores:
        if not isinstance(frame, dict):
            continue
        if frame.get("origin_index") is not None:
            score_by_origin[frame.get("origin_index")] = frame
        second = _safe_float(frame.get("sec"))
        if second is not None:
            score_by_second[second] = frame

    hierarchy = t3.get("summary_hierarchy") if isinstance(t3.get("summary_hierarchy"), dict) else {}
    video_profile = t1.get("video_profile") if isinstance(t1.get("video_profile"), dict) else {}
    shots = t1.get("shot_profiles") if isinstance(t1.get("shot_profiles"), list) else t1.get("shots")
    chapters = t1.get("chapters") if isinstance(t1.get("chapters"), list) else []
    frames = t3.get("frames") if isinstance(t3.get("frames"), list) else []
    if not frames:
        return None

    normalized_chapters = []
    for chapter in chapters[:16]:
        if not isinstance(chapter, dict):
            continue
        normalized_chapters.append(
            {
                "chapter_id": chapter.get("chapter_id"),
                "time_span": [
                    _safe_float(chapter.get("start_sec")),
                    _safe_float(chapter.get("end_sec")),
                ],
                "summary": _clean_text(chapter.get("chapter_summary") or chapter.get("summary"), 520),
                "role": _clean_text(chapter.get("chapter_role") or chapter.get("role"), 100),
            }
        )

    normalized_shots = []
    for shot in (shots or [])[:32]:
        if not isinstance(shot, dict):
            continue
        normalized_shots.append(
            {
                "shot_id": shot.get("shot_id"),
                "chapter_id": (shot.get("metadata") or {}).get("chapter_id") or shot.get("chapter_id"),
                "time_span": [
                    _safe_float(shot.get("start_sec")),
                    _safe_float(shot.get("end_sec")),
                ],
                "summary": _clean_text(shot.get("vlm_summary") or shot.get("summary"), 420),
                "role": _clean_text(shot.get("shot_role") or shot.get("role"), 100),
                "scene_type": _clean_text(shot.get("scene_type"), 100),
                "quality_hint": _clean_text(shot.get("visual_quality_hint"), 100),
                "info_hint": _clean_text(shot.get("informativeness_hint") or shot.get("importance_hint"), 100),
            }
        )

    normalized_frames = []
    normalized_transitions = []
    image_refs = []
    for index, frame in enumerate(frames[:16]):
        if not isinstance(frame, dict):
            continue
        origin = frame.get("origin_index")
        score = score_by_origin.get(origin)
        if score is None:
            score = score_by_second.get(_safe_float(frame.get("sec")))
        score = score or {}
        quality = score.get("quality_profile") if isinstance(score.get("quality_profile"), dict) else {}
        frame_id = f"F{index:02d}"
        image_path = _clean_text(frame.get("img") or score.get("img"), 4096)
        if image_path:
            image_refs.append({"frame_id": frame_id, "path": image_path})
        normalized_frames.append(
            {
                "frame_id": frame_id,
                "shot_id": frame.get("shot_id"),
                "timestamp": _safe_float(frame.get("sec")),
                "caption": _clean_text(frame.get("caption"), 440),
                "quality": {
                    "passed": quality.get("final_pass"),
                    "fast_score": _safe_float(quality.get("fast_quality_score")),
                    "reason": _clean_text(quality.get("final_reason") or quality.get("reason"), 160),
                },
                "value_score": _safe_float((score.get("score_detail") or {}).get("value_score") or score.get("score")),
                "tool_semantics": _tool_semantics(score),
                "evidence_id": f"F{index:02d}_CAP",
                "reference_image_attached": bool(image_path),
            }
        )
        if index < len(frames) - 1:
            transition = _parse_transition(frame.get("transition_to_next"))
            transition.update({"source": frame_id, "target": f"F{index + 1:02d}"})
            normalized_transitions.append(transition)

    first_start = min((value["time_span"][0] for value in normalized_shots if value["time_span"][0] is not None), default=0.0)
    last_end = max((value["time_span"][1] for value in normalized_shots if value["time_span"][1] is not None), default=None)
    video_id = _clean_text(video_profile.get("video_id"), 200)
    if not video_id:
        video_id = Path(segment_path).stem.rsplit("_", 1)[0]
    summary = _clean_text(t3.get("segment_summary") or hierarchy.get("segment_summary") or t1.get("global_summary"), 900)
    if not summary:
        return None

    return {
        "segment": {
            "video_id": video_id,
            "start": first_start,
            "end": last_end,
            "summary": summary,
            "type": _clean_text(t1.get("video_type") or video_profile.get("video_type"), 100),
        },
        "chapters": normalized_chapters,
        "shots": normalized_shots,
        "keyframes": normalized_frames,
        "transitions": normalized_transitions,
        "_audit": {
            "segment_path": segment_path,
            "source_log": str(source_log),
            "source_dataset": source_log.parent.parent.name,
            "metadata_id": hashlib.sha256(segment_path.encode("utf-8")).hexdigest()[:16],
            "images": image_refs,
        },
    }


def _iter_metadata_file(log_path: Path) -> Iterator[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    try:
        stream = log_path.open("r", encoding="utf-8")
    except OSError:
        return
    with stream:
        for raw in stream:
            try:
                row = orjson.loads(raw) if orjson is not None else json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            stage = str(row.get("stage") or "")
            if stage not in {"T1", "T2", "T3"}:
                continue
            segment_path = _segment_key(row)
            output = _stage_output(row, stage)
            if not segment_path or output is None:
                continue
            grouped.setdefault(segment_path, {})[stage] = output
    for segment_path, stages in grouped.items():
        metadata = _extract_metadata(segment_path, stages, log_path)
        if metadata is not None:
            yield metadata


def _iter_metadata(log_roots: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    for root in log_roots:
        for log_path in root.rglob("*_segments.jsonl"):
            yield from _iter_metadata_file(log_path)


def _select_random(log_roots: List[Path], sample_size: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Logs are segment-oriented, so seeded randomized file traversal gives a
    # reproducible sample without parsing the full multi-hundred-GB log tree.
    log_paths: List[Path] = []
    source_log_counts: Counter = Counter()
    for root in log_roots:
        paths = list(root.rglob("*_segments.jsonl"))
        log_paths.extend(paths)
        source_log_counts[root.parent.name] += len(paths)
    rng = random.Random(seed)
    rng.shuffle(log_paths)
    selected: List[Dict[str, Any]] = []
    seen = set()
    selected_source_counts: Counter = Counter()
    scanned = 0
    for log_path in log_paths:
        scanned += 1
        for metadata in _iter_metadata_file(log_path):
            metadata_id = metadata["_audit"]["metadata_id"]
            if metadata_id in seen:
                continue
            seen.add(metadata_id)
            selected_source_counts[metadata["_audit"]["source_dataset"]] += 1
            selected.append(metadata)
            if len(selected) >= sample_size:
                break
        if len(selected) >= sample_size:
            break

    if len(selected) < sample_size:
        raise RuntimeError(
            "Could not find enough completed T1/T2/T3 metadata packages. "
            f"Needed {sample_size}; found {len(selected)}."
        )
    inventory = {
        "candidate_log_files": len(log_paths),
        "source_candidate_log_files": dict(sorted(source_log_counts.items())),
        "scanned_log_files": scanned,
        "selected_unique_metadata": len(selected),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        "population_metadata_count_exact": False,
        "sampling_method": "seeded_shuffled_segment_log_traversal",
        "sample_size": sample_size,
        "seed": seed,
    }
    return selected, inventory


def _reuse_sample(source_sample: Path, sample_size: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_rows = [json.loads(line) for line in source_sample.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(source_rows) != sample_size:
        raise RuntimeError(f"Reuse sample has {len(source_rows)} rows; expected {sample_size}: {source_sample}")

    requested_order = [str(row["_audit"]["metadata_id"]) for row in source_rows]
    requested_by_log: Dict[Path, set[str]] = {}
    for row in source_rows:
        audit = row.get("_audit") or {}
        requested_by_log.setdefault(Path(str(audit["source_log"])), set()).add(str(audit["metadata_id"]))

    recovered: Dict[str, Dict[str, Any]] = {}
    for index, (log_path, requested_ids) in enumerate(requested_by_log.items(), 1):
        for metadata in _iter_metadata_file(log_path):
            metadata_id = str(metadata["_audit"]["metadata_id"])
            if metadata_id in requested_ids:
                recovered[metadata_id] = metadata
        if index % 250 == 0:
            print(f"rehydrated_logs={index}/{len(requested_by_log)}, metadata={len(recovered)}/{sample_size}", flush=True)

    missing = [metadata_id for metadata_id in requested_order if metadata_id not in recovered]
    if missing:
        raise RuntimeError(f"Could not rehydrate {len(missing)} requested metadata IDs; first={missing[:5]}")

    sample = [recovered[metadata_id] for metadata_id in requested_order]
    image_counts = [len((item.get("_audit") or {}).get("images") or []) for item in sample]
    source_counts = Counter(item["_audit"]["source_dataset"] for item in sample)
    inventory = {
        "reused_sample_from": str(source_sample),
        "matched_metadata_ids": True,
        "selected_unique_metadata": len(sample),
        "selected_source_counts": dict(sorted(source_counts.items())),
        "samples_with_images": sum(count > 0 for count in image_counts),
        "samples_without_images": sum(count == 0 for count in image_counts),
        "total_image_refs": sum(image_counts),
        "max_image_refs_per_sample": max(image_counts, default=0),
        "sample_size": sample_size,
    }
    return sample, inventory


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def _normalize_domain(value: Any) -> Optional[str]:
    compact = re.sub(r"[^a-z]", "", str(value or "").lower())
    aliases = {
        "technologytutorial": "Technology&Tutorial",
        "sportsaction": "Sports&Action",
        "dailylifehumanactivities": "DailyLife&HumanActivities",
        "naturescenery": "Nature&Scenery",
        "gamingacg": "Gaming&ACG",
    }
    return aliases.get(compact)


def _parse_response(content: str) -> Dict[str, Any]:
    candidates = [content.strip()]
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        domain = _normalize_domain(parsed.get("domain"))
        decision = str(parsed.get("decision") or "").strip().lower()
        reasons = parsed.get("primary_reasons")
        if domain is not None and decision in {"accept", "reject"} and isinstance(reasons, list):
            cleaned = [_clean_text(reason, 220) for reason in reasons if _clean_text(reason, 220)]
            return {"domain": domain, "decision": decision, "primary_reasons": cleaned[:4]}
    raise ValueError("response is not a valid Gate C JSON decision")


def _image_to_data_uri(path: Path, max_side: int, jpeg_quality: int) -> str:
    if Image is None:
        raise RuntimeError("Pillow is required for multimodal Gate C auditing")
    with Image.open(path) as source:
        image = source.convert("RGB")
        if max_side > 0 and max(image.size) > max_side:
            scale = float(max_side) / float(max(image.size))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _judge_one(metadata: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    package = {key: value for key, value in metadata.items() if key != "_audit"}
    package_text = json.dumps(package, ensure_ascii=False, separators=(",", ":"))
    if len(package_text) > args.metadata_char_limit:
        package_text = package_text[: args.metadata_char_limit]
    user_content = ADMISSION_PROMPT
    images_requested = 0
    images_attached = 0
    images_missing = 0
    message_content: Any
    if args.multimodal:
        image_refs = list((metadata.get("_audit") or {}).get("images") or [])[: args.max_images]
        images_requested = len(image_refs)
        image_parts = []
        attachment_map = []
        for ref in image_refs:
            path = Path(str(ref.get("path") or ""))
            try:
                data_uri = _image_to_data_uri(path, args.image_max_side, args.jpeg_quality)
            except (OSError, ValueError, RuntimeError):
                images_missing += 1
                continue
            image_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            attachment_map.append(f"image_{images_attached + 1}={ref.get('frame_id')}")
            images_attached += 1
        if not image_parts:
            return {
                "decision": "error",
                "primary_reasons": ["no readable reference keyframes for multimodal audit"],
                "raw_response": "",
                "attempts": 0,
                "images_requested": images_requested,
                "images_attached": 0,
                "images_missing": images_missing,
            }
        user_content += (
            "\nREFERENCE KEYFRAMES:\n"
            "The attached images correspond one-to-one to keyframe records as follows: "
            + ", ".join(attachment_map)
            + ". Use them to verify grounding, coherence, and hallucination safety. "
            "Do not reject solely for raw image quality unless that problem is represented in the metadata."
        )
        user_content += "\nMETADATA PACKAGE:\n" + package_text
        message_content = image_parts + [{"type": "text", "text": user_content}]
    else:
        user_content += "\nMETADATA PACKAGE:\n" + package_text
        message_content = user_content
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": message_content}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint_index = int(metadata["_audit"]["metadata_id"][:8], 16) % len(args.base_url)
    base_url = args.base_url[endpoint_index]
    last_error = ""
    for attempt in range(1, args.retries + 1):
        request = urllib.request.Request(
            _chat_url(base_url),
            data=encoded,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + args.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = ((body.get("choices") or [{}])[0].get("message") or {})
            content = str(message.get("content") or "")
            judged = _parse_response(content)
            judged.update(
                {
                    "raw_response": content,
                    "attempts": attempt,
                    "images_requested": images_requested,
                    "images_attached": images_attached,
                    "images_missing": images_missing,
                }
            )
            return judged
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
    return {
        "decision": "error",
        "primary_reasons": [last_error],
        "raw_response": "",
        "attempts": args.retries,
        "images_requested": images_requested,
        "images_attached": images_attached,
        "images_missing": images_missing,
    }


def _load_existing(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            metadata_id = ((row.get("_audit") or {}).get("metadata_id"))
            decision = str((row.get("gate_c") or {}).get("decision") or "")
            if metadata_id and decision in {"accept", "reject"}:
                done.add(str(metadata_id))
    return done


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _build_summary(results_path: Path, sample_size: int, args: argparse.Namespace) -> Dict[str, Any]:
    domain_counts: Dict[str, Counter] = {domain: Counter() for domain in DOMAINS}
    reasons: Dict[str, Counter] = {domain: Counter() for domain in DOMAINS}
    total = Counter()
    image_coverage = Counter()
    latest_rows: Dict[str, Dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            metadata_id = str((row.get("_audit") or {}).get("metadata_id") or "")
            if metadata_id:
                latest_rows[metadata_id] = row

    for row in latest_rows.values():
        gate_c = row.get("gate_c") or {}
        domain = gate_c.get("domain")
        decision = str(gate_c.get("decision") or "error")
        total[decision] += 1
        attached = int(gate_c.get("images_attached") or 0)
        missing = int(gate_c.get("images_missing") or 0)
        image_coverage["images_attached"] += attached
        image_coverage["images_missing"] += missing
        image_coverage["samples_with_images"] += int(attached > 0)
        image_coverage["samples_without_images"] += int(attached == 0)
        if domain not in domain_counts:
            continue
        domain_counts[domain][decision] += 1
        if decision == "reject":
            for reason in gate_c.get("primary_reasons") or []:
                reasons[domain][_clean_text(reason, 180)] += 1

    domains = {}
    for domain in DOMAINS:
        counts = domain_counts[domain]
        decided = counts["accept"] + counts["reject"]
        domains[domain] = {
            "sampled": sum(counts.values()),
            "accept": counts["accept"],
            "reject": counts["reject"],
            "error": counts["error"],
            "rejection_rate": round(counts["reject"] / decided, 4) if decided else None,
            "top_rejection_reasons": [{"reason": reason, "count": count} for reason, count in reasons[domain].most_common(8)],
        }
    decided_total = total["accept"] + total["reject"]
    return {
        "audit": {
            "sample_size_requested": sample_size,
            "seed": args.seed,
            "source_roots": [str(path) for path in args.log_root],
            "base_urls": args.base_url,
            "model": args.model,
            "domain_assignment": "jointly_inferred_by_the_Gate_C_VLM_response",
            "admission_scope": "post_hoc_multimodal_metadata_audit" if args.multimodal else "post_hoc_metadata_only_audit",
            "multimodal": args.multimodal,
            "max_images_per_sample": args.max_images if args.multimodal else 0,
            "image_max_side": args.image_max_side if args.multimodal else 0,
            "jpeg_quality": args.jpeg_quality if args.multimodal else None,
            "image_coverage": dict(image_coverage),
        },
        "overall": {
            "sampled": sum(total.values()),
            "accept": total["accept"],
            "reject": total["reject"],
            "error": total["error"],
            "rejection_rate": round(total["reject"] / decided_total, 4) if decided_total else None,
        },
        "domains": domains,
    }


def _write_markdown(summary: Dict[str, Any], path: Path) -> None:
    multimodal = bool(summary["audit"].get("multimodal"))
    scope_note = (
        "This is a post-hoc multimodal audit over completed T1/T2/T3 metadata with its reference keyframes."
        if multimodal
        else "This is a post-hoc metadata-only audit over completed T1/T2/T3 logs."
    )
    lines = [
        "# Gate C Multimodal Metadata Admission Audit" if multimodal else "# Gate C Metadata Admission Audit",
        "",
        scope_note + " Domain labels are jointly inferred by the Gate C VLM response and are not ground-truth annotations.",
        "",
        "| Domain | Sampled | Accept | Reject | Error | Rejection rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, row in summary["domains"].items():
        rate = "n/a" if row["rejection_rate"] is None else f"{row['rejection_rate']:.1%}"
        lines.append(f"| {domain} | {row['sampled']} | {row['accept']} | {row['reject']} | {row['error']} | {rate} |")
    lines.extend(["", "## Main Rejection Reasons", ""])
    for domain, row in summary["domains"].items():
        lines.append(f"### {domain}")
        reasons = row["top_rejection_reasons"]
        if not reasons:
            lines.append("- None")
        else:
            for item in reasons:
                lines.append(f"- {item['reason']} ({item['count']})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, action="append", required=True, help="A data source directory containing logs/.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--reuse-sample-from", type=Path, help="Rehydrate the exact metadata IDs from an existing metadata_sample.jsonl.")
    parser.add_argument("--multimodal", action="store_true", help="Attach reference keyframe images to each VLM request.")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--image-max-side", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--base-url", action="append", help="OpenAI-compatible vLLM base URL. Repeat to spread calls across endpoints.")
    parser.add_argument("--model", default="Qwen3.5-9B-local")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--metadata-char-limit", type=int, default=14000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.base_url:
        args.base_url = ["http://127.0.0.1:8024/v1"]
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.image_max_side <= 0:
        parser.error("--image-max-side must be positive")
    if not 30 <= args.jpeg_quality <= 95:
        parser.error("--jpeg-quality must be between 30 and 95")
    return args


def main() -> int:
    args = parse_args()
    args.log_root = [root.resolve() for root in args.log_root]
    if args.reuse_sample_from is not None:
        args.reuse_sample_from = args.reuse_sample_from.resolve()
    missing = [str(root) for root in args.log_root if not (root / "logs").is_dir()]
    if missing:
        raise SystemExit("Each --log-root must contain logs/: " + ", ".join(missing))
    log_roots = [root / "logs" for root in args.log_root]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "metadata_sample.jsonl"
    inventory_path = output_dir / "metadata_inventory.json"
    results_path = output_dir / "gate_c_decisions.jsonl"
    summary_path = output_dir / "gate_c_summary.json"
    report_path = output_dir / "gate_c_report.md"
    if sample_path.exists():
        sample = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(sample) != args.sample_size:
            raise SystemExit(f"Existing {sample_path} has {len(sample)} rows, expected {args.sample_size}. Use a new output directory.")
        print(f"Reusing frozen sample: {sample_path} ({len(sample)} rows)")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else None
    else:
        if args.reuse_sample_from is not None:
            sample, inventory = _reuse_sample(args.reuse_sample_from, args.sample_size)
        else:
            sample, inventory = _select_random(log_roots, args.sample_size, args.seed)
        with sample_path.open("x", encoding="utf-8") as stream:
            for metadata in sample:
                stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        _write_json(inventory_path, inventory)
        if args.reuse_sample_from is not None:
            print(
                f"Rehydrated matched sample: {len(sample)} rows, images={inventory['total_image_refs']}, "
                f"without_images={inventory['samples_without_images']}"
            )
        else:
            print(
                f"Candidate segment logs: {inventory['candidate_log_files']} "
                f"(sources={inventory['source_candidate_log_files']})"
            )
            print(
                f"Frozen randomized sample: {len(sample)} rows, seed={args.seed}, "
                f"scanned_logs={inventory['scanned_log_files']}"
            )

    if args.dry_run:
        print(f"Dry run complete. Metadata sample: {sample_path}")
        return 0

    completed = _load_existing(results_path)
    pending = [item for item in sample if item["_audit"]["metadata_id"] not in completed]
    print(f"Gate C requests: pending={len(pending)}, already_complete={len(sample) - len(pending)}, workers={args.workers}")
    lock = threading.Lock()
    with results_path.open("a", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_judge_one, metadata, args): metadata for metadata in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                metadata = futures[future]
                try:
                    gate_c = future.result()
                except Exception as exc:  # Defensive: keep the audit resumable.
                    gate_c = {"decision": "error", "primary_reasons": [f"worker_error: {type(exc).__name__}: {exc}"], "raw_response": "", "attempts": 0}
                row = {"_audit": metadata["_audit"], "gate_c": gate_c}
                with lock:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                if index % 25 == 0 or index == len(pending):
                    print(f"completed {index}/{len(pending)}")

    summary = _build_summary(results_path, args.sample_size, args)
    if inventory is not None:
        summary["audit"]["metadata_inventory"] = inventory
    _write_json(summary_path, summary)
    _write_markdown(summary, report_path)
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
