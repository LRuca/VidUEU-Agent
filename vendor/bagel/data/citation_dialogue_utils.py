# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple


VLM_CITATION_SYSTEM_PROMPT = """You should first output references in <cite></cite> that contain all minimal necessary historical information needed as context for the subsequent current-turn text generation (each citation must be numbered as (1), (2), ...; text evidence must be quoted as original sentence(s), image evidence must be written as Image X bbox:[x1,y1,x2,y2]), then output reasoning in <claim></claim> where each clause segment (before punctuation) must append citation index tag(s) in the form [(1)] or [(1),(2)], and finally output the actual text answer outside tags, i.e. <cite>(1) Turn 2, user: "..." ; (2) Turn 4, assistant: Image 1 bbox:[x1,y1,x2,y2]</cite><claim>reasoning clause A[(1)], reasoning clause B[(1),(2)].</claim>answer here"""

GEN_CITATION_SYSTEM_PROMPT = """You should first output references in <cite></cite> that contain all minimal necessary historical information needed as context for the subsequent current-turn image generation (each citation must be numbered as (1), (2), ...; text evidence must be quoted as original sentence(s), image evidence must be written as Image X bbox:[x1,y1,x2,y2]), then output planning in <claim></claim> where each clause segment (before punctuation) must append citation index tag(s) in the form [(1)] or [(1),(2)], and finally output the actual image result outside tags, i.e. <cite>(1) Turn 1, user: "..." ; (2) Turn 3, user: "..." ; (3) Turn 2, assistant: Image 2 bbox:[x1,y1,x2,y2]</cite><claim>planning clause A[(1)], planning clause B[(2),(3)].</claim>image here"""


# CE groups used by citation-training mode.
CE_GROUP_CITE = 1
CE_GROUP_CLAIM = 2
CE_GROUP_ANSWER_TEXT = 3

# MSE groups used by citation-training mode.
MSE_GROUP_ANSWER_IMAGE = 1


IMAGE_REF_RE = re.compile(r"Image\s*#\s*(\d+)", flags=re.IGNORECASE)
IMAGE_ONLY_RE = re.compile(r"^\s*Image\s*#\s*(\d+)\s*$", flags=re.IGNORECASE)
CLAIM_CITATION_TAG_RE = re.compile(r"\[\((?:\d+)\)(?:,\(\d+\))*\]")
CLAIM_BLOCK_RE = re.compile(r"(?is)<claim>(.*?)</claim>")
INLINE_EID_PAREN_RE = re.compile(r"\((\d+(?:\s*,\s*\d+)*)\)")


def resolve_turn_id(turn: dict, turn_idx: int) -> int:
    raw = turn.get("turn_id", None)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            pass
    return turn_idx + 1


def extract_image_refs(text: str) -> List[int]:
    if not isinstance(text, str) or len(text) == 0:
        return []
    out: List[int] = []
    seen = set()
    for m in IMAGE_REF_RE.finditer(text):
        idx = int(m.group(1))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def is_image_placeholder_text(text: str) -> bool:
    return isinstance(text, str) and IMAGE_ONLY_RE.fullmatch(text.strip()) is not None


def _infer_image_number_from_str(value: str) -> Optional[int]:
    if not isinstance(value, str) or not value:
        return None

    patterns = [
        r"Image\s*#?\s*(\d+)",
        r"_img(\d+)(?:\D|$)",
        r"(?:^|[_\-])img(\d+)(?:\D|$)",
        r"(?:^|[_\-])Image(\d+)(?:\D|$)",
    ]
    for pat in patterns:
        m = re.search(pat, value, flags=re.IGNORECASE)
        if m is not None:
            return int(m.group(1))
    return None


def infer_image_number(
    image_item: dict,
    fallback_numbers: Optional[Sequence[int]] = None,
    fallback_index: int = 0,
) -> Optional[int]:
    if not isinstance(image_item, dict):
        return None

    for key in ("image_id", "uri"):
        num = _infer_image_number_from_str(image_item.get(key, ""))
        if num is not None:
            return num

    if fallback_numbers is not None and 0 <= fallback_index < len(fallback_numbers):
        try:
            return int(fallback_numbers[fallback_index])
        except Exception:
            return None
    return None


def get_first_valid_image_uri(image_items: Sequence[dict]) -> Optional[str]:
    if not isinstance(image_items, Sequence):
        return None
    for item in image_items:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri", None)
        if isinstance(uri, str) and uri:
            return uri
    return None


def _map_images_with_fallback(image_items: Sequence[dict], fallback_numbers: Sequence[int]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not isinstance(image_items, Sequence):
        return mapping

    fallback_ptr = 0
    for item in image_items:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri", None)
        if not (isinstance(uri, str) and uri):
            continue
        num = infer_image_number(item, fallback_numbers=fallback_numbers, fallback_index=fallback_ptr)
        if num is None:
            continue
        if num not in mapping:
            mapping[num] = uri
        fallback_ptr += 1
    return mapping


def extract_turn_image_maps(turn: dict) -> Tuple[Dict[int, str], Dict[int, str]]:
    user = turn.get("user", {}) or {}
    assistant = turn.get("assistant", {}) or {}

    user_text = user.get("text", "") or ""
    assistant_text = assistant.get("text", "") or ""

    user_fallback = extract_image_refs(user_text)
    assistant_fallback = extract_image_refs(assistant_text)

    user_map = _map_images_with_fallback(user.get("images", []) or [], user_fallback)
    assistant_map = _map_images_with_fallback(assistant.get("images_out", []) or [], assistant_fallback)
    return user_map, assistant_map


def merge_image_uri_map(dst: Dict[int, str], src: Dict[int, str]) -> None:
    for k, v in src.items():
        if k not in dst:
            dst[k] = v


def remove_citation_markup_from_history_text(text: str) -> str:
    if not isinstance(text, str) or len(text) == 0:
        return ""

    # remove <cite>...</cite>
    text = re.sub(r"(?is)<cite>.*?</cite>", "", text)

    # keep <claim> body text, but strip [(1),(2)] tags
    def _claim_repl(m):
        body = m.group(1) or ""
        body = CLAIM_CITATION_TAG_RE.sub("", body)
        return body

    text = CLAIM_BLOCK_RE.sub(_claim_repl, text)
    text = text.replace("<claim>", "").replace("</claim>", "")
    text = text.strip()
    return text


def classify_assistant_round(assistant: dict) -> Optional[str]:
    if not isinstance(assistant, dict):
        return None

    text = assistant.get("text", "")
    if not isinstance(text, str):
        text = ""
    text = text.strip()

    images_out = assistant.get("images_out", []) or []
    has_img = any(isinstance(x, dict) and isinstance(x.get("uri", None), str) and x.get("uri", "") for x in images_out)

    # Mixed text+image is treated as text-only supervision.
    if text and (not is_image_placeholder_text(text)):
        return "text"
    if has_img:
        return "image"
    if text:
        return "text"
    return None


def _slice_span_text(text: str, char_start: int, char_end: int) -> str:
    if not isinstance(text, str):
        text = ""
    try:
        s = int(char_start)
        e = int(char_end)
    except Exception:
        return ""
    s = max(0, s)
    e = max(s, e)
    return text[s:e]


def format_bbox_xyxy(bbox_xyxy: Sequence[float]) -> str:
    vals = []
    for v in bbox_xyxy:
        try:
            vals.append(str(float(v)))
        except Exception:
            vals.append(str(v))
    return ",".join(vals)


def build_evidence_index(turns: Sequence[dict], upto_turn_idx: int) -> Tuple[Dict[int, dict], List[str]]:
    evidence: Dict[int, dict] = {}
    errors: List[str] = []

    max_turn = min(len(turns), upto_turn_idx + 1)
    for t_idx in range(max_turn):
        turn = turns[t_idx]
        if not isinstance(turn, dict):
            continue
        turn_id = resolve_turn_id(turn, t_idx)
        user = turn.get("user", {}) or {}
        assistant = turn.get("assistant", {}) or {}

        user_map, assistant_map = extract_turn_image_maps(turn)

        def add_text_spans(role: str, source_text: str, spans: Sequence[dict], kind: str):
            for sp in spans:
                if not isinstance(sp, dict):
                    continue
                sid = sp.get("id", None)
                if not isinstance(sid, int):
                    continue
                if sid in evidence:
                    errors.append(f"duplicate_evidence_id={sid}")
                    continue
                if kind == "claim_span" and isinstance(sp.get("text", None), str):
                    txt = sp.get("text", "")
                else:
                    txt = _slice_span_text(source_text, sp.get("char_start", 0), sp.get("char_end", 0))
                evidence[sid] = {
                    "id": sid,
                    "turn_id": turn_id,
                    "role": role,
                    "kind": "text",
                    "text": txt,
                }

        add_text_spans("user", user.get("text", "") or "", user.get("spans", []) or [], kind="span")
        add_text_spans("assistant", assistant.get("text", "") or "", assistant.get("spans", []) or [], kind="span")
        add_text_spans("assistant", assistant.get("text", "") or "", assistant.get("claim_spans", []) or [], kind="claim_span")

        def add_regions(role: str, images: Sequence[dict], img_map: Dict[int, str]):
            for i, img in enumerate(images):
                if not isinstance(img, dict):
                    continue
                num = infer_image_number(img, fallback_numbers=list(img_map.keys()), fallback_index=i)
                regions = img.get("regions", []) or []
                for rg in regions:
                    if not isinstance(rg, dict):
                        continue
                    rid = rg.get("id", None)
                    if not isinstance(rid, int):
                        continue
                    if rid in evidence:
                        errors.append(f"duplicate_evidence_id={rid}")
                        continue
                    bbox = rg.get("bbox_xyxy", None)
                    if not isinstance(bbox, list):
                        continue
                    evidence[rid] = {
                        "id": rid,
                        "turn_id": turn_id,
                        "role": role,
                        "kind": "bbox",
                        "image_num": int(num) if isinstance(num, int) else None,
                        "bbox_xyxy": bbox,
                    }

        add_regions("user", user.get("images", []) or [], user_map)
        add_regions("assistant", assistant.get("images_out", []) or [], assistant_map)

    return evidence, errors


def build_cite_block(citation_eids: Sequence[int], evidence_index: Dict[int, dict]) -> Tuple[str, Dict[int, int], List[str]]:
    local_idx_map: Dict[int, int] = {}
    entries: List[str] = []
    errors: List[str] = []

    for eid in citation_eids:
        if not isinstance(eid, int):
            errors.append(f"invalid_citation_eid={eid}")
            continue
        if eid in local_idx_map:
            continue
        ev = evidence_index.get(eid, None)
        if ev is None:
            errors.append(f"missing_citation_eid={eid}")
            continue
        local_idx = len(entries) + 1
        local_idx_map[eid] = local_idx

        turn_id = ev.get("turn_id", 0)
        role = ev.get("role", "user")
        kind = ev.get("kind", "")
        if kind == "bbox":
            image_num = ev.get("image_num", None)
            if not isinstance(image_num, int):
                image_num = 0
            bbox = format_bbox_xyxy(ev.get("bbox_xyxy", []))
            entries.append(f"({local_idx}) Turn {turn_id}, {role}: Image {image_num} bbox:[{bbox}]")
        else:
            txt = ev.get("text", "")
            if not isinstance(txt, str):
                txt = ""
            entries.append(f"({local_idx}) Turn {turn_id}, {role}: \"{txt}\"")

    cite_body = " ; ".join(entries)
    return f"<cite>{cite_body}</cite>", local_idx_map, errors


def _dedupe_keep_order(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _strip_inline_claim_eid_markers(text: str, allowed_eids: Sequence[int]) -> str:
    """
    Remove inline legacy evidence-id markers such as (54) or (3,5) from claim text,
    but only when all numbers are in this claim's evidence_ids.
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""
    allowed = set([int(x) for x in allowed_eids if isinstance(x, int)])
    if len(allowed) == 0:
        return text

    def _repl(m: re.Match) -> str:
        raw = m.group(1) or ""
        try:
            nums = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except Exception:
            return m.group(0)
        if len(nums) == 0:
            return m.group(0)
        if all(n in allowed for n in nums):
            return ""
        return m.group(0)

    out = INLINE_EID_PAREN_RE.sub(_repl, text)
    out = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def build_claim_block(claims: Sequence[dict], citation_local_idx_map: Dict[int, int], strict_eids: bool = True) -> Tuple[str, List[str]]:
    errors: List[str] = []
    clauses: List[str] = []
    ending_punct = set([".", ",", ";", ":", "!", "?", "。", "，", "；", "：", "！", "？"])

    for cidx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        text = claim.get("text", "")
        if not isinstance(text, str):
            text = ""
        text = text.strip()

        claim_eids = claim.get("evidence_ids", []) or []
        if not isinstance(claim_eids, list):
            claim_eids = []
        claim_eids = [x for x in claim_eids if isinstance(x, int)]
        text = _strip_inline_claim_eid_markers(text, claim_eids)

        invalid = [x for x in claim_eids if x not in citation_local_idx_map]
        if strict_eids and len(invalid) > 0:
            errors.append(f"claim_{cidx}_invalid_eids={invalid}")
            continue

        mapped_local = _dedupe_keep_order([citation_local_idx_map[x] for x in claim_eids if x in citation_local_idx_map])
        if strict_eids and len(mapped_local) == 0:
            errors.append(f"claim_{cidx}_no_valid_eids")
            continue

        tag = "[" + ",".join([f"({x})" for x in mapped_local]) + "]" if len(mapped_local) > 0 else ""
        if text and tag:
            text_strip = text.rstrip()
            if len(text_strip) > 0 and text_strip[-1] in ending_punct:
                clause = f"{text_strip[:-1].rstrip()}{tag}{text_strip[-1]}"
            else:
                clause = f"{text_strip}{tag}"
        else:
            clause = f"{text}{tag}" if text else tag
        if clause:
            clauses.append(clause)

    claim_body = " ".join(clauses)
    return f"<claim>{claim_body}</claim>", errors


def split_text_with_first_appearance_images(
    text: str,
    image_uri_map: Dict[int, str],
    inserted_image_nums: set,
) -> List[dict]:
    if not isinstance(text, str):
        text = ""

    out: List[dict] = []
    cursor = 0
    has_ref = False
    for m in IMAGE_REF_RE.finditer(text):
        has_ref = True
        if m.start() > cursor:
            out.append({"type": "text", "value": text[cursor:m.start()]})
        ref_text = m.group(0)
        ref_num = int(m.group(1))
        path = image_uri_map.get(ref_num, None)
        if isinstance(path, str) and path and ref_num not in inserted_image_nums:
            inserted_image_nums.add(ref_num)
            out.append({"type": "image", "value": path, "image_num": ref_num})
        else:
            out.append({"type": "text", "value": ref_text})
        cursor = m.end()

    if not has_ref:
        out.append({"type": "text", "value": text})
    elif cursor < len(text):
        out.append({"type": "text", "value": text[cursor:]})

    return out


def collect_first_appearance_images_for_text(
    text: str,
    image_uri_map: Dict[int, str],
    inserted_image_nums: set,
) -> List[dict]:
    """
    Collect images whose symbolic refs (Image #N) appear in this text and are
    first-seen within the current sample.

    Return list items:
      {"image_num": N, "uri": "..."}
    """
    out: List[dict] = []
    seen_in_text = set()
    for image_num in extract_image_refs(text):
        if image_num in seen_in_text:
            continue
        seen_in_text.add(image_num)
        if image_num in inserted_image_nums:
            continue
        uri = image_uri_map.get(image_num, None)
        if not (isinstance(uri, str) and uri):
            continue
        inserted_image_nums.add(image_num)
        out.append({"image_num": image_num, "uri": uri})
    return out


def format_chatml_utterance(role: str, text: str, include_turn: bool = False, turn_no: int = 0) -> str:
    """
    Build one ChatML-like utterance block.

    - role is kept as official user/assistant label.
    - include_turn controls whether turn marker is prepended into content.
    """
    if not isinstance(role, str) or role not in {"user", "assistant"}:
        role = "user"
    if not isinstance(text, str):
        text = ""
    if include_turn:
        text = f"Turn {int(turn_no)}: {text}"
    return f"<|im_start|>{role}\n{text}\n<|im_end|>\n"


def format_chatml_assistant_prefix() -> str:
    return "<|im_start|>assistant\n"
