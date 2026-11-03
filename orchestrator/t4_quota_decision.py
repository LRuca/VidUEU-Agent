from __future__ import annotations

import json
from typing import Any, Dict, List


def build_default_understanding_quota_decision(
    *,
    t4_conf: Dict[str, Any],
    g_total: int,
    mix_warmup: int,
    single_scale: float,
    dialog_scale_up: float,
    dialog_scale_down: float,
    cur_dialog_r: float,
    dialog_min: float,
    max_llava: int,
    max_mmt: int,
    max_mme: int,
    max_dialog_visdial: int,
    max_dialog_mmdu: int,
) -> Dict[str, Any]:
    candidate_types = ["llava_bench", "mmt_bench", "mme", "dialog_visdial_like", "dialog_mmdu_like"]
    quota_limits = {
        "llava_bench": max_llava,
        "mmt_bench": max_mmt,
        "mme": max_mme,
        "dialog_visdial_like": max_dialog_visdial,
        "dialog_mmdu_like": max_dialog_mmdu,
    }
    default_quotas = {
        "llava_bench": max(0, min(max_llava, 1)),
        "mmt_bench": max(0, min(max_mmt, 2)),
        "mme": max(0, min(max_mme, 6)),
        "dialog_visdial_like": max(0, min(max_dialog_visdial, int(t4_conf.get("default_dialog_visdial", 2) or 2))),
        "dialog_mmdu_like": max(0, min(max_dialog_mmdu, int(t4_conf.get("default_dialog_mmdu", 3) or 3))),
    }

    if g_total >= mix_warmup:
        if single_scale < 1.0:
            default_quotas["mme"] = int(round(float(default_quotas.get("mme", 0) or 0) * single_scale))
            default_quotas["mmt_bench"] = int(round(float(default_quotas.get("mmt_bench", 0) or 0) * single_scale))
            default_quotas["llava_bench"] = int(round(float(default_quotas.get("llava_bench", 0) or 0) * single_scale))
        if dialog_scale_up > 1.0:
            default_quotas["dialog_visdial_like"] = int(round(float(default_quotas.get("dialog_visdial_like", 0) or 0) * dialog_scale_up))
            default_quotas["dialog_mmdu_like"] = int(round(float(default_quotas.get("dialog_mmdu_like", 0) or 0) * min(dialog_scale_up, 1.5)))
        if dialog_scale_down < 1.0:
            default_quotas["dialog_mmdu_like"] = int(round(float(default_quotas.get("dialog_mmdu_like", 0) or 0) * dialog_scale_down))
            default_quotas["dialog_visdial_like"] = int(round(float(default_quotas.get("dialog_visdial_like", 0) or 0) * max(dialog_scale_down, 0.5)))

        default_quotas["llava_bench"] = max(0, min(int(default_quotas.get("llava_bench", 0) or 0), max_llava))
        default_quotas["mmt_bench"] = max(0, min(int(default_quotas.get("mmt_bench", 0) or 0), max_mmt))
        default_quotas["mme"] = max(0, min(int(default_quotas.get("mme", 0) or 0), max_mme))
        default_quotas["dialog_visdial_like"] = max(0, min(int(default_quotas.get("dialog_visdial_like", 0) or 0), max_dialog_visdial))
        default_quotas["dialog_mmdu_like"] = max(0, min(int(default_quotas.get("dialog_mmdu_like", 0) or 0), max_dialog_mmdu))

    try:
        if g_total >= mix_warmup and cur_dialog_r < dialog_min:
            default_quotas["dialog_visdial_like"] = min(max_dialog_visdial, max(1, int(default_quotas.get("dialog_visdial_like", 0) or 0)))
            default_quotas["dialog_mmdu_like"] = min(max_dialog_mmdu, max(1, int(default_quotas.get("dialog_mmdu_like", 0) or 0)))
    except Exception:
        pass

    return {
        "candidate_types": candidate_types,
        "quota_limits": quota_limits,
        "default_quotas": default_quotas,
        "decision": {
            "quotas": dict(default_quotas),
            "rationales": [],
            "llm_raw": None,
            "notes": {
                "keyframe_index_space": "pos_0_to_N-1",
                "must_cite_evidence_id": True,
                "dialog_visdial_like": "single-image multi-turn (VisDial-like)",
                "dialog_mmdu_like": "multi-image interleaved (MMDU-like) with <ImageHere>",
            },
        },
    }


def build_understanding_quota_decision_prompt(
    *,
    candidate_types: List[str],
    quota_limits: Dict[str, int],
    evidence_pack_text: str,
    global_mix_note: str,
) -> str:
    quota_lines = "\n".join([f"- {k}: 0..{v}" for k, v in quota_limits.items()])
    return (
        "You are a quota allocator for understanding-only benchmark generation.\n"
        "Decide quotas ONLY for understanding benchmarks in this segment.\n"
        "Do not allocate edit, unify, or generation here.\n\n"
        f"{global_mix_note}\n"
        "Use the evidence pack below. Prefer valid output count, groundedness, and feasible dialog yield.\n"
        "Every rationale for quota > 0 must cite at least one S_* evidence id.\n\n"
        f"Candidate benchmark types: {json.dumps(candidate_types, ensure_ascii=False)}\n"
        f"Quota limits:\n{quota_lines}\n\n"
        "EVIDENCE PACK:\n"
        f"{evidence_pack_text}\n\n"
        "Output STRICT JSON only with this schema:\n"
        "{\n"
        '  "quotas": {\n'
        '    "llava_bench": int,\n'
        '    "mmt_bench": int,\n'
        '    "mme": int,\n'
        '    "dialog_visdial_like": int,\n'
        '    "dialog_mmdu_like": int\n'
        "  },\n"
        '  "rationales": [\n'
        '    {"type": str, "evidence_refs": [str, ...], "reason": str}\n'
        "  ]\n"
        "}\n"
    )


def apply_understanding_quota_decision_response(
    *,
    parsed: Dict[str, Any] | None,
    decision: Dict[str, Any],
    candidate_types: List[str],
    default_quotas: Dict[str, int],
    max_llava: int,
    max_mmt: int,
    max_mme: int,
    max_dialog_visdial: int,
    max_dialog_mmdu: int,
    llm_decision_raw: str,
) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return decision

    quotas = parsed.get("quotas")
    rats = parsed.get("rationales")
    valid_types = set(candidate_types)

    def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
        try:
            iv = int(v)
        except Exception:
            iv = int(default)
        return max(lo, min(iv, hi))

    if isinstance(quotas, dict):
        decision["quotas"] = {
            "llava_bench": _clamp_int(quotas.get("llava_bench", default_quotas["llava_bench"]), 0, max_llava, default_quotas["llava_bench"]),
            "mmt_bench": _clamp_int(quotas.get("mmt_bench", default_quotas["mmt_bench"]), 0, max_mmt, default_quotas["mmt_bench"]),
            "mme": _clamp_int(quotas.get("mme", default_quotas["mme"]), 0, max_mme, default_quotas["mme"]),
            "dialog_visdial_like": _clamp_int(quotas.get("dialog_visdial_like", default_quotas["dialog_visdial_like"]), 0, max_dialog_visdial, default_quotas["dialog_visdial_like"]),
            "dialog_mmdu_like": _clamp_int(quotas.get("dialog_mmdu_like", default_quotas["dialog_mmdu_like"]), 0, max_dialog_mmdu, default_quotas["dialog_mmdu_like"]),
        }

    if isinstance(rats, list):
        rr = []
        for r in rats:
            if not isinstance(r, dict):
                continue
            tp = r.get("type")
            evr = r.get("evidence_refs", [])
            rsn = r.get("reason", "")
            if not isinstance(tp, str) or tp not in valid_types:
                continue
            if not isinstance(evr, list):
                evr = []
            rr.append({"type": tp, "evidence_refs": [str(x) for x in evr[:12]], "reason": str(rsn)[:600]})
        decision["rationales"] = rr

    decision["llm_raw"] = llm_decision_raw
    return decision
