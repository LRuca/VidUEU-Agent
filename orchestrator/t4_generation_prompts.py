from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from executor.tasks.prompt_loader import render_prompt_text, current_prompt_profile
from orchestrator.evidence_selection import complete_evidence_text


def build_fewshot_bank() -> Dict[str, str]:
    bank = {
        "mme": render_prompt_text("t4_generation_prompts", "fewshot_bank", "mme"),
        "mmt_bench": render_prompt_text("t4_generation_prompts", "fewshot_bank", "mmt_bench"),
        "llava_bench": render_prompt_text("t4_generation_prompts", "fewshot_bank", "llava_bench"),
        "edit": render_prompt_text("t4_generation_prompts", "fewshot_bank", "edit"),
    }
    bank["magicbrush"] = bank["edit"]
    return bank


def build_instance_generation_prompt(
    *,
    benchmark: str,
    fewshot_text: str,
    seg_meta: Dict[str, Any],
    default_pos: int,
    default_img: str,
    body: str,
    evidence_pack_text: str,
    quota: int,
    schema: str,
) -> str:
    fewshot_block = (fewshot_text + "\n\n") if fewshot_text else ""
    return render_prompt_text(
        "t4_generation_prompts",
        "instance_generation_base",
        fewshot_block=fewshot_block,
        seg_meta_json=json.dumps(seg_meta, ensure_ascii=False),
        default_pos=default_pos,
        default_img=default_img,
        body=body,
        evidence_pack_text=evidence_pack_text,
        benchmark=benchmark,
        quota=quota,
        schema=schema,
    )


def build_dialog_evidence_pack(
    *,
    pos_list: List[int],
    evidence_id_set: set[str],
    segment_summary: str,
    frames_aligned: List[Dict[str, Any]],
    short_text: Callable[[str, int], str],
    tool_evidence_suffix: Callable[[str], str],
    max_chars: int = 2400,
    evidence_units: List[Dict[str, Any]] | None = None,
) -> str:
    if current_prompt_profile() in {"paper_v5", "paper_v6", "paper_v7", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"}:
        positions = set(pos_list)
        units = []
        required = set()
        for row in evidence_units or []:
            key = row.get("evidence_id", "")
            frame_selected = any(key.startswith(f"F{p:02d}_") for p in positions)
            pair_selected = row.get("pair_from_pos") in positions and row.get("pair_to_pos") in positions
            if key == "S_SEGEMENT_SUM" or frame_selected or pair_selected:
                units.append(row)
                if key == "S_SEGEMENT_SUM" or key.endswith(("_CAP", "_STATE", "_TRANS", "_COMPARE")):
                    required.add(key)
        for p in positions:
            required.add(f"F{p:02d}_CAP")
            required.add(f"F{p:02d}_STATE")
        return complete_evidence_text(units, required, max_chars=max_chars)
    lines = []
    if "S_SEGEMENT_SUM" in evidence_id_set:
        lines.append(f"[S_SEGEMENT_SUM] {short_text(segment_summary, 500)}")
    for p in pos_list:
        p = int(p)
        if 0 <= p < len(frames_aligned):
            fr = frames_aligned[p]
            cap = (fr.get("caption") or "").strip()
            if cap and f"F{p:02d}_CAP" in evidence_id_set:
                lines.append(f"[F{p:02d}_CAP] Frame F{p:02d}: {cap}")
            frame_tool_descs = fr.get("frame_tool_descs", {}) if isinstance(fr.get("frame_tool_descs", {}), dict) else {}
            for tool_name, tool_summary in frame_tool_descs.items():
                eid = f"F{p:02d}_TOOL_{tool_evidence_suffix(tool_name)}"
                if eid not in evidence_id_set:
                    continue
                tool_summary = str(tool_summary or "").strip()
                if tool_summary:
                    lines.append(f"[{eid}] Frame F{p:02d}: {tool_name} semantic hint: {short_text(tool_summary, 180)}")
    packed = "\n".join(lines).strip()
    if len(packed) > max_chars:
        packed = packed[:max_chars].rstrip() + " ..."
    return packed


def build_dialog_repair_prompt(*, kind: str, rounds: int, pos_list: List[int], dlg_ev: str, turns: List[Dict[str, Any]]) -> str:
    want = int(rounds) * 2
    schema = "{\n  \"turns\": [\n    {\"role\": \"user\"|\"assistant\", \"text\": str, \"evidence_refs\": [str, ...]}\n  ]\n}\n"
    return render_prompt_text(
        "t4_generation_prompts",
        "dialog_repair",
        want=want,
        kind=kind,
        image_pos_list=json.dumps([int(p) for p in pos_list], ensure_ascii=False),
        dlg_ev=dlg_ev,
        turns_json=json.dumps(turns, ensure_ascii=False),
        schema=schema,
    )
