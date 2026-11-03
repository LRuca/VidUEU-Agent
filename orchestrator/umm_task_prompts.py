from __future__ import annotations

import json
from typing import Any, Dict

from executor.tasks.prompt_loader import render_prompt_text
from executor.tasks.prompt_loader import current_prompt_profile


def _build_prompt(
    *,
    task_type: str,
    task_payload: Dict[str, Any],
    evidence_pack_text: str,
    schema: Dict[str, Any],
    system_style: str,
) -> str:
    if current_prompt_profile() != "legacy":
        schema = {**schema, "evidence_refs": ["S_SEGEMENT_SUM", "F00_CAP", "P00_TRANS"]}
    return render_prompt_text(
        "umm_task_prompts",
        "build_prompt_base",
        system_style=system_style,
        task_type=task_type,
        task_payload_json=json.dumps(task_payload, ensure_ascii=False, indent=2),
        evidence_pack_text=evidence_pack_text,
        schema_json=json.dumps(schema, ensure_ascii=False, indent=2),
    )


def _benchmark_style_rules() -> str:
    return render_prompt_text("umm_task_prompts", "benchmark_style_rules")


def _seed_system_style() -> str:
    return render_prompt_text("umm_task_prompts", "seed_system_style")


def _weave_system_style() -> str:
    return render_prompt_text("umm_task_prompts", "weave_system_style")


def _comm_system_style() -> str:
    return render_prompt_text("umm_task_prompts", "comm_system_style")


def _isg_system_style() -> str:
    return render_prompt_text("umm_task_prompts", "isg_system_style")


def _join_prompt(parts: list[str]) -> str:
    return "\n\n".join(parts) if current_prompt_profile() != "legacy" else "".join(parts)


def build_seed_prompt(task_payload: Dict[str, Any], evidence_pack_text: str) -> str:
    schema = {
        "plan": {"goal": "", "invariants": [""], "edits": [""]},
        "gen": {"summary": ""},
        "select": {"question": "", "answer": "", "rationale": ""},
    }
    prompt = _build_prompt(
        task_type="unify.seed.l3.gts",
        task_payload=task_payload,
        evidence_pack_text=evidence_pack_text,
        schema=schema,
        system_style=_seed_system_style(),
    )
    return _join_prompt([prompt, _benchmark_style_rules(), render_prompt_text("umm_task_prompts", "seed_extra_rules")])


def build_weave_prompt(task_payload: Dict[str, Any], evidence_pack_text: str) -> str:
    schema = {
        "understand": {"facts": [""]},
        "plan": {"edits": [""]},
        "qa_open": {"question": "", "answer": "", "rubric": [""]},
        "select": {"question": "", "answer": "", "rationale": ""},
    }
    prompt = _build_prompt(
        task_type="unify.weave.uie_select_recall",
        task_payload=task_payload,
        evidence_pack_text=evidence_pack_text,
        schema=schema,
        system_style=_weave_system_style(),
    )
    return _join_prompt([prompt, _benchmark_style_rules(), render_prompt_text("umm_task_prompts", "weave_extra_rules")])


def build_comm_c1_prompt(task_payload: Dict[str, Any], evidence_pack_text: str, step_count: int) -> str:
    schema = {
        "question": "",
        "overall": "",
        "step_texts": [""] * max(1, int(step_count)),
        "select": [{"question": "", "answer": "", "rationale": ""}] * max(1, int(step_count)),
    }
    if current_prompt_profile() in {"paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"}:
        schema.pop("overall")
    prompt = _build_prompt(
        task_type="unify.comm.task4.c1_steps",
        task_payload=task_payload,
        evidence_pack_text=evidence_pack_text,
        schema=schema,
        system_style=_comm_system_style(),
    )
    return _join_prompt([prompt, _benchmark_style_rules(), render_prompt_text("umm_task_prompts", "comm_c1_extra_rules")])


def build_comm_c2_prompt(task_payload: Dict[str, Any], evidence_pack_text: str) -> str:
    schema = {
        "question": "",
        "explanation": "",
        "select": {"question": "", "answer": "", "rationale": ""},
    }
    prompt = _build_prompt(
        task_type="unify.comm.task4.c2_fill_gap",
        task_payload=task_payload,
        evidence_pack_text=evidence_pack_text,
        schema=schema,
        system_style=_comm_system_style(),
    )
    return _join_prompt([prompt, _benchmark_style_rules(), render_prompt_text("umm_task_prompts", "comm_c2_extra_rules")])


def build_isg_prompt(task_payload: Dict[str, Any], evidence_pack_text: str, step_count: int) -> str:
    schema = {
        "overall": "",
        "edits": [""] * max(1, int(step_count)),
        "because": [""] * max(1, int(step_count)),
        "so_that": [""] * max(1, int(step_count)),
        "check": [""] * max(1, int(step_count)),
        "select": [{"question": "", "answer": "", "rationale": ""}] * max(1, int(step_count)),
    }
    prompt = _build_prompt(
        task_type="unify.isg.i2.seq_select",
        task_payload=task_payload,
        evidence_pack_text=evidence_pack_text,
        schema=schema,
        system_style=_isg_system_style(),
    )
    return _join_prompt([prompt, _benchmark_style_rules(), render_prompt_text("umm_task_prompts", "isg_extra_rules")])
