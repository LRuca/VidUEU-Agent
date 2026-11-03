"""Ordered Appendix-D round validation independent of media/model loading."""
from __future__ import annotations

from typing import Any


def validate_paper_rounds(turns: list[dict[str, Any]], task: str) -> None:
    """Require each named generation/selection round, not just a task label.

    BAGEL's legacy multiturn loader accepts several compressed formats. Those
    formats are useful inputs but do not establish the paper's full episodes.
    """
    if task not in {
        "grounded_edit_to_select", "understand_recall_generate_select",
        "stepwise_sequence", "middle_step_gap", "instructional_future_sequence",
    }:
        return
    names = [str(turn.get("t") or "") for turn in turns]
    if task in {"grounded_edit_to_select", "understand_recall_generate_select", "middle_step_gap"}:
        prefix = {
            "grounded_edit_to_select": ["plan"],
            "understand_recall_generate_select": ["understand", "plan", "qa_open"],
            "middle_step_gap": ["explain"],
        }[task]
        expected = [*prefix, "gen", "select_step1"]
        # The historical single-selection name has identical round semantics.
        normalized = ["select_step1" if name == "select" else name for name in names]
    else:
        prefix = (["overall", "steps"] if task == "stepwise_sequence"
                  else ["overall", "edit", "reason", "target", "check"])
        count = sum(name.startswith("gen_step") for name in names)
        if count < (2 if task == "stepwise_sequence" else 1):
            raise ValueError(f"paper_rounds_missing_generation_steps:{task}")
        expected = [*prefix, *[f"gen_step{i+1}" for i in range(count)],
                    *[f"select_step{i+1}" for i in range(count)]]
        normalized = names
    if normalized != expected:
        raise ValueError(f"paper_round_order_mismatch:{task}")
    targets = [turn["render"]["answer"]["images"][0] for name, turn in zip(normalized, turns)
               if (name == "gen" or name.startswith("gen_step"))
               and len(turn.get("render", {}).get("answer", {}).get("images", [])) == 1]
    selection_index = 0
    for name, turn in zip(normalized, turns):
        render = turn.get("render") or {}
        query, answer = render.get("query") or {}, render.get("answer") or {}
        inputs, outputs = query.get("images") or [], answer.get("images") or []
        if name == "gen" or name.startswith("gen_step"):
            if len(outputs) != 1 or not inputs:
                raise ValueError(f"paper_generation_requires_condition_and_one_target:{task}:{name}")
        elif outputs:
            raise ValueError(f"paper_text_round_has_image_target:{task}:{name}")
        if name.startswith("select"):
            if len(inputs) < 2 or len(set(inputs)) != len(inputs):
                raise ValueError(f"paper_selection_requires_real_distinct_candidates:{task}:{name}")
            messages = answer.get("messages") or []
            text = [m.get("content") for m in messages if m.get("type") == "text"]
            allowed = {chr(65+i) for i in range(len(inputs))}
            if len(text) != 1 or text[0] not in allowed:
                raise ValueError(f"paper_selection_requires_valid_letter:{task}:{name}")
            if selection_index >= len(targets) or inputs[ord(text[0])-65] != targets[selection_index]:
                raise ValueError(f"paper_selection_does_not_match_generated_target:{task}:{name}")
            selection_index += 1
