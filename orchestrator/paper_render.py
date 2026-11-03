"""Appendix D round contracts; each sample is consumed as one ordered dialogue."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any
from executor.tasks.prompt_loader import current_prompt_profile


def render_paper_unified(item: dict[str, Any]) -> dict[str, Any]:
    from align_unify_render_format import _frames, _positions, _target_pos, _target_positions
    frames = {int(f["pos"]): f for f in _frames(item)}
    positions = _positions(item)
    original = item.get("turns", {})
    benchmark = item["benchmark"]
    rounds = []
    candidates, answers = {}, {}

    def add(name: str, prompt: str, inputs: list[int], answer: Any = None,
            output_image: int | None = None) -> None:
        images = [frames[p]["img"] for p in inputs]
        messages = [{"role": "user", "type": "text", "content": prompt}]
        messages += [{"role": "user", "type": "image", "content": f"Image #{i+1}"} for i in range(len(images))]
        output = [frames[output_image]["img"]] if output_image is not None else []
        content = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        answer_messages = [{"role": "assistant", "type": "image" if output else "text",
                            "content": "Image #1" if output else content}]
        rounds.append({"t": name, "mode": "edit" if output else "vlm_understand",
                       "in": {"frame_positions": inputs, "images": images},
                       "gt": {"image": output[0]} if output else {"answer": answer},
                       "render": {"query": {"messages": messages, "images": images},
                                  "answer": {"messages": answer_messages, "images": output}}})

    def select(target: int, step: str, rationale: str) -> None:
        pool = list(dict.fromkeys([target] + positions + sorted(frames)))[:4]
        if len(pool) < 2 or any(p not in frames for p in pool):
            raise ValueError("Paper selection requires at least two real candidate frames")
        seed = hashlib.sha256(f"{item.get('unify_id', '')}|{step}|{pool}".encode()).digest()
        random.Random(int.from_bytes(seed[:8], "big")).shuffle(pool)
        rows = [{"id": chr(65+i), "image": frames[p]["img"], "frame": f"F{p:02d}"} for i, p in enumerate(pool)]
        answer = next(r["id"] for r in rows if r["frame"] == f"F{target:02d}")
        candidates[step], answers[step] = rows, answer
        add("select_"+step, "Which candidate matches the generated target image from the preceding generation round? "
            "Reply with the candidate letter only.\n" + "\n".join(f"{r['id']}: Image #{i+1}" for i,r in enumerate(rows)), pool, answer)
        rounds[-1]["rationale"] = rationale

    if benchmark.endswith("seed.l3.gts"):
        source, target = positions[:2]
        plan = original["plan"]
        add("plan", "What should change from the source image to reach the target state? Identify goal, invariants and grounded edits.", [source, target], plan)
        add("gen", "Generate the target image after applying the grounded edit from the plan.", [source], output_image=target)
        select(target, "step1", original.get("select", {}).get("rationale", ""))
    elif benchmark.endswith("weave.uie_select_recall"):
        source, target = positions[:2]
        add("understand", "What visible facts describe the earlier and later image states?", [source, target], original["understand"])
        add("plan", "What grounded visual changes connect the earlier state to the later state?", [source, target], original["plan"])
        qa = original["qa_open"]
        add("qa_open", qa["question"], [source], qa["answer"])
        rounds[-1]["rubric"] = qa.get("rubric", [])
        add("gen", "Generate the recalled later-state image using the preceding dialogue.", [source], output_image=target)
        select(target, "step1", original.get("select", {}).get("rationale", ""))
    elif benchmark.endswith("comm.task4.c1_steps"):
        targets = _target_positions(item)
        add("overall", "What is the overall coherent visual sequence?", [positions[0], positions[-1]], original["overall"])
        add("steps", "What are the grounded step-by-step state changes?", positions, original["steps"])
        for i, target in enumerate(targets):
            add(f"gen_step{i+1}", f"Generate the target image for step {i+1} from the grounded steps.", [positions[i]], output_image=target)
        for i, target in enumerate(targets):
            selects = original.get("select", [])
            select(target, f"step{i+1}", selects[i].get("rationale", "") if i < len(selects) else "")
    elif benchmark.endswith("comm.task4.c2_fill_gap"):
        earlier, middle, later = positions[:3]
        add("explain", "What visual state fits between the earlier and later frames, and why?", [earlier, later], original["explanation"])
        add("gen", "Generate the middle-step image that fits both adjacent transitions.", [earlier, later], output_image=middle)
        select(middle, "step1", original.get("select", {}).get("rationale", ""))
    elif benchmark.endswith("isg.i2.seq_select"):
        context = positions[:2]
        targets = _target_positions(item)
        for name, field, question in [
            ("overall", "overall", "What future visual sequence should follow from these context frames?" if current_prompt_profile() in {'paper_v10', 'paper_v11', 'paper_v13', 'paper_v15'} else "What future visual sequence follows from these context frames?"),
            ("edit", "edits", "What visible edit should be made at each future step?"),
            ("reason", "because", "Why does each future step follow from the context or previous state?"),
            ("target", "so_that", "What visual state should each future step reach?"),
            ("check", "check", "How can each future step be visually verified?"),
        ]:
            add(name, question, context, original[field])
        for i, target in enumerate(targets):
            add(f"gen_step{i+1}", f"Generate the target image for future step {i+1} using the grounded plan.", context+targets[:i], output_image=target)
        for i, target in enumerate(targets):
            selects = original.get("select", [])
            select(target, f"step{i+1}", selects[i].get("rationale", "") if i < len(selects) else "")
    else:
        raise ValueError(f"Unknown paper unified task: {benchmark}")
    out = copy.deepcopy(item)
    out["turns"] = rounds
    out["eval"] = {"candidates": candidates, "answer": answers}
    out["context_mode"] = "ordered_dialogue"
    out["meta"]["legacy_unify"] = {k: copy.deepcopy(item.get(k)) for k in ["turns", "answer", "render", "evidence_refs"]}
    out["meta"]["target_image_source"] = "observed_video_keyframe"
    return out
