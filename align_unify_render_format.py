from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _msg(role: str, typ: str, content: Any) -> Dict[str, str]:
    return {"role": str(role), "type": str(typ), "content": str(content or "")}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _nonempty_text_messages(*values: Any) -> List[Dict[str, str]]:
    return [_msg("user", "text", value) for value in values if str(value or "").strip()]


def _task_name(benchmark: str) -> str:
    text = str(benchmark or "")
    return text[len("unify.") :] if text.startswith("unify.") else text


def _frames(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    frames = meta.get("segment_keyframes")
    if isinstance(frames, list):
        return [dict(x) for x in frames if isinstance(x, dict)]
    return []


def _frame_img(item: Dict[str, Any], pos: int) -> str:
    for fr in _frames(item):
        try:
            if int(fr.get("pos", -1)) == int(pos):
                return str(fr.get("img", "") or "")
        except Exception:
            continue
    return ""


def _frame_caption(item: Dict[str, Any], pos: int) -> str:
    for fr in _frames(item):
        try:
            if int(fr.get("pos", -1)) == int(pos):
                return str(fr.get("caption", "") or "")
        except Exception:
            continue
    return ""


def _positions(item: Dict[str, Any]) -> List[int]:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    out: List[int] = []
    for value in meta.get("frame_positions") if isinstance(meta.get("frame_positions"), list) else []:
        try:
            out.append(int(value))
        except Exception:
            pass
    return out


def _target_pos(item: Dict[str, Any]) -> Optional[int]:
    ans = item.get("answer") if isinstance(item.get("answer"), dict) else {}
    for key in ("target_frame_pos",):
        try:
            return int(ans.get(key))
        except Exception:
            pass
    values = ans.get("target_frame_positions") or ans.get("future_frame_positions")
    if isinstance(values, list) and values:
        try:
            return int(values[0])
        except Exception:
            return None
    return None


def _target_positions(item: Dict[str, Any]) -> List[int]:
    ans = item.get("answer") if isinstance(item.get("answer"), dict) else {}
    for key in ("target_frame_positions", "future_frame_positions"):
        values = ans.get(key)
        if isinstance(values, list):
            out: List[int] = []
            for value in values:
                try:
                    out.append(int(value))
                except Exception:
                    pass
            if out:
                return out
    pos = _target_pos(item)
    return [pos] if pos is not None else []


def _candidate_positions(item: Dict[str, Any], target_pos: int, context: List[int] | None = None, limit: int = 4) -> List[int]:
    frame_positions = []
    for fr in _frames(item):
        try:
            frame_positions.append(int(fr.get("pos", -1)))
        except Exception:
            pass
    out: List[int] = [int(target_pos)]
    for pos in context or []:
        if pos != target_pos:
            out.append(int(pos))
    for delta in (1, -1, 2, -2, 3, -3, 4, -4):
        pos = int(target_pos) + delta
        if pos in frame_positions and pos not in out:
            out.append(pos)
        if len(out) >= limit:
            break
    for pos in frame_positions:
        if pos not in out:
            out.append(pos)
        if len(out) >= limit:
            break
    return out[:limit]


def _candidates(item: Dict[str, Any], target_pos: int, context: List[int] | None = None) -> tuple[List[Dict[str, str]], str]:
    rows: List[Dict[str, str]] = []
    answer = "A"
    for idx, pos in enumerate(_candidate_positions(item, target_pos, context=context)):
        image = _frame_img(item, pos)
        if not image:
            continue
        label = chr(ord("A") + len(rows))
        if int(pos) == int(target_pos):
            answer = label
        rows.append({"id": label, "image": image, "frame": f"F{int(pos):02d}"})
    return rows, answer


def _select_render(candidates: List[Dict[str, str]], answer: Any) -> Dict[str, Any]:
    images = [c["image"] for c in candidates if c.get("image")]
    label_lines = [f"{c.get('id', '')}: Image #{idx + 1}" for idx, c in enumerate(candidates)]
    messages = [
        _msg("user", "text", "Select the real image. Reply with A/B/C/D only."),
        _msg("user", "text", "\n".join(label_lines)),
    ]
    for idx in range(len(images)):
        messages.append(_msg("user", "image", f"Image #{idx + 1}"))
    return {
        "query": {"messages": messages, "images": images},
        "answer": {"messages": [_msg("assistant", "text", answer)], "images": []},
    }


def _multi_select_render(step_candidates: Dict[str, List[Dict[str, str]]], answer: Dict[str, str]) -> Dict[str, Any]:
    messages = [_msg("user", "text", 'Select the real image for EACH step. Reply with JSON like {"step1":"A", ...}.')]
    images: List[str] = []
    img_idx = 1
    for step_key in sorted(step_candidates, key=lambda x: (int("".join(ch for ch in x if ch.isdigit()) or 0), x)):
        labels: List[str] = []
        for cand in step_candidates[step_key]:
            labels.append(f"{cand.get('id', '')}: Image #{img_idx}")
            images.append(cand.get("image", ""))
            img_idx += 1
        messages.append(_msg("user", "text", step_key))
        messages.append(_msg("user", "text", "\n".join(labels)))
    for idx in range(len(images)):
        messages.append(_msg("user", "image", f"Image #{idx + 1}"))
    return {
        "query": {"messages": messages, "images": images},
        "answer": {"messages": [_msg("assistant", "text", _json_text(answer))], "images": []},
    }


def _src_base(item: Dict[str, Any], task: str, images: List[str], frames: List[int]) -> Dict[str, Any]:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return {
        "segment": str(meta.get("segment_path", "") or ""),
        "frames": frames,
        "images": images,
        "pair_keys": list(meta.get("pair_keys") or []),
        "evidence_refs": list(item.get("evidence_refs") or []),
        "benchmark": task,
    }


def _add_turn_render(turn: Dict[str, Any], task_type: str, render: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(turn)
    out["task_type"] = task_type
    out["render"] = render
    return out


def _align_seed(item: Dict[str, Any]) -> Dict[str, Any]:
    turns = item.get("turns") if isinstance(item.get("turns"), dict) else {}
    plan = turns.get("plan", {}) if isinstance(turns.get("plan"), dict) else {}
    gen = turns.get("gen", {}) if isinstance(turns.get("gen"), dict) else {}
    select = turns.get("select", {}) if isinstance(turns.get("select"), dict) else {}
    positions = _positions(item)
    src_pos = positions[0] if positions else 0
    tgt_pos = _target_pos(item) or (positions[1] if len(positions) > 1 else src_pos)
    src_img = _frame_img(item, src_pos)
    tgt_img = _frame_img(item, tgt_pos)
    src_caption = _frame_caption(item, src_pos)
    edits = [{"edit": str(e), "because": "", "so_that": "", "where": "", "check": ""} for e in plan.get("edits", [])]
    candidates, answer = _candidates(item, tgt_pos, context=[src_pos])
    plan_prompt = [ _msg("user", "text", "You are given an image and a goal. Write a plan with invariants and edits."), _msg("user", "image", "Image #1") ]
    plan_prompt.extend(_nonempty_text_messages(src_caption and f"Caption: {src_caption}", plan.get("goal") and f"Goal: {plan.get('goal', '')}"))
    out_turns = [
        _add_turn_render(
            {"t": "plan", "in": {"image": src_img, "caption": src_caption, "goal": plan.get("goal", "")}, "gt": {"invariants": plan.get("invariants", []), "edits": edits}},
            "vlm_understand",
            {
                "query": {
                    "messages": plan_prompt,
                    "images": [src_img] if src_img else [],
                },
                "answer": {"messages": [_msg("assistant", "text", _json_text({"invariants": plan.get("invariants", []), "edits": edits}))], "images": []},
            },
        ),
        _add_turn_render(
            {"t": "gen", "in": {"image": src_img, "edits_ref": "turns[0].gt.edits"}, "gt": {"image": tgt_img}},
            "edit",
            {
                "query": {
                    "messages": [_msg("user", "text", "Edit the image following these edits."), _msg("user", "image", "Image #1"), _msg("user", "text", "Edits:\n" + "\n".join(f"- {e['edit']}" for e in edits))],
                    "images": [src_img] if src_img else [],
                },
                "answer": {"messages": [_msg("assistant", "image", "Image #1")], "images": [tgt_img] if tgt_img else []},
            },
        ),
        _add_turn_render(
            {"t": "select", "in": {"source_image": src_img, "edits_ref": "turns[0].gt.edits", "cand_ref": "eval.candidates.step1"}, "gt": {"choice_ref": "eval.answer.step1"}},
            "vlm_understand",
            _select_render(candidates, answer),
        ),
    ]
    task = _task_name(item.get("benchmark", ""))
    return {
        "id": item.get("unify_id", ""),
        "module": "unify",
        "benchmark": item.get("benchmark", ""),
        "task": task,
        "src": _src_base(item, task, [src_img, tgt_img], [src_pos, tgt_pos]),
        "turns": out_turns,
        "eval": {"candidates": {"step1": candidates}, "answer": {"step1": answer}},
        "tags": ["SEED-L3"],
        "meta": item.get("meta", {}),
    }


def _align_weave(item: Dict[str, Any]) -> Dict[str, Any]:
    turns = item.get("turns") if isinstance(item.get("turns"), dict) else {}
    understand = turns.get("understand", {}) if isinstance(turns.get("understand"), dict) else {}
    plan = turns.get("plan", {}) if isinstance(turns.get("plan"), dict) else {}
    qa = turns.get("qa_open", {}) if isinstance(turns.get("qa_open"), dict) else {}
    positions = _positions(item)
    src_pos = positions[0] if positions else 0
    tgt_pos = _target_pos(item) or (positions[1] if len(positions) > 1 else src_pos)
    src_img = _frame_img(item, src_pos)
    tgt_img = _frame_img(item, tgt_pos)
    src_caption = _frame_caption(item, src_pos)
    edits = [{"edit": str(e), "because": "", "so_that": "", "where": "", "check": ""} for e in plan.get("edits", [])]
    candidates, answer = _candidates(item, tgt_pos, context=[src_pos])
    understand_prompt = [_msg("user", "text", "Describe the image using only directly visible facts."), _msg("user", "image", "Image #1")]
    understand_prompt.extend(_nonempty_text_messages(src_caption and f"Caption: {src_caption}"))
    out_turns = [
        _add_turn_render(
            {"t": "understand", "in": {"image": src_img, "caption": src_caption}, "gt": {"facts": understand.get("facts", [])}},
            "vlm_understand",
            {
                "query": {"messages": understand_prompt, "images": [src_img] if src_img else []},
                "answer": {"messages": [_msg("assistant", "text", _json_text({"facts": understand.get("facts", [])}))], "images": []},
            },
        ),
        _add_turn_render(
            {"t": "plan", "in": {"image": src_img, "caption": src_caption, "goal": "Recall the visible later state."}, "gt": {"edits": edits}},
            "vlm_understand",
            {
                "query": {"messages": [_msg("user", "text", "You are given an image and a goal. Write a plan with invariants and edits."), _msg("user", "image", "Image #1"), _msg("user", "text", "Goal: Recall the visible later state.")], "images": [src_img] if src_img else []},
                "answer": {"messages": [_msg("assistant", "text", _json_text({"edits": edits}))], "images": []},
            },
        ),
        _add_turn_render(
            {"t": "gen", "in": {"image": src_img, "edits_ref": "turns[1].gt.edits"}, "gt": {"image": tgt_img}},
            "edit",
            {
                "query": {"messages": [_msg("user", "text", "Edit the image following these edits."), _msg("user", "image", "Image #1"), _msg("user", "text", "Edits:\n" + "\n".join(f"- {e['edit']}" for e in edits))], "images": [src_img] if src_img else []},
                "answer": {"messages": [_msg("assistant", "image", "Image #1")], "images": [tgt_img] if tgt_img else []},
            },
        ),
        _add_turn_render(
            {"t": "select", "in": {"cand_ref": "eval.candidates.step1"}, "gt": {"choice_ref": "eval.answer.step1"}},
            "vlm_understand",
            _select_render(candidates, answer),
        ),
    ]
    if qa.get("question") or qa.get("answer"):
        out_turns.append(
            _add_turn_render(
                {"t": "qa_open", "in": {"image_prev": src_img, "image": tgt_img, "q": qa.get("question", ""), "judge_rubric": qa.get("rubric", [])}, "gt": {"answer": qa.get("answer", "")}},
                "vlm_understand",
                {
                    "query": {"messages": [_msg("user", "text", qa.get("question", "")), _msg("user", "image", "Image #1"), _msg("user", "image", "Image #2")], "images": [x for x in [src_img, tgt_img] if x]},
                    "answer": {"messages": [_msg("assistant", "text", qa.get("answer", ""))], "images": []},
                },
            )
        )
    task = _task_name(item.get("benchmark", ""))
    return {
        "id": item.get("unify_id", ""),
        "module": "unify",
        "benchmark": item.get("benchmark", ""),
        "task": task,
        "src": _src_base(item, task, [src_img, tgt_img], [src_pos, tgt_pos]),
        "turns": out_turns,
        "eval": {"candidates": {"step1": candidates}, "answer": {"step1": answer}},
        "tags": ["WeaveBench"],
        "meta": item.get("meta", {}),
    }


def _align_comm_c1(item: Dict[str, Any]) -> Dict[str, Any]:
    turns = item.get("turns") if isinstance(item.get("turns"), dict) else {}
    positions = _positions(item)
    start_pos = positions[0] if positions else 0
    targets = _target_positions(item)
    start_img = _frame_img(item, start_pos)
    target_imgs = [_frame_img(item, p) for p in targets]
    interleaved: List[Dict[str, str]] = [{"type": "text", "content": turns.get("overall", "")}]
    for idx, step in enumerate(turns.get("steps", []) if isinstance(turns.get("steps"), list) else []):
        interleaved.append({"type": "text", "content": str(step)})
        if idx < len(target_imgs):
            interleaved.append({"type": "image", "content": target_imgs[idx]})
    answer_messages: List[Dict[str, str]] = []
    img_idx = 1
    for entry in interleaved:
        if entry.get("type") == "text":
            answer_messages.append(_msg("assistant", "text", entry.get("content", "")))
        elif entry.get("type") == "image":
            answer_messages.append(_msg("assistant", "image", f"Image #{img_idx}"))
            img_idx += 1
    step_candidates: Dict[str, List[Dict[str, str]]] = {}
    step_answers: Dict[str, str] = {}
    for idx, pos in enumerate(targets, start=1):
        cands, ans = _candidates(item, pos, context=[start_pos] + targets)
        step_candidates[f"step{idx}"] = cands
        step_answers[f"step{idx}"] = ans
    out_turns = [
        _add_turn_render(
            {"t": "steps", "in": {"question": turns.get("question", ""), "start_image": start_img}, "gt": {"interleaved": interleaved}},
            "interleaved_mm",
            {
                "query": {"messages": [_msg("user", "text", turns.get("question", "")), _msg("user", "image", "Image #1")], "images": [start_img] if start_img else []},
                "answer": {"messages": answer_messages, "images": [x for x in target_imgs if x]},
            },
        ),
        _add_turn_render(
            {"t": "select", "in": {"prompt": "Select the real image for each step.", "cand_ref": "eval.candidates"}, "gt": {"choice_ref": "eval.answer"}},
            "vlm_understand",
            _multi_select_render(step_candidates, step_answers),
        ),
    ]
    task = _task_name(item.get("benchmark", ""))
    return {
        "id": item.get("unify_id", ""),
        "module": "unify",
        "benchmark": item.get("benchmark", ""),
        "task": task,
        "src": _src_base(item, task, [start_img] + target_imgs, [start_pos] + targets),
        "turns": out_turns,
        "eval": {"candidates": step_candidates, "answer": step_answers},
        "tags": ["CoMM-Task4"],
        "meta": item.get("meta", {}),
    }


def _align_comm_c2(item: Dict[str, Any]) -> Dict[str, Any]:
    turns = item.get("turns") if isinstance(item.get("turns"), dict) else {}
    positions = _positions(item)
    earlier = positions[0] if len(positions) > 0 else 0
    middle = _target_pos(item) or (positions[1] if len(positions) > 1 else earlier)
    later = positions[2] if len(positions) > 2 else middle
    earlier_img = _frame_img(item, earlier)
    middle_img = _frame_img(item, middle)
    later_img = _frame_img(item, later)
    cands, ans = _candidates(item, middle, context=[earlier, later])
    out_turns = [
        _add_turn_render(
            {"t": "plan", "in": {"question": turns.get("question", ""), "image_earlier": earlier_img, "image_later": later_img}, "gt": {"explanation": turns.get("explanation", ""), "middle_image": middle_img}},
            "interleaved_mm",
            {
                "query": {"messages": [_msg("user", "text", "Given earlier and later real images, infer the missing middle step. Explain briefly, then output the middle image."), _msg("user", "image", "Image #1"), _msg("user", "image", "Image #2")], "images": [x for x in [earlier_img, later_img] if x]},
                "answer": {"messages": [_msg("assistant", "text", turns.get("explanation", "")), _msg("assistant", "image", "Image #1")], "images": [middle_img] if middle_img else []},
            },
        )
    ]
    task = _task_name(item.get("benchmark", ""))
    return {
        "id": item.get("unify_id", ""),
        "module": "unify",
        "benchmark": item.get("benchmark", ""),
        "task": task,
        "src": _src_base(item, task, [earlier_img, middle_img, later_img], [earlier, middle, later]),
        "turns": out_turns,
        "eval": {"candidates": {"step1": cands}, "answer": {"step1": ans}},
        "tags": ["CoMM-Task4"],
        "meta": item.get("meta", {}),
    }


def _align_isg(item: Dict[str, Any]) -> Dict[str, Any]:
    turns = item.get("turns") if isinstance(item.get("turns"), dict) else {}
    positions = _positions(item)
    context = positions[:2] if len(positions) >= 2 else positions
    targets = _target_positions(item)
    context_imgs = [_frame_img(item, p) for p in context]
    target_imgs = [_frame_img(item, p) for p in targets]
    edits: List[Dict[str, Any]] = []
    raw_edits = turns.get("edits") if isinstance(turns.get("edits"), list) else []
    because = turns.get("because") if isinstance(turns.get("because"), list) else []
    so_that = turns.get("so_that") if isinstance(turns.get("so_that"), list) else []
    check = turns.get("check") if isinstance(turns.get("check"), list) else []
    for idx, edit in enumerate(raw_edits):
        edits.append(
            {
                "step": idx + 1,
                "edit": str(edit),
                "because": str(because[idx]) if idx < len(because) else "",
                "so_that": str(so_that[idx]) if idx < len(so_that) else "",
                "check": str(check[idx]) if idx < len(check) else "",
            }
        )
    step_candidates: Dict[str, List[Dict[str, str]]] = {}
    step_answers: Dict[str, str] = {}
    for idx, pos in enumerate(targets, start=1):
        cands, ans = _candidates(item, pos, context=context + targets)
        step_candidates[f"step{idx}"] = cands
        step_answers[f"step{idx}"] = ans
    plan_messages = [_msg("user", "text", f"You are given {len(context_imgs)} context images. Write a K-step plan for future changes.")]
    for idx in range(len(context_imgs)):
        plan_messages.append(_msg("user", "image", f"Image #{idx + 1}"))
    gen_messages = [_msg("user", "text", f"Given the context images, output the next {len(target_imgs)} REAL images in order.")]
    for idx in range(len(context_imgs)):
        gen_messages.append(_msg("user", "image", f"Image #{idx + 1}"))
    gen_messages.append(_msg("user", "text", "Planned steps:\n" + "\n".join(f"- {e.get('edit', '')}" for e in edits)))
    out_turns = [
        _add_turn_render(
            {"t": "plan", "in": {"context_images": context_imgs}, "gt": {"overall": turns.get("overall", ""), "edits": edits}},
            "vlm_understand",
            {"query": {"messages": plan_messages, "images": [x for x in context_imgs if x]}, "answer": {"messages": [_msg("assistant", "text", _json_text({"overall": turns.get("overall", ""), "edits": edits}))], "images": []}},
        ),
        _add_turn_render(
            {"t": "gen", "in": {"context_images": context_imgs, "k": len(target_imgs), "plan_ref": "turns[0].gt.edits"}, "gt": {"images": target_imgs}},
            "edit",
            {"query": {"messages": gen_messages, "images": [x for x in context_imgs if x]}, "answer": {"messages": [_msg("assistant", "image", f"Image #{idx + 1}") for idx in range(len(target_imgs))], "images": [x for x in target_imgs if x]}},
        ),
        _add_turn_render(
            {"t": "select", "in": {"cand_ref": "eval.candidates"}, "gt": {"choice_ref": "eval.answer"}},
            "vlm_understand",
            _multi_select_render(step_candidates, step_answers),
        ),
    ]
    task = _task_name(item.get("benchmark", ""))
    return {
        "id": item.get("unify_id", ""),
        "module": "unify",
        "benchmark": item.get("benchmark", ""),
        "task": task,
        "src": _src_base(item, task, context_imgs + target_imgs, context + targets),
        "turns": out_turns,
        "eval": {"candidates": step_candidates, "answer": step_answers},
        "tags": ["ISG-Bench"],
        "meta": item.get("meta", {}),
    }


def _legacy_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("unify_id", "turns", "answer", "render", "evidence_refs"):
        if key in item:
            out[key] = copy.deepcopy(item.get(key))
    return out


def _with_legacy_payload(aligned: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, Any]:
    if aligned is original:
        return aligned
    out = dict(aligned)
    meta = copy.deepcopy(out.get("meta")) if isinstance(out.get("meta"), dict) else {}
    meta.setdefault("legacy_unify", _legacy_payload(original))
    out["meta"] = meta
    return out


def align_item(item: Dict[str, Any], profile: str = "legacy") -> Dict[str, Any]:
    if profile == "paper":
        from orchestrator.paper_render import render_paper_unified
        return render_paper_unified(item)
    benchmark = str(item.get("benchmark", "") or "")
    if benchmark.endswith("seed.l3.gts"):
        return _with_legacy_payload(_align_seed(item), item)
    if benchmark.endswith("weave.uie_select_recall"):
        return _with_legacy_payload(_align_weave(item), item)
    if benchmark.endswith("comm.task4.c1_steps"):
        return _with_legacy_payload(_align_comm_c1(item), item)
    if benchmark.endswith("comm.task4.c2_fill_gap"):
        return _with_legacy_payload(_align_comm_c2(item), item)
    if benchmark.endswith("isg.i2.seq_select"):
        return _with_legacy_payload(_align_isg(item), item)
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert T4 unify rows to per-turn benchmark render format.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = _load_jsonl(Path(args.input).resolve())
    aligned = [align_item(row) for row in rows]
    _write_jsonl(Path(args.output).resolve(), aligned)
    print(json.dumps({"input": args.input, "output": args.output, "rows": len(aligned)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
