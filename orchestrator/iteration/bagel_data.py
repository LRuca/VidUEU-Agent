"""Convert selected paper samples to the existing BAGEL multiturn contract."""
from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .jobs import file_fingerprint
from .paper_rounds import validate_paper_rounds
from .state import read_jsonl, write_json, write_jsonl
from .taxonomy import TASK_CAPABILITY, TASK_TYPES, canonical_task

_MODES = {"vlm_understand", "select_mcq", "plan_struct", "t2i", "edit", "interleaved_mm"}
_IMAGE_REF = re.compile(r"Image #(\d+)$")
# Dialog producers use both the explicit ``Image1: <ImageHere>`` form and the
# BAGEL-native bare ``<ImageHere>`` form.  The latter is bound in encounter
# order to the next image that has not already been introduced in the dialog.
_PLACEHOLDER = re.compile(
    r"(?:(?:Image\s*(\d+)\s*:\s*)?<ImageHere>)", re.IGNORECASE
)


def _message(role: str, kind: str, content: str) -> dict[str, str]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty_message")
    return {"role": role, "type": kind, "content": content}


def _side(role: str, text: str, images: list[str]) -> dict[str, Any]:
    messages = [_message(role, "text", text)] if text else []
    messages.extend(_message(role, "image", f"Image #{i + 1}") for i in range(len(images)))
    return {"messages": messages, "images": images}


def _turn(mode: str, question: str, inputs: list[str], answer: str = "",
          outputs: list[str] | None = None) -> dict[str, Any]:
    return {"task_type": mode, "render": {
        "query": _side("user", question, inputs),
        "answer": _side("assistant", answer, outputs or []),
    }}


def _dialog_turns(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    images = [row["image"] if isinstance(row, dict) else row for row in candidate["images"]]
    raw = candidate["turns"]
    if not isinstance(raw, list) or not raw or len(raw) % 2:
        raise ValueError("dialog_requires_complete_qa_pairs")
    result, introduced = [], set()
    next_auto = 1
    for index in range(0, len(raw), 2):
        user, assistant = raw[index:index + 2]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError("dialog_role_order")
        text = user["text"]
        messages, turn_images, cursor = [], [], 0
        for match in _PLACEHOLDER.finditer(text):
            if match.group(1) is None:
                while next_auto in introduced and next_auto <= len(images):
                    next_auto += 1
                number = next_auto
                next_auto += 1
            else:
                number = int(match.group(1))
            if number < 1 or number > len(images):
                raise ValueError("dialog_image_reference_out_of_range")
            if text[cursor:match.start()].strip():
                messages.append(_message("user", "text", text[cursor:match.start()]))
            turn_images.append(images[number - 1])
            introduced.add(number)
            messages.append(_message("user", "image", f"Image #{len(turn_images)}"))
            cursor = match.end()
        if _PLACEHOLDER.search(text[cursor:]):
            raise ValueError("dialog_unbound_image_placeholder")
        if text[cursor:].strip():
            messages.append(_message("user", "text", text[cursor:]))
        if not messages:
            raise ValueError("empty_dialog_query")
        result.append({"task_type": "vlm_understand", "render": {
            "query": {"messages": messages, "images": turn_images},
            "answer": _side("assistant", assistant["text"], []),
        }})
    if introduced != set(range(1, len(images) + 1)):
        raise ValueError("dialog_image_not_introduced")
    return result


def _candidate_turns(candidate: dict[str, Any], task: str) -> list[dict[str, Any]]:
    raw = candidate.get("turns")
    if isinstance(raw, list) and raw and all(isinstance(t, dict) and "render" in t for t in raw):
        turns = copy.deepcopy(raw)
        for turn in turns:
            turn["task_type"] = turn.get("task_type") or turn.get("mode")
        return turns
    if task == "fine_grained_perception":
        answer = candidate.get("_answer")
        if not isinstance(answer, str) or answer not in {"A", "B", "C", "D"}:
            raise ValueError("invalid_mcq_answer")
        options = [f"{letter}. {candidate[letter]}" for letter in "ABCD"]
        query = candidate["question"] + "\n" + "\n".join(options) + "\nReply with the option letter only."
        return [_turn("select_mcq", query, [candidate["image"]], answer)]
    if task == "single_image_reasoning":
        return [_turn("vlm_understand", candidate["question"], [candidate["image"]], candidate["gpt_answer"])]
    if task == "multi_image_logic":
        return _dialog_turns(candidate)
    if task == "localized_editing":
        return [_turn("edit", candidate["instruction"], [candidate["source_img"]],
                      outputs=[candidate["target_img"]])]
    # Only render an existing grounded task; never invent a missing target.
    from orchestrator.paper_render import render_paper_unified

    return _candidate_turns(render_paper_unified(candidate), task)


def _validate_side(side: dict[str, Any], role: str, asset_root: Path,
                   media: dict[str, dict[str, Any]]) -> None:
    if not isinstance(side, dict) or not isinstance(side.get("messages"), list):
        raise ValueError("render_side_requires_messages")
    paths = side.get("images")
    if not isinstance(paths, list):
        raise ValueError("render_side_requires_images")
    resolved = []
    for name in paths:
        if not isinstance(name, str) or not name:
            raise ValueError("invalid_media_path")
        path = Path(name)
        path = (asset_root / path).resolve() if not path.is_absolute() else path.resolve()
        key = str(path)
        if key not in media:
            from PIL import Image

            fingerprint = file_fingerprint(path)
            with Image.open(path) as img:
                img.verify()
            media[key] = fingerprint
        resolved.append(key)
    side["images"] = resolved
    references = []
    for message in side["messages"]:
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError("render_role_mismatch")
        if message.get("type") not in ("text", "image"):
            raise ValueError("unsupported_message_type")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("empty_render_content")
        if message["type"] == "image":
            match = _IMAGE_REF.fullmatch(message["content"])
            if not match:
                raise ValueError("invalid_image_reference")
            references.append(int(match.group(1)))
    if references != list(range(1, len(resolved) + 1)):
        raise ValueError("image_references_must_match_ordered_media")


def validated_turns(candidate: dict[str, Any], task: str, asset_root: Path,
                    media: dict[str, dict[str, Any]], *,
                    require_paper_rounds: bool = False) -> list[dict[str, Any]]:
    """Use the same validation for source admission and final train publication."""
    turns = _candidate_turns(candidate, task)
    if not turns:
        raise ValueError("empty_episode")
    if require_paper_rounds:
        validate_paper_rounds(turns, task)
    for index, turn in enumerate(turns, 1):
        turn["turn_id"] = index
        if turn["task_type"] not in _MODES:
            raise ValueError("unsupported_bagel_mode")
        render = turn["render"]
        for side, role in (("query", "user"), ("answer", "assistant")):
            _validate_side(render[side], role, asset_root, media)
        answer = render["answer"]
        has_image = bool(answer["images"])
        has_text = any(m["type"] == "text" for m in answer["messages"])
        if not (has_image or has_text):
            raise ValueError("missing_training_target")
        mode = turn["task_type"]
        if mode in {"edit", "t2i"} and (len(answer["images"]) != 1 or has_text):
            raise ValueError("generation_mode_requires_one_image_target")
        if mode in {"vlm_understand", "plan_struct", "select_mcq"} and (has_image or not has_text):
            raise ValueError("text_mode_requires_text_target")
        if has_image:
            inputs = {media[p]["sha256"] for p in render["query"]["images"]}
            if any(media[p]["sha256"] in inputs for p in answer["images"]):
                raise ValueError("current_target_is_also_current_input")
        if str(turn.get("t", "")).startswith("select"):
            hashes = [media[p]["sha256"] for p in render["query"]["images"]]
            if len(hashes) != len(set(hashes)):
                raise ValueError("paper_selection_requires_distinct_media_content")
    if require_paper_rounds:
        # Validate again after absolute-path resolution so aliases cannot hide
        # duplicate candidates or break target/selection identity.
        validate_paper_rounds(turns, task)
    return turns


def convert_manifest(manifest: Path, output: Path, *, asset_root: Path,
                     require_nine_tasks: bool = True,
                     require_paper_rounds: bool = False) -> dict[str, Any]:
    """Validate all records before publishing; never mutate source candidates."""
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError("empty_train_manifest")
    episodes, media, seen = [], {}, set()
    for line, row in enumerate(rows, 1):
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"manifest_requires_candidate:{line}")
        key = row.get("record_key")
        task = canonical_task(row.get("task_type") or candidate.get("task_type") or candidate.get("benchmark"))
        if not key or str(key) in seen or not task:
            raise ValueError(f"invalid_or_duplicate_record_key_or_task:{line}")
        seen.add(str(key))
        capability = TASK_CAPABILITY[task]
        if row.get("capability", capability) != capability:
            raise ValueError("task_capability_mismatch")
        row_asset_root = Path(row.get("asset_root") or candidate.get("asset_root") or asset_root).resolve()
        turns = validated_turns(candidate, task, row_asset_root, media,
                                require_paper_rounds=require_paper_rounds)
        provenance = {k: copy.deepcopy(row.get(k)) for k in (
            "source_pointer", "source_key", "source_sha256", "iteration", "policy_domain")}
        for name in ("source_file", "source_line", "source_row_sha256", "source_file_sha256",
                     "media_root", "source_origin", "source_split", "source_audit"):
            provenance[name] = copy.deepcopy(row.get(name, candidate.get(name)))
        provenance.update(input_manifest=str(manifest.resolve()), input_manifest_line=line,
                          input_row=candidate.get("input_row"), record_key=key,
                          asset_root=str(row_asset_root),
                          review_provenance=copy.deepcopy(candidate.get("review_provenance")))
        episodes.append({"id": str(key), "task_type": task, "capability": capability,
                         "turns": turns, "context_mode": "ordered_dialogue", "provenance": provenance})
    task_counts = Counter(e["task_type"] for e in episodes)
    missing = sorted(set(TASK_TYPES) - set(task_counts))
    if missing and require_nine_tasks:
        raise ValueError("missing_train_tasks:" + ",".join(missing))
    report = {"schema_version": 1, "episode_count": len(episodes),
              "paper_round_contract_checked": require_paper_rounds,
              "turn_count": sum(len(e["turns"]) for e in episodes),
              "task_counts": dict(task_counts), "missing_task_types": missing,
              "capability_counts": dict(Counter(e["capability"] for e in episodes)),
              "media": list(media.values()), "source_manifest": file_fingerprint(manifest),
              "coverage_scope": "format_and_media_only_not_training_consumption"}
    write_jsonl(output, episodes)
    report["output"] = file_fingerprint(output)
    write_json(output.with_suffix(".report.json"), report)
    return report
