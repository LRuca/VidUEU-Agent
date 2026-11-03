"""Real BAGEL inference and benchmark-native judging on frozen complete cases."""
from __future__ import annotations

import ast
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .artifacts import content_identity, publish_immutable_json
from .bagel_runtime import build_inferencer, infer_dialogue
from .jobs import file_fingerprint
from .state import digest, read_json, write_json

MMDU_DIMENSIONS = (
    "Creativity", "Richness", "Visual Perception", "Logical Coherence",
    "Answer Accuracy", "Image Relationship Understanding", "Overall Score",
)


def parse_score_object(text: str) -> dict[str, Any]:
    """Accept the official JSON or Python-dict format, without evaluating code."""
    decoder = json.JSONDecoder()
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
    # MMDU's native prompt explicitly requests a single-quoted dictionary.
    for start, char in enumerate(text):
        if char != "{":
            continue
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            try:
                value = ast.literal_eval(text[start:end])
                if isinstance(value, dict):
                    return value
            except (ValueError, SyntaxError):
                continue
    raise ValueError("judge_score_object_missing")


def score_number(value: Any, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("judge_score_must_be_numeric")
    if not math.isfinite(value) or not minimum <= value <= 10:
        raise ValueError("judge_score_out_of_range")
    return float(value)


def validate_mmdu(value: dict[str, Any]) -> dict[str, float]:
    scores = {key: score_number(value[key], minimum=1) for key in MMDU_DIMENSIONS}
    if any(not score.is_integer() for score in scores.values()):
        raise ValueError("mmdu_requires_integer_scores")
    return scores


def validate_vie(value: dict[str, Any]) -> dict[str, Any]:
    scores = value.get("score")
    if not isinstance(scores, list) or len(scores) != 2:
        raise ValueError("vie_requires_two_score_dimensions")
    return {**value, "score": [score_number(score) for score in scores]}


def validate_weave(value: dict[str, Any], *, accuracy: bool = False) -> dict[str, Any]:
    score = score_number(value.get("score"))
    if not score.is_integer() or (accuracy and score not in {0, 5, 10}):
        raise ValueError("invalid_weave_integer_score")
    return {**value, "score": score}


def aggregate_scores(benchmark: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    if not rows:
        raise ValueError("benchmark_scores_empty")
    if benchmark == "MMDU":
        parts = {key: sum(row[key] for row in rows) / len(rows) * 10 for key in MMDU_DIMENSIONS}
        return {"name": "aggregate", "value": parts["Overall Score"], "unit": "percent"}, parts
    if benchmark == "GEdit-Bench-En":
        parts = {key: sum(row[key] for row in rows) / len(rows) for key in ("G_SC", "G_PQ", "G_O")}
        return {"name": "G_O", "value": parts["G_O"], "unit": "score_0_10"}, parts
    if benchmark != "WEAVEbench":
        raise ValueError("unknown_native_benchmark")
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for metric, score in row.items():
            values[metric].append(score_number(score))
    # Match WEAVE summary.py: normalize/round category means before weighting.
    parts = {key: round(sum(scores) / len(scores) / 10, 3) for key, scores in values.items()}
    if all(key in parts for key in ("key_point", "visual_consistency", "image_quality")):
        weights = ({"key_point": .4, "visual_consistency": .1, "image_quality": .2, "accuracy": .3}
                   if "accuracy" in parts else {"key_point": .5, "visual_consistency": .2, "image_quality": .3})
        overall = sum(parts[key] * weight for key, weight in weights.items())
    else:
        overall = sum(parts.values()) / len(parts)
    return {"name": "Overall", "value": round(overall, 3), "unit": "fraction"}, parts


class PersistentJudge:
    def __init__(self, config: dict[str, Any]):
        from executor.models.vllm_vision_api import VLLMVisionApi
        self.client = VLLMVisionApi({**config, "retry_until_success": True})

    def score(self, path: Path, images: list[str], prompt: str,
              validate: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        request = {"images": [file_fingerprint(Path(p)) for p in images], "prompt": digest(prompt),
                   "model": self.client.model, "base_url": self.client.base_url}
        publish_immutable_json(path.with_suffix(".request.json"), request)
        if path.exists():
            return validate(read_json(path)["parsed"])
        attempt = 0
        while True:
            attempt += 1
            raw = self.client.chat(images, prompt, max_new_tokens=4096)
            attestation = dict(self.client.last_attestation or {})
            attempt_path = path.parent / f"{path.stem}.attempt_{attempt:04d}.json"
            # Preserve provider responses before parsing so retries are auditable.
            while attempt_path.exists():
                attempt += 1
                attempt_path = path.parent / f"{path.stem}.attempt_{attempt:04d}.json"
            write_json(attempt_path, {"raw": raw, "attestation": attestation})
            try:
                parsed = validate(parse_score_object(raw))
            except (ValueError, KeyError, TypeError) as exc:
                write_json(path.with_suffix(".pending.json"), {"status": "WAITING", "attempt": attempt,
                                                              "error_type": type(exc).__name__})
                time.sleep(min(30, 2 * attempt))
                continue
            write_json(path, {"parsed": parsed, "raw": raw, "attestation": attestation})
            write_json(path.with_suffix(".pending.json"), {"status": "RESOLVED", "attempt": attempt})
            return parsed

    def close(self):
        self.client.close()


def _resolve_images(ids: list[str], available: dict[str, str]) -> list[str]:
    missing = [key for key in ids if key not in available]
    if missing:
        raise ValueError("unavailable_image_reference:" + ",".join(missing))
    return [available[key] for key in ids]


def run_native_benchmark(*, benchmark: str, checkpoint: Path, checkpoint_digest: str,
                         subset: Path, output: Path, result: Path, runtime_config: dict[str, Any],
                         judge_config: dict[str, Any], decoding: dict[str, Any], prompt_file: Path) -> dict[str, Any]:
    definition = read_json(subset)
    cases = definition["cases"]
    case_ids = [str(case["id"]) for case in cases]
    if (definition["benchmark"] != benchmark or not cases or len(set(case_ids)) != len(case_ids)
            or any(not case.get("turns") for case in cases)):
        raise ValueError("invalid_complete_benchmark_subset")
    identity = content_identity(checkpoint, exclude=("export_manifest.json",))
    if identity["sha256"] != checkpoint_digest:
        raise ValueError("native_benchmark_checkpoint_changed")
    root = output.parent / "cases"
    publish_immutable_json(output.parent / "native_protocol.json", {
        "checkpoint": identity, "subset": content_identity(subset),
        "prompts": content_identity(prompt_file), "runtime": runtime_config, "decoding": decoding,
        "judge": {key: value for key, value in judge_config.items() if key not in {"api_key", "api_key_file"}},
        "media": [file_fingerprint(Path(path)) for path in definition["media_files"]],
    })
    prompts = read_json(prompt_file)
    inferencer, judge = None, PersistentJudge(judge_config)
    scored_rows, raw_cases = [], []
    seed = int(decoding.get("seed", 20261017))
    try:
        for case_index, case in enumerate(cases):
            # Gold target images never enter this inference map. A generated
            # binding becomes available only after its model output is saved.
            available = dict(case.get("input_images", {}))
            messages: list[dict[str, Any]] = []
            turn_results = []
            case_root = root / f"{case_index:04d}"
            for index, turn in enumerate(case["turns"]):
                turn_root = case_root / f"turn_{index:04d}"
                images = _resolve_images(turn.get("input_image_ids", []), available)
                messages.append({"role": "user", "text": turn["prompt"], "images": images})
                inferred_path = turn_root / "inference.json"
                if inferred_path.exists():
                    inferred = read_json(inferred_path)
                    if inferred["output_type"] != turn["output_type"]:
                        raise ValueError("cached_inference_modality_changed")
                    if turn["output_type"] == "image":
                        if file_fingerprint(Path(inferred["image"])) != inferred["image_identity"]:
                            raise ValueError("cached_generated_image_changed")
                else:
                    if inferencer is None:
                        inferencer = build_inferencer(Path(runtime_config["vendor_root"]), checkpoint,
                                                     image_size=int(decoding.get("image_size", 256)))
                    value = infer_dialogue(inferencer, messages, output_type=turn["output_type"],
                                           decoding=decoding, seed=seed + case_index * 1000 + index)
                    inferred = {"output_type": turn["output_type"], "checkpoint_digest": checkpoint_digest}
                    if isinstance(value, Image.Image):
                        turn_root.mkdir(parents=True, exist_ok=True)
                        image_path = turn_root / "generated.png"
                        value.save(image_path)
                        with Image.open(image_path) as saved:
                            saved.verify()
                        inferred.update(image=str(image_path.resolve()), image_identity=file_fingerprint(image_path))
                    else:
                        inferred["text"] = value
                    write_json(inferred_path, inferred)
                if turn["output_type"] == "image":
                    binding = str(turn["output_image_id"])
                    available[binding] = inferred["image"]
                    messages.append({"role": "assistant", "text": f"Image #{binding}", "images": [inferred["image"]]})
                else:
                    messages.append({"role": "assistant", "text": inferred["text"]})
                if benchmark == "MMDU":
                    prompt = (prompts["meta_prompt"] + "\n[Question]\n" + turn["prompt"]
                              + "\n[The Start of Reference Answer]\n" + turn["reference"]
                              + "\n[The End of Reference Answer]\n[The Start of Assistant’s Answer]\n"
                              + inferred["text"] + "\n[The End of Assistant’s Answer]")
                    scores = judge.score(turn_root / "judge.json", list(case["input_images"].values()), prompt, validate_mmdu)
                elif benchmark == "GEdit-Bench-En":
                    context = prompts["_context_no_delimit"]
                    sc_prompt = "\n".join([context, prompts["_prompts_0shot_two_image_edit_rule"],
                                             prompts["_prompts_0shot_tie_rule_SC"]]).replace("<instruction>", turn["prompt"])
                    sc = judge.score(turn_root / "judge_sc.json", images + [inferred["image"]], sc_prompt, validate_vie)
                    pq_prompt = "\n".join([context, prompts["_prompts_0shot_rule_PQ"]])
                    pq = judge.score(turn_root / "judge_pq.json", [inferred["image"]], pq_prompt, validate_vie)
                    scores = {"G_SC": min(sc["score"]), "G_PQ": min(pq["score"])}
                    scores["G_O"] = math.sqrt(scores["G_SC"] * scores["G_PQ"])
                else:
                    scores = {}
                    if turn["output_type"] == "text":
                        prompt = prompts["prompt_txt_acc"].format(standard_answer=turn["reference"],
                                                                 generated_answer=inferred["text"])
                        value = judge.score(turn_root / "judge_accuracy.json", [], prompt,
                                            lambda row: validate_weave(row, accuracy=True))
                        scores["accuracy"] = value["score"]
                    else:
                        for metric, prompt_key in (("key_point", "prompt_keypoint_checklist"),
                                                   ("visual_consistency", "prompt_visual_consistency"),
                                                   ("image_quality", "prompt_image_quality")):
                            refs = []
                            requirements = []
                            if metric != "image_quality":
                                for requirement in turn.get("requirements", []):
                                    requirements.append(requirement["text"])
                                    if requirement.get("image"):
                                        refs.append(requirement["image"])
                                    elif requirement.get("generated_image_id"):
                                        refs.extend(_resolve_images([requirement["generated_image_id"]], available))
                            prompt = prompts[prompt_key].replace("<image>", "Image 1 (generated output)")
                            prompt += "\n" + "\n".join(requirements)
                            if refs:
                                prompt += "\nFollowing images 2 onwards are the reference images, in requirement order."
                            value = judge.score(turn_root / f"judge_{metric}.json", [inferred["image"]] + refs,
                                                prompt, validate_weave)
                            scores[metric] = value["score"]
                scored_rows.append(scores)
                turn_results.append({"turn": index, "inference": inferred, "scores": scores})
                write_json(case_root / "progress.json", {"id": case["id"], "expected_turns": len(case["turns"]),
                                                        "scored_turns": len(turn_results), "turns": turn_results})
            raw_cases.append({"id": case["id"], "turns": turn_results})
    finally:
        judge.close()
    metric, subcategory_scores = aggregate_scores(benchmark, scored_rows)
    write_json(output, {"benchmark": benchmark, "cases": raw_cases, "simulated": False})
    value = {
        "benchmark": benchmark, "simulated": False, "checkpoint_digest": checkpoint_digest,
        "case_ids": case_ids, "expected": len(cases), "inferred": len(cases), "scored": len(cases), "failed": 0,
        "expected_turns": sum(len(case["turns"]) for case in cases), "scored_turns": len(scored_rows),
        "metric": metric, "subcategory_scores": subcategory_scores,
        "judge_identity": {"model": judge.client.model, "base_url": judge.client.base_url},
        "protocol_digest": digest(read_json(output.parent / "native_protocol.json")),
        "model_identity": inferencer.runtime_identity if inferencer else {"cached": True, "checkpoint": str(checkpoint)},
    }
    write_json(result, value)
    return value
