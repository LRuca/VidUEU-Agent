from __future__ import annotations

import json
import re
import traceback
from typing import Any, Dict, List, cast

from executor.tasks.prompt_loader import load_prompt_text


EXPECTED_KEYS = (
    "exposure_brightness",
    "focus_sharpness",
    "motion_blur_camera_shake_ghosting",
    "noise_grain_low_light_degradation",
    "artifacts_compression_resampling_ai_generated",
    "color_fidelity_white_balance_color_cast",
    "contrast_dynamic_range_detail_visibility",
    "occlusion_obstruction_crop_integrity",
    "corruption_abnormal_regions_overlay_contamination",
)


QUALITY_PROMPT = """You are a strict technical image quality inspector for aggressive dataset filtering.

Your task is to inspect the input image and determine, for each quality dimension below, whether it passes or fails a strict high-quality standard.

Important rules:
- Judge only technical image quality and cleanliness, not aesthetics. This is strict high-quality dataset filtering, so be conservative.
- False pass is worse than false fail; if uncertain, choose fail.
- Recognizable content does not justify pass.
- Any clearly visible defect in any meaningful region, including local mild or moderate defects, should usually cause fail for that dimension.
- Evaluate all dimensions independently.
- If one dimension fails, do not automatically fail any other dimension; mark fail only for dimensions with direct visual evidence of that specific problem.
- For each dimension, write exactly one short reason sentence.
- Also produce one short factual caption describing the visible scene or subject for downstream LLM scoring.
- Output exactly 9 reason lines first, then the final JSON as the last part of the response.
- Do not use markdown code fences.
- In the final JSON, the "caption" value must be a short English string, and every quality dimension value must be exactly "pass" or "fail".

Inspect the image using the following dimensions and rules:

1. exposure_brightness
Pass only if:
- The image is well exposed for strict dataset use.
- Important content is clearly visible.
- There is no clearly visible highlight clipping in meaningful regions.
- There is no clearly visible shadow clipping hiding meaningful content.
Fail if any of the following is true:
- The image is visibly too bright or too dark.
- Important content loses visible detail because of overexposure or underexposure.
- Highlights or shadows are visibly clipped in a way that reduces image quality.
- Exposure is visibly unstable, harsh, dim, or compromised.

2. focus_sharpness
Pass only if:
- Important content is clearly in focus.
- Edges and structures look crisp enough for strict dataset filtering.
- There is no visible defocus softness in meaningful content.
Fail if any of the following is true:
- The image is visibly soft or out of focus.
- Important content is not crisp enough.
- Defocus blur or overall softness is visible.
- Fine structures are not clearly resolved.

3. motion_blur_camera_shake_ghosting
Pass only if:
- Important content is clean and stable.
- There is no visible motion blur, camera shake blur, double edge, or ghosting in meaningful regions.
Fail if any of the following is true:
- Any visible motion blur appears in meaningful content.
- Any visible camera shake blur appears.
- Any visible double edge or ghosting appears.
- Partial motion-related degradation is visible and reduces dataset quality.

4. noise_grain_low_light_degradation
Pass only if:
- Noise is minimal or barely perceptible.
- Fine details remain clean.
- There is no visible low-light degradation that meaningfully affects quality.
Fail if any of the following is true:
- Noise or grain is clearly visible.
- Chroma noise, luminance noise, or low-light grain is noticeable.
- Fine detail is weakened by noise.
- The image has a visibly dirty, rough, grainy, or degraded low-light appearance.

5. artifacts_compression_resampling_ai_generated
Pass only if:
- The image is technically clean.
- There is no visible JPEG blocking, ringing, aliasing, pixelation, oversharpening halo, resampling artifact, or AI-generated artifact.
- Local structures are coherent and visually plausible.
Fail if any of the following is true:
- Compression or resampling artifacts are clearly visible.
- Pixelation, ringing, aliasing, halo, blockiness, or sharpening artifacts are visible.
- There are visible AI-generated artifacts or locally implausible details.
- The image looks visibly processed, synthetic, melted, duplicated, broken, or structurally inconsistent in meaningful regions.

6. color_fidelity_white_balance_color_cast
Pass only if:
- Colors look technically reasonable and stable.
- White balance is acceptable for strict dataset use.
- There is no clearly visible problematic color cast.
Fail if any of the following is true:
- A visible yellow, blue, green, magenta, or other problematic color cast exists.
- White balance visibly looks wrong.
- Colors look technically distorted, unstable, or contaminated.
- Color reproduction is visibly off enough to reduce dataset quality.

7. contrast_dynamic_range_detail_visibility
Pass only if:
- Important details are clearly visible.
- Contrast and dynamic range are sufficient for strict dataset filtering.
- The image does not look visibly washed out, muddy, flat, or excessively harsh.
Fail if any of the following is true:
- Important detail is visibly lost or weakened.
- The image looks visibly washed out, flat, muddy, or excessively harsh.
- Dynamic range is poor enough to reduce technical quality.
- Detail visibility is compromised in highlights, shadows, or midtones.

8. occlusion_obstruction_crop_integrity
Pass only if:
- Important content is intact.
- Framing does not accidentally damage useful content.
- No obstruction meaningfully hurts image quality.
Fail if any of the following is true:
- Meaningful content is blocked, cut off, or poorly cropped.
- Finger, object, lens obstruction, accidental foreground blockage, or framing damage is visible.
- The image is technically damaged by obstruction or bad cropping.

9. corruption_abnormal_regions_overlay_contamination
Pass only if:
- The image is clean and intact.
- There is no visible corruption, abnormal region, missing patch, strong banding, severe flare contamination, severe haze/fog contamination, watermark contamination, or disruptive text overlay.
Fail if any of the following is true:
- Any corrupted, abnormal, missing, broken, or contaminated region is visible.
- Strong flare, fogging, haze, dirt, watermark, or overlay interferes with image quality.
- Banding, contamination, or image integrity failure is clearly visible in meaningful regions.

Output format:
- First output exactly 9 plain-text lines, one line per dimension.
- Each line must contain the dimension name, a colon, and exactly one short sentence explaining that dimension's judgment.
- After those 9 lines, output exactly one JSON object as the final part of the response.
- The final JSON must also include a top-level "caption" field containing one concise natural-language caption in English.
- Do not output any extra text before the 9 lines, between the 9 lines and the JSON, or after the JSON.

The 9 lines must follow this order:
1. exposure_brightness
2. focus_sharpness
3. motion_blur_camera_shake_ghosting
4. noise_grain_low_light_degradation
5. artifacts_compression_resampling_ai_generated
6. color_fidelity_white_balance_color_cast
7. contrast_dynamic_range_detail_visibility
8. occlusion_obstruction_crop_integrity
9. corruption_abnormal_regions_overlay_contamination

The final JSON must use exactly these keys:

{
  "caption": "A concise factual caption of the visible scene.",
  "exposure_brightness": "pass",
  "focus_sharpness": "pass",
  "motion_blur_camera_shake_ghosting": "pass",
  "noise_grain_low_light_degradation": "pass",
  "artifacts_compression_resampling_ai_generated": "pass",
  "color_fidelity_white_balance_color_cast": "pass",
  "contrast_dynamic_range_detail_visibility": "pass",
  "occlusion_obstruction_crop_integrity": "pass",
  "corruption_abnormal_regions_overlay_contamination": "pass"
}"""


def extract_first_json_object(text: str) -> str:
    stripped = text.strip()

    fenced_matches = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_matches:
        return fenced_matches[-1].strip()

    starts = [m.start() for m in re.finditer(r"\{", stripped)]
    if not starts:
        raise ValueError("No JSON object start '{' found in model output.")

    last_valid: str | None = None
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(stripped)):
            ch = stripped[index]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : index + 1]
                    try:
                        json.loads(candidate)
                        last_valid = candidate
                    except Exception:
                        pass
                    break

    if last_valid is None:
        raise ValueError("No complete JSON object found in model output.")

    return last_valid


def normalize_result_dict(obj: Dict[str, Any]) -> Dict[str, str]:
    key_aliases = {
        "exposure_brightness": "exposure_brightness",
        "focus_sharpness": "focus_sharpness",
        "motion_blur_camera_shake_ghosting": "motion_blur_camera_shake_ghosting",
        "noise_grain_low_light_degradation": "noise_grain_low_light_degradation",
        "artifacts_compression_resampling_ai_generated": "artifacts_compression_resampling_ai_generated",
        "color_fidelity_white_balance_color_cast": "color_fidelity_white_balance_color_cast",
        "contrast_dynamic_range_detail_visibility": "contrast_dynamic_range_detail_visibility",
        "occlusion_obstruction_crop_integrity": "occlusion_obstruction_crop_integrity",
        "corruption_abnormal_regions_overlay_contamination": "corruption_abnormal_regions_overlay_contamination",
        "corruption_abnormal_regions_overlay": "corruption_abnormal_regions_overlay_contamination",
        "corruption_abnormal_regions": "corruption_abnormal_regions_overlay_contamination",
    }

    remapped: Dict[str, Any] = {}
    for raw_key, raw_value in obj.items():
        mapped_key = key_aliases.get(raw_key)
        if mapped_key is not None:
            remapped[mapped_key] = raw_value

    normalized: Dict[str, str] = {}
    for key in EXPECTED_KEYS:
        if key not in remapped:
            raise KeyError(f"Missing expected key: {key}")
        value = str(remapped[key]).strip().lower()
        if value not in {"pass", "fail"}:
            raise ValueError(f"Invalid value for key {key}: {remapped[key]!r}")
        normalized[key] = value
    return normalized


def extract_caption_from_obj(obj: Dict[str, Any]) -> str:
    caption = str(obj.get("caption", "") or "").strip()
    return caption


def parse_model_response_to_result(text: str) -> Dict[str, Any]:
    try:
        json_text = extract_first_json_object(text)
        obj = json.loads(json_text)
        if not isinstance(obj, dict):
            raise TypeError("Parsed JSON is not an object.")
        obj = cast(Dict[str, Any], obj)
        return {
            "parsed_result": normalize_result_dict(obj),
            "caption": extract_caption_from_obj(obj),
        }
    except Exception:
        pass

    line_based: Dict[str, str] = {}
    for key in EXPECTED_KEYS:
        pattern = rf"{re.escape(key)}\s*[:：]\s*(.*?)(pass|fail)\b"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            line_based[key] = match.group(2).strip().lower()

    if len(line_based) == len(EXPECTED_KEYS):
        caption_match = re.search(r'caption\s*[:：]\s*(.+)', text, flags=re.IGNORECASE)
        return {
            "parsed_result": line_based,
            "caption": str(caption_match.group(1) if caption_match is not None else "").strip(),
        }

    raise ValueError("Failed to parse either JSON output or line-based pass/fail output.")


def build_quality_result(parsed: Dict[str, str]) -> Dict[str, Any]:
    failed_dimensions = [key for key in EXPECTED_KEYS if parsed[key] == "fail"]
    passed_dimensions = [key for key in EXPECTED_KEYS if parsed[key] == "pass"]
    return {
        "parsed_result": parsed,
        "failed_dimensions": failed_dimensions,
        "passed_dimensions": passed_dimensions,
        "failed_count": len(failed_dimensions),
        "passed_count": len(passed_dimensions),
        "all_pass": len(failed_dimensions) == 0,
    }


def _resolve_quality_prompt() -> str:
    try:
        return load_prompt_text("quality_toolcall")
    except Exception:
        return QUALITY_PROMPT


def compute_vlm_quality_scores(
    image_paths: List[str],
    vlm: Any = None,
    model_manager: Any = None,
    prompt: str | None = None,
    max_new_tokens: int = 768,
) -> List[Dict[str, Any]]:
    prompt_text = str(prompt or _resolve_quality_prompt())

    def _run_single(single_vlm: Any, image_path: str) -> Dict[str, Any]:
        raw_response = ""
        try:
            raw_response = str(
                single_vlm.chat(
                    images=[image_path],
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                )
            )
            parsed_payload = parse_model_response_to_result(raw_response)
            parsed = cast(Dict[str, str], parsed_payload.get("parsed_result") or {})
            result = build_quality_result(parsed)
            result["caption"] = str(parsed_payload.get("caption", "") or "")
            result["raw_response"] = raw_response
            result["available"] = True
            result["error"] = ""
            return result

        except Exception as exc:
            tb = traceback.format_exc()

            print(f"[quality_iqa] VLM failed on image: {image_path}")
            print(f"[quality_iqa] exception_type: {type(exc).__name__}")
            print(f"[quality_iqa] exception: {exc}")
            print("[quality_iqa] traceback:")
            print(tb)
            if raw_response:
                print("[quality_iqa] raw_response_before_failure:")
                print(raw_response)

            return {
                "parsed_result": None,
                "failed_dimensions": ["parse_or_inference_error"],
                "passed_dimensions": [],
                "failed_count": len(EXPECTED_KEYS),
                "passed_count": 0,
                "all_pass": False,
                "caption": "",
                "raw_response": raw_response,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "image_path": image_path,
                "exception_type": type(exc).__name__,
            }

    if model_manager is not None:
        return model_manager.run_vlm_jobs(
            image_paths,
            worker_fn=lambda single_vlm, image_path: _run_single(single_vlm, image_path),
        )
    if vlm is None:
        raise ValueError("compute_vlm_quality_scores requires either vlm or model_manager")
    return [_run_single(vlm, image_path) for image_path in image_paths]
