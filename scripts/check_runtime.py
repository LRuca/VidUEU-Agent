#!/usr/bin/env python3
"""Offline runtime validation; real local OCR/OmniParser inference is opt-in."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect(profile, model_root=None, *, inference=False, device="cpu"):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True")
    result = {"profile": profile, "inference": inference, "ok": False}
    if profile in {"pipeline", "omniparser"}:
        from executor.tasks.ocr_runtime import paddle_device_info
        # OmniParser OCR stays on CPU; its vision models use the selected device.
        result["ocr"] = paddle_device_info(device if profile == "pipeline" else "cpu")
        from paddleocr import PaddleOCR
        import torch
        from transformers import AutoProcessor
        if profile == "omniparser":
            import supervision
            from executor.tasks.ultralytics_compat import disable_runtime_installation
            disable_runtime_installation()
            from ultralytics import YOLO
            disable_runtime_installation()
            from executor.tasks.omniparser_bridge import _load_local_florence2
    elif profile == "bagel":
        import torch
        import flash_attn
    elif profile == "vllm":
        import vllm
    if inference:
        if model_root is None or profile not in {"pipeline", "omniparser"}:
            raise ValueError("runtime_inference_requires_model_root_and_tool_profile")
        models = Path(model_root).resolve()
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (640, 256), "white")
        draw = ImageDraw.Draw(im)
        draw.text((30, 20), "SMOKE TEST 123", fill="black", font_size=36)
        draw.rectangle((200, 100, 350, 220), fill="red")
        with tempfile.TemporaryDirectory(prefix="vidueu_runtime_") as temporary:
            path = Path(temporary) / "fixture.png"
            im.save(path)
            config = {"models": {"paddleocr_det": str(models / "PP-OCRv5_server_det"),
                       "paddleocr_rec": str(models / "PP-OCRv5_server_rec")},
                       "tasks": {"T4_ocr": {"device": device, "lang": "en"}}}
            if profile == "pipeline":
                from executor.tasks.paddle_ocr import PaddleOCRExecutor
                executor = PaddleOCRExecutor(config)
                output = executor.run([str(path)])
                lines = output[path.name]
                if not lines:
                    raise RuntimeError("runtime_ocr_empty_fixture_result")
                result["recognized_text"] = [line["text"] for line in lines]
            else:
                from executor.tasks.omniparser import OmniParserExecutor
                config["models"]["omniparser"] = {
                    "repo_root": str(models / "OmniParser"),
                    "weights_dir": str(models / "OmniParser/weights"),
                    "caption_model_path": str(models / "OmniParser/weights/icon_caption"),
                    "caption_processor_path": str(models / "Florence-2-base"),
                    "caption_model_code_path": str(models / "Florence-2-base"),
                    "mode": "subprocess", "python_executable": sys.executable,
                    "device": device, "caption_batch_size": 1, "imgsz": 320,
                }
                output = OmniParserExecutor(config).run([str(path)])
                entry = output[str(path)]
                if not entry["elements"] or not entry["ocr_text"]:
                    raise RuntimeError("runtime_omniparser_empty_fixture_result")
                result["diagnostics"] = entry["diagnostics"]
                result["recognized_text"] = entry["ocr_text"]
                if not all(len(item["bbox"]) == 4 and item["content"] for item in entry["elements"]):
                    raise RuntimeError("runtime_omniparser_invalid_element")
                from executor.tasks.omniparser_bridge import run_batch
                no_text_config = OmniParserExecutor(config)._runtime_model_config()
                no_text_config["text_threshold"] = 1.01
                no_text = run_batch(str(models / "OmniParser"), no_text_config, [str(path)])[str(path)]
                assert no_text["ocr_text"] == [] and no_text["elements"]
                result["no_ocr_boxes"] = "passed"
                # Exercise both no-OCR and no-detection edge cases as well.
                from executor.tasks.omniparser_compat import load_official_utils
                utils = load_official_utils(str(models / "OmniParser"))
                import numpy as np
                import torch
                original = utils.predict_yolo
                utils.predict_yolo = lambda **kw: (torch.empty((0, 4)), torch.empty(0), [])
                try:
                    _, coords, elements = utils.get_som_labeled_img(
                        im, ocr_bbox=[], ocr_text=[], caption_model_processor=None)
                    assert coords == {} and elements == []
                finally:
                    utils.predict_yolo = original
                result["empty_detection"] = "passed"
    result["ok"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pipeline", "omniparser", "bagel", "vllm", "dev"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        report = inspect(args.profile, args.model_root, inference=args.inference, device=args.device)
    except Exception as exc:
        # No config or credential-bearing provider request is used here.
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "profile": args.profile, "error": str(exc)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
