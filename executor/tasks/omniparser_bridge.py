from __future__ import annotations

import argparse
import io
import importlib.util
import json
import os
import sys
import types
from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


_STATE: Dict[str, Any] = {"key": None, "model": None, "caption": None}


def _import_official_utils(repo_root: str):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from executor.tasks.omniparser_compat import load_official_utils
        module = load_official_utils(repo_root)
    except Exception as exc:  # pragma: no cover - runtime import depends on external repo
        raise RuntimeError(
            f"Failed to import OmniParser official utils from repo_root={repo_root}: {exc}"
        ) from exc
    return module.check_ocr_box, module.get_caption_model_processor, module.get_som_labeled_img, module.get_yolo_model


def _build_ocr(repo_root, config):
    """Use managed local Paddle 3.x assets; never download models on import."""
    if not bool(config.get("use_paddleocr", True)):
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, download_enabled=False,
                                model_storage_directory=config.get("easyocr_model_dir"))
        return lambda image, threshold: reader.readtext(image, text_threshold=threshold)
    from executor.tasks.ocr_runtime import paddle_device_info
    from paddleocr import PaddleOCR
    device = paddle_device_info(config.get("ocr_device", "cpu"))["device"]
    model_root = Path(repo_root).resolve().parent
    det = Path(config.get("ocr_det_path") or model_root / "PP-OCRv5_server_det")
    rec = Path(config.get("ocr_rec_path") or model_root / "PP-OCRv5_server_rec")
    for root in (det, rec):
        for name in ("inference.json", "inference.pdiparams"):
            if not (root / name).is_file():
                raise FileNotFoundError(f"omniparser_ocr_asset_missing: {root / name}")
    ocr = PaddleOCR(text_detection_model_dir=str(det), text_recognition_model_dir=str(rec),
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, device=device, enable_mkldnn=False)

    def read(image, threshold):
        result = []
        for item in ocr.predict(image):
            for polygon, text, score in zip(item["rec_polys"], item["rec_texts"], item["rec_scores"]):
                if float(score) >= threshold:
                    result.append((polygon, str(text), float(score)))
        return result
    return read


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("utf-8")


def _load_models(repo_root: str, model_config: Dict[str, Any]):
    key = (
        os.path.abspath(repo_root),
        str(model_config.get("som_model_path", "") or ""),
        str(model_config.get("caption_model_path", "") or ""),
        str(model_config.get("caption_processor_path", "") or ""),
        str(model_config.get("caption_model_code_path", "") or ""),
        str(model_config.get("caption_model_name", "florence2") or "florence2"),
        str(model_config.get("device", "") or ""),
    )
    if _STATE.get("key") == key and _STATE.get("model") is not None and _STATE.get("caption") is not None:
        return _STATE["model"], _STATE["caption"]

    check_ocr_box, get_caption_model_processor, get_som_labeled_img, get_yolo_model = _import_official_utils(repo_root)
    del check_ocr_box, get_som_labeled_img

    som_model_path = str(model_config.get("som_model_path", "") or "").strip()
    caption_model_path = str(model_config.get("caption_model_path", "") or "").strip()
    caption_model_name = str(model_config.get("caption_model_name", "florence2") or "florence2").strip()
    device = str(model_config.get("device", "") or "").strip() or None

    if not som_model_path:
        raise ValueError("OmniParser requires models.omniparser.som_model_path")
    if not caption_model_path:
        raise ValueError("OmniParser requires models.omniparser.caption_model_path")

    som_model = get_yolo_model(model_path=som_model_path)
    if caption_model_name.lower() == "florence2":
        caption_processor_path = str(
            model_config.get("caption_processor_path") or caption_model_path
        ).strip()
        caption_model_code_path = str(
            model_config.get("caption_model_code_path") or caption_model_path
        ).strip()
        caption_model_processor = _load_local_florence2(
            caption_model_path,
            processor_path=caption_processor_path,
            code_path=caption_model_code_path,
            device=device,
        )
    else:
        caption_model_processor = get_caption_model_processor(
            model_name=caption_model_name,
            model_name_or_path=caption_model_path,
            device=device,
        )
    _STATE.update({"key": key, "model": som_model, "caption": caption_model_processor})
    return som_model, caption_model_processor


def _load_local_florence2(
    model_path: str,
    *,
    processor_path: str | None = None,
    code_path: str | None = None,
    device: str | None = None,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if str(device).lower() == "cpu" else torch.float16
    processor = AutoProcessor.from_pretrained(
        processor_path or model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config_cls, model_cls = _load_florence2_classes(code_path or model_path)
    config = config_cls.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = "eager"
    model = model_cls.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device)
    _patch_florence2_generation_compat(model)
    return {"model": model, "processor": processor}


def _patch_florence2_generation_compat(model: Any) -> None:
    language_model = getattr(model, "language_model", None)
    prepare = getattr(language_model, "prepare_inputs_for_generation", None)
    if language_model is None or prepare is None or getattr(language_model, "_agentpack_generation_patch", False):
        return

    def _is_empty_past_key_values(value: Any) -> bool:
        try:
            return value is not None and len(value) > 0 and value[0] and value[0][0] is None
        except Exception:
            return False

    def patched_prepare_inputs_for_generation(decoder_input_ids, past_key_values=None, **kwargs):
        if _is_empty_past_key_values(past_key_values):
            past_key_values = None
        return prepare(
            decoder_input_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    language_model.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
    language_model._agentpack_generation_patch = True


def _load_florence2_classes(code_path: str):
    code_dir = Path(code_path).expanduser().resolve()
    config_file = code_dir / "configuration_florence2.py"
    model_file = code_dir / "modeling_florence2.py"
    if not config_file.is_file() or not model_file.is_file():
        raise FileNotFoundError(
            "Florence2 local code requires configuration_florence2.py and "
            f"modeling_florence2.py under {code_dir}"
        )

    package_name = "_agentpack_florence2_" + str(abs(hash(str(code_dir))))
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(code_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    config_module = _load_module_from_file(
        f"{package_name}.configuration_florence2",
        config_file,
    )
    model_module = _load_module_from_file(
        f"{package_name}.modeling_florence2",
        model_file,
    )
    return config_module.Florence2Config, model_module.Florence2ForConditionalGeneration


def _load_module_from_file(module_name: str, path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _coerce_xyxy(box: Any) -> List[float]:
    if isinstance(box, dict):
        for key in ("bbox", "box", "xyxy", "coords", "coordinates"):
            value = box.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 4:
                return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    return []


def _normalize_element(item: Any, idx: int, label_coordinates: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(item) if isinstance(item, dict) else {"content": str(item or "")}
    element_id = str(
        raw.get("element_id")
        or raw.get("id")
        or raw.get("index")
        or idx
    )
    bbox = _coerce_xyxy(raw)
    if not bbox:
        bbox = _coerce_xyxy(label_coordinates.get(element_id))
    content = str(
        raw.get("content")
        or raw.get("text")
        or raw.get("label")
        or raw.get("caption")
        or ""
    ).strip()
    element_type = str(
        raw.get("type")
        or raw.get("category")
        or ("text" if content else "ui_element")
    ).strip()
    interactable = raw.get("interactable", raw.get("is_interactable"))
    if interactable is None:
        interactable = raw.get("interactivity", raw.get("clickable"))
    if interactable is not None:
        interactable = bool(interactable)
    return {
        "element_id": element_id,
        "bbox": bbox,
        "type": element_type or "ui_element",
        "content": content,
        "interactable": interactable,
        "raw": raw,
    }


def _build_summary(elements: List[Dict[str, Any]]) -> str:
    snippets: List[str] = []
    for element in elements:
        content = str(element.get("content", "") or "").strip()
        if not content:
            continue
        if content not in snippets:
            snippets.append(content)
        if len(snippets) >= 5:
            break
    if not snippets:
        return ""
    return "UI elements: " + " | ".join(snippets)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def run_batch(repo_root: str, model_config: Dict[str, Any], image_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    repo_root = os.path.abspath(repo_root)
    check_ocr_box, _, get_som_labeled_img, _ = _import_official_utils(repo_root)
    som_model, caption_model_processor = _load_models(repo_root, model_config)

    box_threshold = float(model_config.get("box_threshold", 0.05) or 0.05)
    iou_threshold = float(model_config.get("iou_threshold", 0.7) or 0.7)
    batch_size = int(model_config.get("caption_batch_size", 64) or 64)
    imgsz = model_config.get("imgsz")
    read_ocr = _build_ocr(repo_root, model_config)
    text_threshold = float(model_config.get("text_threshold", 0.8) or 0.8)
    output_coord_in_ratio = bool(model_config.get("output_coord_in_ratio", True))

    results: Dict[str, Dict[str, Any]] = {}
    for image_path in image_paths or []:
        image = Image.open(image_path).convert("RGB")
        box_overlay_ratio = max(image.size) / 3200.0
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }
        import numpy as np
        ocr_items = read_ocr(np.asarray(image), text_threshold)
        ocr_text = [item[1] for item in ocr_items]
        ocr_bbox = [[float(np.min(item[0], axis=0)[0]), float(np.min(item[0], axis=0)[1]),
                     float(np.max(item[0], axis=0)[0]), float(np.max(item[0], axis=0)[1])]
                    for item in ocr_items]
        # OmniParser's utility returns None when OCR finds no boxes, but its
        # downstream list construction expects both values to be iterable.
        ocr_text = list(ocr_text or [])
        ocr_bbox = list(ocr_bbox or [])
        labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=output_coord_in_ratio,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=ocr_text,
            use_local_semantics=True,
            iou_threshold=iou_threshold,
            scale_img=False,
            batch_size=batch_size,
            imgsz=imgsz,
        )
        elements = [
            _normalize_element(item, idx, label_coordinates if isinstance(label_coordinates, dict) else {})
            for idx, item in enumerate(list(parsed_content_list or []))
        ]
        results[str(image_path)] = {
            "image_path": str(image_path),
            "image_size": {"width": int(image.size[0]), "height": int(image.size[1])},
            "labeled_image_base64": str(labeled_img or ""),
            "label_coordinates": label_coordinates if isinstance(label_coordinates, dict) else {},
            "ocr_text": list(ocr_text or []),
            "ocr_bbox": list(ocr_bbox or []),
            "parsed_content_list": list(parsed_content_list or []),
            "elements": elements,
            "summary": _build_summary(elements),
            "diagnostics": {
                "element_count": len(elements),
                "ocr_device": str(model_config.get("ocr_device", "cpu")),
                "text_count": sum(1 for x in elements if str(x.get("type", "")).lower() == "text"),
                "interactable_count": sum(1 for x in elements if x.get("interactable") is True),
            },
        }
    return results


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    model_config = dict(payload.get("model_config", {}) or {})
    image_paths = [str(x) for x in list(payload.get("image_paths", []) or []) if str(x or "").strip()]
    result = run_batch(args.repo_root, model_config, image_paths)
    Path(args.output_json).write_text(
        json.dumps(_to_jsonable(result), ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(_main())
