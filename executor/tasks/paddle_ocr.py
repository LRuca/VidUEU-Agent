import inspect
import json
import logging
import os
from typing import Any, Dict, List


class PaddleOCRExecutor:
    def __init__(self, config):
        self.config = config or {}

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.model_root = os.path.join(repo_root, "model", "paddleocr")
        self.paddlex_cache_root = os.path.join(repo_root, "model", "paddlex_cache")
        self.mpl_config_root = os.path.join(self.model_root, "mplconfig")
        os.makedirs(self.model_root, exist_ok=True)
        os.makedirs(self.paddlex_cache_root, exist_ok=True)
        os.makedirs(self.mpl_config_root, exist_ok=True)
        os.environ.setdefault("PADDLE_OCR_BASE_DIR", self.model_root)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", self.paddlex_cache_root)
        os.environ.setdefault("MPLCONFIGDIR", self.mpl_config_root)

        from paddleocr import PaddleOCR

        models_cfg = (self.config.get("models", {}) or {}) if isinstance(self.config, dict) else {}
        t4_cfg = (self.config.get("tasks", {}) or {}).get("T4_ocr", {}) or {}

        self.det_model_dir = models_cfg.get("paddleocr_det") or None
        self.rec_model_dir = models_cfg.get("paddleocr_rec") or None
        self.cls_model_dir = models_cfg.get("paddleocr_cls") or None
        self.det_model_name = (
            models_cfg.get("paddleocr_det_name")
            or self._read_model_name(self.det_model_dir)
        )
        self.rec_model_name = (
            models_cfg.get("paddleocr_rec_name")
            or self._read_model_name(self.rec_model_dir)
        )
        self.cls_model_name = (
            models_cfg.get("paddleocr_cls_name")
            or self._read_model_name(self.cls_model_dir)
        )
        self.lang = t4_cfg.get("lang", "ch")
        self.use_angle_cls = bool(t4_cfg.get("use_angle_cls", False))
        self.use_doc_orientation_classify = bool(
            t4_cfg.get("use_doc_orientation_classify", False)
        )
        self.use_doc_unwarping = bool(t4_cfg.get("use_doc_unwarping", False))
        self.enable_hpi = bool(t4_cfg.get("enable_hpi", False))
        self.enable_mkldnn = bool(t4_cfg.get("enable_mkldnn", False))
        self.enable_cinn = bool(t4_cfg.get("enable_cinn", False))
        self.use_gpu = bool(t4_cfg.get("use_gpu", False))
        self.device = self._normalize_paddle_device(
            t4_cfg.get("device") or ("gpu:0" if self.use_gpu else "cpu")
        )
        self.verbose = bool(t4_cfg.get("verbose", False))
        from executor.tasks.ocr_runtime import paddle_device_info
        self.backend_info = paddle_device_info(self.device)
        self.device = self.backend_info["device"]
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

        if not self.verbose:
            os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")
            logging.getLogger("ppocr").setLevel(logging.ERROR)
            logging.getLogger("ppocr_debug").setLevel(logging.ERROR)

        if self.verbose:
            print("[PaddleOCR] initializing")
            print(f"  model_root     = {self.model_root}")
            print(f"  det_model_dir  = {self.det_model_dir}")
            print(f"  rec_model_dir  = {self.rec_model_dir}")
            print(f"  cls_model_dir  = {self.cls_model_dir}")
            print(f"  lang           = {self.lang}")
            print(f"  use_angle_cls  = {self.use_angle_cls}")
            print(f"  use_doc_orientation_classify = {self.use_doc_orientation_classify}")
            print(f"  use_doc_unwarping            = {self.use_doc_unwarping}")
            print(f"  enable_hpi    = {self.enable_hpi}")
            print(f"  enable_mkldnn = {self.enable_mkldnn}")
            print(f"  enable_cinn   = {self.enable_cinn}")
            print(f"  use_gpu        = {self.use_gpu}")
            print(f"  device         = {self.device}")

        ocr_kwargs = {
            "use_textline_orientation": self.use_angle_cls,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "lang": self.lang,
            "device": self.device,
            "enable_hpi": self.enable_hpi,
            "enable_mkldnn": self.enable_mkldnn,
            "enable_cinn": self.enable_cinn,
        }
        if self.det_model_dir:
            ocr_kwargs["text_detection_model_dir"] = self.det_model_dir
        if self.rec_model_dir:
            ocr_kwargs["text_recognition_model_dir"] = self.rec_model_dir
        if self.cls_model_dir:
            ocr_kwargs["textline_orientation_model_dir"] = self.cls_model_dir

        try:
            sig = inspect.signature(PaddleOCR)
            if self.det_model_name and "text_detection_model_name" in sig.parameters:
                ocr_kwargs["text_detection_model_name"] = self.det_model_name
            if self.rec_model_name and "text_recognition_model_name" in sig.parameters:
                ocr_kwargs["text_recognition_model_name"] = self.rec_model_name
            if self.cls_model_name and "textline_orientation_model_name" in sig.parameters:
                ocr_kwargs["textline_orientation_model_name"] = self.cls_model_name
            if "show_log" in sig.parameters:
                ocr_kwargs["show_log"] = self.verbose
        except Exception:
            pass

        self.ocr = PaddleOCR(**ocr_kwargs)

    @staticmethod
    def _read_model_name(model_dir: Any) -> str | None:
        if not model_dir:
            return None
        config_path = os.path.join(str(model_dir), "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError, TypeError):
            return None

        global_cfg = payload.get("Global", {}) if isinstance(payload, dict) else {}
        value = global_cfg.get("model_name") if isinstance(global_cfg, dict) else None
        if not value and isinstance(payload, dict):
            value = payload.get("model_name")
        return str(value).strip() if value else None

    @staticmethod
    def _normalize_paddle_device(device: Any) -> str:
        if device is None:
            return "cpu"

        raw = str(device).strip()
        if not raw:
            return "cpu"

        lowered = raw.lower()
        if lowered == "cuda":
            return "gpu"
        if lowered.startswith("cuda:"):
            return f"gpu:{raw.split(':', 1)[1]}"
        return lowered

    def _normalize_lines(self, result: Any) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []

        # PaddleOCR 2.x style: [[ [box, (text, conf)], ... ]]
        if isinstance(result, list) and result and isinstance(result[0], list):
            for box in result[0]:
                try:
                    text, conf = box[1]
                except Exception:
                    continue
                lines.append(
                    {
                        "text": str(text),
                        "confidence": round(float(conf), 3),
                    }
                )
            return lines

        # PaddleOCR 3.x style result objects/dicts.
        items = result if isinstance(result, list) else [result]
        for item in items:
            rec_texts = None
            rec_scores = None
            if isinstance(item, dict):
                rec_texts = item.get("rec_texts")
                rec_scores = item.get("rec_scores")
            else:
                rec_texts = getattr(item, "rec_texts", None)
                rec_scores = getattr(item, "rec_scores", None)
                if rec_texts is None and hasattr(item, "res"):
                    res = getattr(item, "res")
                    if isinstance(res, dict):
                        rec_texts = res.get("rec_texts")
                        rec_scores = res.get("rec_scores")

            if not rec_texts:
                continue

            rec_scores = list(rec_scores) if rec_scores is not None else [None] * len(rec_texts)
            for text, conf in zip(rec_texts, rec_scores):
                entry = {"text": str(text)}
                if conf is not None:
                    entry["confidence"] = round(float(conf), 3)
                lines.append(entry)

        return lines

    def run(self, frame_paths):
        if self.verbose:
            print(f"[PaddleOCR] processing {len(frame_paths)} frames")

        ocr_results = {}
        for frame_path in frame_paths:
            img_id = os.path.basename(frame_path)
            result = self.ocr.predict(
                frame_path,
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
                use_textline_orientation=self.use_angle_cls,
            )
            ocr_results[img_id] = self._normalize_lines(list(result))

        if self.verbose:
            print("[PaddleOCR] done")

        return ocr_results
