"""Offline contracts for eager import removal, device honesty and OCR records."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from executor.tasks.ocr_runtime import paddle_device_info
from executor.tasks.omniparser_compat import adapt_utils
from executor.tasks.omniparser import OmniParserExecutor


def test_cpu_paddle_rejects_gpu_request_before_model_creation(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(is_compiled_with_cuda=lambda: False))
    assert paddle_device_info("cpu") == {"device": "cpu", "cuda_build": False, "backend": "cpu"}
    with pytest.raises(RuntimeError, match="ocr_gpu_backend_missing"):
        paddle_device_info("cuda:0")


def test_gpu_paddle_requires_visible_logical_device(monkeypatch):
    fake = SimpleNamespace(is_compiled_with_cuda=lambda: True,
                           device=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)))
    monkeypatch.setitem(sys.modules, "paddle", fake)
    assert paddle_device_info("cuda:0")["device"] == "gpu:0"
    with pytest.raises(RuntimeError, match="unavailable"):
        paddle_device_info("cuda:5")


DEMO = '''
from openai import AzureOpenAI
import easyocr
from paddleocr import PaddleOCR
reader = easyocr.Reader(['en'])
paddle_ocr = PaddleOCR(use_gpu=False)
def get_som_labeled_img(image_source, ocr_bbox, ocr_text):
    if not ocr_bbox:
        ocr_bbox = None
    filtered_boxes_elem = list(zip(ocr_bbox, ocr_text))
    starting_idx = -1
    return filtered_boxes_elem
'''


def test_compat_loader_never_constructs_eager_ocr_or_requires_cloud_sdk():
    tree = adapt_utils(DEMO, "fixture.py")
    namespace = {"_vidueu_empty_som": lambda image: ("png", {}, [])}
    exec(compile(tree, "fixture.py", "exec"), namespace)
    assert namespace["get_som_labeled_img"]("blank", [], []) == ("png", {}, [])
    assert "reader" not in namespace and "paddle_ocr" not in namespace
    assert "openai" not in ast.unparse(tree)


def test_compat_loader_refuses_changed_upstream_layout():
    with pytest.raises(RuntimeError, match="layout_changed"):
        adapt_utils(DEMO.replace("paddle_ocr =", "other ="), "fixture.py")


def test_florence_detection_uses_model_type_not_install_directory():
    source = DEMO + '''
def get_parsed_content_icon(model):
    return 'florence' in model.config.name_or_path
'''
    tree = adapt_utils(source, "fixture.py")
    namespace = {}
    exec(compile(tree, "fixture.py", "exec"), namespace)
    model = SimpleNamespace(config=SimpleNamespace(name_or_path="/weights/icon_caption", model_type="florence2"))
    assert namespace["get_parsed_content_icon"](model) is True


def test_no_ocr_overlap_preserves_element_dictionary():
    source = DEMO + '''
def remove_overlap_new(boxes):
    filtered_boxes = []
    for box1_elem in boxes:
        box1 = box1_elem['bbox']
        filtered_boxes.append(box1)
    return filtered_boxes
'''
    namespace = {}
    exec(compile(adapt_utils(source, "fixture.py"), "fixture.py", "exec"), namespace)
    element = {"bbox": [0, 0, 1, 1], "content": None, "type": "icon"}
    assert namespace["remove_overlap_new"]([element]) == [element]


def test_omniparser_propagates_local_ocr_paths_and_cpu_default():
    executor = OmniParserExecutor({"models": {"paddleocr_det": "/models/det",
        "paddleocr_rec": "/models/rec", "omniparser": {"repo_root": "/models/OmniParser"}}})
    config = executor._runtime_model_config()
    assert config["use_paddleocr"] is True
    assert config["ocr_device"] == "cpu"
    assert config["ocr_det_path"] == "/models/det"
    assert config["ocr_rec_path"] == "/models/rec"


def test_paddle_executor_consumes_predict_generator(tmp_path):
    from executor.tasks.paddle_ocr import PaddleOCRExecutor
    executor = object.__new__(PaddleOCRExecutor)
    executor.verbose = False
    executor.use_doc_orientation_classify = False
    executor.use_doc_unwarping = False
    executor.use_angle_cls = False
    executor.ocr = SimpleNamespace(predict=lambda *a, **kw: iter([
        {"rec_texts": ["SMOKE"], "rec_scores": [0.99]}]))
    output = executor.run([str(tmp_path / "frame.png")])
    assert output["frame.png"] == [{"text": "SMOKE", "confidence": 0.99}]


def test_bridge_missing_local_ocr_is_error_before_paddle_constructor(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from executor.tasks.omniparser_bridge import _build_ocr
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(is_compiled_with_cuda=lambda: False))
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(
        PaddleOCR=lambda **kw: pytest.fail("must not initialize or download")))
    with pytest.raises(FileNotFoundError, match="ocr_asset_missing"):
        _build_ocr(str(tmp_path / "OmniParser"), {})


def test_bridge_preserves_official_interactivity_field():
    pytest.importorskip("PIL")
    from executor.tasks.omniparser_bridge import _normalize_element
    result = _normalize_element({"type": "icon", "content": "Upload", "interactivity": True,
                                 "bbox": [0.1, 0.2, 0.3, 0.4]}, 0, {})
    assert result["interactable"] is True


def test_setup_includes_runtime_probe_before_pip_check(tmp_path, capsys):
    import json
    from scripts.setup_env import main
    main(["--profile", "omniparser", "--env-prefix", str(tmp_path / "envs"), "--dry-run"])
    commands = json.loads(capsys.readouterr().out)["commands"]
    assert "check_runtime.py" in commands[-2][1]
    assert commands[-1][-3:] == ["-m", "pip", "check"]
