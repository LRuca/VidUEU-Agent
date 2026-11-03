from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from executor.tasks import yolov8_detect


def test_missing_yolo_weights_fail_before_library_download(tmp_path, monkeypatch):
    loader = Mock()
    monkeypatch.setattr(yolov8_detect, "YOLO", loader)
    detector = yolov8_detect.YoloDetector(
        {"models": {"yolo": str(tmp_path / "missing.pt")}}
    )
    with pytest.raises(FileNotFoundError, match="existing local file"):
        detector._ensure_model()
    loader.assert_not_called()


def test_existing_yolo_weights_load_once(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture; model loader is mocked")
    model = SimpleNamespace(names={0: "person"})
    loader = Mock(return_value=model)
    monkeypatch.setattr(yolov8_detect, "YOLO", loader)
    detector = yolov8_detect.YoloDetector({"models": {"yolo": str(weights)}})
    assert detector._ensure_model() is model
    assert detector._ensure_model() is model
    loader.assert_called_once_with(str(weights))
