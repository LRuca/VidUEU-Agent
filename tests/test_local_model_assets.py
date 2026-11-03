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


def test_ultralytics_decoder_patch_never_installs_for_corrupt_image(tmp_path, monkeypatch):
    import sys
    from PIL import Image, UnidentifiedImageError
    from executor.tasks.ultralytics_compat import disable_runtime_installation
    original = Image.open
    installer = Mock(side_effect=AssertionError('runtime installation forbidden'))
    checks = SimpleNamespace(AUTOINSTALL=True)
    monkeypatch.setitem(sys.modules,'ultralytics.utils.checks',checks)
    monkeypatch.setitem(sys.modules,'ultralytics.utils.patches',
                        SimpleNamespace(_image_open=original,image_open=installer))
    monkeypatch.setattr(Image,'open',installer)
    disable_runtime_installation()
    assert checks.AUTOINSTALL is False
    assert Image.open is original
    broken = tmp_path/'broken.jpg'
    broken.write_bytes(b'not an image')
    with pytest.raises(UnidentifiedImageError):
        Image.open(broken)
    installer.assert_not_called()
