"""Keep optional Ultralytics dependencies out of runtime package installation."""
import os
import sys


def disable_runtime_installation():
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    checks = sys.modules.get('ultralytics.utils.checks')
    if checks is not None:
        checks.AUTOINSTALL = False
    # Recent Ultralytics replaces PIL.Image.open and invokes pip for any
    # decoding failure, including deliberately corrupt JPG/PNG inputs. Our
    # supported media formats use Pillow directly; retain its original errors.
    patches = sys.modules.get('ultralytics.utils.patches')
    if patches is not None and hasattr(patches, '_image_open'):
        from PIL import Image
        if Image.open is getattr(patches, 'image_open', None):
            Image.open = patches._image_open
