"""Load pinned OmniParser geometry helpers without eager OCR/cloud clients.

The downloaded asset is never rewritten. These narrow, checked adaptations
bridge its PaddleOCR 2.x demo utilities to our local PaddleOCR 3.x runtime.
Upstream utilities: Microsoft OmniParser, CC-BY-4.0 at the pinned revision;
https://github.com/microsoft/OmniParser. Adaptations are described in NOTICE.md.
"""
from __future__ import annotations

import ast
import base64
import io
from pathlib import Path
import sys
import types

_MODULES = {}


def _empty_som(image):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), {}, []


def adapt_utils(source: str, filename: str):
    """Accept the pinned demo's known startup pattern, fail on layout drift."""
    tree = ast.parse(source, filename=filename)
    removed = set()
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {"openai", "paddleocr"}:
            continue
        if isinstance(node, ast.Import):
            node.names = [item for item in node.names if item.name != "easyocr"]
            if not node.names:
                continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"reader", "paddle_ocr"}:
                removed.add(target.id)
                continue
        body.append(node)
    if removed != {"reader", "paddle_ocr"}:
        raise RuntimeError("omniparser_upstream_layout_changed: eager OCR assignments")
    tree.body = body
    som = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "get_som_labeled_img"), None)
    if som is None:
        raise RuntimeError("omniparser_upstream_layout_changed: get_som_labeled_img")
    # The official empty-OCR branch resets [] to None before zip(None, texts).
    fixes = 0
    for node in ast.walk(som):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "ocr_bbox"
                and isinstance(node.value, ast.Constant) and node.value.value is None):
            node.value = ast.List(elts=[], ctx=ast.Load())
            fixes += 1
    if fixes != 1:
        raise RuntimeError("omniparser_upstream_layout_changed: empty OCR branch")
    # Local weights live in icon_caption/, so the path is not a model-type
    # signal. Wrongly taking the BLIP branch forwards a text-only mask.
    class FlorenceType(ast.NodeTransformer):
        def visit_Compare(self, node):
            self.generic_visit(node)
            if (isinstance(node.left, ast.Constant) and node.left.value == "florence"
                    and len(node.comparators) == 1
                    and ast.unparse(node.comparators[0]) == "model.config.name_or_path"):
                node.comparators[0] = ast.parse("model.config.model_type.lower()", mode="eval").body
            return node
    tree = FlorenceType().visit(tree)
    overlap = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "remove_overlap_new"), None)
    if overlap is not None:
        # In the no-OCR branch upstream appends raw coordinates rather than the
        # element dictionary, breaking the following content/type lookups.
        changed = 0
        for node in ast.walk(overlap):
            if (isinstance(node, ast.Call) and ast.unparse(node.func) == "filtered_boxes.append"
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "box1"):
                node.args[0] = ast.Name(id="box1_elem", ctx=ast.Load())
                changed += 1
        if changed != 1:
            raise RuntimeError("omniparser_upstream_layout_changed: no-OCR overlap")
    # No detections is a valid result, not a malformed zero-dimensional tensor.
    for index, node in enumerate(som.body):
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == "starting_idx" for t in node.targets)):
            som.body[index:index] = ast.parse(
                "if not filtered_boxes_elem:\n    return _vidueu_empty_som(image_source)\n").body
            break
    else:
        raise RuntimeError("omniparser_upstream_layout_changed: starting_idx")
    return ast.fix_missing_locations(tree)


def load_official_utils(repo_root: str):
    root = Path(repo_root).resolve()
    if root in _MODULES:
        return _MODULES[root]
    path = root / "util/utils.py"
    tree = adapt_utils(path.read_text(encoding="utf-8"), str(path))
    existing = sys.modules.get("util")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise RuntimeError("omniparser_wrong_util_import")
    sys.path.insert(0, str(root))
    module = types.ModuleType("_vidueu_omniparser_utils")
    module.__file__ = str(path)
    module._vidueu_empty_som = _empty_som
    exec(compile(tree, str(path), "exec"), module.__dict__)
    original_caption = module.get_parsed_content_icon

    def caption(boxes, starting_idx, *args, **kwargs):
        if starting_idx < 0:
            return []
        return original_caption(boxes, starting_idx, *args, **kwargs)

    module.get_parsed_content_icon = caption
    _MODULES[root] = module
    return module
