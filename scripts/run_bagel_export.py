#!/usr/bin/env python3
"""Export one BAGEL training checkpoint and verify its official layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        value.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
    return value.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    vendor = args.vendor_root.resolve()
    base = args.base_model_dir.resolve()
    checkpoint = args.checkpoint_dir.resolve()
    output = args.output_dir.resolve()
    if not (vendor.is_dir() and base.is_dir() and checkpoint.is_dir()):
        raise FileNotFoundError("bagel_export_input_missing")
    script = vendor / "tools" / "export_ckpt_to_official_layout_hardcoded.py"
    command = [args.python_executable, str(script), "--ckpt-dir", str(checkpoint),
               "--base-model-dir", str(base), "--out-dir", str(output),
               "--lora-r", "16", "--lora-alpha", "32", "--out-dtype", "bf16"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(vendor) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=str(vendor), env=env, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    required = ["ema.safetensors", "ae.safetensors", "llm_config.json", "vit_config.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError("bagel_export_missing:" + ",".join(missing))
    from orchestrator.iteration.artifacts import content_identity
    result = {
        "schema_version": 1,
        "backend": "bagel_export",
        "simulated": False,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_digest": _sha256(checkpoint),
        "output_dir": str(output),
        "export_digest": content_identity(output, exclude=("export_manifest.json",))["sha256"],
        "required_files": required,
    }
    _write(output / "export_manifest.json", result)
    _write(args.result_path.resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
