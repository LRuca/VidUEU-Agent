from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


class OmniParserExecutor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.repo_root = os.path.abspath(
            str(self._omni_cfg().get("repo_root", "") or "")
        )
        self.python_executable = str(self._omni_cfg().get("python_executable", "") or "").strip()
        self.mode = str(self._omni_cfg().get("mode", "auto") or "auto").strip().lower()
        self._bridge_module = None

    def _models_cfg(self) -> Dict[str, Any]:
        return (self.config.get("models", {}) or {}) if isinstance(self.config, dict) else {}

    def _omni_cfg(self) -> Dict[str, Any]:
        models_cfg = self._models_cfg()
        return (models_cfg.get("omniparser", {}) or {}) if isinstance(models_cfg, dict) else {}

    def _agentpack_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _bridge_script_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "omniparser_bridge.py")

    def _runtime_model_config(self) -> Dict[str, Any]:
        omni = dict(self._omni_cfg())
        repo_root = str(omni.get("repo_root", "") or "").strip()
        weights_dir = str(omni.get("weights_dir", "") or "").strip()
        if not weights_dir and repo_root:
            weights_dir = repo_root
            omni["weights_dir"] = weights_dir
        if weights_dir:
            omni.setdefault("som_model_path", os.path.join(weights_dir, "icon_detect", "model.pt"))
            omni.setdefault("caption_model_path", os.path.join(weights_dir, "icon_caption"))
        omni.setdefault("caption_model_name", "florence2")
        omni.setdefault("device", "cuda")
        omni.setdefault("box_threshold", 0.05)
        omni.setdefault("iou_threshold", 0.7)
        omni.setdefault("caption_batch_size", 64)
        omni.setdefault("imgsz", 640)
        omni.setdefault("use_paddleocr", True)
        omni.setdefault("ocr_device", "cpu")
        omni.setdefault("ocr_det_path", self._models_cfg().get("paddleocr_det"))
        omni.setdefault("ocr_rec_path", self._models_cfg().get("paddleocr_rec"))
        omni.setdefault("text_threshold", 0.8)
        omni.setdefault("output_coord_in_ratio", True)
        return omni

    def _validate(self) -> None:
        if not self.repo_root:
            raise ValueError("Missing models.omniparser.repo_root in config")
        if not os.path.isdir(self.repo_root):
            raise FileNotFoundError(f"OmniParser repo_root does not exist: {self.repo_root}")

    def _prefer_subprocess(self) -> bool:
        if self.mode == "subprocess":
            return True
        if self.mode == "import":
            return False
        if self.python_executable:
            current = os.path.abspath(sys.executable)
            target = os.path.abspath(self.python_executable)
            return current != target
        return False

    def _run_subprocess(self, frame_paths: List[str]) -> Dict[str, Any]:
        model_config = dict(self._runtime_model_config())
        env = os.environ.copy()
        device = str(model_config.get("device", "") or "").strip().lower()
        if device.startswith("cuda:"):
            # Config uses logical ordinals, like Torch. Respect the parent's
            # visibility mask before isolating the child onto local cuda:0.
            ordinal = int(device.split(":", 1)[1])
            inherited = env.get("CUDA_VISIBLE_DEVICES")
            if inherited is not None:
                visible = [entry.strip() for entry in inherited.split(",") if entry.strip()]
                if ordinal < 0 or ordinal >= len(visible):
                    raise ValueError("OmniParser device is outside CUDA_VISIBLE_DEVICES")
                gpu_idx = visible[ordinal]
            else:
                if ordinal < 0:
                    raise ValueError("OmniParser GPU index must be nonnegative")
                gpu_idx = str(ordinal)
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx
            model_config["device"] = "cuda:0"
        elif device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        payload = {
            "model_config": model_config,
            "image_paths": [str(x) for x in frame_paths if str(x or "").strip()],
        }
        with tempfile.TemporaryDirectory(prefix="agentpack_omniparser_") as tmpdir:
            input_json = os.path.join(tmpdir, "input.json")
            output_json = os.path.join(tmpdir, "output.json")
            Path(input_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            python_exe = self.python_executable or sys.executable
            cmd = [
                python_exe,
                self._bridge_script_path(),
                "--repo-root",
                self.repo_root,
                "--input-json",
                input_json,
                "--output-json",
                output_json,
            ]
            proc = subprocess.run(
                cmd,
                cwd=self._agentpack_root(),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "OmniParser subprocess failed"
                    f"\ncmd={' '.join(cmd)}"
                    f"\nstdout={proc.stdout.strip()}"
                    f"\nstderr={proc.stderr.strip()}"
                )
            return json.loads(Path(output_json).read_text(encoding="utf-8"))

    def _run_import(self, frame_paths: List[str]) -> Dict[str, Any]:
        if self._bridge_module is None:
            from executor.tasks import omniparser_bridge as bridge

            self._bridge_module = bridge
        return self._bridge_module.run_batch(
            self.repo_root,
            self._runtime_model_config(),
            [str(x) for x in frame_paths if str(x or "").strip()],
        )

    def run(self, frame_paths: List[str]) -> Dict[str, Any]:
        self._validate()
        frame_paths = [str(x) for x in frame_paths if str(x or "").strip()]
        if not frame_paths:
            return {}
        if self._prefer_subprocess():
            return self._run_subprocess(frame_paths)
        return self._run_import(frame_paths)

    def close(self) -> None:
        self._bridge_module = None
