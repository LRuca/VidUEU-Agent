# dataset/builder.py
# -*- coding: utf-8 -*-
"""
Unified JSONL dataset writer used by the pipeline.

Rules:
- Constructor receives the full pipeline config dict.
- Output path resolution priority:
  1) config["dataset"]["output_file"]
  2) config["dataset"]["output_path"]
  3) config["tasks"]["T4_dataset"]["output_file"] / output_path
  4) config["paths"]["dataset_output"]
  5) default: dataset/output/data.jsonl
- Writes JSONL, one item per line.
- Thread-safe for shared planner / multi-video execution.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict


class DatasetBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        paths_cfg = config.get("paths", {}) or {}
        ds_cfg = config.get("dataset", {}) or {}
        t4_cfg = (config.get("tasks", {}).get("T4_dataset", {}) or {})

        output_file = (
            ds_cfg.get("output_file")
            or ds_cfg.get("output_path")
            or t4_cfg.get("output_file")
            or t4_cfg.get("output_path")
            or paths_cfg.get("dataset_output")
        )

        if not output_file or isinstance(output_file, dict):
            output_file = os.path.join("dataset", "output", "data.jsonl")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        self.output_file = output_file
        print(f"[DatasetBuilder] Writing dataset to: {self.output_file}")

        self._fp = open(self.output_file, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def _write_line(self, item: Dict[str, Any]) -> None:
        line = json.dumps(item, ensure_ascii=False)
        with self._lock:
            self._fp.write(line + "\n")
            self._fp.flush()

    def add_item(self, item: Dict[str, Any]) -> None:
        if not isinstance(item, dict):
            raise TypeError(f"DatasetBuilder.add_item() expects dict, got {type(item)}")
        self._write_line(item)

    def write(self, item: Dict[str, Any]) -> None:
        self.add_item(item)

    def append(self, item: Dict[str, Any]) -> None:
        self.add_item(item)

    def close(self) -> None:
        try:
            if not self._fp.closed:
                self._fp.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
