#!/usr/bin/env python3
"""Build a bounded historical training pool from explicit read-only sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.iteration.source_pool import prepare_source_pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = prepare_source_pool(config["sources"], args.output_dir,
                                 max_per_task=config.get("max_per_task", 16),
                                 forbidden_media=[Path(p) for p in config.get("forbidden_media", [])])
    print(json.dumps({key: report[key] for key in ("status", "candidate_count", "task_counts", "missing_task_types")}))
    return 0 if not report["missing_task_types"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
