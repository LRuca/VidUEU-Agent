#!/usr/bin/env python3
"""Audit a candidate pool against one canonical iteration policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.iteration.distribution import audit_distribution
from orchestrator.iteration.inventory import load_inventory
from orchestrator.iteration.policy import default_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_inventory(args.candidate_pool)
    report = audit_distribution(rows)
    report["target_distribution"] = {
        key: default_policy(args.iteration)[key] for key in ("capability", "domain", "task_type")
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
