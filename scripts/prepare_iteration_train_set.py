#!/usr/bin/env python3
"""Prepare a deterministic smoke train manifest from a candidate pool."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.iteration.distribution import select_records
from orchestrator.iteration.inventory import load_inventory
from orchestrator.iteration.policy import default_policy
from orchestrator.iteration.state import write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--target-train-samples", type=int, default=10000)
    parser.add_argument("--actual-sample-cap", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    rows = load_inventory(args.candidate_pool)
    result = select_records(rows, default_policy(args.iteration), target_train_samples=args.target_train_samples,
                            actual_sample_cap=args.actual_sample_cap, seed=args.seed)
    manifest_rows = [{"record_key": row["record_key"], "candidate": row,
                      "source_pointer": row.get("source_pointer"), "source_sha256": row.get("source_sha256")}
                     for row in result["selected"]]
    write_jsonl(args.output_jsonl, manifest_rows)
    metadata = {
        "iteration": args.iteration,
        "target_train_samples": args.target_train_samples,
        "actual_samples": len(manifest_rows),
        "shortage": len(manifest_rows) < args.target_train_samples,
        "record_keys_unique": len({row["record_key"] for row in manifest_rows}) == len(manifest_rows),
        "distribution_report": result["report"],
    }
    write_json(args.output_meta, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
