#!/usr/bin/env python3
"""Validate and convert selected nine-task samples for BAGEL training."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.iteration.bagel_data import convert_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--allow-partial-coverage", action="store_true")
    parser.add_argument("--require-paper-rounds", action="store_true")
    args = parser.parse_args()
    report = convert_manifest(args.manifest, args.output, asset_root=args.asset_root,
                              require_nine_tasks=not args.allow_partial_coverage,
                              require_paper_rounds=args.require_paper_rounds)
    print(json.dumps({k: v for k, v in report.items() if k != "media"}, indent=2))


if __name__ == "__main__":
    main()
