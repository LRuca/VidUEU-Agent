#!/usr/bin/env python3
"""Run complete frozen MMDU/GEdit/WEAVE cases with real inference and judging."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from orchestrator.iteration.native_benchmarks import run_native_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "subset", "output", "result", "config", "prompt-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--benchmark", required=True, choices=("MMDU", "GEdit-Bench-En", "WEAVEbench"))
    args = parser.parse_args()
    cfg = yaml.safe_load(os.path.expandvars(args.config.read_text(encoding="utf-8")))
    runtime = cfg["benchmark_runtime"]
    if runtime.get("visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime["visible_devices"])
    judges = cfg.get("benchmark_judges", {})
    judge = judges.get(args.benchmark, cfg.get("benchmark_judge"))
    if not isinstance(judge, dict):
        raise ValueError("benchmark_judge_config_missing")
    run_native_benchmark(benchmark=args.benchmark, checkpoint=args.checkpoint,
                         checkpoint_digest=args.checkpoint_digest, subset=args.subset,
                         output=args.output, result=args.result, runtime_config=runtime,
                         judge_config=judge, decoding=cfg["benchmark_decoding"],
                         prompt_file=args.prompt_file)


if __name__ == "__main__":
    main()
