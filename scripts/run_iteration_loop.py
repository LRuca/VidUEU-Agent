#!/usr/bin/env python3
"""Run the resumable outer data-iteration loop.

This entry point is intentionally independent from ``run_pipeline.py``.  It
consumes an already produced candidate pool and validates the outer-loop
contracts before any future real trainer/evaluator backend is enabled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow ``python scripts/run_iteration_loop.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.iteration.controller import IterationController, LoopPending
from orchestrator.iteration.jobs import JobObservationTimeout
from vidueu_config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/resume the paper-style iteration loop")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--target-train-samples", type=int)
    parser.add_argument("--actual-sample-cap", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reflection-backend", choices=("fixture", "deepseek"))
    parser.add_argument("--trainer-backend", choices=("dry_run", "smoke_simulator", "bagel", "real_smoke"))
    parser.add_argument("--evaluator-backend", choices=("fixture", "smoke_simulator", "benchmark", "real_smoke"))
    parser.add_argument("--generation-backend", choices=("candidate_pool", "run_pipeline"))
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--pipeline-input")
    parser.add_argument("--python-executable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    loop = config.setdefault("loop", {})
    output_root = args.output_root or Path(loop.get("output_root", "outputs/iteration_smoke"))
    generation_backend = args.generation_backend or loop.get("generation_backend", "candidate_pool")
    candidate_value = args.candidate_pool or loop.get("candidate_pool")
    # A fresh run_pipeline generation can start from a video input without a
    # pre-existing candidate pool.  The controller will inventory a synthetic
    # source record and replace it with the pipeline output during GENERATED.
    if not candidate_value and not (generation_backend == "run_pipeline" and (args.pipeline_input or loop.get("pipeline_input"))):
        raise SystemExit("candidate pool is required via --candidate-pool or loop.candidate_pool")
    candidate_pool = Path(candidate_value) if candidate_value else output_root / "candidate_pool.input.jsonl"
    controller = IterationController(
        config,
        candidate_pool=candidate_pool,
        output_root=output_root,
        rounds=args.rounds,
        target_train_samples=args.target_train_samples,
        actual_sample_cap=args.actual_sample_cap,
        seed=args.seed,
        reflection_backend=args.reflection_backend,
        trainer_backend=args.trainer_backend,
        evaluator_backend=args.evaluator_backend,
        generation_backend=args.generation_backend,
        pipeline_config=args.pipeline_config,
        pipeline_input=args.pipeline_input,
        python_executable=args.python_executable,
    )
    try:
        summary = controller.run()
    except (LoopPending, JobObservationTimeout) as exc:
        print(json.dumps({"status": "WAITING", "reason": str(exc), "output_root": str(output_root)}))
        return 75
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
