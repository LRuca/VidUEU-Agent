#!/usr/bin/env python3
"""Inspect native packed batches before GPU submission; this is not training."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--expected-num-tokens", type=int, default=1024)
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=4096)
    args = parser.parse_args()
    if min(args.nproc_per_node, args.steps, args.gradient_accumulation_steps) < 1:
        parser.error("ranks, steps and accumulation must be positive")
    sys.path.insert(0, str(args.vendor_root.resolve()))
    import torch
    import yaml
    from data.data_utils import add_special_tokens
    from data.dataset_info import DATASET_INFO
    from data.dataset_base import DataConfig, PackedDataset
    from modeling.qwen2 import Qwen2Tokenizer
    from orchestrator.iteration.state import read_jsonl, write_json
    from orchestrator.iteration.taxonomy import TASK_TYPES
    from orchestrator.iteration.jobs import file_fingerprint
    from scripts.run_bagel_train import _prepare_dataset_registry

    rows = read_jsonl(args.episodes)
    name = "iteration_coverage_plan"
    config_path = _prepare_dataset_registry(args.vendor_root, name, args.episodes.resolve(), len(rows))
    DATASET_INFO.setdefault("multiturn_jsonl", {})[name] = {
        "data_dir": "/", "jsonl_path": str(args.episodes.resolve()), "num_total_samples": len(rows)}
    tokenizer = Qwen2Tokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    tokenizer, special_tokens, _ = add_special_tokens(tokenizer)
    seen = {str(row["id"]): set() for row in rows}
    ranks = []
    for rank in range(args.nproc_per_node):
        random.seed(20261017 + rank)
        torch.manual_seed(20261017 + rank)
        config = DataConfig(grouped_datasets=yaml.safe_load(config_path.read_text()),
                            text_cond_dropout_prob=0.0, vit_cond_dropout_prob=0.0,
                            vae_cond_dropout_prob=0.0, max_latent_size=64)
        dataset = PackedDataset(config, tokenizer, special_tokens, local_rank=rank,
                                world_size=args.nproc_per_node, num_workers=0,
                                expected_num_tokens=args.expected_num_tokens, max_num_tokens=args.max_num_tokens,
                                max_num_tokens_per_sample=args.max_num_tokens_per_sample)
        batches, iterator = [], iter(dataset)
        for micro in range(args.steps * args.gradient_accumulation_steps):
            batch = next(iterator)
            indexes = [item["data_indexes"] for item in batch["batch_data_indexes"]]
            for index in indexes:
                seen[str(index[1])].add(index[2])
            batches.append({"micro_step": micro + 1, "tokens": batch["sequence_length"], "indexes": indexes})
        ranks.append({"rank": rank, "batches": batches})
    complete = [row for row in rows if seen[str(row["id"])] == set(range(1, len(row["turns"]) + 1))]
    tasks = sorted({row["task_type"] for row in complete})
    missing = sorted(set(TASK_TYPES) - set(tasks))
    report = {"schema_version": 1, "status": "shortage" if missing else "ready",
              "coverage_scope": "native_packing_plan_only_not_optimizer_consumption",
              "episodes": file_fingerprint(args.episodes), "steps": args.steps,
              "gradient_accumulation_steps": args.gradient_accumulation_steps,
              "complete_episode_ids": [row["id"] for row in complete], "complete_task_types": tasks,
              "missing_complete_task_types": missing, "ranks": ranks}
    write_json(args.report, report)
    print(json.dumps({k: report[k] for k in ("status", "steps", "gradient_accumulation_steps", "complete_task_types", "missing_complete_task_types")}))
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
