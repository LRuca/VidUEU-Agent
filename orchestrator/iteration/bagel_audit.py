"""In-process instrumentation of the unmodified native BAGEL trainer.

Only batches whose forward succeeded and whose optimizer step returned enter
the consumption report.  Hooks live in one worker process and are restored on
exit.  Parameter checks cover complete local FSDP shards, not sampled values.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

from .state import write_json


def tensor_digest(tensor) -> str:
    import torch

    # Viewing bf16 as bytes avoids numpy's lack of bf16 support.
    data = tensor.detach().reshape(-1)
    result = hashlib.sha256()
    result.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    for chunk in data.split(1024 * 1024):
        result.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class TrainingAudit:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.steps: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.batch: dict[str, Any] | None = None
        self.modules = []
        self._restore = []

    def _patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self._restore.append((owner, name, original))
        setattr(owner, name, replacement)
        return original

    def install(self) -> None:
        import torch
        from data.dataset_base import SimpleCustomBatch
        from modeling.bagel import Bagel
        import modeling.autoencoder as autoencoder
        import train.fsdp_utils as fsdp_utils

        original_to_dict = SimpleCustomBatch.to_dict
        def to_dict(batch):
            value = original_to_dict(batch)
            self.batch = {"data_indexes": copy.deepcopy(value.get("batch_data_indexes", [])),
                          "ce_tokens": int(value.get("ce_loss_indexes", torch.empty(0)).numel()),
                          "mse_tokens": int(value.get("mse_loss_indexes", torch.empty(0)).numel())}
            return value
        self._patch(SimpleCustomBatch, "to_dict", to_dict)

        original_forward = Bagel.forward
        def forward(model, *args, **kwargs):
            result = original_forward(model, *args, **kwargs)
            if self.batch is None:
                raise RuntimeError("native_forward_without_audited_batch")
            record = copy.deepcopy(self.batch)
            record["loss_means"] = {}
            for name in ("ce", "mse"):
                term = result.get(name)
                if term is not None and term.numel():
                    if not bool(torch.isfinite(term).all()):
                        raise FloatingPointError(f"nonfinite_native_loss:{name}")
                    record["loss_means"][name] = float(term.detach().float().mean())
            self.pending.append(record)
            return result
        self._patch(Bagel, "forward", forward)

        original_fsdp = fsdp_utils.fsdp_wrapper
        def fsdp_wrapper(*args, **kwargs):
            model = original_fsdp(*args, **kwargs)
            self.modules.append(("model", model))
            return model
        self._patch(fsdp_utils, "fsdp_wrapper", fsdp_wrapper)

        original_load_ae = autoencoder.load_ae
        def load_ae(*args, **kwargs):
            model, config = original_load_ae(*args, **kwargs)
            self.modules.append(("vae", model))
            return model, config
        self._patch(autoencoder, "load_ae", load_ae)

        original_step = torch.optim.AdamW.step
        def step(optimizer, *args, **kwargs):
            if not self.pending:
                raise RuntimeError("optimizer_step_without_successful_forward")
            before = self._parameter_state()
            grad_squares = []
            grad_parameters = 0
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("nonfinite_optimizer_gradient")
                    grad_squares.append(float(parameter.grad.detach().double().square().sum()))
                    grad_parameters += 1
            result = original_step(optimizer, *args, **kwargs)
            after = self._parameter_state()
            changed = [key for key in before if before[key]["sha256"] != after[key]["sha256"]]
            frozen_changed = [key for key in changed if not before[key]["trainable"]]
            if frozen_changed:
                raise RuntimeError("frozen_parameters_changed:" + ",".join(frozen_changed[:8]))
            row = {"optimizer_step": len(self.steps) + 1,
                   "grad_parameters": grad_parameters,
                   "gradient_l2": math.sqrt(sum(grad_squares)),
                   "changed_parameters": changed,
                   "frozen_parameters_checked": sum(not item["trainable"] for item in before.values()),
                   "frozen_changed": frozen_changed,
                   "parameter_before": before, "parameter_after": after,
                   "batches": copy.deepcopy(self.pending)}
            self.steps.append(row)
            self.pending.clear()
            write_json(self.output, {"schema_version": 1, "real_optimizer_steps": len(self.steps),
                                     "steps": self.steps, "coverage_scope": "successful_optimizer_updates"})
            return result
        self._patch(torch.optim.AdamW, "step", step)

    def _parameter_state(self) -> dict[str, Any]:
        import torch

        result = {}
        for prefix, model in self.modules:
            for name, parameter in model.named_parameters():
                if parameter.numel() and not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(f"nonfinite_parameter:{prefix}.{name}")
                result[prefix + "." + name] = {
                    "sha256": tensor_digest(parameter), "trainable": parameter.requires_grad,
                    "numel": parameter.numel(), "shape": list(parameter.shape),
                }
        if not result:
            raise RuntimeError("training_audit_has_no_model")
        return result

    def close(self) -> None:
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        self._restore.clear()


def summarize_training_audits(audits: list[dict[str, Any]], episodes: list[dict[str, Any]],
                              expected_steps: int) -> dict[str, Any]:
    from .taxonomy import TASK_TYPES

    if not audits or any(a.get("real_optimizer_steps") != expected_steps for a in audits):
        raise ValueError("observed_optimizer_steps_do_not_match_request")
    task_by_id = {str(episode["id"]): episode["task_type"] for episode in episodes}
    expected_turns = {str(episode["id"]): set(range(1, len(episode.get("turns", [])) + 1))
                      for episode in episodes}
    consumed_turns: dict[str, set[int]] = {key: set() for key in task_by_id}
    consumed, changed, grad_nonzero = set(), set(), False
    for audit in audits:
        if len(audit.get("steps", [])) != expected_steps:
            raise ValueError("incomplete_training_audit")
        for step in audit["steps"]:
            if step.get("frozen_changed"):
                raise ValueError("frozen_parameters_changed")
            gradient = step.get("gradient_l2")
            if isinstance(gradient, bool) or not isinstance(gradient, (int, float)) or not math.isfinite(gradient):
                raise ValueError("invalid_gradient_evidence")
            grad_nonzero |= gradient > 0
            changed.update(step.get("changed_parameters", []))
            for batch in step["batches"]:
                for index in batch["data_indexes"]:
                    record_id = str(index["data_indexes"][1])
                    if record_id not in task_by_id:
                        raise ValueError("consumed_unknown_episode")
                    consumed.add(record_id)
                    native_index = index["data_indexes"]
                    if len(native_index) >= 3:
                        turn = native_index[2]
                        if isinstance(turn, bool) or not isinstance(turn, int) or turn not in expected_turns[record_id]:
                            raise ValueError("consumed_unknown_episode_turn")
                        consumed_turns[record_id].add(turn)
    if not changed or not grad_nonzero:
        raise ValueError("no_observed_parameter_update")
    tasks = sorted({task_by_id[key] for key in consumed})
    complete_ids = sorted(key for key in task_by_id
                          if expected_turns[key] and consumed_turns[key] == expected_turns[key])
    complete_tasks = sorted({task_by_id[key] for key in complete_ids})
    return {"real_optimizer_steps": expected_steps, "consumed_episode_ids": sorted(consumed),
            "consumed_task_types": tasks, "missing_task_types": sorted(set(TASK_TYPES) - set(tasks)),
            "consumed_episode_turns": {key: sorted(value) for key, value in consumed_turns.items() if value},
            "complete_episode_ids": complete_ids, "complete_task_types": complete_tasks,
            "missing_complete_task_types": sorted(set(TASK_TYPES) - set(complete_tasks)),
            "unconsumed_episode_ids": sorted(set(task_by_id) - consumed),
            "parameter_updates_verified": True, "frozen_parameters_unchanged": True}
