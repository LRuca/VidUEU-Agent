"""Resumable paper-style autonomous data iteration controller.

The controller deliberately keeps generation, training, and evaluation behind
small backends.  The first deployment uses the existing reviewed candidate
pool, a dry-run trainer, and fixture metrics; replacing those backends later
does not change the state or manifest contracts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml
from vidueu_config import load_config as load_yaml_config

from .backends import dry_run_train, evaluate_fixture, smoke_evaluate, smoke_train
from .real_backends import RealBackendError, run_bagel_export, run_bagel_train
from .benchmark_backends import benchmark_protocol, run_benchmark_evaluation
from .artifacts import content_identity, publish_immutable_json
from .bagel_data import convert_manifest
from .source_pool import load_audited_pool
from .distribution import audit_distribution, select_records
from .feedback import validate_feedback, reflection_feedback, reflection_distribution
from .manifest import validate_manifest_metadata
from .jobs import JobObservationTimeout, file_lock
from .policy import apply_config_overlay, default_policy, policy_diff
from .reflection import deepseek_reflection, fixture_reflection, next_policy, validate_reflection
from .state import digest, read_json, read_jsonl, write_json, write_jsonl
from .taxonomy import label_details_for_record, labels_for_record
from .labels import canonical_iteration_labels, propagate_iteration_labels
from .scheduler import select_source_records, source_key


STAGES = (
    "PLANNED",
    "INVENTORIED",
    "GENERATED",
    "DISTRIBUTION_AUDITED",
    "TRAIN_MANIFEST_READY",
    "TRAIN_DRY_RUN",
    "EXPORT_DRY_RUN",
    "EVAL_DRY_RUN",
    "REFLECTED",
    "COMPLETED",
)
TERMINAL_FAILURE_STAGES = ("FAILED", "BLOCKED")
RESUMABLE_STAGES = (*TERMINAL_FAILURE_STAGES, "WAITING")
_UNEXPANDED_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# These fields affect placement, credentials, or worker liveness.  They are
# deliberately excluded from the semantic identity of a run so that a job
# which failed before generation can be resumed after moving it to an
# available GPU.  Any field not listed here remains semantic by default.
_DEPLOYMENT_KEYS = frozenset({
    "adaptive_gpu_memory_utilization",
    "allowed_local_media_path",
    "api_key",
    "api_key_file",
    "auto_start",
    "auto_trim_by_free_vram",
    "base_port",
    "base_url",
    "bind_host",
    "connect_host",
    "cooldown_base_s_429",
    "cooldown_base_s_other",
    "cooldown_max_s",
    "device",
    "devices",
    "empty_cache_on_return",
    "extra_args",
    "gpu_memory_utilization",
    "gpu_memory_utilization_margin_gb",
    "gpu_placement",
    "global_wait_timeout_s",
    "idle_unload_s",
    "load_timeout_s",
    "log_dir",
    "manager_assign_devices",
    "max_concurrency",
    "max_expensive_tool_calls_per_segment",
    "max_gpu_resident_models",
    "max_inflight",
    "max_waiters",
    "min_free_vram_gb",
    "min_free_vram_gb_per_replica",
    "model_device_overrides",
    "nproc_per_node",
    "output_root",
    "per_model",
    "per_resource_devices",
    "poll_interval_s",
    "pool_connections",
    "pool_maxsize",
    "prefer_spread",
    "proxy_url",
    "queue_timeout_s",
    "reload_key_file",
    "remote_b",
    "replicas",
    "reserve_vram_gb",
    "reserved_margin_mb",
    "resources",
    "scheduler",
    "service",
    "shell_executable",
    "single_gpu_scheduler",
    "startup_on_init",
    "startup_timeout_s",
    "target_replicas",
    "timeout_s",
    "tool_runtimes",
    "verify_ssl",
    "video_parallelism",
    "visible_devices",
    "workdir",
    "yolo_device",
})


def _semantic_projection(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    """Return a config projection whose identity excludes deployment knobs.

    The projection is intentionally conservative: only explicitly known
    placement/transport fields are removed.  A new semantic field therefore
    invalidates a run until it is reviewed, while a GPU relocation does not.
    ``pipeline_config`` and ``python_executable`` are handled separately as
    deployment references in the outer loop configuration.
    """
    if isinstance(value, dict):
        projected: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in _DEPLOYMENT_KEYS:
                continue
            if path == ("loop",) and key in {
                "output_root", "pipeline_config", "pipeline_input", "python_executable",
            }:
                continue
            if path == ("bagel",) and key in {
                "vendor_root", "python_executable", "visible_devices", "master_port",
            }:
                continue
            projected[key] = _semantic_projection(item, path=path + (key,))
        return projected
    if isinstance(value, list):
        return [_semantic_projection(item, path=path) for item in value]
    if isinstance(value, tuple):
        return [_semantic_projection(item, path=path) for item in value]
    return value


def _pipeline_semantic_identity(path: Path | None) -> dict[str, Any] | None:
    """Hash the semantic portion of a pipeline YAML independently of placement."""
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        # The full file identity still records the input.  A malformed YAML
        # file must not become silently compatible with another malformed file.
        raw = {"unparsed_sha256": _sha256_file(path)}
    return {
        "semantic_sha256": digest(_semantic_projection(raw)),
        "full_sha256": _sha256_file(path),
    }


class LoopPending(RuntimeError):
    """A dependency needs observation/retry; no completed stage is discarded."""


def _input_identity(value: str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(value).resolve()
    if path.is_dir():
        files = [{"path": str(item.relative_to(path)), "sha256": _sha256_file(item)}
                 for item in sorted(path.rglob("*")) if item.is_file()]
        return {"path": str(path), "files_digest": digest(files), "file_count": len(files)}
    return {"path": str(path), "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _expand_config_text(text: str, *, source: Path) -> str:
    """Expand environment variables and fail before launching a pipeline.

    ``os.path.expandvars`` intentionally leaves an unknown variable literal in
    place.  That behavior is unsafe for an iteration run: a missing API key or
    endpoint would otherwise reach model construction and fail much later.  We
    keep expansion compatible with the existing YAML configs, while making
    unresolved ``${NAME}`` placeholders an explicit preflight error.
    """
    expanded = os.path.expandvars(text)
    missing = sorted(set(_UNEXPANDED_ENV_RE.findall(expanded)))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"unexpanded_environment_variables:{names}:config={source}")
    return expanded


def _source_pointer(record: dict[str, Any]) -> str | None:
    provenance = record.get("review_provenance")
    if isinstance(provenance, dict):
        pointer = provenance.get("source_pointer") or provenance.get("source_file")
        if pointer:
            return str(pointer)
    for key in ("source_pointer", "source_file", "image", "video_path"):
        if record.get(key):
            return str(record[key])
    meta = record.get("meta")
    if isinstance(meta, dict):
        for key in ("source_pointer", "source_file", "video_path", "segment_path"):
            if meta.get(key):
                return str(meta[key])
    return None


def _source_hash(record: dict[str, Any]) -> str:
    if record.get("source_origin") == "historical_training_source" and record.get("source_row_sha256"):
        return str(record["source_row_sha256"])
    provenance = record.get("review_provenance")
    if isinstance(provenance, dict) and provenance.get("source_sha256"):
        return str(provenance["source_sha256"])
    pointer = _source_pointer(record)
    if pointer:
        file_hash = _sha256_file(Path(pointer))
        if file_hash:
            return file_hash
    return digest(record)


def normalize_candidate(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Attach stable provenance and canonical labels without changing content."""
    if not isinstance(record, dict):
        raise ValueError(f"candidate_{index}_must_be_object")
    item = copy.deepcopy(record)
    item.setdefault("record_key", digest(record))
    item["record_key"] = str(item["record_key"])
    item["source_pointer"] = _source_pointer(item)
    item["source_sha256"] = _source_hash(item)
    item["input_row"] = item.get("input_row", index + 1)
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    if item.get("video_type") in (None, "") and meta.get("video_type") not in (None, ""):
        item["video_type"] = meta.get("video_type")
    # Mirror labels from nested metadata/global perception before the generic
    # taxonomy pass.  This is the final guard for legacy benchmark exports
    # whose top-level fields were flattened away by T4.
    item = propagate_iteration_labels(
        item,
        canonical_iteration_labels(item),
        provenance="iteration_inventory",
    )
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    labels = labels_for_record(item)
    label_details = label_details_for_record(item)
    item["capability"] = item.get("capability") or labels["capability"]
    item["task_type"] = item.get("task_type") or labels["task_type"]
    item["policy_domain"] = item.get("policy_domain") or labels["domain"]
    item["domain_source"] = item.get("domain_source") or label_details.get("domain_source")
    item["policy_domain_source"] = item.get("domain_source")
    item["domain_confidence"] = label_details.get("domain_confidence")
    item["label_source"] = item.get("label_source") or label_details.get("domain_source") or "unknown"
    item["label_version"] = item.get("label_version") or label_details.get("label_version") or "domain_v1"
    item["loop_provenance"] = {
        "source_pointer": item["source_pointer"],
        "source_sha256": item["source_sha256"],
        "input_row": item["input_row"],
    }
    return item


def collect_generation_candidates(pipeline_output: Path) -> list[dict[str, Any]]:
    """Read accepted exports, preserving media roots and duplicate provenance."""
    reviewed = sorted(pipeline_output.rglob("reviewed_candidates.jsonl"))
    data_files = sorted(pipeline_output.rglob("data.jsonl"))
    candidates: list[dict[str, Any]] = []
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    provenance_keys = {"input_row", "input_source_file", "input_source_line", "input_sources"}
    for path in reviewed or data_files:
        for source_line, original in enumerate(read_jsonl(path), 1):
            row = copy.deepcopy(original)
            if not isinstance(row, dict):
                raise ValueError(f"generated_candidate_must_be_object:{path}:{source_line}")
            if not row.get("asset_root"):
                if path.parent.name == "output":
                    row["asset_root"] = str(path.parent.parent.resolve())
                elif path.parent.name in {"reviewed", "quality_review"}:
                    row["asset_root"] = str(path.parent.parent.resolve())
                else:
                    row["asset_root"] = str(path.parent.resolve())
            content = {key: value for key, value in row.items() if key not in provenance_keys}
            signature = digest(content)
            key = str(row.get("record_key") or signature)
            pointer = {"file": str(path.resolve()), "line": source_line}
            if key in seen:
                prior_signature, prior = seen[key]
                if prior_signature != signature:
                    raise ValueError(f"generated_record_key_conflict:{key}")
                prior["input_sources"].append(pointer)
                continue
            row.setdefault("record_key", key)
            row.setdefault("input_row", source_line)
            row.setdefault("input_source_file", str(path.resolve()))
            row.setdefault("input_source_line", source_line)
            row["input_sources"] = [pointer]
            seen[key] = (signature, row)
            candidates.append(row)
    if not candidates:
        raise ValueError("pipeline_produced_no_candidate_exports")
    return [normalize_candidate(row, i) for i, row in enumerate(candidates)]


class IterationController:
    """Run and resume the configured outer loop."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        candidate_pool: Path,
        output_root: Path,
        rounds: int | None = None,
        target_train_samples: int | None = None,
        actual_sample_cap: int | None = None,
        seed: int | None = None,
        reflection_backend: str | None = None,
        trainer_backend: str | None = None,
        evaluator_backend: str | None = None,
        generation_backend: str | None = None,
        pipeline_config: Path | None = None,
        pipeline_input: str | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        iteration_cfg = self.config.get("iteration")
        if isinstance(iteration_cfg, dict) and iteration_cfg.get("enabled") is False:
            raise ValueError("iteration_loop_disabled")
        loop = self.config.setdefault("loop", {})
        if loop.get("enabled") is False:
            raise ValueError("iteration_loop_disabled")
        self.rounds = int(rounds if rounds is not None else loop.get("rounds", 2))
        self.target_train_samples = int(
            target_train_samples if target_train_samples is not None else loop.get("target_train_samples", 10000)
        )
        self.actual_sample_cap = int(
            actual_sample_cap if actual_sample_cap is not None else loop.get("actual_sample_cap", 32)
        )
        self.seed = int(seed if seed is not None else loop.get("seed", 20261016))
        self.reflection_backend = reflection_backend or loop.get("reflection_backend", "fixture")
        self.trainer_backend = trainer_backend or loop.get("trainer_backend", "dry_run")
        self.evaluator_backend = evaluator_backend or loop.get("evaluator_backend", "fixture")
        self.generation_backend = generation_backend or loop.get("generation_backend", "candidate_pool")
        self.generation_source_cap = int(loop.get("generation_source_cap", 0) or 0)
        self.audited_source_pool = (Path(loop["audited_source_pool"]).resolve()
                                    if loop.get("audited_source_pool") else None)
        self.audited_source_digest = _sha256_file(self.audited_source_pool) if self.audited_source_pool else None
        if self.audited_source_pool:
            load_audited_pool(self.audited_source_pool)
        configured_pipeline = pipeline_config or loop.get("pipeline_config")
        self.pipeline_config = Path(configured_pipeline) if configured_pipeline else None
        self.pipeline_input = pipeline_input or loop.get("pipeline_input")
        self.python_executable = python_executable or loop.get("python_executable") or sys.executable
        self.real_smoke = (loop.get("mode") == "real_smoke"
                           or self.trainer_backend == "real_smoke"
                           or self.evaluator_backend == "real_smoke")
        if self.real_smoke:
            expected = {
                "generation": (self.generation_backend, {"run_pipeline"}),
                "trainer": (self.trainer_backend, {"bagel", "real_smoke"}),
                "evaluator": (self.evaluator_backend, {"benchmark", "real_smoke"}),
                "reflection": (self.reflection_backend, {"deepseek"}),
            }
            for name, (selected, allowed) in expected.items():
                if selected not in allowed:
                    raise ValueError(f"real_smoke_requires_real_backend:{name}:{selected}")
        if self.rounds <= 0 or self.target_train_samples <= 0 or self.actual_sample_cap < 0:
            raise ValueError("invalid_loop_limits")
        self.candidate_pool = Path(candidate_pool)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        loop.update(
            {
                "rounds": self.rounds,
                "target_train_samples": self.target_train_samples,
                "actual_sample_cap": self.actual_sample_cap,
                "seed": self.seed,
                "reflection_backend": self.reflection_backend,
                "trainer_backend": self.trainer_backend,
                "evaluator_backend": self.evaluator_backend,
                "generation_backend": self.generation_backend,
                "pipeline_config": str(self.pipeline_config.resolve()) if self.pipeline_config else None,
                "pipeline_input": self.pipeline_input,
                "python_executable": self.python_executable,
            }
        )
        self._write_run_manifest()

    @property
    def run_manifest_path(self) -> Path:
        return self.output_root / "loop_manifest.json"

    def _run_manifest_payload(self) -> dict[str, Any]:
        """Build the immutable semantic/deployment identity for this run."""
        full_config = _jsonable(self.config)
        pipeline_identity = _pipeline_semantic_identity(self.pipeline_config)
        semantic = {
            "candidate_pool": str(self.candidate_pool.resolve()),
            "candidate_pool_sha256": _sha256_file(self.candidate_pool),
            "pipeline_input_identity": _input_identity(self.pipeline_input),
            "pipeline_semantic_identity": (
                {"semantic_sha256": pipeline_identity["semantic_sha256"]}
                if pipeline_identity else None
            ),
            "rounds": self.rounds,
            "target_train_samples": self.target_train_samples,
            "actual_sample_cap": self.actual_sample_cap,
            "seed": self.seed,
            "backends": {
                "reflection": self.reflection_backend,
                "trainer": self.trainer_backend,
                "evaluator": self.evaluator_backend,
                "generation": self.generation_backend,
            },
            "config_semantic_digest": digest(_semantic_projection(full_config)),
            "training_mode": "dry_run" if self.trainer_backend == "dry_run" else "configured",
            "paper_claim": "smoke_chain_only",
        }
        if self.audited_source_pool:
            semantic["audited_source_pool"] = str(self.audited_source_pool)
            semantic["audited_source_pool_sha256"] = self.audited_source_digest
        deployment = {
            "pipeline_config": str(self.pipeline_config.resolve()) if self.pipeline_config else None,
            "pipeline_config_identity": _input_identity(self.pipeline_config),
            "pipeline_semantic_full_sha256": pipeline_identity.get("full_sha256") if pipeline_identity else None,
            "python_executable": self.python_executable,
            "config_full_digest": digest(full_config),
        }
        return {
            "schema_version": 3,
            "semantic": semantic,
            "deployment": deployment,
            # Keep the explicit fields duplicated at the top level for old
            # inspection scripts; compatibility decisions use ``semantic``.
            "candidate_pool": semantic["candidate_pool"],
            "candidate_pool_sha256": semantic["candidate_pool_sha256"],
            "pipeline_input_identity": semantic["pipeline_input_identity"],
            "pipeline_config_identity": deployment["pipeline_config_identity"],
            "rounds": self.rounds,
            "target_train_samples": self.target_train_samples,
            "actual_sample_cap": self.actual_sample_cap,
            "seed": self.seed,
            "backends": semantic["backends"],
            "config_digest": deployment["config_full_digest"],
            "training_mode": semantic["training_mode"],
            "paper_claim": semantic["paper_claim"],
        }

    def _legacy_manifest_can_migrate(
        self,
        existing: dict[str, Any],
        current: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Allow one narrowly-scoped migration for the failed v2 run.

        Legacy schema-2 manifests did not retain a semantic projection.  We
        therefore only migrate a run whose first incomplete stage is
        GENERATED, whose pipeline copy proves semantic equality, and whose
        explicit input/backend identities still match.  A run that reached
        training or evaluation is never migrated on this heuristic.
        """
        reasons: list[str] = []
        semantic = current["semantic"]
        checks = (
            (existing.get("candidate_pool_sha256"), semantic["candidate_pool_sha256"], "candidate_pool"),
            (existing.get("pipeline_input_identity"), semantic["pipeline_input_identity"], "pipeline_input"),
            (existing.get("rounds"), semantic["rounds"], "rounds"),
            (existing.get("target_train_samples"), semantic["target_train_samples"], "target_train_samples"),
            (existing.get("actual_sample_cap"), semantic["actual_sample_cap"], "actual_sample_cap"),
            (existing.get("seed"), semantic["seed"], "seed"),
            (existing.get("backends"), semantic["backends"], "backends"),
        )
        for old, new, name in checks:
            if old != new:
                reasons.append(f"{name}_changed")

        state_paths = sorted(self.output_root.glob("iteration_*/state.json"))
        if not state_paths:
            reasons.append("no_failed_stage_state")
        for state_path in state_paths:
            try:
                state = read_json(state_path)
            except Exception:
                reasons.append(f"invalid_state:{state_path.name}")
                continue
            if state.get("status") != "FAILED" or state.get("failed_stage") != "GENERATED":
                reasons.append(f"state_not_generated_failure:{state_path.parent.name}")
            # There must be no downstream artifact that could represent a
            # partially completed training/evaluation run.
            for name in ("train_result.json", "export_result.json", "eval_result.json", "reflection.json"):
                if state_path.parent.joinpath(name).exists():
                    reasons.append(f"downstream_artifact_present:{state_path.parent.name}/{name}")

        old_pipeline = self.output_root / "iteration_01" / "pipeline_config.yaml"
        old_identity = _pipeline_semantic_identity(old_pipeline)
        new_identity = semantic.get("pipeline_semantic_identity")
        if not old_identity or not new_identity:
            reasons.append("pipeline_semantic_identity_missing")
        elif old_identity.get("semantic_sha256") != new_identity.get("semantic_sha256"):
            reasons.append("pipeline_semantics_changed")

        # A deployment-only recovery is safe only when the old run has not
        # produced any downstream round artifacts.  Keep the explicit audit
        # reasons so a future operator can see why migration was accepted.
        return not reasons, reasons

    def _write_run_manifest(self) -> None:
        manifest = self._run_manifest_payload()
        with file_lock(self.output_root / "manifest.lock"):
            if not self.run_manifest_path.exists():
                write_json(self.run_manifest_path, manifest)
                return

            existing = read_json(self.run_manifest_path)
            if not isinstance(existing, dict):
                raise ValueError("invalid_run_manifest")
            if existing.get("schema_version") == 3:
                if existing.get("semantic") != manifest.get("semantic"):
                    raise ValueError("run_manifest_semantics_changed_use_new_output_root")
                if existing.get("deployment") != manifest.get("deployment"):
                    revision_path = self.output_root / "deployment_revisions.json"
                    revisions = read_json(revision_path) if revision_path.exists() else []
                    if not isinstance(revisions, list):
                        raise ValueError("invalid_deployment_revisions")
                    revision = {
                        "from": existing.get("deployment"),
                        "to": manifest.get("deployment"),
                    }
                    if not revisions or revisions[-1] != revision:
                        revisions.append(revision)
                        write_json(revision_path, revisions)
                return

            if existing.get("schema_version") == 2:
                # A prior controller may already have accepted the narrowly
                # scoped migration and then failed at a later stage.  The
                # migration record is the durable proof of the semantic
                # comparison; do not require the original GENERATED failure
                # state on every subsequent resume.
                migration_path = self.output_root / "manifest_migration.json"
                if migration_path.exists():
                    migration = read_json(migration_path)
                    if (
                        isinstance(migration, dict)
                        and migration.get("from_schema") == 2
                        and migration.get("to_schema") == 3
                        and migration.get("semantic_manifest") == manifest.get("semantic")
                    ):
                        return
                compatible, reasons = self._legacy_manifest_can_migrate(existing, manifest)
                if compatible:
                    write_json(
                        self.output_root / "manifest_migration.json",
                        {
                            "from_schema": 2,
                            "to_schema": 3,
                            "reason": "deployment_only_change_before_generated",
                            "semantic_manifest": manifest.get("semantic"),
                            "deployment_manifest": manifest.get("deployment"),
                        },
                    )
                    # Preserve the original schema-2 identity.  The migration
                    # record and deployment revisions make the new effective
                    # configuration auditable without rewriting history.
                    return
                detail = ",".join(reasons) or "legacy_identity_mismatch"
                raise ValueError(f"run_manifest_changed_use_new_output_root:{detail}")

            raise ValueError("unsupported_run_manifest_schema_use_new_output_root")

    def _round_dir(self, iteration: int) -> Path:
        path = self.output_root / f"iteration_{iteration:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _state_path(self, iteration: int) -> Path:
        return self._round_dir(iteration) / "state.json"

    def _load_state(self, iteration: int) -> dict[str, Any]:
        path = self._state_path(iteration)
        if path.exists():
            state = read_json(path)
            if not isinstance(state, dict):
                raise ValueError(f"invalid_state:{path}")
            if state.get("status") in RESUMABLE_STAGES:
                # A failed stage is resumable: restore the last successful
                # stage and let the same action run again.
                completed = [entry.get("stage") for entry in state.get("stage_history", [])]
                state["status"] = completed[-1] if completed else "PLANNED"
                state["resumed"] = True
            return state
        state = {
            "schema_version": 1,
            "iteration": iteration,
            "status": "PLANNED",
            "stage_history": [],
            "resumed": False,
        }
        write_json(path, state)
        return state

    def _save_state(self, iteration: int, state: dict[str, Any]) -> None:
        write_json(self._state_path(iteration), state)

    def _stage_done(self, state: dict[str, Any], stage: str) -> bool:
        current = str(state.get("status", "PLANNED"))
        if current in RESUMABLE_STAGES:
            return False
        return STAGES.index(current) >= STAGES.index(stage)

    def _advance(
        self,
        iteration: int,
        state: dict[str, Any],
        stage: str,
        action: Callable[[], dict[str, Any] | None],
    ) -> None:
        if self._stage_done(state, stage):
            state["resumed"] = True
            return
        try:
            result = action() or {}
        except Exception as exc:
            state["status"] = "WAITING" if isinstance(exc, (LoopPending, JobObservationTimeout)) else "FAILED"
            state["failed_stage"] = stage
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self._save_state(iteration, state)
            raise
        state["status"] = stage
        state.pop("failed_stage", None)
        state.pop("error", None)
        state.setdefault("stage_history", []).append(
            {"stage": stage, "artifact_digest": digest(_jsonable(result))}
        )
        self._save_state(iteration, state)

    def _load_candidates(self) -> list[dict[str, Any]]:
        if not self.candidate_pool.exists():
            if self.generation_backend == "run_pipeline" and self.pipeline_input:
                source = {"video_path": str(self.pipeline_input), "source_pointer": str(self.pipeline_input)}
                return [normalize_candidate(source, 0)]
            raise FileNotFoundError(f"candidate_pool_not_found:{self.candidate_pool}")
        candidates = [normalize_candidate(row, i) for i, row in enumerate(read_jsonl(self.candidate_pool))]
        return candidates

    def _prepare_generation_source_schedule(
        self,
        round_dir: Path,
        iteration: int,
        policy: dict[str, Any],
    ) -> Path | None:
        """Create the next-round source manifest from prior inventory labels."""
        if iteration <= 1:
            return None
        prior_inventory = self._round_dir(iteration - 1) / "source_inventory.jsonl"
        if not prior_inventory.exists():
            return None
        records = read_jsonl(prior_inventory)
        if not records:
            return None
        result = select_source_records(
            records,
            policy,
            source_cap=self.generation_source_cap,
            seed=self.seed + iteration,
        )
        schedule_path = round_dir / "pre_generation_source_schedule.jsonl"
        write_jsonl(schedule_path, result["schedule"])
        write_json(round_dir / "pre_generation_source_scheduler_report.json", result)
        # Zero is audit-only.  Do not replace the caller's input directory or
        # manifest with a list derived from the previous candidate pool.
        if self.generation_source_cap == 0:
            return None
        selected_paths = []
        for row in result["selected"]:
            pointer = row.get("video_path") or row.get("source_pointer")
            if pointer and str(pointer) not in selected_paths:
                selected_paths.append(str(pointer))
        if not selected_paths:
            return None
        list_path = round_dir / "pre_generation_video_list.txt"
        list_path.write_text("\n".join(selected_paths) + "\n", encoding="utf-8")
        return list_path

    def _run_pipeline_generation(
        self,
        round_dir: Path,
        policy: dict[str, Any],
        *,
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Run the existing T1-T4 entry point when explicitly selected."""
        if self.pipeline_config is None:
            raise ValueError("run_pipeline_generation_requires_pipeline_config")
        if not self.pipeline_input:
            raise ValueError("run_pipeline_generation_requires_pipeline_input")
        pipeline_config_path = self.pipeline_config
        if not pipeline_config_path.is_absolute():
            pipeline_config_path = Path.cwd() / pipeline_config_path
        if not pipeline_config_path.exists():
            raise FileNotFoundError(f"pipeline_config_not_found:{pipeline_config_path}")
        base = load_yaml_config(pipeline_config_path)
        if not isinstance(base, dict):
            raise ValueError("pipeline_config_must_be_object")
        overlay = apply_config_overlay(base, policy)
        pipeline_output = round_dir / "pipeline_output"
        overlay.setdefault("runtime", {})["iteration"] = int(policy.get("iteration", 1))
        overlay["runtime"]["iteration_policy"] = copy.deepcopy(policy)
        overlay["runtime"]["output_root"] = str(pipeline_output)
        source_list = self._prepare_generation_source_schedule(round_dir, iteration, policy)
        if source_list is not None:
            scan = overlay["runtime"].setdefault("video_scan", {})
            scan["video_list_file"] = str(source_list)
        overlay_path = round_dir / "pipeline_config.yaml"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text(yaml.safe_dump(_jsonable(overlay), sort_keys=False, allow_unicode=True), encoding="utf-8")
        repo_root = Path(__file__).resolve().parents[2]
        command = [
            self.python_executable,
            str(repo_root / "run_pipeline.py"),
            "--config",
            str(overlay_path),
            "--input",
            str(self.pipeline_input),
            "--output-root",
            str(pipeline_output),
            "--no-dashboard",
        ]
        completed = subprocess.run(command, cwd=str(repo_root), text=True, capture_output=True, check=False)
        (round_dir / "pipeline_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (round_dir / "pipeline_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"run_pipeline_failed:exit={completed.returncode}")
        return collect_generation_candidates(pipeline_output)

    def _supplement_candidates(self, pipeline_rows: list[dict[str, Any]], round_dir: Path) -> list[dict[str, Any]]:
        """Keep historical data separate from the run_pipeline generation count."""
        source_rows: list[dict[str, Any]] = []
        if self.audited_source_pool is not None:
            if _sha256_file(self.audited_source_pool) != self.audited_source_digest:
                raise ValueError("audited_source_pool_changed_during_run")
            source_rows = load_audited_pool(self.audited_source_pool)
            source_rows = [copy.deepcopy(row) for row in source_rows]
            for row in source_rows:
                row["generation_origin"] = "audited_historical_training_source"
                row["generation_backend"] = "audited_source_pool"
                row.setdefault("asset_root", row.get("media_root"))
            write_jsonl(round_dir / "audited_source_pool_forwarded.jsonl", source_rows)
        combined: list[dict[str, Any]] = []
        seen: dict[str, dict[str, Any]] = {}
        duplicate_count = 0
        for row in [*pipeline_rows, *source_rows]:
            key = str(row.get("record_key") or digest(row))
            if key in seen:
                if digest(seen[key]) != digest(row):
                    raise ValueError(f"supplemented_record_key_conflict:{key}")
                duplicate_count += 1
                continue
            seen[key] = row
            combined.append(row)
        write_json(round_dir / "generation_sources_report.json", {
            "pipeline_backend": "run_pipeline",
            "pipeline_generated": len(pipeline_rows),
            "audited_source_pool": str(self.audited_source_pool) if self.audited_source_pool else None,
            "audited_source_pool_forwarded": len(source_rows),
            "combined_candidates": len(combined),
            "duplicate_record_keys_dropped": duplicate_count,
            "historical_sources_are_explicit": bool(source_rows),
        })
        return combined

    def _policy_for_round(self, iteration: int) -> tuple[dict[str, Any], str]:
        if iteration == 1:
            return default_policy(1), "paper_iteration_1"
        prior_reflection = self._round_dir(iteration - 1) / "reflection.json"
        if prior_reflection.exists():
            reflection = read_json(prior_reflection)
            policy = reflection.get("next_policy") if isinstance(reflection, dict) else None
            if isinstance(policy, dict):
                if reflection.get("fallback_to_previous_policy"):
                    raise LoopPending("prior_reflection_not_validated")
                return policy, f"iteration_{iteration - 1:02d}_reflection"
        if self.reflection_backend == "deepseek":
            raise LoopPending("prior_reflection_required")
        prior_policy_path = self._round_dir(iteration - 1) / "policy.json"
        if prior_policy_path.exists():
            return read_json(prior_policy_path), f"iteration_{iteration - 1:02d}_policy_fallback"
        return default_policy(iteration), "paper_default_fallback"

    def _previous_feedback(self, iteration: int) -> dict[str, Any] | None:
        if iteration <= 1:
            return None
        path = self._round_dir(iteration - 1) / "eval_result.json"
        return read_json(path) if path.exists() else None

    def _reflection_config(self) -> dict[str, Any]:
        """Provide DeepSeek's model config without duplicating secret settings.

        The real-generation smoke profile keeps model endpoints and key-file
        paths in the pipeline overlay because that is also what ``run_pipeline``
        consumes.  The outer controller must use the same text-model config for
        reflection; copy only the model section into an isolated config so the
        controller and pipeline cannot drift.  Secrets remain referenced by
        file/environment path and are never copied into output manifests.
        """
        config = copy.deepcopy(self.config)
        path = self.pipeline_config
        if path is None or not path.exists():
            return config
        try:
            overlay = load_yaml_config(path)
        except Exception:
            return config
        if not isinstance(overlay, dict):
            return config
        models = overlay.get("models")
        if not isinstance(models, dict):
            return config
        qwen_llm = models.get("qwen_llm")
        if not isinstance(qwen_llm, dict):
            return config
        config.setdefault("models", {})["qwen_llm"] = copy.deepcopy(qwen_llm)
        return config

    def _ensure_baseline(self) -> dict[str, Any] | None:
        if self.evaluator_backend not in {"benchmark", "real_smoke"}:
            return None
        directory = self.output_root / "baseline"
        directory.mkdir(parents=True, exist_ok=True)
        with file_lock(directory / "controller.lock", blocking=False):
            bagel = self.config.get("bagel") or self.config.get("trainer", {}).get("bagel", {})
            checkpoint = Path(bagel.get("model_path") or self.config.get("trainer", {}).get("init_checkpoint", ""))
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"baseline_checkpoint_missing:{checkpoint}")
            cached_manifest_path = directory / "manifest.json"
            cached_state_path = directory / "state.json"
            cached_result_path = directory / "eval_result.json"
            if cached_manifest_path.exists() and cached_state_path.exists() and cached_result_path.exists():
                cached_state = read_json(cached_state_path)
                cached_manifest = read_json(cached_manifest_path)
                if cached_state.get("status") == "COMPLETED" and isinstance(cached_manifest, dict):
                    cached_result = read_json(cached_result_path)
                    expected_digest = ((cached_manifest.get("checkpoint") or {}).get("sha256"))
                    if (cached_result.get("checkpoint_digest") == expected_digest
                            and cached_result.get("simulated") is False):
                        # Baseline is immutable for a run.  The benchmark job
                        # layer already revalidates its result/artifact
                        # fingerprints on resume; avoid re-reading the full
                        # 28GB initial checkpoint on every controller restart.
                        return cached_result
            identity = content_identity(checkpoint, exclude=("export_manifest.json",))
            protocol = benchmark_protocol(self.config)
            manifest = {"schema_version": 1, "checkpoint": identity, "protocol": protocol}
            publish_immutable_json(directory / "manifest.json", manifest)
            state_path = directory / "state.json"
            try:
                write_json(state_path, {"status": "RUNNING", "manifest_digest": digest(manifest)})
                # Completed jobs validate their published artifacts on each
                # observation, so a cached aggregate cannot hide corruption.
                result = run_benchmark_evaluation(
                    checkpoint=checkpoint, checkpoint_digest=identity["sha256"],
                    round_dir=directory, iteration=0, config=self.config,
                )
            except Exception as exc:
                write_json(state_path, {
                    "status": "WAITING" if isinstance(exc, JobObservationTimeout) else "FAILED",
                    "manifest_digest": digest(manifest), "error_type": type(exc).__name__,
                })
                raise
            if result.get("checkpoint_digest") != identity["sha256"] or result.get("simulated") is not False:
                raise RealBackendError("baseline_result_identity_mismatch")
            write_json(state_path, {"status": "COMPLETED", "manifest_digest": digest(manifest),
                                    "result_digest": digest(result)})
            return result

    def run_round(self, iteration: int) -> dict[str, Any]:
        with file_lock(self._round_dir(iteration) / "controller.lock", blocking=False):
            return self._run_round(iteration)

    def _run_round(self, iteration: int) -> dict[str, Any]:
        round_dir = self._round_dir(iteration)
        # Keep the per-round layout stable even for candidate_pool smoke runs;
        # the run_pipeline backend writes its workspace into this directory.
        (round_dir / "pipeline_output").mkdir(parents=True, exist_ok=True)
        state = self._load_state(iteration)
        policy, policy_source = self._policy_for_round(iteration)
        policy_path = round_dir / "policy.json"
        if not policy_path.exists():
            write_json(policy_path, policy)
        previous_feedback = self._previous_feedback(iteration)
        write_json(
            round_dir / "round_inputs.json",
            {
                "iteration": iteration,
                "policy_source": policy_source,
                "previous_feedback_consumed": previous_feedback is not None,
                "previous_feedback_digest": digest(previous_feedback) if previous_feedback is not None else None,
            },
        )

        self._advance(iteration, state, "PLANNED", lambda: {"policy_source": policy_source})

        inventory_path = round_dir / "inventory.jsonl"
        inventory_report_path = round_dir / "inventory_report.json"

        def inventory() -> dict[str, Any]:
            rows = self._load_candidates()
            write_jsonl(inventory_path, rows)
            write_jsonl(round_dir / "source_inventory.jsonl", rows)
            report = audit_distribution(rows)
            report.update(
                {
                    "candidate_pool": str(self.candidate_pool),
                    "candidate_pool_sha256": _sha256_file(self.candidate_pool),
                    "source_hashes_present": sum(bool(row.get("source_sha256")) for row in rows),
                }
            )
            write_json(inventory_report_path, report)
            return report

        self._advance(iteration, state, "INVENTORIED", inventory)

        generated_path = round_dir / "generated_candidates.jsonl"
        candidate_pool_path = round_dir / "candidate_pool.jsonl"

        def generated() -> dict[str, Any]:
            if self.generation_backend == "candidate_pool":
                rows = read_jsonl(inventory_path)
                backend = "existing_candidate_pool"
                generated_count = 0
            elif self.generation_backend == "run_pipeline":
                rows = self._run_pipeline_generation(
                    round_dir,
                    read_json(policy_path),
                    iteration=iteration,
                )
                backend = "run_pipeline"
                generated_count = len(rows)
                write_jsonl(round_dir / "pipeline_generated_candidates.jsonl", rows)
                rows = self._supplement_candidates(rows, round_dir)
            else:
                raise ValueError(f"unknown_generation_backend:{self.generation_backend}")
            write_jsonl(generated_path, rows)
            write_jsonl(candidate_pool_path, rows)
            result = {"backend": backend, "generated": generated_count, "forwarded": len(rows)}
            write_json(round_dir / "generation_report.json", result)
            return result

        self._advance(iteration, state, "GENERATED", generated)

        selection: dict[str, Any] = {}

        def distribution() -> dict[str, Any]:
            nonlocal selection
            rows = read_jsonl(generated_path)
            policy_for_selection = read_json(policy_path)
            selection = select_records(
                rows,
                policy_for_selection,
                target_train_samples=self.target_train_samples,
                actual_sample_cap=self.actual_sample_cap,
                seed=self.seed + iteration,
            )
            report = selection["report"]
            write_json(round_dir / "distribution_report.json", report)
            write_jsonl(round_dir / "gate_c_selection.jsonl", report.get("selection_audit", []))
            write_jsonl(round_dir / "source_schedule.jsonl", report.get("source_schedule", []))
            write_json(
                round_dir / "source_scheduler_report.json",
                {
                    "iteration": iteration,
                    "selection_seed": self.seed + iteration,
                    "source_count": len(report.get("source_schedule", [])),
                    "selected_source_count": sum(
                        1 for row in report.get("source_schedule", []) if row.get("selected")
                    ),
                    "unknown_domain_source_count": sum(
                        1 for row in report.get("source_schedule", []) if not row.get("policy_domain")
                    ),
                    "domain_audit_status_counts": report.get("domain_audit_status_counts", {}),
                    "schedule": report.get("source_schedule", []),
                },
            )
            write_jsonl(round_dir / "selected_candidates.jsonl", selection["selected"])
            write_jsonl(round_dir / "withheld_candidates.jsonl", selection["withheld"])
            return report

        self._advance(iteration, state, "DISTRIBUTION_AUDITED", distribution)
        if not selection:
            selection = {
                "selected": read_jsonl(round_dir / "selected_candidates.jsonl"),
                "withheld": read_jsonl(round_dir / "withheld_candidates.jsonl"),
                "report": read_json(round_dir / "distribution_report.json"),
            }

        train_manifest_path = round_dir / "train_manifest.jsonl"
        train_manifest_meta_path = round_dir / "train_manifest.json"

        def train_manifest() -> dict[str, Any]:
            rows = []
            for row in selection["selected"]:
                rows.append(
                    {
                        "iteration": iteration,
                        "record_key": str(row["record_key"]),
                        "candidate": row,
                        "source_pointer": row.get("source_pointer"),
                        "source_key": source_key(row),
                        "source_sha256": row.get("source_sha256"),
                        "capability": row.get("capability"),
                        "task_type": row.get("task_type"),
                        "policy_domain": row.get("policy_domain"),
                    }
                )
            write_jsonl(train_manifest_path, rows)
            metadata = {
                "iteration": iteration,
                "target_train_samples": self.target_train_samples,
                "actual_samples": len(rows),
                "shortage": len(rows) < self.target_train_samples,
                "pool_size": selection["report"].get("pool_size", 0),
                "record_keys_unique": len({row["record_key"] for row in rows}) == len(rows),
                "source_pointers": [row.get("source_pointer") for row in rows],
                "source_keys": [source_key(row) for row in selection["selected"]],
                "init_checkpoint": self.config.get("trainer", {}).get("init_checkpoint"),
                "optimizer_steps": 0,
                "records_path": str(train_manifest_path),
                "record_keys": [row["record_key"] for row in rows],
            }
            validate_manifest_metadata(metadata)
            write_json(train_manifest_meta_path, metadata)
            if self.trainer_backend in {"bagel", "real_smoke"}:
                bagel_cfg = self.config.get("bagel")
                if not isinstance(bagel_cfg, dict):
                    bagel_cfg = self.config.get("trainer", {}).get("bagel", {})
                if not isinstance(bagel_cfg, dict):
                    raise ValueError("bagel_config_missing")
                episodes_path = round_dir / "bagel_episodes.jsonl"
                asset_root = Path(bagel_cfg.get("asset_root", "/"))
                conversion = convert_manifest(
                    train_manifest_path,
                    episodes_path,
                    asset_root=asset_root,
                    require_nine_tasks=bool(bagel_cfg.get("require_nine_tasks", False)),
                    require_paper_rounds=bool(bagel_cfg.get("require_paper_rounds", self.real_smoke)),
                )
                write_json(round_dir / "bagel_data_report.json", conversion)
                metadata["bagel_episodes_path"] = str(episodes_path)
                metadata["bagel_task_counts"] = conversion.get("task_counts", {})
                metadata["bagel_missing_task_types"] = conversion.get("missing_task_types", [])
                write_json(train_manifest_meta_path, metadata)
            return metadata

        self._advance(iteration, state, "TRAIN_MANIFEST_READY", train_manifest)

        train_result_path = round_dir / "train_result.json"

        def train() -> dict[str, Any]:
            if self.trainer_backend == "dry_run":
                result = dry_run_train(train_manifest_path, iteration=iteration, config=self.config)
            elif self.trainer_backend == "smoke_simulator":
                result = smoke_train(
                    train_manifest_path,
                    iteration=iteration,
                    round_dir=round_dir,
                    config=self.config,
                )
                state.setdefault("lifecycle_events", []).extend(
                    ["TRAIN_SUBMITTED", "TRAIN_WAITING", "TRAIN_COMPLETED"]
                )
            elif self.trainer_backend in {"bagel", "real_smoke"}:
                result = run_bagel_train(
                    train_manifest=train_manifest_path,
                    round_dir=round_dir,
                    iteration=iteration,
                    config=self.config,
                )
                state.setdefault("lifecycle_events", []).extend(
                    ["TRAIN_SUBMITTED", "TRAIN_WAITING", "TRAIN_COMPLETED"]
                )
            else:
                raise NotImplementedError(f"trainer_backend_not_implemented:{self.trainer_backend}")
            write_json(train_result_path, result)
            return result

        self._advance(iteration, state, "TRAIN_DRY_RUN", train)

        export_result_path = round_dir / "export_result.json"

        def export() -> dict[str, Any]:
            train_result = read_json(train_result_path)
            if self.trainer_backend in {"bagel", "real_smoke"}:
                result = run_bagel_export(
                    train_result=train_result,
                    round_dir=round_dir,
                    config=self.config,
                )
                state.setdefault("lifecycle_events", []).extend(
                    ["EXPORT_SUBMITTED", "EXPORT_WAITING", "EXPORT_COMPLETED"]
                )
            else:
                result = {
                    "schema_version": 1,
                    "backend": "skipped_for_non_real_trainer",
                    "simulated": self.trainer_backend == "smoke_simulator",
                    "checkpoint_digest": train_result.get("checkpoint_digest"),
                    "output_dir": train_result.get("output_checkpoint"),
                }
            write_json(export_result_path, result)
            return result

        self._advance(iteration, state, "EXPORT_DRY_RUN", export)

        eval_result_path = round_dir / "eval_result.json"

        def evaluate() -> dict[str, Any]:
            train_result = read_json(train_result_path)
            if self.evaluator_backend == "fixture":
                result = evaluate_fixture(iteration=iteration, distribution=selection["report"])
            elif self.evaluator_backend == "smoke_simulator":
                result = smoke_evaluate(
                    iteration=iteration,
                    distribution=selection["report"],
                    train_result=train_result,
                    policy=read_json(policy_path),
                    round_dir=round_dir,
                    config=self.config,
                )
                state.setdefault("lifecycle_events", []).extend(
                    ["EVAL_SUBMITTED", "EVAL_WAITING", "EVAL_COMPLETED"]
                )
            elif self.evaluator_backend in {"benchmark", "real_smoke"}:
                exported = read_json(export_result_path)
                if exported.get("simulated") is not False or not exported.get("output_dir"):
                    raise RealBackendError("real_evaluator_requires_exported_checkpoint")
                result = run_benchmark_evaluation(
                    checkpoint=Path(exported["output_dir"]),
                    checkpoint_digest=exported["export_digest"],
                    round_dir=round_dir, iteration=iteration, config=self.config,
                )
                result["loss_curve_summary"] = train_result.get("loss_curve_summary", {"available": False})
                state.setdefault("lifecycle_events", []).extend(
                    ["EVAL_SUBMITTED", "EVAL_WAITING", "EVAL_COMPLETED"]
                )
            else:
                raise NotImplementedError(f"evaluator_backend_not_implemented:{self.evaluator_backend}")
            result["training"] = train_result
            result["export"] = read_json(export_result_path)
            result["consumed_train_result_digest"] = digest(train_result)
            validate_feedback(result)
            write_json(eval_result_path, result)
            write_json(round_dir / "feedback.json", result)
            return result

        self._advance(iteration, state, "EVAL_DRY_RUN", evaluate)

        reflection_path = round_dir / "reflection.json"

        def reflect() -> dict[str, Any]:
            feedback = read_json(eval_result_path)
            history = []
            for prior in range(1, iteration):
                directory = self._round_dir(prior)
                history.append({
                    "iteration": prior,
                    "feedback": reflection_feedback(read_json(directory / "eval_result.json")),
                    "distribution": reflection_distribution(read_json(directory / "distribution_report.json")),
                    "policy_diff": read_json(directory / "reflection.json").get("policy_diff", {}),
                })
            payload = {
                "iteration": iteration,
                "policy": read_json(policy_path),
                "feedback": reflection_feedback(feedback),
                "compressed_history": history,
                "distribution_report": reflection_distribution(selection["report"]),
            }
            baseline = self.output_root / "baseline" / "eval_result.json"
            if iteration == 1 and baseline.exists():
                payload["baseline"] = reflection_feedback(read_json(baseline))
            if self.reflection_backend == "fixture":
                reflection = fixture_reflection(iteration, read_json(policy_path), feedback)
            elif self.reflection_backend == "deepseek":
                try:
                    reflection = deepseek_reflection(self._reflection_config(), payload)
                    validate_reflection(read_json(policy_path), reflection)
                except Exception as exc:
                    write_json(round_dir / "reflection_pending.json", {
                        "status": "WAITING", "backend": "deepseek",
                        "error_type": type(exc).__name__,
                        "input_feedback_digest": digest(feedback),
                    })
                    # Keep raw provider exceptions out of persisted state:
                    # they can contain request headers or key-bearing URLs.
                    raise LoopPending(f"reflection_retry_required:{type(exc).__name__}") from None
            else:
                raise ValueError(f"unknown_reflection_backend:{self.reflection_backend}")
            current = read_json(policy_path)
            updated = next_policy(current, reflection)
            reflection["next_policy"] = updated
            reflection["policy_diff"] = policy_diff(current, updated)
            reflection["input_feedback_digest"] = digest(feedback)
            write_json(reflection_path, reflection)
            pending_path = round_dir / "reflection_pending.json"
            if pending_path.exists():
                pending = read_json(pending_path)
                pending.update(status="RESOLVED", reflection_digest=digest(reflection))
                write_json(pending_path, pending)
            overlay = apply_config_overlay(self.config, updated)
            write_json(round_dir / "config_overlay.json", _jsonable(overlay))
            return reflection

        self._advance(iteration, state, "REFLECTED", reflect)

        state["status"] = "COMPLETED"
        state["completed"] = True
        state["summary"] = {
            "actual_samples": read_json(train_manifest_meta_path)["actual_samples"],
            "target_train_samples": self.target_train_samples,
            "optimizer_steps": read_json(train_result_path)["optimizer_steps"],
            "policy_diff_count": len(read_json(reflection_path).get("policy_diff", {})),
        }
        if iteration > 1 and (self._round_dir(iteration - 1) / "policy.json").exists():
            previous_policy = read_json(self._round_dir(iteration - 1) / "policy.json")
            state["summary"]["policy_diff_from_previous_count"] = len(
                policy_diff(previous_policy, read_json(policy_path))
            )
        else:
            state["summary"]["policy_diff_from_previous_count"] = None
        self._save_state(iteration, state)
        write_json(
            round_dir / "iteration_manifest.json",
            {
                "iteration": iteration,
                "status": state["status"],
                "state_path": str(self._state_path(iteration)),
                "policy_path": str(policy_path),
                "source_inventory_path": str(round_dir / "source_inventory.jsonl"),
                "candidate_pool_path": str(candidate_pool_path),
                "distribution_report_path": str(round_dir / "distribution_report.json"),
                "source_schedule_path": str(round_dir / "source_schedule.jsonl"),
                "source_scheduler_report_path": str(round_dir / "source_scheduler_report.json"),
                "pre_generation_source_schedule_path": (
                    str(round_dir / "pre_generation_source_schedule.jsonl")
                    if (round_dir / "pre_generation_source_schedule.jsonl").exists()
                    else None
                ),
                "train_manifest_path": str(train_manifest_path),
                "train_result_path": str(train_result_path),
                "feedback_path": str(round_dir / "feedback.json"),
                "reflection_path": str(reflection_path),
                "train_job_path": str(round_dir / "train_job.json") if (round_dir / "train_job.json").exists() else None,
                "eval_job_path": str(round_dir / "eval_job.json") if (round_dir / "eval_job.json").exists() else None,
                "lifecycle_events": state.get("lifecycle_events", []),
                "policy_diff_from_previous_count": state["summary"]["policy_diff_from_previous_count"],
            },
        )
        return {
            "iteration": iteration,
            "status": state["status"],
            "actual_samples": state["summary"]["actual_samples"],
            "target_train_samples": self.target_train_samples,
            "previous_feedback_consumed": previous_feedback is not None,
            "policy_diff_count": state["summary"]["policy_diff_count"],
            "policy_diff_from_previous_count": state["summary"]["policy_diff_from_previous_count"],
            "round_dir": str(round_dir),
        }

    def run(self) -> dict[str, Any]:
        baseline = self._ensure_baseline()
        rounds = [self.run_round(iteration) for iteration in range(1, self.rounds + 1)]
        checkpoints = []
        for iteration in range(1, self.rounds + 1):
            train_path = self._round_dir(iteration) / "train_result.json"
            if train_path.exists():
                checkpoints.append(read_json(train_path).get("init_checkpoint"))
        same_checkpoint = len(set(checkpoints)) <= 1
        if not same_checkpoint:
            raise ValueError("iteration_init_checkpoint_changed")
        summary = {
            "status": "COMPLETED" if all(item["status"] == "COMPLETED" for item in rounds) else "INCOMPLETE",
            # The persistent job contract uses a boolean completion gate in
            # addition to the human-readable status string.
            "completed": all(item["status"] == "COMPLETED" for item in rounds),
            "rounds": rounds,
            "output_root": str(self.output_root),
            "target_train_samples": self.target_train_samples,
            "actual_sample_cap": self.actual_sample_cap,
            "same_init_checkpoint_recorded": same_checkpoint,
            "paper_claim": "smoke_chain_only",
            "baseline_digest": digest(baseline) if baseline is not None else None,
        }
        write_json(self.output_root / "summary.json", summary)
        return summary


__all__ = ["IterationController", "STAGES", "normalize_candidate"]
