#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
orchestrator/planner.py

PipelinePlanner with:
  - persistent heavy models (VLM / LLM / BLIP2 / YOLO / OCR / ASR)
    - fixed-length pre-splitting for long videos (configurable)
    - semantic segmentation across pre-split segments using:
      * VLM global summaries (semantic features)
      * RAFT deep optical flow between segments
      * TF-IDF + cosine similarity
  - classic T1-T4 task structure preserved

All prompts and model outputs are expected to be English only.
"""

from __future__ import annotations

import os
import json
import re
import math
import copy
import threading
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import numpy as np
import traceback 
import subprocess
import shlex
import shutil
import glob
from executor.tasks.representative_selector import RepresentativeSelector

def parse_llm_json(text: str) -> Optional[Any]:
    """
    Best-effort extraction of a JSON object or array from an LLM response.

    Strategy:
      1) Try ``json.loads`` on the full text.
      2) Try extracting a ```json ... ``` fenced block.
      3) Try the substring from the first ``{`` to the last ``}``.
      4) Try the substring from the first ``[`` to the last ``]``.
    Returns the parsed value on success, otherwise ``None``.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    # 1) Try parsing the full response directly.
    try:
        import json
        return json.loads(text)
    except Exception:
        pass

    # 2) Try a fenced ```json ... ``` block.
    import re
    m = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) Try the outermost JSON object span when prose wraps the payload.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4) Try the outermost JSON array span.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None



from executor.tasks.yolov8_detect import YoloDetector
try:
    from executor.tasks.blip2_caption import Blip2Captioner
except Exception as e:
    Blip2Captioner = None
    print(f"[PipelinePlanner] BLIP2 import skipped: {e}")
try:
    from executor.models.qwen_llm import QwenLLM
except Exception as e:
    QwenLLM = None
    print(f"[PipelinePlanner] QwenLLM import skipped: {e}")
from executor.models.qwen_vlm import QwenVLM
from executor.tasks.keyframe_selector import KeyframeSelector
from dataset.builder import DatasetBuilder
try:
    from executor.tasks.image_embedding_diversity import ImageEmbeddingDiversity
except Exception as e:
    ImageEmbeddingDiversity = None
    print(f"[PipelinePlanner] ImageEmbeddingDiversity import skipped: {e}")
from executor.runtime.model_manager import ModelManager
from executor.runtime.chat_logger import ChatTranscriptLogger
from executor.tasks.shot_profiler import ShotProfiler
from executor.tasks.chapter_builder import ChapterBuilder
from executor.tasks.structure_profiler import StructureProfiler
from executor.tasks.quota_planner import QuotaPlanner
from executor.tasks.tool_planner import ToolPlanner
from executor.tasks.tool_executor import ToolExecutor
from executor.tasks.frame_optimizer import FrameOptimizer
from executor.tasks.recall_reviewer import RecallReviewer
from executor.tasks.gate_a_shot_admission import GateAShotAdmission
from executor.tasks.gate_b_quality import GateBQuality
from executor.tasks.frame_signal_semanticizer import FrameSignalSemanticizer
from executor.tasks.llm_value_scorer import LLMValueScorer
from executor.tasks.gate_c_diversity import GateCDiversity
from orchestrator.planner_helpers import (
    build_t1_debug_pack,
    chapters_to_split_points,
    chapter_from_jsonable,
    decide_total_frame_budget,
    estimate_structure_confidence,
    ensure_stage_workdir,
    prepare_t1_frame_items as build_t1_frame_items,
    prepare_t2_frame_pool as build_t2_frame_pool,
    shot_from_jsonable,
    to_jsonable_record,
    video_profile_from_jsonable,
)
from orchestrator.t1_structured import run_t1_structured
from orchestrator.t2_structured import run_t2_structured
from orchestrator.t3_structured import run_t3_structured
from orchestrator.t4_structured import run_t4_structured
from orchestrator.iteration.labels import canonical_iteration_labels, propagate_iteration_labels

# deep optical flow (RAFT) wrapper
try:
    from executor.tasks.optical_flow_raft import RAFTOpticalFlow
except Exception as e:
    RAFTOpticalFlow = None
    print(f"[PipelinePlanner] RAFTOpticalFlow import failed; RAFT backend disabled: {e}")


@dataclass
class SegmentRecord:
    """Record per fixed-length segment during pre-splitting."""

    video_path: str
    segment_path: str
    index: int
    start_sec: float
    end_sec: float
    vlm_summary: str = ""
    vlm_type: str = ""
    tfidf_vector: Optional[np.ndarray] = None
    motion_score: float = 0.0


class PipelinePlanner:
    """
    High-level orchestrator for T1-T4 stages.

    This version keeps heavy models persistent (initialized once) and
    uses fixed-length pre-splitting + semantic segmentation across segments.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # ================ paths & basic config ================
        paths = config.get("paths", {})
        self.temp_root = paths.get("temp_dir", "data/temp")
        self.output_root = paths.get("output_dir", "dataset/output")
        self.semantic_root = paths.get("semantic_dir", "metadata/semantic")
        self.log_root = paths.get("log_dir", "logs/pipeline")

        os.makedirs(self.temp_root, exist_ok=True)
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.semantic_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)
        # ===================== NEW: workspace runtime hints (for path relativization) =====================
        rt = (config.get("runtime", {}) or {}) if isinstance(config, dict) else {}
        self.source_root = rt.get("source_root")  # e.g. Dataset_output/source1
        self.work_root = rt.get("work_root")      # e.g. Dataset_output/source1/sheet0

        # normalize to absolute paths (optional)
        try:
            if self.source_root:
                self.source_root = os.path.abspath(self.source_root)
            if self.work_root:
                self.work_root = os.path.abspath(self.work_root)
        except Exception:
            pass
        # ===================== END NEW =====================

        # dataset builder
        self.dataset_builder = DatasetBuilder(config)
        self._io_lock = threading.RLock()
        self._counter_lock = threading.RLock()
        self._chat_thread_state = threading.local()
        self._concurrent_video_mode = False

        # ===================== [NEW] global benchmark mix counters (across all videos) =====================
        # Accumulates across multiple run_pipeline() calls in the same Python process.
        self._global_bench_counts = {
            "llava_bench": 0,
            "mmt_bench": 0,
            "mme": 0,
            "edit": 0,
            "dialog_visdial_like": 0,
            "dialog_mmdu_like": 0,
            "unify": 0,
            "generation": 0,
        }

        # ===================== [END NEW] =====================

        # ===================== [NEW] global CATEGORY counters (across all videos) =====================
        # For balancing per-benchmark categories (mme/mmt/llava) across multiple run_pipeline() calls.
        self._global_mme_cat_counts: Dict[str, int] = {}
        self._global_mmt_cat_counts: Dict[str, int] = {}
        self._global_llava_cat_counts: Dict[str, int] = {}
        # ===================== [END NEW] =====================

        # ================== core tasks (heavy models are loaded once) ==================
        self.captioner = Blip2Captioner(config) if Blip2Captioner is not None else None

        ks_cfg = (config.get("keyframe_selector", None) or {})
        if not isinstance(ks_cfg, dict):
            ks_cfg = {}
        # The autonomous iteration loop updates keyframe scoring through the
        # runtime policy.  Inject only the small, explicit score multipliers so
        # the existing selector configuration and legacy runs remain intact.
        runtime_policy = ((config.get("runtime", {}) or {}).get("iteration_policy") or {})
        if isinstance(runtime_policy, dict) and isinstance(runtime_policy.get("keyframe_score_weights"), dict):
            ks_cfg = dict(ks_cfg)
            ks_cfg["iteration_score_weights"] = dict(runtime_policy["keyframe_score_weights"])
        try:
            self.keyframe_selector = KeyframeSelector(**ks_cfg)
        except TypeError as e:
            print(f"[PipelinePlanner] keyframe_selector config ignored due to TypeError: {e}")
            self.keyframe_selector = KeyframeSelector()
        
        self.model_manager = ModelManager(config)

        self.modules = type("PlannerModules", (), {})()
        self.modules.shot_profiler = ShotProfiler(config)
        self.modules.chapter_builder = ChapterBuilder(config)
        self.modules.structure_profiler = StructureProfiler(config)
        self.modules.quota_planner = QuotaPlanner(config)
        self.modules.tool_planner = ToolPlanner(config)
        self.modules.tool_executor = ToolExecutor(config)
        self.modules.frame_optimizer = FrameOptimizer(config)
        self.modules.recall_reviewer = RecallReviewer(config)
        self.modules.gate_a_shot_admission = GateAShotAdmission(config)
        self.modules.gate_b_quality = GateBQuality(config)
        self.modules.frame_signal_semanticizer = FrameSignalSemanticizer(config)
        self.modules.llm_value_scorer = LLMValueScorer(config)
        self.modules.gate_c_diversity = GateCDiversity(config)
        self.modules.representative_selector = RepresentativeSelector(config)
        
        out_conf = config.get("tasks", {}).get("T4_dataset", {})
        self.dataset_format = out_conf.get("format", "generic_qa")

        # RAFT optical-flow (optional)
        of_cfg = config.get("optical_flow", {})
        self.raft_backend = str(of_cfg.get("backend", "") or "")
        self.raft_enabled = bool(self.raft_backend == "raft" and RAFTOpticalFlow is not None)
        if not self.raft_enabled:
            print("[PipelinePlanner] RAFT optical-flow disabled by config.")

    @property
    def llm(self):
        return self.model_manager.get_resource("llm")

    @property
    def vlm(self):
        return self.model_manager.get_resource("vlm")

    @property
    def detector(self):
        try:
            return self.model_manager.get_resource("yolo")
        except Exception:
            return None

    @property
    def img_embed(self):
        try:
            return self.model_manager.get_resource("embedder")
        except Exception:
            return None
        
    @property
    def text_embed(self):
        try:
            embedder = self.model_manager.get_resource("embedder")
            if not hasattr(embedder, "compute_text_embeddings"):
                return None
            return embedder
        except Exception:
            return None
    @property
    def asr(self):
        try:
            return self.model_manager.get_resource("asr")
        except Exception:
            return None

    @property
    def ocr(self):
        try:
            return self.model_manager.get_resource("ocr")
        except Exception:
            return None
        
    def on_segment_end(self) -> None:
        if getattr(self, "model_manager", None) is None:
            return
        if self._concurrent_video_mode:
            return
        self.model_manager.on_segment_end()

    def on_video_end(self) -> None:
        self._clear_chat_logging()
        if getattr(self, "model_manager", None) is None:
            return
        if self._concurrent_video_mode:
            return
        self.model_manager.on_video_end()

    def enable_concurrent_video_mode(self, enabled: bool) -> None:
        self._concurrent_video_mode = bool(enabled)
        if getattr(self, "model_manager", None) is not None:
            self.model_manager.set_scope_release_suspended(self._concurrent_video_mode)

    def warm_vlm_pool(self) -> None:
        if getattr(self, "model_manager", None) is None:
            return
        self.model_manager.warm_vlm_pool()

    def acquire_video_vlm_worker(self, video_path: str = "") -> None:
        if getattr(self, "model_manager", None) is None:
            return
        self.model_manager.acquire_video_vlm_slot(video_path=video_path)

    def release_video_vlm_worker(self) -> None:
        if getattr(self, "model_manager", None) is None:
            return
        self.model_manager.release_video_vlm_slot()

    def close(self) -> None:
        self._clear_chat_logging()
        try:
            if getattr(self, "dataset_builder", None) is not None:
                self.dataset_builder.close()
        except Exception:
            pass
        if getattr(self, "model_manager", None) is not None:
            self.model_manager.set_scope_release_suspended(False)
            self.model_manager.close_all()

    def get_global_bench_counts_snapshot(self) -> Dict[str, int]:
        with self._counter_lock:
            return dict(getattr(self, "_global_bench_counts", {}) or {})

    def get_global_category_counts_snapshot(self, benchmark_name: str) -> Dict[str, int]:
        attr_name = {
            "mme": "_global_mme_cat_counts",
            "mmt_bench": "_global_mmt_cat_counts",
            "llava_bench": "_global_llava_cat_counts",
        }.get(str(benchmark_name or "").strip())
        if not attr_name:
            return {}
        with self._counter_lock:
            return dict(getattr(self, attr_name, {}) or {})

    def bump_global_item_category(self, benchmark_name: str, item_obj: Dict[str, Any]) -> None:
        if not isinstance(item_obj, dict):
            return
        category = str(item_obj.get("category", "") or "").strip().lower()
        if not category:
            return
        attr_name = {
            "mme": "_global_mme_cat_counts",
            "mmt_bench": "_global_mmt_cat_counts",
            "llava_bench": "_global_llava_cat_counts",
        }.get(str(benchmark_name or "").strip())
        if not attr_name:
            return
        with self._counter_lock:
            counter = getattr(self, attr_name, None)
            if not isinstance(counter, dict):
                return
            counter[category] = int(counter.get(category, 0) or 0) + 1

    def _run_single_vlm_chat(
        self,
        images: Any,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if getattr(self, "model_manager", None) is None:
            if max_new_tokens is None:
                return str(self.vlm.chat(images, prompt) or "")
            return str(self.vlm.chat(images, prompt, max_new_tokens=max_new_tokens) or "")

        def _worker(single_vlm: Any, job: Dict[str, Any]) -> str:
            if job.get("max_new_tokens") is None:
                return str(single_vlm.chat(job["images"], job["prompt"]) or "")
            return str(
                single_vlm.chat(
                    job["images"],
                    job["prompt"],
                    max_new_tokens=int(job["max_new_tokens"]),
                )
                or ""
            )

        result = self.model_manager.run_vlm_jobs(
            [
                {
                    "images": images,
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                }
            ],
            worker_fn=_worker,
            max_workers=1,
        )
        return str(result[0] if result else "" or "")

    def _chat_log_path(self, video_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(self.log_root, f"{base}_chat_history.jsonl")

    def _set_chat_logging(
        self,
        video_path: str,
        stage_name: str,
        segment: Optional[SegmentRecord] = None,
    ) -> None:
        log_path = self._chat_log_path(video_path)
        logger = getattr(self._chat_thread_state, "logger", None)
        logger_path = getattr(self._chat_thread_state, "logger_path", None)
        if logger is None or logger_path != log_path:
            logger = ChatTranscriptLogger(log_path)
            self._chat_thread_state.logger = logger
            self._chat_thread_state.logger_path = log_path

        context = {
            "video_path": str(video_path),
            "stage": str(stage_name or ""),
        }
        if segment is not None:
            context["segment_index"] = int(segment.index)
            context["segment_path"] = str(segment.segment_path)
            context["segment_start_sec"] = float(segment.start_sec)
            context["segment_end_sec"] = float(segment.end_sec)

        self.model_manager.set_chat_logger(logger)
        self.model_manager.set_chat_context(context)

    def _clear_chat_logging(self) -> None:
        if getattr(self, "model_manager", None) is None:
            return
        self.model_manager.clear_chat_context()
        self.model_manager.clear_chat_logger()
        self._chat_thread_state.logger = None
        self._chat_thread_state.logger_path = None

    def _get_raft_flow(self):
        raft = getattr(self, "raft_flow", None)
        if raft is not None:
            return raft
        if not getattr(self, "raft_enabled", False):
            return None
        try:
            return self.model_manager.get_resource("raft")
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------
    # ===================== NEW: relativize path fields for json/jsonl portability =====================
    def _relativize_path(self, p: Any) -> Any:
        """
        Convert absolute path -> relative-to self.source_root if possible.
        Keep non-string and empty values unchanged.
        Always use '/' separators in output for portability.
        """
        import os

        if not self.source_root:
            return p
        if not isinstance(p, str) or not p:
            return p

        try:
            ap = os.path.abspath(p)
            base = self.source_root

            # only relativize when ap is under base
            if ap == base or ap.startswith(base + os.sep):
                rp = os.path.relpath(ap, base)
                return rp.replace("\\", "/")
            return p
        except Exception:
            return p

    def _relativize_item_paths_inplace(self, obj: Any) -> Any:
        """
        In-place (but safe) conversion for known path-like fields inside item/meta.
        We do NOT touch arbitrary strings; only keys that are likely paths.
        """
        PATH_KEYS = {
            "video_path", "segment_path",
            "image", "image_path",
            "source_img", "target_img", "mask_img",
            "audio_path", "pdf_path", "preview_pdf", "preview_path",
            "log_path", "json_path", "jsonl_path",
            "frame_path", "img",  # frames often store {"img": "..."}
        }

        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k in PATH_KEYS or k.endswith("_path") or k.endswith("_dir"):
                    obj[k] = self._relativize_path(v)
                else:
                    self._relativize_item_paths_inplace(v)
            return obj

        if isinstance(obj, list):
            for i in range(len(obj)):
                self._relativize_item_paths_inplace(obj[i])
            return obj

        return obj
    # ===================== END NEW =====================

    def _cuda_cleanup(self, tag: str = "", *, aggressive: bool = False) -> None:
        """
        Best-effort CUDA/PyTorch memory cleanup to mitigate fragmentation.
        - aggressive=False: safe, cheap
        - aggressive=True: includes ipc_collect + synchronize (slower)
        """
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        try:
            import torch
        except Exception:
            return

        if not torch.cuda.is_available():
            return

        try:
            # release PyTorch cached blocks back to the CUDA driver
            torch.cuda.empty_cache()
        except Exception:
            pass

        if aggressive:
            try:
                # clear IPC-related memory (useful sometimes in multiprocess / conda-run scenarios)
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                # synchronize to ensure pending kernels are done before future big allocs
                torch.cuda.synchronize()
            except Exception:
                pass

        # Optional: debug print current allocator stats
        try:
            if getattr(self, "debug", False) or getattr(self, "verbose", False):
                reserved = float(torch.cuda.memory_reserved()) / (1024**3)
                allocated = float(torch.cuda.memory_allocated()) / (1024**3)
                self._dbg(f"[cuda_cleanup]{'['+tag+']' if tag else ''} allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB aggressive={aggressive}")
        except Exception:
            pass

    # =====================================================================
    # Helper: probe and optionally repair a video with ffmpeg
    # =====================================================================
    #DEPRECATED!!!
    # def _probe_and_maybe_repair_video(self, src_path: str) -> str:
    #     """
    #     Probe a source video and repair it only when needed.
    #     - If probing succeeds, return the original ``src_path``.
    #     - If probing fails, re-encode into a sibling ``*_fixed.mp4`` file and return it.
    #     - If repair also fails, warn and fall back to the original ``src_path``.
    #     """
    #     import os
    #     import subprocess
    #     import shlex

    #     if not os.path.exists(src_path):
    #         print(f"[PreFix] source video not found, skip repair: {src_path}")
    #         return src_path

    #     # 1) Probe the first few seconds with ffmpeg and only inspect the return code.
    #     probe_cmd = [
    #         "ffmpeg",
    #         "-v", "error",          # only emit actual errors
    #         "-read_intervals", "0%+3",  # decode the first 3 seconds
    #         "-i", src_path,
    #         "-f", "null",
    #         "-"                      # null sink
    #     ]
    #     try:
    #         print(f"[PreFix] Probing video with ffmpeg: {src_path}")
    #         probe_res = subprocess.run(
    #             probe_cmd,
    #             stdout=subprocess.DEVNULL,
    #             stderr=subprocess.PIPE,
    #             check=False,
    #         )
    #     except Exception as e:
    #         print(f"[PreFix] ffmpeg probe failed unexpectedly: {e}")
    #         # Be conservative and keep using the original source on unexpected probe failures.
    #         return src_path

    #     if probe_res.returncode == 0:
    #         # Probe passed, so no repair is needed.
    #         print("[PreFix] Probe OK, skip repair.")
    #         return src_path

    #     print("[PreFix] Probe FAILED (ffmpeg returncode != 0), try to repair with re-encode.")

    #     # 2) Build the sibling repaired path.
    #     root, ext = os.path.splitext(src_path)
    #     fixed_path = root + "_fixed" + (ext or ".mp4")

    #     # Reuse an existing repaired file if one is already present.
    #     if os.path.exists(fixed_path):
    #         print(f"[PreFix] Found existing fixed video, use it: {fixed_path}")
    #         return fixed_path

    #     repair_cmd = [
    #         "ffmpeg",
    #         "-y",                      # overwrite output
    #         "-err_detect", "ignore_err",
    #         "-i", src_path,
    #         "-c:v", "libx264",
    #         "-preset", "fast",
    #         "-crf", "20",
    #         "-c:a", "aac",
    #         "-b:a", "128k",
    #         fixed_path,
    #     ]
    #     try:
    #         print(f"[PreFix] Re-encoding video to clean bitstream:\n  {' '.join(shlex.quote(c) for c in repair_cmd)}")
    #         repair_res = subprocess.run(
    #             repair_cmd,
    #             stdout=subprocess.DEVNULL,
    #             stderr=subprocess.STDOUT,
    #             check=True,
    #         )
    #         print(f"[PreFix] Repair succeeded, use: {fixed_path}")
    #         return fixed_path
    #     except Exception as e:
    #         print(f"[PreFix] Repair FAILED, fallback to original video. error={e}")
    #         return src_path

    def _video_id(self, video_path: str) -> str:
        """
        Derive a stable video id from the basename without the file extension.
        Example: ``data/input/-17SPG-3Jis.mp4`` -> ``-17SPG-3Jis``.
        """
        base = os.path.basename(video_path)      # "-17SPG-3Jis.mp4"
        vid, _ = os.path.splitext(base)         # ("-17SPG-3Jis", ".mp4")
        return vid

    def _segment_dir(self, video_path: str, segment_seconds: int) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(self.temp_root, base, f"segments_{int(segment_seconds)}s")


    def _segment_path(self, video_path: str, segment_seconds: int, seg_idx: int) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(
            self._segment_dir(video_path, segment_seconds),
            f"{base}_{seg_idx:03d}.mp4",
        )


    def _semantic_out_path(self, video_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(self.semantic_root, f"{base}_semantic.json")

    def _log_path(self, video_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(self.log_root, f"{base}_segments.jsonl")

    def _stage_debug_path(self, video_path: str, segment_index: int, stage_name: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]
        debug_root = (self.config.get("paths", {}) or {}).get("debug_dir", "logs/debug")
        os.makedirs(debug_root, exist_ok=True)
        return os.path.join(debug_root, f"{base}_seg{int(segment_index):03d}_{stage_name}.json")

    def _dump_stage_debug(
        self,
        *,
        video_path: str,
        segment_index: int,
        stage_name: str,
        payload: Any,
    ) -> str:
        debug_path = self._stage_debug_path(video_path, segment_index, stage_name)
        try:
            safe_payload = copy.deepcopy(payload)
        except Exception:
            safe_payload = payload
        try:
            safe_payload = self._relativize_item_paths_inplace(safe_payload)
        except Exception:
            pass
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(safe_payload, f, ensure_ascii=False, indent=2, default=str)
        return debug_path

    def _dump_stage_error(
        self,
        *,
        video_path: str,
        segment_index: int,
        stage_name: str,
        exc: BaseException,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "stage": stage_name,
            "segment_index": int(segment_index),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }
        return self._dump_stage_debug(
            video_path=video_path,
            segment_index=segment_index,
            stage_name=f"{stage_name}_error",
            payload=payload,
        )

    def _ensure_t1_workdir(self, video_id: str, seg_id: str) -> str:
        return ensure_stage_workdir(self.temp_root, video_id, seg_id, "T1")

    def _ensure_t2_workdir(self, video_id: str, seg_id: str) -> str:
        return ensure_stage_workdir(self.temp_root, video_id, seg_id, "T2")

    def _to_jsonable_shot(self, shot: Any) -> Dict[str, Any]:
        return to_jsonable_record(shot)

    def _to_jsonable_chapter(self, chapter: Any) -> Dict[str, Any]:
        return to_jsonable_record(chapter)

    def _to_jsonable_video_profile(self, vp: Any) -> Dict[str, Any]:
        return to_jsonable_record(vp)

    def _from_jsonable_shot(self, obj: Dict[str, Any]) -> Any:
        return shot_from_jsonable(obj)

    def _from_jsonable_chapter(self, obj: Dict[str, Any]) -> Any:
        return chapter_from_jsonable(obj)

    def _from_jsonable_video_profile(self, obj: Dict[str, Any]) -> Any:
        return video_profile_from_jsonable(obj)

    def _prepare_t1_frame_items(
        self,
        video_path: str,
        t0_frame_items: List[Dict[str, Any]],
        raw_shots: List[Dict[str, Any]],
        t1_workdir: str,
    ) -> List[Dict[str, Any]]:
        return build_t1_frame_items(
            video_path=video_path,
            t0_frame_items=t0_frame_items,
            raw_shots=raw_shots,
            t1_workdir=t1_workdir,
        )

    def _chapters_to_split_points(
        self,
        chapters: List[Any],
        segment_start_sec: float,
        segment_end_sec: float,
    ) -> List[float]:
        return chapters_to_split_points(chapters, segment_start_sec, segment_end_sec)

    def _estimate_structure_confidence(
        self,
        shots: List[Any],
        chapters: List[Any],
        pairwise_merge_decisions: List[Dict[str, Any]],
        smoothing_meta: Dict[str, Any],
    ) -> float:
        _ = (shots, smoothing_meta)
        return estimate_structure_confidence(chapters, pairwise_merge_decisions)

    def _build_t1_debug_pack(
        self,
        shots: List[Any],
        chapters: List[Any],
        pairwise_merge_decisions: List[Dict[str, Any]],
        smoothing_meta: Dict[str, Any],
        shot_detector_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        return build_t1_debug_pack(shots, chapters, pairwise_merge_decisions, smoothing_meta, shot_detector_meta)

    def _decide_total_frame_budget(
        self,
        t1_stage: Dict[str, Any],
        segment: Any,
        shots: List[Any],
        chapters: List[Any],
        video_profile: Any,
    ) -> int:
        _ = (t1_stage, video_profile)
        return decide_total_frame_budget(segment, shots, chapters)

    def _prepare_t2_frame_pool(
        self,
        video_path: str,
        shots: List[Any],
        quota_plan: Dict[str, Any],
        t2_workdir: str,
    ) -> List[Dict[str, Any]]:
        _ = quota_plan
        return build_t2_frame_pool(
            video_path=video_path,
            shots=shots,
            t2_workdir=t2_workdir,
        )
    
    def _safe_subclip_with_reencode(self, in_path: str, start: float, end: float, out_path: str) -> bool:
        """
        Extract a subclip with ffmpeg via explicit re-encoding.
        This is more robust than MoviePy for damaged inputs, short windows, or timestamp drift.
        """
        import subprocess
        import json
        import os

        # ---------- Probe source duration ----------
        try:
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "json",
                in_path
            ]
            probe_out = subprocess.check_output(probe_cmd).decode("utf-8")
            duration = float(json.loads(probe_out)["format"]["duration"])
        except Exception as e:
            print(f"[T1 WARN] ffprobe failed on {in_path}: {e}")
            return False

        # ---------- Clamp end to the source duration ----------
        end = min(float(end), duration)
        start = float(start)

        dur = end - start  # ffmpeg needs clip duration rather than absolute end time
        if dur < 0.5:
            print(f"[T1 WARN] _safe_subclip: interval too short ({dur:.3f}s), skip.")
            return False

        # ---------- Extract with ffmpeg ----------
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", in_path,
                "-t", str(dur),              # use duration, not absolute end timestamp
                "-c:v", "libx264",           # re-encode for better split stability
                "-c:a", "aac",
                "-movflags", "+faststart",
                out_path
            ]

            timeout_s = max(60.0, min(600.0, float(dur) * 20.0 + 30.0))
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_s)
            if p.returncode != 0:
                print(f"[T1 WARN] ffmpeg failed (returncode={p.returncode}): {out_path}")
                return False

            # Validate the output file before reporting success.
            if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                print(f"[T1 WARN] ffmpeg output invalid: {out_path}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print(f"[T1 WARN] ffmpeg reencode timed out after {timeout_s:.1f}s: {out_path}")
            return False
        except Exception as e:
            print(f"[T1 WARN] ffmpeg reencode failed: {e}")
            return False


    # -------------------------------------------------------------------------
    # Optional: pre-repair a potentially broken source video by re-encoding
    # -------------------------------------------------------------------------
    def _probe_video_ok(self, video_path: str, probe_sec: int = 3) -> bool:
        if not video_path or not os.path.exists(video_path):
            return False

        cmd = [
            "ffmpeg",
            "-v", "error",
            "-read_intervals", f"0%+{probe_sec}",
            "-i", video_path,
            "-f", "null",
            "-"
        ]
        try:
            r = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=probe_sec * 10,
            )
            return r.returncode == 0
        except Exception:
            return False


    def _repair_video_if_needed(self, video_path: str) -> str:
        """
        Only repair when needed.
        If probe ok -> return original video_path.
        If probe fails -> write a sibling file with suffix '_fixed' and return it.
        """
        # 1) Probe first; only repair when the source actually looks broken.
        if self._probe_video_ok(video_path, probe_sec=3):
            return video_path

        # 2) Build a sibling ``_fixed`` path.
        root, ext = os.path.splitext(video_path)
        fixed_path = f"{root}_fixed{ext if ext else '.mp4'}"

        # 3) Reuse an existing repaired file if it already looks valid.
        if os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 1024:
            return fixed_path

        # 4) First try a stream-copy remux, which is faster and preserves quality.
        cmd_copy = [
            "ffmpeg", "-y",
            "-v", "error",
            "-i", video_path,
            "-c", "copy",
            "-movflags", "+faststart",
            fixed_path,
        ]
        r1 = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r1.returncode == 0 and os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 1024:
            return fixed_path

        # 5) If remux fails, fall back to full re-encoding for maximum compatibility.
        cmd_reenc = [
            "ffmpeg", "-y",
            "-v", "error",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            fixed_path,
        ]
        r2 = subprocess.run(cmd_reenc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r2.returncode == 0 and os.path.exists(fixed_path) and os.path.getsize(fixed_path) > 1024:
            return fixed_path

        # 6) If all repairs fail, keep using the original source.
        return video_path

    def _fixed_video_path_for_source(self, video_path: str) -> str:
        root, ext = os.path.splitext(video_path)
        return f"{root}_fixed{ext if ext else '.mp4'}"

    def _cleanup_sibling_fixed_video(self, video_path: str, safe_video_path: str, records: List["SegmentRecord"]) -> None:
        if not safe_video_path or safe_video_path == video_path:
            return
        fixed_path = self._fixed_video_path_for_source(video_path)
        if os.path.normcase(os.path.abspath(safe_video_path)) != os.path.normcase(os.path.abspath(fixed_path)):
            return
        if not records:
            return
        safe_path_in_use = any(
            os.path.normcase(os.path.abspath(str(getattr(x, "segment_path", "") or "")))
            == os.path.normcase(os.path.abspath(safe_video_path))
            for x in records
        )
        if safe_path_in_use:
            return
        try:
            if os.path.exists(safe_video_path):
                os.remove(safe_video_path)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # T0: pre-splitting + basic ASR/OCR on segments
    # -------------------------------------------------------------------------
    def _probe_duration_sec(self, path: str) -> float:
        """Return video duration from ffprobe, or ``0`` on failure."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            return float(out.decode().strip() or "0")
        except Exception:
            return 0.0



    def pre_split_video(self, video_path: str, segment_seconds: int = 30) -> List["SegmentRecord"]:
        """
        Use ffmpeg to pre-split a long video into fixed-length segments.
        Returns a list of SegmentRecord with segment paths & basic metadata.

        MIN-MOD FIX:
        - For AV1 sources, DO NOT use -c copy for segmenting, because segments often start without sequence header.
        That yields undecodable segments and causes [av1] Missing Sequence Header in T0/T2.
        - So: detect codec of safe_video_path; if av1 => force re-encode split.
        """
        seg_dir = self._segment_dir(video_path, segment_seconds)

        # Clean previous segment workdirs such as ``segments_XXs`` and sibling ``seg_*`` folders.
        video_temp_root = os.path.dirname(seg_dir)
        if os.path.exists(video_temp_root):
            if os.path.exists(seg_dir):
                shutil.rmtree(seg_dir, ignore_errors=True)
            for p in glob.glob(os.path.join(video_temp_root, "seg_*")):
                shutil.rmtree(p, ignore_errors=True)
            for p in glob.glob(os.path.join(video_temp_root, "**/*_refine_cache"), recursive=True):
                shutil.rmtree(p, ignore_errors=True)

        os.makedirs(seg_dir, exist_ok=True)

        # Repair the source first if probing suggests the original file is unsafe to segment.
        safe_video_path = self._repair_video_if_needed(video_path)

        # ---- MIN-MOD: detect video codec (critical) ----
        def _probe_video_codec_name(path: str) -> str:
            try:
                cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=nw=1:nk=1",
                    path,
                ]
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
                return (r.stdout or "").strip().lower()
            except Exception:
                return ""

        src_codec = _probe_video_codec_name(safe_video_path)
        force_reencode = (src_codec == "av1")
        if force_reencode:
            print("[pre_split_video] source codec=av1; force re-encode splitting and skip stream copy.")

        # Probe the duration from the safe path.
        duration = self._probe_duration_sec(safe_video_path)

        if duration <= 0:
            print(f"[pre_split_video] invalid duration for {safe_video_path}; falling back to a single segment.")
            return [
                SegmentRecord(
                    video_path=safe_video_path,
                    segment_path=safe_video_path,
                    index=0,
                    start_sec=0.0,
                    end_sec=0.0,
                )
            ]

        num_segments = math.ceil(duration / segment_seconds)
        records: List[SegmentRecord] = []

        print(f"[pre_split_video] segmenting: chunk={segment_seconds}s, duration={duration:.1f}s, segments={num_segments}")

        for i in range(num_segments):
            start = i * segment_seconds
            end = min((i + 1) * segment_seconds, duration)
            dur = max(end - start, 0.0)

            out_path = self._segment_path(video_path, segment_seconds, i)

            if not os.path.exists(out_path):
                print(
                    f"[pre_split_video] splitting segment {i + 1}/{num_segments}: "
                    f"{start:.2f}-{end:.2f}s -> {out_path}",
                    flush=True,
                )
                if force_reencode:
                    # AV1 sources are split by direct re-encoding for stability.
                    ok = self._safe_subclip_with_reencode(safe_video_path, start, end, out_path)
                    if not ok:
                        print(f"[pre_split_video] re-encode failed on segment {i}: {start:.2f}-{end:.2f}")
                        continue
                else:
                    # 1) Try a fast stream-copy split first.
                    cmd_copy = [
                        "ffmpeg",
                        "-y",
                        "-ss", f"{start:.3f}",
                        "-i", safe_video_path,
                        "-t", f"{dur:.3f}",
                        "-c:v", "copy",
                        "-c:a", "copy",
                        "-reset_timestamps", "1",
                        "-avoid_negative_ts", "make_zero",
                        out_path,
                    ]

                    copy_ok = False
                    try:
                        timeout_s = max(60.0, min(300.0, float(dur) * 10.0 + 30.0))
                        subprocess.run(
                            cmd_copy,
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=timeout_s,
                        )
                        copy_ok = (
                            os.path.exists(out_path)
                            and os.path.getsize(out_path) > 0
                            and self._probe_duration_sec(out_path) > 0.05
                        )
                    except subprocess.CalledProcessError:
                        copy_ok = False
                    except subprocess.TimeoutExpired:
                        print(
                            f"[pre_split_video] stream-copy timed out after {timeout_s:.1f}s "
                            f"on segment {i}: {start:.2f}-{end:.2f}; retry re-encode.",
                            flush=True,
                        )
                        copy_ok = False

                    # 2) If stream-copy fails, fall back to re-encoding.
                    if not copy_ok:
                        # Clean up any partial file before retrying.
                        try:
                            if os.path.exists(out_path):
                                os.remove(out_path)
                        except Exception:
                            pass

                        ok = self._safe_subclip_with_reencode(safe_video_path, start, end, out_path)
                        if not ok:
                            print(f"[pre_split_video] both stream-copy and re-encode failed on segment {i}: {start:.2f}-{end:.2f}")
                            continue

            records.append(
                SegmentRecord(
                    video_path=video_path,    # keep the original parent source path
                    segment_path=out_path,    # actual materialized split clip path
                    index=i,
                    start_sec=start,
                    end_sec=end,
                )
            )

        print(f"[pre_split_video] segmentation finished: chunk={segment_seconds}s, clips={len(records)} in {seg_dir}")
        self._cleanup_sibling_fixed_video(video_path, safe_video_path, records)
        return records




    # -------------------------------------------------------------------------
    # RAFT-based motion estimation (optional)
    # -------------------------------------------------------------------------
    def compute_motion_scores_for_frames(
        self,
        frame_paths: List[str],
        sample_step: int = 1,
        resize_short_side: Optional[int] = 384,
    ) -> List[float]:
        """
        Use RAFT to compute optical-flow magnitudes between neighboring frames.

        :param frame_paths: list of frame image paths.
        :param sample_step: frame interval.
        :param resize_short_side: resize short side to this value for speed.
        :return: list of motion scores (len == len(frame_paths) - 1).
        """
        raft_flow = self._get_raft_flow()
        if raft_flow is None or len(frame_paths) < 2:
            print("[PipelinePlanner] RAFT optical-flow not enabled or not enough frames.")
            return []

        return raft_flow.compute_flow_for_frame_paths(
            frame_paths=frame_paths,
            sample_step=sample_step,
            iters=20,
            resize_short_side=resize_short_side,
        )

    # -------------------------------------------------------------------------
    # T1 helpers
    # -------------------------------------------------------------------------

    def _parse_vlm_output(self, vlm_raw: str) -> dict:
        """Parse the structured six-line VLM summary block into a dict."""
        out = {
            "video_type": "other",
            "global_summary": "",
            "modality_hint": "mixed",
            "motion_hint": "medium_motion",
            "suitable_for_dataset": True,
            "dataset_reason": "",
            "should_split": False,
            "split_reason": ""
        }
        if not vlm_raw:
            return out

        lines = [ln.strip() for ln in vlm_raw.splitlines() if ln.strip()]
        fallback_text = vlm_raw.lower()

        # summary
        for ln in lines:
            if ln.lower().startswith("1)") or "summary" in ln.lower():
                out["global_summary"] = ln.split(":", 1)[-1].strip()
                break

        # video_type
        video_type_options = ["lecture", "vlog", "gaming", "surveillance", "drama", "tutorial", "sports", "other"]
        for vt in video_type_options:
            if vt in fallback_text:
                out["video_type"] = vt
                break

        # modality_hint
        if "mostly_speech" in fallback_text or "mostly speech" in fallback_text:
            out["modality_hint"] = "mostly_speech"
        elif "mostly_visual" in fallback_text or "mostly visual" in fallback_text:
            out["modality_hint"] = "mostly_visual"
        elif "mixed" in fallback_text:
            out["modality_hint"] = "mixed"

        # motion_hint
        if "high_motion" in fallback_text or "high motion" in fallback_text:
            out["motion_hint"] = "high_motion"
        elif "low_motion" in fallback_text or "low motion" in fallback_text:
            out["motion_hint"] = "low_motion"
        elif "medium_motion" in fallback_text or "medium motion" in fallback_text:
            out["motion_hint"] = "medium_motion"

        # suitable_for_dataset
        for ln in lines:
            if ln.lower().startswith("5)") or "suitable_for_dataset" in ln.lower():
                if "no" in ln.lower():
                    out["suitable_for_dataset"] = False
                if "-" in ln:
                    out["dataset_reason"] = ln.split("-", 1)[1].strip()
                break

        # should_split
        for ln in lines:
            if ln.lower().startswith("6)") or "should_split" in ln.lower():
                if "yes" in ln.lower():
                    out["should_split"] = True
                if "-" in ln:
                    out["split_reason"] = ln.split("-", 1)[1].strip()
                break

        return out
    
    def _bind_ocr_to_frames(
        self,
        base_frames: List[str],
        base_ts: List[float],
        ocr_raw: Dict[str, Any],
        frame_items: List[Dict[str, Any]],
        max_gap: float = 1.0,
    ) -> List[str]:
        """
        Bind OCR results onto ``frame_items`` as per-frame text.

        It first builds a path-to-text lookup from the OCR output, then tries exact
        matches by full path or basename. If that misses, it falls back to the
        nearest timestamp from ``base_frames`` / ``base_ts`` within ``max_gap``.

        Returns a list of OCR strings aligned with ``frame_items``.
        """
        import os
        import bisect
        from typing import Dict, Any, List, Tuple

        # OCR payload must be a dict; otherwise return empty strings for every frame.
        if not isinstance(ocr_raw, dict):
            return [""] * len(frame_items)

        # 1) Normalize OCR entries into a key -> text map by full path and basename.
        path2text: Dict[str, str] = {}
        for k, v in ocr_raw.items():
            if not isinstance(k, str):
                continue

            if isinstance(v, dict):
                # Simple {"text": "..."} payload.
                text = (v.get("text", "") or "").strip()

            elif isinstance(v, list):
                # PaddleOCR-style payload: list[{"text","confidence"}] or list[str].
                parts: List[str] = []
                for line in v:
                    if isinstance(line, dict):
                        t = (line.get("text", "") or "").strip()
                        if t:
                            parts.append(t)
                    elif isinstance(line, str):
                        t = line.strip()
                        if t:
                            parts.append(t)
                text = "\n".join(parts)

            else:
                text = (str(v) if v is not None else "").strip()

            path2text[k] = text
            path2text[os.path.basename(k)] = text

        # 2) Build timestamp -> text pairs from OCR entries aligned to base frames.
        #    This supports nearest-neighbor fallback when path matching misses.
        ts_text_pairs: List[Tuple[float, str]] = []
        if (
            isinstance(base_frames, list)
            and isinstance(base_ts, list)
            and len(base_frames) == len(base_ts)
            and len(base_frames) > 0
        ):
            for p, t in zip(base_frames, base_ts):
                try:
                    ts = float(t)
                except Exception:
                    continue
                bn = os.path.basename(p) if isinstance(p, str) else ""
                txt = ""
                if isinstance(p, str) and p in path2text:
                    txt = path2text[p]
                elif bn and bn in path2text:
                    txt = path2text[bn]
                ts_text_pairs.append((ts, txt))

            # Keep pairs sorted for binary-search lookup.
            ts_text_pairs.sort(key=lambda x: x[0])

        # Extract sorted timestamps for bisect lookup.
        ts_sorted: List[float] = [x[0] for x in ts_text_pairs]

        def _nearest_text_by_time(sec: Any) -> str:
            """Return OCR text from the nearest base timestamp within ``max_gap``."""
            if not ts_text_pairs:
                return ""
            try:
                s = float(sec)
            except Exception:
                return ""

            i = bisect.bisect_left(ts_sorted, s)
            cand_idxs: List[int] = []
            if 0 <= i < len(ts_sorted):
                cand_idxs.append(i)
            if 0 <= i - 1 < len(ts_sorted):
                cand_idxs.append(i - 1)

            best_txt = ""
            best_gap = None
            for j in cand_idxs:
                gap = abs(ts_sorted[j] - s)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_txt = ts_text_pairs[j][1]

            if best_gap is None or best_gap > float(max_gap):
                return ""
            return best_txt or ""

        # 3) Resolve OCR for each frame by path, basename, then nearest timestamp.
        per_frame_ocr: List[str] = []
        for item in frame_items:
            img_path = item.get("img", "") if isinstance(item, dict) else ""
            basename = os.path.basename(img_path) if isinstance(img_path, str) else ""

            txt = ""
            if isinstance(img_path, str) and img_path in path2text:
                txt = path2text[img_path]
            elif basename and basename in path2text:
                txt = path2text[basename]
            else:
                # Path lookup missed; fall back to the frame timestamp.
                sec = item.get("sec", None) if isinstance(item, dict) else None
                txt = _nearest_text_by_time(sec)

            per_frame_ocr.append(txt)

        return per_frame_ocr



    def _bind_asr_to_frames(
        self,
        asr_segments: List[Dict[str, Any]],
        frame_items: List[Dict[str, Any]],
        delta: float = 1.5,
    ) -> List[str]:
        """
        Bind ASR segments to T2 frame items as per-frame text snippets.
        - ``frame_items`` is a list like ``[{"img": ..., "sec": ...}, ...]``.
        - ``asr_segments`` is typically Whisper output with ``timestamp`` / ``ts`` / ``time``.

        Returns:
            ``per_frame_asr_text`` aligned to ``frame_items``.
        """
        per_frame_asr: List[str] = []

        for f in frame_items:
            t = float(f.get("sec", 0.0))
            pieces = []
            for seg in asr_segments or []:
                ts = seg.get("timestamp") or seg.get("ts") or seg.get("time")
                if not ts or len(ts) < 2:
                    continue
                try:
                    s, e = float(ts[0]), float(ts[1])
                except Exception:
                    continue

                # Keep ASR segments whose time span overlaps the frame-centered window.
                if e >= t - delta and s <= t + delta:
                    txt = (seg.get("text") or "").strip()
                    if txt:
                        pieces.append(txt)

            per_frame_asr.append(" ".join(pieces).strip())

        return per_frame_asr

    def _llm_semantic_score_for_frames(
        self,
        frame_sem_texts: List[str],
        modality_hint: str = "mixed",
    ) -> List[float]:
        """
        Score semantic usefulness of ASR+OCR snippets with the LLM on a 0..1 scale.
        Assumptions:
        - This is only an auxiliary weighting signal, never a hard gate.
        - The returned score list is aligned with ``frame_sem_texts``.
        - If the text is uniformly uninformative, the method falls back to zeros.
        """
        import json
        from typing import Any

        n = len(frame_sem_texts)
        if n == 0:
            return []

        # Default to zeros.
        scores: List[float] = [0.0] * n

        # Keep only sufficiently informative snippets before calling the LLM.
        items = []
        for idx, txt in enumerate(frame_sem_texts):
            t = (txt or "").strip()
            if not t or len(t) < 10:
                # Empty or ultra-short snippets are treated as ``none`` with score 0.
                continue
            # Truncate overly long snippets to keep the prompt stable.
            if len(t) > 300:
                t = t[:300] + " ..."
            items.append({"idx": idx, "text": t})

        if not items:
            # Nothing usable to score, so keep all scores at zero.
            return scores

        # Build the grading prompt and ask the LLM for coarse labels.
        modality_hint_str = {
            "mostly_speech": "Most information is in speech.",
            "mostly_visual": "Most information is in visuals, text is auxiliary.",
            "mixed": "Both speech and visuals carry information.",
        }.get(modality_hint, "Both speech and visuals carry information.")

        prompt = f"""
            You are a quality checker for short text snippets extracted from video (ASR/OCR).
            Your job is to label each snippet with a coarse semantic quality label.

            Context about the video:
            - modality hint: {modality_hint} ({modality_hint_str})

            For each item, decide:
            - "none" : meaningless noise, random characters, or totally irrelevant.
            - "poor" : somewhat meaningful but very noisy, fragmented, or unclear.
            - "ok"   : reasonably meaningful and can be used as auxiliary caption/context.

            Input is a JSON array of objects with fields:
            - "idx": integer index of the frame
            - "text": the snippet text

            You MUST return ONLY a JSON array of the same length and same order, where each
            element is one of: "none", "poor", "ok".

            Example input:
            [
            {{ "idx": 0, "text": "A man is assembling a metal block on the table." }},
            {{ "idx": 1, "text": "..." }}
            ]

            Example output:
            ["ok", "none"]

            Now here is the real input (JSON):
            {json.dumps(items, ensure_ascii=False, indent=2)}

            Return ONLY the JSON array, nothing else.
            """.strip()

        try:
            resp = self.llm.chat(prompt, max_new_tokens=2048)
        except Exception:
            # If the LLM call fails, fall back to zeros.
            return scores

        if isinstance(resp, str):
            text = resp
        else:
            try:
                text = json.dumps(resp, ensure_ascii=False)
            except Exception:
                text = str(resp)

        # Parse JSON via the robust helper.
        parsed: Any = None
        try:
            parsed = parse_llm_json(text)
        except Exception:
            parsed = None

        if not isinstance(parsed, list):
            # If parsing fails, give each attempted item a weak non-zero prior.
            for it in items:
                scores[it["idx"]] = 0.3
            # Otherwise map labels into coarse numeric scores.
        else:
            # Zip defensively in case the model returns the wrong length.
            for label, it in zip(parsed, items):
                label_str = str(label).strip().lower()
                if label_str == "ok":
                    v = 1.0
                elif label_str == "poor":
                    v = 0.3
                else:  # treat anything else as ``none``
                    v = 0.0
                scores[it["idx"]] = v

        # If all scores collapse to the same value, return zeros after normalization.
        max_v = max(scores) if scores else 0.0
        min_v = min(scores) if scores else 0.0
        if max_v - min_v < 1e-6:
            # A constant vector carries no ranking information for downstream weighting.
            return [0.0] * n

        # Otherwise min-max normalize into [0, 1].
        norm_scores: List[float] = []
        denom = (max_v - min_v) + 1e-6
        for v in scores:
            norm_scores.append((v - min_v) / denom)

        return norm_scores



    def _vlm_check_keyframes_for_dataset(
            self,
            keyframes: List[Dict[str, Any]],
            summary: str = "",
            asr_snippet: str = "",
            ocr_snippet: str = "",
        ) -> Tuple[bool, str]:
            """
            Use the VLM to decide whether a keyframe set is suitable for the dataset.
            The prompt is intentionally conservative and favors rejection for weak,
            blurry, redundant, or low-information visual content.

            Returns:
                ``(ok: bool, reason: str)``
            ``ok = True`` means the keyframe set passed the VLM quality check.
            """
            from PIL import Image
            import json
            import re

            if not keyframes:
                return False, "no_keyframes"

            # -----------------------------
            # config: only for text cleaning limits (NO frame sampling)
            # -----------------------------
            try:
                _t2_cfg = ((getattr(self, "config", {}) or {}).get("tasks", {}) or {}).get("T2_vlm_check", {}) or {}
                MAX_ASR_CHARS = int(_t2_cfg.get("max_asr_chars", 600))
                MAX_OCR_CHARS = int(_t2_cfg.get("max_ocr_chars", 400))
            except Exception:
                MAX_ASR_CHARS = 600
                MAX_OCR_CHARS = 400

            # -----------------------------
            # Clean ASR/OCR to reduce noise; do not invent content.
            # -----------------------------
            def _clean_text_for_vlm(text: str, kind: str, max_chars: int) -> str:
                if not text:
                    return ""
                t = str(text)

                # Strip timestamps / floating numbers / repeated whitespace.
                t = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", t)
                t = re.sub(r"\b\d+\.\d+\b", " ", t)
                t = re.sub(r"[ \t]+", " ", t)

                parts = re.split(r"[\n\r]+", t)
                cleaned = []
                seen = set()
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue

                    if kind.upper() == "OCR":
                        if len(p) <= 1:
                            continue
                        bad_ratio = sum(
                            1 for ch in p if not (ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"))
                        ) / max(1, len(p))
                        if bad_ratio > 0.6:
                            continue
                    else:
                        if len(p) < 3:
                            continue

                    if p in seen:
                        continue
                    seen.add(p)
                    cleaned.append(p)

                if kind.upper() == "OCR":
                    cleaned = sorted(cleaned, key=lambda s: len(s), reverse=True)[:12]

                out = "\n".join(cleaned).strip()
                if len(out) > max_chars:
                    out = out[:max_chars].rstrip() + " ..."
                return out

            # Apply the cleaner before injecting text into the VLM prompt.
            asr_snippet = _clean_text_for_vlm(asr_snippet, "ASR", MAX_ASR_CHARS) if asr_snippet else ""
            ocr_snippet = _clean_text_for_vlm(ocr_snippet, "OCR", MAX_OCR_CHARS) if ocr_snippet else ""

            # -----------------------------
            # load images (NO sampling: feed all original keyframes)
            # -----------------------------
            images: List[Image.Image] = []
            extra_images: List[Image.Image] = []  # Track resized images so they can be closed in finally.
            try:
                for kf in keyframes:
                    try:
                        img = Image.open(kf["img"]).convert("RGB")
                        images.append(img)
                    except Exception:
                        return False, "image_load_failed"

                if not images:
                    return False, "no_keyframes_loaded"

                # -----------------------------
                # prompt: reduce ASR/OCR weight; STRICT reject gesture-only long talking-head
                # -----------------------------
                prompt = f"""
                    You are a STRICT quality inspector for a video-image-text dataset.
                    Your primary goal is to find reasons to REJECT bad or borderline keyframe sets.

                    Input:
                    - A group of keyframes from ONE video segment/chapter.
                    - IMPORTANT: You are seeing the ORIGINAL keyframes output by the pipeline (no downsampling / no diversity cherry-picking).
                    - Optional context (LOW CONFIDENCE HINTS):
                    - Global summary (may be empty or noisy).
                    - ASR snippet (may be incomplete / wrong).
                    - OCR snippet (may contain recognition errors).

                    IMPORTANT WEIGHTING:
                    - Visual quality and visual usefulness are PRIMARY.
                    - ASR/OCR are SECONDARY and may be noisy.
                    - You MUST NOT reject solely because ASR/OCR looks confusing, contradictory, or messy.
                    Only use ASR/OCR if it is clearly coherent AND consistent with the visuals.

                    You MUST apply these HARD-FAIL rules:
                    1) If ANY frame is:
                    - severely blurred or out of focus,
                    - extremely dark or extremely bright so that main content is hard to see,
                    - dominated by heavy compression artifacts, noise, or glitches,
                    then you MUST set ok = false.

                    2) Redundancy / low-diversity:
                    - If keyframes are almost all near-duplicates with no meaningful visual progression,
                        you SHOULD set ok = false.
                    - SPECIAL CASE (STRICT):
                        If the frames show a person talking continuously with only minor pose/hand/gesture changes
                        (a long talking-head monologue with no meaningful visual evolution),
                        you MUST treat it as low-diversity content and set ok = false,
                        even if the person is clear and the frames are sharp.

                    3) If most semantic information is missing or unusable visually (tiny objects, no context),
                    you SHOULD set ok = false.

                    You SHOULD ONLY set ok = true when ALL are true:
                    - No severely bad frames.
                    - Several frames contain clear, informative visual content.
                    - There is meaningful visual progression/diversity OR strong informative anchors
                    (e.g., slide/page changes, demonstrated objects, readable on-screen text, step-by-step actions).
                    - A human could reliably write a meaningful multi-sentence interleaved description grounded in these frames.

                    Also estimate a continuous quality_score in [0.0, 1.0]:
                    - 0.0-0.2: almost unusable.
                    - 0.2-0.4: very poor, should be rejected.
                    - 0.4-0.6: borderline / mixed quality, usually reject.
                    - 0.6-0.8: generally OK but with visible issues.
                    - 0.8-1.0: clearly good, clean and informative.

                    IMPORTANT:
                    - If ANY frame is severely blurred / unreadable or broken,
                    quality_score MUST be < 0.6 and ok MUST be false.
                    - For borderline cases, err on the side of rejection.

                    Context for this segment:
                    - Global summary: {summary}

                    - ASR snippet (may be empty):
                    {asr_snippet}

                    - OCR snippet (may be empty):
                    {ocr_snippet}

                    Respond in STRICT JSON with this exact format:
                    {{
                    "ok": true or false,
                    "reason": "<short English explanation>",
                    "quality_score": <float between 0.0 and 1.0>,
                    "num_frames": <int>,
                    "num_bad_frames": <int>,
                    "num_low_quality_frames": <int>
                    }}

                    ONLY output this JSON object, nothing else.
                    """

                # -----------------------------
                # VLM call (NO change to max_new_tokens):
                # - first try original images
                # - if CUDA OOM, retry with ALL frames but progressively downscaled resolution
                # -----------------------------
                def _is_cuda_oom(err: Exception) -> bool:
                    s = str(err).lower()
                    return ("cuda out of memory" in s) or ("out of memory" in s and "cuda" in s)

                raw = ""

                # try to mitigate fragmentation before the heaviest VLM call
                try:
                    self._cuda_cleanup(tag="before_vlm_check", aggressive=True)
                except Exception:
                    pass

                try:
                    raw = self._run_single_vlm_chat(images, prompt)
                except Exception as e:
                    if not _is_cuda_oom(e):
                        print(f"[T2] VLM dataset check failed: {e}")
                        return False, f"vlm_error: {e}"

                    print(f"[T2 WARN] VLM OOM on original images; retrying with downscaled images: {e}")

                    import gc
                    try:
                        import torch
                    except Exception:
                        torch = None  # type: ignore[assignment]

                    # Retry with all frames preserved but progressively smaller resolutions.
                    for max_side in (1280, 1024, 768, 512):
                        try:
                            gc.collect()
                        except Exception:
                            pass
                        if torch is not None:
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    if hasattr(torch.cuda, "ipc_collect"):
                                        torch.cuda.ipc_collect()
                            except Exception:
                                pass

                        resized: List[Image.Image] = []
                        created_this_round: List[Image.Image] = []
                        try:
                            for im in images:
                                w, h = im.size
                                m = max(w, h)
                                if m > int(max_side):
                                    scale = float(max_side) / float(m)
                                    nw = max(1, int(round(w * scale)))
                                    nh = max(1, int(round(h * scale)))
                                    rim = im.resize((nw, nh), Image.BICUBIC)
                                    resized.append(rim)
                                    created_this_round.append(rim)
                                else:
                                    resized.append(im)

                            print(f"[T2 WARN] VLM OOM retry with max_side={max_side}, n_imgs={len(resized)}")
                            raw = self._run_single_vlm_chat(resized, prompt)  # DO NOT change max_new_tokens
                            extra_images.extend(created_this_round)  # close these resized images in finally
                            break
                        except Exception as e2:
                            # This round failed, so close any resized images created here.
                            for rim in created_this_round:
                                try:
                                    rim.close()
                                except Exception:
                                    pass

                            if _is_cuda_oom(e2):
                                print(f"[T2 WARN] Still OOM at max_side={max_side}: {e2}")
                                continue

                            print(f"[T2] VLM dataset check failed during retry (non-OOM): {e2}")
                            return False, f"vlm_error: {e2}"
                    else:
                        return False, f"vlm_error: {e}"

                # debug raw output
                try:
                    _raw_preview = raw.replace("\n", " ") if isinstance(raw, str) else str(raw)
                    if len(_raw_preview) > 500:
                        _raw_preview = _raw_preview[:500] + " ..."
                    print(f"[T2 DEBUG] VLM raw output (truncated): {_raw_preview}")
                except Exception:
                    print("[T2 DEBUG] Failed to print raw VLM output")

                # -----------------------------
                # robust JSON parse
                # -----------------------------
                def _robust_parse_json(s):
                    if isinstance(s, dict):
                        return s
                    if not isinstance(s, str):
                        return None
                    try:
                        return json.loads(s)
                    except Exception:
                        pass
                    try:
                        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
                        if m:
                            return json.loads(m.group(0))
                    except Exception:
                        pass
                    return None

                data = None
                try:
                    # Prefer the shared parser first, then fall back to the local robust parser.
                    if "parse_llm_json" in globals():
                        try:
                            data = parse_llm_json(raw)  # type: ignore[name-defined]
                        except Exception:
                            data = None
                    if not isinstance(data, dict):
                        data = _robust_parse_json(raw)
                except Exception:
                    data = _robust_parse_json(raw)

                if not isinstance(data, dict):
                    print(f"[T2] Failed to parse VLM dataset JSON: {raw}")
                    return False, "vlm_parse_failed"

                ok_flag = bool(data.get("ok", False))
                reason = (data.get("reason") or "").strip() or "no_reason"

                quality_score = data.get("quality_score", 0.0)
                try:
                    quality_score = float(quality_score)
                except Exception:
                    quality_score = 0.0

                num_bad_frames = int(data.get("num_bad_frames", 0) or 0)
                num_low_quality_frames = int(data.get("num_low_quality_frames", 0) or 0)
                num_frames = int(data.get("num_frames", len(keyframes)) or len(keyframes))

                STRICT_MIN_SCORE = 0.6
                final_ok = ok_flag and (quality_score >= STRICT_MIN_SCORE) and (num_bad_frames == 0)

                print(
                    f"[T2 DEBUG] dataset_check: ok_flag={ok_flag}, "
                    f"final_ok={final_ok}, quality_score={quality_score:.3f}, "
                    f"num_frames={num_frames}, num_bad_frames={num_bad_frames}, "
                    f"num_low_quality_frames={num_low_quality_frames}, reason={reason}"
                )

                return final_ok, reason

            finally:
                for im in images:
                    try:
                        im.close()
                    except Exception:
                        pass
                for im in extra_images:
                    try:
                        im.close()
                    except Exception:
                        pass






    def run_T1_structured(
        self,
        segment: SegmentRecord,
    ) -> Dict[str, Any]:
        return run_t1_structured(self, segment)

    def run_T2_structured(
        self,
        t1_stage: Dict[str, Any],
        segment: SegmentRecord,
    ) -> Dict[str, Any]:
        return run_t2_structured(self, t1_stage, segment)

    # =====================================================================
    # =======================================================
    def run_T3(
        self,
        t1_stage: Dict[str, Any],
        t2_stage: Dict[str, Any],
        segment: SegmentRecord,
    ) -> Dict[str, Any]:
        return run_t3_structured(self, t1_stage, t2_stage, segment)

    def run_T4(
        self,
        t1_stage: Dict[str, Any],
        t2_stage: Dict[str, Any],
        t3_stage: Dict[str, Any],
        segment: SegmentRecord,
    ) -> Dict[str, Any]:
        from orchestrator.metadata_admission import (
            paper_profile,
            build_metadata,
            admit_metadata_with_recovery,
        )
        if paper_profile(self.config):
            t3_out = t3_stage.get("output", {}) or {}
            filtered = t3_out.get("filtered_t2_output") or t2_stage.get("output", {})
            metadata = build_metadata(t1_stage.get("output", {}), filtered, t3_out, segment)
            gate = admit_metadata_with_recovery(self, metadata)
            self._dump_stage_debug(video_path=segment.video_path, segment_index=segment.index,
                                   stage_name="METADATA", payload={"metadata": metadata, "gate_c": gate})
            t3_out["metadata_id"] = metadata["metadata_id"]
            t3_out["gate_c"] = gate
            if gate.get("decision") != "accept":
                return {"instances": {}, "decision": {"selected": [], "tasks": {}},
                        "runtime": {"gate_c": gate,
                                    "module_route_reason": f"metadata_{gate.get('decision', 'held')}"}}
        return run_t4_structured(self, t1_stage, t2_stage, t3_stage, segment)

    def run_pipeline(self, video_path: str) -> List[Dict[str, Any]]:
        try:
            return self._run_pipeline_impl(video_path)
        finally:
            self.on_video_end()

    def _run_pipeline_impl(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Orchestrate the full multi-stage pipeline for one video.

        Returns merged dataset items across all segments.
        """
        import os
        import json
        from typing import List, Dict, Any

        print(f"[PipelinePlanner] Running full pipeline for: {video_path}")

        self._set_chat_logging(video_path, "PRE_SPLIT")
        segment_seconds = float(self.config.get("pipeline", {}).get("segment_seconds", 30))
        if not np.isfinite(segment_seconds) or segment_seconds <= 0:
            raise ValueError("pipeline.segment_seconds must be finite and positive")
        segments = self.pre_split_video(video_path, segment_seconds=segment_seconds)
        print(f"[PipelinePlanner] Episode-level segmentation: segments={len(segments)}")

        all_items: List[dict] = []
        log_path = self._log_path(video_path)

        # ===================== [MINIMAL NEW] per-benchmark global jsonl + counters =====================
        BENCH_KEYS = ("llava_bench", "edit", "mmt_bench", "mme", "dialog_visdial_like", "dialog_mmdu_like", "unify", "generation")
        total_counts = {k: 0 for k in BENCH_KEYS}

        # Persist per-benchmark jsonl outputs under output/benchmarks_jsonl/*.jsonl.
        # output_root already points at the active workspace output directory.
        #   outputroot/source_/sheet_/output
        base_dir = self.output_root
        bench_out_dir = os.path.join(base_dir, "benchmarks_jsonl")
        bench_detail_out_dir = os.path.join(bench_out_dir, "detail")
        os.makedirs(bench_out_dir, exist_ok=True)
        os.makedirs(bench_detail_out_dir, exist_ok=True)

        def _append_jsonl(fp: str, rows: List[dict]) -> None:
            if not rows:
                return
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with self._io_lock:
                with open(fp, "a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

        def _append_stage_log(segment_obj: Any, stage_name: str, payload: Dict[str, Any]) -> None:
            _append_jsonl(
                log_path,
                [
                    {
                        "segment_index": int(segment_obj.index),
                        "segment_path": segment_obj.segment_path,
                        "stage": stage_name,
                        **payload,
                    }
                ],
            )

        def _flatten_instances(instances: Dict[str, Any]) -> Dict[str, List[dict]]:
            out = {k: [] for k in BENCH_KEYS}
            if not isinstance(instances, dict):
                return out

            def _append_bucket(dst_key: str, rows: Any) -> None:
                if not isinstance(rows, list):
                    return
                out[dst_key].extend(x for x in rows if isinstance(x, dict))

            for raw_key, rows in instances.items():
                k = str(raw_key or "")
                if k == "magicbrush":
                    _append_bucket("edit", rows)
                elif k.startswith("edit."):
                    _append_bucket("edit", rows)
                elif k in {"geneval", "wise"}:
                    _append_bucket("generation", rows)
                elif k.startswith("generation."):
                    _append_bucket("generation", rows)
                elif k.startswith("unify."):
                    _append_bucket("unify", rows)
                elif k.startswith("understanding."):
                    bench = k.split(".", 1)[1]
                    if bench in out:
                        _append_bucket(bench, rows)
                elif k in out:
                    _append_bucket(k, rows)
            return out

        def _detail_key_from_item(item: Dict[str, Any]) -> str:
            module = str(item.get("module", "") or "").strip()
            bench = str(item.get("benchmark", "") or "").strip()
            if module == "edit":
                return "edit.magicbrush" if bench in {"", "edit", "magicbrush"} else f"edit.{bench}"
            if module == "generation":
                return f"generation.{bench or 'generation'}"
            if module == "understanding":
                return f"understanding.{bench or 'unknown'}"
            if module == "unify":
                return bench if bench.startswith("unify.") else f"unify.{bench or 'task'}"
            if bench == "magicbrush":
                return "edit.magicbrush"
            if bench in {"geneval", "wise", "generation"}:
                return f"generation.{bench}"
            if bench.startswith("unify."):
                return bench
            return bench or module or "unknown.unknown"

        def _detail_key_from_instance_key(raw_key: Any, rows: Any) -> str:
            k = str(raw_key or "").strip()
            if k.startswith(("edit.", "generation.", "understanding.", "unify.")):
                return k
            if k == "edit" or k == "magicbrush":
                return "edit.magicbrush"
            if k in {"geneval", "wise", "generation"}:
                return f"generation.{k}"
            if k in {"llava_bench", "mmt_bench", "mme", "dialog_visdial_like", "dialog_mmdu_like"}:
                return f"understanding.{k}"
            if isinstance(rows, list):
                first = next((x for x in rows if isinstance(x, dict)), None)
                if first:
                    return _detail_key_from_item(first)
            return k or "unknown.unknown"

        def _safe_detail_filename(detail_key: str) -> str:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(detail_key or "").strip())
            return name or "unknown.unknown"

        def _flatten_instances_detail(instances: Dict[str, Any]) -> Dict[str, List[dict]]:
            out: Dict[str, List[dict]] = {}
            if not isinstance(instances, dict):
                return out
            for raw_key, rows in instances.items():
                if not isinstance(rows, list):
                    continue
                detail_key = _detail_key_from_instance_key(raw_key, rows)
                out.setdefault(detail_key, []).extend(x for x in rows if isinstance(x, dict))
            return out

        def _split_flat_items(items: List[dict]) -> Dict[str, List[dict]]:
            out = {k: [] for k in BENCH_KEYS}
            for it in items or []:
                if not isinstance(it, dict):
                    continue
                b = str(it.get("benchmark") or "")
                if b == "magicbrush":
                    out["edit"].append(it)
                elif b in {"geneval", "wise"}:
                    out["generation"].append(it)
                elif b.startswith("unify."):
                    out["unify"].append(it)
                elif b in out:
                    out[b].append(it)
            return out

        def _split_flat_items_detail(items: List[dict]) -> Dict[str, List[dict]]:
            out: Dict[str, List[dict]] = {}
            for it in items or []:
                if not isinstance(it, dict):
                    continue
                detail_key = _detail_key_from_item(it)
                out.setdefault(detail_key, []).append(it)
            return out

        def _print_seg_counts(seg_idx: int, seg_path: str, seg_counts: Dict[str, int]) -> None:
            print(
                f"[Pipeline] Segment #{seg_idx} done | "
                f"llava={seg_counts.get('llava_bench', 0)} "
                f"edit={seg_counts.get('edit', 0)} "
                f"mmt={seg_counts.get('mmt_bench', 0)} "
                f"mme={seg_counts.get('mme', 0)} "
                f"visdial={seg_counts.get('dialog_visdial_like', 0)} "
                f"mmdu={seg_counts.get('dialog_mmdu_like', 0)} "
                f"generation={seg_counts.get('generation', 0)} | "
                f"TOTAL llava={total_counts['llava_bench']} "
                f"edit={total_counts['edit']} "
                f"mmt={total_counts['mmt_bench']} "
                f"mme={total_counts['mme']} "
                f"visdial={total_counts.get('dialog_visdial_like', 0)} "
                f"mmdu={total_counts.get('dialog_mmdu_like', 0)} "
                f"generation={total_counts.get('generation', 0)} | "
                f"path={seg_path}"
            )
        # ===================== [END MINIMAL NEW] =====================

        i = 0
        while i < len(segments):
            segment = segments[i]
            print(f"[PipelinePlanner] Segment {i}: {segment.segment_path}")

            try:
                self._set_chat_logging(video_path, "T1", segment)
                t1_stage = self.run_T1_structured(segment)
            except Exception as e:
                self._dump_stage_error(
                    video_path=video_path,
                    segment_index=segment.index,
                    stage_name="T1",
                    exc=e,
                    context={"segment_path": segment.segment_path},
                )
                _append_stage_log(
                    segment,
                    "T1_ERROR",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            self._dump_stage_debug(video_path=video_path, segment_index=segment.index, stage_name="T1", payload=t1_stage)
            _append_stage_log(segment, "T1", {"T1": t1_stage})

            t1_out = (t1_stage or {}).get("output", {}) or {}

            t1_suitable_raw = t1_out.get("suitable_for_dataset", 1)

            def _t1_rejected(val) -> bool:
                # Support bool flags directly.
                if isinstance(val, bool):
                    return not val
                # For numeric values, treat <= 0 as rejected and > 0 as accepted.
                if isinstance(val, (int, float)):
                    return val <= 0
                # Also accept string prefixes like "no - xxx" / "yes - xxx".
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v.startswith("no"):
                        return True
                    if v.startswith("yes"):
                        return False
                # Otherwise keep the segment by default.
                return False

            if _t1_rejected(t1_suitable_raw):
                print(
                    f"[Pipeline] Segment #{segment.index} rejected by T1 dataset check, "
                    f"suitable_for_dataset={t1_suitable_raw!r}. Skip T2/T3/T4."
                )

                _append_stage_log(
                    segment,
                    "T1_REJECT",
                    {"T1": t1_stage, "T2": None, "T3": None, "T4": None},
                )

                # [MINIMAL NEW] print 0-count segment summary
                _print_seg_counts(segment.index, segment.segment_path, {k: 0 for k in BENCH_KEYS})

                # Release per-segment resources before continuing.
                self.on_segment_end()
                i += 1
                continue
            # ------------------------------------------------------------------

            print(f"[Pipeline] Segment #{segment.index} uses fixed pre-split only; run T2/T3/T4 directly.")

            # ---------- Disabled branch: further T1-based sub-splitting ----------
            if False:

                # Parent episode timing in the original video coordinate system.
                parent_start = float(segment.start_sec)
                parent_end = float(segment.end_sec)
                parent_dur = parent_end - parent_start

                # ``split_points_sec`` is relative to the current segment, e.g. [8.0, 17.0].
                # Convert it into local interval boundaries [local_start, local_end).
                rel_cuts = [float(x) for x in split_points_sec]
                rel_cuts = [0.0] + rel_cuts + [parent_dur]

                sub_segments: List[SegmentRecord] = []
                for j in range(len(rel_cuts) - 1):
                    local_start = rel_cuts[j]
                    local_end = rel_cuts[j + 1]
                    if local_end - local_start < 1e-3:
                        continue

                    # Translate child clip timing back into the parent video timeline.
                    child_start = parent_start + local_start
                    child_end = parent_start + local_end

                    child_path = split_paths[j]  # run_T1 names these like *_s1 / *_s2 / *_s3.mp4

                    child_seg = SegmentRecord(
                        video_path=segment.video_path,
                        segment_path=child_path,
                        index=next_seg_index,
                        start_sec=child_start,
                        end_sec=child_end,
                    )
                    next_seg_index += 1

                    # Mark children so they are not split again downstream.
                    child_seg.no_more_split = True

                    sub_segments.append(child_seg)

                # Debug-print the generated child segments.
                for s_idx, s in enumerate(sub_segments):
                    print(
                        f"[Pipeline]   sub#{s_idx}: idx={s.index} "
                        f"[{s.start_sec:.2f}s -> {s.end_sec:.2f}s] path={s.segment_path}"
                    )

                # Replace the current segment in-place with the derived sub-segments.
                segments[i:i+1] = sub_segments

                # Restart the loop so the new child segments go through T2/T3/T4.
                continue

            # ----- T2: quality filtering plus frame selection -----
            try:
                self._set_chat_logging(video_path, "T2", segment)
                t2_stage = self.run_T2_structured(t1_stage, segment)
            except Exception as e:
                self._dump_stage_error(
                    video_path=video_path,
                    segment_index=segment.index,
                    stage_name="T2",
                    exc=e,
                    context={"segment_path": segment.segment_path, "t1_stage": t1_stage},
                )
                _append_stage_log(
                    segment,
                    "T2_ERROR",
                    {
                        "T1": t1_stage,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            self._dump_stage_debug(video_path=video_path, segment_index=segment.index, stage_name="T2", payload=t2_stage)
            _append_stage_log(segment, "T2", {"T1": t1_stage, "T2": t2_stage})
            t2_out = (t2_stage or {}).get("output", {}) or {}
            quality_passed = bool(t2_out.get("quality_passed", False))

            # T2 may reject a segment explicitly via ``quality_passed = False``.
            if not quality_passed:
                print(
                    f"[Pipeline] Segment #{segment.index} rejected by T2 dataset check, "
                    f"reason={t2_out.get('reason', '')!r}. Skip T3/T4."
                )

                _append_stage_log(
                    segment,
                    "T2_REJECT",
                    {"T1": t1_stage, "T2": t2_stage, "T3": None, "T4": None},
                )

                # [MINIMAL NEW] print 0-count segment summary
                _print_seg_counts(segment.index, segment.segment_path, {k: 0 for k in BENCH_KEYS})

                # Release per-segment resources before continuing.
                self.on_segment_end()
                i += 1
                continue

            # ----- Segment passed T2, so continue into T3/T4 -----
            try:
                self._set_chat_logging(video_path, "T3", segment)
                t3_stage = self.run_T3(t1_stage, t2_stage, segment)
            except Exception as e:
                self._dump_stage_error(
                    video_path=video_path,
                    segment_index=segment.index,
                    stage_name="T3",
                    exc=e,
                    context={"segment_path": segment.segment_path, "t1_stage": t1_stage, "t2_stage": t2_stage},
                )
                _append_stage_log(
                    segment,
                    "T3_ERROR",
                    {
                        "T1": t1_stage,
                        "T2": t2_stage,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            self._dump_stage_debug(video_path=video_path, segment_index=segment.index, stage_name="T3", payload=t3_stage)
            _append_stage_log(segment, "T3", {"T1": t1_stage, "T2": t2_stage, "T3": t3_stage})
            try:
                self._set_chat_logging(video_path, "T4", segment)
                t4_stage = self.run_T4(t1_stage, t2_stage, t3_stage, segment)
            except Exception as e:
                self._dump_stage_error(
                    video_path=video_path,
                    segment_index=segment.index,
                    stage_name="T4",
                    exc=e,
                    context={"segment_path": segment.segment_path, "t1_stage": t1_stage, "t2_stage": t2_stage, "t3_stage": t3_stage},
                )
                _append_stage_log(
                    segment,
                    "T4_ERROR",
                    {
                        "T1": t1_stage,
                        "T2": t2_stage,
                        "T3": t3_stage,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            self._dump_stage_debug(video_path=video_path, segment_index=segment.index, stage_name="T4", payload=t4_stage)
            _append_stage_log(
                segment,
                "T4",
                {"T1": t1_stage, "T2": t2_stage, "T3": t3_stage, "T4": t4_stage},
            )

            # ===================== [MINIMAL NEW] per-seg split + persist + print =====================
            seg_by_bench = {k: [] for k in BENCH_KEYS}
            seg_by_detail: Dict[str, List[dict]] = {}

            if isinstance(t4_stage, dict):
                # NEW: prefer instances
                inst = t4_stage.get("instances", None)
                if isinstance(inst, dict):
                    seg_by_bench = _flatten_instances(inst)
                    seg_by_detail = _flatten_instances_detail(inst)
                else:
                    # fallback: old flat items
                    flat = t4_stage.get("items", []) or []
                    if isinstance(flat, list):
                        seg_by_bench = _split_flat_items(flat)
                        seg_by_detail = _split_flat_items_detail(flat)
            else:
                # very old: t4_stage itself might be a list of items
                if isinstance(t4_stage, list):
                    seg_by_bench = _split_flat_items(t4_stage)
                    seg_by_detail = _split_flat_items_detail(t4_stage)

            # Carry Global Perception's iteration labels into every T4 row.
            # T4 may construct a fresh metadata object for each benchmark, so
            # relying on nested metadata alone loses the source-domain label at
            # the iteration boundary.  Preserve the original row and evidence
            # while adding only provenance fields when they are absent.
            iteration_labels = canonical_iteration_labels(t1_out)
            for bucket in seg_by_bench.values():
                for pos, row in enumerate(bucket):
                    if not isinstance(row, dict):
                        continue
                    bucket[pos] = propagate_iteration_labels(
                        row, iteration_labels, provenance="T1_Global_Perception"
                    )
            # Rebuild detail buckets after row enrichment so detail JSONL and
            # flat data.jsonl share the same propagated metadata.
            seg_by_detail = {}
            for bucket in seg_by_bench.values():
                for row in bucket:
                    if isinstance(row, dict):
                        seg_by_detail.setdefault(_detail_key_from_item(row), []).append(row)

            # Split outputs by benchmark and append them to benchmark-specific jsonl files.
            for k in BENCH_KEYS:
                out_fp = os.path.join(bench_out_dir, f"{k}.jsonl")  # one file per benchmark
                _append_jsonl(out_fp, seg_by_bench[k])
            for detail_key, rows in sorted(seg_by_detail.items()):
                out_fp = os.path.join(bench_detail_out_dir, f"{_safe_detail_filename(detail_key)}.jsonl")
                _append_jsonl(out_fp, rows)

            # Update per-run counters.
            seg_counts = {k: len(seg_by_bench[k]) for k in BENCH_KEYS}
            for k in BENCH_KEYS:
                total_counts[k] += seg_counts[k]

            # ===================== [NEW] accumulate global counts (across all videos) =====================
            try:
                if hasattr(self, "_global_bench_counts") and isinstance(self._global_bench_counts, dict):
                    with self._counter_lock:
                        for k in BENCH_KEYS:
                            self._global_bench_counts[k] = int(self._global_bench_counts.get(k, 0) or 0) + int(seg_counts.get(k, 0) or 0)
            except Exception:
                pass
            # ===================== [END NEW] =====================


            _print_seg_counts(segment.index, segment.segment_path, seg_counts)

            # Merge all benchmark items into the flat return list.
            for k in BENCH_KEYS:
                all_items.extend(seg_by_bench[k])
            # ===================== [END MINIMAL NEW] =====================

            self.on_segment_end()
            i += 1

        # 3) save semantic info (optional)
        out_json = self._semantic_out_path(video_path)
        sem_result = {
            "video_path": video_path,
            "segments": [seg.__dict__ for seg in segments],
        }
        try:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(sem_result, f, ensure_ascii=False, indent=2)
            print(f"[PipelinePlanner] Semantic segmentation result saved to {out_json}")
        except Exception as e:
            print(f"[PipelinePlanner] Failed to save semantic segmentation result: {e}")

        # 4) per-segment logs already appended incrementally
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"[PipelinePlanner] Per-segment logs appended to {log_path}")

        print(
            f"[PipelinePlanner] Finished pipeline for {video_path}, items={len(all_items)} | "
            f"by_bench: llava={total_counts['llava_bench']}, edit={total_counts['edit']}, "
            f"mmt={total_counts['mmt_bench']}, mme={total_counts['mme']}, unify={total_counts['unify']}, generation={total_counts.get('generation', 0)} | "
            f"visdial={total_counts.get('dialog_visdial_like', 0)}, mmdu={total_counts.get('dialog_mmdu_like', 0)} | "
            f"bench_out_dir={bench_out_dir}"
        )
        return all_items
