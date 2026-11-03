#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict export from raw training checkpoint to official-like BAGEL inference package.

This script follows checkpoint organization directly (no base-weight guessing):
1) Read raw non-LoRA tensors from checkpoint and normalize key-space to runtime keys.
2) Verify LoRA roots and merge LoRA into real linear weights:
     W <- W_base + (alpha / r) * (B @ A)
3) Keep all other tensors exactly as stored in checkpoint.
4) Pack to official-like 13-file layout with single-file `ema.safetensors`.

Example:
  python tools/export_ckpt_to_official_layout_hardcoded.py \
    --ckpt-dir /path/to/checkpoints/0000800 \
    --base-model-dir /path/to/models/BAGEL-7B-MoT \
    --out-dir /path/to/official_layout/0000800_official_like
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# =========================
# Hardcoded settings
# =========================
BASE_MODEL_DIR = None
CKPT_DIR = None
OUT_DIR = None

PREFER_EMA = True  # if checkpoint has ema.safetensors, prefer it.

# If set to None, script will parse lora_r / lora_alpha from train.log / log.txt
# under the run root (../.. of CKPT_DIR), and fail if not found.
LORA_R: Optional[int] = None
LORA_ALPHA: Optional[float] = None

FLOW_HEAD_PREFIXES = ("llm2vae.", "vae2llm.", "time_embedder.")

# Optional diagnostics: compare ckpt base_layer vs official base on sampled LoRA roots.
DIAG_COMPARE_OFFICIAL = True
DIAG_SAMPLE_ROOTS = 24
OUT_DTYPE = "bf16"  # bf16 | fp16 | fp32 | keep

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OFFICIAL_FILES = [
    "README.md",
    "ae.safetensors",
    "config.json",
    "ema.safetensors",
    "generation_config.json",
    "llm_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vit_config.json",
    "vocab.json",
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export raw BAGEL training checkpoint to official-like 13-file inference layout "
            "with strict checkpoint-driven LoRA merge."
        )
    )
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--no-prefer-ema", action="store_true")
    parser.add_argument("--no-diag-official", action="store_true")
    parser.add_argument("--diag-sample-roots", type=int, default=DIAG_SAMPLE_ROOTS)
    parser.add_argument("--out-dtype", default=OUT_DTYPE)
    return parser.parse_args()


def _fmt_gib(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GiB"


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _copy_if_exists(src: str, dst: str):
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _write_json_atomic(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_json_if_valid(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    if os.path.getsize(path) <= 0:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _normalize_key(key: str) -> str:
    while key.startswith("_fsdp_wrapped_module."):
        key = key[len("_fsdp_wrapped_module.") :]

    if key.startswith("language_model.base_model.model."):
        key = "language_model." + key[len("language_model.base_model.model.") :]

    key = key.replace(".base_layer.weight", ".weight")
    key = key.replace(".base_layer.bias", ".bias")
    return key


def _pick_ckpt_weight_source(ckpt_dir: str) -> str:
    ema = os.path.join(ckpt_dir, "ema.safetensors")
    model = os.path.join(ckpt_dir, "model.safetensors")
    idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if PREFER_EMA and os.path.isfile(ema):
        return ema
    if os.path.isfile(model):
        return model
    if os.path.isfile(idx):
        return ckpt_dir
    if os.path.isfile(ema):
        return ema
    raise FileNotFoundError(f"No ckpt weights found under {ckpt_dir}")


def _pick_ckpt_mode_and_ref(ckpt_dir: str) -> Tuple[str, str]:
    trainable = os.path.join(ckpt_dir, "trainable_model.safetensors")
    if os.path.isfile(trainable):
        return "trainable_only", trainable
    return "full", _pick_ckpt_weight_source(ckpt_dir)


def _pick_official_weight_source(base_dir: str) -> Optional[str]:
    ema = os.path.join(base_dir, "ema.safetensors")
    single = os.path.join(base_dir, "model.safetensors")
    idx = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.isfile(ema):
        return ema
    if os.path.isfile(single):
        return single
    if os.path.isfile(idx):
        return base_dir
    return None


def _run_root_from_ckpt_dir(ckpt_dir: str) -> Optional[str]:
    # .../<run_name>/checkpoints/<step>
    ckpt_parent = os.path.dirname(os.path.abspath(ckpt_dir))
    if os.path.basename(ckpt_parent) != "checkpoints":
        return None
    return os.path.dirname(ckpt_parent)


def _read_run_logs(run_root: str) -> str:
    txts: List[str] = []
    for name in ("train.log", "log.txt"):
        p = os.path.join(run_root, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txts.append(f.read())
            except Exception:
                pass
    return "\n".join(txts)


def _find_logged_dataclass_line(log_text: str, prefix: str) -> Optional[str]:
    for line in log_text.splitlines():
        if prefix in line:
            return line.strip()
    return None


def _parse_logged_field(line: str, field_name: str):
    pattern = re.compile(rf"{re.escape(field_name)}=(.*?)(?:,\s+\w+=|\)$)")
    m = pattern.search(line)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none":
            return None
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except Exception:
            return raw


def _reconstruct_bagel_config_from_logs(base_model_dir: str, ckpt_dir: str) -> dict:
    run_root = _run_root_from_ckpt_dir(ckpt_dir)
    if run_root is None:
        raise RuntimeError(
            "Cannot reconstruct config.json: ckpt_dir does not match .../<run>/checkpoints/<step> layout."
        )
    log_text = _read_run_logs(run_root)
    if not log_text.strip():
        raise RuntimeError(
            f"Cannot reconstruct config.json: no train.log/log.txt found under run root {run_root}"
        )

    model_line = _find_logged_dataclass_line(log_text, "Model arguments ModelArguments(")
    train_line = _find_logged_dataclass_line(log_text, "Training arguments TrainingArguments(")
    if model_line is None or train_line is None:
        raise RuntimeError(
            "Cannot reconstruct config.json: failed to find logged ModelArguments / TrainingArguments lines."
        )

    required_model_fields = [
        "latent_patch_size",
        "max_latent_size",
        "vit_max_num_patch_per_side",
        "connector_act",
        "interpolate_pos",
    ]
    required_train_fields = [
        "visual_gen",
        "visual_und",
        "timestep_shift",
    ]

    parsed_model = {k: _parse_logged_field(model_line, k) for k in required_model_fields}
    parsed_train = {k: _parse_logged_field(train_line, k) for k in required_train_fields}
    missing = [k for k, v in {**parsed_model, **parsed_train}.items() if v is None]
    if missing:
        raise RuntimeError(
            "Cannot reconstruct config.json from logs; missing parsed fields: "
            f"{missing}"
        )

    base_cfg = _load_json_if_valid(os.path.join(base_model_dir, "config.json"))
    if base_cfg is None:
        raise RuntimeError(
            f"Cannot reconstruct config.json: invalid base config at {os.path.join(base_model_dir, 'config.json')}"
        )

    llm_cfg = _load_json_if_valid(os.path.join(ckpt_dir, "llm_config.json"))
    vit_cfg = _load_json_if_valid(os.path.join(ckpt_dir, "vit_config.json"))
    vae_cfg = _load_json_if_valid(os.path.join(ckpt_dir, "vae_config.json"))
    if llm_cfg is None:
        raise RuntimeError("Cannot reconstruct config.json: invalid checkpoint llm_config.json")
    if bool(parsed_train["visual_und"]) and vit_cfg is None:
        raise RuntimeError("Cannot reconstruct config.json: invalid checkpoint vit_config.json")
    if bool(parsed_train["visual_gen"]) and vae_cfg is None:
        raise RuntimeError("Cannot reconstruct config.json: invalid checkpoint vae_config.json")

    reconstructed = dict(base_cfg)
    reconstructed.update(
        {
            "visual_gen": bool(parsed_train["visual_gen"]),
            "visual_und": bool(parsed_train["visual_und"]),
            "llm_config": llm_cfg,
            "vit_config": vit_cfg if bool(parsed_train["visual_und"]) else None,
            "vae_config": vae_cfg if bool(parsed_train["visual_gen"]) else None,
            "latent_patch_size": int(parsed_model["latent_patch_size"]),
            "max_latent_size": int(parsed_model["max_latent_size"]),
            "vit_max_num_patch_per_side": int(parsed_model["vit_max_num_patch_per_side"]),
            "connector_act": str(parsed_model["connector_act"]),
            "interpolate_pos": bool(parsed_model["interpolate_pos"]),
            "timestep_shift": float(parsed_train["timestep_shift"]),
        }
    )
    return reconstructed


def _parse_lora_from_logs(run_root: str) -> Tuple[Optional[int], Optional[float]]:
    txts: List[str] = []
    for name in ("train.log", "log.txt"):
        p = os.path.join(run_root, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txts.append(f.read())
            except Exception:
                pass

    if not txts:
        return None, None

    rs = set()
    alphas = set()
    pat_r = re.compile(r"lora_r=(\d+)")
    pat_a = re.compile(r"lora_alpha=([0-9]+(?:\.[0-9]+)?)")
    for t in txts:
        for m in pat_r.findall(t):
            rs.add(int(m))
        for m in pat_a.findall(t):
            alphas.add(float(m))

    r = None
    a = None
    if len(rs) == 1:
        r = next(iter(rs))
    if len(alphas) == 1:
        a = next(iter(alphas))
    return r, a


def _resolve_lora_hparams() -> Tuple[int, float]:
    run_root = _run_root_from_ckpt_dir(CKPT_DIR)
    parsed_r = None
    parsed_a = None
    if run_root is not None:
        parsed_r, parsed_a = _parse_lora_from_logs(run_root)

    final_r = LORA_R if LORA_R is not None else parsed_r
    final_a = LORA_ALPHA if LORA_ALPHA is not None else parsed_a

    if final_r is None or final_a is None:
        raise RuntimeError(
            "Cannot resolve LoRA hyper-params. Set LORA_R/LORA_ALPHA in script, "
            "or ensure train.log/log.txt contains unique lora_r/lora_alpha."
        )

    if parsed_r is not None and LORA_R is not None and parsed_r != LORA_R:
        raise RuntimeError(f"LORA_R mismatch: script={LORA_R}, log={parsed_r}")
    if parsed_a is not None and LORA_ALPHA is not None and abs(parsed_a - LORA_ALPHA) > 1e-12:
        raise RuntimeError(f"LORA_ALPHA mismatch: script={LORA_ALPHA}, log={parsed_a}")

    print(
        f"[INFO] LoRA hyper-params: r={final_r}, alpha={final_a} "
        f"(log_detected_r={parsed_r}, log_detected_alpha={parsed_a})"
    )
    return int(final_r), float(final_a)


class SafeTensorReader:
    """Read tensors from single or sharded safetensors source with key lookup."""

    def __init__(self, source: str):
        self.source = source
        self.is_dir = os.path.isdir(source)
        self.kind = "single"
        self.single_path: Optional[str] = None
        self.weight_map: Dict[str, str] = {}
        self._keys: List[str] = []
        self._key_set = set()

        if self.is_dir:
            idx_path = os.path.join(source, "model.safetensors.index.json")
            if not os.path.isfile(idx_path):
                raise FileNotFoundError(f"Missing index json: {idx_path}")
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            wm = idx.get("weight_map", {})
            if not isinstance(wm, dict) or len(wm) == 0:
                raise RuntimeError(f"Invalid weight_map in {idx_path}")
            self.kind = "sharded"
            self.weight_map = wm
            self._keys = sorted(wm.keys())
            self._key_set = set(self._keys)
            for shard in sorted(set(wm.values())):
                p = os.path.join(source, shard)
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing shard referenced by index: {p}")
        else:
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Missing safetensors file: {source}")
            if not source.endswith(".safetensors"):
                raise ValueError(f"Not a safetensors file: {source}")
            self.kind = "single"
            self.single_path = source
            with safe_open(source, framework="pt", device="cpu") as f:
                self._keys = sorted(list(f.keys()))
                self._key_set = set(self._keys)

    def keys(self) -> List[str]:
        return list(self._keys)

    def has(self, key: str) -> bool:
        return key in self._key_set

    def get_tensor(self, key: str) -> torch.Tensor:
        if self.kind == "single":
            with safe_open(self.single_path, framework="pt", device="cpu") as f:  # type: ignore[arg-type]
                return f.get_tensor(key)
        shard = self.weight_map[key]
        with safe_open(os.path.join(self.source, shard), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def load_selected(self, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
        uniq = []
        seen = set()
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            uniq.append(k)

        out: Dict[str, torch.Tensor] = {}
        if self.kind == "single":
            with safe_open(self.single_path, framework="pt", device="cpu") as f:  # type: ignore[arg-type]
                for k in uniq:
                    out[k] = f.get_tensor(k)
            return out

        by_shard: Dict[str, List[str]] = defaultdict(list)
        for k in uniq:
            by_shard[self.weight_map[k]].append(k)
        shard_list = sorted(by_shard.keys())
        for i, shard in enumerate(shard_list, start=1):
            shard_path = os.path.join(self.source, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for k in by_shard[shard]:
                    out[k] = f.get_tensor(k)
            print(f"[INFO] load shard {i}/{len(shard_list)} for selected keys: {shard}")
        return out


def _collect_lora_roots(keys: Iterable[str]) -> List[str]:
    suffix = ".lora_A.default.weight"
    roots = []
    for k in keys:
        if k.endswith(suffix):
            roots.append(k[: -len(suffix)])
    return sorted(set(roots))


def _summarize_ckpt(reader: SafeTensorReader):
    keys = reader.keys()
    c_pref = Counter(k.split(".", 1)[0] for k in keys)
    c_lora_a = sum(k.endswith(".lora_A.default.weight") for k in keys)
    c_lora_b = sum(k.endswith(".lora_B.default.weight") for k in keys)
    c_base_layer = sum(".base_layer." in k for k in keys)
    c_lm_head = sum("lm_head" in k for k in keys)
    print(
        "[INFO] CKPT summary: "
        f"total_keys={len(keys)} loraA={c_lora_a} loraB={c_lora_b} "
        f"base_layer_keys={c_base_layer} lm_head_keys={c_lm_head}"
    )
    print(f"[INFO] CKPT top-level prefixes: {c_pref.most_common(12)}")


def _validate_lora_roots(
    ckpt_reader: SafeTensorReader,
    roots: List[str],
    lora_r: int,
) -> Dict[str, int]:
    missing_a = 0
    missing_b = 0
    missing_base = 0
    rank_mismatch = 0
    shape_mismatch = 0
    rank_hist = Counter()

    for r in roots:
        ka = r + ".lora_A.default.weight"
        kb = r + ".lora_B.default.weight"
        kw = r + ".base_layer.weight"

        if not ckpt_reader.has(ka):
            missing_a += 1
            continue
        if not ckpt_reader.has(kb):
            missing_b += 1
            continue
        if not ckpt_reader.has(kw):
            missing_base += 1
            continue

        # Stream per-root to avoid loading all base_layer weights into RAM.
        a = ckpt_reader.get_tensor(ka)
        b = ckpt_reader.get_tensor(kb)
        w = ckpt_reader.get_tensor(kw)
        if a.ndim != 2 or b.ndim != 2 or w.ndim != 2:
            shape_mismatch += 1
            continue

        rank_hist[int(a.shape[0])] += 1
        if int(a.shape[0]) != int(lora_r):
            rank_mismatch += 1
        if int(a.shape[0]) != int(b.shape[1]):
            shape_mismatch += 1
            continue
        if tuple(w.shape) != (int(b.shape[0]), int(a.shape[1])):
            shape_mismatch += 1
            continue

    print(f"[INFO] LoRA roots: {len(roots)}, rank_hist={dict(rank_hist)}")
    print(
        "[INFO] LoRA root checks: "
        f"missingA={missing_a} missingB={missing_b} missing_base={missing_base} "
        f"rank_mismatch={rank_mismatch} shape_mismatch={shape_mismatch}"
    )

    if missing_a or missing_b or missing_base or rank_mismatch or shape_mismatch:
        raise RuntimeError("LoRA root integrity check failed. Stop to avoid incorrect merge.")

    return {"roots": len(roots)}


def _select_norm_candidate(
    norm_key: str,
    candidates: List[str],
    lora_target_norm_keys: set[str],
) -> str:
    def score(raw: str) -> Tuple[int, int, str]:
        has_wrap = 1 if raw.startswith("_fsdp_wrapped_module.") else 0
        is_base_layer = 1 if ".base_layer." in raw else 0
        # For LoRA targets, prefer .base_layer.* tensor; otherwise prefer plain tensor.
        if norm_key in lora_target_norm_keys:
            base_pref = 0 if is_base_layer else 1
        else:
            base_pref = 1 if is_base_layer else 0
        return (has_wrap, base_pref, raw)

    ordered = sorted(candidates, key=score)
    return ordered[0]


def _build_ckpt_non_lora_norm_map(
    ckpt_reader: SafeTensorReader,
    lora_target_norm_keys: set[str],
) -> Tuple[Dict[str, str], int]:
    bucket: Dict[str, List[str]] = defaultdict(list)
    for raw in ckpt_reader.keys():
        if ".lora_A.default.weight" in raw or ".lora_B.default.weight" in raw:
            continue
        nk = _normalize_key(raw)
        bucket[nk].append(raw)

    norm_map: Dict[str, str] = {}
    duplicate_norm_keys = 0
    for nk, raws in bucket.items():
        if len(raws) > 1:
            duplicate_norm_keys += 1
        pick = _select_norm_candidate(nk, raws, lora_target_norm_keys)
        norm_map[nk] = pick

    # Strict conflict check on duplicates: values must match exactly.
    conflict_examples = []
    for nk, raws in bucket.items():
        if len(raws) <= 1:
            continue
        selected = norm_map[nk]
        need = [selected] + [r for r in raws if r != selected]
        tensors = ckpt_reader.load_selected(need)
        base = tensors[selected]
        for r in raws:
            if r == selected:
                continue
            t = tensors[r]
            if tuple(t.shape) != tuple(base.shape):
                conflict_examples.append((nk, selected, r, "shape"))
                break
            if not torch.equal(t, base):
                conflict_examples.append((nk, selected, r, "value"))
                break

    if conflict_examples:
        first = conflict_examples[0]
        raise RuntimeError(
            "Normalized key conflict detected (non-identical duplicate tensors): "
            f"norm_key={first[0]}, selected={first[1]}, other={first[2]}, kind={first[3]}. "
            "Stop to avoid incorrect export."
        )

    return norm_map, duplicate_norm_keys


def _materialize_base_state_from_ckpt(
    ckpt_reader: SafeTensorReader,
    norm_map: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    raw_keys = [norm_map[nk] for nk in sorted(norm_map.keys())]
    raw_sd = ckpt_reader.load_selected(raw_keys)
    out: Dict[str, torch.Tensor] = {}
    for nk in sorted(norm_map.keys()):
        raw = norm_map[nk]
        out[nk] = raw_sd.pop(raw)
    del raw_sd
    return out


def _flow_head_counts(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    out = {}
    for p in FLOW_HEAD_PREFIXES:
        out[p] = sum(k.startswith(p) for k in sd.keys())
    return out


def _merge_lora_into_state(
    final_sd: Dict[str, torch.Tensor],
    ckpt_reader: SafeTensorReader,
    roots: List[str],
    lora_r: int,
    lora_alpha: float,
) -> int:
    lora_keys = []
    for r in roots:
        lora_keys.append(r + ".lora_A.default.weight")
        lora_keys.append(r + ".lora_B.default.weight")
    lora_sd = ckpt_reader.load_selected(lora_keys)

    scale = float(lora_alpha) / float(lora_r)
    merged_count = 0
    for r in roots:
        ka = r + ".lora_A.default.weight"
        kb = r + ".lora_B.default.weight"
        target = _normalize_key(r + ".base_layer.weight")
        if target not in final_sd:
            raise RuntimeError(f"LoRA merge target missing in base state: {target}")
        a = lora_sd[ka].float()  # [r, in]
        b = lora_sd[kb].float()  # [out, r]
        if int(a.shape[0]) != int(lora_r):
            raise RuntimeError(f"LoRA rank mismatch at {ka}: got {a.shape[0]}, expected {lora_r}")
        delta = (b @ a) * scale
        base = final_sd[target].float()
        if tuple(base.shape) != tuple(delta.shape):
            raise RuntimeError(
                f"LoRA merge shape mismatch for {target}: "
                f"base={tuple(base.shape)} delta={tuple(delta.shape)}"
            )
        final_sd[target] = (base + delta).to(dtype=final_sd[target].dtype)
        merged_count += 1

    return merged_count


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean())


def _parse_out_dtype(name: str) -> Optional[torch.dtype]:
    n = str(name).strip().lower()
    if n in ("keep", "same", "same_as_input"):
        return None
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported --out-dtype: {name}")


def _cast_state_dict_floats(sd: Dict[str, torch.Tensor], out_dtype: Optional[torch.dtype]):
    if out_dtype is None:
        return
    for k, v in list(sd.items()):
        if v.is_floating_point() and v.dtype != out_dtype:
            sd[k] = v.to(out_dtype)


def _diag_compare_ckpt_base_vs_official(
    ckpt_reader: SafeTensorReader,
    official_reader: SafeTensorReader,
    roots: List[str],
):
    if len(roots) == 0:
        return
    sample = roots[: min(len(roots), DIAG_SAMPLE_ROOTS)]
    diffs = []
    miss = 0
    for r in sample:
        kw = r + ".base_layer.weight"
        ko = _normalize_key(kw)
        if not (ckpt_reader.has(kw) and official_reader.has(ko)):
            miss += 1
            continue
        tw = ckpt_reader.get_tensor(kw)
        to = official_reader.get_tensor(ko)
        if tuple(tw.shape) != tuple(to.shape):
            miss += 1
            continue
        diffs.append(_mean_abs(tw, to))

    if not diffs:
        print("[INFO] diag official compare: no valid samples")
        return
    diffs_sorted = sorted(diffs)
    print(
        "[INFO] diag ckpt.base_layer vs official weight "
        f"(sample={len(diffs_sorted)}, miss={miss}): "
        f"mean_abs min={diffs_sorted[0]:.6g} "
        f"median={diffs_sorted[len(diffs_sorted)//2]:.6g} "
        f"max={diffs_sorted[-1]:.6g}"
    )


def _torch_dtype_to_safetensors_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.bool: "BOOL",
        torch.uint8: "U8",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported tensor dtype for safetensors export: {dtype}")
    return mapping[dtype]


def _cast_tensor_if_needed(t: torch.Tensor, out_dtype: Optional[torch.dtype]) -> torch.Tensor:
    if out_dtype is not None and t.is_floating_point() and t.dtype != out_dtype:
        return t.to(out_dtype)
    return t


def _validate_delta_lora_roots(
    delta_reader: SafeTensorReader,
    base_reader: SafeTensorReader,
    roots: List[str],
    lora_r: int,
):
    missing_a = 0
    missing_b = 0
    missing_base = 0
    rank_mismatch = 0
    shape_mismatch = 0
    rank_hist = Counter()

    for r in roots:
        ka = r + ".lora_A.default.weight"
        kb = r + ".lora_B.default.weight"
        target = _normalize_key(r + ".base_layer.weight")

        if not delta_reader.has(ka):
            missing_a += 1
            continue
        if not delta_reader.has(kb):
            missing_b += 1
            continue
        if not base_reader.has(target):
            missing_base += 1
            continue

        a = delta_reader.get_tensor(ka)
        b = delta_reader.get_tensor(kb)
        w = base_reader.get_tensor(target)
        if a.ndim != 2 or b.ndim != 2 or w.ndim != 2:
            shape_mismatch += 1
            continue
        rank_hist[int(a.shape[0])] += 1
        if int(a.shape[0]) != int(lora_r):
            rank_mismatch += 1
        if int(a.shape[0]) != int(b.shape[1]):
            shape_mismatch += 1
            continue
        if tuple(w.shape) != (int(b.shape[0]), int(a.shape[1])):
            shape_mismatch += 1
            continue

    print(f"[INFO] delta LoRA roots: {len(roots)}, rank_hist={dict(rank_hist)}")
    print(
        "[INFO] delta LoRA checks: "
        f"missingA={missing_a} missingB={missing_b} missing_base={missing_base} "
        f"rank_mismatch={rank_mismatch} shape_mismatch={shape_mismatch}"
    )
    if missing_a or missing_b or missing_base or rank_mismatch or shape_mismatch:
        raise RuntimeError("Delta LoRA integrity check failed. Stop to avoid incorrect export.")


def _build_delta_direct_replace_norm_map(
    delta_reader: SafeTensorReader,
    lora_target_norm_keys: set[str],
) -> Tuple[Dict[str, str], int]:
    bucket: Dict[str, List[str]] = defaultdict(list)
    for raw in delta_reader.keys():
        if ".lora_A.default.weight" in raw or ".lora_B.default.weight" in raw:
            continue
        nk = _normalize_key(raw)
        bucket[nk].append(raw)

    norm_map: Dict[str, str] = {}
    duplicate_norm_keys = 0
    for nk, raws in bucket.items():
        if len(raws) > 1:
            duplicate_norm_keys += 1
        norm_map[nk] = _select_norm_candidate(nk, raws, lora_target_norm_keys)

    conflict_examples = []
    for nk, raws in bucket.items():
        if len(raws) <= 1:
            continue
        selected = norm_map[nk]
        tensors = delta_reader.load_selected([selected] + [r for r in raws if r != selected])
        base = tensors[selected]
        for r in raws:
            if r == selected:
                continue
            t = tensors[r]
            if tuple(t.shape) != tuple(base.shape):
                conflict_examples.append((nk, selected, r, "shape"))
                break
            if not torch.equal(t, base):
                conflict_examples.append((nk, selected, r, "value"))
                break
    if conflict_examples:
        first = conflict_examples[0]
        raise RuntimeError(
            "Delta normalized key conflict detected: "
            f"norm_key={first[0]}, selected={first[1]}, other={first[2]}, kind={first[3]}"
        )

    return norm_map, duplicate_norm_keys


def _final_tensor_from_delta_export(
    *,
    key: str,
    base_reader: SafeTensorReader,
    delta_reader: SafeTensorReader,
    direct_replace_norm_map: Dict[str, str],
    roots_by_target: Dict[str, str],
    lora_r: int,
    lora_alpha: float,
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if key in direct_replace_norm_map:
        t = delta_reader.get_tensor(direct_replace_norm_map[key])
        return _cast_tensor_if_needed(t, out_dtype)

    if key in roots_by_target:
        root = roots_by_target[key]
        ka = root + ".lora_A.default.weight"
        kb = root + ".lora_B.default.weight"
        a = delta_reader.get_tensor(ka).float()
        b = delta_reader.get_tensor(kb).float()
        if int(a.shape[0]) != int(lora_r):
            raise RuntimeError(f"LoRA rank mismatch at {ka}: got {a.shape[0]}, expected {lora_r}")
        base = base_reader.get_tensor(key).float()
        delta = (b @ a) * (float(lora_alpha) / float(lora_r))
        if tuple(base.shape) != tuple(delta.shape):
            raise RuntimeError(
                f"LoRA merge shape mismatch for {key}: base={tuple(base.shape)} delta={tuple(delta.shape)}"
            )
        merged = base + delta
        return _cast_tensor_if_needed(merged.to(dtype=base_reader.get_tensor(key).dtype), out_dtype)

    t = base_reader.get_tensor(key)
    return _cast_tensor_if_needed(t, out_dtype)


def _stream_save_delta_export(
    *,
    out_dir: str,
    base_reader: SafeTensorReader,
    delta_reader: SafeTensorReader,
    direct_replace_norm_map: Dict[str, str],
    roots_by_target: Dict[str, str],
    lora_r: int,
    lora_alpha: float,
    out_dtype: Optional[torch.dtype],
    metadata: Dict[str, str],
) -> List[str]:
    final_keys = sorted(base_reader.keys())
    ema_out = os.path.join(out_dir, "ema.safetensors")
    idx_path = os.path.join(out_dir, "model.safetensors.index.json")

    plans = []
    total_size = 0
    offset = 0
    for i, key in enumerate(final_keys, start=1):
        t = _final_tensor_from_delta_export(
            key=key,
            base_reader=base_reader,
            delta_reader=delta_reader,
            direct_replace_norm_map=direct_replace_norm_map,
            roots_by_target=roots_by_target,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            out_dtype=out_dtype,
        )
        nbytes = _tensor_nbytes(t)
        plans.append(
            (
                key,
                {
                    "dtype": _torch_dtype_to_safetensors_name(t.dtype),
                    "shape": list(t.shape),
                    "data_offsets": [offset, offset + nbytes],
                },
            )
        )
        offset += nbytes
        total_size += nbytes
        if i % 500 == 0 or i == len(final_keys):
            print(f"[INFO] delta export plan {i}/{len(final_keys)} tensors")
        del t

    header = {"__metadata__": metadata}
    for key, entry in plans:
        header[key] = entry
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    with open(ema_out, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, byteorder="little", signed=False))
        f.write(header_bytes)
        for i, key in enumerate(final_keys, start=1):
            t = _final_tensor_from_delta_export(
                key=key,
                base_reader=base_reader,
                delta_reader=delta_reader,
                direct_replace_norm_map=direct_replace_norm_map,
                roots_by_target=roots_by_target,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                out_dtype=out_dtype,
            ).detach().cpu().contiguous()
            raw = t.view(torch.uint8).numpy()
            f.write(raw.tobytes())
            if i % 500 == 0 or i == len(final_keys):
                print(f"[INFO] delta export write {i}/{len(final_keys)} tensors")

    idx = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": {k: "ema.safetensors" for k in final_keys},
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

    print(f"[OK] streamed ema: {ema_out} ({_fmt_gib(os.path.getsize(ema_out))})")
    print(f"[OK] streamed index: {idx_path} total={_fmt_gib(total_size)}")
    return final_keys


def _save_ema_and_index(
    out_dir: str,
    final_sd: Dict[str, torch.Tensor],
    metadata: Dict[str, str],
):
    ema_out = os.path.join(out_dir, "ema.safetensors")
    save_file(final_sd, ema_out, metadata=metadata)

    total_size = 0
    weight_map = {}
    for k, v in final_sd.items():
        weight_map[k] = "ema.safetensors"
        total_size += _tensor_nbytes(v)

    idx = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": weight_map,
    }
    idx_path = os.path.join(out_dir, "model.safetensors.index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

    print(f"[OK] save ema: {ema_out} ({_fmt_gib(os.path.getsize(ema_out))})")
    print(f"[OK] save index: {idx_path} total={_fmt_gib(total_size)}")


def _export_tokenizer_json_if_missing(out_dir: str):
    tok_json = os.path.join(out_dir, "tokenizer.json")
    if os.path.isfile(tok_json):
        return
    try:
        from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

        tok = Qwen2Tokenizer.from_pretrained(CKPT_DIR)
        tok.save_pretrained(out_dir)
        print(f"[INFO] tokenizer exported from ckpt (len={len(tok)})")
    except Exception as e:
        print(f"[WARN] tokenizer export failed: {e}")


def _assemble_assets(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Base template from official
    for name in [
        "README.md",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "llm_config.json",
        "vit_config.json",
        "ae.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
    ]:
        _copy_if_exists(os.path.join(BASE_MODEL_DIR, name), os.path.join(out_dir, name))

    # Runtime assets from ckpt should override official when present.
    runtime_override_pairs = [
        ("generation_config.json", "generation_config.json"),
        ("preprocessor_config.json", "preprocessor_config.json"),
        ("llm_config.json", "llm_config.json"),
        ("vit_config.json", "vit_config.json"),
        ("ae.safetensors", "ae.safetensors"),
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("merges.txt", "merges.txt"),
        ("vocab.json", "vocab.json"),
    ]
    for src_name, dst_name in runtime_override_pairs:
        _copy_if_exists(os.path.join(CKPT_DIR, src_name), os.path.join(out_dir, dst_name))

    ckpt_bagel_cfg = _load_json_if_valid(os.path.join(CKPT_DIR, "bagel_config.json"))
    if ckpt_bagel_cfg is None:
        ckpt_bagel_cfg = _load_json_if_valid(os.path.join(CKPT_DIR, "config.json"))
    if ckpt_bagel_cfg is None:
        print("[WARN] checkpoint bagel_config.json/config.json invalid or empty; reconstructing config.json from logs + sub-configs.")
        ckpt_bagel_cfg = _reconstruct_bagel_config_from_logs(BASE_MODEL_DIR, CKPT_DIR)
    _write_json_atomic(os.path.join(out_dir, "config.json"), ckpt_bagel_cfg)

    _export_tokenizer_json_if_missing(out_dir)


def _cleanup_to_official_layout(out_dir: str):
    keep = set(OFFICIAL_FILES)
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        if name in keep:
            continue
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            os.remove(p)

    missing = [n for n in OFFICIAL_FILES if not os.path.isfile(os.path.join(out_dir, n))]
    if missing:
        raise RuntimeError(f"Missing official files after export: {missing}")


def _validate_final_state_dict(final_sd: Dict[str, torch.Tensor], merged_count: int):
    if merged_count <= 0:
        raise RuntimeError("No LoRA weights were merged; refusing to export.")

    bad_prefix = []
    bad_base_layer = []
    bad_lora = []
    for key in final_sd.keys():
        if key.startswith("_fsdp_wrapped_module."):
            bad_prefix.append(key)
        if ".base_layer." in key:
            bad_base_layer.append(key)
        if ".lora_A.default.weight" in key or ".lora_B.default.weight" in key:
            bad_lora.append(key)

    if bad_prefix:
        raise RuntimeError(f"Found FSDP-prefixed keys after normalization, sample={bad_prefix[:3]}")
    if bad_base_layer:
        raise RuntimeError(f"Found .base_layer.* keys after normalization, sample={bad_base_layer[:3]}")
    if bad_lora:
        raise RuntimeError(f"Found LoRA tensors in final state, sample={bad_lora[:3]}")


def _validate_exported_layout(out_dir: str, expected_keys: List[str]):
    ema_path = os.path.join(out_dir, "ema.safetensors")
    idx_path = os.path.join(out_dir, "model.safetensors.index.json")
    if not os.path.isfile(ema_path):
        raise RuntimeError(f"Missing exported ema.safetensors: {ema_path}")
    if not os.path.isfile(idx_path):
        raise RuntimeError(f"Missing exported model.safetensors.index.json: {idx_path}")

    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise RuntimeError("Invalid weight_map in exported index.")

    expected_set = set(expected_keys)
    weight_map_set = set(weight_map.keys())
    if weight_map_set != expected_set:
        miss = sorted(expected_set - weight_map_set)[:5]
        extra = sorted(weight_map_set - expected_set)[:5]
        raise RuntimeError(
            f"Exported index key mismatch: missing={miss} extra={extra}"
        )
    bad_targets = sorted({v for v in weight_map.values() if v != "ema.safetensors"})
    if bad_targets:
        raise RuntimeError(f"Exported index points to unexpected files: {bad_targets}")

    with safe_open(ema_path, framework="pt", device="cpu") as f:
        ema_keys = sorted(list(f.keys()))
    if set(ema_keys) != expected_set:
        miss = sorted(expected_set - set(ema_keys))[:5]
        extra = sorted(set(ema_keys) - expected_set)[:5]
        raise RuntimeError(
            f"Exported ema.safetensors key mismatch: missing={miss} extra={extra}"
        )
    bad_keys = [
        k for k in ema_keys
        if k.startswith("_fsdp_wrapped_module.")
        or ".base_layer." in k
        or ".lora_A.default.weight" in k
        or ".lora_B.default.weight" in k
    ]
    if bad_keys:
        raise RuntimeError(f"Unexpected normalized/LoRA keys remained in export, sample={bad_keys[:3]}")

    for json_name in [
        "config.json",
        "generation_config.json",
        "llm_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vit_config.json",
    ]:
        path = os.path.join(out_dir, json_name)
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing JSON asset: {path}")
        if os.path.getsize(path) <= 0:
            raise RuntimeError(f"Empty JSON asset: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            raise RuntimeError(f"Invalid JSON asset {path}: {e}") from e


def main():
    global BASE_MODEL_DIR
    global CKPT_DIR
    global OUT_DIR
    global LORA_R
    global LORA_ALPHA
    global PREFER_EMA
    global DIAG_COMPARE_OFFICIAL
    global DIAG_SAMPLE_ROOTS
    global OUT_DTYPE

    args = _parse_args()
    BASE_MODEL_DIR = args.base_model_dir
    CKPT_DIR = args.ckpt_dir
    OUT_DIR = args.out_dir
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    PREFER_EMA = not args.no_prefer_ema
    DIAG_COMPARE_OFFICIAL = not args.no_diag_official
    DIAG_SAMPLE_ROOTS = int(args.diag_sample_roots)
    OUT_DTYPE = str(args.out_dtype)

    print(
        "[INFO] args: "
        f"base_model_dir={BASE_MODEL_DIR}, ckpt_dir={CKPT_DIR}, out_dir={OUT_DIR}, "
        f"lora_r={LORA_R}, lora_alpha={LORA_ALPHA}, "
        f"prefer_ema={PREFER_EMA}, diag_official={DIAG_COMPARE_OFFICIAL}, "
        f"diag_sample_roots={DIAG_SAMPLE_ROOTS}, out_dtype={OUT_DTYPE}"
    )

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    _assemble_assets(OUT_DIR)

    lora_r, lora_alpha = _resolve_lora_hparams()

    out_dtype = _parse_out_dtype(OUT_DTYPE)
    official_ref = _pick_official_weight_source(BASE_MODEL_DIR)
    if official_ref is None:
        raise FileNotFoundError(f"No official/base weight source found under {BASE_MODEL_DIR}")

    ckpt_mode, ckpt_ref = _pick_ckpt_mode_and_ref(CKPT_DIR)
    print(f"[INFO] ckpt mode: {ckpt_mode}")
    print(f"[INFO] ckpt weight ref: {ckpt_ref}")

    if ckpt_mode == "trainable_only":
        base_reader = SafeTensorReader(official_ref)
        delta_reader = SafeTensorReader(ckpt_ref)
        print(f"[INFO] official weight ref for export: {official_ref}")
        _summarize_ckpt(delta_reader)

        roots = _collect_lora_roots(delta_reader.keys())
        _validate_delta_lora_roots(delta_reader, base_reader, roots, lora_r=lora_r)
        lora_target_norm_keys = {_normalize_key(r + ".base_layer.weight") for r in roots}
        direct_replace_norm_map, duplicate_norm_keys = _build_delta_direct_replace_norm_map(
            delta_reader, lora_target_norm_keys=lora_target_norm_keys
        )
        print(
            f"[INFO] normalized delta non-LoRA keys={len(direct_replace_norm_map)} "
            f"(duplicate_norm_keys={duplicate_norm_keys})"
        )

        base_keys = set(base_reader.keys())
        direct_replace_keys = set(direct_replace_norm_map.keys())
        extra_direct = sorted(direct_replace_keys - base_keys)
        if extra_direct:
            raise RuntimeError(
                "Delta checkpoint contains non-LoRA keys absent from base model. "
                f"Sample={extra_direct[:5]}"
            )
        for nk, raw in direct_replace_norm_map.items():
            dt = delta_reader.get_tensor(raw)
            bt = base_reader.get_tensor(nk)
            if tuple(dt.shape) != tuple(bt.shape):
                raise RuntimeError(
                    "Delta direct-replace tensor shape differs from base; this export path "
                    "does not support structural weight changes. "
                    f"key={nk} delta_shape={tuple(dt.shape)} base_shape={tuple(bt.shape)}"
                )

        flow_before = {p: sum(k.startswith(p) for k in direct_replace_keys) for p in FLOW_HEAD_PREFIXES}
        print(f"[INFO] flow-head key counts in delta state: {flow_before}")
        has_any_flow_head = any(c > 0 for c in flow_before.values())
        if has_any_flow_head:
            for p, c in flow_before.items():
                if c <= 0:
                    raise RuntimeError(
                        "Partial flow-head tensors found in delta state; "
                        f"missing expected prefix: {p}"
                    )
        else:
            print("[INFO] no flow-head tensors found in delta state; treat as understanding-only delta export.")

        roots_by_target = {_normalize_key(r + ".base_layer.weight"): r for r in roots}

        metadata = {
            "format": "pt",
            "export_mode": "base_plus_trainable_delta_merge",
            "base_model_dir": BASE_MODEL_DIR,
            "ckpt_dir": CKPT_DIR,
            "ckpt_weight_ref": ckpt_ref,
            "base_weight_ref": official_ref,
            "lora_r": str(lora_r),
            "lora_alpha": str(lora_alpha),
            "lora_scale": str(float(lora_alpha) / float(lora_r)),
            "out_dtype": OUT_DTYPE,
            "flow_prefixes": ",".join(FLOW_HEAD_PREFIXES),
            "flow_key_count_llm2vae": str(flow_before["llm2vae."]),
            "flow_key_count_vae2llm": str(flow_before["vae2llm."]),
            "flow_key_count_time_embedder": str(flow_before["time_embedder."]),
            "lora_merged": str(len(roots)),
        }
        final_keys = _stream_save_delta_export(
            out_dir=OUT_DIR,
            base_reader=base_reader,
            delta_reader=delta_reader,
            direct_replace_norm_map=direct_replace_norm_map,
            roots_by_target=roots_by_target,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            out_dtype=out_dtype,
            metadata=metadata,
        )
        _cleanup_to_official_layout(OUT_DIR)
        _validate_exported_layout(OUT_DIR, expected_keys=final_keys)
    else:
        ckpt_reader = SafeTensorReader(ckpt_ref)
        _summarize_ckpt(ckpt_reader)

        roots = _collect_lora_roots(ckpt_reader.keys())
        _validate_lora_roots(ckpt_reader, roots, lora_r=lora_r)
        lora_target_norm_keys = {_normalize_key(r + ".base_layer.weight") for r in roots}

        ckpt_norm_map, duplicate_norm_keys = _build_ckpt_non_lora_norm_map(
            ckpt_reader, lora_target_norm_keys=lora_target_norm_keys
        )
        print(
            f"[INFO] normalized non-LoRA keys={len(ckpt_norm_map)} "
            f"(duplicate_norm_keys={duplicate_norm_keys})"
        )

        final_sd = _materialize_base_state_from_ckpt(ckpt_reader, ckpt_norm_map)
        flow_before = _flow_head_counts(final_sd)
        print(f"[INFO] flow-head key counts in ckpt base state: {flow_before}")
        for p, c in flow_before.items():
            if c <= 0:
                raise RuntimeError(f"Expected flow-head tensors not found for prefix: {p}")

        merged_count = _merge_lora_into_state(
            final_sd=final_sd,
            ckpt_reader=ckpt_reader,
            roots=roots,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        if merged_count != len(roots):
            raise RuntimeError(f"LoRA merge count mismatch: merged={merged_count}, roots={len(roots)}")
        print(
            f"[INFO] lora merge completed: merged={merged_count}, "
            f"scale={float(lora_alpha)/float(lora_r):.6f}"
        )
        _validate_final_state_dict(final_sd, merged_count)

        _cast_state_dict_floats(final_sd, out_dtype)
        print(f"[INFO] cast output floating tensors to: {out_dtype if out_dtype is not None else 'keep'}")

        dtype_bytes = Counter()
        for v in final_sd.values():
            dtype_bytes[str(v.dtype)] += _tensor_nbytes(v)
        print(f"[INFO] output dtype-bytes: {dict(dtype_bytes)}")

        if DIAG_COMPARE_OFFICIAL:
            print(f"[INFO] official weight ref for diagnostics: {official_ref}")
            official_reader = SafeTensorReader(official_ref)
            _diag_compare_ckpt_base_vs_official(ckpt_reader, official_reader, roots)

        metadata = {
            "format": "pt",
            "export_mode": "ckpt_non_lora_plus_lora_merge",
            "base_model_dir": BASE_MODEL_DIR,
            "ckpt_dir": CKPT_DIR,
            "ckpt_weight_ref": ckpt_ref,
            "lora_r": str(lora_r),
            "lora_alpha": str(lora_alpha),
            "lora_scale": str(float(lora_alpha) / float(lora_r)),
            "out_dtype": OUT_DTYPE,
            "flow_prefixes": ",".join(FLOW_HEAD_PREFIXES),
            "flow_key_count_llm2vae": str(flow_before["llm2vae."]),
            "flow_key_count_vae2llm": str(flow_before["vae2llm."]),
            "flow_key_count_time_embedder": str(flow_before["time_embedder."]),
            "lora_merged": str(merged_count),
        }
        _save_ema_and_index(OUT_DIR, final_sd, metadata=metadata)
        _cleanup_to_official_layout(OUT_DIR)
        _validate_exported_layout(OUT_DIR, expected_keys=sorted(final_sd.keys()))

    print("[OK] export complete.")
    print(f"[OK] output dir: {OUT_DIR}")
    print("[OK] final files:")
    for n in sorted(os.listdir(OUT_DIR)):
        fp = os.path.join(OUT_DIR, n)
        print(f"  - {n} ({os.path.getsize(fp)} bytes)")


if __name__ == "__main__":
    main()
