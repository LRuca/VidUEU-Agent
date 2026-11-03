# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    # .../executor/models/blurmap_model.py -> repo_root=/home/rtjjjj/video_agent
    return Path(__file__).resolve().parents[2]


def _find_ckpt(default_ckpt: Optional[str] = None) -> str:
    """
    Find a valid TF checkpoint base path (without .index/.data suffix).
    Accept base even if base file itself doesn't exist, as long as base+'.index' exists.
    """
    cand: List[Path] = []
    if default_ckpt:
        cand.append(Path(default_ckpt))

    envp = os.environ.get("BLURMAP_CKPT_PATH", "").strip()
    if envp:
        cand.append(Path(envp))

    repo = _repo_root()
    model_dir = repo / "third_party" / "deep_blur_detection_and_classification" / "model"
    if model_dir.exists():
        for p in sorted(model_dir.glob("*.ckpt*")):
            cand.append(p)

    def _base_from_any(p: Path) -> Optional[Path]:
        s = str(p)
        if s.endswith(".index"):
            return Path(s[:-6])  # strip ".index"
        if ".data-" in s and ".ckpt" in s:
            return Path(s.split(".data-")[0])  # strip ".data-xxxx"
        if s.endswith(".ckpt"):
            return Path(s)
        return None

    checked = set()
    for p in cand:
        base = _base_from_any(p)
        if not base:
            continue
        b = str(base)
        if b in checked:
            continue
        checked.add(b)

        # ✅ 关键：不要求 base 文件存在，只要 base.index 存在就算有效
        if Path(b + ".index").exists():
            return b

    raise FileNotFoundError(
        f"blurmap ckpt not found. Tried default/env and {model_dir}. "
        f"Set env BLURMAP_CKPT_PATH to the ckpt base (e.g. .../final_model.ckpt)."
    )



def _set_device_env(device: str) -> Dict[str, str]:
    """
    device:
      - "auto": do nothing
      - "cpu": disable CUDA
      - "cuda"/"gpu": use all visible (do nothing)
      - "0"/"1"... : set CUDA_VISIBLE_DEVICES to that id
    Also enforce TF allow_growth to reduce pre-alloc OOM.
    """
    env = os.environ.copy()

    # TF1 / TF2 both accept this in many builds
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["TF_CPP_MIN_LOG_LEVEL"] = env.get("TF_CPP_MIN_LOG_LEVEL", "2")

    d = (device or "auto").strip().lower()
    if d in ("cpu",):
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif d.isdigit():
        env["CUDA_VISIBLE_DEVICES"] = d
    else:
        # auto/cuda/gpu: keep current
        pass
    return env


def _run_one_batch(
    batch_pairs: List[Tuple[int, str]],  # (global_idx, frame_path)
    out_dir: Path,
    conda_env: str,
    device: str,
    timeout_s: int,
    max_images: int,
    use_copy: bool,
    ckpt_path: str,
) -> Dict[int, str]:
    """
    Run third_party on this batch. Return {global_idx: out_png_path_or_empty}
    """
    repo = _repo_root()
    third_main = repo / "third_party" / "deep_blur_detection_and_classification" / "main.py"
    if not third_main.exists():
        raise FileNotFoundError(f"third_party main.py not found: {third_main}")

    # Create batch work dir
    work = out_dir / f".blurmap_batch_{uuid.uuid4().hex[:10]}"
    inp = work / "input"
    outp = work / "output"
    inp.mkdir(parents=True, exist_ok=True)
    outp.mkdir(parents=True, exist_ok=True)

    # Copy/symlink images into input with deterministic names: 000123.jpg
    for gidx, src in batch_pairs:
        ext = Path(src).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            ext = ".jpg"
        dst = inp / f"{gidx:06d}{ext}"
        if dst.exists():
            continue
        if use_copy:
            shutil.copy2(src, dst)
        else:
            # symlink (fast); fallback to copy if symlink fails
            try:
                os.symlink(os.path.abspath(src), str(dst))
            except Exception:
                shutil.copy2(src, dst)

    # Call conda run third_party
    cmd = [
        "conda", "run", "-n", str(conda_env), "--no-capture-output",
        "python", str(third_main),
        "--is_train", "false",
        "--index", "0",
        "--input_dir", str(inp),
        "--output_dir", str(outp),
        "--ckpt_path", str(ckpt_path),
        "--max_images", str(int(max_images)),
    ]

    env = _set_device_env(device=device)

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, timeout=int(timeout_s))

    if proc.returncode != 0:
        raise RuntimeError(
            f"third_party blurmap failed rc={proc.returncode}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    # Collect outputs: try match by prefix 000123*.png
    result: Dict[int, str] = {}
    for gidx, _ in batch_pairs:
        candidates = sorted(outp.glob(f"{gidx:06d}*.png"))
        if not candidates:
            # sometimes the script may output with different naming; fallback to any png count alignment is unsafe
            result[gidx] = ""
            continue
        src_png = candidates[0]
        dst_png = out_dir / f"{gidx:06d}.png"
        try:
            shutil.copy2(src_png, dst_png)
            result[gidx] = str(dst_png)
        except Exception:
            result[gidx] = ""

    # cleanup batch work (optional). keep if you want debug.
    try:
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass

    return result


def compute_blurmap(
    frame_paths: List[str],
    out_dir: str,
    conda_env: str = "blurmap_tf1_gpu",
    device: str = "auto",          # auto|cpu|cuda|gpu_id
    timeout_s: int = 1800,
    max_images: int = 1000000,
    use_copy: bool = False,
    batch_size: int = 64,          # ✅ NEW
    ckpt_path: Optional[str] = None # ✅ NEW
) -> List[str]:
    """
    Run deep blur detection (blurmap 3-class) and output per-frame PNG blur maps.

    Output files:
      out_dir/{i:06d}.png

    Returns:
      out_png_paths aligned with frame_paths (missing -> "").
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Resolve ckpt
    ckpt = _find_ckpt(ckpt_path)

    n = len(frame_paths)
    if n == 0:
        return []

    # Write frames json for debugging
    frames_json = out_dir_p / "blurmap_frames.json"
    try:
        frames_json.write_text(json.dumps([str(x) for x in frame_paths], ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    # batching with auto-split on failure
    bs = int(batch_size) if int(batch_size) > 0 else n

    # final outputs
    out_pngs: List[str] = [""] * n
    ok_map: Dict[int, str] = {}

    def _run_range(st: int, ed: int, cur_bs: int) -> None:
        nonlocal ok_map
        pairs = [(i, frame_paths[i]) for i in range(st, ed)]

        try:
            got = _run_one_batch(
                batch_pairs=pairs,
                out_dir=out_dir_p,
                conda_env=conda_env,
                device=device,
                timeout_s=timeout_s,
                max_images=max_images,
                use_copy=use_copy,
                ckpt_path=ckpt,
            )
            ok_map.update(got)
            return
        except Exception as e:
            msg = str(e)
            # if failed and we can split -> split
            if (ed - st) > 1 and cur_bs > 1:
                mid = (st + ed) // 2
                # recurse with smaller “conceptual bs”
                _run_range(st, mid, max(1, cur_bs // 2))
                _run_range(mid, ed, max(1, cur_bs // 2))
                return
            # single image still failed -> mark as missing
            ok_map[st] = ""
            return

    # iterate chunks, but each chunk can further split on failure
    i = 0
    while i < n:
        j = min(n, i + bs)
        _run_range(i, j, bs)
        i = j

    # fill final aligned list
    for i in range(n):
        p = ok_map.get(i, "")
        out_pngs[i] = str(p) if p and Path(p).exists() else ""

    # write manifest
    manifest = []
    for i in range(n):
        manifest.append({"i": i, "ok": bool(out_pngs[i]), "out_png": out_pngs[i]})
    mf = out_dir_p / f"blurmap_manifest_{_now_tag()}.json"
    try:
        mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return out_pngs
