# iqa_musiq.py
from __future__ import annotations
from typing import List, Union, Optional
import os
import json
import math
import subprocess
import tempfile
import torch

# 你原来的 musiq_score / get_musiq_metric 保留即可
# ...

def compute_musiq_scores(
    image_paths: List[str],
    conda_env: str = "video_iqa",
    device: str = "auto",
    timeout_s: int = 600,
) -> List[float]:
    """
    Run MUSIQ inside `conda_env` via `conda run -n <env> python -c ...`.
    Always returns len(image_paths) scores; failures are NaN (not zeros).
    """
    import subprocess
    import tempfile
    import json
    import math
    import os

    n = len(image_paths)
    if n <= 0:
        return []

    payload = {"paths": image_paths, "device": device}

    with tempfile.TemporaryDirectory(prefix="musiq_") as td:
        in_json = os.path.join(td, "in.json")
        with open(in_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        worker = r"""
import os, sys, json, math
import torch
import pyiqa

in_json = sys.argv[1]
with open(in_json, "r", encoding="utf-8") as f:
    payload = json.load(f)

paths = payload.get("paths", [])
device = (payload.get("device") or "auto").lower().strip()

def decide_dev(d: str) -> str:
    if d in ("auto", "", "none"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"

def run(dev: str):
    m = pyiqa.create_metric("musiq", device=dev)
    m.eval()
    autocast_on = (dev == "cuda")
    out = []
    for p in paths:
        if not p or (not os.path.exists(p)):
            out.append(float("nan"))
            continue
        try:
            if autocast_on:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    s = m(p)
            else:
                s = m(p)
            try:
                v = float(s.item())
            except Exception:
                v = float(s.reshape(-1)[0].item())
            out.append(v if math.isfinite(v) else float("nan"))
        except Exception:
            out.append(float("nan"))
    return out

dev = decide_dev(device)
try:
    out = run(dev)
except RuntimeError as e:
    msg = str(e).lower()
    # OOM / CUDA error -> fallback to CPU once
    if ("out of memory" in msg or "cuda error" in msg) and dev == "cuda":
        out = run("cpu")
    else:
        raise

print(json.dumps(out))
""".strip()

        cmd = ["conda", "run", "-n", str(conda_env), "python", "-c", worker, in_json]

        env = os.environ.copy()
        # 继承外部 CUDA_VISIBLE_DEVICES（你已经在 shell export 过）
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=int(timeout_s),
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"musiq conda-run nonzero exit (env={conda_env}, code={proc.returncode}): {proc.stderr.strip()}"
            )

        # stdout 可能有 pyiqa 打印，倒序找最后一个 JSON 行
        lines = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
        json_line = None
        for l in reversed(lines):
            if l.startswith("[") and l.endswith("]"):
                json_line = l
                break
        if json_line is None:
            raise RuntimeError(f"musiq conda-run no JSON in stdout. stdout_head={(proc.stdout or '')[:300]}")

        arr = json.loads(json_line)
        out: List[float] = []
        if isinstance(arr, list):
            for v in arr:
                try:
                    fv = float(v)
                    out.append(fv if math.isfinite(fv) else float("nan"))
                except Exception:
                    out.append(float("nan"))
        else:
            out = [float("nan")] * n

        # 保证长度一致，避免 selector 里被静默置零
        if len(out) != n:
            if len(out) < n:
                out = out + [float("nan")] * (n - len(out))
            else:
                out = out[:n]

        return out
