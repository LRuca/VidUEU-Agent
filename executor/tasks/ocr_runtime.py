"""Explicit Paddle device contract: never silently downgrade a GPU request."""
from __future__ import annotations


def paddle_device_info(device="cpu"):
    import paddle
    raw = str(device).strip().lower().replace("cuda", "gpu")
    gpu = raw == "gpu" or raw.startswith("gpu:")
    if raw != "cpu" and not gpu:
        raise ValueError("ocr_invalid_device: use cpu or cuda:<logical index>")
    built = bool(paddle.is_compiled_with_cuda())
    if gpu:
        if not built:
            raise RuntimeError("ocr_gpu_backend_missing: installed Paddle is CPU-only; "
                               "set tasks.T4_ocr.device=cpu or install a matching GPU Paddle in a separate environment")
        index = int(raw.split(":", 1)[1]) if ":" in raw else 0
        if not 0 <= index < paddle.device.cuda.device_count():
            raise RuntimeError("ocr_gpu_device_unavailable")
        raw = f"gpu:{index}"
    return {"device": raw, "cuda_build": built, "backend": "gpu" if gpu else "cpu"}
