"""Stable diagnostic codes and safe, actionable messages."""
from __future__ import annotations

import re

HINTS = {
    "api_budget_exhausted": "Choose a new authorized run ledger or explicitly plan additional budget. Restarting never resets the ledger.",
    "api_budget_limit_changed": "Keep the existing ledger's limit unchanged; select a new output for a separately budgeted run.",
    "output_inside_input": "Keep outputs outside the source video directory to avoid rediscovering generated clips.",
    "dotenv_reference_cycle": "Remove cyclic variable references from the adjacent .env file.",
    "file_not_found": "Check that the requested configuration or source path exists.",
    "invalid_api_base_url": "Use an HTTP(S) endpoint without embedded credentials, query parameters or fragments.",
    "dataset_write_failed": "Check disk space, permissions and the output log. Failed writes are not counted as successful items.",
    "recipe_deployment_incomplete": "Run reproduce plan and fill the listed deployment fields. The recipe alone does not start training.",
    "image_outside_asset_roots": "Only explicitly authorized image roots may be exported. Check the path or add --asset-root.",
    "duplicate_record_key": "Resolve duplicate producer record keys before exporting; rows are never silently dropped.",
    "endpoint_unreachable": "Check the server address and the child process network/proxy mode; no billable request is used by doctor.",
    "unexpanded_environment_variables": "Set the named variable in the config-adjacent .env or pass an explicit CLI override.",
    "duplicate_config_key": "Remove the duplicate YAML key; configuration keys must occur once.",
    "unknown_config_field": "Check the field spelling with `vidueu config example`.",
    "input_missing": "Set input to an existing video or directory; relative paths use the small config's directory.",
    "no_video_inputs": "Supply a supported video file or a directory containing videos.",
    "api_credentials_missing": "Set the configured API key environment variable or provide an existing key file.",
    "api_model_or_url_missing": "Set both the provider base_url and the model it actually serves.",
    "vlm_served_model_mismatch": "Use the exact model ID returned by your vision server's /v1/models endpoint.",
    "missing_system_command": "Install FFmpeg and ffprobe through your system package manager.",
    "device_outside_visible_mask": "Tool cuda:N uses logical indices inside hardware.gpus / CUDA_VISIBLE_DEVICES.",
    "low_disk": "Choose an output filesystem with sufficient free space. No automatic cleanup is performed.",
    "output_not_writable": "Choose an output directory whose existing parent is writable.",
    "asset_file_missing": "Run `vidueu assets download --config ...` or correct the local asset path.",
    "model_directory_missing": "Download the selected assets or set model_root / the advanced path override.",
    "model_config_missing": "Provide the complete local model snapshot including config.json.",
    "model_weights_missing": "Complete the model download; a config alone is not a model.",
    "missing_weight_shard": "Complete every weight shard referenced by the model index.",
    "omni_python_missing": "Install the omniparser environment and set tool_python, or deselect that tool.",
    "ocr_gpu_backend_missing": "The default installer uses CPU Paddle. Set tasks.T4_ocr.device=cpu; a GPU request requires a separately prepared matching GPU Paddle environment.",
    "ocr_gpu_device_unavailable": "OCR uses logical CUDA indices. Check CUDA_VISIBLE_DEVICES and the Paddle CUDA build.",
    "omniparser_ocr_asset_missing": "Download ocr_det and ocr_rec, or set models.paddleocr_det/paddleocr_rec to the complete local models.",
    "omniparser_upstream_layout_changed": "Use the pinned omni_code asset from manifests/models.yaml; do not mix upstream revisions.",
    "run_already_active": "Inspect `vidueu status` and stop that owned run before starting another.",
    "preflight_failed": "Fix the reported checks, then start again. No pipeline was launched.",
    "metadata_snapshot_missing": "Export a run with structured METADATA snapshots; older logs need an explicit converter.",
    "image_missing": "Restore the referenced image before export or supply the correct --asset-root.",
    "image_corrupt": "Restore a decodable source image; export never silently drops image references.",
    "export_source_changed": "Wait for the producer to finish and export to a new directory again.",
    "export_destination_exists": "Choose a new empty export destination. Existing data is never overwritten.",
    "shell_substitution_disallowed_in_dotenv": "Use literal NAME=value entries; .env is read as data, not executed.",
    "unsafe_source_name": "Use a single folder name without slashes, '.' or '..'.",
}


def diagnostic(exc, *, check=None):
    raw = str(exc)
    match = re.match(r"^([a-z][a-z0-9_]+)(?::|$)", raw)
    code = match.group(1) if match else "operation_failed"
    field = raw.split(":", 1)[1] if ":" in raw and code in {
        "unexpanded_environment_variables", "duplicate_config_key", "unknown_config_field",
        "config_field_must_be_object", "invalid_config_choices", "invalid_dotenv_line"} else None
    if code == "operation_failed":
        if isinstance(exc, FileNotFoundError):
            code = "file_not_found"
        elif isinstance(exc, ConnectionError) or type(exc).__name__ in {"URLError", "HTTPError", "TimeoutError"}:
            code = "endpoint_unreachable"
    result = {"code": code, "error": type(exc).__name__,
              "hint": HINTS.get(code, "Inspect the named check and configuration; secrets and provider response bodies are withheld.")}
    if field and re.fullmatch(r"[A-Za-z0-9_.,-]+", field):
        result["field"] = field
    if check:
        result["check"] = check
    location = getattr(exc, "source_location", None)
    if isinstance(location, dict) and set(location) == {"source_file", "source_line"}:
        result["source_location"] = location
    return result
