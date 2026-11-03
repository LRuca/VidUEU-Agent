# Repository Guidelines

## Project Structure & Module Organization

This repository contains the V2UG video understanding and generation pipeline.
Core stages live in `orchestrator/` and `executor/`; tool adapters and model integrations are under `tools/` and `model/`. Top-level Python scripts handle execution, dataset preparation, metadata alignment, VLM judging, and packaging. Tests are in `tests/`; operational helpers are in `scripts/`. Keep generated logs and experiment outputs in `logs/`, `tmp/`, and `outputs/`. Treat `data/`, model weights, and source media as inputs unless a task explicitly creates a new output directory.

## Build, Test, and Development Commands

Install dependencies in an environment with a matching PyTorch/CUDA build:

```bash
python -m pip install -r requirements.txt
```

Run the full test suite:

```bash
pytest -q
```

Run focused regression tests while iterating:

```bash
pytest -q tests/test_run_pipeline_mock.py tests/test_t2_structured_regressions.py
```

Use `python run_pipeline.py --config config.yaml` for a configured run. Prefer repository preparation scripts for dataset work, and run smoke tests before long GPU jobs.

## Coding Style & Naming Conventions

Use Python 4-space indentation, descriptive `snake_case` names, and type hints where useful. Preserve existing dataclass/schema contracts. Use ASCII by default, avoid broad refactors, and comment only non-obvious control flow. No repository-wide formatter or linter is configured.

## Testing Guidelines

Add or update regression tests in `tests/` for behavior changes. Name files `test_<area>.py` and test functions `test_<behavior>`. Mock network calls, model services, and GPU-dependent components when possible. For changes involving JSONL schemas or provenance, test key uniqueness, round-trip parsing, and preservation of source references.

## Commit & Pull Request Guidelines

Use short imperative commit subjects, for example `Harden video collection against missing temp dirs`. Keep commits focused. Pull requests should explain the affected pipeline stage, configuration or GPU assumptions, validation commands and results, and any new output paths. Include representative logs or screenshots for dashboard, data-format, or visualization changes. Never commit credentials, API keys, model weights, raw videos, or large generated outputs.

## Security & Data Handling

Keep secrets in environment variables or local configuration ignored by Git. Do not overwrite or delete source JSONL, manifests, `temp` data, model files, or backups during experiments. Write derived artifacts to a new, clearly named output directory and retain provenance fields such as `input_row`, `record_key`, source file, and source line.
