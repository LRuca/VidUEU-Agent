#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
exec "${VIDUEU_PYTHON:-python3}" "$root/scripts/manage_run.py" "$@" --kind vllm --python "${VIDUEU_VLLM_PYTHON:-python3}"
