#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${VIDUEU_BOOTSTRAP_PYTHON:-python3}" "$root/scripts/setup_env.py" "$@"
