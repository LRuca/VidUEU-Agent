#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
python="${VIDUEU_PYTHON:-$root/environments/pipeline/bin/python}"
if [[ ! -x "$python" ]]; then
  echo "Pipeline environment missing. Run bash deploy.sh first." >&2
  exit 1
fi
exec "$python" -m vidueu run "$@"
