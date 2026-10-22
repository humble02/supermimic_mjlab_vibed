#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "Run bash tools/setup_local.sh first." >&2
  exit 1
fi
export VIRTUAL_ENV="$REPO/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export WARP_CACHE_PATH="$REPO/.cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$REPO/.cache/inductor"
export MPLCONFIGDIR="$REPO/.cache/matplotlib"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"
exec "$VIRTUAL_ENV/bin/python" "$@"
