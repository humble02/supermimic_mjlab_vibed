#!/usr/bin/env bash
# Foreground launcher; use tmux externally if desired.
set -euo pipefail
cd "$(dirname "$0")"
RUN_NAME=${1:-lafan1_mjlab}
if [[ $# -gt 0 ]]; then shift; fi
exec ./run_local.sh supermimic/train.py --log_dir "logs/$RUN_NAME" "$@"
