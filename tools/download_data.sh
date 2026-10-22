#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./run_local.sh tools/prepare_data.py "$@"
./run_local.sh tools/urdf_to_mjcf.py
