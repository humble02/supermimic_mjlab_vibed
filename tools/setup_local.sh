#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/" >&2; exit 1; }
uv sync --locked
./run_local.sh -c 'import torch, mjlab, mujoco; print("MuJoCo:", mujoco.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "none")'
