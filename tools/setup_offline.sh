#!/usr/bin/env bash
# Install a prepared wheel bundle without accessing package indexes or GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/offline-bundle" >&2
  exit 2
fi
BUNDLE=$(realpath "$1")
command -v uv >/dev/null || { echo "Install/copy uv locally before running this script." >&2; exit 1; }
test -f "$BUNDLE/requirements.txt"
test -d "$BUNDLE/wheels"
export UV_OFFLINE=1
if [[ -f "$BUNDLE/sha256.json" ]]; then
  python3 - "$BUNDLE" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
bundle = Path(sys.argv[1])
checksums = json.loads((bundle / 'sha256.json').read_text())
for name, expected in checksums.items():
    with (bundle / 'wheels' / name).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
print(f'Verified {len(checksums)} offline wheels', flush=True)
PY
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python python3.12 .venv
fi
uv pip sync --python .venv/bin/python --no-index --find-links "$BUNDLE/wheels" "$BUNDLE/requirements.txt"
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
uv pip check --python .venv/bin/python
./run_local.sh -c 'import torch, mujoco; print("MuJoCo:", mujoco.__version__); print("CUDA:", torch.cuda.is_available()); print("Visible GPUs:", torch.cuda.device_count())'
