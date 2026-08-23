#!/usr/bin/env bash
# Set up the training environment on a RunPod A100 (or any Linux box with a
# CUDA 12 driver). Idempotent: re-running is a no-op once the marker exists.
#
#   bash scripts/bootstrap.sh [WORKSPACE]
#
# WORKSPACE defaults to /workspace when it exists (RunPod's persistent volume),
# otherwise the repo root. The venv lives there too, so a pod restart does not
# mean a 4 GB reinstall.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAXUED_SHA="0f8f1284677375b889e4f13a32c9617cd009f8c4"

if [ "${1-}" != "" ]; then
  WORKSPACE="$1"
elif [ -d /workspace ]; then
  WORKSPACE=/workspace
else
  WORKSPACE="$REPO_ROOT"
fi

VENV="$WORKSPACE/venvs/jaxued"
MARKER="$VENV/.bootstrap_ok"

echo "repo:      $REPO_ROOT"
echo "workspace: $WORKSPACE"
echo "venv:      $VENV"

# --- jaxued at the pinned commit, never modified ---------------------------
if [ ! -d "$REPO_ROOT/third_party/jaxued/.git" ]; then
  mkdir -p "$REPO_ROOT/third_party"
  git clone https://github.com/DramaCow/jaxued.git "$REPO_ROOT/third_party/jaxued"
fi
git -C "$REPO_ROOT/third_party/jaxued" fetch --quiet origin
git -C "$REPO_ROOT/third_party/jaxued" checkout --quiet "$JAXUED_SHA"
echo "jaxued @ $(git -C "$REPO_ROOT/third_party/jaxued" rev-parse --short HEAD)"

if [ -f "$MARKER" ]; then
  echo "environment already bootstrapped ($MARKER)"
  echo "$VENV/bin/python"
  exit 0
fi

# --- isolated interpreter --------------------------------------------------
# The base image's preinstalled torch/jax/numpy-2 would fight the pins below,
# so build a clean 3.11 environment instead of installing into system python.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

mkdir -p "$WORKSPACE/venvs"
uv venv --python 3.11 "$VENV"
PY="$VENV/bin/python"

"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
# constraints.txt bounds the nvidia-* CUDA wheels: jax 0.4.30 declares only
# lower bounds on them, and the current releases (CUDA 12.9 / cuDNN 9.24)
# segfault against a jaxlib built for CUDA ~12.4.
uv pip install --python "$PY" -r "$REPO_ROOT/requirements.txt" --constraint "$REPO_ROOT/constraints.txt"
uv pip install --python "$PY" --no-deps -e "$REPO_ROOT/third_party/jaxued"
uv pip install --python "$PY" --no-deps -e "$REPO_ROOT"

# gymnax imports matplotlib.pyplot at import time. If this script was launched
# from a Jupyter kernel, MPLBACKEND points at matplotlib_inline, which does not
# exist in this venv - force a headless backend.
export MPLBACKEND=Agg
# The image's profile puts /usr/local/cuda/lib64 on LD_LIBRARY_PATH for
# interactive shells. Those system CUDA libraries shadow the pip CUDA wheels
# jax was installed with, and a mismatched cuBLAS fails at the first matmul
# with "INTERNAL: the library was not initialized".
unset LD_LIBRARY_PATH

"$PY" - <<'PYCHECK'
from jaxued.environments import Maze  # the import that actually pulls gymnax in
import jax, tlab_ued
from tlab_ued.teachers import TEACHERS
print("jax", jax.__version__, "devices:", jax.devices())
print("teachers:", sorted(TEACHERS))
assert any(d.platform == "gpu" for d in jax.devices()), "no GPU visible to JAX"
PYCHECK

touch "$MARKER"
echo "bootstrap complete"
echo "$PY"
