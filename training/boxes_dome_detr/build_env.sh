#!/bin/bash
# Build the Dome-DETR training environment with uv.
# The MultiScaleDeformableAttention CUDA extension must be compiled from source
# against the installed torch; it's kept OUT of pyproject.toml on purpose (uv cannot
# resolve it). CUDA ops are built as a fat binary for both L4 (sm_89, hpg-turin) and
# B200 (sm_100, hpg-b200) so the same venv runs on either partition.
# Build pins live here; see README.md.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees-dome-detr
ENVDIR=$REPO/training/boxes_dome_detr
DOME_DETR=/blue/ewhite/b.weinstein/src/Dome-DETR

step() { echo "===== [$(date +%H:%M:%S)] $* ====="; }

module load cuda/12.8.1
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
# Fat binary: cubins for L4 (8.9) and B200 (10.0) + PTX from 10.0 for forward compat.
export TORCH_CUDA_ARCH_LIST="8.9 10.0+PTX"
export MAX_JOBS=4
# Built on a login node with no visible GPU, so torch.cuda.is_available() is False;
# force the CUDA extensions (e.g. MultiScaleDeformableAttention) to compile.
export FORCE_CUDA=1

step "uv venv (python 3.11)"
cd "$ENVDIR"
uv venv --python 3.11
source .venv/bin/activate

step "torch 2.7.1 + cu128 (matches nvcc 12.8 / L4 sm_89)"
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

step "Build Dome-DETR + CUDA extension (MultiScaleDeformableAttention)"
nvcc --version | tail -2

# Patch setup.py to force CUDA even on login node (no GPU device needed for compilation)
# Original: checks torch.cuda.is_available() which is False on login nodes
# Fixed: check CUDA_HOME instead, which is set and sufficient for compilation
SETUP_PY="$DOME_DETR/src/zoo/dome/ops/setup.py"
sed -i 's/if torch\.cuda\.is_available() and CUDA_HOME is not None:/if CUDA_HOME is not None:/g' "$SETUP_PY"

cd "$DOME_DETR/src/zoo/dome/ops"
sh make.sh
# Dome-DETR is not a proper Python package (no setup.py/pyproject.toml at root),
# so we just rely on PYTHONPATH and the compiled CUDA op above

step "Install Dome-DETR requirements + MillionTrees + Comet ML"
cd "$ENVDIR"
uv pip install calflops transformers scikit-image faster-coco-eval-aitod
uv pip install -e .

step "Configure PYTHONPATH for Dome-DETR source"
# Dome-DETR is not installed as a package; add it to PYTHONPATH so imports work
export PYTHONPATH="$DOME_DETR:$PYTHONPATH"

step "Smoke import check"
python - <<'PY'
import sys
import torch
from src.zoo.dome.ops import MultiScaleDeformableAttention
import milliontrees
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
print("MultiScaleDeformableAttention imported OK")
print("milliontrees imported OK")
print("PYTHONPATH setup OK")
PY

step "BUILD COMPLETE"
