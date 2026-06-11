#!/bin/bash
# Build the CanopyRS eval environment with uv (mirrors existing_models/detectree2).
# detrex + Detectron2 are the only pieces that must be compiled from source against
# the installed torch; everything else is a normal wheel. CUDA ops are built as a fat
# binary for both L4 (sm_89, hpg-turin) and B200 (sm_100, hpg-b200) so the same venv
# runs on either partition. Build pins live in detrex.toml; see README.md.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees/.claude/worktrees/add-selvamask-dataset
ENVDIR=$REPO/existing_models/canopyrs
CANOPYRS=/blue/ewhite/b.weinstein/src/CanopyRS

step() { echo "===== [$(date +%H:%M:%S)] $* ====="; }

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
# Fat binary: cubins for L4 (8.9) and B200 (10.0) + PTX from 10.0 for forward compat.
export TORCH_CUDA_ARCH_LIST="8.9 10.0+PTX"
export MAX_JOBS=4
# Built on a login node with no visible GPU, so torch.cuda.is_available() is False;
# force the detectron2 / detrex CUDA extensions (e.g. ms_deform_attn) to compile.
export FORCE_CUDA=1

step "Clone CanopyRS (+ detrex/detectron2 submodules)"
if [ ! -d "$CANOPYRS/.git" ]; then
  git clone https://github.com/hugobaudchon/CanopyRS.git "$CANOPYRS"
fi
git -C "$CANOPYRS" submodule update --init --recursive

step "uv venv (python 3.10)"
cd "$ENVDIR"
uv venv --python 3.10
source .venv/bin/activate

step "torch 2.7.1 + cu128 (matches nvcc 12.8 / L4 sm_89)"
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

step "GDAL python binding matching the gdal/3.7.0 module"
uv pip install "GDAL==$(gdal-config --version)"

step "Build Detectron2 + detrex from source (sm_89, against installed torch)"
nvcc --version | tail -2
uv pip install --no-build-isolation -e "$CANOPYRS/detrex/detectron2"
uv pip install --no-build-isolation -e "$CANOPYRS/detrex"

step "CanopyRS package (torch/GDAL pins already satisfied -> not re-resolved)"
uv pip install -e "$CANOPYRS"

step "SAM 3 transformers + MillionTrees"
uv pip install "transformers>=5.0.0rc1"
uv pip install -e "$REPO"

step "Smoke import check"
python - <<'PY'
import torch, detectron2, detrex, canopyrs, transformers, milliontrees
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
print("detectron2", detectron2.__version__, "transformers", transformers.__version__)
print("canopyrs + detrex + milliontrees import OK")
PY

step "BUILD COMPLETE"
