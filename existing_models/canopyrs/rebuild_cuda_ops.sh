#!/bin/bash
# Rebuild the CanopyRS CUDA extensions (detectron2 + detrex) for BOTH L4 (sm_89)
# and B200 (sm_100). The original build_env.sh pinned TORCH_CUDA_ARCH_LIST=8.9,
# so detrex's ms_deform_attn op had no sm_100 kernel image and crashed on B200
# with "no kernel image is available for execution on the device".
# Run on a GPU node, e.g.:
#   srun -p hpg-b200 --gpus=1 --cpus-per-task=4 --mem=32G --time=1:00:00 \
#        existing_models/canopyrs/rebuild_cuda_ops.sh
set -euo pipefail

ENVDIR=/blue/ewhite/b.weinstein/src/MillionTrees/existing_models/canopyrs
CANOPYRS=/blue/ewhite/b.weinstein/src/CanopyRS

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
# Fat binary: cubins for L4 (8.9) and B200 (10.0) + PTX from 10.0 for forward compat.
export TORCH_CUDA_ARCH_LIST="8.9 10.0+PTX"
export MAX_JOBS=4
export FORCE_CUDA=1

cd "$ENVDIR"
source .venv/bin/activate

echo "===== nvcc / torch ====="
nvcc --version | tail -2
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# ninja speeds the CUDA compile dramatically; wheel avoids the bdist_wheel error path.
echo "===== ensure build tools ====="
uv pip install ninja wheel

# Editable installs already point at these source trees, so recompiling the
# extensions in place (build_ext --inplace) updates the imported .so directly.
echo "===== rebuild detectron2 ops in place (force recompile) ====="
( cd "$CANOPYRS/detrex/detectron2" && python setup.py build_ext --inplace --force )

echo "===== rebuild detrex ops in place (force recompile) ====="
( cd "$CANOPYRS/detrex" && python setup.py build_ext --inplace --force )

echo "===== verify embedded SASS arches ====="
for SO in "$CANOPYRS"/detrex/detrex/_C*.so "$CANOPYRS"/detrex/detectron2/detectron2/_C*.so; do
  echo "--- $SO ---"
  cuobjdump "$SO" 2>/dev/null | grep -iE "arch = sm" | sort -u || true
done

echo "===== runtime smoke test of ms_deform_attn CUDA op on this GPU ====="
python - <<'PY'
import torch
from detrex.layers.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch,
)

dev = "cuda"
torch.manual_seed(0)
bs, n_heads, embed = 2, 8, 32          # value last dim = per-head embed
n_levels, n_points, n_query = 4, 4, 120
shapes = torch.as_tensor([[32, 32], [16, 16], [8, 8], [4, 4]],
                         dtype=torch.long, device=dev)
level_start = torch.cat((shapes.new_zeros(1),
                         (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]))
S = int((shapes[:, 0] * shapes[:, 1]).sum())

value = torch.randn(bs, S, n_heads, embed, device=dev)
sampling_locations = torch.rand(bs, n_query, n_heads, n_levels, n_points, 2, device=dev)
attn = torch.rand(bs, n_query, n_heads, n_levels, n_points, device=dev) + 1e-5
attn = attn / attn.sum(-1, keepdim=True)

cuda_out = MultiScaleDeformableAttnFunction.apply(
    value, shapes, level_start, sampling_locations, attn, 64)
torch.cuda.synchronize()
ref_out = multi_scale_deformable_attn_pytorch(value, shapes, sampling_locations, attn)
max_err = (cuda_out - ref_out).abs().max().item()
print("ms_deform_attn CUDA op ran on", torch.cuda.get_device_name(0))
print("out shape", tuple(cuda_out.shape), "max|cuda-ref|", max_err)
assert max_err < 1e-3, f"CUDA op disagrees with reference: {max_err}"
print("SMOKE TEST PASSED")
PY

echo "===== REBUILD COMPLETE ====="
