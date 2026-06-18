# Detectree2 baseline (TreePolygons)

[Detectree2](https://github.com/PatBall1/detectree2) is a Detectron2 Mask R-CNN
for tree-crown delineation. We evaluate a pretrained `model_garden` checkpoint on
the MillionTrees `TreePolygons` task and report the same metrics as the other
`existing_models` runs.

## Install

Detectron2 has no PyPI release and must be built against the *exact* installed
PyTorch, so it is **not** in `pyproject.toml`. The pinned, reproducible build
environment (torch version, CUDA module, arch flags, detectron2 git source) lives
in [`detectron2.toml`](detectron2.toml). Install in steps (see also the
[detectree2 install docs](https://patball1.github.io/detectree2/installation.html)):

```bash
cd existing_models/detectree2
uv venv --python 3.11
source .venv/bin/activate

# 1. PyTorch — pinned to the build detectron2 compiles against (L4 / sm_89 here)
uv pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# 2. Detectron2 (built from source against the torch above)
export CUDA_HOME=/apps/compilers/cuda/12.4.1   # provides nvcc matching cu124
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
uv pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 3. Detectree2 + MillionTrees (editable)
uv pip install detectree2
uv pip install -e ../..
```

See [`detectron2.toml`](detectron2.toml) for the exact pins.

## Download a checkpoint

The `250312_flexi.pth` general RGB model (closed-canopy + urban) is a sensible
default for the diverse MillionTrees imagery:

```bash
wget https://zenodo.org/records/15863800/files/250312_flexi.pth
```

Other options live in the [model garden](https://github.com/PatBall1/detectree2/tree/master/model_garden).

## Run

`--no-sync` keeps the hand-built detectron2 + pinned torch (neither is in
`pyproject.toml`); without it `uv run` would re-resolve and clobber them.

```bash
uv run --no-sync python eval_polygons.py \
  --root-dir "$MT_ROOT" \
  --split-scheme within-distribution \
  --model-path 250312_flexi.pth \
  --device cuda \
  --output-dir outputs/within-distribution
```

Smoke test (a couple of batches, CPU):

```bash
uv run --no-sync python eval_polygons.py --mini --max-batches 2 --device cpu \
  --model-path 250312_flexi.pth
```

Outputs `results_polygons_<split>.txt` / `.json`, matching `existing_models/sam3`.
