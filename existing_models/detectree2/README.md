# Detectree2 baseline (TreePolygons)

[Detectree2](https://github.com/PatBall1/detectree2) is a Detectron2 Mask R-CNN
for tree-crown delineation. We evaluate a pretrained `model_garden` checkpoint on
the MillionTrees `TreePolygons` task and report the same metrics as the other
`existing_models` runs.

## Install

Detectron2 has no PyPI release and must be built against the installed PyTorch, so
install in steps (see the [detectree2 install docs](https://patball1.github.io/detectree2/installation.html)):

```bash
cd existing_models/detectree2
uv venv --python 3.11
source .venv/bin/activate

# 1. PyTorch (pick the CUDA build for your cluster; CPU shown here)
uv pip install torch torchvision

# 2. Detectron2 (built from source against the torch above)
uv pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 3. Detectree2 + MillionTrees (editable)
uv pip install detectree2
uv pip install -e ../..
```

## Download a checkpoint

The `250312_flexi.pth` general RGB model (closed-canopy + urban) is a sensible
default for the diverse MillionTrees imagery:

```bash
wget https://zenodo.org/records/15863800/files/250312_flexi.pth
```

Other options live in the [model garden](https://github.com/PatBall1/detectree2/tree/master/model_garden).

## Run

```bash
uv run python eval_polygons.py \
  --root-dir "$MT_ROOT" \
  --split-scheme random \
  --model-path 250312_flexi.pth \
  --device cuda \
  --output-dir outputs/random
```

Smoke test (a couple of batches, CPU):

```bash
uv run python eval_polygons.py --mini --max-batches 2 --device cpu \
  --model-path 250312_flexi.pth
```

Outputs `results_polygons_<split>.txt` / `.json`, matching `existing_models/sam3`.
