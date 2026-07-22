# Dome-DETR Integration Status

## ✅ Completed

1. **Environment Setup** (build_env.sh)
   - Python 3.11.13 isolated venv created
   - torch 2.7.1 + cu128 installed
   - All Dome-DETR requirements (calflops, transformers, scikit-image)
   - milliontrees and comet-ml installed
   - Activation script: source training/boxes_dome_detr/activate_dome_env.sh

2. **Training Pipeline** (train.py)
   - Converts TreeBoxes CSV → COCO JSON format
   - Renders config templates with hyperparameters
   - Calls vendored Dome-DETR via subprocess
   - Comet logging with naming scheme

3. **Evaluation** (eval.py)
   - MillionTrees eval API integration
   - Threshold sweep support
   - Model loading (stub - needs full implementation)

4. **SLURM Integration**
   - train_boxes_dome_detr.sbatch for both splits
   - eval_boxes_dome_detr.sbatch for evaluation

5. **Data Conversion**
   - convert_to_coco.py converts TreeBoxes to COCO format
   - Cached on first run

## ⚠️ Known Issues

1. **CUDA Op** — needs compilation on compute node (deferred to SLURM)
2. **eval.py** — model loading is stubbed (needs implementation)
3. **Config templates** — only dome_m is complete

## 🚀 Quick Start

```bash
cd /blue/ewhite/b.weinstein/src/MillionTrees-dome-detr
source training/boxes_dome_detr/activate_dome_env.sh

# Smoke test
python training/boxes_dome_detr/train.py --split-scheme within-distribution --smoke-test --gpus 1

# Full training
sbatch training/slurm/train_boxes_dome_detr.sbatch
```

See DOME_DETR_INTEGRATION.md for full details.
