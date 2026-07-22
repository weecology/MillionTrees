# Dome-DETR on MillionTrees TreeBoxes

Fine-tuning [Dome-DETR](https://github.com/RicePasteM/Dome-DETR) ("DETR with Density-Oriented
Feature-Query Manipulation for Efficient Tiny Object Detection", ACM MM 2025) on MillionTrees
tree crown detection.

## Setup

### 1. Build the isolated Python environment

Dome-DETR includes a compiled CUDA extension (`MultiScaleDeformableAttention`) that must be
built against the specific torch/CUDA versions installed. This cannot be managed by the shared
project `uv` environment, so we build it in an isolated venv:

```bash
cd training/boxes_dome_detr
bash build_env.sh
```

This script:
- Creates `training/boxes_dome_detr/.venv` with Python 3.11
- Installs torch 2.7.1 + CUDA 12.8 (matches the cluster's nvcc)
- Builds the CUDA extension as a fat binary (L4 sm_89 + B200 sm_100)
- Installs Dome-DETR, milliontrees, and comet-ml

The build takes ~10-15 minutes. Check `build_env.sh` output for any compilation errors.

### 2. (Optional) Test locally before submitting SLURM

```bash
# Activate the environment
source training/boxes_dome_detr/.venv/bin/activate

# Smoke test: 2 epochs, 2 batches (quick validation)
python training/boxes_dome_detr/train.py \
  --split-scheme within-distribution \
  --batch-size 4 \
  --gpus 1 \
  --smoke-test
```

## Training

### Full SLURM submission (recommended for production)

```bash
sbatch training/slurm/train_boxes_dome_detr.sbatch
# Check logs
tail -f /home/b.weinstein/logs/mt_train_boxes_dome_<JOBID>.out
```

This trains both `within-distribution` and `out-of-distribution` splits (array job) on 2×L4 GPUs.

### Manual training (for debugging)

```bash
source training/boxes_dome_detr/.venv/bin/activate
python training/boxes_dome_detr/train.py \
  --split-scheme within-distribution \
  --batch-size 8 \
  --max-epochs 80 \
  --lr 0.001 \
  --gpus 1 \
  --comet
```

## Evaluation

### After training completes

```bash
sbatch training/slurm/eval_boxes_dome_detr.sbatch
```

Or manually:

```bash
source training/boxes_dome_detr/.venv/bin/activate
python training/boxes_dome_detr/eval.py \
  --checkpoint training/boxes_dome_detr/outputs/within-distribution/best_ckpt.pth \
  --split-scheme within-distribution \
  --score-threshold 0.1 \
  --output-dir training/boxes_dome_detr/outputs/within-distribution
```

## Configuration

### Model sizes

Dome-DETR has three sizes available: `s` (13.2M), `m` (23.9M, default), `l` (33.4M).
Use `--model-size {s,m,l}` in train.py. Default is `m` (best AP/compute tradeoff).

### Hyperparameters

- `--batch-size`: 8 (default, fits L4's 22GB). May need to reduce on smaller GPUs.
- `--max-epochs`: 80 (within-dist), 100 (OOD, harder convergence).
- `--lr`: 0.001 (default learning rate).
- `--model-size`: `m` (default; `s` for speed, `l` for accuracy).
- `--init-mode`: `pretrained` (VisDrone, default) or `scratch`.

### Pretraining

Dome-DETR ships pretrained checkpoints on two domains:
- **VisDrone** (default): Drone/aerial imagery — closest to MillionTrees box domains
- **AI-TOD-V2**: Tiny dense objects — more distant from tree crown imagery

The wrapper defaults to VisDrone; change via model loading in `train.py` if needed.

## Leaderboard integration

Once training + eval completes, results appear as:
- `training/boxes_dome_detr/outputs/{within-distribution,out-of-distribution}/results_<split>.txt`

Add rows to `docs/leaderboard.md` comparing Dome-DETR against DeepForest:

| Model       | Fine-tuned | Avg Recall | Mask-Aware Precision | Script |
|-------------|:---:|--------|--------|--------|
| Dome-DETR   | ✓   | X.XXX  | 0.XXX  | `source training/boxes_dome_detr/.venv/bin/activate && python training/boxes_dome_detr/eval.py ...` |

## Notes

- **CUDA extension**: The compiled CUDA op in `training/boxes_dome_detr/.venv` is specific to
  L4 (sm_89) and B200 (sm_100). If running on different GPUs (e.g., A100), rebuild the env
  with appropriate `TORCH_CUDA_ARCH_LIST`.
- **Data conversion**: TreeBoxes → COCO format is done automatically on first run; subsequent
  runs reuse the cached JSON files under `training/boxes_dome_detr/outputs/coco_annotations/`.
- **Comet logging**: Pass `--comet` to log metrics, hyperparameters, and visualization overlays
  to Comet ML (project: `milliontrees-boxes`, tag: `model-domedetr`).

## Troubleshooting

### "MultiScaleDeformableAttention not found"
The CUDA extension didn't compile. Run `bash build_env.sh` again and check for `nvcc` errors.

### "RuntimeError: CUDA out of memory"
Reduce `--batch-size` (e.g., 4 or 2) or use `--gpus 1` (single GPU).

### "ModuleNotFoundError: No module named 'milliontrees'"
Forgot to activate the venv: `source .venv/bin/activate`

### Training is very slow
Check `nvidia-smi` to confirm both GPUs are being used. If stuck on one GPU, check NCCL
logs: `export NCCL_DEBUG=INFO`.
