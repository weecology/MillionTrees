# MillionTrees v0.23 manuscript tables

Generated 2026-08-08 19:45. All AP is **AP40** (IoU 0.4). Recall is detection recall (boxes),
keypoint accuracy (points), or mask recall (polygons); Precision is mask-aware precision;
F1 is their harmonic mean. Validation = held-out Allen et al. 2025 + Frey et al. 2026 split,
which exists only for the out-of-distribution models.

All point rows are now on the repaired v0.23 data (post 08-07 repackage, which restored
187 Amirkolaee et al. 2023 images / 32,452 annotations). Boxes and polygons unaffected by that fix.

Caveats: TreePolygons emits no counting_mae metric, so that column is empty by construction.

Point rows updated 2026-08-10: the fine-tuned point model is now the **count-loss fix** run
(39014685, `training/slurm/train_points_896_countfix.sbatch`, `--loss-preset pretrain`), which
replaces every earlier point fine-tune. All prior fine-tunes inherited `enforce_count=True`, which
made `count_loss` identically zero, so the model learned where trees are but never how many; those
runs also never converged (they always selected the last epoch). With the count term live, val_loss
bottoms mid-run — within-distribution kept epoch 8/20, out-of-distribution epoch 17/20 — and the
counting metrics are transformed (within-distribution MAE 118.67 → 19.99, nMAE 1.611 → 0.214;
out-of-distribution MAE 66.77 → 38.40, slope now positive on both splits). The validation cell
(39081455) is the countfix out-of-distribution checkpoint on the held-out Allen+Frey split.

### TreeBoxes

| Model | Fine-tuned | Split type | Counting MAE | Test Recall | Test Precision | Test F1 | Test AP40 | Val Recall | Val Precision | Val F1 | Val AP40 | Val MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CanopyRS | Yes* | Out-of-distribution | 13.8 | 0.80 | 0.86 | 0.83 | 0.66 | 0.58 | 0.89 | 0.70 | 0.47 | 25.9 |
| DeepForest | Yes | Out-of-distribution | 18.1 | 0.64 | 0.76 | 0.70 | 0.48 | 0.33 | 0.88 | 0.48 | 0.16 | 37.0 |
| SAM3 | No | Out-of-distribution | 40.2 | 0.72 | 0.58 | 0.65 | 0.50 | 0.57 | 0.60 | 0.58 | 0.40 | 59.0 |
| DeepForest | No | Out-of-distribution | 13.1 | 0.47 | 0.78 | 0.58 | 0.30 | 0.29 | 0.97 | 0.44 | 0.13 | 22.2 |
| CanopyRS | Yes* | Within-distribution | 14.5 | 0.66 | 0.78 | 0.72 | 0.49 | - | - | - | - | - |
| DeepForest | Yes | Within-distribution | 15.6 | 0.64 | 0.61 | 0.62 | 0.47 | - | - | - | - | - |
| SAM3 | No | Within-distribution | 38.2 | 0.58 | 0.48 | 0.52 | 0.39 | - | - | - | - | - |
| DeepForest | No | Within-distribution | 11.4 | 0.39 | 0.60 | 0.47 | 0.24 | - | - | - | - | - |

### TreePolygons

| Model | Fine-tuned | Split type | Counting MAE | Test Recall | Test Precision | Test F1 | Test AP40 | Val Recall | Val Precision | Val F1 | Val AP40 | Val MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CanopyRS | Yes* | Out-of-distribution | - | 0.82 | 0.86 | 0.84 | 0.42 | 0.55 | 0.91 | 0.68 | 0.36 | - |
| Mask R-CNN (Detectron2) | Yes | Out-of-distribution | - | 0.56 | 0.90 | 0.69 | 0.39 | 0.46 | 0.97 | 0.62 | 0.33 | - |
| Detectree2 | No | Out-of-distribution | - | 0.50 | 0.63 | 0.56 | 0.25 | 0.50 | 0.63 | 0.56 | 0.35 | - |
| SAM3 | No | Out-of-distribution | - | 0.47 | 0.67 | 0.55 | 0.25 | 0.52 | 0.58 | 0.55 | 0.34 | - |
| DeepForest Mask R-CNN | Yes | Out-of-distribution | - | 0.50 | 0.56 | 0.53 | 0.12 | - | - | - | - | - |
| CanopyRS | Yes* | Within-distribution | - | 0.76 | 0.87 | 0.81 | 0.38 | - | - | - | - | - |
| Mask R-CNN (Detectron2) | Yes | Within-distribution | - | 0.65 | 0.92 | 0.76 | 0.45 | - | - | - | - | - |
| SAM3 | No | Within-distribution | - | 0.58 | 0.62 | 0.60 | 0.29 | - | - | - | - | - |
| Detectree2 | No | Within-distribution | - | 0.53 | 0.60 | 0.56 | 0.22 | - | - | - | - | - |
| DeepForest Mask R-CNN | Yes | Within-distribution | - | 0.43 | 0.71 | 0.54 | 0.07 | - | - | - | - | - |

### TreePoints

| Model | Fine-tuned | Split type | Counting MAE | Test Recall | Test Precision | Test F1 | Test AP40 | Val Recall | Val Precision | Val F1 | Val AP40 | Val MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TreeFormer | No | Out-of-distribution | 66.77 | 0.76 | 0.74 | 0.75 | - | 0.35 | 0.94 | 0.51 | - | 46.88 |
| TreeFormer (count loss) | Yes | Out-of-distribution | 38.40 | 0.69 | 0.66 | 0.67 | - | 0.27 | 0.66 | 0.38 | - | 136.46 |
| SAM3 | No | Out-of-distribution | 37.39 | 0.69 | 0.62 | 0.65 | - | 0.43 | 0.79 | 0.56 | - | 58.96 |
| TreeFormer (count loss) | Yes | Within-distribution | 19.99 | 0.78 | 0.79 | 0.78 | - | - | - | - | - | - |
| TreeFormer | No | Within-distribution | 118.67 | 0.74 | 0.71 | 0.73 | - | - | - | - | - | - |
| SAM3 | No | Within-distribution | 43.54 | 0.68 | 0.60 | 0.64 | - | - | - | - | - | - |
