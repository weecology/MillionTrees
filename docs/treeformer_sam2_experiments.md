# TreeFormer + SAM2 cross-geometry experiments

> **AP columns are AP50** (IoU 0.5), the metric in force when these runs were made. AP is now
> AP40 (IoU 0.4) project-wide, so these AP values are historical and are not comparable to the
> current leaderboard.

Model: **TreeFormer** (point detector) → **SAM2** (mask prompting), evaluated on the
`TreePolygons` `crossgeometry` split. Source of runs:
`existing_models/treeformer_sam2/eval_polygons_crossgeometry.py` via
`existing_models/slurm/eval_treeformer_sam2_run{A..N}.sbatch`.

These A–N runs used the **pretrained** `weecology/deepforest-tree-point` checkpoint.
Run M is now codified as the script defaults (SAM2 = `sam2.1-hiera-large`, image-size = 1024,
no tiling, negative-prompts = on, score-thresh-tf = 0.3, batch-size = 1); a fresh evaluation
with the fine-tuned TreeFormer checkpoint at these defaults is pending re-run.

Defaults filled where a flag was omitted (for the historical A–N rows below):
SAM2 = `sam2.1-hiera-small`, image-size = 448, patch-size = 0 (resize whole image, no tiling),
negative-prompts = on, score-thresh-tf = 0.3, nms-distance-thresh = 5.0.

## Configurations

| Run | SAM2 model | Image size | Tiling | Neg. prompts | score-thresh-tf | nms-dist | Notes |
|-----|------------|-----------|--------|--------------|-----------------|----------|-------|
| A | small | 448 | none (resize) | on | 0.10 | 5.0 | |
| B | small | 448 | none (resize) | on | 0.20 | 3.0 | |
| C | small | 224 | patch 512 | on | 0.30 | 5.0 | |
| D | base-plus | 448 | none (resize) | on | 0.30 | 5.0 | |
| E | large | 448 | none (resize) | on | 0.30 | 5.0 | |
| F | large | 224 | patch 512 | on | 0.30 | 5.0 | |
| G | large | 448 | patch 1024 | on | 0.30 | 5.0 | |
| H | large | 448 | patch 1024 | off | 0.30 | 5.0 | |
| I | large | 448 | patch 1000 | on | 0.30 | 5.0 | |
| J | large | 448 | patch-fraction 0.5 | on | 0.30 | 5.0 | ~4 patches/img |
| K | large | 448 | full-res | off | 0.30 | 5.0 | **no results (run incomplete)** |
| L | large | 1024 | none (resize) | off | 0.30 | 5.0 | profiled |
| M | large | 1024 | none (resize) | on | 0.30 | 5.0 | profiled |
| N | large | 1024 | patch 800 | off | 0.30 | 5.0 | profiled |

## Results

Metrics are source-averaged over the crossgeometry split (5 sources, n = 13 / 1322 / 151 / 584 / 2453).
Accuracy = mask_acc. AP50 = −1 means not computed for that run.

| Run | Accuracy | Recall | Mask precision | AP50 | Worst-grp acc | Worst-grp recall | Merge commission |
|-----|----------|--------|----------------|------|---------------|------------------|------------------|
| A | 0.252 | 0.398 | 0.784 | — | 0.083 | 0.185 | 0.008 |
| B | 0.255 | 0.391 | 0.811 | — | 0.083 | 0.184 | 0.008 |
| C | 0.049 | 0.492 | 0.179 | — | 0.014 | 0.140 | 0.007 |
| D | 0.267 | 0.403 | 0.820 | — | 0.066 | 0.159 | 0.008 |
| **E** | **0.316** | 0.472 | 0.808 | — | **0.146** | **0.280** | 0.009 |
| F | 0.052 | 0.104 | 0.716 | — | 0.025 | 0.088 | 0.000 |
| G | 0.083 | 0.236 | 0.642 | — | 0.025 | 0.091 | 0.000 |
| H | 0.059 | 0.679 | 0.155 | — | 0.019 | 0.317 | 0.009 |
| I | 0.081 | 0.231 | 0.644 | — | 0.025 | 0.102 | 0.000 |
| J | 0.081 | 0.231 | 0.644 | — | 0.025 | 0.102 | 0.000 |
| K | — | — | — | — | — | — | — |
| L | 0.148 | 0.528 | 0.404 | 0.082 | 0.059 | 0.433 | 0.007 |
| M | 0.204 | 0.404 | 0.784 | 0.217 | 0.028 | 0.102 | 0.001 |
| N | 0.043 | 0.650 | 0.136 | 0.020 | 0.016 | 0.329 | 0.007 |

## Takeaways

- **Run E is the best overall config**: SAM2-large, image-size 448, whole-image resize (no
  tiling), negative prompts on, score-thresh 0.3 — top accuracy (0.316) and best worst-group
  numbers (acc 0.146, recall 0.280).
- **SAM2 size helps**: small (A/B ≈ 0.25) → base-plus (D 0.267) → large (E 0.316) at the same
  448/resize setup.
- **Tiling hurts accuracy** at these settings (C, F, G, I, J, N): high recall but precision
  collapses (e.g. H recall 0.679 / precision 0.155), driving mask_acc down.
- **AP50 was only computed on the profiled runs** (L/M/N); M (large, 1024, neg-prompts on)
  gives the best AP50 (0.217).
- **Run K produced no results** — the job appears to have not completed.
