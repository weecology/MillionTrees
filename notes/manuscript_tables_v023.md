# MillionTrees v0.23 manuscript tables

Generated 2026-08-08 19:45; **Val AP60** column added 2026-08-22. The headline AP is **AP40**
(IoU 0.4); **AP60** (IoU 0.6) is the strict-localisation complement — same predictions, same
ranking, only the overlap a match requires changes — so the AP40 → AP60 drop separates finding
trees from delineating them. Recall is detection recall (boxes), keypoint accuracy (points), or
mask recall (polygons); Precision is mask-aware precision; F1 is their harmonic mean.
Validation = held-out Allen et al. 2025 + Frey et al. 2026 split, which exists only for the
out-of-distribution models. The TreePoints table carries no AP column of either IoU (keypoint
matching is distance-based), so it is unchanged.

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

| Model | Fine-tuned | Split type | Counting MAE | Test Recall | Test Precision | Test F1 | Test AP40 | Val Recall | Val Precision | Val F1 | Val AP40 | Val AP60 | Val MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CanopyRS | Yes* | Out-of-distribution | 13.8 | 0.80 | 0.86 | 0.83 | 0.66 | 0.58 | 0.89 | 0.70 | 0.47 | 0.28 | 25.9 |
| DeepForest | Yes | Out-of-distribution | 18.1 | 0.64 | 0.76 | 0.70 | 0.48 | 0.33 | 0.88 | 0.48 | 0.16 | 0.05 | 37.0 |
| SAM3 | No | Out-of-distribution | 40.2 | 0.72 | 0.58 | 0.65 | 0.50 | 0.57 | 0.60 | 0.58 | 0.40 | 0.20 | 59.0 |
| DeepForest | No | Out-of-distribution | 13.1 | 0.47 | 0.78 | 0.58 | 0.30 | 0.29 | 0.97 | 0.44 | 0.13 | 0.01 | 22.2 |
| CanopyRS | Yes* | Within-distribution | 14.5 | 0.66 | 0.78 | 0.72 | 0.49 | - | - | - | - | - | - |
| DeepForest | Yes | Within-distribution | 15.6 | 0.64 | 0.61 | 0.62 | 0.47 | - | - | - | - | - | - |
| SAM3 | No | Within-distribution | 38.2 | 0.58 | 0.48 | 0.52 | 0.39 | - | - | - | - | - | - |
| DeepForest | No | Within-distribution | 11.4 | 0.39 | 0.60 | 0.47 | 0.24 | - | - | - | - | - | - |

Val AP60 preserves the box ranking but widens the spread: retention (AP60/AP40) runs 0.58
CanopyRS · 0.48 SAM3 · 0.32 fine-tuned DeepForest · **0.05 pretrained DeepForest**. Pretrained
DeepForest's 0.13 AP40 is almost entirely loose matches — at IoU 0.6 it scores 0.007. Fine-tuning
lifts AP40 by 26 % (0.130 → 0.164) but AP60 by 7.6× (0.007 → 0.053), so the strict column is where
the fine-tuning gain actually shows.

### TreePolygons

| Model | Fine-tuned | Split type | Counting MAE | Test Recall | Test Precision | Test F1 | Test AP40 | Val Recall | Val Precision | Val F1 | Val AP40 | Val AP60 | Val MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CanopyRS | Yes* | Out-of-distribution | - | 0.82 | 0.86 | 0.84 | 0.42 | 0.55 | 0.91 | 0.68 | 0.36 | 0.13 | - |
| Mask R-CNN (Detectron2) | Yes | Out-of-distribution | - | 0.56 | 0.90 | 0.69 | 0.39 | 0.46 | 0.97 | 0.62 | 0.33 | 0.13 | - |
| Detectree2 | No | Out-of-distribution | - | 0.50 | 0.63 | 0.56 | 0.25 | 0.50 | 0.63 | 0.56 | 0.35 | 0.14 | - |
| SAM3 | No | Out-of-distribution | - | 0.47 | 0.67 | 0.55 | 0.25 | 0.52 | 0.58 | 0.55 | 0.34 | 0.14 | - |
| DeepForest Mask R-CNN | Yes | Out-of-distribution | - | 0.50 | 0.56 | 0.53 | 0.12 | - | - | - | - | - | - |
| CanopyRS | Yes* | Within-distribution | - | 0.76 | 0.87 | 0.81 | 0.38 | - | - | - | - | - | - |
| Mask R-CNN (Detectron2) | Yes | Within-distribution | - | 0.65 | 0.92 | 0.76 | 0.45 | - | - | - | - | - | - |
| SAM3 | No | Within-distribution | - | 0.58 | 0.62 | 0.60 | 0.29 | - | - | - | - | - | - |
| Detectree2 | No | Within-distribution | - | 0.53 | 0.60 | 0.56 | 0.22 | - | - | - | - | - | - |
| DeepForest Mask R-CNN | Yes | Within-distribution | - | 0.43 | 0.71 | 0.54 | 0.07 | - | - | - | - | - | - |

> ⚠️ **The polygon Val cells above are pooled, not source-balanced — do not put them beside the
> box table without reading this.** The box rows' headline "Average" is a **macro** mean over
> sources (`standard_group_eval`, which states that intent explicitly); the polygon rows come from
> `TreePolygonsStreamingEvalState`, whose headline is a **pooled** mean over images
> (`sum/n`, and one global torchmetrics object for AP). The two geometries therefore aggregate
> differently, which is invisible when sources score alike and large when they do not:
>
> | Polygon model | Val AP40 pooled | Val AP40 macro | Val AP60 pooled | Val AP60 macro |
> |---|--:|--:|--:|--:|
> | CanopyRS | 0.364 | **0.402** | 0.132 | **0.174** |
> | Mask R-CNN (D2) | 0.326 | 0.326 | 0.130 | 0.128 |
> | Detectree2 | 0.352 | 0.352 | 0.142 | 0.142 |
> | SAM3 | 0.341 | 0.342 | 0.139 | 0.139 |
>
> Only CanopyRS moves, and it moves enough to change the ranking: pooled AP60 reads as a
> four-way tie with CanopyRS third, macro AP60 puts CanopyRS first by 0.03. Its per-source AP40
> is 0.341 Allen / 0.462 Frey — a pooled AP of 0.364 lands *below both*, because pooling
> recomputes one joint PR curve across sources whose score distributions differ. **For the
> manuscript, use the macro column**: it is the convention the box table already uses and the
> convention the codebase documents. Macro values are recomputed from the printed per-source
> cells (3 dp), so they carry ±0.001.

Whichever convention is used, polygon retention (AP60/AP40) is 0.39–0.43 and far tighter than the
boxes' 0.05–0.58. Mask IoU 0.6 is a near-uniform ceiling for tree crowns — interlocking canopy
makes the outer fifth of a crown boundary ambiguous in the reference data itself — so the polygon
AP60 column mainly says "no model delineates crowns tightly", and it separates the polygon models
far less than the box column separates the box models.

### Val AP60 provenance

The `Val AP60` cells come from a full re-run of all eight box/polygon validation evals on
2026-08-22 (jobs 39981327 canopyrs · 39981328 sam3 · 39981329 detectree2 · 39981330 deepforest ·
39981522 fine-tuned boxes · 39981332 fine-tuned polygons), after `AP60` was registered as a
metric on `TreeBoxesDataset` / `TreePolygonsDataset`. Nothing else changed: same v0.23 data, same
checkpoints, same score thresholds. **Every re-run reproduced its existing Val Recall, Val
Precision and Val AP40 to three decimals**, which is the check that the AP60 column describes the
same predictions the rest of the row does.

One correction found while doing this: `training/slurm/eval_validation_boxes.sbatch` still
defaulted to `boxes-best-v2.ckpt`, which despite the name is the 2026-06-26 v0.19-era model, not
the v0.23 one. Job 38825227 had overridden `CKPT` by hand on 08-06 to score `boxes-best.ckpt` and
its ledger entry asked for the default to be fixed; it was not, so the first re-run (39981331)
silently scored the June model — AP40 landed at a coincidentally-similar 0.163, but recall/precision
came out 0.338/0.825 against the published 0.328/0.880. The default is now `boxes-best.ckpt`, and
39981522 reproduces the published row exactly. Any future run of that script picks the right model.

### TreePoints

| Model | Fine-tuned | Split type | Counting MAE | Test Recall | Test Precision | Test F1 | Test AP40 | Val Recall | Val Precision | Val F1 | Val AP40 | Val MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TreeFormer | No | Out-of-distribution | 66.77 | 0.76 | 0.74 | 0.75 | - | 0.35 | 0.94 | 0.51 | - | 46.88 |
| TreeFormer (count loss) | Yes | Out-of-distribution | 38.40 | 0.69 | 0.66 | 0.67 | - | 0.27 | 0.66 | 0.38 | - | 136.46 |
| SAM3 | No | Out-of-distribution | 37.39 | 0.69 | 0.62 | 0.65 | - | 0.43 | 0.79 | 0.56 | - | 58.96 |
| TreeFormer (count loss) | Yes | Within-distribution | 19.99 | 0.78 | 0.79 | 0.78 | - | - | - | - | - | - |
| TreeFormer | No | Within-distribution | 118.67 | 0.74 | 0.71 | 0.73 | - | - | - | - | - | - |
| SAM3 | No | Within-distribution | 43.54 | 0.68 | 0.60 | 0.64 | - | - | - | - | - | - |
