# Validation Leaderboard (held-out TLS set)

Manuscript **Table 5 — Validation columns**. Scores on the **held-out validation split**:
TLS-derived reference geometry excluded from every train/test split and reserved for one
final evaluation. Two sources, **85 images**, evaluated together:

- **Allen et al. 2025** — 24 images (TLS-validated crowns; closed canopy)
- **Frey et al. 2026** — 61 images (EcoSense TLS, ortho_3919)

Dataset **v0.22** throughout. Fine-tuned rows = the **out-of-distribution** model (the leaderboard
model for its geometry). Pretrained/existing rows are **zero-shot** (no MillionTrees training), so
their validation score is split-independent — one number, valid for any split-type row.

Every model uses the **same eval config as its leaderboard test row** (only the split differs).
`F1 = 2·R·P/(R+P)` from the overall columns (same harmonic-mean convention as the Test F1
column; the evaluator does not emit F1 for this split). `CountingMAE` is computed only on
completely-annotated sources (Allen); Frey is partial.

**AP is AP40** (IoU 0.4) everywhere, matching the IoU the recall / mask-aware precision behind
F1 use — AP50 is no longer scored (see `ap50_vs_ap40_existing_models.md` for the comparison that
motivated the switch; AP40 − AP50 is +0.04 to +0.11 on this split). The box and polygon rows are
the v0.22 re-runs of 2026-07-27, the same predictions `validation_ap_completeness.md` re-scores;
they replace the v0.21 values previously shown here, which were depressed by a ~12 % unannotated
border on every v0.21 Allen tile (removed in v0.22). v0.21 can no longer load this split at all
(Allen imagery no longer matches its tree-coverage masks).

> ⚠️ **Table 5 correction:** the polygon Mask R-CNN OOD validation cell in the manuscript draft
> shows AP = **0.57**, which is the **F1** (0.573), not an AP. The measured **AP40 = 0.257**.
> Recall (0.41) and Precision (0.98) in that cell are correct.

## Overall (fill-in values)

### TreeBoxes
| Model | Fine-tuned | Detection Recall | Detection Precision | F1 | AP40 |
|---|:--:|--:|--:|--:|--:|
| CanopyRS DINO (thr 0.30) | ✗ | 0.549 | 0.887 | 0.678 | 0.404 |
| SAM3 | ✗ | 0.538 | 0.613 | 0.573 | 0.340 |
| DeepForest | ✓ (OOD) | 0.328 | 0.818 | 0.468 | 0.150 |
| DeepForest | ✗ | 0.287 | 0.967 | 0.443 | 0.121 |

### TreePoints  *(no crown segmentation / AP)* — **v0.22**
| Model | Fine-tuned | Keypoint Accuracy | Detection Precision | F1 | CountingMAE |
|---|:--:|--:|--:|--:|--:|
| TreeFormer (img 896) | ✓ (OOD) | 0.378 | 0.846 | 0.522 | 91.4 |
| TreeFormer (img 896) | ✗ | 0.348 | 0.940 | 0.508 | 46.9 |
| SAM3 | ✗ | 0.387 | 0.795 | 0.521 | 59.0 |

Re-run on **v0.22** on 2026-08-04 (jobs 38691232 / 38691233 / 38691234), same configs as the
v0.21 rows they replace, so the deltas are dataset-version only. v0.21 values for reference:
0.418 / 0.847 / 0.560 / 92.2 (✓), 0.387 / 0.950 / 0.550 / 55.4 (✗ TreeFormer),
0.387 / 0.798 / 0.521 / 74.8 (SAM3); files kept alongside each result as `*_v0.21.{txt,json}`.
Trimming the Allen border mostly removes false-positive *counts* — CountingMAE drops 15–21 %
for both zero-shot models — while matched-detection rates barely move, except TreeFormer's
Allen keypoint accuracy, which falls in both the fine-tuned (0.541 → 0.462) and pretrained
(0.505 → 0.426) rows. The v0.22 Allen tile is smaller (3007×3090 vs 3766×3848), so at the
fixed 896 resize each crown covers more pixels; that is the same resolution sensitivity that
motivated the 448 → 896 switch, and it is untested at 896-and-above scale. Frey is unchanged
in every row (0.294 / 0.270 / 0.337) — those tiles were not re-tiled, a useful control.

### TreePolygons
| Model | Fine-tuned | Detection Recall | Detection Precision | F1 | Crown Seg. AP40 |
|---|:--:|--:|--:|--:|--:|
| CanopyRS DINO+SAM3 (thr 0.30) | ✗ | 0.479 | 0.929 | 0.632 | 0.317 |
| Mask R-CNN (Detectron2) | ✓ (OOD) | 0.405 | 0.977 | 0.573 | 0.257 |
| Detectree2 | ✗ | 0.461 | 0.634 | 0.534 | 0.287 |
| SAM3 | ✗ | 0.477 | 0.604 | 0.533 | 0.270 |

## Per-source (Allen / Frey)

### TreeBoxes
| Model | FT | Allen R | Allen P | Allen AP40 | Frey R | Frey P | Frey AP40 |
|---|:--:|--:|--:|--:|--:|--:|--:|
| DeepForest | ✓ | 0.386 | 0.656 | 0.181 | 0.271 | 0.980 | 0.118 |
| CanopyRS DINO (0.30) | ✗ | 0.653 | 0.869 | 0.454 | 0.445 | 0.906 | 0.354 |
| SAM3 | ✗ | 0.612 | 0.663 | 0.348 | 0.463 | 0.563 | 0.331 |
| DeepForest | ✗ | 0.356 | 0.937 | 0.148 | 0.219 | 0.997 | 0.093 |

### TreePoints  *(KeypointAccuracy / Precision)*
| Model | FT | Allen K | Allen P | Frey K | Frey P |
|---|:--:|--:|--:|--:|--:|
| TreeFormer (896) | ✓ | 0.462 | 0.760 | 0.294 | 0.931 |
| TreeFormer (896) | ✗ | 0.426 | 0.886 | 0.270 | 0.994 |
| SAM3 | ✗ | 0.437 | 0.751 | 0.337 | 0.839 |

All rows are **v0.22**. v0.21 points per-source, for reference: ✓ 0.541 / 0.762, ✗ 0.505 / 0.907,
SAM3 0.438 / 0.756 (Allen K / Allen P; Frey identical to the v0.22 values shown).

### TreePolygons
| Model | FT | Allen R | Allen P | Allen AP40 | Frey R | Frey P | Frey AP40 |
|---|:--:|--:|--:|--:|--:|--:|--:|
| Mask R-CNN (D2) | ✓ | 0.488 | 0.963 | 0.325 | 0.372 | 0.982 | 0.228 |
| CanopyRS DINO+SAM3 (0.30) | ✗ | 0.583 | 0.882 | 0.341 | 0.438 | 0.948 | 0.319 |
| SAM3 | ✗ | 0.505 | 0.676 | 0.249 | 0.448 | 0.532 | 0.291 |
| Detectree2 | ✗ | 0.488 | 0.604 | 0.274 | 0.434 | 0.664 | 0.300 |

## Config / provenance

| Model | Geometry | Checkpoint / weights | Threshold | Env |
|---|---|---|---|---|
| DeepForest ✓ | boxes | `boxes-best-v2` | 0.10 | main |
| DeepForest ✗ | boxes | release weights | 0.10 | deepforest venv |
| CanopyRS ✗ | boxes, polygons | DINO(+SAM3) | **0.30** | canopyrs venv |
| SAM3 ✗ | all | facebook/sam3 | 0.10 | sam3 venv |
| TreeFormer ✓ | points | 896 isolated ckpt (`point_recall≈0.89`) | 0.10 · img 896 | `.venv-treeformer` |
| TreeFormer ✗ | points | pretrained | 0.10 · img 896 | `.venv-treeformer` |
| Detectree2 ✗ | polygons | `250312_flexi.pth` | 0.10 | detectree2 venv |
| Mask R-CNN ✓ | polygons | `detectron2/out-of-distribution/model_final.pth` | 0.15 · img 448 | canopyrs venv |

Jobs (boxes + polygons, v0.22, 2026-07-27) — the `--save-predictions` dump runs, which also feed
`validation_ap_completeness.md`: 38129318 (canopyrs) / 38129319 (sam3) / 38129320 (detectree2) /
38129321 (deepforest) / 38129730 (fine-tuned boxes) / 38129323 (fine-tuned polygons). These
superseded the v0.21 runs 36329925/6/7 (fine-tuned) and 36352675–9 (existing) whose numbers this
table previously carried.

**Points v0.22 re-run (2026-08-04):** 38691232 (fine-tuned TreeFormer) / 38691233 (pretrained
TreeFormer) / 38691234 (SAM3, points only via the new
`existing_models/slurm/eval_validation_sam3_points.sbatch` — the all-geometry SAM3 script would
have re-run boxes+polygons and clobbered their current v0.22 results from 38128357/38129319).
Configs unchanged from the v0.21 jobs. The pretrained-TreeFormer `*_v0.21.{txt,json}` files in
`existing_models/treeformer/outputs/validation_896_isolated/` are **reconstructed** from the
36352679 job log (the backup copy was made while `/blue` was full and produced a 0-byte file);
their numbers match this table's v0.21 values. The other four v0.21 backups are original.

## Prediction visualizations (10 per source)

Ground-truth vs prediction overlays, ~20 PNGs each (10 Allen + 10 Frey):

- Boxes ✓: `outputs/viz/validation_boxes/` · ✗ DeepForest: `existing_models/deepforest/outputs/validation/viz/` · CanopyRS: `existing_models/canopyrs/outputs/validation/viz_boxes/` · SAM3: `existing_models/sam3/outputs/validation/viz/`
- Points ✓: `outputs/viz/validation_points/` · ✗ TreeFormer: `existing_models/treeformer/outputs/validation_896_isolated/viz/` · SAM3: (in sam3 viz)
- Polygons ✓: `training/polygons/outputs/detectron2/out-of-distribution/viz_validation/` · ✗ Detectree2: `existing_models/detectree2/outputs/validation/viz/` · CanopyRS: `existing_models/canopyrs/outputs/validation/viz_polygons/` · SAM3: `existing_models/sam3/outputs/validation/viz/`
