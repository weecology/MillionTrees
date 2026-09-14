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

**The headline AP is AP40** (IoU 0.4) everywhere, matching the IoU the recall / mask-aware
precision behind F1 use — AP50 is not scored (see `ap50_vs_ap40_existing_models.md` for the
comparison that motivated the switch; AP40 − AP50 is +0.04 to +0.11 on this split).
**AP60** (IoU 0.6) is reported beside it as the strict-localisation complement: identical
predictions and identical ranking, only the overlap a match requires changes, so the AP40 → AP60
drop is a direct read of how well each model *delineates* the crowns it finds rather than how
many it finds. Both columns come from the same predictions; neither is meaningful alone. The box and polygon rows are
the v0.22 re-runs of 2026-07-27, the same predictions `validation_ap_completeness.md` re-scores;
they replace the v0.21 values previously shown here, which were depressed by a ~12 % unannotated
border on every v0.21 Allen tile (removed in v0.22). v0.21 can no longer load this split at all
(Allen imagery no longer matches its tree-coverage masks).

> ⚠️ **Table 5 correction:** the polygon Mask R-CNN OOD validation cell in the manuscript draft
> shows AP = **0.57**, which is the **F1** (0.573), not an AP. The measured **AP40 = 0.257**.
> Recall (0.41) and Precision (0.98) in that cell are correct.

## Overall (fill-in values)

### TreeBoxes
| Model | Fine-tuned | Detection Recall | Detection Precision | F1 | AP40 | AP60 | AP60/AP40 |
|---|:--:|--:|--:|--:|--:|--:|--:|
| CanopyRS DINO (thr 0.30) | ✗ | 0.549 | 0.887 | 0.678 | 0.404 | 0.214 | 0.53 |
| SAM3 | ✗ | 0.538 | 0.613 | 0.573 | 0.340 | 0.143 | 0.42 |
| DeepForest | ✓ (OOD) | 0.328 | 0.818 | 0.468 | 0.150 | 0.043 | 0.29 |
| DeepForest | ✗ | 0.287 | 0.967 | 0.443 | 0.121 | 0.007 | 0.06 |

The ranking is unchanged by the stricter IoU, but the **spread widens sharply**. CanopyRS keeps
53 % of its AP going from IoU 0.4 to 0.6; pretrained DeepForest keeps 6 %. Nearly every box
pretrained DeepForest gets credit for at IoU 0.4 is a loose match that fails at 0.6, so its 0.121
AP40 describes finding trees, not delineating them. Fine-tuning moves that retention from 0.06 to
0.29 — a larger relative gain than the AP40 column alone shows (0.121 → 0.150).

### TreePoints  *(no crown segmentation / AP)* — **v0.22**, except the countfix row (**v0.23**)
| Model | Fine-tuned | Keypoint Accuracy | Detection Precision | F1 | CountingMAE |
|---|:--:|--:|--:|--:|--:|
| SAM3 | ✗ | 0.387 | 0.795 | 0.521 | 59.0 |
| TreeFormer (img 896) | ✓ (OOD) | 0.378 | 0.846 | 0.522 | 91.4 |
| TreeFormer (img 896) | ✗ | 0.348 | 0.940 | 0.508 | 46.9 |
| TreeFormer countfix (img 896) — **v0.23** | ✓ (OOD) | 0.265 | 0.658 | 0.378 | 136.5 |

**The fine-tuned point rows are the generalization result, not a broken run.** The countfix
model (job 39014685) is the best fine-tuned point model MillionTrees has: on the
within-distribution test split it beats the pretrained checkpoint on both detection
(F1 0.782 vs 0.726) and counting (nMAE 0.214 vs 1.611), and it is the first run where the
count loss produces a gradient at all. It is nevertheless the *worst* row in this table. Its
Allen counting slope is **−0.309** — per-image predicted counts are anti-correlated with
truth — against the pretrained checkpoint's 46.9 MAE on the identical images.

Read across the three evaluations, fine-tuning TreeFormer helps in proportion to how closely
the evaluation resembles its training sources (within-distribution: large gain;
out-of-distribution: loses detection, keeps a counting gain; held-out TLS: loses badly). The
parameters adapt to specific sources rather than learning transferable tree density. Fixing
the dead count path (39014685) removed one confound — the count-blind fine-tune scored an even
worse Allen MAE of 302.2 / nMAE 4.508 here — but it did not change the direction of the
finding, which is why both fine-tuned rows are shown.

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
| Model | Fine-tuned | Detection Recall | Detection Precision | F1 | Crown Seg. AP40 | Crown Seg. AP60 | AP60/AP40 |
|---|:--:|--:|--:|--:|--:|--:|--:|
| CanopyRS DINO+SAM3 (thr 0.30) | ✗ | 0.479 | 0.929 | 0.632 | 0.317 | 0.122 | 0.37 |
| Mask R-CNN (Detectron2) | ✓ (OOD) | 0.405 | 0.977 | 0.573 | 0.257 | 0.092 | 0.33 |
| Detectree2 | ✗ | 0.461 | 0.634 | 0.534 | 0.287 | 0.100 | 0.35 |
| SAM3 | ✗ | 0.477 | 0.604 | 0.533 | 0.270 | 0.095 | 0.35 |

Polygons behave very differently from boxes here: **all four models retain 0.33–0.37 of their
AP40 at IoU 0.6**, a band narrow enough that AP60 re-ranks nothing and separates nobody. Mask
IoU 0.6 is a hard bar for tree crowns — interlocking canopy means the outer 20 % of a crown
boundary is ambiguous in the reference data itself — so on this split AP60 is best read as a
shared ceiling rather than a discriminator between polygon models. The box column is where the
strict threshold does work.

## Per-source (Allen / Frey)

### TreeBoxes
| Model | FT | Allen R | Allen P | Allen AP40 | Allen AP60 | Frey R | Frey P | Frey AP40 | Frey AP60 |
|---|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DeepForest | ✓ | 0.386 | 0.656 | 0.181 | 0.054 | 0.271 | 0.980 | 0.118 | 0.031 |
| CanopyRS DINO (0.30) | ✗ | 0.653 | 0.869 | 0.454 | 0.180 | 0.445 | 0.906 | 0.354 | 0.249 |
| SAM3 | ✗ | 0.612 | 0.663 | 0.348 | 0.086 | 0.463 | 0.563 | 0.331 | 0.200 |
| DeepForest | ✗ | 0.356 | 0.937 | 0.148 | 0.006 | 0.219 | 0.997 | 0.093 | 0.009 |

**The two sources swap places under the strict threshold.** Every box model scores higher AP40 on
Allen than on Frey, and every one except fine-tuned DeepForest *reverses* that at AP60 (CanopyRS
0.180 Allen vs 0.249 Frey; SAM3 0.086 vs 0.200). Allen is closed canopy where crowns interlock and
a loose box still overlaps something; Frey's TLS crowns are better separated, so a detection that
lands is tight. An AP40-only table reads Allen as the easier source; it is only easier if you do
not ask for precise extents.

### TreePoints  *(KeypointAccuracy / Precision)*
| Model | FT | Allen K | Allen P | Frey K | Frey P |
|---|:--:|--:|--:|--:|--:|
| TreeFormer (896) | ✓ | 0.462 | 0.760 | 0.294 | 0.931 |
| TreeFormer (896) | ✗ | 0.426 | 0.886 | 0.270 | 0.994 |
| SAM3 | ✗ | 0.437 | 0.751 | 0.337 | 0.839 |
| TreeFormer countfix (896) — **v0.23** | ✓ | 0.349 | 0.535 | 0.180 | 0.782 |

All rows are **v0.22** except the countfix row (**v0.23**, job 39081455). v0.21 points
per-source, for reference: ✓ 0.541 / 0.762, ✗ 0.505 / 0.907, SAM3 0.438 / 0.756
(Allen K / Allen P; Frey identical to the v0.22 values shown). The countfix model degrades on
*both* sources, Frey (0.294 → 0.180 K) proportionally more than Allen — the generalization
loss is not an Allen-specific tiling artifact.

### TreePolygons
| Model | FT | Allen R | Allen P | Allen AP40 | Allen AP60 | Frey R | Frey P | Frey AP40 | Frey AP60 |
|---|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Mask R-CNN (D2) | ✓ | 0.488 | 0.963 | 0.325 | 0.062 | 0.372 | 0.982 | 0.228 | 0.122 |
| CanopyRS DINO+SAM3 (0.30) | ✗ | 0.583 | 0.882 | 0.341 | 0.060 | 0.438 | 0.948 | 0.319 | 0.184 |
| SAM3 | ✗ | 0.505 | 0.676 | 0.249 | 0.043 | 0.448 | 0.532 | 0.291 | 0.146 |
| Detectree2 | ✗ | 0.488 | 0.604 | 0.274 | 0.050 | 0.434 | 0.664 | 0.300 | 0.149 |

The same reversal, and stronger: on Allen **every** polygon model collapses to 0.043–0.062 AP60
(17–19 % of its AP40) while holding 0.122–0.184 on Frey (50–58 %). Allen's masks at IoU 0.6 are
close to unscorable for all four models, which is why the overall polygon AP60 column above is so
compressed — it is a macro-average of one near-floor source and one ordinary one.

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

### Where the AP60 column comes from

AP60 was **not** re-run: it is re-scored offline from the same v0.22 `--save-predictions` dumps
in `outputs/validation_preds/*.pkl` that produced the AP40 column, by

```bash
python scripts/rescore_validation_ap.py \
  --predictions outputs/validation_preds/*.pkl --regimes all \
  --ap-ious 0.4 0.5 0.6 --out-csv notes/validation_ap40_ap60.csv
```

Same predictions, same score thresholds, same macro-over-sources aggregation — only the matching
IoU differs, which is the whole point of the column. `notes/validation_ap40_ap60.csv` holds every
cell plus AP50 and the pooled/per-source variants.

The re-scorer reproduces the published AP40 **exactly in all sixteen per-source cells and all four
TreeBoxes overall cells**. Two polygon overall cells differ: CanopyRS 0.330 vs the 0.317 tabled
here, Mask R-CNN 0.276 vs 0.257 — in both cases the tabled value is a *pooled* average over all 85
images while the re-scorer's is the *macro* average over the two sources (0.341/0.319 and
0.325/0.228), which is the convention `MillionTreesDataset.eval()` and the rest of this table use.

That is not a re-scorer artifact: **`TreePolygonsStreamingEvalState` aggregates differently from
`standard_group_eval`.** The latter's headline `Average X:` is an unweighted mean over sources (it
says so in the comment); the streaming polygon evaluator's is `sum/n` over images for the
elementwise metrics and a single global torchmetrics object for AP. So every polygon `results_*.txt`
overall row is pooled while every box one is macro. The streaming class's docstring claims its
"results match `standard_group_eval` semantics", and `test_TreePolygons_eval_stream_matches_legacy`
passes — but that test's polygon split is a single image (`polygon_data.loc[2, "split"] = "test"`),
so it has one source and pooled ≡ macro by construction. The difference is invisible when sources
score alike and reaches 0.04 of AP when they do not.
The AP60 values and the AP60/AP40 ratios in the polygon overall table are computed on the macro
basis throughout, so dividing the printed AP60 by the printed AP40 for those two rows gives 0.38
and 0.36 rather than the tabled 0.37 and 0.33. The per-source polygon table is unaffected.

From v0.23 onward this reconciliation is unnecessary: `AP60` is now a registered metric on both
`TreeBoxesDataset` and `TreePolygonsDataset` (and streamed by `TreePolygonsStreamingEvalState`),
so every eval writes `Average AP60:` beside `Average AP40:` into its `results_*.txt`.

## Prediction visualizations (10 per source)

Ground-truth vs prediction overlays, ~20 PNGs each (10 Allen + 10 Frey):

- Boxes ✓: `outputs/viz/validation_boxes/` · ✗ DeepForest: `existing_models/deepforest/outputs/validation/viz/` · CanopyRS: `existing_models/canopyrs/outputs/validation/viz_boxes/` · SAM3: `existing_models/sam3/outputs/validation/viz/`
- Points ✓: `outputs/viz/validation_points/` · ✗ TreeFormer: `existing_models/treeformer/outputs/validation_896_isolated/viz/` · SAM3: (in sam3 viz)
- Polygons ✓: `training/polygons/outputs/detectron2/out-of-distribution/viz_validation/` · ✗ Detectree2: `existing_models/detectree2/outputs/validation/viz/` · CanopyRS: `existing_models/canopyrs/outputs/validation/viz_polygons/` · SAM3: `existing_models/sam3/outputs/validation/viz/`
