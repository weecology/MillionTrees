# MillionTrees v0.24 manuscript tables

Generated 2026-08-28 from the v0.24 leaderboard refresh (test jobs `40371580`–`40371592`,
submitted 2026-08-27 via `slurm/submit_leaderboard_v024.sh`; detectree2 OOD recovered as
`40405315_1`) and the v0.24 validation-split re-run (jobs `40510226`–`40510234`, 2026-08-28).
Regenerate `docs/leaderboard.md` with `uv run python scripts/make_benchmark_table.py`.

The headline AP is **AP40** (IoU 0.4); **AP60** (IoU 0.6) is the strict-localisation
complement — same predictions, same ranking. Recall is detection recall (boxes), keypoint
accuracy (points), or mask recall (polygons); Precision is mask-aware precision; F1 is their
harmonic mean. Test/Val AP values are the pooled headline `Average AP*` from the result files
(same convention `docs/leaderboard.md` uses); the one place pooled ≠ macro is CanopyRS polygon
validation — see the Table 4 note.

Validation = the held-out Allen et al. 2025 + Frey et al. 2026 TLS split (3661 annotations,
byte-identical CSV in v0.23 and v0.24). It exists only for the out-of-distribution models, so
within-distribution rows have no Val cells. **Every zero-shot Val row reproduced its v0.23
value to 3 dp** (confirming the validation set did not change); the fine-tuned Val rows moved
because those models were retrained on v0.24.

## ⚠️ Status / caveats

1. **Table 5 / Table 6 (weak supervision, boxes + polygons) is still SUSPENDED** (see
   `notes/weak_supervision_pretraining_table.md`, audit 2026-08-21). The v0.24 OOD
   box-pretrained polygon arm (`40371590_1`) was still running at generation time. Do not put
   v0.24 box/polygon weak-supervision numbers in the manuscript until the stage-1 audit is
   closed. **The points weak-supervision arm (Table 7 below) IS complete and clear**
   (`notes/weak_supervision_points_table.md`, 2026-09-02) — it is a separate two-stage
   experiment with its own verified stage 1, not part of the suspended box/polygon audit.

2. **Box AP is not comparable across versions.** v0.24 box AP40/AP60 uses
   `max_detection_thresholds=[1,10,1000]`; the v0.23 manuscript box AP used the torchmetrics
   default top-100 cap (fixed 2026-08-26). Combined with the OOD-split repair and the retrain,
   box AP40 test deltas are confounded by **three** changes at once. **Test AP60 is a new
   column for v0.24** — there is no v0.23 test AP60.

3. **DeepForest Mask R-CNN polygons has no validation cell** — there is no
   `eval_validation_*` sbatch for that row, and its OOD test run is a failed configuration
   (see Table 4).

---

## Table 2 — TreePoints

| Model | Fine-tuned | Split | Test MAE | Test R | Test P | Test F1 | Val MAE | Val R | Val P | Val F1 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| TreeFormer (count-loss fix) | Yes | Within-distribution | 19.55 | 0.781 | 0.759 | 0.770 | – | – | – | – |
| TreeFormer (release) | No | Within-distribution | 123.43 | 0.743 | 0.702 | 0.722 | – | – | – | – |
| SAM3 | No | Within-distribution | 40.60 | 0.704 | 0.610 | 0.654 | – | – | – | – |
| TreeFormer (release) | No | Out-of-distribution | 17.22 | 0.841 | 0.781 | 0.810 | 46.88 | 0.350 | 0.938 | 0.510 |
| TreeFormer (count-loss fix) | Yes | Out-of-distribution | 50.07 | 0.827 | 0.795 | 0.811 | 74.33 | 0.371 | 0.946 | 0.533 |
| SAM3 | No | Out-of-distribution | 37.18 | 0.693 | 0.669 | 0.681 | 58.96 | 0.429 | 0.788 | 0.556 |

**Change vs v0.23 (Doc Table 2 / `notes/manuscript_tables_v023.md`):**

| Row | Metric | v0.23 | v0.24 | Δ |
|---|---|--:|--:|--:|
| TF countfix, WD | F1 | 0.78 | 0.770 | ≈0 |
| TF release, WD | F1 / MAE | 0.73 / 118.7 | 0.722 / 123.4 | ≈0 |
| SAM3, WD | F1 / MAE | 0.64 / 43.5 | 0.654 / 40.6 | +0.01 |
| **TF release, OOD test** | **F1 / MAE** | **0.75 / 66.8** | **0.810 / 17.2** | **+0.06 / −49.6** |
| **TF countfix, OOD test** | **F1 / MAE** | **0.67 / 38.4** | **0.811 / 50.1** | **+0.14 / +11.7** |
| SAM3, OOD test | F1 / MAE | 0.65 / 37.4 | 0.681 / 37.2 | ≈0 |
| TF release, OOD **val** | F1 / MAE | 0.51 / 46.9 | 0.510 / 46.9 | reproduces (zero-shot) |
| **TF countfix, OOD val** | **F1 / MAE** | **0.38 / 136.5** | **0.533 / 74.3** | **+0.15 / −62.1** |
| SAM3, OOD val | F1 / MAE | 0.56 / 59.0 | 0.556 / 59.0 | reproduces |

- Within-distribution points are essentially unchanged.
- **Out-of-distribution points moved a lot** — the v0.24 OOD point hold-out was rebuilt
  (Amirkolaee et al. 2023, TreeFormer's own pretraining source, dropped; Dubrovin et al. 2024
  added). On the new hold-out the manuscript's central point claim weakens:
  - v0.23: fine-tuned TreeFormer "did not learn general features out-of-distribution"
    (test F1 0.67 vs pretrained 0.75). **v0.24: fine-tuned ties pretrained on test F1**
    (0.811 vs 0.810) and beats it on the TLS validation split (F1 0.533 vs 0.510).
  - The split still runs *against* fine-tuning on **counting**: pretrained MAE 17.2 test /
    46.9 val, fine-tuned 50.1 / 74.3. Story becomes "fine-tuning matches OOD detection but
    degrades OOD counting", not "fine-tuning fails to generalize".
  - The fine-tuned val per-image count slope is now **+0.42** (was negative in v0.23) — the
    "anti-correlated counts" failure is gone.
- CenterNet (in the Doc only) has no v0.24 run.

---

## Table 3 — TreeBoxes

| Model | Fine-tuned | Split | Test MAE | Test R | Test P | Test F1 | Test AP40 | Test AP60 | Val R | Val P | Val F1 | Val AP40 | Val AP60 | Val MAE |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CanopyRS | Yes* | Within-distribution | 12.73 | 0.711 | 0.760 | 0.735 | 0.528 | 0.405 | – | – | – | – | – | – |
| DeepForest | Yes | Within-distribution | 12.80 | 0.681 | 0.669 | 0.675 | 0.514 | 0.400 | – | – | – | – | – | – |
| SAM3 | No | Within-distribution | 37.06 | 0.657 | 0.468 | 0.547 | 0.421 | 0.304 | – | – | – | – | – | – |
| DeepForest | No | Within-distribution | 11.92 | 0.413 | 0.642 | 0.503 | 0.260 | 0.093 | – | – | – | – | – | – |
| CanopyRS | Yes* | Out-of-distribution | 13.79 | 0.842 | 0.864 | 0.853 | 0.633 | 0.472 | 0.582 | 0.886 | 0.701 | 0.473 | 0.275 | 25.92 |
| DeepForest | Yes | Out-of-distribution | 14.72 | 0.684 | 0.798 | 0.737 | 0.469 | 0.304 | 0.374 | 0.890 | 0.527 | 0.248 | 0.095 | 35.75 |
| SAM3 | No | Out-of-distribution | 40.15 | 0.722 | 0.580 | 0.643 | 0.460 | 0.294 | 0.569 | 0.599 | 0.584 | 0.409 | 0.196 | 59.00 |
| DeepForest | No | Out-of-distribution | 13.10 | 0.540 | 0.781 | 0.639 | 0.298 | 0.123 | 0.289 | 0.966 | 0.445 | 0.130 | 0.007 | 22.25 |

**Change vs v0.23 (Doc Table 3):**

| Row | v0.23 test R/P/F1/AP40 | v0.24 test R/P/F1/AP40 | v0.23 val R/P/F1/AP40/AP60 | v0.24 val R/P/F1/AP40/AP60 |
|---|--:|--:|--:|--:|
| CanopyRS WD | 0.66/0.78/0.72/0.49 | 0.71/0.76/0.735/0.53 | – | – |
| DeepForest FT WD | 0.64/0.61/0.62/0.47 | 0.68/0.67/0.675/0.51 | – | – |
| SAM3 WD | 0.58/0.48/0.52/0.39 | 0.66/0.47/0.547/0.42 | – | – |
| DeepForest pre WD | 0.39/0.60/0.47/0.24 | 0.41/0.64/0.503/0.26 | – | – |
| CanopyRS OOD | 0.80/0.86/0.83/0.66 | 0.84/0.86/0.853/0.633 | 0.58/0.89/0.70/0.47/0.28 | 0.58/0.89/0.701/0.473/0.275 (reproduces) |
| **DeepForest FT OOD** | 0.64/0.76/0.70/0.48 | **0.68/0.80/0.737/0.469** | 0.33/0.88/0.48/0.16/0.05 | **0.37/0.89/0.527/0.248/0.095** |
| SAM3 OOD | 0.73/0.58/0.65/0.50 | 0.72/0.58/0.643/0.460 | 0.57/0.60/0.58/0.40/0.20 | 0.57/0.60/0.584/0.409/0.196 (reproduces) |
| DeepForest pre OOD | 0.47/0.78/0.58/0.30 | 0.54/0.78/0.639/0.298 | 0.29/0.97/0.44/0.13/0.01 | 0.29/0.97/0.445/0.130/0.007 (reproduces) |

- Within-distribution box F1 rises ~+0.03–0.05 across all four rows (v0.24 fixed the
  within-distribution row-level split leak, so the test set is now genuinely disjoint).
- **Fine-tuned DeepForest OOD improves on both splits**: test F1 0.70 → 0.737, and the
  TLS validation row moves up materially (F1 0.48 → 0.527, AP40 0.16 → 0.248, AP60 0.05 →
  0.095, MAE 37.0 → 35.8). This *strengthens* the manuscript's "fine-tuning on MillionTrees
  improves generalization" claim for boxes.
- Test AP40 moves are **not** clean version deltas (caveat #2). The interpretable
  within-version signal is the AP40→AP60 retention: CanopyRS 75%/75% (test/val OOD),
  fine-tuned DeepForest 65%/38%, pretrained DeepForest 41%/5%.

---

## Table 4 — TreePolygons

| Model | Fine-tuned | Split | Test R | Test P | Test F1 | Test AP40 | Test AP60 | Val R | Val P | Val F1 | Val AP40 | Val AP60 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CanopyRS | Yes* | Within-distribution | 0.726 | 0.868 | 0.791 | 0.390 | 0.244 | – | – | – | – | – |
| Mask R-CNN (Detectron2) | Yes | Within-distribution | 0.641 | 0.909 | 0.752 | 0.483 | 0.319 | – | – | – | – | – |
| SAM3 | No | Within-distribution | 0.563 | 0.614 | 0.587 | 0.291 | 0.206 | – | – | – | – | – |
| DeepForest Mask R-CNN | Yes | Within-distribution | 0.473 | 0.708 | 0.567 | 0.137 | 0.077 | – | – | – | – | – |
| detectree2 | No | Within-distribution | 0.534 | 0.577 | 0.555 | 0.257 | 0.171 | – | – | – | – | – |
| CanopyRS | Yes* | Out-of-distribution | 0.803 | 0.878 | 0.839 | 0.511 | 0.415 | 0.546 | 0.911 | 0.683 | 0.364 † | 0.132 † |
| detectree2 | No | Out-of-distribution | 0.617 | 0.640 | 0.628 | 0.306 | 0.222 | 0.497 | 0.629 | 0.555 | 0.352 | 0.142 |
| SAM3 | No | Out-of-distribution | 0.561 | 0.674 | 0.612 | 0.256 | 0.175 | 0.519 | 0.584 | 0.549 | 0.341 | 0.139 |
| Mask R-CNN (Detectron2) | Yes | Out-of-distribution | 0.435 | 0.937 | 0.594 | 0.273 | 0.208 | 0.425 | 0.973 | 0.591 | 0.290 | 0.115 |
| DeepForest Mask R-CNN | Yes | Out-of-distribution | 0.132 | 0.521 | 0.211 | 0.007 | 0.005 | – | – | – | – | – |
| TreeFormer + SAM2 | No | Cross-geometry | 0.353 | 0.786 | 0.487 | 0.170 | 0.118 | 0.265 | 0.721 | 0.388 | 0.134 | 0.030 |
| SAM3 (cross-geometry baseline) | No | Cross-geometry | 0.561 | 0.674 | 0.612 | 0.256 | 0.175 | – | – | – | – | – |

† **CanopyRS polygon validation AP is pooled**; the macro (source-balanced, the box-table
convention) value is **AP40 0.402 / AP60 0.174**. Only CanopyRS diverges — its per-source
score distributions differ enough that one joint PR curve under-reads. For the manuscript use
the macro value for this cell (carried unchanged from the v0.23 note; the pooled↔macro gap
reproduced exactly). All other polygon Val cells have pooled ≈ macro.

**Change vs v0.23 (Doc Table 4):**

| Row | v0.23 test R/P/F1/AP40 | v0.24 test R/P/F1/AP40 | v0.23 val R/P/F1/AP40 | v0.24 val R/P/F1/AP40 |
|---|--:|--:|--:|--:|
| CanopyRS WD | 0.76/0.87/0.81/0.38 | 0.73/0.87/0.791/0.39 | – | – |
| Mask R-CNN WD | 0.65/0.92/0.76/0.45 | 0.64/0.91/0.752/0.483 | – | – |
| SAM3 WD | 0.58/0.62/0.60/0.29 | 0.56/0.61/0.587/0.291 | – | – |
| DetectTree2 WD | 0.53/0.60/0.56/0.22 | 0.53/0.58/0.555/0.257 | – | – |
| CanopyRS OOD | 0.82/0.86/0.84/0.42 | 0.80/0.88/0.839/0.511 | 0.55/0.91/0.68/0.40(macro) | 0.55/0.91/0.683/0.402(macro) (reproduces) |
| **Mask R-CNN OOD** | 0.56/0.90/0.69/0.39 | **0.44/0.94/0.594/0.273** | 0.46/0.97/0.62/0.33 | **0.43/0.97/0.591/0.290** |
| **DetectTree2 OOD** | 0.50/0.63/0.56/0.25 | **0.62/0.64/0.628/0.306** | 0.50/0.63/0.56/0.35 | 0.50/0.63/0.555/0.352 (reproduces) |
| SAM3 OOD | 0.47/0.67/0.55/0.25 | 0.56/0.67/0.612/0.256 | 0.51/0.58/0.55/0.34 | 0.52/0.58/0.549/0.341 (reproduces) |
| **TreeFormer+SAM2 xgeom** | 0.46/0.65/0.54/0.21 | **0.35/0.79/0.487/0.170** | – (Doc blank) | 0.27/0.72/0.388/0.134 (new cell) |

- **Within-distribution polygons are stable** (all rows within ±0.02 F1).
- **Out-of-distribution polygons were re-shuffled hard.** v0.24 removed 5 undeclared polygon
  sources (27.8% of the OOD polygon test set) and moved SelvaBox to train-only. The OOD test
  set is now Alejandro Miranda / Bohlman 2008 / Lefebvre 2024 / NEON MultiTemporal / Takeshige
  2025 / Troles 2024 — a harder, more urban/temperate mix. On it:
  - **Fine-tuned Mask R-CNN (Detectron2) OOD drops**: test F1 0.69 → 0.594, AP40 0.39 → 0.273;
    it now **loses to zero-shot detectree2 and SAM3 on test F1**. The validation row moves
    less (F1 0.62 → 0.591). This softens the manuscript's "fine-tuned Mask R-CNN is the best
    OOD polygon model" on the test split but not on validation.
  - **CanopyRS OOD test AP40 rises 0.42 → 0.511**; zero-shot detectree2 / SAM3 OOD recall
    rise ~+0.10.
- **Polygon AP40 → AP60 (pooled), fine-tuned models:**

  | Model | Split | AP40 | AP60 | AP60/AP40 |
  |---|---|--:|--:|--:|
  | Mask R-CNN (Detectron2) | WD test | 0.483 | 0.319 | 66% |
  | Mask R-CNN (Detectron2) | OOD test | 0.273 | 0.208 | 76% |
  | Mask R-CNN (Detectron2) | OOD validation | 0.290 | 0.115 | 40% |
  | DeepForest Mask R-CNN | WD test | 0.137 | 0.077 | 56% |
  | DeepForest Mask R-CNN | OOD test | 0.007 | 0.005 | (failed run) |

  On the two test splits the Detectron2 model keeps 66–76% of its AP40 at the stricter IoU;
  the held-out TLS validation split is where crown delineation actually breaks down (40%).
- **DeepForest Mask R-CNN OOD is a failed run** (test recall 0.132, AP40 0.007). Its
  within-distribution run is healthy (recall 0.473). Best checkpoint is epoch 9/100 by
  val_loss — the known polygon-Mask-R-CNN early-overfit + val_loss-selection artifact
  (`notes/polygon_training_review.md`), amplified by the harder OOD set. **Recommend dropping
  this row from the manuscript OOD polygon table** or flagging it as a failed configuration.
- **Cross-geometry TreeFormer + SAM2 drops** (test F1 0.54 → 0.487, recall 0.46 → 0.35;
  validation F1 0.39) — fewer, higher-precision points propagate to fewer masks. A zero-shot
  **SAM3 cross-geometry baseline beats it on test** (F1 0.612 vs 0.487). Consider whether the
  cross-geometry row should become SAM3.

---

## Table 5 / 6 — weak supervision (boxes, polygons)

**No update.** Box/polygon analysis suspended (`notes/weak_supervision_pretraining_table.md`);
v0.24 OOD box-pretrained polygon arm (`40371590_1`) still running 2026-08-28. Do not
transcribe v0.24 box/polygon weak-supervision numbers into the manuscript until the stage-1
audit is closed.

## Table 7 — weak supervision (points): AutoArborist continue-pretraining

Complete 2026-09-02 (`notes/weak_supervision_points_table.md`; jobs `40777951`–`40777955`,
control reused from `40735121`). The point analogue of Table 6: does inserting a
continue-pretraining stage on **AutoArborist** (Beery et al. 2022 *unsupervised*, ~20k
municipal-inventory point images, 100% train in both split schemes) between the released
TreeFormer checkpoint and the supervised MillionTrees fine-tune beat fine-tuning **directly**
from the released checkpoint? Both arms share one stage-2 recipe (pretrained init, lr 2e-4,
20 epochs, 896 px, `val_loss` selection); the only difference is the stage-2 `--checkpoint`.

Recall = keypoint accuracy; Precision = mask-aware precision; F1 = their harmonic mean.
Validation = held-out Allen et al. 2025 + Frey et al. 2026 TLS (the clean read); Test uses
the MillionTrees test split for checkpoint selection in both arms and is optimistic.

| Split | Arm | Test R | Test P | Test F1 | Test MAE | Val R | Val P | Val F1 | Val MAE† |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Within-distribution | Direct fine-tune | 0.848 | 0.657 | 0.740 | 66.99 | 0.361 | 0.867 | 0.510 | 101.29 |
| Within-distribution | + AutoArborist pretrain | 0.772 | 0.845 | 0.807 | 28.45 | 0.377 | 0.936 | 0.538 | 66.04 |
| Within-distribution | **Δ** | −0.076 | +0.188 | **+0.067** | −38.5 | +0.016 | +0.069 | **+0.028** | −35.2 |
| Out-of-distribution | Direct fine-tune | 0.857 | 0.785 | 0.819 | 29.42 | 0.374 | 0.940 | 0.535 | 60.54 |
| Out-of-distribution | + AutoArborist pretrain | 0.814 | 0.788 | 0.801 | 52.52 | 0.425 | 0.937 | 0.585 | 115.46 |
| Out-of-distribution | **Δ** | −0.043 | +0.003 | **−0.018** | +23.1 | +0.051 | −0.003 | **+0.050** | +54.9 |

† Val MAE is Allen et al. 2025 only (Frey carries no counting ground truth in the eval).

- **On the clean held-out validation split, AutoArborist pretraining improves point F1 on
  both splits (+0.028 WD, +0.050 OOD), entirely through recall** (+0.016 / +0.051), with
  precision already saturated near 0.94. Both validation sources move the same way (Allen
  0.431→0.458 WD / 0.449→0.490 OOD; Frey 0.290→0.296 WD / 0.298→0.359 OOD), so it is not a
  single-source or tiling artifact.
- **This is the first positive weak-supervision result in the paper.** Box→box and
  box→polygon pretraining were null at every transfer depth (Table 6 note): at MillionTrees
  box scale the initialization barely matters. The v0.24 point task has only 3,403
  supervised train images after the AutoArborist demotion, leaving the headroom the boxes
  did not have.
- **Counting stays unstable** — MAE improves WD (test −39, val −35) and degrades OOD (test
  +23, val +55). Report F1/recall as the outcome; MAE is context, consistent with the
  broader TreeFormer point-counting behaviour.
- **Stage-1 sanity:** on a held-out slice of the supervised train split, AutoArborist
  pretraining leaves keypoint accuracy flat (0.739→0.737 WD, 0.575→0.573 OOD) and improves
  count calibration (MAE 132→115 WD, 178→136 OOD) — the stage-2 gain comes from the
  pretraining exposure, not from stage 1 already being a stronger detector.
- **Caveat for absolute numbers:** the direct-fine-tune arm here uses the plain leaderboard
  recipe, not the `--loss-preset pretrain` count-loss fix behind Doc Table 2, so these
  absolute values are not comparable to Table 2's TreeFormer rows — only the within-ablation
  Δ is.

---

## Provenance

Data version confirmed `v0.24` in every job log (`[MillionTrees] Loaded Tree* v0.24 (full)`),
including the validation re-run.

- Boxes FT test/val: `training/boxes/outputs/{split}/results_{split}[_validation].txt`
  (val checkpoint `boxes-best-v1.ckpt`, job 40371580_1)
- Points FT test/val: `training/points/outputs/{split}_896_countfix/results_{split}[_validation].txt`
  (val checkpoint `treeformer-epoch=13-val_loss=1.9105.ckpt`, job 40371581_1)
- Polygons FT (D2) test/val: `training/polygons/outputs/detectron2/{split}/results_{split}[_validation].txt`
- Polygons FT (DeepForest): `training/polygons/outputs/deepforest_annotationsafecrop/{split}/results_{split}.txt`
- Existing models test: `existing_models/<model>/outputs/{split}/results_<geom>_{split}.txt`
- Existing models val: `existing_models/<model>/outputs/validation[_896_isolated]/results_<geom>_out-of-distribution.txt`
- Cross-geometry test: `existing_models/treeformer_sam2/outputs/crossgeometry_v024/…`,
  `existing_models/sam3/outputs/crossgeometry/…`
- Cross-geometry val: `existing_models/treeformer_sam2/outputs/validation/results_polygons_crossgeometry.txt`
  (same v0.24 WD countfix checkpoint `treeformer-epoch=18-val_loss=1.7955.ckpt` as the test row)
