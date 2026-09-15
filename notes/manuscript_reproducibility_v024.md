# Manuscript reproducibility audit — Doc Tables 2, 3, 4 (v0.24)

Audit date **2026-09-03**. Source of truth for the manuscript:
[Google Doc](https://docs.google.com/document/d/1Ah9ZsmdeVYXUt5FtKJeQt8L2up24-vZPGSI5rXtBeYM/edit)
Tables 2 (TreePoints), 3 (TreeBoxes), 4 (TreePolygons). Compared cell-by-cell against the
result files on disk, then reproduction jobs were launched for every row that a re-run can
independently confirm.

**Headline: 22 of 25 checkable manuscript cells reproduce exactly from files on disk.**
Three problems remain, in descending severity:

1. **Table 3's fine-tuned DeepForest rows do not come from the production recipe** — they
   come from an A/B arm (`aug_ab/*_aug-schedf1`) that no committed leaderboard script
   submits. Re-running `train_boxes.sbatch` today produced materially different validation
   numbers (AP40 0.169 vs the Doc's 0.280).
2. **CanopyRS out-of-distribution rows are missing from Tables 3 and 4** — they exist on
   file and are the best OOD scores in both tables.
3. **CenterNet (Table 2, 2 rows / 8 numbers) has no v0.24 run, no output directory, and no
   sbatch script anywhere in the repo.** Not reproducible at all from this codebase.

---

## 1. Table 2 — TreePoints

| Doc row | Doc (MAE / R / P / F1) | On file | Verdict |
|---|---|---|---|
| TreeFormer FT, WD | 19.55 / 0.78 / 0.76 / 0.77 | 19.549 / 0.781 / 0.759 / 0.770 | reproduces |
| TreeFormer release, WD | 123.43 / 0.74 / 0.70 / 0.72 | 123.43 / 0.743 / 0.702 / 0.722 | reproduces |
| SAM3, WD | 40.60 / 0.70 / 0.61 / 0.65 | 40.60 / 0.704 / 0.610 / 0.654 | reproduces |
| **CenterNet, WD** | 41.8 / 0.69 / 0.64 / 0.66 | **no file** | **unreproducible** |
| TreeFormer release, OOD test | 17.2 / 0.84 / 0.78 / 0.81 | 17.22 / 0.841 / 0.781 / 0.810 | reproduces |
| TreeFormer release, OOD val | 46.9 / 0.35 / 0.94 / 0.51 | 46.88 / 0.350 / 0.938 / 0.510 | reproduces |
| TreeFormer FT, OOD test | 50.1 / 0.83 / 0.80 / 0.81 | 50.068 / 0.827 / 0.795 / 0.811 | reproduces |
| TreeFormer FT, OOD val | 74.3 / 0.37 / 0.94 / 0.53 | 74.333 / 0.371 / 0.946 / 0.533 | reproduces |
| SAM3, OOD test | 37.2 / 0.69 / 0.67 / 0.68 | 37.18 / 0.693 / 0.669 / 0.681 | reproduces |
| SAM3, OOD val | 59.0 / 0.43 / 0.79 / 0.56 | 58.96 / 0.429 / 0.788 / 0.556 | reproduces |
| **CenterNet, OOD** | 83.4 / 0.61 / 0.61 / 0.61; val 248.7 / 0.48 / 0.73 / 0.58 | **no file** | **unreproducible** |

### CenterNet is not in the repository
```
find . -ipath '*centernet*'        -> nothing
grep -rl -i centernet --include=*.sbatch  -> nothing
```
There is no `existing_models/centernet/`, no result file, and no submit script. The eight
CenterNet numbers in Doc Table 2 predate v0.24 and cannot be regenerated with current
instructions. **Either drop both CenterNet rows or restore the code + re-run before
submission** — as written they are the only cells in Tables 2–4 with no provenance at all.

---

## 2. Table 3 — TreeBoxes

| Doc row | Doc | On file (canonical dir) | Verdict |
|---|---|---|---|
| CanopyRS, WD | 12.7 / 0.71 / 0.76 / 0.74 | 12.73 / 0.711 / 0.760 / 0.735 | reproduces |
| **DeepForest FT, WD** | 11.62 / 0.68 / 0.74 / 0.71 | 10.82 / 0.677 / 0.748 / 0.711 | **wrong source, see below** |
| SAM3, WD | 37.0 / 0.66 / 0.48 / 0.55 | 37.06 / 0.657 / 0.468 / 0.547 | reproduces |
| DeepForest release, WD | 11.9 / 0.41 / 0.64 / 0.50 | 11.92 / 0.413 / 0.642 / 0.503 | reproduces |
| **CanopyRS, OOD** | **row absent** | 13.79 / 0.842 / 0.864 / **0.853**; val 0.582 / 0.886 / 0.703 / AP40 0.473 / AP60 0.275 / MAE 25.92 | **missing row** |
| **DeepForest FT, OOD** | 13.2 / 0.70 / 0.83 / 0.76; val 0.39 / 0.89 / 0.53 / 0.28 / 0.12 | 13.67 / 0.690 / 0.824 / 0.751; val 0.328 / 0.880 / **0.478** / **0.169** / **0.055** | **wrong source + unstable** |
| SAM3, OOD | 40.2 / 0.72 / 0.58 / 0.64; val 0.57 / 0.60 / 0.58 / 0.41 / 0.20 | 40.15 / 0.722 / 0.580 / 0.643; val 0.569 / 0.599 / 0.584 / 0.409 / 0.196 | reproduces |
| DeepForest release, OOD | 13.1 / 0.54 / 0.78 / 0.64; val 0.29 / 0.97 / 0.44 / 0.13 / 0.01 | 13.10 / 0.540 / 0.781 / 0.639; val 0.289 / 0.966 / 0.445 / 0.130 / 0.007 | reproduces |

### 2a. The fine-tuned DeepForest rows were transcribed from an A/B arm, not the leaderboard

The Doc's numbers match `training/boxes/outputs/aug_ab/*_aug-schedf1/` to three decimals:

| Cell | Doc | `aug_ab/*_aug-schedf1` | `outputs/<split>` (production) |
|---|--:|--:|--:|
| WD MAE | 11.62 | **11.616** | 10.823 |
| WD R / P / F1 | 0.68 / 0.74 / 0.71 | **0.677 / 0.743 / 0.708** | 0.677 / 0.748 / 0.711 |
| OOD test MAE | 13.2 | **13.232** | 13.671 |
| OOD test R / P / F1 | 0.70 / 0.83 / 0.76 | **0.703 / 0.833 / 0.762** | 0.690 / 0.824 / 0.751 |
| OOD val R / P / F1 | 0.39 / 0.89 / 0.53 | **0.386 / 0.888 / 0.538** | 0.328 / 0.880 / 0.478 |
| OOD val AP40 / AP60 | 0.28 / 0.12 | **0.280 / 0.116** | **0.169 / 0.055** |

`aug_ab/*_aug-schedf1` is job `40785325` — an augmentation × LR-schedule × F1-checkpoint-
selection A/B, run out of `train_boxes_aug_ab.sbatch`. **No script in
`slurm/submit_leaderboard_v024.sh` produces it.** The committed production recipe
(`training/slurm/train_boxes.sbatch`, which now carries `--augment` permanently) was re-run
as job `40874551` and finished 2026-09-03; those are the "production" numbers above.

A reader following the repo's own instructions gets the production column, not the Doc column.

### 2b. The OOD validation cell is the least stable number in the paper

Three runs of the same model family on the same held-out Allen+Frey TLS split:

| Run | Recipe | Val R | Val P | Val F1 | Val AP40 | Val AP60 |
|---|---|--:|--:|--:|--:|--:|
| `_v024_noaug_backup` (40371580) | no augmentation | 0.374 | 0.890 | 0.527 | 0.248 | 0.095 |
| `aug_ab/..._aug-schedf1` (40785325) | aug + sched + F1 select | 0.386 | 0.888 | 0.538 | **0.280** | 0.116 |
| `outputs/out-of-distribution` (40874551/52) | aug only (**production**) | 0.328 | 0.880 | 0.478 | **0.169** | 0.055 |

Val AP40 spans **0.169 → 0.280 (a 66 % relative spread)** and AP60 spans 0.055 → 0.116
(**2.1×**) across three runs of nominally the same fine-tuned box model. Test-split numbers
are far tighter (F1 0.737–0.762). This cell carries the manuscript's "fine-tuning on
MillionTrees improves generalization" claim for boxes and should not be reported from a
single draw.

### 2c. CanopyRS OOD row is missing

`existing_models/canopyrs/outputs/out-of-distribution/results_boxes_out-of-distribution.txt`
exists and gives **test F1 0.853 / AP40 0.633** — the highest box score in the study, and
higher than every row Table 3 currently prints. The WD CanopyRS row *is* in the table, so
the omission is asymmetric. Same for its validation cells (F1 0.703 / AP40 0.473).

---

## 3. Table 4 — TreePolygons

| Doc row | Doc (R / P / F1; val R / P / F1 / AP40 / AP60) | On file | Verdict |
|---|---|---|---|
| CanopyRS, WD | 0.73 / 0.87 / 0.79 | 0.726 / 0.868 / 0.791 | reproduces |
| Mask R-CNN (D2), WD | 0.64 / 0.91 / 0.75 | 0.641 / 0.909 / 0.752 | reproduces |
| SAM3, WD | 0.56 / 0.61 / 0.59 | 0.563 / 0.614 / 0.587 | reproduces |
| DetectTree2, WD | 0.53 / 0.58 / 0.55 | 0.534 / 0.577 / 0.555 | reproduces |
| **CanopyRS, OOD** | **row absent** | 0.803 / 0.878 / **0.839**; val 0.546 / 0.911 / 0.683 / AP40 0.364 pooled (**0.402 macro**) / 0.132 | **missing row** |
| DetectTree2, OOD | 0.62 / 0.64 / 0.63; 0.50 / 0.63 / 0.56 / 0.35 / 0.14 | 0.617 / 0.640 / 0.628; 0.497 / 0.629 / 0.555 / 0.352 / 0.142 | reproduces |
| Mask R-CNN (D2), OOD | 0.43 / 0.94 / 0.59; 0.43 / 0.97 / 0.59 / 0.29 / 0.11 | 0.435 / 0.937 / 0.594; 0.425 / 0.973 / 0.592 / 0.290 / 0.115 | reproduces |
| SAM3, OOD | 0.56 / 0.67 / 0.61; 0.51 / 0.58 / 0.55 / 0.34 / 0.14 | 0.561 / 0.674 / 0.612; 0.519 / 0.584 / 0.550 / 0.341 / 0.139 | reproduces |
| TreeFormer + SAM2, xgeom | 0.35 / 0.79 / 0.49; 0.27 / 0.72 / 0.39 / 0.13 / 0.03 | 0.353 / 0.786 / 0.487; 0.265 / 0.721 / 0.388 / 0.134 / 0.030 | reproduces |

Every present row reproduces. Two notes:

- **CanopyRS OOD is missing here too**, and it is again the best OOD polygon result
  (test F1 0.839 vs the printed best of 0.628). If the row is added, its **validation AP must
  use the macro value 0.402 / 0.174, not the pooled 0.364 / 0.132** — CanopyRS is the one row
  where pooled ≠ macro (per-source score distributions differ enough that one joint PR curve
  under-reads). Every other polygon Val cell has pooled ≈ macro.
- **Dropping both DeepForest Mask R-CNN rows is correct and intentional** ("Mask R-CNN" =
  Detectron2 throughout the manuscript; the OOD run was also a failed configuration,
  recall 0.132 / AP40 0.007). No action.
- The zero-shot **SAM3 cross-geometry baseline** (test F1 0.612) is on file but not in the
  table; it beats the TreeFormer+SAM2 cross-geometry row (0.487). Editorial call, not an error.

---

## 4. Reproduction jobs launched 2026-09-03

Two classes. Zero-shot evals are inference-only and **must** reproduce to 3 dp — any drift is
a determinism or data-version bug. The retrain measures genuine seed noise.

All reproduction runs write to a **separate output namespace** so the manuscript-backing files
are never clobbered: the four eval scripts gained an `MT_OUT_SUBDIR` override (default
unchanged), and the box retrain writes to `training/boxes/outputs/repro_seed1/`.

| Job | Script | Rows covered | Expectation |
|---|---|---|---|
| `40991191` | `eval_sam3.sbatch` | 6 SAM3 cells, Tables 2/3/4, both splits | exact |
| `40991192` | `eval_deepforest.sbatch` | DeepForest release, Table 3, both splits | exact |
| `40991193` | `eval_canopyrs.sbatch` | CanopyRS boxes + polygons, Tables 3/4 — **confirms the two missing OOD rows** | exact |
| `40991194` | `eval_detectree2.sbatch` | detectree2, Table 4, both splits (canonical OOD came from recovery job `40405315_1`) | exact |
| `40991199` | `train_boxes_repro.sbatch` (new) | Table 3 fine-tuned box rows, production recipe at `--seed 1` | seed spread |
| `40991200` | `eval_validation_boxes_repro.sbatch` (new) | Table 3 Val cells — a third draw on the §2b unstable cell | seed spread |

`train_boxes_repro.sbatch` is byte-identical to `train_boxes.sbatch` except `--seed 1` and the
output dir, so the prod-vs-repro delta isolates stochasticity from recipe.

Also right-sized `--mem` per CLAUDE.md §6 while touching these scripts: sam3 100G→56G
(observed peak 41G), deepforest 64G→24G (peak 8G), canopyrs 150G→88G (peak 67G). detectree2
audits OK at 200G.

## 5. Recommended actions before submission

1. **Decide which box recipe Table 3 cites**, then make the repo produce it. Either
   (a) re-transcribe the fine-tuned rows from `train_boxes.sbatch` output, or (b) promote the
   `aug-schedf1` configuration into `train_boxes.sbatch` and re-run. Today the Doc cites a
   configuration the leaderboard scripts cannot produce.
2. **Report the OOD box validation cell with its spread**, or fix the checkpoint-selection
   criterion — AP40 0.169–0.280 across three draws is too wide to state as one number.
3. **Add the CanopyRS OOD rows to Tables 3 and 4** (macro AP for the polygon val cell).
4. **Drop or regenerate CenterNet** (Table 2).
