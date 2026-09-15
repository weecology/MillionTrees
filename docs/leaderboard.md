# Leaderboard

# Tasks

## Out-of-distribution

The first task evaluates generalization across geography and acquisition conditions.
Selected source datasets are held out from the train split; models are fine-tuned on
the remaining sources and evaluated on the held-out test sources (no images from test
localities appear in train).

## Within-distribution

The second task is to create the best global detector for individual trees given a set of training and test data. Datasets are split randomly, reflecting information within localities. This is consistent with how most applied users engage with models, by fine-tuning backbone models with sample data from a desired locality.

## Cross-geometry

Off the shelf tools often limit users for a single annotation type. We have 'point' models, 'box' models and 'polygon' models. To create truly global models for biological inference, we need models that can use all available data, not just one annotation geometry. In particular, polygon annotations are very time consuming to create, but are often desirable for downstream usecases. We opted against polygon training sources, for example polygons to points, as this is an unrealistic, or atleast, very uncommon downstream use case.

### Boxes to Polygons

All box sources are used to train and predict all polygon sources. There is no local data from the test localities in train.

### Points to Polygons

All point sources are used to train and predict all polygon sources

### Points to Boxes

All point sources are used to train and predict all box sources.

# Results

All scores below are computed on MillionTrees **v0.24**, on the **test** split of the named
split scheme. Every table in this page is generated from the evaluation result files by
`scripts/make_benchmark_table.py`, which reads an explicit registry of published runs — so
the numbers, and the configuration table that accompanies them, always describe the same run.

Fine-tuned panel figures (within-distribution + out-of-distribution splits, ground truth vs
prediction) are generated from training checkpoints via
`scripts/create_finetuned_visualizations.py` (PNG + SVG under `docs/`).
See [repository_structure.md](repository_structure.md).

## How to read the metrics

- **Recall** is geometry-specific: detection recall (boxes), keypoint accuracy (points),
  or mask recall (polygons).
- **Mask-aware precision** is partial-annotation aware. An unmatched prediction that lands on
  tree-covered pixels is *ignored* rather than counted as a false positive, because most
  MillionTrees sources are incompletely annotated. This is why precision reads high, and it is
  why precision here is not comparable to a standard COCO precision.
- **F1** is the harmonic mean of that recall and that mask-aware precision, computed post-hoc.
- **AP is AP40** — average precision at **IoU 0.4**, the same match threshold recall and
  mask-aware precision use, so AP and F1 agree on what counts as a match. There is no AP50
  column; AP40 runs roughly +0.04 to +0.11 above AP50 on the same predictions, so these numbers
  are not comparable to MillionTrees tables published before 2026-08-04. See
  [notes/ap50_vs_ap40_existing_models.md](https://github.com/weecology/MillionTrees/blob/main/notes/ap50_vs_ap40_existing_models.md).
- **Point matching** uses a GSD-normalized radius per source (a fixed ground distance, not a
  fixed pixel count), so a point prediction is matched at the same real-world tolerance
  regardless of the source's resolution.
- **Counting MAE** is computed only over completely-annotated sources. TreePolygons emits no
  counting metric, so polygon tables have no counting column.

## Reproducing a row

Two things determine a number as much as the model does, and both are recorded in the
**Run configuration** table at the bottom of this page:

1. **Dataset version.** The loaders default to the newest key in their `_versions_dict`, so
   the same command run after a release produces different numbers. Pin it explicitly —
   `TreeBoxesDataset(version="0.24")` — when reproducing anything here. All rows on this page
   are v0.24, supervised sources only (`include_unsupervised=False`, the default).
2. **Score threshold.** The standardized operating point is **0.10**. Two exceptions are
   marked in the configuration table: the CanopyRS box and polygon rows use **0.30**, the
   best-F1 point from a full-test-set sweep (at 0.10 they are badly recall-heavy), and the
   Detectron2 polygon model uses its built-in **0.15** `ROI_HEADS.SCORE_THRESH_TEST`. These are
   per-model tuned exceptions, not a re-standardization.

Beyond those, the configuration table lists the checkpoint, evaluation image size, Python
environment, and full training hyperparameters per row. Note that several published runs
deliberately differ from the script defaults — the point rows run at image size 896 (default
448) with `--loss-preset pretrain`, and the box rows use different early-stopping patience on
each split — so running a bare command from this page without those flags will not reproduce
the number.

### Known caveats

- **Neither the box nor the point trainer sets a random seed**, so those rows are not
  bit-reproducible; expect run-to-run variation. The polygon trainers use seed 42.
- All fine-tuned rows (boxes, points, Detectron2 and DeepForest polygons) were retrained on
  v0.24 for this refresh; every row on this page is trained and scored on the same version.
- **Cross-geometry** is defined as predicting polygons from another annotation geometry. It is
  not applicable to box or point prediction, so those tables are absent rather than zero-filled.

## TreeFormer fine-tunes well and generalizes poorly

The point rows show a clear gradient, and it is a result rather than an artifact. Fine-tuning
TreeFormer helps in proportion to how closely the evaluation resembles the training data, and
hurts once it does not:

| Evaluation | Relationship to train | Fine-tuned | Pretrained |
|---|---|---|---|
| Within-distribution test | same sources | F1 **0.782**, counting nMAE **0.214** | F1 0.726, nMAE 1.611 |
| Out-of-distribution test | held-out aerial sources | F1 0.675, nMAE **0.350** | F1 **0.750**, nMAE 0.489 |
| Validation (held-out TLS) | different reference geometry | keypoint acc 0.265, Allen MAE 136.5 (nMAE 2.035) | Allen MAE **46.9** |

Within-distribution, fine-tuning is decisively better on both detection and counting.
Out-of-distribution it already loses detection F1 while still improving counting. On the
held-out TLS validation split it loses badly, and the per-image count slope turns negative
(−0.309 on Allen), meaning predicted counts are anti-correlated with truth. The conclusion is
that TreeFormer's parameters adapt to the specific sources they are fine-tuned on rather than
learning a transferable notion of tree density — it is a strong fine-tuning target and a weak
generalizer. Applied users fine-tuning on local data should expect the within-distribution
column; nobody should expect the validation column to follow from it.

![Pretrained vs fine-tuned TreeFormer across the three evaluation regimes](public/treeformer_generalization_figure.png)

One row per regime, pretrained (left) against fine-tuned (right). Every prediction is
classified the way `MaskAwareKeypointPrecision` does — matched (green), counted false
positive (red), and the mask-aware exemption for unmatched predictions on tree-covered
pixels (grey) — so panel precision is the reported quantity rather than a generic dot plot.

The source-level breakdown says the same thing. Amirkolaee is the within-distribution source
where fine-tuning moves both axes the way the split does (recall 0.31 → 0.61, precision
0.78 → 0.92, counting MAE 112 → 43). OFO is 1387 of the 3260 out-of-distribution test images
and is what drives that split's aggregate precision drop: the fine-tune finds nearly every
tree (recall 0.95 → 0.98) while precision falls 0.44 → 0.18, which is over-prediction, not a
detection failure. Allen carries the validation counting collapse (MAE 47 → 136).

Precision figures here are mask-aware: an unmatched prediction landing on tree-covered pixels
is exempted rather than counted as a false positive, so precision swings image to image on
closed-canopy sources and only the source-level aggregate is meaningful. That is why the tile
drawn in each panel is not a free choice: of nine candidates from the row's source, the one
shown is whichever has per-tile recall and precision closest to the published aggregate for
**both** models, so no panel contradicts the numbers printed above it.

Regenerate with the frozen TreeFormer venv (CPU, no SLURM job needed). The `--version` pin is
load-bearing — the loader default follows `_versions_dict` and would otherwise draw tiles from
a different packaging than the result files the aggregates come from:

```bash
.venv-treeformer/bin/python scripts/make_generalization_figure.py \
    --version 0.23 --out docs/public/treeformer_generalization_figure.png
# re-style without redoing inference (panels are cached next to the output):
.venv-treeformer/bin/python scripts/make_generalization_figure.py --from-cache \
    --out docs/public/treeformer_generalization_figure.svg
```

![TreePoints: model predictions by split](leaderboard_predictions_points.png)

![TreeBoxes: model predictions by split](leaderboard_predictions_boxes.png)

![TreePolygons: model predictions by split](leaderboard_predictions_polygons.png)

# Submissions

## Submit to the leaderboard

Once you have trained a model and evaluated its performance, you can submit your results to the MillionTrees leaderboard. Here's how:

1. Create a public repository with your code and model training scripts. Make sure to include:
   - Clear instructions for reproducing your results
   - Requirements file listing all dependencies
   - Training configuration files/parameters
   - Code for data preprocessing and augmentation
   - Model architecture definition
   - Evaluation code

2. Generate predictions on the test split, pinning the dataset version explicitly:
   ```python
   dataset = TreeBoxesDataset(version="0.24")  # pin the version; do not rely on the default
   test_dataset = dataset.get_subset("test")   # Use test split
   test_loader = get_eval_loader("standard", test_dataset, batch_size=16)

   predictions = []
   for metadata, images, _ in test_loader:
       pred = model(images)
       predictions.append(pred)
   ```

3. Report the run configuration. A submission is only comparable to the published rows if it
   states each of the following — these are exactly the columns of the **Run configuration**
   table below:

   | Field | Why it matters |
   |---|---|
   | Dataset version and whether unsupervised sources were included | Scores are not comparable across releases |
   | Score threshold | The single largest lever on the recall/precision balance; standard is 0.10 |
   | Weights / checkpoint the score came from | Distinguishes a run from its neighbours in a sweep |
   | Evaluation image size and tiling / inference mode | Changes effective object scale |
   | Training hyperparameters: batch size, GPUs, learning rate, epochs, LR schedule, early stopping / checkpoint-selection criterion, augmentation | Needed to retrain |
   | Random seed, or an explicit statement that none was set | Bounds how much of a difference is noise |
   | Software environment (framework versions, any non-release branch) | Several published rows require specific branches |

4. Submit a pull request to the [MillionTrees repository](https://github.com/weecology/MillionTrees) with:
   - Link to your code repository
   - Model description and approach
   - Performance metrics on test set, with the configuration from step 3
   - Example prediction visualizations
   - Instructions for reproducing results

## Benchmark Results

Fine-tuned models (trained on the MillionTrees train split) vs. pretrained models evaluated zero-shot, all on MillionTrees **v0.24**. All AP is **AP40** (IoU 0.4), the same match threshold behind recall and mask-aware precision; F1 is their harmonic mean. Rows are generated from the result files by `scripts/make_benchmark_table.py` -- see the run configuration table below for what produced each number.

### Split: within-distribution

### TreeBoxes

| Model | Fine-tuned | DetectionRecall | MaskAwarePrecision | F1 | AP40 | CountingMAE |
|---|---|---|---|---|---|---|
| CanopyRS DINO Swin-L | ✗ | 0.711 | 0.760 | 0.735 | 0.528 | 12.729 |
| DeepForest (RetinaNet) | ✓ | 0.681 | 0.669 | 0.675 | 0.514 | 12.803 |
| SAM3 | ✗ | 0.657 | 0.468 | 0.547 | 0.421 | 37.056 |
| DeepForest (release weights) | ✗ | 0.413 | 0.642 | 0.503 | 0.260 | 11.923 |

### TreePoints

| Model | Fine-tuned | KeypointAccuracy | MaskAwarePrecision | F1 | CountingMAE |
|---|---|---|---|---|---|
| TreeFormer (count-loss fix) | ✓ | 0.781 | 0.759 | 0.770 | 19.549 |
| TreeFormer (release weights) | ✗ | 0.743 | 0.702 | 0.722 | 123.431 |
| SAM3 | ✗ | 0.704 | 0.610 | 0.654 | 40.602 |

### TreePolygons

| Model | Fine-tuned | MaskRecall | MaskAwarePrecision | F1 | MaskAccuracy | AP40 |
|---|---|---|---|---|---|---|
| CanopyRS DINO + SAM3 (SelvaMask) | ✗ | 0.726 | 0.868 | 0.791 | 0.293 | 0.390 |
| Mask R-CNN (Detectron2) | ✓ | 0.641 | 0.909 | 0.752 | 0.413 | 0.483 |
| SAM3 | ✗ | 0.563 | 0.614 | 0.587 | 0.159 | 0.291 |
| DeepForest Mask R-CNN | ✓ | 0.473 | 0.708 | 0.567 | 0.129 | 0.137 |
| detectree2 | ✗ | 0.534 | 0.577 | 0.555 | 0.146 | 0.257 |

### Split: out-of-distribution

### TreeBoxes

| Model | Fine-tuned | DetectionRecall | MaskAwarePrecision | F1 | AP40 | CountingMAE |
|---|---|---|---|---|---|---|
| CanopyRS DINO Swin-L | ✗ | 0.842 | 0.864 | 0.853 | 0.633 | 13.792 |
| DeepForest (RetinaNet) | ✓ | 0.684 | 0.798 | 0.737 | 0.469 | 14.719 |
| SAM3 | ✗ | 0.722 | 0.580 | 0.643 | 0.460 | 40.154 |
| DeepForest (release weights) | ✗ | 0.540 | 0.781 | 0.639 | 0.298 | 13.104 |

### TreePoints

| Model | Fine-tuned | KeypointAccuracy | MaskAwarePrecision | F1 | CountingMAE |
|---|---|---|---|---|---|
| TreeFormer (count-loss fix) | ✓ | 0.827 | 0.795 | 0.811 | 50.068 |
| TreeFormer (release weights) | ✗ | 0.841 | 0.781 | 0.810 | 17.216 |
| SAM3 | ✗ | 0.693 | 0.669 | 0.681 | 37.182 |

### TreePolygons

| Model | Fine-tuned | MaskRecall | MaskAwarePrecision | F1 | MaskAccuracy | AP40 |
|---|---|---|---|---|---|---|
| CanopyRS DINO + SAM3 (SelvaMask) | ✗ | 0.803 | 0.878 | 0.839 | 0.322 | 0.511 |
| detectree2 | ✗ | 0.617 | 0.640 | 0.628 | 0.172 | 0.306 |
| SAM3 | ✗ | 0.561 | 0.674 | 0.612 | 0.169 | 0.256 |
| Mask R-CNN (Detectron2) | ✓ | 0.435 | 0.937 | 0.594 | 0.301 | 0.273 |
| DeepForest Mask R-CNN | ✓ | 0.132 | 0.521 | 0.211 | 0.015 | 0.007 |

### Split: crossgeometry

### TreePolygons

| Model | Fine-tuned | MaskRecall | MaskAwarePrecision | F1 | MaskAccuracy | AP40 |
|---|---|---|---|---|---|---|
| TreeFormer + SAM2 | ✗ | 0.353 | 0.786 | 0.487 | 0.177 | 0.170 |

## Run configuration

Everything needed to reproduce a row beyond the dataset version. Score thresholds are the standardized 0.10 unless marked otherwise.

| Model | Geometry | Fine-tuned | Weights / checkpoint | Score threshold | Eval image size | Environment | Training configuration |
|---|---|---|---|---|---|---|---|
| DeepForest (RetinaNet) | TreeBoxes | ✓ | trained from the DeepForest release backbone | 0.10 | native tile | shared `.venv` (`uv run`) | `training/slurm/train_boxes.sbatch`: batch 32 x 2 GPUs, lr 0.01, <=200 epochs, early-stop patience 10 (within) / 15 (OOD), no gradient clipping, no seed set |
| CanopyRS DINO Swin-L | TreeBoxes | ✗ | CanopyRS DINO Swin-L release | **0.30** (per-model tuned; see `notes/canopyrs_threshold_sweep.md`) | CanopyRS default tiling | `existing_models/canopyrs/.venv` | not trained on MillionTrees |
| DeepForest (release weights) | TreeBoxes | ✗ | DeepForest release weights | 0.10 | native tile | deepforest venv | not trained on MillionTrees |
| SAM3 | TreeBoxes | ✗ | `facebook/sam3` | 0.10 | SAM3 default | sam3 venv | not trained on MillionTrees |
| TreeFormer (count-loss fix) | TreePoints | ✓ | `weecology/deepforest-tree-point`, fine-tuned (job 39014685) | 0.10 (`score_integration_radius` 2) | 896 | frozen `.venv-treeformer` (DeepForest `treeformer-training` branch) | `training/slurm/train_points_896_countfix.sbatch`: batch 16 x 3 GPUs, lr 2e-4, 20 epochs, image 896, `--loss-preset pretrain` (enforce_count False, losses count/ot/density_l1, mae_weight 0.025, density_l1_weight 0.05), checkpoint on val_loss, early stopping off, no seed set |
| SAM3 | TreePoints | ✗ | `facebook/sam3` | 0.10 | SAM3 default | sam3 venv | not trained on MillionTrees |
| TreeFormer (release weights) | TreePoints | ✗ | `weecology/deepforest-tree-point` | 0.10 (`score_integration_radius` 2) | 896 | frozen `.venv-treeformer` | not trained on MillionTrees |
| DeepForest Mask R-CNN | TreePolygons | ✓ | torchvision Mask R-CNN, COCO init | 0.10 | 448, tiled stream eval | shared `.venv` (`uv run`) | `training/slurm/train_polygons_deepforest_annotationsafecrop.sbatch`: batch 16, lr 0.01, <=100 epochs, image 448, `--train-aug annotationsafecrop`, `--eval-inference tiled`, `--data-scope subset`, seed 42 |
| Mask R-CNN (Detectron2) | TreePolygons | ✓ | COCO Mask R-CNN R50-FPN 3x; checkpoint `model_final.pth` | 0.15 (`ROI_HEADS.SCORE_THRESH_TEST`) | 448 (`min-size-test` 800, `max-size` 1333) | CanopyRS uv venv (from-source Detectron2) | `training/slurm/train_polygons_detectron2.sbatch`: batch 8, lr 0.01, 50 epochs, image 448, `min-size-train` 640-800, LR steps at 70%/90% of iters, warmup <=1000 iters, seed 42 |
| CanopyRS DINO + SAM3 (SelvaMask) | TreePolygons | ✗ | CanopyRS DINO detector + SAM3 SelvaMask segmenter | **0.30** (per-model tuned) | CanopyRS default tiling | `existing_models/canopyrs/.venv` | not trained on MillionTrees |
| SAM3 | TreePolygons | ✗ | `facebook/sam3` | 0.10 | SAM3 default | sam3 venv | not trained on MillionTrees |
| TreeFormer + SAM2 | TreePolygons | ✗ | TreeFormer points -> SAM2 mask prompting | 0.10 | 896 (points stage) | frozen `.venv-treeformer` + SAM2 | not trained on MillionTrees |
| detectree2 | TreePolygons | ✗ | `250312_flexi.pth` | 0.10 | detectree2 default tiling | detectree2 venv | not trained on MillionTrees |
