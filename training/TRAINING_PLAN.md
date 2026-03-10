# MillionTrees Training Plan & New-User Audit

## Overview

This document describes the training approaches for each MillionTrees task (Boxes, Points, Polygons),
the obstacles encountered as a new user, and recommendations for documentation and code improvements.

---

## Training Approaches

### Boxes: Fine-tuned DeepForest RetinaNet

**Script:** `training/boxes/train_boxes.py`

**Strategy:** The pretrained DeepForest model (`weecology/deepforest-tree`) already achieves ~0.395 average
recall on the random split *without any training on this dataset*. Fine-tuning the RetinaNet backbone
on the MillionTrees training split should improve recall across diverse global sources.

- Model: DeepForest RetinaNet (ResNet-50 backbone)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: Cosine annealing
- Early stopping on validation loss (patience=5)
- Best checkpoint selection

**Key implementation detail:** DeepForest's RetinaNet uses `num_classes=1` with label index 0
for "Tree". This differs from standard torchvision detectors which use label 0 for background.
The `training.md` doc example uses `num_classes=2` with `FastRCNNPredictor`, which would be
correct for Faster R-CNN but breaks for DeepForest's RetinaNet. This mismatch between the
doc example and the actual baseline model architecture is confusing.

### Points: DeepForest with Pseudo-Box Supervision

**Script:** `training/points/train_points.py`

**Strategy:** Point detection is fundamentally a localization task. We convert ground truth
point annotations to small pseudo bounding boxes (30x30 pixels by default), train DeepForest
on these pseudo-boxes, then convert predicted box centroids back to points at inference time.

- Pseudo-box size is configurable (default 30px, tunable via `--pseudo-box-size`)
- Same backbone and optimizer as the box model
- At inference: predicted box centroids become point predictions

**Rationale:** This approach allows using the powerful pre-trained detection backbone rather than
building a point-specific architecture from scratch. The baseline already uses this centroid approach
for evaluation, so training directly with pseudo-boxes is a natural extension.

### Polygons: Mask R-CNN from torchvision

**Script:** `training/polygons/train_polygons.py`

**Strategy:** DeepForest only produces boxes, not masks. For polygon/instance segmentation, we use
torchvision's Mask R-CNN (ResNet-50-FPN-V2) which jointly predicts bounding boxes and per-instance
binary masks.

- Model: `maskrcnn_resnet50_fpn_v2` with COCO pretrained weights
- Modified heads for 2-class output (background + tree)
- Same optimizer strategy (AdamW + cosine annealing)
- Predictions produce binary masks thresholded at 0.5

**Key consideration:** The polygon dataset returns masks as `(N, H, W)` uint8 tensors and
bounding boxes as `BoundingBoxes` objects. The Mask R-CNN expects float masks and standard
tensors for boxes, requiring careful format conversion.

---

## SLURM Submission

All three tasks have SLURM array jobs that sweep the three split schemes:

```bash
# Submit all training jobs
bash training/slurm/submit_all_training.sh
```

Each job runs as `--array=0-2` for `(random, zeroshot, crossgeometry)`.

---

## New-User Audit: Issues Found

### Critical Bugs (Breaks Out of the Box)

**1. Albumentations 2.x / DeepForest Incompatibility**
- `deepforest==1.5.2` imports `from albumentations import functional as F`
- `albumentations>=2.0` removed this module
- **Impact:** Every DeepForest-based example is broken (`baseline_boxes.py`, `baseline_points.py`, `baseline_polygons.py`)
- **Fix:** Pin `albumentations<2.0` in dependencies, or upgrade DeepForest

**2. `format_geometry` Not in Installed DeepForest**
- All baseline examples import `from deepforest.utilities import format_geometry`
- This function does not exist in `deepforest==1.5.2`
- **Impact:** Even after fixing albumentations, the examples crash on import
- **Fix:** Pin a compatible DeepForest version or rewrite examples to use the DataFrame output

**3. Mini Dataset Re-downloads Every Time**
- Mini datasets extract successfully but no `RELEASE_v{version}.txt` file is created
- `dataset_exists_locally()` checks for this file
- With `download=False`, the dataset is not found even though data exists
- With `download=True`, it re-downloads every time
- **Impact:** Users waste bandwidth and see confusing behavior
- **Fix:** Create the release file after successful mini extraction, or modify `dataset_exists_locally` to handle mini datasets

**4. Progress Bar Shows Full Dataset Size for Mini Downloads**
- Mini versions inherit `compressed_size` from full datasets in `_versions_dict`
- TreeBoxes mini shows `42,922,274,383 bytes` (42GB!) in the progress bar
- The actual download is ~60MB
- **Impact:** Users see a terrifying 42GB progress bar for a small test download
- **Fix:** Set correct `compressed_size` in `_get_mini_versions_dict()`

### Documentation Issues

**5. Duplicated Section in `getting_started.md`**
- The "Mini Datasets for Development" section appears twice (lines 15-36 and 37-57)
- Identical content copy-pasted

**6. `training.md` Example Uses Wrong Architecture**
- The doc shows `FastRCNNPredictor` with `num_classes=2` for a Faster R-CNN
- The actual baseline uses DeepForest's RetinaNet with `num_classes=1`
- Label indexing differs between the two (0=bg/1=tree vs 0=tree)
- A new user following `training.md` would build a Faster R-CNN while the baselines use RetinaNet
- The training example also doesn't show how to evaluate using the MillionTrees eval API

**7. `dataset_structure.md` Shows Wrong Iteration Order**
- The doc shows `for image, label, metadata in dataset:` (line ~58)
- The actual order is `(metadata, image, targets)`, not `(image, label, metadata)`
- This inconsistency also appears in other places in the doc

**8. No Training Example That Actually Trains**
- `training.md` shows a skeleton but the existing examples only do *evaluation*
- There is no runnable end-to-end training example
- A new user has to figure out the target format, label encoding, and loss computation entirely on their own

**9. Evaluation Doc Lacks Format Specification**
- `evaluation.md` says predictions need `y`, `labels`, and `scores` keys
- But doesn't specify: what dtype? what shape? what does `labels` contain (0 or 1)?
- The polygon evaluation expects masks in `y`, boxes in `bboxes` -- this is not mentioned

**10. Cross-geometry Split Has 0.000 Scores on Leaderboard**
- Both DeepForest and SAM3 show `0.000` for cross-geometry on Points and Boxes
- This suggests the cross-geometry split may not be populated for those tasks
- Not explained in the docs -- confusing for users trying all three splits

### Warnings and Polish Issues

**11. False "Entire Dataset" Warning**
- `get_train_loader()` warns "You are loading the entire dataset" even when called with a proper subset
- `MillionTreesSubset` inherits from `MillionTreesDataset`, so the isinstance check triggers falsely

**12. `torch.tensor(sourceTensor)` Deprecation Warning**
- `milliontrees_dataset.py:45` uses `torch.tensor(self.metadata_array[idx])`
- Should use `.detach().clone()` instead

**13. `FutureWarning` on `DataFrameGroupBy.apply`**
- All three dataset classes trigger pandas FutureWarning about groupby behavior
- Should add `include_groups=False` to `.groupby().apply()`

**14. TreePoints Docstring Says "Camera Traps"**
- `TreePointsDataset` docstring: "RGB images from camera traps"
- These are aerial images, not camera traps (copy-paste from WILDS template)
- Same issue in `TreePolygonsDataset`

**15. No `image_size` Parameter on TreePoints**
- TreeBoxes and TreePolygons accept `image_size` parameter
- TreePoints hardcodes `448` in `_transform_()` (line 271)
- Inconsistent API across dataset types

### Suggestions for Better Usability

**16. Add a Quick-Start Training Script**
- Provide `docs/examples/train_boxes.py` that actually trains (not just evaluates)
- Include the eval loop at the end
- Show the complete target format with comments

**17. Document the Target Format Per Task**
- Boxes: `{"y": Tensor[N, 4], "labels": Tensor[N]}`
- Points: `{"y": Tensor[N, 2], "labels": Tensor[N]}`
- Polygons: `{"y": Tensor[N, H, W], "bboxes": Tensor[N, 4], "labels": Tensor[N]}`
- This is the single most important thing a new ML developer needs to know

**18. Add a `verify_installation()` Utility**
- A function that checks all dependencies are compatible
- Downloads one mini batch and prints shapes
- Would save users hours of debugging

**19. Document DeepForest Label Encoding**
- DeepForest uses `label_dict = {'Tree': 0}` with `num_classes=1`
- This is non-standard (usually 0=background) and causes CUDA asserts if wrong
- Worth a prominent note in the training docs

**20. Provide a `format_predictions()` Helper**
- Every example script has 40+ lines of prediction formatting code
- A utility function in `milliontrees.common.utils` would reduce boilerplate
- Something like: `milliontrees.utils.format_box_predictions(model_output) -> eval_dict`

---

## File Inventory

```
training/
  boxes/
    train_boxes.py          # DeepForest fine-tuning for box detection
  points/
    train_points.py         # DeepForest with pseudo-box supervision
  polygons/
    train_polygons.py       # Mask R-CNN for instance segmentation
  slurm/
    train_boxes.sbatch      # SLURM array job (3 splits)
    train_points.sbatch     # SLURM array job (3 splits)
    train_polygons.sbatch   # SLURM array job (3 splits)
    submit_all_training.sh  # Submit all jobs at once
  TRAINING_PLAN.md          # This document
```

---

## Next Steps (Pending Approval)

1. Fix the albumentations/deepforest dependency conflict in the environment
2. Run mini dataset training for all three tasks to validate scripts end-to-end
3. Run full dataset training on SLURM (via `submit_all_training.sh`)
4. Compare trained model results against the leaderboard baselines
5. If performance is unsatisfying with DeepForest, explore alternative architectures
   (e.g., DINO, Co-DETR for boxes; P2PNet for points; SAM-based for polygons)
