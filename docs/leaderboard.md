# Leaderboard

# Tasks

## Zero-shot

The first task evaluates generalization across geography and acquisition conditions.
Selected source datasets are held out from the train split; models are fine-tuned on
the remaining sources and evaluated on the held-out test sources (no images from test
localities appear in train).

## Random

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

Fine-tuned panel figures (random + zeroshot splits, ground truth vs prediction) are
generated from training checkpoints via `scripts/create_finetuned_visualizations.py`
(PNG + SVG under `docs/`). See [repository_structure.md](repository_structure.md).

## TreePoints

Fine-tuned (✓) rows train on the MillionTrees train split (`training/points/train.py`);
pretrained (✗) rows evaluate released weights on the test split with no MillionTrees
training. The TreeFormer ✗ rows use the KCL TreeFormer HF checkpoint
(J. Veitch-Michaelis, `$KCL_CHECKPOINT` = `/home/veitchmichaelisj/code/DeepForest_jvm/kcl_hf_checkpoint`),
evaluated via `training/points/eval.py` (2026-06-10).
The TreeFormer point model needs `uv sync --group treeformer` (DeepForest
[`treeformer-training`](https://github.com/jveitchmichaelis/DeepForest/tree/treeformer-training)
branch until merged to main).

### Random

| Model | Fine-tuned | Counting MAE | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| TreeFormer | ✗ | 53.150 | 0.830 | <small>`uv run --group treeformer python training/points/eval.py --checkpoint $KCL_CHECKPOINT --split-scheme random`</small> |
| SAM3 | ✗ | 54.593 | 0.711 | <small>`uv run python existing_models/sam3/eval_points.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |
| TreeFormer | ✓ | 57.523 | 0.782 | <small>`uv run --group treeformer python training/points/train.py --split-scheme random`</small> |

### Zeroshot split

Fine-tuned (✓) rows train on the zeroshot **train** split (non-held-out sources)
and are scored on geographically held-out **test** sources. Pretrained (✗) rows use a
released checkpoint with no MillionTrees training at all.

| Model | Fine-tuned | Counting MAE | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| TreeFormer | ✗ | 16.451 | 0.789 | <small>`uv run --group treeformer python training/points/eval.py --checkpoint $KCL_CHECKPOINT --split-scheme zeroshot`</small> |
| SAM3 | ✗ | 14.878 | 0.759 | <small>`uv run python existing_models/sam3/eval_points.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |
| TreeFormer | ✓ | 17.007 | 0.872 | <small>`uv run --group treeformer python training/points/train.py --split-scheme zeroshot`</small> |

### Cross-geometry

> **Note:** Cross-geometry is designed for predicting polygons from other annotation geometries; it is not applicable to point prediction.

![TreePoints: model predictions by split](leaderboard_predictions_points.png)

## TreeBoxes

### Random

| Model | Fine-tuned | Avg Recall | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| CanopyRS DINO Swin-L | ✗ | 0.688 | 0.679 | <small>`python existing_models/canopyrs/eval_boxes.py --device cuda --split-scheme random`</small> |
| DeepForest | ✓ | 0.547 | 0.592 | <small>`uv run python training/boxes/train.py --split-scheme random`</small> |
| DeepForest | ✗ | 0.407 | 0.731 | <small>`uv run python existing_models/deepforest/eval_boxes.py --split-scheme random`</small> |
| SAM3 | ✗ | 0.190 | 0.608 | <small>`uv run python existing_models/sam3/eval_boxes.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |

### Zero-shot

| Model | Fine-tuned | Avg Recall | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| CanopyRS DINO Swin-L | ✗ | 0.885 | 0.815 | <small>`python existing_models/canopyrs/eval_boxes.py --device cuda --split-scheme zeroshot`</small> |
| DeepForest | ✓ | 0.525 | 0.947 | <small>`uv run python training/boxes/train.py --split-scheme zeroshot`</small> |
| DeepForest | ✗ | 0.432 | 0.962 | <small>`uv run python existing_models/deepforest/eval_boxes.py --split-scheme zeroshot`</small> |
| SAM3 | ✗ | 0.209 | 0.798 | <small>`uv run python existing_models/sam3/eval_boxes.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |

### Cross-geometry

> **Note:** Cross-geometry splits are designed for predicting polygons from other annotation geometries. The 0.000 scores below reflect that this split is not applicable to box prediction.

| Model | Fine-tuned | Avg Recall | Script |
|---|:---:|---|---|
| DeepForest | ✗ | 0.000 | <small>`uv run python existing_models/deepforest/eval_boxes.py --split-scheme crossgeometry`</small> |
| SAM3 | ✗ | 0.000 | <small>`uv run python existing_models/sam3/eval_boxes.py --device cuda --split-scheme crossgeometry --hf-token $HF_TOKEN`</small> |

![TreeBoxes: model predictions by split](leaderboard_predictions_boxes.png)

## TreePolygons

Fine-tuned (✓) uses Mask R-CNN (`training/polygons/train.py`). Pretrained (✗) uses
SAM3 and detectree2.

> **Note:** CanopyRS DINO + SAM3 (SelvaMask) polygon rows are marked "rerun pending".
> The 2026-06-10 random run scored ≈ 0 on every source except Khan_et_al._2026 (0.350),
> and no zeroshot polygon run was produced. Cause: the polygon evaluator filters every
> metric (recall/precision/AP50) at `eval_score_threshold = 0.5`, while TreeBoxes uses
> 0.1. The CanopyRS DINO detector finds the trees (see TreeBoxes above) but its
> confidences on the polygon imagery mostly fall below 0.5, so nearly all predictions
> are dropped at eval time. Re-run and log the detector score distribution before
> publishing; do **not** lower the polygon threshold, since detectree2/SAM3/Mask R-CNN
> are all scored at the same 0.5 floor.

### Random

| Model | Fine-tuned | Avg Mask Accuracy | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| Mask R-CNN | ✓ | 0.416 | 0.900 | <small>`uv run python training/polygons/train.py --split-scheme random`</small> |
| detectree2 | ✗ | 0.304 | 0.891 | <small>`uv run python existing_models/detectree2/eval_polygons.py --split-scheme random`</small> |
| SAM3 | ✗ | 0.186 | 0.619 | <small>`uv run python existing_models/sam3/eval_polygons.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |
| CanopyRS DINO + SAM3 (SelvaMask) | ✗ | rerun pending | rerun pending | <small>`python existing_models/canopyrs/eval_polygons.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |

### Zero-shot

| Model | Fine-tuned | Avg Mask Accuracy | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| detectree2 | ✗ | 0.375 | 0.945 | <small>`uv run python existing_models/detectree2/eval_polygons.py --split-scheme zeroshot`</small> |
| SAM3 | ✗ | 0.165 | 0.663 | <small>`uv run python existing_models/sam3/eval_polygons.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |
| Mask R-CNN | ✓ | 0.064 | 0.814 | <small>`uv run python training/polygons/train.py --split-scheme zeroshot`</small> |
| CanopyRS DINO + SAM3 (SelvaMask) | ✗ | rerun pending | rerun pending | <small>`python existing_models/canopyrs/eval_polygons.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |

> **Note:** The Mask R-CNN ✓ zeroshot row is from the pre-fix polygon evaluator
> (before GT-mask binarization / AP50; commit b2ff776) and is not directly comparable
> to the post-fix random row above. Rerun to refresh.

### Cross-geometry

| Model | Fine-tuned | Avg Mask Accuracy | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| TreeFormer+SAM2 | ✗ | 0.254 | 0.828 | <small>`uv run python existing_models/treeformer_sam2/eval_polygons_crossgeometry.py`</small> |
| SAM3 | ✗ | 0.165 | 0.663 | <small>`uv run python existing_models/sam3/eval_polygons.py --device cuda --split-scheme crossgeometry --hf-token $HF_TOKEN`</small> |

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

2. Generate predictions on the test split:
   ```python
   test_dataset = dataset.get_subset("test")  # Use test split
   test_loader = get_eval_loader("standard", test_dataset, batch_size=16)

   predictions = []
   for metadata, images, _ in test_loader:
       pred = model(images)
       predictions.append(pred)
   ```

3. Submit a pull request to the [MillionTrees repository](https://github.com/weecology/MillionTrees) with:
   - Link to your code repository
   - Model description and approach
   - Performance metrics on test set
   - Example prediction visualizations
   - Instructions for reproducing results

## Benchmark Results

Comparison of fine-tuned models (trained on MillionTrees) vs. pretrained models evaluated zero-shot.

### Split: random

### TreeBoxes

| Model | DetectionAccuracy | DetectionRecall | CountingMAE |
|---||---||---||---|
| CanopyRS-DINO-SwinL | 0.169 | 0.750 | 130.693 |
| DeepForest-finetuned | 0.269 | 0.547 | 27.142 |
| DeepForest-pretrained | 0.268 | 0.407 | 24.069 |
| SAM3 | 0.207 | 0.554 | 43.950 |

### TreePoints

| Model | KeypointAccuracy | CountingMAE |
|---||---||---|
| SAM3 | 0.182 | 54.593 |
| TreeFormer-finetuned | 0.278 | 57.444 |
| TreeFormer-pretrained | 0.239 | 56.519 |

### TreePolygons

| Model | MaskAccuracy | MaskRecall | AP50 |
|---||---||---||---|
| CanopyRS-DINO-SAM3-SelvaMask | 0.327 | 0.593 | 0.176 |
| Detectree2 | 0.277 | 0.465 | 0.163 |
| MaskRCNN-finetuned | 0.387 | 0.658 | 0.278 |
| SAM3 | 0.186 | 0.282 | 0.064 |

### Split: zeroshot

### TreeBoxes

| Model | DetectionAccuracy | DetectionRecall | CountingMAE |
|---||---||---||---|
| CanopyRS-DINO-SwinL | 0.220 | 0.919 | 203.784 |
| DeepForest-finetuned | 0.377 | 0.525 | 30.365 |
| DeepForest-pretrained | 0.302 | 0.432 | 28.741 |
| SAM3 | 0.379 | 0.732 | 47.854 |

### TreePoints

| Model | KeypointAccuracy | CountingMAE |
|---||---||---|
| SAM3 | 0.129 | 14.878 |
| TreeFormer-finetuned | 0.314 | 17.124 |
| TreeFormer-pretrained | 0.291 | 15.717 |

### TreePolygons

| Model | MaskAccuracy | MaskRecall | AP50 |
|---||---||---||---|
| CanopyRS-DINO-SAM3-SelvaMask | 0.380 | 0.635 | 0.298 |
| Detectree2 | 0.375 | 0.520 | 0.366 |
| MaskRCNN-finetuned | 0.194 | 0.408 | 0.117 |
| SAM3 | 0.165 | 0.214 | 0.131 |

