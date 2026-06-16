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
SAM3, detectree2, and CanopyRS DINO + SAM3 (SelvaMask).

### Random

| Model | Fine-tuned | Avg Mask Accuracy | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| Mask R-CNN | ✓ | 0.416 | 0.900 | <small>`uv run python training/polygons/train.py --split-scheme random`</small> |
| CanopyRS DINO + SAM3 (SelvaMask) | ✗ | 0.312 | 0.879 | <small>`python existing_models/canopyrs/eval_polygons.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |
| detectree2 | ✗ | 0.304 | 0.891 | <small>`uv run python existing_models/detectree2/eval_polygons.py --split-scheme random`</small> |
| SAM3 | ✗ | 0.186 | 0.619 | <small>`uv run python existing_models/sam3/eval_polygons.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |

### Zero-shot

| Model | Fine-tuned | Avg Mask Accuracy | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| detectree2 | ✗ | 0.375 | 0.945 | <small>`uv run python existing_models/detectree2/eval_polygons.py --split-scheme zeroshot`</small> |
| CanopyRS DINO + SAM3 (SelvaMask) | ✗ | 0.355 | 0.912 | <small>`python existing_models/canopyrs/eval_polygons.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |
| SAM3 | ✗ | 0.165 | 0.663 | <small>`uv run python existing_models/sam3/eval_polygons.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |
| Mask R-CNN | ✓ | 0.064 | 0.814 | <small>`uv run python training/polygons/train.py --split-scheme zeroshot`</small> |

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
|---|---|---|---|
| CanopyRS-DINO-SwinL | 0.177 | 0.841 | 169.941 |
| DeepForest-finetuned | 0.327 | 0.591 | 32.844 |
| DeepForest-pretrained | 0.261 | 0.399 | 28.399 |
| SAM3 | 0.239 | 0.635 | 53.283 |

### TreePoints

| Model | KeypointAccuracy | CountingMAE |
|---|---|---|
| SAM3 | 0.218 | 38.336 |
| TreeFormer-finetuned | 0.411 | 65.230 |
| TreeFormer-pretrained | 0.318 | 52.938 |

### TreePolygons

| Model | MaskAccuracy | MaskRecall | AP50 |
|---|---|---|---|
| CanopyRS-DINO-SAM3-SelvaMask | 0.312 | 0.565 | 0.220 |
| Detectree2 | 0.215 | 0.378 | 0.106 |
| MaskRCNN-finetuned | 0.375 | 0.604 | 0.284 |
| SAM3 | 0.154 | 0.235 | 0.045 |

### Split: zeroshot

### TreeBoxes

| Model | DetectionAccuracy | DetectionRecall | CountingMAE |
|---|---|---|---|
| CanopyRS-DINO-SwinL | 0.214 | 0.901 | 199.248 |
| DeepForest-finetuned | 0.341 | 0.534 | 34.757 |
| DeepForest-pretrained | 0.277 | 0.397 | 29.918 |
| SAM3 | 0.357 | 0.706 | 48.121 |

### TreePoints

| Model | KeypointAccuracy | CountingMAE |
|---|---|---|
| SAM3 | 0.243 | 52.164 |
| TreeFormer-finetuned | 0.460 | 34.506 |
| TreeFormer-pretrained | 0.392 | 18.627 |

### TreePolygons

| Model | MaskAccuracy | MaskRecall | AP50 |
|---|---|---|---|
| CanopyRS-DINO-SAM3-SelvaMask | 0.355 | 0.615 | 0.277 |
| Detectree2 | 0.247 | 0.354 | 0.142 |
| MaskRCNN-finetuned | 0.197 | 0.282 | 0.219 |
| SAM3 | 0.098 | 0.124 | 0.058 |

