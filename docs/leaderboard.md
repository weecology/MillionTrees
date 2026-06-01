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

## TreePoints

Fine-tuned (✓) rows train on the MillionTrees train split (`training/points/train.py`);
pretrained (✗) rows evaluate released weights on the test split (`existing_models/`).
The TreeFormer point model needs `uv sync --extra treeformer` (DeepForest
[`treeformer-training`](https://github.com/jveitchmichaelis/DeepForest/tree/treeformer-training)
branch until merged to main).

### Random

| Model | Fine-tuned | Counting MAE | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| TreeFormer | ✗ | pending | pending | <small>`uv run python existing_models/treeformer/eval_points.py --split-scheme random`</small> |
| SAM3 | ✗ | 26.675 | 0.714 | <small>`uv run python existing_models/sam3/eval_points.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |
| TreeFormer | ✓ | pending | pending | <small>`uv run --extra treeformer python training/points/train.py --split-scheme random`</small> |

### Zeroshot split

Fine-tuned (✓) rows train on the zeroshot **train** split (non-held-out sources)
and are scored on geographically held-out **test** sources. Pretrained (✗) rows use a
released checkpoint with no MillionTrees training at all.

| Model | Fine-tuned | Counting MAE | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| TreeFormer | ✗ | pending | pending | <small>`uv run python existing_models/treeformer/eval_points.py --split-scheme zeroshot`</small> |
| SAM3 | ✗ | 51.860 | 0.544 | <small>`uv run python existing_models/sam3/eval_points.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |
| TreeFormer | ✓ | pending | pending | <small>`uv run --extra treeformer python training/points/train.py --split-scheme zeroshot`</small> |

### Cross-geometry

> **Note:** Cross-geometry is designed for predicting polygons from other annotation geometries; it is not applicable to point prediction.

![TreePoints: model predictions by split](leaderboard_predictions_points.png)

## TreeBoxes

### Random

| Model | Fine-tuned | Avg Recall | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| DeepForest | ✓ | 0.721 | 0.610 | <small>`uv run python training/boxes/train.py --split-scheme random`</small> |
| DeepForest | ✗ | 0.414 | 0.760 | <small>`uv run python existing_models/deepforest/eval_boxes.py --split-scheme random`</small> |
| SAM3 | ✗ | 0.175 | 0.619 | <small>`uv run python existing_models/sam3/eval_boxes.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |

### Zero-shot

| Model | Fine-tuned | Avg Recall | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| DeepForest | ✓ | 0.460 | 0.900 | <small>`uv run python training/boxes/train.py --split-scheme zeroshot`</small> |
| DeepForest | ✗ | 0.416 | 0.959 | <small>`uv run python existing_models/deepforest/eval_boxes.py --split-scheme zeroshot`</small> |
| SAM3 | ✗ | 0.201 | 0.810 | <small>`uv run python existing_models/sam3/eval_boxes.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |

### Cross-geometry

> **Note:** Cross-geometry splits are designed for predicting polygons from other annotation geometries. The 0.000 scores below reflect that this split is not applicable to box prediction.

| Model | Fine-tuned | Avg Recall | Script |
|---|:---:|---|---|
| DeepForest | ✗ | 0.000 | <small>`uv run python existing_models/deepforest/eval_boxes.py --split-scheme crossgeometry`</small> |
| SAM3 | ✗ | 0.000 | <small>`uv run python existing_models/sam3/eval_boxes.py --device cuda --split-scheme crossgeometry --hf-token $HF_TOKEN`</small> |

![TreeBoxes: model predictions by split](leaderboard_predictions_boxes.png)

## TreePolygons

Fine-tuned (✓) uses Mask R-CNN (`training/polygons/train.py`). Pretrained (✗) uses SAM3.

### Random

| Model | Fine-tuned | Avg Mask Accuracy | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| Mask R-CNN | ✓ | 0.232 | 0.872 | <small>`uv run python training/polygons/train.py --split-scheme random`</small> |
| SAM3 | ✗ | 0.223 | 0.681 | <small>`uv run python existing_models/sam3/eval_polygons.py --device cuda --split-scheme random --hf-token $HF_TOKEN`</small> |

### Zero-shot

| Model | Fine-tuned | Avg Mask Accuracy | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| Mask R-CNN | ✓ | 0.146 | 0.758 | <small>`uv run python training/polygons/train.py --split-scheme zeroshot`</small> |
| SAM3 | ✗ | 0.180 | 0.719 | <small>`uv run python existing_models/sam3/eval_polygons.py --device cuda --split-scheme zeroshot --hf-token $HF_TOKEN`</small> |

### Cross-geometry

| Model | Fine-tuned | Avg Mask Accuracy | Mask-Aware Precision | Script |
|---|:---:|---|---|---|
| SAM3 | ✗ | pending | pending | <small>`uv run python existing_models/sam3/eval_polygons.py --device cuda --split-scheme crossgeometry --hf-token $HF_TOKEN`</small> |

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
