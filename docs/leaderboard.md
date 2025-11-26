# Leaderboard

# Tasks

There are three tasks within the MillionTrees package. 

## Zero-shot

The first task is to create a zero-shot detection system to generalize across geography and aquisition conditions. Selected datasets are held out from training completely and used for evaluation in new conditions. This is a challenging task with no local training data.

## Random

The second task is to create the best global detector for individual trees given a set of training and test data. Datasets are split randomly, reflecting information within localities. This is consistant with how most applied users engage with models, by fine-tuning backbone models with sample data from a desired locality.

## Cross-geometry

Off the shelf tools often limit users for a single annotation type. We have 'point' models, 'box' models and 'polygon' models. To create truly global models for biological inference, we need models that can use all available data, not just one annotation geometry. In particular, polygon annotations are very time consuming to create, but are often desirable for downstream usecases. We opted against polygon training sources, for example polygons to points, as this is an unrealistic, or atleast, very uncommon downstream use case. 


### Boxes to Polygons

All box sources are used to train and predict all polygon sources. There is no local data from the test localities in train.

### Points to Polygons

All point sources are used to train and predict all polygon sources

### Points to Boxes 

All point sources are used to train and predict all box sources.


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

3. Save visual examples of your model's predictions:
   ```python
   # Save a few example predictions
   dataset.visualize_predictions(
       predictions[:5], 
       save_dir="prediction_examples/"
   )
   ```

4. Submit a pull request to the [MillionTrees repository](https://github.com/weecology/MillionTrees) with:
   - Link to your code repository
   - Model description and approach
   - Performance metrics on test set
   - Example prediction visualizations
   - Instructions for reproducing results

## Mini dataset quick results

These runs use the mini datasets (one image per source) for fast validation.

| Model (script) | Task | Root dir | Key metrics |
|---|---|---|---|
| sam3_points.py (SAM3 native, GPU) | TreePoints | data-mini | KeypointAccuracy: 0.000; Counting MAE: 1164.000 |
| sam3_boxes.py (SAM3 native, GPU) | TreeBoxes | data-mini | Detection Acc: 0.083; Recall: 0.084 |
| baseline_points.py (DeepForest) | TreePoints | /orange/ewhite/web/public/MillionTrees | KeypointAccuracy: 0.000; Counting MAE: 104.250 |
| baseline_boxes.py (DeepForest) | TreeBoxes | /orange/ewhite/web/public/MillionTrees | Detection Acc: 0.559; Recall: 0.794 |
| sam3_polygons.py (SAM3) | TreePolygons | data-mini | Pending (reduce batch/frames if OOM) |
| torchvision_fasterrcnn_treeboxes.py | TreeBoxes | — | Pending |
| yolo_treeboxes.py | TreeBoxes | — | Pending |

Reproduce:
- Points (SAM3 native, GPU): `uv run python docs/examples/sam3_points.py --backend native --root-dir data-mini --mini --download --device cuda --batch-size 2 --num-workers 0 --max-batches 2`
- Boxes (SAM3 native, GPU): `uv run python docs/examples/sam3_boxes.py --backend native --root-dir data-mini --mini --download --device cuda --batch-size 2 --num-workers 0 --max-batches 2`
- Points/Boxes (DeepForest, shared root):  
  `uv run python docs/examples/baseline_points.py --root-dir /orange/ewhite/web/public/MillionTrees --mini --batch-size 4 --max-batches 2`  
  `uv run python docs/examples/baseline_boxes.py --root-dir /orange/ewhite/web/public/MillionTrees --mini --batch-size 4 --max-batches 2`

## Full dataset

Version: 0.8

| Model (script) | Task | Key metrics |
|---|---|---|
| sam3_points.py (SAM3) | TreePoints | KeypointAccuracy: 0.000; Counting MAE: 91.700 |
| sam3_boxes.py (SAM3) | TreeBoxes | Detection Acc: 0.037; Recall: 0.041 |
| baseline_points.py (DeepForest) | TreePoints | KeypointAccuracy: 0.000; Counting MAE: 104.250 |
| baseline_boxes.py (DeepForest) | TreeBoxes | Detection Acc: 0.559; Recall: 0.794 |
| sam3_polygons.py / baseline_polygons.py | TreePolygons | N/A on this root (missing 'polygon' column) |

## Generated results

| Model | Task | Split | Dataset | Size | Script |
|---|---|---|---|---|---|
| sam3_points.py | TreePoints | zeroshot | TreePoints | mini | `uv run python docs/examples/sam3_points.py --backend native --root-dir /orange/ewhite/web/public/MillionTrees --device cuda --max-batches 2 --mini --split-scheme zeroshot` |
| sam3_boxes.py | TreeBoxes | zeroshot | TreeBoxes | mini | `uv run python docs/examples/sam3_boxes.py --backend native --root-dir /orange/ewhite/web/public/MillionTrees --device cuda --max-batches 2 --mini --split-scheme zeroshot` |
| baseline_points.py | TreePoints | zeroshot | TreePoints | mini | `uv run python docs/examples/baseline_points.py --root-dir /orange/ewhite/web/public/MillionTrees --max-batches 2 --mini --split-scheme zeroshot` |
| baseline_boxes.py | TreeBoxes | zeroshot | TreeBoxes | mini | `uv run python docs/examples/baseline_boxes.py --root-dir /orange/ewhite/web/public/MillionTrees --max-batches 2 --mini --split-scheme zeroshot` |
