# Evaluation

Once you have a model, evaluate it on the test split. The evaluation API expects per-image predictions in the same format as the ground truth targets: a dict with keys `y` (geometry tensor), `labels`, and `scores`.

## Expected Prediction Format

Each prediction is a dict with the following keys:

| Task         | Key        | Shape / Dtype                 | Description                                 |
|--------------|------------|-------------------------------|---------------------------------------------|
| TreeBoxes    | `"y"`      | `Tensor[N, 4]` float32        | Bounding boxes `[xmin, ymin, xmax, ymax]`   |
| TreePoints   | `"y"`      | `Tensor[N, 2]` float32        | Point coordinates `[x, y]`                  |
| TreePolygons | `"y"`      | `Tensor[N, H, W]` uint8       | Binary instance masks                       |
| All          | `"labels"` | `Tensor[N]` int64             | Class labels (typically all 0 for "tree")   |
| All          | `"scores"` | `Tensor[N]` float32           | Confidence scores (required for predictions)|

```python
from milliontrees.common.data_loaders import get_eval_loader
import torch

# Use the test split for evaluation
test_dataset = dataset.get_subset("test")
test_loader = get_eval_loader("standard", test_dataset, batch_size=16)

all_y_pred = []  # list[dict]
all_y_true = []  # list[dict]

for metadata, images, targets in test_loader:
    # Run your model to produce predictions for this batch
    batch_preds = MyModel(images)

    # Accumulate per-image predictions and targets
    for pred, target in zip(batch_preds, targets):
        all_y_pred.append(pred)
        all_y_true.append(target)

# Evaluate. Pass the metadata array from the same subset used for evaluation
results, results_str = dataset.eval(all_y_pred, all_y_true, metadata=test_dataset.metadata_array)
print(results_str)
```

The evaluation returns a dictionary of metrics and a formatted string with per-source breakdowns and averages.

## Evaluation visualizations

For qualitative debugging, pass **`viz_dir`** and optionally **`viz_n_per_source`** (default `4`) to `eval()`. The library writes PNGs under `viz_dir`, grouped in subfolders by source name, with up to `viz_n_per_source` images per source (in dataloader order).

- **Purple**: ground-truth geometry (boxes, points, or mask fill).
- **Orange**: predictions with `scores` above the dataset’s eval score threshold (same rule as metrics).

Images are resized to the dataset’s eval `image_size` so coordinates match `y_pred` and `y_true` from `get_eval_loader`.

When `viz_dir` is set, **`results["eval_visualization_paths"]`** lists the written PNG paths (strings).

```python
results, results_str = dataset.eval(
    all_y_pred,
    all_y_true,
    metadata=test_dataset.metadata_array[: len(all_y_pred)],
    viz_dir="work/eval_viz",
    viz_n_per_source=3,
)
```

### Example overlay (TreeBoxes + DeepForest)

Below: one TreeBoxes test image from the onboarding split, ground-truth boxes in purple, and boxes from the pretrained **`weecology/deepforest-tree`** model in orange (same resize as eval).

![TreeBoxes evaluation overlay: purple ground truth, orange DeepForest predictions](public/eval_visualization_sample.png)

To regenerate this image after changing overlay styles or the sample batch, install dev extras and run:

```bash
uv sync --dev
uv run python docs/scripts/generate_eval_viz_sample.py --root-dir onboarding_data
```

The script uses **`include_unsupervised=True`** so a local tree layout such as `onboarding_data/TreeBoxes_v0.12/` is found (the default supervised-only layout uses a different directory name). Use **`--mini --download`** only if you rely on the mini zip URL instead.

## External Model Adapter Example (Segmentation / DeepTrees-style)

If your model outputs instance masks (for example, a segmentation model such as DeepTrees), use the adapter example:

`docs/examples/external_segmentation_adapter.py`

Run a full smoke test without any external dependency:

```bash
python docs/examples/external_segmentation_adapter.py \
  --mini --download --mock --root-dir onboarding_data
```

Then replace `run_external_model_batch(images)` in the script with your model call.  
The adapter function `adapt_segmentation_prediction(...)` handles conversion to MillionTrees format:

- `masks` -> `y` (`Tensor[N, H, W]`, uint8)
- optional `scores` -> `scores` (`Tensor[N]`, float32, defaults to ones)
- optional `labels` -> `labels` (`Tensor[N]`, int64, defaults to zeros)

## TreeBoxes

The following can be recreated in the git repo: https://github.com/weecology/MillionTrees/blob/main/docs/examples/baseline_boxes.py

### Accuracy

First are the detection accuracy results. Detection accuracy measures the proportion of ground truth objects that are correctly detected (matched to a prediction above the IoU threshold), averaged across images.

$$
\text{Detection Accuracy} = \frac{1}{|I|} \sum_{i \in I} \frac{\text{matched}_i}{\text{total\_gt}_i}
$$

where $|I|$ is the number of images, $\text{matched}_i$ is the number of ground truth objects in image $i$ that have a matching prediction (above the IoU threshold), and $\text{total\_gt}_i$ is the total number of ground truth objects in image $i$.
```
============================================================
ACCURACY RESULTS
============================================================

Source-wise Results:
--------------------------------------------------
Source                    Score      Count     
--------------------------------------------------
NEON_benchmark            0.576      0         
Radogoshi et al. 2021     0.469      0         
Kwon et al. 2023          0.426      0         
SelvaBox                  0.320      0         
Weecology_University_Florida 0.313      0         
Velasquez-Camacho et al. 2023 0.227      0         
Zamboni et al. 2021       0.210      0         
Dumortier et al. 2025     0.154      0         
Reiersen et al. 2022      0.114      0         
Sun et al. 2022           0.097      0         
OAM-TCD                   0.059      0         
World Resources Institute 0.057      0         
Santos et al. 2019        0.001      0         

Summary Statistics:
----------------------------------------
Average accuracy: 0.211
Worst-group accuracy: 0.001
Min accuracy: 0.001
Max accuracy: 0.576
Std accuracy: 0.171
```

### Recall

Recall is proportion of correctly predicted true positives.

```
============================================================
RECALL RESULTS
============================================================

Source-wise Results:
--------------------------------------------------
Source                    Score      Count     
--------------------------------------------------
NEON_benchmark            0.779      0         
Radogoshi et al. 2021     0.594      0         
Kwon et al. 2023          0.506      0         
Zamboni et al. 2021       0.484      0         
Velasquez-Camacho et al. 2023 0.483      0         
Weecology_University_Florida 0.444      0         
SelvaBox                  0.426      0         
Dumortier et al. 2025     0.228      0         
Reiersen et al. 2022      0.183      0         
World Resources Institute 0.108      0         
Sun et al. 2022           0.104      0         
OAM-TCD                   0.103      0         
Santos et al. 2019        0.016      0         

Summary Statistics:
----------------------------------------
Average recall: 0.311
Worst-group recall: 0.016
Min recall: 0.016
Max recall: 0.779
Std recall: 0.224
Average detection_acc across source: nan
Average detection_accuracy: 0.211
  source_id = 0  [n =     94]:  detection_accuracy = 0.154
  source_id = 1  [n =      5]:  detection_accuracy = 0.426
  source_id = 2  [n =     33]:  detection_accuracy = 0.576
  source_id = 3  [n =    784]:  detection_accuracy = 0.059
  source_id = 4  [n =    286]:  detection_accuracy = 0.469
  source_id = 5  [n =     12]:  detection_accuracy = 0.114
  source_id = 6  [n =     95]:  detection_accuracy = 0.001
  source_id = 7  [n =    253]:  detection_accuracy = 0.320
  source_id = 8  [n =     27]:  detection_accuracy = 0.097
  source_id = 9  [n =    176]:  detection_accuracy = 0.227
  source_id = 10  [n =    461]: detection_accuracy = 0.313
  source_id = 11  [n =     94]: detection_accuracy = 0.057
  source_id = 12  [n =     42]: detection_accuracy = 0.210
Worst-group detection_accuracy: 0.001
Average detection_recall: 0.311
  source_id = 0  [n =     94]:  detection_recall = 0.228
  source_id = 1  [n =      5]:  detection_recall = 0.506
  source_id = 2  [n =     33]:  detection_recall = 0.779
  source_id = 3  [n =    784]:  detection_recall = 0.103
  source_id = 4  [n =    286]:  detection_recall = 0.594
  source_id = 5  [n =     12]:  detection_recall = 0.183
  source_id = 6  [n =     95]:  detection_recall = 0.016
  source_id = 7  [n =    253]:  detection_recall = 0.426
  source_id = 8  [n =     27]:  detection_recall = 0.104
  source_id = 9  [n =    176]:  detection_recall = 0.483
  source_id = 10  [n =    461]: detection_recall = 0.444
  source_id = 11  [n =     94]: detection_recall = 0.108
  source_id = 12  [n =     42]: detection_recall = 0.484
Worst-group detection_recall: 0.016
```

To see more examples for formatted and output of models, see examples/ in the git repo. 
