# Evaluation

Once you have a model, evaluate it on the test split. The evaluation API expects per-image predictions in the same format as the ground truth targets: a dict with keys `y` (geometry tensor), `labels`, and `scores`.

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
<<<<<<< HEAD


## TreeBoxes

The following can be recreated in the git repo: https://github.com/weecology/MillionTrees/blob/main/docs/examples/baseline_boxes.py

### Accuracy

First are the accuracy results, accuracy is the number of correctly predicted ground truth averaged across images.

The accuracy metric is defined as:

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left(\hat{y}_i = y_i\right)
$$

where $N$ is the number of samples, $\hat{y}_i$ is the predicted label, $y_i$ is the true label, and $\mathbb{1}$ is the indicator function that returns 1 if the prediction is correct and 0 otherwise.
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
=======
>>>>>>> 3bd5c82 (OpenForestObservatory unsupervised data and improvements to point dataset)
