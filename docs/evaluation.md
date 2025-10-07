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
