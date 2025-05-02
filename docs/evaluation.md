# Evaluation

Once we have trained a model, we can evaluate it on the test set.

```
from milliontrees.common.data_loaders import get_eval_loader
import torch

test_dataset = dataset.get_subset("train")
test_loader = get_eval_loader("standard", test_dataset, batch_size=16)

# Show one batch of the loader
all_y_pred = []
all_y_true = []
all_metadata = []

# For the sake of this example, we will make up some predictions to show format
for metadata, images, targets in test_loader:
    all_metadata.append(metadata)
    all_y_true.append(targets)
    all_y_pred.append(MyModel(images))

# Evaluate
dataset.eval(all_y_pred, all_y_true, all_metadata)
```

The evaluation dataset will return a dictionary of metrics for the given dataset and split.
