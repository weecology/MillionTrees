# Getting Started

## Installation

```
pip install MillionTrees
```

To be able to recreate the training examples, install the optional packages

```
MillionTrees[training]
```

## Load the data

```
from milliontrees.datasets.TreePoints import TreePointsDataset
dataset = TreePointsDataset(download=True, root_dir=<directory>) 
for image, label, metadata in dataset:
    image.shape == (3, 100, 100)
    label.shape == (2,)
    # Two fine-grained domain and a label of the coarse domain? This is still unclear see L82 of milliontrees_dataset.py
    assert len(metadata) == 2
    break
```
### Train a model

```
trainer.fit(model, train_dataloader)
```

## Evaluate predictions

```
from milliontrees.common.data_loaders import get_eval_loader

# Get the test set
test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

# Prepare the data loader
test_loader = get_eval_loader("standard", test_data, batch_size=16)

# Get predictions for the full test set
for x, y_true, metadata in test_loader:
    y_pred = model(x)
    # Accumulate y_true, y_pred, metadata

# Evaluate
dataset.eval(all_y_pred, all_y_true, all_metadata)
# {'recall_macro_all': 0.66, ...}
```

## Submit to the leaderboard

We accept submissions as .csv files 