# Getting Started

## Installation

```
pip install MillionTrees
```

## Load the data

The aim of the package is to provide a single interface to load data directly into pytorch without needing to deal with the details of the data format. Users download the data and yield training and evaluation examples from the dataloaders.

```
from milliontrees.common.data_loaders import get_train_loader
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

# Download the data; this will take a while
dataset = TreeBoxesDataset(download=True)

train_dataset = dataset.get_subset("train")

# View the first image in the dataset
metadata, image, targets = train_dataset[0]
print(f"Metadata length: {len(metadata)}")
print(f"Image shape: {image.shape}, Image type: {type(image)}")
print(f"Targets keys: {targets.keys()}, Label type: {type(targets)}")
```

### Dataloaders

Datasets are batched into lists of target dictionaries, tensors of images, and tensors of metadata.
Each target dictionary contains tensors of the ground truth with the keys dict_keys
(['y', 'labels']). 'y' differs among the TreeGeometry datasets.

```
train_loader = get_train_loader("standard", train_dataset, batch_size=2)

# Show one batch of the loader
for metadata, image, targets in train_loader:
    print("Targets is a list of dictionaries with the following keys: ", targets[0].keys())
    print(f"Image shape: {image.shape}, Image type: {type(image)}")
    print(f"Annotation shape of the first image: {targets[0]['y'].shape}")
```

## Training models

We recommend using pytorch-lightning to train models for maximum reproducibility. Imagine a simple object detection model that predicts the bounding boxes of trees in an image. Of course users are welcome to use any other framework or model, but this is a simple example to get started.

```
# Create a simple PyTorch Lightning object detection model
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

class ObjectDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Use a pre-trained Faster R-CNN model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Modify the box predictor for our use case
        num_classes = 2  # Background + Tree
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        metadata, images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9)

# Initialize model and trainer
model = ObjectDetectionModel()
trainer = pl.Trainer(max_epochs=10, accelerator='auto')

# Create data loader
train_dataloader = get_train_loader("standard", train_dataset, batch_size=2)

trainer.fit(model, train_dataloader)
```

## Evaluate predictions

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
    all_y_pred.append(ObjectDetectionModel(images))

# Evaluate
dataset.eval(all_y_pred, all_y_true, all_metadata)
```

The evaluation dataset will return a dictionary of metrics for the given dataset and split.

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

