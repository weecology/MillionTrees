# Training models

MillionTrees is a dataset package, it is designed to be general and flexible. We recommend using pytorch-lightning to train models for maximum reproducibility. Imagine a simple object detection model that predicts the bounding boxes of trees in an image. Of course users are welcome to use any other framework or model, but this is a simple example to get started.

### Data setup

```python
from milliontrees.common.data_loaders import get_train_loader
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

# Download the data; this will take a while
dataset = TreeBoxesDataset(download=True)

train_dataset = dataset.get_subset("train")
train_loader = get_train_loader("standard", train_dataset, batch_size=2)
```

### Model Definition

```python
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
