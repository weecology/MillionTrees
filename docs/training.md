# Training Models

MillionTrees is a dataset package designed to be general and flexible. We recommend using PyTorch Lightning to train models for maximum reproducibility. Below is a complete example using Faster R-CNN for bounding box detection. Users are welcome to use any framework or model.

## Batch Format

All MillionTrees dataloaders return batches as `(metadata, images, targets)`:

- **metadata**: `Tensor[B, 2]` — `[filename_id, source_id]` per image
- **images**: `Tensor[B, 3, H, W]` — batch of RGB images
- **targets**: `list[dict]` — one dict per image with keys:

| Task         | `"y"` shape      | `"labels"` shape | Description                        |
|--------------|-------------------|-------------------|------------------------------------|
| TreeBoxes    | `Tensor[N, 4]`    | `Tensor[N]`       | `[xmin, ymin, xmax, ymax]` boxes   |
| TreePoints   | `Tensor[N, 2]`    | `Tensor[N]`       | `[x, y]` point coordinates         |
| TreePolygons | `Tensor[N, H, W]` | `Tensor[N]`       | Binary masks per instance          |

## Data Setup

```python
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

# Use mini=True for development; remove for full training
dataset = TreeBoxesDataset(download=True, mini=True)

train_dataset = dataset.get_subset("train")
val_dataset = dataset.get_subset("test")

train_loader = get_train_loader("standard", train_dataset, batch_size=2)
val_loader = get_eval_loader("standard", val_dataset, batch_size=2)
```

## Model Definition

```python
import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class TreeDetector(pl.LightningModule):
    def __init__(self, num_classes=2, lr=5e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def training_step(self, batch, batch_idx):
        metadata, images, targets = batch
        formatted = [
            {"boxes": t["y"], "labels": torch.ones(len(t["y"]), dtype=torch.int64, device=images.device)}
            for t in targets
        ]
        loss_dict = self.model(images, formatted)
        loss = sum(loss_dict.values())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)

model = TreeDetector()
trainer = pl.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_loader)
```

**Note:** Torchvision detection models expect `labels` as integers starting from 1 (0 is background). The MillionTrees target dict uses `"y"` for coordinates and `"labels"` for class labels, so you need to reformat targets as shown above.

## Evaluation

After training, use the MillionTrees evaluation API to compute metrics:

```python
from milliontrees.common.eval import Evaluator

evaluator = dataset.eval  # built-in evaluator

model.eval()
all_predictions, all_targets = [], []
with torch.no_grad():
    for metadata, images, targets in val_loader:
        outputs = model.model(images)
        for output, target in zip(outputs, targets):
            pred = {
                "y": output["boxes"],
                "labels": output["labels"],
                "scores": output["scores"],
            }
            all_predictions.append(pred)
            all_targets.append(target)

results = evaluator(all_predictions, all_targets)
print(results)
```
