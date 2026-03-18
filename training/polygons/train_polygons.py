"""Train a Mask R-CNN model on the MillionTrees TreePolygons dataset.

Strategy: Fine-tune a torchvision Mask R-CNN (ResNet-50-FPN backbone) on
TreePolygons. The MillionTrees polygon dataset provides per-instance binary
masks and bounding boxes. Mask R-CNN is the natural fit since it jointly
predicts boxes + instance masks.

DeepForest only produces boxes, not masks, so we use torchvision directly.
"""

import argparse
import os
import warnings
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader


def get_mask_rcnn(num_classes=2):
    """Build a Mask R-CNN with pretrained backbone, 2 classes (bg + tree)."""
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


class MaskRCNNPolygonTrainer(pl.LightningModule):

    def __init__(self, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_mask_rcnn(num_classes=2)

    def _prepare_targets(self, targets_list, device):
        rt = []
        for t in targets_list:
            masks = t["y"]
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)
            masks = masks.to(device).float()

            if "bboxes" in t:
                boxes = t["bboxes"]
                if isinstance(boxes, np.ndarray):
                    boxes = torch.from_numpy(boxes)
                boxes = boxes.to(device).float()
            else:
                if len(masks) > 0:
                    from torchvision.ops import masks_to_boxes
                    boxes = masks_to_boxes(masks.byte())
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)

            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                masks = torch.zeros((0, 1, 1), dtype=torch.float32, device=device)

            labels = torch.ones(len(boxes), dtype=torch.int64, device=device)
            rt.append({"boxes": boxes, "labels": labels, "masks": masks})
        return rt

    def training_step(self, batch, batch_idx):
        metadata, images, targets_list = batch
        rt = self._prepare_targets(targets_list, images.device)
        loss_dict = self.model(images, rt)
        loss = sum(l for l in loss_dict.values())
        self.log("train_loss", loss, prog_bar=True, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        metadata, images, targets_list = batch
        rt = self._prepare_targets(targets_list, images.device)
        self.model.train()
        loss_dict = self.model(images, rt)
        self.model.eval()
        loss = sum(l for l in loss_dict.values())
        self.log("val_loss", loss, prog_bar=True, batch_size=len(images), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]


def format_predictions_for_eval(images, model, device, mask_threshold=0.5):
    """Run Mask R-CNN inference, return predictions in MillionTrees eval format."""
    model.model.eval()
    model.model.to(device)
    with torch.no_grad():
        outputs = model.model(images.to(device))

    batch_y_pred = []
    for output in outputs:
        masks = output.get("masks", torch.zeros((0, 1, 1, 1)))
        boxes = output.get("boxes", torch.zeros((0, 4)))
        scores = output.get("scores", torch.zeros((0,)))
        labels_out = output.get("labels", torch.zeros((0,), dtype=torch.int64))

        if len(masks) == 0:
            y_pred = {
                "y": torch.zeros((0, images.shape[2], images.shape[3]), dtype=torch.uint8),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            }
        else:
            binary_masks = (masks[:, 0] > mask_threshold).byte().cpu()
            y_pred = {
                "y": binary_masks,
                "labels": labels_out.cpu(),
                "scores": scores.cpu(),
            }
        batch_y_pred.append(y_pred)
    return batch_y_pred


def evaluate(model, dataset, test_subset, batch_size=8, device="cuda"):
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    all_y_pred, all_y_true = [], []
    model.eval()
    for batch in test_loader:
        metadata, images, targets = batch
        preds = format_predictions_for_eval(images, model, device)
        for y_pred, image_targets in zip(preds, targets):
            all_y_pred.append(y_pred)
            all_y_true.append(image_targets)
    results, results_str = dataset.eval(
        all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)]
    )
    return results, results_str


def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on MillionTrees TreePolygons")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/polygons/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--comet", action="store_true", help="Log to Comet ML (requires .comet.config or COMET_API_KEY)")
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=50,
        help="Number of training batches per epoch (for fast debugging)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=50,
        help="Number of validation batches per epoch (for fast debugging)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    polygon_dataset = get_dataset(
        "TreePolygons",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
    )

    train_subset = polygon_dataset.get_subset("train")
    test_subset = polygon_dataset.get_subset("test")

    if len(train_subset) == 0:
        print("No training samples for this split; skipping training.")
        return

    train_loader = get_train_loader(
        "standard", train_subset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = get_eval_loader(
        "standard", test_subset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = MaskRCNNPolygonTrainer(lr=args.lr)

    has_val = len(val_loader) > 0

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="polygons-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    early_stop_cb = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

    loggers = []
    if args.comet:
        try:
            from pytorch_lightning.loggers import CometLogger
            loggers.append(CometLogger(
                project_name="milliontrees-polygons",
                tags=[f"split-{args.split_scheme}", "geometry-polygons"],
            ))
        except Exception as e:
            print(f"Comet ML logging disabled: {e}")

    callbacks = [checkpoint_cb, early_stop_cb] if has_val else []

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.gpus,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        val_check_interval=0.5,
        logger=loggers if loggers else True,
        enable_checkpointing=has_val,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches if has_val else 0,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\\n=== Evaluating best checkpoint ===")
    best_path = checkpoint_cb.best_model_path
    if best_path:
        print(f"Loading best checkpoint: {best_path}")
        model = MaskRCNNPolygonTrainer.load_from_checkpoint(best_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results, results_str = evaluate(
        model, polygon_dataset, test_subset, batch_size=args.batch_size, device=device
    )
    print(results_str)

    results_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
    with open(results_path, "w") as f:
        f.write(results_str)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
