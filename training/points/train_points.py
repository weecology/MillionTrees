"""Train a point detection model on MillionTrees TreePoints.

Fine-tunes DeepForest on TreePoints by converting ground truth points to
pseudo-boxes for training, and converting predicted box centroids back to
points for evaluation.
"""

import argparse
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader

PSEUDO_BOX_HALF_SIZE = 15


def points_to_pseudo_boxes(points, image_size=448):
    """Convert (N, 2) point coordinates to (N, 4) pseudo bounding boxes."""
    if len(points) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    half = PSEUDO_BOX_HALF_SIZE
    xmin = (points[:, 0] - half).clamp(min=0)
    ymin = (points[:, 1] - half).clamp(min=0)
    xmax = (points[:, 0] + half).clamp(max=image_size)
    ymax = (points[:, 1] + half).clamp(max=image_size)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


class DeepForestPointTrainer(pl.LightningModule):

    def __init__(self, lr=1e-4, weight_decay=1e-5, image_size=448):
        super().__init__()
        self.save_hyperparameters()
        self.df_model = df_main.deepforest()
        self.df_model.load_model("weecology/deepforest-tree")
        self.retinanet = self.df_model.model

    def _prepare_targets(self, targets_list, device):
        rt = []
        for t in targets_list:
            points = t["y"]
            if points.dim() == 1:
                points = points.unsqueeze(0)
            boxes = points_to_pseudo_boxes(points, self.hparams.image_size).to(device)
            labels = torch.zeros(len(boxes), dtype=torch.int64, device=device)
            rt.append({"boxes": boxes, "labels": labels})
        return rt

    def training_step(self, batch, batch_idx):
        metadata, images, targets_list = batch
        rt = self._prepare_targets(targets_list, images.device)
        loss_dict = self.retinanet(images, rt)
        loss = sum(l for l in loss_dict.values())
        self.log("train_loss", loss, prog_bar=True, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        metadata, images, targets_list = batch
        rt = self._prepare_targets(targets_list, images.device)
        self.retinanet.train()
        loss_dict = self.retinanet(images, rt)
        self.retinanet.eval()
        loss = sum(l for l in loss_dict.values())
        self.log("val_loss", loss, prog_bar=True, batch_size=len(images), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.retinanet.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]


def predict_batch(model, images, batch_index):
    """Run inference and convert box predictions to point centroids."""
    warnings.filterwarnings("ignore")
    model.df_model.model = model.retinanet
    model.df_model.model.eval()
    images_tensor = images if isinstance(images, torch.Tensor) else torch.tensor(images)
    predictions = model.df_model.predict_step(images_tensor, batch_index)

    batch_y_pred = []
    for pred_df in predictions:
        if pred_df is None or len(pred_df) == 0:
            y_pred = {
                "y": torch.zeros((0, 2), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            }
        else:
            boxes = pred_df[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            centroids = np.stack([cx, cy], axis=1)
            y_pred = {
                "y": torch.tensor(centroids),
                "labels": torch.zeros(len(pred_df), dtype=torch.int64),
                "scores": torch.tensor(pred_df["score"].values.astype("float32")),
            }
        batch_y_pred.append(y_pred)
    return batch_y_pred


def evaluate(model, dataset, test_subset, batch_size=16):
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    all_y_pred, all_y_true = [], []
    model.eval()
    for batch_index, batch in enumerate(test_loader):
        metadata, images, targets = batch
        preds = predict_batch(model, images, batch_index)
        for y_pred, image_targets in zip(preds, targets):
            all_y_pred.append(y_pred)
            all_y_true.append(image_targets)
    results, results_str = dataset.eval(
        all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)]
    )
    return results, results_str


def main():
    parser = argparse.ArgumentParser(description="Train DeepForest on MillionTrees TreePoints")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/points/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--pseudo-box-size", type=int, default=30,
                        help="Full width/height of pseudo boxes around points")
    args = parser.parse_args()

    global PSEUDO_BOX_HALF_SIZE
    PSEUDO_BOX_HALF_SIZE = args.pseudo_box_size // 2

    os.makedirs(args.output_dir, exist_ok=True)

    point_dataset = get_dataset(
        "TreePoints", download=args.download, mini=args.mini,
        root_dir=args.root_dir, split_scheme=args.split_scheme,
    )

    train_subset = point_dataset.get_subset("train")
    test_subset = point_dataset.get_subset("test")

    train_loader = get_train_loader(
        "standard", train_subset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = get_eval_loader(
        "standard", test_subset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = DeepForestPointTrainer(lr=args.lr)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="points-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=3,
    )
    early_stop_cb = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator="auto", devices=args.gpus,
        callbacks=[checkpoint_cb, early_stop_cb],
        default_root_dir=args.output_dir, log_every_n_steps=10,
        val_check_interval=1.0,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n=== Evaluating best checkpoint ===")
    best_path = checkpoint_cb.best_model_path
    if best_path:
        print(f"Loading best checkpoint: {best_path}")
        model = DeepForestPointTrainer.load_from_checkpoint(best_path)

    results, results_str = evaluate(model, point_dataset, test_subset, batch_size=args.batch_size)
    print(results_str)

    results_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
    with open(results_path, "w") as f:
        f.write(results_str)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
