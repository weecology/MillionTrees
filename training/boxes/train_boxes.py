"""Train a Faster R-CNN (DeepForest) on MillionTrees TreeBoxes.

Fine-tunes the pretrained DeepForest RetinaNet backbone on the
MillionTrees box training split. Evaluates using the MillionTrees eval API.
"""

import argparse
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader


class DeepForestBoxTrainer(pl.LightningModule):

    def __init__(self, lr=1e-4, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.df_model = df_main.deepforest()
        self.df_model.load_model("weecology/deepforest-tree")
        self.retinanet = self.df_model.model

    def _prepare_targets(self, targets_list, device):
        rt = []
        for t in targets_list:
            boxes = t["y"]
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels = torch.zeros(len(boxes), dtype=torch.int64, device=device)
            rt.append({"boxes": boxes.to(device), "labels": labels})
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
    """Run DeepForest inference, returns list of MillionTrees eval dicts."""
    warnings.filterwarnings("ignore")
    device = next(model.parameters()).device
    model.df_model.model = model.retinanet.to(device)
    model.df_model.model.eval()
    images_tensor = images if isinstance(images, torch.Tensor) else torch.tensor(images)
    images_tensor = images_tensor.to(device)
    predictions = model.df_model.predict_step(images_tensor, batch_index)

    batch_y_pred = []
    for pred_df in predictions:
        if pred_df is None or len(pred_df) == 0:
            y_pred = {
                "y": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            }
        else:
            y_pred = {
                "y": torch.tensor(pred_df[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")),
                "labels": torch.zeros(len(pred_df), dtype=torch.int64),
                "scores": torch.tensor(pred_df["score"].values.astype("float32")),
            }
        batch_y_pred.append(y_pred)
    return batch_y_pred


def evaluate(model, dataset, test_subset, batch_size=12):
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


def _flatten_results_for_comet(results, prefix="eval"):
    """Convert nested eval results dict into flat scalar dict for Comet metrics."""
    flat = {}
    for k, v in results.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    v2 = v2.item() if v2.numel() == 1 else float(v2.cpu().numpy())
                if isinstance(v2, (int, float, np.floating, np.integer)):
                    flat[f"{key}/{k2}"] = float(v2)
        else:
            if isinstance(v, torch.Tensor):
                v = v.item() if v.numel() == 1 else float(v.cpu().numpy())
            if isinstance(v, (int, float, np.floating, np.integer)):
                flat[key] = float(v)
    return flat


def _draw_boxes_on_image(image_tensor, gt_boxes, pred_boxes, pred_scores, score_threshold=0.1):
    """Overlay GT (green) and predicted (red) boxes on image. Returns PIL Image."""
    # image_tensor: (C, H, W) float 0-1; move to CPU for numpy
    t = image_tensor.cpu() if image_tensor.is_cuda else image_tensor
    img_np = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil, "RGBA")
    gt_boxes = gt_boxes.cpu() if gt_boxes.is_cuda else gt_boxes
    if gt_boxes.dim() == 1:
        gt_boxes = gt_boxes.unsqueeze(0)
    for b in gt_boxes:
        x1, y1, x2, y2 = b.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=2)
    pred_boxes = pred_boxes.cpu() if pred_boxes.is_cuda else pred_boxes
    pred_scores = pred_scores.cpu() if pred_scores.is_cuda else pred_scores
    if len(pred_boxes) > 0:
        pred_boxes = pred_boxes[pred_scores > score_threshold]
    if pred_boxes.dim() == 1 and len(pred_boxes) > 0:
        pred_boxes = pred_boxes.unsqueeze(0)
    for b in pred_boxes:
        x1, y1, x2, y2 = b.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)
    return pil


def log_validation_images_to_comet(experiment, model, test_subset, batch_size, num_images=6, score_threshold=0.1):
    """Run a few validation batches, overlay predictions and annotations, log images to Comet."""
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    model.eval()
    logged = 0
    for batch_index, batch in enumerate(test_loader):
        if logged >= num_images:
            break
        metadata, images, targets = batch
        preds = predict_batch(model, images, batch_index)
        for i in range(len(images)):
            if logged >= num_images:
                break
            pil = _draw_boxes_on_image(
                images[i],
                targets[i]["y"],
                preds[i]["y"],
                preds[i]["scores"],
                score_threshold=score_threshold,
            )
            experiment.log_image(pil, name=f"val_pred_{logged}")
            logged += 1


def main():
    parser = argparse.ArgumentParser(description="Train DeepForest on MillionTrees TreeBoxes")
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
    parser.add_argument("--output-dir", type=str, default="training/boxes/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Number of training batches per epoch (default: None = full epoch)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Number of validation batches per epoch (default: None = full epoch)",
    )
    parser.add_argument("--comet", action="store_true", help="Log to Comet ML (requires .comet.config or COMET_API_KEY)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    box_dataset = get_dataset(
        "TreeBoxes",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
    )

    train_subset = box_dataset.get_subset("train")
    test_subset = box_dataset.get_subset("test")

    if len(train_subset) == 0:
        print("No training samples for this split; skipping training.")
        return

    train_loader = get_train_loader(
        "standard", train_subset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = get_eval_loader(
        "standard", test_subset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = DeepForestBoxTrainer(lr=args.lr)

    has_val = len(val_loader) > 0

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="boxes-{epoch:02d}-{val_loss:.4f}",
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
                project_name="milliontrees-boxes",
                tags=[f"split-{args.split_scheme}", "geometry-boxes"],
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
        val_check_interval=1.0,
        logger=loggers if loggers else True,
        enable_checkpointing=has_val,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches if has_val else 0,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n=== Evaluating best checkpoint ===")
    best_path = checkpoint_cb.best_model_path if has_val else ""
    if best_path and os.path.isfile(best_path):
        print(f"Loading best checkpoint: {best_path}")
        model = DeepForestBoxTrainer.load_from_checkpoint(best_path)

    try:
        results, results_str = evaluate(model, box_dataset, test_subset, batch_size=args.batch_size)
        print(results_str)
        results_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
        with open(results_path, "w") as f:
            f.write(results_str)
        print(f"Results saved to {results_path}")

        if args.comet and loggers:
            from pytorch_lightning.loggers import CometLogger
            for logger in loggers:
                if isinstance(logger, CometLogger):
                    experiment = logger.experiment
                    flat = _flatten_results_for_comet(results, prefix="eval")
                    experiment.log_metrics(flat)
                    log_validation_images_to_comet(
                        experiment,
                        model,
                        test_subset,
                        batch_size=args.batch_size,
                        num_images=6,
                        score_threshold=box_dataset.eval_score_threshold,
                    )
                    break
    except Exception as e:
        print(f"Evaluation failed: {e}", flush=True)
        raise


if __name__ == "__main__":
    main()
