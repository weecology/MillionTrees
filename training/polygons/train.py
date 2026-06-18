"""Train a Mask R-CNN model on the MillionTrees TreePolygons dataset.

Strategy: Fine-tune a torchvision Mask R-CNN (ResNet-50-FPN backbone) on
TreePolygons. The MillionTrees polygon dataset provides per-instance binary
masks and bounding boxes. Mask R-CNN is the natural fit since it jointly
predicts boxes + instance masks.

DeepForest only produces boxes, not masks, so we use torchvision directly.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader
from milliontrees.datasets.polygon_stream_eval import (
    TreePolygonsStreamingEvalState,
    merge_viz_samples,
)
from milliontrees.common.metrics.all_metrics import (
    compute_polygon_mask_elementwise_batch,
)


def get_mask_rcnn(num_classes=2, init_mode="coco"):
    """Build a Mask R-CNN with configurable initialization (2 classes: bg + tree)."""
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


class MaskRCNNPolygonTrainer(pl.LightningModule):

    def __init__(self, lr=1e-4, weight_decay=1e-5, init_mode="coco",
                 eval_metrics=None):
        super().__init__()
        # eval_metrics holds live dataset metric objects (not serializable and
        # only needed during training-time validation), so keep it off the
        # saved hyperparameters / checkpoint.
        self.save_hyperparameters(ignore=["eval_metrics"])
        self.model = get_mask_rcnn(num_classes=2, init_mode=init_mode)
        self._eval_metrics = eval_metrics
        self._val_acc_sum = 0.0
        self._val_acc_n = 0

    def _prepare_targets(self, targets_list, device):
        rt = []
        for t in targets_list:
            masks = t["y"]
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)
            # GT masks are rasterized with foreground value 255 (see
            # TreePolygons.create_polygon_mask). Mask R-CNN's mask loss is
            # binary_cross_entropy_with_logits against these as targets, which
            # requires {0, 1}; passing 255 makes the -x*target term explode and
            # drives the loss arbitrarily negative. Binarize at this boundary.
            masks = (masks.to(device) > 0).float()

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
        # Break out the per-component losses so we can see whether the model is
        # learning at all (a flat total can hide a classifier that learns while
        # box/mask regression stalls, or vice versa).
        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, batch_size=len(images))
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_acc_sum = 0.0
        self._val_acc_n = 0

    def validation_step(self, batch, batch_idx):
        metadata, images, targets_list = batch
        rt = self._prepare_targets(targets_list, images.device)
        self.model.train()
        loss_dict = self.model(images, rt)
        self.model.eval()
        loss = sum(l for l in loss_dict.values())
        self.log("val_loss", loss, prog_bar=True, batch_size=len(images), sync_dist=True)
        for name, value in loss_dict.items():
            self.log(f"val_{name}", value, batch_size=len(images), sync_dist=True)

        # Track the actual eval metric (mask_acc) so we can select checkpoints on
        # it instead of the noisy summed val_loss surrogate. Requires an extra
        # eval-mode forward pass to get scored predictions.
        if self._eval_metrics is not None:
            preds = format_predictions_for_eval(images, self, images.device)
            # format_predictions_for_eval returns CPU preds; Lightning has moved
            # the batch targets to GPU. The metric does masks_to_boxes on both, so
            # they must share a device — mirror the CPU test-eval path.
            targets_cpu = [
                {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                 for k, v in t.items()}
                for t in targets_list
            ]
            ew = compute_polygon_mask_elementwise_batch(
                preds,
                targets_cpu,
                accuracy_metric=self._eval_metrics["accuracy"],
                recall_metric=self._eval_metrics["recall"],
                maskaware_metric=self._eval_metrics["maskaware_precision"],
                merge_metric=self._eval_metrics["merge_commission"],
            )
            acc = ew["accuracy"].float()
            self._val_acc_sum += float(acc.sum().item())
            self._val_acc_n += int(acc.numel())

    def on_validation_epoch_end(self):
        if self._eval_metrics is None:
            return
        mask_acc = (self._val_acc_sum / self._val_acc_n) if self._val_acc_n else 0.0
        self.log("val_mask_acc", mask_acc, prog_bar=True, sync_dist=True)

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


def _extract_box_pretrained_backbone(checkpoint_path):
    """Load a DeepForest checkpoint; return backbone state_dict keys for Mask R-CNN."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    prefix = "model.backbone.body."
    mapped = {}
    for key, value in state.items():
        if key.startswith(prefix):
            mapped[f"backbone.body.{key[len(prefix):]}"] = value
    if not mapped:
        raise ValueError(
            f"No DeepForest backbone keys found in {checkpoint_path}. "
            "Expected prefix 'model.backbone.body.'."
        )
    return mapped


def apply_box_pretrained_backbone(lightning_model, checkpoint_path):
    """Apply ResNet backbone weights from a box checkpoint to Mask R-CNN."""
    backbone_state = _extract_box_pretrained_backbone(checkpoint_path)
    missing, unexpected = lightning_model.model.load_state_dict(
        backbone_state,
        strict=False,
    )
    if unexpected:
        raise ValueError(f"Unexpected keys when loading box backbone: {unexpected}")
    loaded_prefix = "backbone.body."
    loaded_count = sum(1 for k in backbone_state if k.startswith(loaded_prefix))
    if loaded_count == 0:
        raise ValueError("No backbone.body keys loaded from box pretrained checkpoint.")
    return {"missing": missing, "loaded_backbone_keys": loaded_count}


def flatten_numeric_metrics(results):
    flat = {}
    for k, v in results.items():
        if isinstance(v, (int, float)):
            flat[k] = float(v)
        elif isinstance(v, torch.Tensor) and v.ndim == 0:
            flat[k] = float(v.item())
    return flat


def write_run_metadata(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def evaluate(
    model,
    dataset,
    test_subset,
    batch_size=8,
    device="cuda",
    viz_dir=None,
    *,
    eval_mode="stream",
    viz_n_per_source=4,
):
    """Run test-set evaluation.

    ``eval_mode``:
        - ``stream`` (default): update metrics per batch; does not accumulate all
          masks in Python lists (lower peak memory).
        - ``legacy``: accumulate full ``y_pred`` / ``y_true`` lists then call
          ``dataset.eval()`` once (previous behavior).
    """
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    model.eval()

    if eval_mode == "legacy":
        all_y_pred, all_y_true = [], []
        for batch in test_loader:
            metadata, images, targets = batch
            preds = format_predictions_for_eval(images, model, device)
            for y_pred, image_targets in zip(preds, targets):
                all_y_pred.append(y_pred)
                all_y_true.append(image_targets)
        return dataset.eval(
            all_y_pred,
            all_y_true,
            test_subset.metadata_array[: len(all_y_true)],
            viz_dir=viz_dir,
            viz_n_per_source=viz_n_per_source,
        )

    if eval_mode != "stream":
        raise ValueError(f"Unknown eval_mode: {eval_mode!r}; use 'stream' or 'legacy'.")

    state = TreePolygonsStreamingEvalState(dataset)
    viz_cap: dict[int, int] = {}
    viz_y_pred, viz_y_true, viz_rows = [], [], []
    for batch in test_loader:
        metadata, images, targets = batch
        preds = format_predictions_for_eval(images, model, device)
        state.update(preds, targets, metadata)
        if viz_dir is not None:
            merge_viz_samples(
                viz_cap,
                metadata,
                preds,
                targets,
                viz_y_pred=viz_y_pred,
                viz_y_true=viz_y_true,
                viz_rows=viz_rows,
                n_per_source=viz_n_per_source,
            )

    viz_meta = torch.stack(viz_rows, dim=0) if viz_rows else None
    return state.finalize(
        viz_dir=viz_dir,
        viz_y_pred=viz_y_pred or None,
        viz_y_true=viz_y_true or None,
        viz_metadata=viz_meta,
        viz_n_per_source=viz_n_per_source,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on MillionTrees TreePolygons")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
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
        default=None,
        help="Number of training batches per epoch (omit for full dataset)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Number of validation batches per epoch (omit for full dataset)",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="stream",
        choices=["stream", "legacy"],
        help="Test eval: 'stream' avoids holding the full test set in memory; 'legacy' matches old behavior.",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="coco",
        choices=["coco", "box_pretrained"],
        help="How to initialize Mask R-CNN before polygon training.",
    )
    parser.add_argument(
        "--box-backbone-checkpoint",
        type=str,
        default=None,
        help="Path to DeepForest/box checkpoint when --init-mode=box_pretrained.",
    )
    parser.add_argument(
        "--include-unsupervised",
        action="store_true",
        help="Include unsupervised rows in TreePolygons (full zip URLs).",
    )
    parser.add_argument(
        "--data-scope",
        type=str,
        default="subset",
        choices=["subset", "full"],
        help="Tag for experiment aggregation (subset vs full data pull).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-overfit",
        action="store_true",
        help="Sanity check: make the val/eval set identical to the train set, "
             "disable checkpointing + early stopping, and skip writing leaderboard "
             "results. Answers 'is the model learning and is eval wired up?' — on "
             "identical data, loss should fall and mask_acc/AP50 should climb high.",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable EarlyStopping on val_loss. Use --no-early-stopping "
             "to train the full --max-epochs (e.g. for LR sweeps).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="EarlyStopping patience (val checks) when --early-stopping is set. "
             "With once-per-epoch validation this equals epochs of no improvement.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply train-time augmentation (flips, 90-deg rotations, mild "
             "brightness/contrast). Disabled automatically under --debug-overfit.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    polygon_dataset = get_dataset(
        "TreePolygons",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
        include_unsupervised=args.include_unsupervised,
    )

    # Augmentation is train-only: build the train subset with the augmenting
    # transform, leave test on the deterministic resize. Skip it under
    # --debug-overfit so the train==val sanity check stays truly identical.
    use_augment = args.augment and not args.debug_overfit
    train_transform = polygon_dataset._train_transform_() if use_augment else None
    train_subset = polygon_dataset.get_subset("train", transform=train_transform)
    test_subset = polygon_dataset.get_subset("test")
    print(f"[augment] train-time augmentation: {'on' if use_augment else 'off'}")

    if len(train_subset) == 0:
        print("No training samples for this split; skipping training.")
        return

    if args.debug_overfit:
        # Sanity check: train and evaluate on the *same* rows. The only loader
        # difference is shuffle (transforms are a deterministic resize), so this
        # removes any generalization gap. A healthy model + eval pipeline must
        # drive loss down and push mask_acc/AP50 high on this set.
        print(
            "[debug-overfit] val/eval set := train set "
            f"({len(train_subset)} rows); checkpointing + early stopping disabled; "
            "leaderboard results will NOT be written."
        )
        test_subset = train_subset

    train_loader = get_train_loader(
        "standard", train_subset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = get_eval_loader(
        "standard", test_subset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    model = MaskRCNNPolygonTrainer(
        lr=args.lr, init_mode=args.init_mode, eval_metrics=polygon_dataset.metrics
    )
    init_details = {"init_mode": args.init_mode}
    if args.init_mode == "box_pretrained":
        if args.box_backbone_checkpoint is None:
            raise ValueError(
                "--box-backbone-checkpoint is required for --init-mode=box_pretrained"
            )
        details = apply_box_pretrained_backbone(model, args.box_backbone_checkpoint)
        init_details.update({
            "box_backbone_checkpoint": str(Path(args.box_backbone_checkpoint).resolve()),
            "loaded_backbone_keys": details["loaded_backbone_keys"],
        })

    has_val = len(val_loader) > 0

    # Select + early-stop on the actual eval metric (mask_acc), not the noisy
    # summed val_loss surrogate, which rises from classifier overfitting while
    # detection quality is still flat/improving.
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="polygons-{epoch:02d}-{val_mask_acc:.4f}",
        monitor="val_mask_acc",
        mode="max",
        save_top_k=3,
    )
    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor="val_mask_acc", patience=args.patience, mode="max"
    )

    loggers = []
    if args.comet:
        try:
            from pytorch_lightning.loggers import CometLogger
            loggers.append(CometLogger(
                project_name="milliontrees-polygons",
                tags=[
                    f"split-{args.split_scheme}",
                    "geometry-polygons",
                    f"lr-{args.lr:g}",
                    f"init-{args.init_mode}",
                ] + (["debug-overfit"] if args.debug_overfit else []),
            ))
        except Exception as e:
            print(f"Comet ML logging disabled: {e}")

    enable_ckpt = has_val and not args.debug_overfit
    callbacks = []
    if enable_ckpt:
        callbacks.append(checkpoint_cb)
        if args.early_stopping:
            callbacks.append(early_stop_cb)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.gpus,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        val_check_interval=1.0,
        logger=loggers if loggers else True,
        enable_checkpointing=enable_ckpt,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=(args.limit_val_batches if args.limit_val_batches is not None else 1.0) if has_val else 0,
    )

    trainer.fit(model, train_loader, val_loader)

    if args.debug_overfit:
        print("\\n=== [debug-overfit] Evaluating final model on the TRAIN set ===")
        best_path = None
    else:
        print("\\n=== Evaluating best checkpoint ===")
        best_path = checkpoint_cb.best_model_path if enable_ckpt else None
        if best_path:
            print(f"Loading best checkpoint: {best_path}")
            model = MaskRCNNPolygonTrainer.load_from_checkpoint(
                best_path,
                weights_only=False,
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results, results_str = evaluate(
        model,
        polygon_dataset,
        test_subset,
        batch_size=args.batch_size,
        device=device,
        viz_dir=os.path.join(args.output_dir, "viz"),
        eval_mode=args.eval_mode,
    )
    print(results_str)

    if args.debug_overfit:
        # Sanity run: print metrics (and Comet has the curves) but do not emit the
        # leaderboard-shaped JSON/results files.
        print("[debug-overfit] skipping leaderboard results/JSON writes.")
        return

    results_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
    with open(results_path, "w") as f:
        f.write(results_str)
    print(f"Results saved to {results_path}")

    metrics_flat = flatten_numeric_metrics(results)
    json_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.json")
    payload = {
        "model": "trained-polygons",
        "task": "TreePolygons",
        "split": args.split_scheme,
        "metrics": metrics_flat,
        "run_metadata": {
            "seed": args.seed,
            "data_scope": args.data_scope,
            "include_unsupervised": args.include_unsupervised,
            "eval_mode": args.eval_mode,
            "best_checkpoint_path": best_path if best_path else None,
            **init_details,
        },
    }
    write_run_metadata(json_path, payload)
    print(f"JSON results saved to {json_path}")


if __name__ == "__main__":
    main()
