"""Train DeepForest (RetinaNet) on MillionTrees TreeBoxes.

Adapts MillionTrees dataloaders to the DeepForest training API so that
DeepForest's own LightningModule and Trainer do all the heavy lifting.
Custom code here is limited to two things that DeepForest doesn't cover:

  MillionTreesBatchAdapter  — translates batch format (metadata→path, y→boxes)
  evaluate()                — uses MillionTrees eval API (DeepForest's is CSV-based)
"""

import argparse
import glob
import math
import os
import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


# ---------------------------------------------------------------------------
# Batch format adapter
# ---------------------------------------------------------------------------

class _AdaptCollate:
    """collate_fn that emits DeepForest-format (images, targets, paths) batches.

    Wraps the MillionTrees subset collate, then translates batch format:
      MillionTrees: (metadata[B,2], images[B,C,H,W], [{"y": boxes, "labels": int64}])
      DeepForest:   (images[B,C,H,W], [{"boxes": boxes, "labels": int64}], list[str])

    Implemented as a top-level callable (not a closure) so it pickles cleanly to
    DataLoader workers. Crucially, returning a *real* DataLoader with this
    collate — instead of a hand-rolled iterable wrapper — lets Lightning inject a
    DistributedSampler under DDP, so each GPU trains on its own data shard.
    Images are already CHW float32 0-1 from MillionTrees, so no conversion needed.
    """

    def __init__(self, base_collate, filename_id_to_path=None):
        self.base_collate = base_collate
        self.filename_id_to_path = filename_id_to_path or {}

    def __call__(self, batch):
        metadata, images, targets = self.base_collate(batch)
        paths = [
            self.filename_id_to_path.get(int(metadata[i, 0]), str(int(metadata[i, 0])))
            for i in range(len(metadata))
        ]
        adapted = []
        for t in targets:
            boxes = t["y"]
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = t["labels"]
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.int64)
            else:
                labels = labels.long()
            adapted.append({"boxes": boxes.float(), "labels": labels})
        return images, adapted, paths


# ---------------------------------------------------------------------------
# Inference helper (MillionTrees eval API needs torchvision-style dicts)
# ---------------------------------------------------------------------------

def predict_batch(model, images):
    """Run DeepForest inference; returns MillionTrees-format prediction dicts."""
    warnings.filterwarnings("ignore")
    device = next(model.parameters()).device
    images = images.to(device) if isinstance(images, torch.Tensor) else torch.tensor(images).to(device)
    model.model.eval()
    with torch.no_grad():
        predictions = model.model(images)

    result = []
    for pred in predictions:
        boxes = pred.get("boxes", torch.zeros((0, 4)))
        if len(boxes) == 0:
            result.append({
                "y": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
        else:
            result.append({
                "y": boxes.detach().float().cpu(),
                "labels": pred["labels"].detach().cpu().long(),
                "scores": pred["scores"].detach().float().cpu(),
            })
    return result


# ---------------------------------------------------------------------------
# Evaluation (DeepForest's evaluate() is CSV-based; use MillionTrees API)
# ---------------------------------------------------------------------------

def collect_predictions(model, test_subset, batch_size=12, max_batches=None):
    """Run inference over test_subset, returning (all_y_pred, all_y_true).

    Factored out of evaluate() so a threshold sweep can reuse a single inference
    pass (eval_sweep.run_threshold_sweep) instead of re-running the model per
    threshold.
    """
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    all_y_pred, all_y_true = [], []
    for i, batch in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        _, images, targets = batch
        preds = predict_batch(model, images)
        all_y_pred.extend(preds)
        all_y_true.extend(targets)
    return all_y_pred, all_y_true


def evaluate(model, dataset, test_subset, batch_size=12, viz_dir=None, max_batches=None):
    all_y_pred, all_y_true = collect_predictions(
        model, test_subset, batch_size=batch_size, max_batches=max_batches)
    results, results_str = dataset.eval(
        all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)],
        viz_dir=viz_dir,
    )
    return results, results_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DeepForest on MillionTrees TreeBoxes")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for SGD (DeepForest default optimizer)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument(
        "--include-unsupervised",
        action="store_true",
        help="Use TreeBoxes_v* layout with full-zip URLs.",
    )
    parser.add_argument(
        "--remove-incomplete",
        action="store_true",
        help="Train only on complete=True (exhaustively annotated) sources. "
             "Filters the TRAIN split only; the test set is always left "
             "unchanged so results are comparable to the full-train baseline.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/boxes/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        help="Lightning accelerator (use 'cpu' for debugging).",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--comet", action="store_true",
                        help="Log to Comet ML (requires .comet.config or COMET_API_KEY)")
    parser.add_argument("--comet-name", type=str, default=None,
                        help="Comet experiment name. Defaults to "
                             "boxes-<split>-lr<lr> (see CLAUDE.md naming scheme).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Limit to 2 train/val batches and 1 epoch for local testing")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_epochs = 1
        args.early_stop_patience = 1

    os.makedirs(args.output_dir, exist_ok=True)

    box_dataset = get_dataset(
        "TreeBoxes",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        include_unsupervised=args.include_unsupervised,
        remove_incomplete=args.remove_incomplete,
    )

    train_subset = box_dataset.get_subset("train")
    test_subset = box_dataset.get_subset("test")

    if len(train_subset) == 0:
        print("No training samples for this split; skipping training.")
        return

    # Real DataLoaders (not a custom iterable) so Lightning can inject a
    # DistributedSampler under DDP and shard data across GPUs. The collate_fn
    # translates MillionTrees batches into DeepForest's (images, targets, paths).
    adapt_collate = _AdaptCollate(train_subset.collate, box_dataset._filename_id_to_code)
    has_val = len(test_subset) > 0

    train_adapted = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=adapt_collate,
        num_workers=args.num_workers,
    )
    val_adapted = (
        DataLoader(
            test_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=adapt_collate,
            num_workers=args.num_workers,
        )
        if has_val
        else None
    )

    # Build DeepForest model and load pretrained weights
    model = df_main.deepforest(
        config_args={
            "train": {
                "epochs": args.max_epochs,
                "lr": args.lr,
                "root_dir": str(box_dataset._data_dir / "images"),
            },
            "validation": {
                "root_dir": str(box_dataset._data_dir / "images"),
                # Compute box_precision/box_recall/mAP every epoch. The base
                # config defaults this to 20, so with ~20 epochs the metrics
                # would only ever log on the final epoch and you'd see nothing
                # but the losses. These metrics come from the val dataloader,
                # not a csv_file.
                "val_accuracy_interval": 1,
                },
            "batch_size": args.batch_size,
            "devices": args.gpus,
            "accelerator": args.accelerator,
            "workers": args.num_workers,
        },
        existing_train_dataloader=train_adapted,
        existing_val_dataloader=val_adapted,
    )
    model.load_model("weecology/deepforest-tree")

    # Loggers
    loggers = []
    if args.comet:
        try:
            import json
            from pytorch_lightning.loggers import CometLogger

            class _SafeCometLogger(CometLogger):
                """CometLogger that skips non-JSON-serializable hparams.

                DeepForest calls save_hyperparameters() which includes
                existing_train_dataloader (a MillionTreesBatchAdapter).
                Comet tries to serialize it, crashes its FallbackStreamer
                background thread, then the main thread blocks forever
                waiting on the dead queue. Filtering hparams here prevents
                that hang.
                """
                def log_hyperparams(self, params):
                    safe = {}
                    for k, v in params.items():
                        try:
                            json.dumps(v)
                            safe[k] = v
                        except (TypeError, ValueError):
                            safe[k] = type(v).__name__
                    super().log_hyperparams(safe)

            comet_name = args.comet_name or f"boxes-{args.split_scheme}-lr{args.lr:g}"
            loggers.append(_SafeCometLogger(
                project_name="milliontrees-boxes",
                name=comet_name,
                tags=[f"split-{args.split_scheme}", "geometry-boxes"],
            ))
        except Exception as e:
            print(f"Comet ML logging disabled: {e}")

    # Callbacks
    callbacks = []
    checkpoint_cb = None
    if has_val:
        # Monitor box_recall (mode=max) rather than the regression loss: the
        # loss can drift upward while detections stay good, so it's a poor
        # signal for checkpointing/early stopping.
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="boxes-best",
            monitor="box_recall",
            mode="max",
            save_last=True,
            save_top_k=1,
        )
        callbacks.append(checkpoint_cb)
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="box_recall",
            patience=args.early_stop_patience,
            mode="max",
        ))

    trainer_kwargs = {}
    if has_val:
        trainer_kwargs["limit_val_batches"] = 1.0
        trainer_kwargs["num_sanity_val_steps"] = 2
    if args.smoke_test:
        trainer_kwargs["limit_train_batches"] = 2
        trainer_kwargs["limit_val_batches"] = 2
    model.create_trainer(
        logger=loggers[0] if loggers else None,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    model.trainer.fit(model)

    # Under DDP all ranks return from fit(); only rank 0 runs the final
    # MillionTrees eval and writes results (avoids redundant work / file races).
    if not model.trainer.is_global_zero:
        return

    print("\n=== Evaluating best checkpoint ===")
    if checkpoint_cb is not None:
        best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
        if best_path:
            print(f"Loading best checkpoint: {best_path}")
            model = df_main.deepforest.load_from_checkpoint(best_path, weights_only=False)

    eval_max_batches = 2 if args.smoke_test else None
    viz_dir = os.path.join(args.output_dir, "viz")
    results, results_str = evaluate(model, box_dataset, test_subset, batch_size=args.batch_size,
                                    viz_dir=viz_dir,
                                    max_batches=eval_max_batches)
    print(results_str)

    if loggers:
        exp = loggers[0].experiment
        safe = {k: float(v.item() if hasattr(v, "item") else v)
                for k, v in results.items()
                if isinstance(v, (int, float)) or (hasattr(v, "ndim") and v.ndim == 0)}
        exp.log_metrics({k: v for k, v in safe.items() if math.isfinite(v)})
        for img_path in sorted(glob.glob(os.path.join(viz_dir, "**", "*.png"), recursive=True)):
            exp.log_image(img_path, name=os.path.relpath(img_path, viz_dir))

    results_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
    with open(results_path, "w") as f:
        f.write(results_str)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
