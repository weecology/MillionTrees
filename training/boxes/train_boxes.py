"""Train DeepForest (RetinaNet) on MillionTrees TreeBoxes.

Adapts MillionTrees dataloaders to the DeepForest training API so that
DeepForest's own LightningModule and Trainer do all the heavy lifting.
Custom code here is limited to two things that DeepForest doesn't cover:

  MillionTreesBatchAdapter  — translates batch format (metadata→path, y→boxes)
  evaluate()                — uses MillionTrees eval API (DeepForest's is CSV-based)

Requires two small DeepForest fixes (submitted upstream):
  1. on_fit_start: don't raise when existing_train_dataloader is set
  2. create_trainer: enable validation when existing_val_dataloader is set
"""

import argparse
import os
import warnings

import pytorch_lightning as pl
import torch

from deepforest import main as df_main
from deepforest.callbacks import images_callback

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader


# ---------------------------------------------------------------------------
# Batch format adapter
# ---------------------------------------------------------------------------

class MillionTreesBatchAdapter:
    """Wraps a MillionTrees dataloader to yield (path, images, targets) batches
    compatible with the DeepForest training API.

    MillionTrees: (metadata[B,2], images[B,C,H,W], [{"y": boxes, "labels": int64}])
    DeepForest:   (list[str],     images[B,C,H,W], [{"boxes": boxes, "labels": int64}])

    Images are already CHW float32 0-1 from MillionTrees, so no conversion needed.
    """

    def __init__(self, loader, filename_id_to_path=None):
        self.loader = loader
        self.filename_id_to_path = filename_id_to_path or {}

    def __iter__(self):
        for metadata, images, targets in self.loader:
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
            yield paths, images, adapted

    def __len__(self):
        return len(self.loader)


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

def evaluate(model, dataset, test_subset, batch_size=12):
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    all_y_pred, all_y_true = [], []
    for batch in test_loader:
        _, images, targets = batch
        preds = predict_batch(model, images)
        all_y_pred.extend(preds)
        all_y_true.extend(targets)
    results, results_str = dataset.eval(
        all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)]
    )
    return results, results_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DeepForest on MillionTrees TreeBoxes")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
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
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/boxes/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        help="Lightning accelerator (use 'cpu' for debugging).",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--vis-every-n-epochs", type=int, default=5,
                        help="Save val images every N epochs (0 to disable)")
    parser.add_argument("--vis-n-images", type=int, default=4)
    parser.add_argument("--comet", action="store_true",
                        help="Log to Comet ML (requires .comet.config or COMET_API_KEY)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    box_dataset = get_dataset(
        "TreeBoxes",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        include_unsupervised=args.include_unsupervised,
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
    has_val = len(val_loader) > 0

    # Adapt batch format: (metadata, images, {"y":…}) → (path, images, {"boxes":…})
    filename_id_to_path = box_dataset._filename_id_to_code
    train_adapted = MillionTreesBatchAdapter(train_loader, filename_id_to_path)
    val_adapted = MillionTreesBatchAdapter(val_loader, filename_id_to_path) if has_val else None

    # images_callback reads images from disk using config["validation"]["root_dir"]
    images_dir = str(box_dataset._data_dir / "images")

    # Build DeepForest model and load pretrained weights
    model = df_main.deepforest(
        config_args={
            "train": {"epochs": args.max_epochs, "lr": args.lr},
            "validation": {"root_dir": images_dir},
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

            loggers.append(_SafeCometLogger(
                project_name="milliontrees-boxes",
                tags=[f"split-{args.split_scheme}", "geometry-boxes"],
            ))
        except Exception as e:
            print(f"Comet ML logging disabled: {e}")

    # Callbacks
    callbacks = []
    checkpoint_cb = None
    if has_val:
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="boxes-best",
            monitor="val_bbox_regression",
            mode="min",
            save_last=True,
            save_top_k=1,
        )
        callbacks.append(checkpoint_cb)
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val_bbox_regression",
            patience=args.early_stop_patience,
            mode="min",
        ))
        if args.vis_every_n_epochs > 0:
            callbacks.append(images_callback(
                savedir=os.path.join(args.output_dir, "val_images"),
                n=args.vis_n_images,
                every_n_epochs=args.vis_every_n_epochs,
            ))

    # create_trainer reads config for devices/accelerator/epochs; kwargs override the rest
    model.create_trainer(
        logger=loggers[0] if loggers else None,
        callbacks=callbacks,
    )

    model.trainer.fit(model)

    print("\n=== Evaluating best checkpoint ===")
    if checkpoint_cb is not None:
        best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
        if best_path:
            print(f"Loading best checkpoint: {best_path}")
            model = df_main.deepforest.load_from_checkpoint(best_path, weights_only=False)

    results, results_str = evaluate(model, box_dataset, test_subset, batch_size=args.batch_size)
    print(results_str)

    results_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
    with open(results_path, "w") as f:
        f.write(results_str)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
