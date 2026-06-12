"""Fine-tune DeepForest TreeFormer on MillionTrees TreePoints.

Trains on the MillionTrees train split and evaluates on test. Use
``--split-scheme random`` or ``--split-scheme zeroshot`` (zeroshot holds out
entire source datasets from train, but fine-tuning still runs on the remaining
sources).

Uses the point-detection stack from the DeepForest ``treeformer-training`` branch
(https://github.com/jveitchmichaelis/DeepForest/tree/treeformer-training).
Install with ``uv sync --group treeformer`` until this lands on weecology/DeepForest main.
"""

import argparse
import json
import os
import warnings

import pytorch_lightning as pl
import torch

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader


class MillionTreesPointBatchAdapter:
    """Wrap MillionTrees loaders for DeepForest point training.

    MillionTrees: (metadata, images[B,C,H,W], [{"y": (N,2), "labels": ...}])
    DeepForest:   (images, [{"points": (N,2), "labels": ...}], image_paths)
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
                points = t["y"]
                if points.dim() == 1:
                    points = points.unsqueeze(0)
                if len(points) == 0:
                    points = torch.zeros((0, 2), dtype=torch.float32)
                labels = t.get("labels", torch.zeros(len(points), dtype=torch.int64))
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.int64)
                else:
                    labels = labels.long()
                adapted.append({"points": points.float(), "labels": labels})
            yield images, adapted, paths

    def __len__(self):
        return len(self.loader)


def predict_batch(model, images):
    """Run TreeFormer inference; return MillionTrees-format prediction dicts."""
    warnings.filterwarnings("ignore")
    device = next(model.parameters()).device
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        preds = model.predict_step(images, 0)

    batch_y_pred = []
    for pred in preds:
        points = pred.get("points", torch.zeros((0, 2)))
        if len(points) == 0:
            batch_y_pred.append({
                "y": torch.zeros((0, 2), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
        else:
            scores = pred.get("scores", torch.ones(len(points), dtype=torch.float32))
            labels = pred.get("labels", torch.zeros(len(points), dtype=torch.int64))
            batch_y_pred.append({
                "y": points.detach().float().cpu(),
                "labels": labels.detach().cpu().long(),
                "scores": scores.detach().float().cpu(),
            })
    return batch_y_pred


def evaluate(model, dataset, test_subset, batch_size=8, viz_dir=None, max_batches=None):
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    all_y_pred, all_y_true = [], []
    for i, batch in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        _, images, targets = batch
        preds = predict_batch(model, images)
        all_y_pred.extend(preds)
        all_y_true.extend(targets)
    return dataset.eval(
        all_y_pred,
        all_y_true,
        test_subset.metadata_array[:len(all_y_true)],
        viz_dir=viz_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepForest TreeFormer on MillionTrees TreePoints"
    )
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate. Use ~1e-5 for pretrained fine-tuning, ~2e-4 for random-weights training.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/points/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        help="Lightning accelerator (use 'cpu' for debugging).",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--comet", action="store_true",
                        help="Log to Comet ML (requires .comet.config or COMET_API_KEY)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weecology/deepforest-tree-point",
        help="Hugging Face repo for TreeFormer weights. Ignored when --random-weights is set.",
    )
    parser.add_argument(
        "--random-weights", action="store_true",
        help="Initialize from ImageNet backbone only (no pretrained TreeFormer head). "
             "Use a higher LR (e.g. 2e-4) with this flag.",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Limit to 2 train/val batches and 1 epoch")
    parser.add_argument("--score-thresh", type=float, default=0.1,
                        help="Relative peak threshold for density_to_points (standard 0.1).")
    parser.add_argument("--score-integration-radius", type=int, default=2,
                        help="peak_local_max min_distance in density-map px (~4x image px). "
                             "Standard default 2 (tuned from the deepforest default of 5).")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_epochs = 1
        args.early_stop_patience = 1

    os.makedirs(args.output_dir, exist_ok=True)

    point_dataset = get_dataset(
        "TreePoints",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
    )

    train_subset = point_dataset.get_subset("train")
    test_subset = point_dataset.get_subset("test")

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

    filename_id_to_path = point_dataset._filename_id_to_code
    train_adapted = MillionTreesPointBatchAdapter(train_loader, filename_id_to_path)
    val_adapted = MillionTreesPointBatchAdapter(val_loader, filename_id_to_path) if has_val else None

    config_args = {
        "architecture": "treeformer",
        "train": {"epochs": args.max_epochs, "lr": args.lr},
        "validation": {
            "root_dir": str(point_dataset._data_dir / "images"),
            # Compute point_precision/point_recall/val_mae every epoch. The base
            # config defaults this to 20, so with ~20 epochs the metrics would
            # only ever log on the final epoch and you'd see nothing but val_loss.
            # These metrics come from the val dataloader, not a csv_file.
            "val_accuracy_interval": 1,
        },
        "batch_size": args.batch_size,
        "devices": args.gpus,
        "accelerator": args.accelerator,
        "workers": args.num_workers,
    }
    if args.gpus > 1:
        # TreeFormer has heads (box detection / cropmodel) that don't contribute
        # to the point loss, so DDP must be told to expect unused parameters.
        config_args["strategy"] = "ddp_find_unused_parameters_true"

    model = df_main.deepforest(
        config_args=config_args,
        existing_train_dataloader=train_adapted,
        existing_val_dataloader=val_adapted,
    )

    if args.random_weights:
        # ImageNet backbone + randomly-initialized head; skip pretrained TreeFormer weights.
        print("Initializing from ImageNet backbone only (random head weights).")
        model.load_model(model_name=None)
    else:
        print(f"Loading pretrained TreeFormer checkpoint: {args.checkpoint}")
        model.load_model(args.checkpoint)

    # Standardized point-extraction hyperparams (see existing_models/treeformer
    # grid). peak_local_max min_distance=2 (down from deepforest's 5) recovers
    # recall in dense canopy; set on the submodule since postprocess_density
    # reads model.model.* (config is ignored after load_model). score_thresh
    # stays 0.1 (the dataset eval_score_threshold is the binding score filter).
    model.model.score_thresh = args.score_thresh
    model.model.score_integration_radius = args.score_integration_radius
    print(f"score_thresh: {model.model.score_thresh} "
          f"score_integration_radius: {model.model.score_integration_radius}")

    init_label = "random-weights" if args.random_weights else "pretrained"
    print(f"Init: {init_label}  |  LR: {args.lr}  |  Split: {args.split_scheme}")

    loggers = []
    if args.comet:
        try:
            from pytorch_lightning.loggers import CometLogger

            loggers.append(CometLogger(
                project_name="milliontrees-treeformer-points",
                tags=[f"split-{args.split_scheme}", "geometry-points", "treeformer", init_label],
            ))
        except Exception as e:
            print(f"Comet ML logging disabled: {e}")

    callbacks = []
    checkpoint_cb = None
    if has_val:
        # Monitor point_recall (mode=max) rather than val_loss: TreeFormer's
        # val_loss can drift upward while detections stay good, so it's a poor
        # signal for checkpointing/early stopping.
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="treeformer-{epoch:02d}-{point_recall:.4f}",
            monitor="point_recall",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_cb)
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="point_recall",
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

    print("\n=== Evaluating best checkpoint on MillionTrees test split ===")
    if checkpoint_cb is not None:
        best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
        if best_path:
            print(f"Loading best checkpoint: {best_path}")
            model = df_main.deepforest.load_from_checkpoint(best_path, weights_only=False)
            # load_from_checkpoint rebuilds the treeformer submodule from config,
            # reverting score_thresh/score_integration_radius to deepforest
            # defaults (5). Re-apply the standardized values so the final eval
            # matches the pretrained-baseline post-processing.
            model.model.score_thresh = args.score_thresh
            model.model.score_integration_radius = args.score_integration_radius
            print(f"score_thresh: {model.model.score_thresh} "
                  f"score_integration_radius: {model.model.score_integration_radius}")

    eval_max_batches = 2 if args.smoke_test else None
    results, results_str = evaluate(
        model, point_dataset, test_subset,
        batch_size=args.batch_size,
        viz_dir=os.path.join(args.output_dir, "viz"),
        max_batches=eval_max_batches,
    )
    print(results_str)

    results_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(results_str)

    json_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.json")
    flat = {
        k: float(v) if hasattr(v, "item") else v
        for k, v in results.items()
        if isinstance(v, (int, float, torch.Tensor))
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "TreeFormer-finetuned",
            "task": "TreePoints",
            "split": args.split_scheme,
            "checkpoint": "random-weights" if args.random_weights else args.checkpoint,
            "lr": args.lr,
            "metrics": flat,
        }, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
