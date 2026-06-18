"""Pretrain DeepForest on TreeBoxes and export backbone weights for polygon Mask R-CNN."""

import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from deepforest import main as df_main

from milliontrees import get_dataset
from training.boxes.train import _AdaptCollate, evaluate


def _flatten_numeric_metrics(results):
    flat = {}
    for k, v in results.items():
        if isinstance(v, (int, float)):
            flat[k] = float(v)
        elif isinstance(v, torch.Tensor) and v.ndim == 0:
            flat[k] = float(v.item())
    return flat


def export_backbone_weights(model, output_path):
    """Export inner torchvision backbone weights using Lightning checkpoint key prefix."""
    state_dict = model.model.state_dict()
    backbone = {
        f"model.backbone.body.{k[len('backbone.body.'):]}": v.detach().cpu()
        for k, v in state_dict.items()
        if k.startswith("backbone.body.")
    }
    if not backbone:
        raise ValueError("No backbone.body keys found; cannot export transferable weights.")
    torch.save({"state_dict": backbone}, output_path)
    return len(backbone)


def main():
    parser = argparse.ArgumentParser(
        description="Train on TreeBoxes and export backbone weights for polygon pretraining."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"),
    )
    parser.add_argument(
        "--split-scheme",
        type=str,
        default="within-distribution",
        choices=["within-distribution", "out-of-distribution", "crossgeometry"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-unsupervised", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/boxes/pretrain_outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-scope",
        type=str,
        default="subset",
        choices=["subset", "full"],
    )
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--comet", action="store_true",
                        help="Log to Comet ML (requires .comet.config or COMET_API_KEY)")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    box_dataset = get_dataset(
        "TreeBoxes",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        include_unsupervised=args.include_unsupervised,
        include_sources=["*unsupervised*"] if args.include_unsupervised else None,
    )
    train_subset = box_dataset.get_subset("train")
    test_subset = box_dataset.get_subset("test")

    if len(train_subset) == 0:
        raise RuntimeError("No training samples for this split; cannot pretrain.")

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

    model = df_main.deepforest(
        config_args={
            "train": {"epochs": args.max_epochs, "lr": args.lr},
            "validation": {"root_dir": str(box_dataset._data_dir / "images")},
            "batch_size": args.batch_size,
            "devices": args.gpus,
            "accelerator": args.accelerator,
            "workers": args.num_workers,
        },
        existing_train_dataloader=train_adapted,
        existing_val_dataloader=val_adapted,
    )
    model.load_model("weecology/deepforest-tree")
    model.config["train"]["csv_file"] = "existing_train_dataloader"

    loggers = []
    if args.comet:
        try:
            import json as _json
            from pytorch_lightning.loggers import CometLogger

            class _SafeCometLogger(CometLogger):
                def log_hyperparams(self, params):
                    safe = {}
                    for k, v in params.items():
                        try:
                            _json.dumps(v)
                            safe[k] = v
                        except (TypeError, ValueError):
                            safe[k] = type(v).__name__
                    super().log_hyperparams(safe)

            loggers.append(_SafeCometLogger(
                project_name="milliontrees-pretrain",
                tags=[f"split-{args.split_scheme}", "geometry-boxes", "backbone-pretrain"],
            ))
        except Exception as e:
            print(f"Comet ML logging disabled: {e}")

    callbacks = []
    checkpoint_cb = None
    if has_val:
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="box-pretrain-best",
            monitor="val_bbox_regression",
            mode="min",
            save_last=True,
            save_top_k=1,
        )
        callbacks.append(checkpoint_cb)

    trainer_kwargs = {}
    if has_val:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches if args.limit_val_batches else 1.0
        trainer_kwargs["num_sanity_val_steps"] = 2
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None and has_val:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    model.create_trainer(callbacks=callbacks, logger=loggers[0] if loggers else False, **trainer_kwargs)
    model.trainer.fit(model)

    best_path = None
    if checkpoint_cb is not None:
        best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
        if best_path:
            model = df_main.deepforest.load_from_checkpoint(best_path, weights_only=False)

    results, results_str = evaluate(
        model,
        box_dataset,
        test_subset,
        batch_size=args.batch_size,
        max_batches=args.eval_max_batches,
    )
    txt_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(results_str)

    backbone_path = os.path.join(args.output_dir, f"box_backbone_{args.split_scheme}.pt")
    num_keys = export_backbone_weights(model, backbone_path)

    json_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.json")
    payload = {
        "model": "box-pretrain",
        "task": "TreeBoxes",
        "split": args.split_scheme,
        "metrics": _flatten_numeric_metrics(results),
        "run_metadata": {
            "seed": args.seed,
            "data_scope": args.data_scope,
            "include_unsupervised": args.include_unsupervised,
            "mini": args.mini,
            "best_checkpoint_path": best_path,
            "backbone_export_path": str(Path(backbone_path).resolve()),
            "exported_backbone_keys": num_keys,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Backbone export written to {backbone_path}")
    print(f"Metadata written to {json_path}")


if __name__ == "__main__":
    main()
