"""Train a Mask R-CNN on MillionTrees TreePolygons via the DeepForest stack.

This driver wires the MillionTrees TreePolygons dataset into DeepForest's
polygon training stack (the ``cursor/polygon-maskrcnn-workflow`` PR):

  * Training uses DeepForest's ``PolygonDataset`` + ``MaskRCNN``. The dataset
    emits *panoptic-encoded* targets (a single ``(H, W)`` uint16 instance map +
    surviving id list) instead of a dense ``(N, H, W)`` mask stack, and the
    model decodes them on-device. This is the memory optimization that lets
    dense tiles train without materializing gigabytes of masks up front.
  * The recipe (OAM-TCD Detectron2 alignment) lives in the self-contained
    ``deepforest_polygon.yaml`` next to this file and is loaded via
    ``deepforest.utilities.load_config`` with runtime overrides for the data
    paths / epochs / lr. See that file for why it's vendored here.
  * Evaluation stays on the MillionTrees side: the trained DeepForest model is
    scored with the TreePolygons metrics (mask accuracy / recall / mask-aware
    precision / AP50 / merge-commission) so the numbers remain leaderboard
    comparable.

The bridge from MillionTrees splits to DeepForest is a generated annotation CSV
(``image_path``, ``geometry`` WKT, ``label``) pointed at the packaged images dir.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image

from deepforest import utilities
from deepforest.main import deepforest

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.datasets.polygon_stream_eval import (
    TreePolygonsStreamingEvalState,
    merge_viz_samples,
)

# Default recipe: vendored flattened DeepForest polygon/oam config.
DEFAULT_CONFIG = str(Path(__file__).with_name("deepforest_polygon.yaml"))


# --------------------------------------------------------------------------- #
# MillionTrees -> DeepForest annotation bridge
# --------------------------------------------------------------------------- #
def build_annotation_csvs(data_dir, split_scheme, output_dir, include_unsupervised):
    """Write DeepForest-format train/val annotation CSVs from a MillionTrees split.

    The MillionTrees ``<split_scheme>.csv`` carries one polygon per row with a
    WKT ``polygon`` column, a ``filename`` (relative to ``<data_dir>/images``),
    a ``source`` and a ``split`` (train/test). DeepForest's ``read_file`` wants
    ``image_path`` + ``geometry`` (WKT) + ``label``, so we slim and rename.

    Returns ``(train_csv, val_csv, images_dir)`` where ``val_csv`` is built from
    the ``test`` rows (MillionTrees has no separate val split) or ``None`` if the
    test split is empty for this configuration.
    """
    split_csv = Path(data_dir) / f"{split_scheme}.csv"
    df = pd.read_csv(split_csv, low_memory=False)

    # Mirror TreePolygonsDataset's default source filtering: drop unsupervised
    # sources unless the caller opted in. (Other dataset filters use their
    # defaults; expose more here if a run needs them.)
    if not include_unsupervised:
        is_unsup = df["source"].astype(str).str.contains("unsupervised", case=False)
        df = df[~is_unsup]

    df = df[df["polygon"].notna()].copy()
    # Labels arrive mixed-case ("Tree"/"tree"); the config label_dict is {Tree: 0}.
    df["label"] = "Tree"

    images_dir = str(Path(data_dir) / "images")
    os.makedirs(output_dir, exist_ok=True)

    def _write(split_name, path):
        rows = df[df["split"] == split_name]
        if len(rows) == 0:
            return None, 0
        out = rows[["filename", "polygon", "label"]].rename(
            columns={"filename": "image_path", "polygon": "geometry"}
        )
        out.to_csv(path, index=False)
        return path, len(out)

    train_csv, n_train = _write("train", os.path.join(output_dir, "deepforest_train.csv"))
    val_csv, n_val = _write("test", os.path.join(output_dir, "deepforest_val.csv"))
    print(
        f"[bridge] wrote {n_train} train / {n_val} val polygon annotations "
        f"from {split_csv} (images: {images_dir})"
    )
    return train_csv, val_csv, images_dir


def build_config(args, train_csv, val_csv, images_dir, log_root):
    """Load the vendored DeepForest config and apply MillionTrees overrides."""
    overrides = {
        "workers": args.num_workers,
        "batch_size": args.batch_size,
        "devices": args.gpus,
        "log_root": log_root,
        "train": {
            "csv_file": train_csv,
            "root_dir": images_dir,
            "lr": args.lr,
            "epochs": args.max_epochs,
        },
        "validation": {
            "csv_file": val_csv,
            "root_dir": images_dir if val_csv else None,
        },
    }
    if not args.augment:
        # Keep the geometric/photometric pipeline off for sanity runs.
        overrides["train"]["augmentations"] = []

    cfg = utilities.load_config(config_name=args.config, overrides=overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Box-pretrained backbone (weak-supervision ablation, Table 6)
# --------------------------------------------------------------------------- #
def _extract_box_pretrained_backbone(checkpoint_path):
    """Load a DeepForest box checkpoint; return backbone keys for the Mask R-CNN."""
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


def apply_box_pretrained_backbone(deepforest_module, checkpoint_path):
    """Overwrite the Mask R-CNN ResNet backbone with box-pretrained weights.

    Leaves DeepForest's COCO-initialized FPN / RPN / heads in place and only
    swaps ``backbone.body.*`` — the same surgery the previous torchvision-based
    driver did, retargeted onto ``deepforest_module.model`` (the ``MaskRCNN``).
    """
    backbone_state = _extract_box_pretrained_backbone(checkpoint_path)
    missing, unexpected = deepforest_module.model.load_state_dict(
        backbone_state, strict=False
    )
    if unexpected:
        raise ValueError(f"Unexpected keys when loading box backbone: {unexpected}")
    loaded = sum(1 for k in backbone_state if k.startswith("backbone.body."))
    if loaded == 0:
        raise ValueError("No backbone.body keys loaded from box pretrained checkpoint.")
    return {"missing": missing, "loaded_backbone_keys": loaded}


# --------------------------------------------------------------------------- #
# MillionTrees-side evaluation (unchanged metric path)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Tiled native-resolution inference (matches the DeepForest training scale)
# --------------------------------------------------------------------------- #
# The model is trained on RandomResizedCrop 640 crops of *native-resolution*
# imagery (see deepforest_polygon.yaml). Feeding the MillionTrees eval loader's
# whole-image-resized-to-448 tensors put trees at a scale the model never saw
# and produced oversized blob masks. Instead we run DeepForest's sliding-window
# ``predict_tile`` at the training patch size over the native image, then project
# the returned native-coordinate polygons into the 448 metric space (where the
# ground-truth masks live) so the leaderboard numbers stay comparable.
POLYGON_EVAL_PATCH_SIZE = 640        # == training RandomResizedCrop size
POLYGON_EVAL_PATCH_OVERLAP = 0.1     # window overlap so edge trees aren't split
POLYGON_EVAL_TILE_IOU = 0.15         # cross-window polygon NMS (predict_tile default)


def ensure_predict_trainer(model):
    """Give a (possibly freshly loaded) DeepForest model a single-device predict trainer.

    ``predict_tile`` calls ``self.trainer.predict``; a checkpoint loaded via
    ``deepforest.load_from_checkpoint`` has no trainer, and under multi-GPU
    training the eval runs on rank 0 only — so force ``devices=1`` to keep the
    per-image predict loop on a single device. We also drop the dataloader
    workers to 0: each image yields only a handful of windows, so spawning the
    training ``workers`` processes per image is pure overhead (and thrashes
    low-core nodes).
    """
    model.config.workers = 0
    model.create_trainer(
        logger=False,
        devices=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )


def _resolve_native_path(dataset, filename_id):
    """Map a loader ``filename_id`` back to the native (un-resized) image on disk."""
    filename = dataset._filename_id_to_code[int(filename_id)]
    return os.path.join(dataset._data_dir, "images", filename)


def _rasterize_geoms_to_masks(geoms, scale_x, scale_y, target_size):
    """Rasterize native-coordinate shapely polygons into a ``(N, S, S)`` uint8 stack.

    Each polygon's vertices are scaled by ``(scale_x, scale_y)`` so a polygon
    defined in native image pixels is drawn directly at the ``target_size`` metric
    resolution, matching how the dataset rasterizes ground-truth masks.
    """
    masks = []
    for geom in geoms:
        mask = np.zeros((target_size, target_size), dtype=np.uint8)
        if geom is not None and not geom.is_empty:
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                if poly.is_empty:
                    continue
                coords = np.asarray(poly.exterior.coords, dtype=np.float64)
                coords[:, 0] *= scale_x
                coords[:, 1] *= scale_y
                cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
        masks.append(mask)
    if masks:
        return np.stack(masks)
    return np.zeros((0, target_size, target_size), dtype=np.uint8)


def tiled_predict_for_eval(
    model,
    dataset,
    metadata,
    *,
    patch_size=POLYGON_EVAL_PATCH_SIZE,
    patch_overlap=POLYGON_EVAL_PATCH_OVERLAP,
    iou_threshold=POLYGON_EVAL_TILE_IOU,
):
    """Tiled native-resolution inference for one batch, in MillionTrees eval format.

    For each sample (identified by its ``filename_id`` in ``metadata``) the native
    image is run through ``predict_tile`` and the resulting polygons are projected
    into the dataset's ``image_size`` space and rasterized to per-instance masks.
    Returns one ``{"y", "labels", "scores"}`` dict per sample, aligned with the
    loader batch order.
    """
    target_size = dataset.image_size
    if not isinstance(metadata, torch.Tensor):
        metadata = torch.as_tensor(metadata)

    batch_y_pred = []
    for row in metadata:
        path = _resolve_native_path(dataset, int(row[0]))
        with Image.open(path) as im:
            native_w, native_h = im.size

        gdf = model.predict_tile(
            path=path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            iou_threshold=iou_threshold,
        )

        if gdf is None or len(gdf) == 0:
            batch_y_pred.append({
                "y": torch.zeros((0, target_size, target_size), dtype=torch.uint8),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
            continue

        masks = _rasterize_geoms_to_masks(
            list(gdf.geometry),
            target_size / native_w,
            target_size / native_h,
            target_size,
        )
        batch_y_pred.append({
            "y": torch.from_numpy(masks),
            # Single foreground class ("Tree" -> 0), matching the GT labels.
            "labels": torch.zeros((len(masks),), dtype=torch.int64),
            "scores": torch.as_tensor(gdf["score"].to_numpy(), dtype=torch.float32),
        })
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
    """Run MillionTrees test-set evaluation on a trained DeepForest model.

    ``eval_mode``:
        - ``stream`` (default): update metrics per batch (lower peak memory).
        - ``legacy``: accumulate full pred/true lists then call ``dataset.eval``.
    """
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    model.eval()
    ensure_predict_trainer(model)

    if eval_mode == "legacy":
        all_y_pred, all_y_true = [], []
        for batch in test_loader:
            metadata, images, targets = batch
            preds = tiled_predict_for_eval(model, dataset, metadata)
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
        preds = tiled_predict_for_eval(model, dataset, metadata)
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


# --------------------------------------------------------------------------- #
# Training orchestration
# --------------------------------------------------------------------------- #
def build_trainer(model, args, log_root):
    """Attach loggers + checkpointing and (re)build the DeepForest trainer."""
    callbacks = []
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(log_root, "checkpoints"),
        filename="polygons-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(checkpoint_cb)
    if args.early_stopping:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, mode="min"
            )
        )

    loggers = []
    if args.comet:
        try:
            from pytorch_lightning.loggers import CometLogger

            loggers.append(
                CometLogger(
                    project_name="milliontrees-polygons",
                    experiment_name=args.comet_name,
                    tags=[
                        f"split-{args.split_scheme}",
                        "geometry-polygons",
                        "stack-deepforest",
                        f"lr-{args.lr:g}",
                        f"init-{args.init_mode}",
                    ],
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"Comet ML logging disabled: {e}")

    trainer_kwargs = {"callbacks": callbacks, "logger": loggers if loggers else True}
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    # Multi-GPU: DDP with unused-parameter detection. Mask R-CNN's RPN/ROI heads
    # take data-dependent branches (e.g. zero-proposal images), so some params
    # miss a backward each step and the default DDP reducer would error.
    if args.gpus and args.gpus > 1:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"

    model.create_trainer(**trainer_kwargs)
    return checkpoint_cb


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepForest Mask R-CNN on MillionTrees TreePolygons"
    )
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to the DeepForest polygon config YAML (vendored recipe by default).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/polygons/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=448,
                        help="Resolution of the MillionTrees eval loader (the DeepForest "
                             "model itself runs at native/crop resolution).")
    parser.add_argument("--comet", action="store_true",
                        help="Log to Comet ML (requires .comet.config or COMET_API_KEY)")
    parser.add_argument("--comet-name", type=str, default=None,
                        help="Explicit Comet experiment name. Defaults to "
                             "polygons-<split>-<init>-lr<lr>.")
    parser.add_argument("--limit-train-batches", type=float, default=None)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--eval-mode", type=str, default="stream",
                        choices=["stream", "legacy"])
    parser.add_argument("--init-mode", type=str, default="coco",
                        choices=["coco", "box_pretrained"],
                        help="coco: DeepForest COCO-initialized Mask R-CNN. "
                             "box_pretrained: overwrite the ResNet backbone with a box checkpoint.")
    parser.add_argument("--box-backbone-checkpoint", type=str, default=None,
                        help="DeepForest/box checkpoint when --init-mode=box_pretrained.")
    parser.add_argument("--include-unsupervised", action="store_true",
                        help="Include unsupervised sources in the training annotations.")
    parser.add_argument("--data-scope", type=str, default="subset",
                        choices=["subset", "full"],
                        help="Tag for experiment aggregation (subset vs full data pull).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the config's train augmentations. --no-augment strips them.")
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=False,
                        help="EarlyStopping on val_loss (off by default; the OAM recipe trains full epochs).")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--debug-overfit", action="store_true",
                        help="Sanity check: validate/evaluate on the TRAIN annotations and "
                             "skip writing leaderboard results.")
    args = parser.parse_args()

    if args.comet_name is None:
        args.comet_name = f"polygons-{args.split_scheme}-{args.init_mode}-lr{args.lr:g}"

    os.makedirs(args.output_dir, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # MillionTrees dataset: resolves the packaged data dir (download/version/mini)
    # and provides the eval metrics + test subset.
    polygon_dataset = get_dataset(
        "TreePolygons",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
        include_unsupervised=args.include_unsupervised,
    )
    data_dir = polygon_dataset._data_dir

    train_csv, val_csv, images_dir = build_annotation_csvs(
        data_dir, args.split_scheme, args.output_dir, args.include_unsupervised
    )
    if train_csv is None:
        print("No training annotations for this split; skipping training.")
        return

    if args.debug_overfit:
        print("[debug-overfit] validation := train annotations; "
              "leaderboard results will NOT be written.")
        val_csv = train_csv

    log_root = os.path.join(args.output_dir, "logs")
    cfg = build_config(args, train_csv, val_csv, images_dir, log_root)

    model = deepforest(config=cfg)
    init_details = {"init_mode": args.init_mode, "stack": "deepforest-maskrcnn"}
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

    checkpoint_cb = build_trainer(model, args, log_root)
    model.trainer.fit(model)

    # Under DDP all ranks return from fit(); only rank 0 runs the final
    # MillionTrees eval and writes results (avoids redundant work / file races).
    if not model.trainer.is_global_zero:
        return

    # Score the best checkpoint with the MillionTrees metrics.
    best_path = checkpoint_cb.best_model_path or None
    if best_path and os.path.exists(best_path):
        print(f"\n=== Loading best checkpoint: {best_path} ===")
        model = deepforest.load_from_checkpoint(best_path)
    else:
        print("\n=== No checkpoint saved; evaluating the in-memory model ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_subset = train_subset_or_test(polygon_dataset, args.debug_overfit)
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
            "best_checkpoint_path": best_path,
            **init_details,
        },
    }
    write_run_metadata(json_path, payload)
    print(f"JSON results saved to {json_path}")


def train_subset_or_test(polygon_dataset, debug_overfit):
    """Pick the MillionTrees subset to evaluate on (train under --debug-overfit)."""
    return polygon_dataset.get_subset("train" if debug_overfit else "test")


if __name__ == "__main__":
    main()
