"""Train a *native Detectron2* Mask R-CNN on MillionTrees TreePolygons.

This is the implementation-control twin of ``training/polygons/train.py``.
``train.py`` runs Mask R-CNN through the DeepForest / torchvision stack; this
driver runs the *same data, same metric, same eval scale* through stock
Detectron2 (``COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x``). Comparing the
two MillionTrees polygon scores answers the question the DeepForest run can't on
its own: are the low polygon numbers a **data** problem (both stacks land in the
same place) or a DeepForest **implementation** problem (Detectron2 does better on
identical data)?

Why Detectron2 lives here and not via DeepForest: the Detectron2 build is the one
compiled from source for the CanopyRS baselines (``existing_models/canopyrs``),
so this script is meant to run under *that* venv -- it imports only
``milliontrees`` (editable in that env) + ``detectron2`` and deliberately does
NOT import ``deepforest`` (absent from that env).

The pipeline mirrors train.py's *resize* arm so the comparison is apples-to-apples:

  * Data bridge: the MillionTrees ``<split>.csv`` (one WKT polygon per row) is
    grouped per image into Detectron2 standard dataset dicts (polygon
    ``segmentation`` + ``XYXY_ABS`` bbox, single ``Tree`` class).
  * Training: stock Detectron2 ``DefaultTrainer`` with COCO-pretrained weights and
    the standard ResizeShortestEdge recipe.
  * Arms: ``--init-mode`` selects one leg of the weak-supervision ablation --
    ``coco`` (tree-naive control), ``box_pretrained_full`` (the whole stage-1 network
    from ``pretrain_detectron2.py``), or ``cotrain`` (COCO init, weak TreeBoxes tiles
    mixed into the polygon training set with the mask loss held off the box-only rows).
  * Eval: stays on the MillionTrees side. Each native test image runs through the
    Detectron2 predictor; predicted instance masks are resized into the dataset's
    ``image_size`` square (where the GT masks are rasterized) and scored with
    ``TreePolygonsStreamingEvalState`` -- the exact metric path train.py uses, so
    the recall / mask-aware precision / AP40 numbers are leaderboard-comparable.
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from shapely import wkt

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.datasets.polygon_stream_eval import (
    TreePolygonsStreamingEvalState,
    merge_viz_samples,
)
from milliontrees.common.prediction_dump import add_dump_args, make_dumper

import d2_weak_supervision as d2ws

_COCO_MASKRCNN = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# Pre-fetched COCO weights (compute nodes may have no outbound network).
_DEFAULT_WEIGHTS = str(
    Path(__file__).with_name("detectron2_assets") / "model_final_f10217.pkl"
)


# --------------------------------------------------------------------------- #
# MillionTrees -> Detectron2 annotation bridge
# --------------------------------------------------------------------------- #
def _filter_split_df(data_dir, split_scheme, include_unsupervised):
    """Read ``<split>.csv`` and apply TreePolygonsDataset's default source filter."""
    df = pd.read_csv(Path(data_dir) / f"{split_scheme}.csv", low_memory=False)
    if not include_unsupervised:
        is_unsup = df["source"].astype(str).str.contains("unsupervised", case=False)
        df = df[~is_unsup]
    df = df[df["polygon"].notna()].copy()
    return df


def _polygon_to_segmentations(geom, width, height):
    """Flatten a shapely (Multi)Polygon to Detectron2 segmentation lists + bbox.

    Returns ``(segmentations, [x0, y0, x1, y1])`` in absolute pixels, clipped to
    the image, or ``(None, None)`` if no ring has >= 3 vertices.
    """
    if geom is None or geom.is_empty:
        return None, None
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    segs = []
    xs, ys = [], []
    for poly in polys:
        if poly.is_empty:
            continue
        coords = np.asarray(poly.exterior.coords, dtype=np.float64)
        if coords.shape[0] < 4:  # need >= 3 distinct points (ring repeats first)
            continue
        coords[:, 0] = np.clip(coords[:, 0], 0, width)
        coords[:, 1] = np.clip(coords[:, 1], 0, height)
        segs.append(coords.reshape(-1).tolist())
        xs.extend(coords[:, 0].tolist())
        ys.extend(coords[:, 1].tolist())
    if not segs:
        return None, None
    return segs, [min(xs), min(ys), max(xs), max(ys)]


def build_detectron2_dicts(df, split_name, images_dir):
    """Group one split's rows into Detectron2 standard dataset dicts (per image)."""
    rows = df[df["split"] == split_name]
    records = []
    for image_id, (filename, group) in enumerate(rows.groupby("filename")):
        image_path = os.path.join(images_dir, filename)
        with Image.open(image_path) as im:
            width, height = im.size
        annotations = []
        for geom_wkt in group["polygon"]:
            segs, bbox = _polygon_to_segmentations(wkt.loads(geom_wkt), width, height)
            if segs is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            annotations.append({
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": segs,
                "category_id": 0,
            })
        if not annotations:
            continue
        records.append({
            "file_name": image_path,
            "image_id": image_id,
            "height": height,
            "width": width,
            "annotations": annotations,
        })
    return records


def register_datasets(df, images_dir, prefix, extra_train_records=()):
    """Register train/val Detectron2 catalogs; return (train_name, n_train_images).

    ``extra_train_records`` are the box-only weak tiles of the co-training arm. They are
    flagged ``real_mask=False`` upstream, so :class:`~d2_weak_supervision.MaskAwareROIHeads`
    keeps them out of the mask loss while they still contribute RPN and box losses.
    """
    train_name = f"{prefix}_train"
    val_name = f"{prefix}_val"
    for name in (train_name, val_name):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)

    train_records = d2ws.mark_real_masks(
        build_detectron2_dicts(df, "train", images_dir), real=True)
    if extra_train_records:
        offset = len(train_records)
        for i, rec in enumerate(extra_train_records):
            rec["image_id"] = offset + i
        train_records = train_records + list(extra_train_records)
        print(f"[bridge] co-training set: {offset} polygon images + "
              f"{len(extra_train_records)} weak box-only images")
    DatasetCatalog.register(train_name, lambda r=train_records: r)
    MetadataCatalog.get(train_name).set(thing_classes=["Tree"])

    val_records = build_detectron2_dicts(df, "test", images_dir)
    DatasetCatalog.register(val_name, lambda r=val_records: r)
    MetadataCatalog.get(val_name).set(thing_classes=["Tree"])

    print(f"[bridge] registered {len(train_records)} train / {len(val_records)} val "
          f"images ({prefix}) from {images_dir}")
    return train_name, len(train_records)


# --------------------------------------------------------------------------- #
# Detectron2 config
# --------------------------------------------------------------------------- #
class AblationTrainer(DefaultTrainer):
    """DefaultTrainer whose loader carries the per-image ``real_mask`` flag."""

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=d2ws.WeakMaskDatasetMapper(cfg, is_train=True))


def build_cfg(args, train_name, n_train_images, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_COCO_MASKRCNN))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = (
        args.weights if os.path.exists(args.weights)
        else model_zoo.get_checkpoint_url(_COCO_MASKRCNN)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # Test gate matches the DeepForest polygon recipe (score_thresh 0.15): low
    # enough that the MillionTrees evaluator sees a near-complete score
    # distribution for AP40, but high enough to keep the streaming segm-mAP
    # tractable (a 0.05 gate floods ~300 low-conf masks/image and makes eval
    # crawl). detections_per_img 300 also mirrors the DeepForest recipe.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    # Every arm gets the same ROI heads, so the co-training arm differs from the others
    # only in its data. With no box-only images present these behave exactly like
    # StandardROIHeads.
    cfg.MODEL.ROI_HEADS.NAME = "MaskAwareROIHeads"

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    iters_per_epoch = max(1, math.ceil(n_train_images / args.batch_size))
    # The ablation pins --max-iter identically across arms. Without it the co-training
    # arm's epoch is ~5x longer, so "50 epochs" would silently hand it 5x the updates
    # and the comparison would confound the weak labels with the compute budget.
    cfg.SOLVER.MAX_ITER = args.max_iter or iters_per_epoch * args.max_epochs
    cfg.SOLVER.STEPS = (
        int(0.7 * cfg.SOLVER.MAX_ITER),
        int(0.9 * cfg.SOLVER.MAX_ITER),
    )
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch * max(1, args.max_epochs // 10)
    cfg.SOLVER.WARMUP_ITERS = min(1000, iters_per_epoch)
    # The box round's stage 1 diverged for want of this; stage 2 has always clipped.
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = args.grad_clip
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    cfg.INPUT.MIN_SIZE_TRAIN = tuple(args.min_size_train)
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST = args.max_size

    cfg.OUTPUT_DIR = output_dir
    cfg.SEED = args.seed
    os.makedirs(output_dir, exist_ok=True)
    print(f"[cfg] {n_train_images} imgs | {iters_per_epoch} iters/epoch x "
          f"{args.max_epochs} ep = {cfg.SOLVER.MAX_ITER} iters | lr {args.lr} | "
          f"steps {cfg.SOLVER.STEPS}")
    return cfg


# --------------------------------------------------------------------------- #
# MillionTrees-side evaluation (same metric path as train.py)
# --------------------------------------------------------------------------- #
def _resolve_native_path(dataset, filename_id):
    filename = dataset._filename_id_to_code[int(filename_id)]
    return os.path.join(dataset._data_dir, "images", filename)


def detectron2_predict_for_eval(predictor, dataset, metadata):
    """Run the Detectron2 predictor on each native image; return MillionTrees preds.

    Predicted instance masks come back at native resolution; each is resized into
    the dataset's ``image_size`` square (the GT mask space) so the metric is scale
    consistent with the DeepForest resize eval.
    """
    target_size = dataset.image_size
    if not isinstance(metadata, torch.Tensor):
        metadata = torch.as_tensor(metadata)

    batch_y_pred = []
    for row in metadata:
        path = _resolve_native_path(dataset, int(row[0]))
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"cv2 could not read image: {path}")
        instances = predictor(img_bgr)["instances"].to("cpu")
        n = len(instances)
        if n == 0:
            batch_y_pred.append({
                "y": torch.zeros((0, target_size, target_size), dtype=torch.uint8),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
            continue

        masks = instances.pred_masks.numpy().astype(np.uint8)  # (n, Hn, Wn)
        resized = np.zeros((n, target_size, target_size), dtype=np.uint8)
        for i in range(n):
            resized[i] = cv2.resize(
                masks[i], (target_size, target_size), interpolation=cv2.INTER_NEAREST
            )
        batch_y_pred.append({
            "y": torch.from_numpy(resized),
            "labels": torch.zeros((n,), dtype=torch.int64),
            "scores": instances.scores.to(torch.float32),
        })
    return batch_y_pred


def evaluate(predictor, dataset, test_subset, batch_size, viz_dir, viz_n_per_source=10,
             dumper=None):
    """MillionTrees streaming eval over the test subset using the Detectron2 predictor."""
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    state = TreePolygonsStreamingEvalState(dataset)
    viz_cap: dict[int, int] = {}
    viz_y_pred, viz_y_true, viz_rows = [], [], []
    for batch in test_loader:
        metadata, images, targets = batch
        preds = detectron2_predict_for_eval(predictor, dataset, metadata)
        if dumper is not None:
            dumper.update(preds, targets, metadata)
        state.update(preds, targets, metadata)
        if viz_dir is not None:
            merge_viz_samples(
                viz_cap, metadata, preds, targets,
                viz_y_pred=viz_y_pred, viz_y_true=viz_y_true, viz_rows=viz_rows,
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


def flatten_numeric_metrics(results):
    flat = {}
    for k, v in results.items():
        if isinstance(v, (int, float)):
            flat[k] = float(v)
        elif isinstance(v, torch.Tensor) and v.ndim == 0:
            flat[k] = float(v.item())
    return flat


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Train a native Detectron2 Mask R-CNN on MillionTrees TreePolygons"
    )
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--version", type=str, default=None,
                        help="MillionTrees version (default: latest in the loader's dict).")
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Detectron2 SOLVER.IMS_PER_BATCH.")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448,
                        help="MillionTrees eval/metric resolution (GT mask space).")
    parser.add_argument("--min-size-train", type=int, nargs="+",
                        default=[640, 672, 704, 736, 768, 800])
    parser.add_argument("--min-size-test", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--weights", type=str, default=_DEFAULT_WEIGHTS,
                        help="Init weights (.pkl). Falls back to COCO model_zoo URL.")
    parser.add_argument("--init-mode", type=str, default="coco",
                        choices=["coco", "box-pretrained", "box_pretrained_full", "cotrain"],
                        help="Which leg of the weak-supervision ablation to run. "
                             "coco = tree-naive control, verified tensor-for-tensor "
                             "against the released COCO Mask R-CNN. "
                             "box_pretrained_full = the whole stage-1 network from "
                             "pretrain_detectron2.py (backbone + FPN + RPN + ROI box "
                             "head; mask head stays COCO), passed via --weights. "
                             "cotrain = COCO init with the weak TreeBoxes tiles mixed "
                             "into the polygon training set in one stage. "
                             "box-pretrained = the legacy trunk-only arm "
                             "(convert_box_backbone_to_d2.py), kept for provenance.")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Explicit SOLVER.MAX_ITER, overriding --max-epochs. The "
                             "ablation sets this identically on every arm so they are "
                             "compared at matched compute.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coco-pkl", type=str, default=_DEFAULT_WEIGHTS,
                        help="Released COCO Mask R-CNN pkl used to verify --init-mode coco.")
    parser.add_argument("--boxes-dataset", type=str, default="TreeBoxes_v0.23",
                        help="TreeBoxes release supplying the weak tiles for --init-mode cotrain.")
    parser.add_argument("--weak-sources", type=str, nargs="+", default=["*unsupervised*"],
                        help="Source glob(s) counted as weak supervision when co-training.")
    parser.add_argument("--limit-weak-images", type=int, default=None,
                        help="Smoke-test knob: cap the co-training weak set.")
    parser.add_argument("--exclude-sources", type=str, nargs="*", default=[],
                        help="Source glob(s) held out of the weak set, so co-training and "
                             "stage-1 pretraining see the identical weak tiles.")
    parser.add_argument("--output-dir", type=str, default="training/polygons/outputs/detectron2")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--include-unsupervised", action="store_true")
    parser.add_argument("--eval-score-threshold", type=float, default=None,
                        help="MillionTrees evaluator score threshold (default: dataset default).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate cfg.MODEL.WEIGHTS / model_final.pth.")
    parser.add_argument("--eval-split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Subset to evaluate (default test). 'validation' = the "
                             "held-out Allen et al. 2025 + Frey et al. 2026 TLS set.")
    parser.add_argument("--complete-tiles-only", action="store_true",
                        help="Drop eval tiles that are not annotated wall to wall "
                             "(see notes/validation_ap_completeness.md).")
    add_dump_args(parser)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--comet-name", type=str, default=None)
    args = parser.parse_args()

    if args.comet_name is None:
        args.comet_name = (
            f"polygons-{args.split_scheme}-{args.init_mode}-lr{args.lr:g}-detectron2"
        )

    setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_kwargs = dict(
        root_dir=args.root_dir, version=args.version, mini=args.mini,
        split_scheme=args.split_scheme, image_size=args.image_size,
        include_unsupervised=args.include_unsupervised,
        complete_tiles_only=args.complete_tiles_only,
    )
    if args.eval_score_threshold is not None:
        dataset_kwargs["eval_score_threshold"] = args.eval_score_threshold
    dataset = get_dataset("TreePolygons", **dataset_kwargs)
    data_dir = dataset._data_dir
    images_dir = os.path.join(data_dir, "images")

    df = _filter_split_df(data_dir, args.split_scheme, args.include_unsupervised)
    prefix = f"mt_polygons_{args.split_scheme}"

    weak_records = ()
    if args.init_mode == "cotrain":
        boxes_dir = Path(args.root_dir) / args.boxes_dataset
        print(f"[cotrain] weak tiles from {boxes_dir} "
              f"sources={args.weak_sources} exclude_sources={args.exclude_sources}")
        weak_records = d2ws.load_weak_box_dicts(
            boxes_dir, args.split_scheme, split="train",
            source_patterns=tuple(args.weak_sources),
            exclude_patterns=tuple(args.exclude_sources),
            limit_images=args.limit_weak_images,
        )
        n_weak_boxes = sum(len(r["annotations"]) for r in weak_records)
        print(f"[cotrain] {len(weak_records)} weak images / {n_weak_boxes} weak boxes")

    train_name, n_train = register_datasets(df, images_dir, prefix, weak_records)
    if n_train == 0:
        print("No training images for this split; aborting.")
        return

    # The sequential arm loads the whole stage-1 network; merging it onto the COCO
    # release here (rather than trusting a pre-merged file on disk) means the log
    # carries an exact receipt of what was replaced and what stayed at COCO init.
    merge_report = None
    if args.init_mode == "box_pretrained_full":
        merged_path = os.path.join(args.output_dir, "stage1_merged_init.pkl")
        merge_report = d2ws.merge_transferable_into_coco(
            args.weights, args.coco_pkl, merged_path)
        args.weights = merged_path

    cfg = build_cfg(args, train_name, n_train, args.output_dir)

    comet_exp = None
    if args.comet:
        try:
            import comet_ml
            comet_exp = comet_ml.Experiment(
                project_name="milliontrees-polygons",
                auto_metric_logging=False,
            )
            comet_exp.set_name(args.comet_name)
            comet_exp.add_tags([
                f"split-{args.split_scheme}", "geometry-polygons",
                "stack-detectron2", f"lr-{args.lr:g}", f"init-{args.init_mode}",
            ])
            comet_exp.log_parameters(vars(args))
        except Exception as e:  # noqa: BLE001
            print(f"Comet ML logging disabled: {e}")
            comet_exp = None

    if not args.eval_only:
        trainer = AblationTrainer(cfg)
        trainer.resume_or_load(resume=False)
        # Verify after the checkpointer has run, so this describes the real init.
        if args.init_mode in ("coco", "cotrain"):
            d2ws.verify_coco_init(trainer.model, args.coco_pkl)
        elif args.init_mode == "box_pretrained_full":
            d2ws.assert_not_coco_init(trainer.model, args.coco_pkl)
            print(f"Loaded FULL pretrained network: {merge_report['replaced']} stage-1 "
                  f"tensors; {merge_report['mask_head_keys_at_coco']} mask-head tensors "
                  f"left at COCO init.")
        trainer.train()

    # ---- MillionTrees eval on the best (final) checkpoint ----
    final_weights = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if os.path.exists(final_weights):
        cfg.MODEL.WEIGHTS = final_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15
    predictor = DefaultPredictor(cfg)

    eval_subset = dataset.get_subset(args.eval_split)
    # Keep the canonical test filenames for the leaderboard; suffix any non-test
    # eval (e.g. validation) so it never clobbers the test results/viz.
    tag = args.split_scheme if args.eval_split == "test" else f"{args.split_scheme}_{args.eval_split}"
    viz_dir = os.path.join(
        args.output_dir, "viz" if args.eval_split == "test" else f"viz_{args.eval_split}")
    dumper = make_dumper(args, dataset, model="detectron2-maskrcnn", task="TreePolygons")
    results, results_str = evaluate(
        predictor, dataset, eval_subset,
        batch_size=args.batch_size, viz_dir=viz_dir, dumper=dumper,
    )
    if dumper is not None:
        dumper.close()
    print(results_str)

    results_path = os.path.join(args.output_dir, f"results_{tag}.txt")
    with open(results_path, "w") as f:
        f.write(results_str)
    print(f"Results saved to {results_path}")

    metrics_flat = flatten_numeric_metrics(results)
    json_path = os.path.join(args.output_dir, f"results_{tag}.json")
    payload = {
        "model": "detectron2-maskrcnn",
        "task": "TreePolygons",
        "split": args.split_scheme,
        "eval_split": args.eval_split,
        "metrics": metrics_flat,
        "run_metadata": {
            "stack": "detectron2",
            "init_mode": args.init_mode,
            "max_iter": cfg.SOLVER.MAX_ITER,
            "n_train_images": n_train,
            "n_weak_images": len(weak_records),
            "exclude_sources": args.exclude_sources,
            "merge_report": merge_report,
            "seed": args.seed,
            "include_unsupervised": args.include_unsupervised,
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "weights": cfg.MODEL.WEIGHTS,
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON results saved to {json_path}")

    if comet_exp is not None:
        finite = {k: v for k, v in metrics_flat.items() if math.isfinite(v)}
        comet_exp.log_metrics(finite)
        for img_path in sorted(glob.glob(os.path.join(viz_dir, "**", "*.png"), recursive=True)):
            comet_exp.log_image(img_path, name=os.path.relpath(img_path, viz_dir))
        comet_exp.end()


if __name__ == "__main__":
    main()
