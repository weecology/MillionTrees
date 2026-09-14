"""Pretrain DeepForest on TreeBoxes and export backbone weights for polygon Mask R-CNN."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.datasets.milliontrees_dataset import MillionTreesSubset
from training.boxes.comet_viz import CometDetectionViz, sample_fixed_batch
from training.boxes.train import _AdaptCollate, assert_coco_trunk, evaluate

# The Mask R-CNN arm of this script pretrains the *polygon* architecture on weak
# boxes, so it has to build the identical model stage 2 builds -- same config
# file, same COCO verification.
POLYGON_CONFIG = str(
    Path(__file__).resolve().parents[1] / "polygons" / "deepforest_polygon.yaml"
)

# The only tensors a box-supervised stage 1 cannot donate: weak boxes carry no
# mask supervision, so the mask head is never trained and stays at COCO init.
MASK_HEAD_PREFIXES = ("roi_heads.mask_head.", "roi_heads.mask_predictor.")


def _flatten_numeric_metrics(results):
    flat = {}
    for k, v in results.items():
        if isinstance(v, (int, float)):
            flat[k] = float(v)
        elif isinstance(v, torch.Tensor) and v.ndim == 0:
            flat[k] = float(v.item())
    return flat


def build_supervised_val_subset(root_dir, split_scheme, n_images, seed, verbose=False):
    """Hold out a slice of the SUPERVISED train split to validate the pretraining.

    Why this exists: the pretraining stage trains on ``include_sources=['*unsupervised*']``,
    and every unsupervised row is in the *train* split (46,823/0 train/test). So the
    stage had **no validation set at all** -- ``has_val`` was False, which meant no
    ModelCheckpoint, no early stopping, no scheduler (DeepForest's
    ``configure_optimizers`` only attaches one when ``validation.csv_file`` is set), and
    an export of whatever the *last* epoch happened to be. The stage's own
    ``results_<split>.txt`` came out all-``nan`` for the same reason, so nobody could
    tell whether 20 epochs on 7.1M boxes had learned anything.

    We validate on *supervised* data because that is the question the ablation actually
    asks: does pretraining on weak boxes make a better tree detector on human-labelled
    trees? Loss on held-out *unsupervised* boxes would only measure how well the model
    reproduces its own pseudo-labels.

    The slice is taken from the supervised **train** split, never test or validation:
    the manuscript validation split is 59 images from 2 sources (too narrow to monitor
    on), and using test would contaminate the benchmark. Sampling is stratified by
    source and seeded, so the same images are held out across arms.

    NOTE: these images stay in stage 2's training set. That is intentional -- stage 1 and
    stage 2 are separate models, and shrinking stage 2's train split would change the
    baseline this ablation is measured against.
    """
    sup = get_dataset(
        "TreeBoxes",
        download=False,
        root_dir=root_dir,
        split_scheme=split_scheme,
        include_unsupervised=False,
        verbose=verbose,
    )
    train_idx = np.where(sup.split_array == sup.split_dict["train"])[0]
    if len(train_idx) == 0:
        raise RuntimeError("Supervised train split is empty; cannot build a val set.")
    source_of = {int(i): int(sup._metadata_array[int(i), 1]) for i in train_idx}
    by_source = {}
    for i in train_idx:
        by_source.setdefault(source_of[int(i)], []).append(int(i))

    rng = np.random.default_rng(seed)
    n_images = min(n_images, len(train_idx))
    per_source = max(1, n_images // max(1, len(by_source)))
    picked = []
    for src, idxs in sorted(by_source.items()):
        take = min(per_source, len(idxs))
        picked.extend(rng.choice(idxs, size=take, replace=False).tolist())
    # top up to the requested size from whatever is left
    remaining = sorted(set(int(i) for i in train_idx) - set(picked))
    if len(picked) < n_images and remaining:
        extra = rng.choice(remaining, size=min(n_images - len(picked), len(remaining)),
                           replace=False)
        picked.extend(int(i) for i in extra)
    picked = np.sort(np.array(sorted(set(picked))[:n_images]))

    subset = MillionTreesSubset(sup, picked, None, sup.geometry_name)
    names = sup._metadata_map.get("source_id", [])
    counts = {}
    for i in picked:
        sid = int(sup._metadata_array[int(i), 1])
        counts[names[sid] if sid < len(names) else str(sid)] = counts.get(
            names[sid] if sid < len(names) else str(sid), 0) + 1
    print(f"[supervised-val] {len(picked)} held-out SUPERVISED train images "
          f"from {len(counts)} sources (seed {seed})")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"[supervised-val]   {k}: {counts[k]}")
    return sup, subset


def export_backbone_weights(model, output_path):
    """Export inner torchvision backbone weights using Lightning checkpoint key prefix.

    Trunk only (backbone.body.*, 265 tensors). This is the format the polygon
    (box -> Mask R-CNN) ablation consumes: there the ResNet trunk is the only
    architecturally compatible piece, because RetinaNet's dense heads have no
    counterpart in Mask R-CNN's RPN/ROI/mask heads. Kept unchanged so that
    ablation keeps working; the box -> box ablation uses export_full_network().
    """
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


def export_full_network(model, output_path):
    """Export the whole RetinaNet: trunk + FPN + detection heads (301 tensors).

    For the same-geometry (box -> box) ablation both stages are the identical
    DeepForest RetinaNetHub with num_classes=1, so every tensor is transferable
    and there is no reason to truncate. Exporting the trunk alone discarded the
    FPN (reset to COCO) and both detection heads (reset to *random* init, since
    RetinaNetHub always builds fresh heads -- COCO's 91-class head does not fit
    a 1-class model), i.e. it threw away every layer where "this is a tree, and
    this is its extent" is represented.
    """
    state_dict = {
        f"model.{k}": v.detach().cpu() for k, v in model.model.state_dict().items()
    }
    if not state_dict:
        raise ValueError("Empty state_dict; cannot export transferable weights.")
    torch.save({"state_dict": state_dict}, output_path)
    return len(state_dict)


def disable_mask_branch(model):
    """Make the polygon Mask R-CNN a pure box detector for the pretraining stage.

    Stage 1 trains on the 6.4M unsupervised *boxes*, which carry no mask
    supervision at all. Torchvision's ``RoIHeads.forward`` reads ``t["masks"]``
    from every target whenever ``has_mask()`` is true, so a box-only target dict
    would raise ``KeyError`` -- and the obvious workaround, synthesising
    rectangular masks from the boxes, is worse than useless: it would train the
    mask head to emit rectangles and then transfer that into stage 2.

    Setting ``mask_roi_pool = None`` makes ``has_mask()`` false, which gates the
    entire mask block in both training and inference. The mask head and
    predictor keep their COCO weights, receive no gradient, and are excluded
    from the export (see :func:`export_maskrcnn_transferable`).
    """
    roi_heads = model.model.roi_heads
    if not roi_heads.has_mask():
        raise RuntimeError(
            "Expected a Mask R-CNN with a live mask branch; got a model whose "
            "roi_heads.has_mask() is already False."
        )
    roi_heads.mask_roi_pool = None
    if roi_heads.has_mask():
        raise RuntimeError("Failed to disable the mask branch.")
    print("[stage1] mask branch disabled: training on boxes only, no loss_mask")


def use_box_task_metrics(model):
    """Point DeepForest's validation metrics at boxes rather than masks.

    ``deepforest.main.__init__`` picks its metric set from ``self.model.task``,
    and ``MaskRCNN.task`` is ``"polygon"`` -- which builds a segmentation-mAP
    metric and a ``validation_step`` branch that decodes ``panoptic_masks`` from
    every target. Stage 1's targets are boxes, so that branch would fail on the
    first validation batch.

    ``task = "box"`` is not a hack here, it is the literal truth about stage 1:
    a Mask R-CNN with its mask branch off, supervised by boxes, validated on
    boxes. Rebuilding the three metrics mirrors main.py's box branch exactly, so
    ``box_recall`` / ``box_precision`` mean the same thing they mean in the
    RetinaNet arm -- which matters, because ``--ckpt-monitor box_recall`` selects
    the exported weights on one of them.
    """
    from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision

    from deepforest.metrics import RecallPrecision

    model.model.task = "box"
    max_dets = [1, 10, max(100, model.config.detections_per_img)]
    model.iou_metric = IntersectionOverUnion(
        class_metrics=True, iou_threshold=model.config.validation.iou_threshold
    )
    model.mAP_metric = MeanAveragePrecision(
        backend="faster_coco_eval", max_detection_thresholds=max_dets
    )
    model.precision_recall_metric = RecallPrecision(
        iou_threshold=model.config.validation.iou_threshold,
        label_dict=model.label_dict,
        task="box",
    )
    print("[stage1] validation metrics switched to the box task "
          f"(IoU {model.config.validation.iou_threshold}, max_dets {max_dets})")


def export_maskrcnn_transferable(model, output_path):
    """Export every Mask R-CNN tensor stage 2 can consume: all but the mask head.

    This is the polygon analogue of :func:`export_full_network`, and the whole
    point of the Mask R-CNN arm. Because stage 1 and stage 2 are now the *same*
    architecture (``deepforest_polygon.yaml``'s Mask R-CNN v2, num_classes=1),
    the transfer is no longer confined to the ResNet trunk:

        backbone.body 318 + backbone.fpn 48 + rpn 8
        + roi_heads.box_head 26 + roi_heads.box_predictor 4  =  404 of 432

    Only the 28-tensor mask head is held back, and it has to be -- stage 1 never
    trains it (see :func:`disable_mask_branch`).

    The RetinaNet stage 1 could only ever donate ``backbone.body``, because
    RetinaNet has no RPN, no ROI box head and no mask head to give. Worse, its
    export carried 265 keys into a 318-key body, leaving 53 at COCO. That
    truncation is the standing explanation for the polygon null, and closing it
    is exactly what this arm tests.
    """
    state_dict = {
        f"model.{k}": v.detach().cpu()
        for k, v in model.model.state_dict().items()
        if not k.startswith(MASK_HEAD_PREFIXES)
    }
    if not state_dict:
        raise ValueError("Empty state_dict; cannot export transferable weights.")
    held = sum(
        1 for k in model.model.state_dict() if k.startswith(MASK_HEAD_PREFIXES)
    )
    torch.save({"state_dict": state_dict}, output_path)
    return len(state_dict), held


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
    parser.add_argument("--comet-name", type=str, default=None,
                        help="Explicit Comet experiment name. If omitted, defaults to "
                             "boxes-<split>-pretrain-lr<lr> following the CLAUDE.md scheme.")
    parser.add_argument(
        "--viz-images",
        type=int,
        default=8,
        help="Per-epoch Comet detection overlays: this many fixed tiles from the "
             "unsupervised TRAIN split and this many from the supervised val "
             "holdout, each logged with ground-truth and prediction layers so the "
             "epoch slider shows whether the boxes are converging. 0 disables. "
             "Requires --comet.",
    )
    parser.add_argument(
        "--viz-score-thresh",
        type=float,
        default=0.05,
        help="Confidence floor for boxes drawn in the Comet overlays. Low on "
             "purpose: stage-1 models top out around score 0.30, so the usual 0.5 "
             "would render an empty image for a model that is in fact predicting.",
    )
    parser.add_argument(
        "--viz-nms-thresh",
        type=float,
        default=0.4,
        help="NMS IoU for the overlays only (never for the reported metrics). "
             "DeepForest's inference default is 0.05, which discards any box "
             "overlapping a better one by >5% IoU -- on tiles averaging 143 "
             "overlapping crowns that hides most of what the model found.",
    )
    parser.add_argument(
        "--viz-max-boxes",
        type=int,
        default=300,
        help="Cap on boxes drawn per layer, highest-confidence first. Keeps the "
             "Comet viewer responsive on tiles with hundreds of detections.",
    )
    parser.add_argument(
        "--supervised-val-images",
        type=int,
        default=512,
        help="Hold out this many SUPERVISED train images to validate the pretraining "
             "on (stratified by source, seeded). 0 restores the old behaviour of no "
             "validation at all -- which meant no checkpoint selection, no scheduler, "
             "and a blind last-epoch export. See build_supervised_val_subset().",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "multistep-warmup"],
        help="LR schedule for the pretraining stage. DeepForest's default is None, "
             "i.e. ConstantLR -- which is why train_loss plateaued around epoch 5 and "
             "then oscillated for 15 more epochs instead of converging.",
    )
    parser.add_argument(
        "--val-every-n-epochs",
        type=int,
        default=1,
        help="Run validation every N epochs. MUST be set: DeepForest's create_trainer "
             "passes check_val_every_n_epoch=config.validation.val_accuracy_interval, "
             "whose default is 20 -- so a 20-epoch run validates exactly ONCE, at the "
             "last epoch, and ModelCheckpoint only ever sees a single value (making "
             "'best' identical to 'last' and defeating the point of monitoring).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient-norm clipping. Stage 2 (train.py) has always passed --clip 1.0; "
             "the pretraining stage had none, which is why lr 1e-2 diverged to NaN from "
             "epoch 0 in job 39879762_2 while stage 2 runs at that same LR happily.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Linear LR warmup (multistep-warmup only).")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop when supervised val_loss stops improving. 0 disables.")
    parser.add_argument("--ckpt-monitor", type=str, default="val_loss",
                        help="Metric the ModelCheckpoint selects on. The exported "
                             "weights come from the BEST epoch by this metric, not the "
                             "last -- the OOD stage-1 run 39614225 ended at "
                             "classification loss 0.651 having hit 0.529 at epoch 7, "
                             "and exported the worse model because nothing was watching.")
    parser.add_argument("--ckpt-mode", type=str, default="min", choices=["min", "max"])
    parser.add_argument(
        "--init-mode",
        type=str,
        default="deepforest",
        choices=["deepforest", "coco"],
        help="Starting weights for the pretraining stage. "
             "deepforest = weecology/deepforest-tree (NEON-pretrained: the exported "
             "trunk then carries tree supervision, so the downstream ablation cannot "
             "attribute anything to the unsupervised boxes); "
             "coco = torchvision COCO RetinaNet, verified tensor-for-tensor before "
             "training, so whatever the trunk learns comes from the boxes alone.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="retinanet",
        choices=["retinanet", "maskrcnn"],
        help="Architecture for the pretraining stage. "
             "retinanet = DeepForest's box model; its only architecturally "
             "transferable piece is the ResNet trunk, because RetinaNet has no "
             "RPN/ROI/mask head to donate to a Mask R-CNN. "
             "maskrcnn = build the POLYGON architecture (deepforest_polygon.yaml) "
             "and train it on weak boxes with the mask branch off, so backbone + "
             "FPN + RPN + ROI box head (404/432 tensors) all transfer into stage 2. "
             "This is the polygon analogue of the box->box whole-network arm.",
    )
    parser.add_argument(
        "--trainable-backbone-layers",
        type=int,
        default=None,
        help="ResNet stages left unfrozen (maskrcnn only). The polygon config "
             "pins 3 (Detectron2 FREEZE_AT=2: stem + layer1 frozen), which is "
             "right for stage 2 but wrong for stage 1 -- freezing the stem there "
             "means the weak boxes can never reach it. Defaults to 5 (train "
             "everything) for --arch maskrcnn, so stage 2 then freezes those "
             "layers at weak-label-adapted values instead of at COCO.",
    )
    args = parser.parse_args()

    if args.arch == "maskrcnn" and args.init_mode != "coco":
        parser.error(
            "--arch maskrcnn requires --init-mode coco. The 'deepforest' init "
            "loads weecology/deepforest-tree, a NEON-trained RetinaNet whose keys "
            "do not match a Mask R-CNN, and which would contaminate the ablation "
            "with human tree labels even if they did."
        )
    if args.trainable_backbone_layers is None:
        args.trainable_backbone_layers = 5 if args.arch == "maskrcnn" else None

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

    if len(train_subset) == 0:
        raise RuntimeError("No training samples for this split; cannot pretrain.")

    # Validation for the pretraining stage comes from held-out SUPERVISED train
    # images, not from box_dataset: with include_sources=['*unsupervised*'] every
    # row lands in train, so box_dataset.get_subset("test") is EMPTY and the stage
    # silently ran blind. See build_supervised_val_subset() for the full rationale.
    sup_dataset = None
    val_subset = None
    if args.supervised_val_images > 0:
        sup_dataset, val_subset = build_supervised_val_subset(
            args.root_dir,
            args.split_scheme,
            args.supervised_val_images,
            args.seed,
        )
    else:
        print("[supervised-val] DISABLED (--supervised-val-images 0): no validation "
              "loss, no checkpoint selection, last epoch exported.")

    # Real DataLoaders (not a custom iterable) so Lightning can inject a
    # DistributedSampler under DDP and shard data across GPUs. The collate_fn
    # translates MillionTrees batches into DeepForest's (images, targets, paths).
    adapt_collate = _AdaptCollate(train_subset.collate, box_dataset._filename_id_to_code)
    has_val = val_subset is not None and len(val_subset) > 0
    val_collate = (
        _AdaptCollate(val_subset.collate, sup_dataset._filename_id_to_code)
        if has_val else None
    )

    train_adapted = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=adapt_collate,
        num_workers=args.num_workers,
    )
    val_adapted = (
        DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_collate,
            num_workers=args.num_workers,
        )
        if has_val
        else None
    )

    # DeepForest attaches an LR scheduler ONLY when validation.csv_file is set
    # (see deepforest.main.deepforest.configure_optimizers). existing_val_dataloader
    # short-circuits val_dataloader() before the path is ever read, so this
    # placeholder enables scheduling without touching the filesystem -- the same
    # trick train.py uses for train.csv_file.
    train_cfg = {"epochs": args.max_epochs, "lr": args.lr}
    if args.scheduler == "none":
        # Set explicitly rather than relying on a default: deepforest_polygon.yaml
        # ships a multistepLR schedule (milestones [50, 80, 90]) that the maskrcnn
        # arm would otherwise inherit without anyone asking for it.
        train_cfg["scheduler"] = {"type": None}
    elif args.scheduler == "cosine":
        train_cfg["scheduler"] = {
            "type": "cosine",
            "params": {"T_max": args.max_epochs, "eta_min": args.lr * 0.01},
        }
    elif args.scheduler == "multistep-warmup":
        train_cfg["scheduler"] = {
            "type": "multistepLR",
            "params": {
                "milestones": [int(args.max_epochs * 0.6), int(args.max_epochs * 0.85)],
                "gamma": 0.1,
                "warmup_epochs": args.warmup_epochs,
                "warmup_start_factor": 0.001,
            },
        }

    config_args = {
        "train": train_cfg,
        "validation": {
            "root_dir": str(box_dataset._data_dir / "images"),
            "csv_file": "existing_val_dataloader" if has_val else None,
        },
        "batch_size": args.batch_size,
        "devices": args.gpus,
        "accelerator": args.accelerator,
        "workers": args.num_workers,
    }

    # See training/boxes/train.py: model.name must be None at construction time,
    # because deepforest.__init__ -> create_model() downloads the configured hub
    # checkpoint ("weecology/deepforest-tree" by default) before this script gets
    # control. num_classes/label_dict normally arrive with that checkpoint.
    if args.init_mode == "coco":
        config_args["model"] = {"name": None}
        config_args["num_classes"] = 1
        config_args["label_dict"] = {"Tree": 0}

    if args.arch == "maskrcnn":
        # Build the model stage 2 will build. Passing the polygon config file
        # itself (rather than copying knobs across) is what guarantees the two
        # state_dicts line up tensor-for-tensor -- score/nms thresholds,
        # rpn_post_nms_top_n, detections_per_img and the v2 head shapes all come
        # from one place. Only trainable_backbone_layers is deliberately
        # different; see the --trainable-backbone-layers help.
        config_args["maskrcnn"] = {
            "trainable_backbone_layers": args.trainable_backbone_layers
        }
        model = df_main.deepforest(
            config=POLYGON_CONFIG,
            config_args=config_args,
            existing_train_dataloader=train_adapted,
            existing_val_dataloader=val_adapted,
        )
    else:
        model = df_main.deepforest(
            config_args=config_args,
            existing_train_dataloader=train_adapted,
            existing_val_dataloader=val_adapted,
        )

    if args.arch == "maskrcnn":
        # Same guarantee the RetinaNet arm gets from assert_coco_trunk(), but
        # against maskrcnn_resnet50_fpn_v2 COCO_V1. Reused from the polygon
        # trainer so both stages verify their init the identical way.
        from training.polygons.train import assert_coco_trunk as assert_coco_trunk_maskrcnn

        assert_coco_trunk_maskrcnn(model)
        disable_mask_branch(model)
        use_box_task_metrics(model)
    elif args.init_mode == "coco":
        assert_coco_trunk(model)
    else:
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

            comet_name = args.comet_name or (
                f"boxes-{args.split_scheme}-pretrain-{args.arch}-lr{args.lr}")
            loggers.append(_SafeCometLogger(
                project_name="milliontrees-pretrain",
                experiment_name=comet_name,
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
            monitor=args.ckpt_monitor,
            mode=args.ckpt_mode,
            save_last=True,
            save_top_k=1,
        )
        callbacks.append(checkpoint_cb)
        # NOTE: do NOT add a LearningRateMonitor here. DeepForest's create_trainer()
        # already appends one whenever validation is active, and it guards on
        # `logger is not None` -- so passing logger=False (as this script used to)
        # satisfies that check and then trips Lightning's "no logger" assertion.
        # We pass logger=None below instead, which makes DeepForest build a CSVLogger.
        if args.early_stop_patience > 0:
            callbacks.append(pl.callbacks.EarlyStopping(
                monitor=args.ckpt_monitor,
                mode=args.ckpt_mode,
                patience=args.early_stop_patience,
            ))

    # Detection overlays. Scalar curves cannot distinguish "learning slowly" from
    # "fitting noise" on 7.1M weak boxes; the pictures can. Built here (after the
    # loaders and the logger, before create_trainer) so the same fixed tiles are
    # reused for the whole run. See training/boxes/comet_viz.py.
    viz_cb = None
    if args.viz_images > 0 and loggers:
        panels = {}
        train_panel = sample_fixed_batch(
            train_subset, adapt_collate, args.viz_images, args.seed)
        if train_panel is not None:
            panels["train-weak"] = train_panel
        if has_val:
            val_panel = sample_fixed_batch(
                val_subset, val_collate, args.viz_images, args.seed)
            if val_panel is not None:
                panels["val-supervised"] = val_panel
        if panels:
            viz_cb = CometDetectionViz(
                loggers[0].experiment,
                panels,
                score_thresh=args.viz_score_thresh,
                nms_thresh=args.viz_nms_thresh,
                max_boxes=args.viz_max_boxes,
            )
            callbacks.append(viz_cb)
            print(f"[viz] Comet detection overlays enabled for panels "
                  f"{sorted(panels)} ({args.viz_images} tiles each, "
                  f"score>={args.viz_score_thresh}, nms={args.viz_nms_thresh})")
    elif args.viz_images > 0:
        print("[viz] Comet detection overlays skipped: --viz-images requires --comet")

    trainer_kwargs = {}
    if args.grad_clip and args.grad_clip > 0:
        trainer_kwargs["gradient_clip_val"] = args.grad_clip
    if has_val:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches if args.limit_val_batches else 1.0
        trainer_kwargs["num_sanity_val_steps"] = 2
        # Two independent gates both default to 20 and both have to be opened:
        #   1. Lightning's check_val_every_n_epoch -> whether validation_step runs
        #      at all (this is what produces val_loss).
        #   2. config.validation.val_accuracy_interval -> whether
        #      on_validation_epoch_end computes box_recall / box_precision / mAP.
        # trainer_args.update(kwargs) in create_trainer means our value wins for (1).
        trainer_kwargs["check_val_every_n_epoch"] = args.val_every_n_epochs
        model.config.validation.val_accuracy_interval = args.val_every_n_epochs
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None and has_val:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    # logger=None (not False) so DeepForest falls back to a CSVLogger under
    # config.log_root: the per-epoch val_loss lands in metrics.csv next to the run,
    # which is the whole point of this change -- "did the loss drop?" must be
    # answerable from the output directory, without Comet.
    model.create_trainer(
        callbacks=callbacks,
        logger=loggers[0] if loggers else None,
        **trainer_kwargs,
    )

    # Score the untrained (COCO-init) model on the same held-out supervised images
    # BEFORE any pretraining, so the run reports a delta instead of an isolated
    # number. Without this baseline "AP40 0.21 after pretraining" is uninterpretable.
    baseline_str = ""
    if has_val:
        print("\n=== BASELINE: COCO-init model on held-out supervised images (pre-training) ===")
        base_results, baseline_str = evaluate(
            model, sup_dataset, val_subset,
            batch_size=args.batch_size, max_batches=args.eval_max_batches,
        )
        print(f"[baseline] AP40={base_results.get('AP40', {}).get('AP40_avg', float('nan'))}")

    # Step 0 = the initialisation, so the Comet epoch slider starts from "what
    # COCO weights predict on a tree tile" rather than from epoch 0's output.
    if viz_cb is not None:
        viz_cb.log_panels(model, step=0)

    model.trainer.fit(model)

    # Per-epoch loss table, so the .out log is self-documenting and nobody has to
    # open Comet to answer "did the loss actually drop?".
    hist = getattr(model.trainer, "logged_metrics", {})
    print("\n=== stage-1 final logged metrics ===")
    for k in sorted(hist):
        try:
            print(f"  {k}: {float(hist[k]):.4f}")
        except (TypeError, ValueError):
            pass

    best_path = None
    if checkpoint_cb is not None:
        best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
        print(f"\n=== checkpoint selection ({args.ckpt_monitor}, mode={args.ckpt_mode}) ===")
        print(f"  best : {checkpoint_cb.best_model_path}")
        print(f"  score: {checkpoint_cb.best_model_score}")
        print(f"  last : {checkpoint_cb.last_model_path}")
        if best_path:
            model = df_main.deepforest.load_from_checkpoint(best_path, weights_only=False)

    # Final eval on the held-out SUPERVISED images. The old code scored
    # box_dataset's test subset, which for an unsupervised-only selection is empty,
    # so every results_<split>.txt from this stage came out `nan`.
    eval_dataset = sup_dataset if has_val else box_dataset
    eval_subset = val_subset if has_val else box_dataset.get_subset("test")
    results, results_str = evaluate(
        model,
        eval_dataset,
        eval_subset,
        batch_size=args.batch_size,
        max_batches=args.eval_max_batches,
    )
    if baseline_str:
        results_str = (
            "########## BASELINE: COCO init, BEFORE pretraining ##########\n"
            + baseline_str
            + "\n########## AFTER pretraining on unsupervised boxes ##########\n"
            + results_str
        )
    txt_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(results_str)

    backbone_path = os.path.join(args.output_dir, f"box_backbone_{args.split_scheme}.pt")
    num_keys = export_backbone_weights(model, backbone_path)

    if args.arch == "maskrcnn":
        # 404/432 tensors: everything except the untrained mask head. This is
        # what --init-mode box_pretrained_maskrcnn consumes in stage 2.
        network_path = os.path.join(
            args.output_dir, f"maskrcnn_transferable_{args.split_scheme}.pt")
        num_network_keys, num_held = export_maskrcnn_transferable(model, network_path)
        print(f"[stage1] held back {num_held} mask-head tensors (never trained)")
    else:
        # Whole network for the box -> box ablation (see export_full_network).
        network_path = os.path.join(args.output_dir, f"box_network_{args.split_scheme}.pt")
        num_network_keys = export_full_network(model, network_path)

    json_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.json")
    payload = {
        "model": "box-pretrain",
        "task": "TreeBoxes",
        "split": args.split_scheme,
        "metrics": _flatten_numeric_metrics(results),
        "run_metadata": {
            "arch": args.arch,
            "trainable_backbone_layers": args.trainable_backbone_layers,
            "seed": args.seed,
            "data_scope": args.data_scope,
            "include_unsupervised": args.include_unsupervised,
            "mini": args.mini,
            "best_checkpoint_path": best_path,
            "backbone_export_path": str(Path(backbone_path).resolve()),
            "exported_backbone_keys": num_keys,
            "network_export_path": str(Path(network_path).resolve()),
            "exported_network_keys": num_network_keys,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Backbone export written to {backbone_path} ({num_keys} tensors, trunk only)")
    if args.arch == "maskrcnn":
        print(f"Transferable Mask R-CNN export written to {network_path} "
              f"({num_network_keys} tensors: backbone + FPN + RPN + ROI box head; "
              "mask head excluded)")
    else:
        print(f"Full network export written to {network_path} "
              f"({num_network_keys} tensors)")
    print(f"Metadata written to {json_path}")


if __name__ == "__main__":
    main()
