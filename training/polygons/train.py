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
    precision / AP40 / merge-commission) so the numbers remain leaderboard
    comparable.

The bridge from MillionTrees splits to DeepForest is a generated annotation CSV
(``image_path``, ``geometry`` WKT, ``label``) pointed at the packaged images dir.
"""

import argparse
import fnmatch
import glob
import json
import math
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
def build_annotation_csvs(data_dir, split_scheme, output_dir, include_unsupervised,
                         train_sources=None, train_frac=1.0, seed=42):
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
        # Shrink the TRAIN split only; val/test stay identical to a full-train
        # run so the comparison against the baseline is apples to apples.
        if split_name == "train":
            if train_sources:
                patterns = [str(p).lower() for p in train_sources]
                src = rows["source"].astype(str).str.lower()
                keep = src.apply(
                    lambda v: any(fnmatch.fnmatch(v, pat) for pat in patterns))
                rows = rows[keep]
                if len(rows) == 0:
                    raise ValueError(
                        f"--train-sources {train_sources} matched no train rows. "
                        f"Available: {sorted(df[df['split'] == 'train']['source'].unique())}")
            if train_frac < 1.0:
                imgs = np.sort(rows["filename"].unique())
                n_keep = max(1, int(round(len(imgs) * train_frac)))
                rng = np.random.default_rng(seed)
                keep_imgs = set(rng.permutation(imgs)[:n_keep])
                rows = rows[rows["filename"].isin(keep_imgs)]
            print(f"[bridge] train subset: {rows['source'].nunique()} source(s), "
                  f"{rows['filename'].nunique()} images, {len(rows)} polygons")
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
    if args.warmup_epochs > 0:
        # Ramp the LR in from near-zero over the first N epochs instead of
        # hitting a transferred network with the full rate at step 0.
        #
        # This matters far more for --init-mode box_pretrained_maskrcnn than it
        # ever did for the trunk-only arm. Stage 1 trains at lr 1e-3; stage 2 runs
        # at 1e-2, a 10x jump applied from the very first iteration. When the only
        # thing transferred was a ResNet trunk that was survivable, but the
        # Mask R-CNN arm also transfers the RPN and the ROI box head -- the layers
        # that actually encode "this is a tree and this is its extent", and the
        # ones a large unwarmed step is most able to wash out. The stage-1 audit
        # named this LR discontinuity as a sufficient mechanism for the null on
        # its own (notes/weak_supervision_stage1_audit.md, section 4).
        #
        # Default is 0, so every existing run keeps its recipe and stays
        # comparable; when this is on, BOTH arms of an ablation must use it.
        overrides["train"]["scheduler"] = {
            "type": "multistepLR",
            "params": {
                "milestones": [50, 80, 90],
                "gamma": 0.1,
                "warmup_epochs": args.warmup_epochs,
                "warmup_start_factor": args.warmup_start_factor,
            },
        }

    if not args.augment:
        # Keep the geometric/photometric pipeline off for sanity runs.
        overrides["train"]["augmentations"] = []

    # Populated by whichever --train-aug branch below runs; the non-flip crop/
    # resize op only, so --val-aug-match-train can reapply it to validation
    # without also flipping validation images every epoch.
    val_match_aug = None

    if args.train_aug == "crop":
        # The config default recipe (no override needed for training itself).
        # Named here only so --val-aug-match-train has something to mirror --
        # see deepforest_polygon.yaml's own train.augmentations for the source
        # of truth this must stay in sync with.
        val_match_aug = [{
            "RandomResizedCrop": {"size": [640, 640], "scale": [0.64, 1.0], "ratio": [1.0, 1.0], "p": 1.0}
        }]

    if args.train_aug == "resize":
        # Whole-image resize instead of RandomResizedCrop: scale each image (and
        # its annotations) to a fixed ``image_size`` square so the model sees the
        # entire scene rather than a 640 crop of native-resolution imagery.
        # ``Resize`` (Kornia LongestMaxSize) preserves aspect ratio; ``PadIfNeeded``
        # squares it off so the batch stays stackable. Validation uses the same
        # scale. Pair with ``--eval-inference resize`` so train and eval run at one
        # consistent scale (the GT masks are rasterized at ``image_size`` too).
        resize_aug = [
            {"Resize": {"max_size": args.image_size}},
            {"PadIfNeeded": {"size": [args.image_size, args.image_size]}},
        ]
        if args.augment:
            overrides["train"]["augmentations"] = [
                {"HorizontalFlip": {"p": 0.5}},
                {"VerticalFlip": {"p": 0.5}},
                *resize_aug,
            ]
        else:
            overrides["train"]["augmentations"] = list(resize_aug)
        overrides["validation"]["augmentations"] = list(resize_aug)

    if args.train_aug == "nativecrop":
        # True native-resolution 640 window. The config's default ``crop`` recipe
        # uses ``RandomResizedCrop scale=[0.64,1.0]``, whose ``scale`` is an *area*
        # fraction -- on the median 2000px MillionTrees tile that keeps 64-100% of
        # the image and downsamples ~3x to 640, i.e. a whole-image resize in
        # disguise (small trees shrink to ~7px) that also mismatches the native
        # ``predict_tile`` eval scale. A plain ``RandomCrop`` instead carves a real
        # 640px window at native GSD: small trees keep their pixels and the train
        # scale matches ``--eval-inference tiled``. ``pad_if_needed`` covers the
        # ~12% of tiles smaller than 640 on a side. NOTE: with incomplete sources
        # (only ~25% of train images fully annotated) some windows are blank or
        # contain unlabeled trees scored as background -- an accepted trade-off for
        # this arm (see the crop-vs-resize comparison).
        crop_aug = [{"RandomCrop": {"size": [640, 640], "pad_if_needed": True, "p": 1.0}}]
        if args.augment:
            overrides["train"]["augmentations"] = [
                {"HorizontalFlip": {"p": 0.5}},
                {"VerticalFlip": {"p": 0.5}},
                *crop_aug,
            ]
        else:
            overrides["train"]["augmentations"] = list(crop_aug)
        val_match_aug = crop_aug
        # Validation keeps the config default (whole-image Resize) unless
        # --val-aug-match-train is set; the leaderboard eval runs tiled
        # predict_tile, not this transform, so this in-loop val_loss/map is a
        # separate scale question from the final eval.

    if args.train_aug == "annotationsafecrop":
        # Annotation-safe 640 crop: RandomSizedBBoxSafeCrop guarantees every crop
        # window contains at least one annotated polygon (by centering the crop on
        # the union of bounding boxes of all polygons in the image, with random
        # context padding). This fixes the blank-crop problem from ``nativecrop``
        # where only ~25% of images are fully annotated. The crop is resized to
        # 640 so eval with ``--eval-inference tiled`` (predict_tile at patch 640)
        # is the correct paired inference mode. Requires DeepForest's
        # ``bbox_augmentation_context`` (PR bw4sz/DeepForest#2) in the venv.
        crop_aug = [{
            "RandomSizedBBoxSafeCrop": {
                "size": [640, 640],
                "context_scale_range": [1.0, 2.0],
                "erosion_rate": 0.0,
                "p": 1.0,
            }
        }]
        if args.augment:
            overrides["train"]["augmentations"] = [
                {"HorizontalFlip": {"p": 0.5}},
                {"VerticalFlip": {"p": 0.5}},
                *crop_aug,
            ]
        else:
            overrides["train"]["augmentations"] = list(crop_aug)
        val_match_aug = crop_aug
        # Validation keeps the config default unless --val-aug-match-train is
        # set; leaderboard eval uses predict_tile regardless.

    if args.val_aug_match_train and val_match_aug is not None:
        # Mirror the train crop/resize op (minus flips) onto validation so the
        # in-loop val_loss/map Comet logs are scored at the same pixel scale
        # the model trains on, instead of the config default whole-image
        # Resize(max_size=1024). Doesn't touch --eval-inference/predict_tile,
        # which already runs at the training scale for the final leaderboard
        # numbers -- this only affects the per-epoch metrics DeepForest logs
        # during fit().
        overrides["validation"]["augmentations"] = list(val_match_aug)

    # Both ablation arms must start from torchvision's COCO Mask R-CNN, so
    # model.name has to be None *at construction time*: create_model() takes the
    # load_model() branch for any non-None name, and the base config default is
    # "weecology/deepforest-tree". That path calls MaskRCNN.from_pretrained(),
    # which builds a cold-start shell and then loads a RetinaNet checkpoint whose
    # keys never match the Mask R-CNN — so nothing lands on the backbone and the
    # "COCO" arm trains a ResNet-50 from scratch (measured: 0/318 backbone.body
    # tensors equal COCO_V1, all 53 BN running_mean still 0). Nulling the name
    # routes through models.maskrcnn.Model.create_model(pretrained=None) ->
    # MaskRCNN(backbone_weights="COCO_V1"), i.e. real COCO backbone+FPN+heads.
    overrides["model"] = {"name": None}

    cfg = utilities.load_config(config_name=args.config, overrides=overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Box-pretrained backbone (weak-supervision ablation, Table 6)
# --------------------------------------------------------------------------- #
def assert_coco_trunk(model):
    """Raise unless the Mask R-CNN backbone+FPN is exactly torchvision COCO_V1.

    Table 6 is only interpretable if the control arm really is a generic COCO
    backbone. Jobs 38750867/68/69 were not: the config default
    model.name="weecology/deepforest-tree" sent construction through
    MaskRCNN.from_pretrained(), which silently left the ResNet randomly
    initialized (see build_config). The published +0.083 AP40 gain was therefore
    box-pretrained-vs-*random*, not box-pretrained-vs-COCO.

    Guarding beats commenting: compare every backbone tensor against a freshly
    constructed torchvision maskrcnn_resnet50_fpn_v2 COCO_V1 -- the same factory
    deepforest.models.maskrcnn uses -- and refuse to train on a mismatch.
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

    reference = maskrcnn_resnet50_fpn_v2(weights="COCO_V1").backbone.state_dict()
    actual = model.model.backbone.state_dict()

    missing = sorted(set(reference) - set(actual))
    extra = sorted(set(actual) - set(reference))
    if missing or extra:
        raise RuntimeError(
            "Backbone does not have the torchvision Mask R-CNN v2 structure "
            f"(missing={missing[:5]}, unexpected={extra[:5]})."
        )

    mismatched = [
        key for key, value in reference.items()
        if not torch.equal(value, actual[key].detach().cpu().to(value.dtype))
    ]
    if mismatched:
        raise RuntimeError(
            f"UNPRETRAINED INIT: {len(mismatched)}/{len(reference)} backbone/FPN "
            "tensors differ from torchvision Mask R-CNN v2 COCO_V1 - this is not "
            "a COCO baseline. Most likely config.model.name is non-None (default "
            "'weecology/deepforest-tree'), which leaves the ResNet at random init. "
            f"First mismatches: {mismatched[:5]}"
        )

    print(
        f"Verified COCO init: all {len(reference)} backbone/FPN tensors match "
        "torchvision Mask R-CNN v2 COCO_V1"
    )
    return len(reference)


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
    print(f"Loaded {loaded} backbone keys; {len(missing)} keys left at COCO init.")
    return {"missing": missing, "loaded_backbone_keys": loaded}


MASK_HEAD_PREFIXES = ("roi_heads.mask_head.", "roi_heads.mask_predictor.")


def _extract_box_pretrained_maskrcnn(checkpoint_path):
    """Load a stage-1 Mask R-CNN export; return its Mask R-CNN-keyed tensors.

    Rejects the two exports that would *silently* under-transfer if they were
    accepted here. Both live in the same output directory as the right one and
    differ only by filename, which is exactly the mistake worth failing loudly on:

    * ``box_backbone_<split>.pt`` -- trunk only. ``load_state_dict(strict=False)``
      would happily take its 318 keys, leave RPN and the ROI box head at COCO,
      and produce a run indistinguishable from the old ``box_pretrained`` arm
      while claiming to be the new one.
    * ``box_network_<split>.pt`` -- a RetinaNet. Its keys are ``head.*`` /
      ``anchor_generator.*``; none match, so the load would be a no-op.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    prefix = "model."
    mapped = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not mapped:
        raise ValueError(
            f"No 'model.'-prefixed keys found in {checkpoint_path}."
        )

    has_rpn = any(k.startswith("rpn.") for k in mapped)
    has_box_head = any(k.startswith("roi_heads.box_predictor.") for k in mapped)
    if not (has_rpn and has_box_head):
        raise ValueError(
            f"{checkpoint_path} is not a Mask R-CNN stage-1 export: it carries "
            f"no {'rpn.*' if not has_rpn else 'roi_heads.box_predictor.*'} "
            "tensors. --init-mode box_pretrained_maskrcnn needs the "
            "'maskrcnn_transferable_<split>.pt' file written by "
            "pretrain_backbone_for_polygons.py --arch maskrcnn. "
            "(box_backbone_*.pt is trunk-only and box_network_*.pt is a "
            "RetinaNet; use --init-mode box_pretrained for the former.)"
        )
    leaked = sorted(k for k in mapped if k.startswith(MASK_HEAD_PREFIXES))
    if leaked:
        raise ValueError(
            f"{checkpoint_path} contains {len(leaked)} mask-head tensors "
            f"(e.g. {leaked[0]}). Stage 1 trains on boxes with the mask branch "
            "disabled, so it cannot have learned a mask head -- this export is "
            "not from the arch this flag expects."
        )
    return mapped


def apply_box_pretrained_maskrcnn(deepforest_module, checkpoint_path):
    """Transplant a whole box-pretrained Mask R-CNN except its mask head.

    The polygon analogue of the box->box whole-network arm. Where
    :func:`apply_box_pretrained_backbone` moves only ``backbone.body.*`` -- and
    in practice only 265 of that body's 318 tensors, leaving the FPN, the RPN
    and both ROI heads at COCO or random init -- this moves all 404 tensors a
    box-supervised stage 1 is able to learn. The 28 mask-head tensors stay at
    COCO by construction: no weak box carries mask supervision.

    That truncation was the standing explanation for the polygon null (see
    notes/weak_supervision_pretraining_table.md section 3). Closing it is the
    experiment.
    """
    state = _extract_box_pretrained_maskrcnn(checkpoint_path)
    missing, unexpected = deepforest_module.model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(
            f"Unexpected keys when loading the stage-1 Mask R-CNN: {unexpected[:5]}"
        )
    unexplained = [k for k in missing if not k.startswith(MASK_HEAD_PREFIXES)]
    if unexplained:
        raise ValueError(
            f"{len(unexplained)} tensors were left un-transferred and are not "
            f"mask-head tensors: {unexplained[:5]}. The stage-1 export does not "
            "cover this architecture -- refusing to train a partially "
            "initialized ablation arm."
        )
    # Count the untouched tensors from the module, not from ``missing``:
    # ``_NormBase._load_from_state_dict`` silently defaults an absent
    # ``num_batches_tracked`` to 0 instead of reporting it, so ``missing``
    # undercounts the mask head by one tensor per BN layer.
    held = sum(
        1 for k in deepforest_module.model.state_dict()
        if k.startswith(MASK_HEAD_PREFIXES)
    )
    print(
        f"Loaded {len(state)} stage-1 Mask R-CNN tensors "
        f"(backbone + FPN + RPN + ROI box head); {held} mask-head tensors "
        "left at COCO init."
    )
    return {"loaded_keys": len(state), "mask_head_keys_at_coco": held}


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


def resized_predict_for_eval(model, dataset, metadata, *, device):
    """Whole-image-resize inference for one batch, in MillionTrees eval format.

    The scale-matched counterpart to ``tiled_predict_for_eval`` for models trained
    with ``--train-aug resize``. Each native image is stretched to the dataset's
    ``image_size`` square -- the exact space the GT masks are rasterized in (see
    ``TreePolygons`` ``cv2.resize``/``A.Resize`` to ``image_size``) -- run through
    the Mask R-CNN once, and the predicted instance masks are vectorised to
    polygons that already live in the metric space (scale 1, no projection).
    """
    target_size = dataset.image_size
    if not isinstance(metadata, torch.Tensor):
        metadata = torch.as_tensor(metadata)

    batch_y_pred = []
    for row in metadata:
        path = _resolve_native_path(dataset, int(row[0]))
        with Image.open(path) as im:
            arr = np.asarray(
                im.convert("RGB").resize((target_size, target_size), Image.BILINEAR)
            ).astype("float32")
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            pred = model.model(img_t)[0]

        gdf = utilities.format_geometry(
            pred, geom_type=utilities.determine_geometry_type(pred)
        )
        if gdf is None or len(gdf) == 0:
            batch_y_pred.append({
                "y": torch.zeros((0, target_size, target_size), dtype=torch.uint8),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
            continue

        masks = _rasterize_geoms_to_masks(list(gdf.geometry), 1.0, 1.0, target_size)
        batch_y_pred.append({
            "y": torch.from_numpy(masks),
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
    eval_inference="tiled",
    viz_n_per_source=4,
):
    """Run MillionTrees test-set evaluation on a trained DeepForest model.

    ``eval_mode``:
        - ``stream`` (default): update metrics per batch (lower peak memory).
        - ``legacy``: accumulate full pred/true lists then call ``dataset.eval``.

    ``eval_inference``:
        - ``tiled`` (default): native-resolution ``predict_tile`` at the training
          patch size (matches ``--train-aug crop``).
        - ``resize``: whole-image resize to ``image_size`` (matches
          ``--train-aug resize``). Train/eval scale must match.
    """
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    model.eval()
    if eval_inference == "resize":
        # Whole-image predict runs the Mask R-CNN forward directly (no trainer);
        # move the loaded model onto the eval device once.
        model.to(device)

        def predict_batch(meta):
            return resized_predict_for_eval(model, dataset, meta, device=device)
    elif eval_inference == "tiled":
        ensure_predict_trainer(model)

        def predict_batch(meta):
            return tiled_predict_for_eval(model, dataset, meta)
    else:
        raise ValueError(
            f"Unknown eval_inference: {eval_inference!r}; use 'tiled' or 'resize'."
        )

    if eval_mode == "legacy":
        all_y_pred, all_y_true = [], []
        for batch in test_loader:
            metadata, images, targets = batch
            preds = predict_batch(metadata)
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
        preds = predict_batch(metadata)
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
        save_top_k=-1 if getattr(args, "keep_all_checkpoints", False) else 1,
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
    parser.add_argument("--version", type=str, default=None,
                        help="MillionTrees dataset version (e.g. 0.19). Defaults to the "
                             "latest key in the loader's _versions_dict. Pin an older "
                             "version to isolate data changes from recipe changes.")
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
    parser.add_argument("--eval-inference", type=str, default="tiled",
                        choices=["tiled", "resize"],
                        help="tiled: native-resolution predict_tile at the training patch "
                             "size (matches --train-aug crop). resize: whole-image resize to "
                             "--image-size (matches --train-aug resize). Scale must match training.")
    parser.add_argument("--train-aug", type=str, default="crop",
                        choices=["crop", "resize", "nativecrop", "annotationsafecrop"],
                        help="crop: RandomResizedCrop 640 (area scale 0.64-1.0; on big tiles a "
                             "downsample-resize in disguise). resize: whole-image Resize+Pad to "
                             "--image-size, no cropping (use with --eval-inference resize). "
                             "nativecrop: true 640px RandomCrop at native GSD (use with "
                             "--eval-inference tiled). annotationsafecrop: RandomSizedBBoxSafeCrop "
                             "that guarantees each crop contains at least one annotation (use with "
                             "--eval-inference tiled).")
    parser.add_argument("--val-aug-match-train", action=argparse.BooleanOptionalAction, default=False,
                         help="Mirror --train-aug's crop/resize op (minus flips) onto the "
                              "in-loop validation set instead of the config default whole-image "
                              "Resize(max_size=1024). Scores the Comet-logged val_loss/map at the "
                              "same pixel scale training runs at. Off by default so existing "
                              "recipes are unaffected; does not change --eval-inference/"
                              "predict_tile, which already matches scale for the leaderboard eval.")
    parser.add_argument("--init-mode", type=str, default="coco",
                        choices=["coco", "box_pretrained",
                                 "box_pretrained_maskrcnn"],
                        help="coco: DeepForest COCO-initialized Mask R-CNN. "
                             "box_pretrained: overwrite the ResNet backbone with a box checkpoint "
                             "(trunk only, 265 tensors -- RetinaNet stage 1). "
                             "box_pretrained_maskrcnn: transplant a whole box-pretrained "
                             "Mask R-CNN except its mask head (404 of 432 tensors), which "
                             "requires a stage 1 run with --arch maskrcnn.")
    parser.add_argument("--box-backbone-checkpoint", type=str, default=None,
                        help="DeepForest/box checkpoint when --init-mode=box_pretrained.")
    parser.add_argument("--include-unsupervised", action="store_true",
                        help="Include unsupervised sources in the training annotations.")
    parser.add_argument("--data-scope", type=str, default="subset",
                        choices=["subset", "full"],
                        help="Tag for experiment aggregation (subset vs full data pull).")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup over the first N epochs. 0 (default) "
                             "keeps the historical recipe. Strongly recommended for "
                             "--init-mode box_pretrained_maskrcnn, which transfers the "
                             "RPN and ROI box head into a stage running at 10x the "
                             "learning rate that trained them. Both arms of an "
                             "ablation must use the same value.")
    parser.add_argument("--warmup-start-factor", type=float, default=0.001,
                        help="LR multiplier at step 0 of the warmup ramp.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sources", type=str, nargs="+", default=None,
                        help="Restrict the TRAIN split to these source names "
                             "(fnmatch wildcards allowed). Val/test untouched, so the "
                             "run stays comparable to the full-train baseline.")
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="Randomly keep this fraction of TRAIN images (after "
                             "--train-sources), seeded by --seed.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the config's train augmentations. --no-augment strips them.")
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=False,
                        help="EarlyStopping on val_loss (off by default; the OAM recipe trains full epochs).")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--keep-all-checkpoints", action="store_true",
                        help="Keep a checkpoint at every validation epoch "
                             "(save_top_k=-1) instead of only the best-val_loss one. "
                             "Use when a downstream pass selects the checkpoint on the "
                             "held-out validation split (val_loss overfits by ~epoch 9 "
                             "on the polygon Mask R-CNN).")
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
        version=args.version,
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
        include_unsupervised=args.include_unsupervised,
    )
    data_dir = polygon_dataset._data_dir

    train_csv, val_csv, images_dir = build_annotation_csvs(
        data_dir, args.split_scheme, args.output_dir, args.include_unsupervised,
        train_sources=args.train_sources, train_frac=args.train_frac, seed=args.seed
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
    if args.init_mode == "coco":
        assert_coco_trunk(model)
    elif args.init_mode == "box_pretrained":
        if args.box_backbone_checkpoint is None:
            raise ValueError(
                "--box-backbone-checkpoint is required for --init-mode=box_pretrained"
            )
        # Verify the base is COCO *before* the swap, so a failure here is
        # unambiguously the base and not the exported backbone.
        assert_coco_trunk(model)
        details = apply_box_pretrained_backbone(model, args.box_backbone_checkpoint)
        init_details.update({
            "box_backbone_checkpoint": str(Path(args.box_backbone_checkpoint).resolve()),
            "loaded_backbone_keys": details["loaded_backbone_keys"],
        })
    elif args.init_mode == "box_pretrained_maskrcnn":
        if args.box_backbone_checkpoint is None:
            raise ValueError(
                "--box-backbone-checkpoint is required for "
                "--init-mode=box_pretrained_maskrcnn (pass the "
                "maskrcnn_transferable_<split>.pt written by "
                "pretrain_backbone_for_polygons.py --arch maskrcnn)"
            )
        assert_coco_trunk(model)
        details = apply_box_pretrained_maskrcnn(model, args.box_backbone_checkpoint)
        init_details.update({
            "box_backbone_checkpoint": str(Path(args.box_backbone_checkpoint).resolve()),
            "loaded_keys": details["loaded_keys"],
            "mask_head_keys_at_coco": details["mask_head_keys_at_coco"],
        })

    checkpoint_cb = build_trainer(model, args, log_root)
    model.trainer.fit(model)

    # Under DDP all ranks return from fit(); only rank 0 runs the final
    # MillionTrees eval and writes results (avoids redundant work / file races).
    if not model.trainer.is_global_zero:
        return

    # Capture loggers before checkpoint reload replaces the model object.
    train_loggers = model.trainer.loggers if model.trainer else []

    # Score the best checkpoint with the MillionTrees metrics.
    best_path = checkpoint_cb.best_model_path or None
    if best_path and os.path.exists(best_path):
        print(f"\n=== Loading best checkpoint: {best_path} ===")
        model = deepforest.load_from_checkpoint(best_path)
    else:
        print("\n=== No checkpoint saved; evaluating the in-memory model ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_subset = train_subset_or_test(polygon_dataset, args.debug_overfit)
    viz_dir = os.path.join(args.output_dir, "viz")
    results, results_str = evaluate(
        model,
        polygon_dataset,
        test_subset,
        batch_size=args.batch_size,
        device=device,
        viz_dir=viz_dir,
        eval_mode=args.eval_mode,
        eval_inference=args.eval_inference,
    )
    print(results_str)

    if train_loggers:
        exp = train_loggers[0].experiment
        safe = {k: float(v.item() if hasattr(v, "item") else v)
                for k, v in results.items()
                if isinstance(v, (int, float)) or (hasattr(v, "ndim") and v.ndim == 0)}
        exp.log_metrics({k: v for k, v in safe.items() if math.isfinite(v)})
        for img_path in sorted(glob.glob(os.path.join(viz_dir, "**", "*.png"), recursive=True)):
            exp.log_image(img_path, name=os.path.relpath(img_path, viz_dir))

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
