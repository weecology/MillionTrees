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
# Box-pretrained backbone (weak-supervision ablation)
# ---------------------------------------------------------------------------

def _extract_box_pretrained_backbone(checkpoint_path):
    """Load a DeepForest box checkpoint; return backbone keys for the RetinaNet."""
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


def assert_coco_trunk(model):
    """Raise unless the RetinaNet backbone+FPN is exactly torchvision COCO_V1.

    The ablation is only interpretable if the "clean" arm has never seen a tree
    label. That is easy to get wrong: DeepForest's config default is
    model.name = "weecology/deepforest-tree", and deepforest.__init__ ->
    create_model() -> load_model() pulls it before train.py runs, so simply
    *not* calling load_model() leaves a NEON-trained detector in place. Jobs
    38840964 and 39089781 both shipped as "clean COCO" that way and reproduced
    the contaminated baseline's AP40 to three decimals.

    Guarding beats commenting: compare every backbone tensor against a freshly
    constructed torchvision COCO_V1 and refuse to train on a mismatch.
    """
    from torchvision.models.detection import retinanet_resnet50_fpn

    reference = retinanet_resnet50_fpn(weights="COCO_V1").backbone.state_dict()
    actual = model.model.backbone.state_dict()

    missing = sorted(set(reference) - set(actual))
    extra = sorted(set(actual) - set(reference))
    if missing or extra:
        raise RuntimeError(
            "Backbone does not have the torchvision RetinaNet structure "
            f"(missing={missing[:5]}, unexpected={extra[:5]})."
        )

    mismatched = [
        key for key, value in reference.items()
        if not torch.equal(value, actual[key].detach().cpu().to(value.dtype))
    ]
    if mismatched:
        raise RuntimeError(
            f"CONTAMINATED INIT: {len(mismatched)}/{len(reference)} backbone/FPN "
            "tensors differ from torchvision RetinaNet COCO_V1 — this is not a "
            "tree-naive baseline. Most likely config.model.name still points at "
            "a Hugging Face checkpoint (default 'weecology/deepforest-tree'). "
            f"First mismatches: {mismatched[:5]}"
        )

    print(
        f"Verified COCO init: all {len(reference)} backbone/FPN tensors match "
        "torchvision RetinaNet COCO_V1"
    )
    return len(reference)


def _extract_box_pretrained_network(checkpoint_path):
    """Load a box-pretrained checkpoint; return every RetinaNet tensor."""
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
            f"No DeepForest weights found in {checkpoint_path}. "
            "Expected keys prefixed with 'model.'."
        )
    return mapped


def apply_box_pretrained_backbone(model, checkpoint_path):
    """Overwrite the RetinaNet ResNet trunk only (backbone.body.*).

    Retained for the polygon (box -> Mask R-CNN) ablation, where the trunk is
    genuinely the only transferable piece. For box -> box use
    apply_box_pretrained_network(): both stages are the same architecture, so
    truncating here silently resets the FPN to COCO and the detection heads to
    random init, discarding everything the pretraining learned about trees.
    """
    backbone_state = _extract_box_pretrained_backbone(checkpoint_path)
    missing, unexpected = model.model.load_state_dict(backbone_state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys when loading box backbone: {unexpected}")
    loaded = sum(1 for k in backbone_state if k.startswith("backbone.body."))
    if loaded == 0:
        raise ValueError("No backbone.body keys loaded from box pretrained checkpoint.")
    print(f"Loaded {loaded} backbone keys; {len(missing)} head keys left at COCO init.")
    return {"missing": len(missing), "loaded_backbone_keys": loaded}


def apply_box_pretrained_network(model, checkpoint_path):
    """Transfer the ENTIRE pretrained RetinaNet: trunk + FPN + detection heads.

    Both the pretraining stage and this fine-tune build the identical
    DeepForest RetinaNetHub with num_classes=1, so all 301 tensors are
    shape-compatible and load strictly. Anything less than a strict full load
    means the two stages disagree about architecture, which would silently
    reintroduce the truncation this function exists to remove — so refuse.
    """
    state = _extract_box_pretrained_network(checkpoint_path)
    expected = model.model.state_dict()

    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not cover the RetinaNet exactly: "
            f"{len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}). "
            "A trunk-only export (box_backbone_*.pt) will trip this — the "
            "box -> box ablation needs box_network_*.pt."
        )

    model.model.load_state_dict(state, strict=True)
    groups = {
        "backbone.body": sum(1 for k in state if k.startswith("backbone.body.")),
        "backbone.fpn": sum(1 for k in state if k.startswith("backbone.fpn.")),
        "head": sum(1 for k in state if k.startswith("head.")),
    }
    print(
        f"Loaded FULL pretrained network: {len(state)} tensors "
        f"(trunk {groups['backbone.body']}, FPN {groups['backbone.fpn']}, "
        f"heads {groups['head']}); nothing left at COCO/random init."
    )
    return {"loaded_keys": len(state), **groups}


def apply_imagenet_backbone(model):
    """Replace the COCO backbone with an ImageNet trunk + randomly-init FPN.

    DeepForest's create_model() hardcodes backbone_weights="COCO_V1", so there
    is no config route to a non-COCO init; the backbone is swapped after
    construction instead. torchvision's retinanet_resnet50_fpn(weights=None)
    keeps weights_backbone=ResNet50_Weights.IMAGENET1K_V1, which is exactly the
    conventional detection baseline: ImageNet trunk, fresh FPN, fresh heads.
    The heads are already random here — RetinaNetHub builds them for
    num_classes=1, so COCO's 91-class head never loads on any arm.
    """
    from torchvision.models.detection import retinanet_resnet50_fpn

    replacement = retinanet_resnet50_fpn(weights=None).backbone
    if replacement.out_channels != model.model.backbone.out_channels:
        raise RuntimeError(
            "ImageNet backbone out_channels "
            f"({replacement.out_channels}) does not match the constructed model "
            f"({model.model.backbone.out_channels}); the head would be invalid."
        )
    model.model.backbone = replacement
    print("Swapped backbone to ImageNet trunk + randomly-initialized FPN.")
    return replacement


def assert_imagenet_trunk(model):
    """Raise unless the trunk is torchvision ResNet50 IMAGENET1K_V1 and the FPN is not COCO.

    The mirror of assert_coco_trunk(), for the same reason: this arm is only
    interpretable if it has seen no detection pretraining at all. Note that
    ImageNet and COCO trunks agree on 223/265 tensors — COCO RetinaNet is itself
    ImageNet-initialized and freezes the early layers — so "differs from COCO"
    is far too weak a check on its own. Verify against ImageNet positively.
    """
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.models.detection import retinanet_resnet50_fpn

    actual = model.model.backbone.state_dict()
    reference = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()

    body = [k for k in actual if k.startswith("body.")]
    if not body:
        raise RuntimeError(
            f"Backbone has no body.* tensors (keys: {sorted(actual)[:5]})."
        )
    mismatched = [
        k for k in body
        if not torch.equal(
            reference[k[len("body."):]],
            actual[k].detach().cpu().to(reference[k[len("body."):]].dtype),
        )
    ]
    if mismatched:
        raise RuntimeError(
            f"NOT an ImageNet trunk: {len(mismatched)}/{len(body)} tensors differ "
            "from torchvision ResNet50 IMAGENET1K_V1. First mismatches: "
            f"{mismatched[:5]}"
        )

    # The FPN must NOT be COCO's: that is the piece the COCO arm actually gets.
    coco_fpn = retinanet_resnet50_fpn(weights="COCO_V1").backbone.state_dict()
    fpn = [k for k in actual if k.startswith("fpn.")]
    fpn_same = [
        k for k in fpn
        if torch.equal(coco_fpn[k], actual[k].detach().cpu().to(coco_fpn[k].dtype))
    ]
    if fpn_same:
        raise RuntimeError(
            f"CONTAMINATED INIT: {len(fpn_same)}/{len(fpn)} FPN tensors still match "
            "COCO — the backbone swap did not take effect."
        )

    print(
        f"Verified ImageNet init: all {len(body)} trunk tensors match torchvision "
        f"ResNet50 IMAGENET1K_V1; all {len(fpn)} FPN tensors are freshly initialized "
        "(no COCO detection pretraining)."
    )
    return len(body)


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
        help="Read the full TreeBoxes_v* release instead of the supervised-only "
             "zip, which also MIXES the weakly-labelled sources into the TRAIN "
             "split (Weinstein et al. 2018 unsupervised, 41,691 images / 6.4M "
             "pseudo-boxes, plus Young et al. 2025 weak supervised, 11,739 / "
             "99,626). This is the co-training arm of the weak-supervision "
             "ablation: weak and human labels optimized together in one stage, "
             "as opposed to the pretrain-then-finetune route through "
             "pretrain_backbone_for_polygons.py. Test/validation are unaffected "
             "— every weak row is in train — so the score stays comparable to "
             "the supervised-only baseline.",
    )
    parser.add_argument(
        "--exclude-sources", type=str, nargs="+", default=None,
        help="Drop these source names (fnmatch wildcards, case-insensitive) from "
             "ALL splits. Used by the co-training arm to hold the weak data "
             "identical to what stage-1 pretraining saw: that stage trains on "
             "include_sources=['*unsupervised*'], i.e. Weinstein et al. 2018 only, "
             "so co-training passes --exclude-sources '*Young*' to keep the extra "
             "weak source out and leave pretraining-vs-co-training as the single "
             "difference. Excluding a source that only appears in train (both weak "
             "sources do) leaves the eval set untouched.",
    )
    parser.add_argument(
        "--remove-incomplete",
        action="store_true",
        help="Train only on complete=True (exhaustively annotated) sources. "
             "Filters the TRAIN split only; the test set is always left "
             "unchanged so results are comparable to the full-train baseline.",
    )
    parser.add_argument(
        "--train-sources", type=str, nargs="+", default=None,
        help="Restrict the TRAIN split to these source names (fnmatch wildcards "
             "allowed, e.g. 'Weecology*'). Validation/test are untouched, so the "
             "run is scored on the same eval set as the full-train baseline. "
             "Used by the weak-supervision data-scaling ablation.",
    )
    parser.add_argument(
        "--train-frac", type=float, default=1.0,
        help="Randomly keep this fraction of TRAIN images (after --train-sources). "
             "Seeded by --seed so every arm of an ablation sees the identical subset.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=1.0,
        help="Fraction of the test split used for the per-epoch validation metric "
             "that drives early stopping / checkpoint selection. The final scored "
             "evaluation always uses the full test split.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the --train-frac subsample (and torch/numpy global seeding).",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/boxes/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        help="Lightning accelerator (use 'cpu' for debugging).",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument(
        "--gradient-clip-val", type=float, default=0.0,
        help="Clip gradient norm to this value (0 = off, Lightning default). "
             "The box-pretrained arm diverged to NaN at lr 0.01 (job 36058068_0), "
             "so the pretraining ablation runs with 1.0.",
    )
    parser.add_argument("--comet", action="store_true",
                        help="Log to Comet ML (requires .comet.config or COMET_API_KEY)")
    parser.add_argument("--comet-name", type=str, default=None,
                        help="Comet experiment name. Defaults to "
                             "boxes-<split>-<init>-lr<lr> (see CLAUDE.md naming scheme).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Limit to 2 train/val batches and 1 epoch for local testing")
    parser.add_argument(
        "--init-mode",
        type=str,
        default="deepforest",
        choices=["deepforest", "coco", "imagenet", "box_pretrained", "box_pretrained_full"],
        help="Backbone initialization: "
             "deepforest = load weecology/deepforest-tree (NEON-pretrained, not a clean baseline); "
             "coco = torchvision COCO RetinaNet only (clean baseline, no NEON exposure); "
             "imagenet = ImageNet ResNet50 trunk with a freshly initialized FPN and heads, "
             "i.e. no detection pretraining of any kind — the cold-start floor for asking "
             "what COCO and the weak-label pretraining each add; "
             "box_pretrained = COCO RetinaNet with only the ResNet trunk swapped from "
             "--box-backbone-checkpoint (FPN reset to COCO, heads reset to random; kept for "
             "reproducibility of the 39129721 runs and for the polygon ablation); "
             "box_pretrained_full = the ENTIRE pretrained RetinaNet (trunk + FPN + heads) from "
             "a box_network_*.pt export — the correct same-geometry transfer.",
    )
    parser.add_argument(
        "--box-backbone-checkpoint",
        type=str,
        default=None,
        help="Path to a .pt export from pretrain_backbone_for_polygons.py: "
             "box_backbone_*.pt for --init-mode box_pretrained, "
             "box_network_*.pt for --init-mode box_pretrained_full.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.max_epochs = 1
        args.early_stop_patience = 1

    os.makedirs(args.output_dir, exist_ok=True)

    # TreeBoxes only excludes '*unsupervised*' by default when exclude_sources is
    # left as None; passing any list at all REPLACES that default. Re-add it here
    # unless --include-unsupervised was asked for, so --exclude-sources can never
    # smuggle the weak sources into a supervised-only arm.
    exclude_sources = args.exclude_sources
    if exclude_sources is not None and not args.include_unsupervised:
        exclude_sources = list(exclude_sources) + ["*unsupervised*"]

    box_dataset = get_dataset(
        "TreeBoxes",
        download=args.download,
        mini=args.mini,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        include_unsupervised=args.include_unsupervised,
        remove_incomplete=args.remove_incomplete,
        train_sources=args.train_sources,
        exclude_sources=exclude_sources,
    )

    # Seed before get_subset: --train-frac draws its images with np.random, and
    # every arm of an ablation must train on the identical subsample.
    pl.seed_everything(args.seed, workers=True)

    train_subset = box_dataset.get_subset("train", frac=args.train_frac)
    test_subset = box_dataset.get_subset("test")
    # The val loader only drives early stopping / checkpoint selection; the final
    # MillionTrees eval below always scores the FULL test split. Subsampling it
    # keeps a small-train run from spending most of its wall clock validating
    # 3k images after every 7-step epoch.
    val_subset = (test_subset if args.val_frac >= 1.0
                  else box_dataset.get_subset("test", frac=args.val_frac))
    print(f"Train images: {len(train_subset)} | val (monitoring) images: "
          f"{len(val_subset)} | test (scored) images: {len(test_subset)}")

    if len(train_subset) == 0:
        print("No training samples for this split; skipping training.")
        return

    # Real DataLoaders (not a custom iterable) so Lightning can inject a
    # DistributedSampler under DDP and shard data across GPUs. The collate_fn
    # translates MillionTrees batches into DeepForest's (images, targets, paths).
    adapt_collate = _AdaptCollate(train_subset.collate, box_dataset._filename_id_to_code)
    has_val = len(val_subset) > 0

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
            collate_fn=adapt_collate,
            num_workers=args.num_workers,
        )
        if has_val
        else None
    )

    # Build DeepForest model and load pretrained weights
    config_args = {
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
    }

    # Both clean arms must start from torchvision COCO, so model.name has to be
    # None *at construction time*: deepforest.__init__ -> create_model() takes
    # the load_model() branch for any non-None name, and the config default is
    # "weecology/deepforest-tree". Nulling it here routes through
    # RetinaNetHub(backbone_weights="COCO_V1"), i.e. COCO trunk+FPN with fresh
    # heads, which is what the ablation claims to compare. num_classes and
    # label_dict come from the hub checkpoint on the deepforest path, so they
    # must be supplied explicitly once nothing is downloaded.
    if args.init_mode in ("coco", "imagenet", "box_pretrained", "box_pretrained_full"):
        config_args["model"] = {"name": None}
        config_args["num_classes"] = 1
        config_args["label_dict"] = {"Tree": 0}

    model = df_main.deepforest(
        config_args=config_args,
        existing_train_dataloader=train_adapted,
        existing_val_dataloader=val_adapted,
    )

    # deepforest = load weecology/deepforest-tree (NEON-exposed; not a clean ablation baseline).
    # coco = torchvision COCO RetinaNet only (clean COCO baseline, no NEON exposure).
    # box_pretrained = COCO RetinaNet with backbone replaced by NEON-pretrained weights
    #   (COCO heads + NEON backbone — clean pair for comparing against coco init).
    if args.init_mode == "deepforest":
        model.load_model("weecology/deepforest-tree")
    elif args.init_mode == "coco":
        assert_coco_trunk(model)
    elif args.init_mode == "imagenet":
        # Confirm the base really was COCO before the swap, so a later guard
        # failure is unambiguously the swap and not an unexpected starting model.
        assert_coco_trunk(model)
        apply_imagenet_backbone(model)
        assert_imagenet_trunk(model)
    elif args.init_mode == "box_pretrained":
        if not args.box_backbone_checkpoint:
            raise ValueError("--box-backbone-checkpoint is required with --init-mode box_pretrained")
        # Verify the base is COCO *before* the swap, so a failure here is
        # unambiguously the base and not the exported backbone.
        assert_coco_trunk(model)
        apply_box_pretrained_backbone(model, args.box_backbone_checkpoint)
    elif args.init_mode == "box_pretrained_full":
        if not args.box_backbone_checkpoint:
            raise ValueError(
                "--box-backbone-checkpoint is required with --init-mode box_pretrained_full"
            )
        # Same ordering rationale as above: confirm the base is COCO before the
        # pretrained weights land on top of it, so a guard failure is
        # unambiguously the base and not the export.
        assert_coco_trunk(model)
        apply_box_pretrained_network(model, args.box_backbone_checkpoint)

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

            comet_name = args.comet_name or f"boxes-{args.split_scheme}-{args.init_mode}-lr{args.lr:g}"
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
    if args.gradient_clip_val > 0:
        trainer_kwargs["gradient_clip_val"] = args.gradient_clip_val
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
