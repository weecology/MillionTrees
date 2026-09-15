"""Stage 1 of the TreePoints weak-supervision ablation: continue-pretrain the
released TreeFormer point checkpoint on AutoArborist (Beery et al. 2022).

This mirrors the box/polygon weak-supervision arms (``training/boxes/
pretrain_backbone_for_polygons.py`` + ``training/weak_supervision/``). The
question is the point analogue of the box->box whole-network arm:

    does inserting a pretraining stage on the ~22k AutoArborist images
    (``Beery et al. 2022 unsupervised``, municipal-inventory points loosely
    aligned to NAIP imagery) beat fine-tuning straight from the released
    ``weecology/deepforest-tree-point`` checkpoint?

Stage 2 is the *unchanged* ``training/points/train.py`` run on the supervised
MillionTrees points train split. The only difference between the two arms is
whether stage 2's ``--checkpoint`` points at this script's HF export
(``<output-dir>/hf_checkpoint``) or at ``weecology/deepforest-tree-point``.

Why AutoArborist counts as weak supervision: in v0.24 ``Beery et al. 2022``
ships as ``Beery et al. 2022 unsupervised`` -- its labels are municipal tree
inventory records, not image annotations (the TCD canopy filter discards ~50%
of the raw points), and it is already barred from evaluation
(``TRAIN_ONLY_SOURCES``). Every AutoArborist row is in the *train* split for
both split schemes, exactly like the unsupervised boxes.

Validation for this stage comes from a held-out, source-stratified, seeded
slice of the *supervised* MillionTrees points train split -- never test, never
the manuscript validation split. We validate on supervised points because that
is the question the ablation asks (does pretraining on noisy inventory points
make a better detector of human-labelled trees?); loss on held-out AutoArborist
points would only measure how well the model reproduces municipal records.
Rationale copied verbatim from ``build_supervised_val_subset`` in the box arm.

Runs in the frozen ``.venv-treeformer`` (DeepForest ``treeformer-training``
branch), called directly -- NOT ``uv run`` (see memory points-wrong-branch-collapse).
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import pytorch_lightning as pl
import torch

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.datasets.milliontrees_dataset import MillionTreesSubset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader

from training.points.train import MillionTreesPointBatchAdapter, evaluate

# AutoArborist ships under this source name in v0.24 (see TreePoints._versions_dict
# and TRAIN_ONLY_SOURCES). fnmatch pattern, case-insensitive in the loader.
AUTOARBORIST_SOURCE_GLOB = "Beery et al. 2022*"


def build_supervised_val_subset(root_dir, split_scheme, image_size, n_images, seed):
    """Hold out a source-stratified, seeded slice of the SUPERVISED points train
    split to validate the pretraining on.

    Never touches test or the manuscript validation split. These images stay in
    stage 2's training set -- stage 1 and stage 2 are separate models and
    shrinking stage 2's train split would move the baseline this ablation is
    measured against. Mirrors training/boxes/pretrain_backbone_for_polygons.py.
    """
    sup = get_dataset(
        "TreePoints",
        download=False,
        root_dir=root_dir,
        split_scheme=split_scheme,
        image_size=image_size,
        include_unsupervised=False,
    )
    train_idx = np.where(sup.split_array == sup.split_dict["train"])[0]
    if len(train_idx) == 0:
        raise RuntimeError("Supervised points train split is empty; cannot build a val set.")

    by_source = {}
    for i in train_idx:
        src = int(sup._metadata_array[int(i), 1])
        by_source.setdefault(src, []).append(int(i))

    rng = np.random.default_rng(seed)
    n_images = min(n_images, len(train_idx))
    per_source = max(1, n_images // max(1, len(by_source)))
    picked = []
    for src, idxs in sorted(by_source.items()):
        take = min(per_source, len(idxs))
        picked.extend(rng.choice(idxs, size=take, replace=False).tolist())
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
        key = names[sid] if sid < len(names) else str(sid)
        counts[key] = counts.get(key, 0) + 1
    print(f"[supervised-val] {len(picked)} held-out SUPERVISED points train images "
          f"from {len(counts)} sources (seed {seed})")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"[supervised-val]   {k}: {counts[k]}")
    return sup, subset


def main():
    parser = argparse.ArgumentParser(
        description="Continue-pretrain TreeFormer on AutoArborist, export an HF "
                    "checkpoint for training/points/train.py --checkpoint."
    )
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="within-distribution",
                        choices=["within-distribution", "out-of-distribution", "crossgeometry"],
                        help="AutoArborist is all-train for every scheme, so stage 1 "
                             "trains on identical rows; the scheme only selects which "
                             "split's supervised train the val holdout is drawn from.")
    parser.add_argument("--init-checkpoint", type=str,
                        default="weecology/deepforest-tree-point",
                        help="Starting weights. Default = the released TreeFormer point "
                             "checkpoint; the control arm starts stage 2 from this same "
                             "checkpoint, so the ablation isolates the AutoArborist stage.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Continued-pretraining LR. Kept low (1e-5) so the stage "
                             "adapts the released checkpoint to more imagery rather than "
                             "overwriting it with noisy inventory points.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=896,
                        help="Match the current leaderboard points hyperparameters "
                             "(the _896_isolated runs) so the backbone adapts at the "
                             "resolution stage 2 fine-tunes and eval scores at.")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--supervised-val-images", type=int, default=512,
                        help="Held-out SUPERVISED points train images for stage-1 "
                             "model selection (stratified by source, seeded). 0 "
                             "disables validation entirely (blind last-epoch export).")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--ckpt-monitor", type=str, default="val_loss",
                        choices=["point_recall", "val_loss"])
    parser.add_argument("--output-dir", type=str,
                        default="training/points/weak_supervision_outputs/pretrain")
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument("--score-integration-radius", type=int, default=2)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--comet-name", type=str, default=None,
                        help="Defaults to points-<split>-autoarborist-pretrain-lr<lr>.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="2 train/val batches, 1 epoch, tiny holdout.")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_epochs = 1
        args.early_stop_patience = 1
        args.supervised_val_images = min(args.supervised_val_images, 16)

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Stage-1 training data: AutoArborist only -------------------------------
    autoarborist = get_dataset(
        "TreePoints",
        download=False,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
        include_unsupervised=True,
        include_sources=[AUTOARBORIST_SOURCE_GLOB],
    )
    selected = sorted(set(autoarborist._source_id_to_code.values()))
    if not selected or any("beery" not in s.lower() for s in selected):
        raise RuntimeError(
            f"Expected only AutoArborist (Beery et al. 2022) rows; got {selected}. "
            "Check the source name in this dataset version."
        )
    train_subset = autoarborist.get_subset("train")
    if len(train_subset) == 0:
        raise RuntimeError("No AutoArborist train images for this split.")
    print(f"[stage1] AutoArborist train images: {len(train_subset)}  sources: {selected}")

    # --- Stage-1 validation: held-out SUPERVISED points train -------------------
    sup_dataset = val_subset = None
    if args.supervised_val_images > 0:
        sup_dataset, val_subset = build_supervised_val_subset(
            args.root_dir, args.split_scheme, args.image_size,
            args.supervised_val_images, args.seed,
        )
    else:
        print("[supervised-val] DISABLED: no checkpoint selection, last epoch exported.")

    train_loader = get_train_loader(
        "standard", train_subset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    has_val = val_subset is not None and len(val_subset) > 0
    val_loader = (
        get_eval_loader("standard", val_subset, batch_size=args.batch_size,
                        num_workers=args.num_workers)
        if has_val else None
    )

    train_adapted = MillionTreesPointBatchAdapter(
        train_loader, autoarborist._filename_id_to_code)
    val_adapted = (
        MillionTreesPointBatchAdapter(val_loader, sup_dataset._filename_id_to_code)
        if has_val else None
    )

    # --- Model: same construction as training/points/train.py ------------------
    _df_point_distance_threshold = int(round(0.067 * args.image_size))
    config_args = {
        "architecture": "treeformer",
        "train": {"epochs": args.max_epochs, "lr": args.lr},
        "validation": {
            "root_dir": str(autoarborist._data_dir / "images"),
            "val_accuracy_interval": 1,
        },
        "point": {
            "distance_threshold": _df_point_distance_threshold,
            # conf/point_pretrain.yaml -- the config the released checkpoint was
            # trained under (enforce_count=False keeps the count loss live).
            # Re-applied post-load below; config alone is ignored after load_model.
            "enforce_count": False,
            "losses": ["count", "ot", "density_l1"],
            "mae_weight": 0.025,
            "ot_weight": 0.1,
            "density_l1_weight": 0.05,
            "score_integration_radius": 2,
        },
        "batch_size": args.batch_size,
        "devices": args.gpus,
        "accelerator": args.accelerator,
        "workers": args.num_workers,
    }

    model = df_main.deepforest(
        config_args=config_args,
        existing_train_dataloader=train_adapted,
        existing_val_dataloader=val_adapted,
    )
    print(f"[stage1] loading init checkpoint: {args.init_checkpoint}")
    model.load_model(args.init_checkpoint)

    model.model.score_thresh = args.score_thresh
    model.model.score_integration_radius = args.score_integration_radius

    # loss-preset=pretrain MUST be applied on the submodule after load_model:
    # TreeFormer.create_model() forwards only score_thresh/score_integration_radius
    # to from_pretrained(), so the checkpoint's saved _hub_mixin_config otherwise
    # wins and count_loss stays identically 0. See training/points/train.py.
    m = model.model
    m.enforce_count = False
    m.losses = ["count", "ot", "density_l1"]
    m.active_losses = set(m.losses)
    m.mae_weight = 0.025
    m.ot_weight = 0.1
    m.density_l1_weight = 0.05
    m.update_config()
    print(f"[stage1] loss-preset=pretrain APPLIED: enforce_count={m.enforce_count} "
          f"losses={sorted(m.active_losses)}")

    loggers = []
    if args.comet:
        try:
            from pytorch_lightning.loggers import CometLogger

            comet_name = (args.comet_name
                          or f"points-{args.split_scheme}-autoarborist-pretrain-lr{args.lr:g}")
            loggers.append(CometLogger(
                project_name="milliontrees-pretrain",
                name=comet_name,
                tags=[f"split-{args.split_scheme}", "geometry-points", "treeformer",
                      "autoarborist-pretrain"],
            ))
        except Exception as e:
            print(f"Comet ML logging disabled: {e}")

    callbacks = []
    checkpoint_cb = None
    if has_val:
        monitor_mode = "min" if args.ckpt_monitor == "val_loss" else "max"
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, args.split_scheme, "checkpoints"),
            filename="autoarborist-pretrain-{epoch:02d}-{%s:.4f}" % args.ckpt_monitor,
            monitor=args.ckpt_monitor,
            mode=monitor_mode,
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_cb)
        if args.early_stop_patience > 0:
            callbacks.append(pl.callbacks.EarlyStopping(
                monitor=args.ckpt_monitor, patience=args.early_stop_patience,
                mode=monitor_mode,
            ))

    trainer_kwargs = {}
    if args.gpus > 1:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
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

    # Baseline: score the init checkpoint on the held-out supervised images
    # BEFORE pretraining, so the run reports a delta not an isolated number.
    baseline_str = ""
    if has_val:
        print("\n=== BASELINE: init checkpoint on held-out supervised points (pre-training) ===")
        _, baseline_str = evaluate(
            model, sup_dataset, val_subset, batch_size=args.batch_size,
            max_batches=2 if args.smoke_test else None,
        )
        print(baseline_str)

    model.trainer.fit(model)

    # --- Select best epoch, export an HF-format TreeFormer checkpoint ----------
    best_path = None
    if checkpoint_cb is not None:
        best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
        print(f"\n=== checkpoint selection ({args.ckpt_monitor}) ===")
        print(f"  best : {checkpoint_cb.best_model_path}  ({checkpoint_cb.best_model_score})")
        print(f"  last : {checkpoint_cb.last_model_path}")
        if best_path:
            model = df_main.deepforest.load_from_checkpoint(best_path, weights_only=False)
            model.model.score_thresh = args.score_thresh
            model.model.score_integration_radius = args.score_integration_radius

    hf_dir = os.path.join(args.output_dir, args.split_scheme, "hf_checkpoint")
    os.makedirs(hf_dir, exist_ok=True)
    # TreeFormerModel is a PyTorchModelHubMixin -> save_pretrained writes
    # config.json + model.safetensors, the exact format train.py/eval.py's
    # --checkpoint consumes via TreeFormerModel.from_pretrained(<local dir>).
    model.model.save_pretrained(hf_dir)
    print(f"[stage1] HF checkpoint exported to {hf_dir}")

    # Post-pretraining eval on the same held-out supervised images.
    after_str = ""
    if has_val:
        print("\n=== AFTER pretraining: held-out supervised points ===")
        after_results, after_str = evaluate(
            model, sup_dataset, val_subset, batch_size=args.batch_size,
            max_batches=2 if args.smoke_test else None,
        )
        print(after_str)
        if loggers:
            exp = loggers[0].experiment
            safe = {k: float(v) for k, v in after_results.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))}
            exp.log_metrics(safe)

    results_path = os.path.join(args.output_dir, args.split_scheme,
                                f"pretrain_results_{args.split_scheme}.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        if baseline_str:
            f.write("########## BASELINE: init checkpoint, BEFORE AutoArborist pretraining ##########\n")
            f.write(baseline_str + "\n")
        if after_str:
            f.write("########## AFTER AutoArborist pretraining ##########\n")
            f.write(after_str + "\n")

    json_path = os.path.join(args.output_dir, args.split_scheme,
                             f"pretrain_results_{args.split_scheme}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "stage": "autoarborist-pretrain",
            "task": "TreePoints",
            "split": args.split_scheme,
            "init_checkpoint": args.init_checkpoint,
            "lr": args.lr,
            "image_size": args.image_size,
            "train_images": len(train_subset),
            "best_checkpoint": best_path,
            "hf_checkpoint": os.path.abspath(hf_dir),
        }, f, indent=2)
    print(f"[stage1] metadata written to {json_path}")
    print(f"\nStage 2 (treatment arm):\n"
          f"  training/points/train.py --checkpoint {os.path.abspath(hf_dir)} "
          f"--split-scheme {args.split_scheme} --image-size {args.image_size}")


if __name__ == "__main__":
    main()
