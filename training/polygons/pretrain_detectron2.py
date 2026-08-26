"""Stage 1 of the Detectron2 polygon weak-supervision ablation.

Trains a **native Detectron2 Mask R-CNN with the mask branch switched off** on the
unsupervised Weinstein et al. 2018 boxes in TreeBoxes, then exports every tensor the
polygon stage can reuse -- backbone + FPN + RPN + ROI box head. The mask head is never
touched here, so it reaches stage 2 still at COCO init.

This is the Detectron2 twin of ``training/boxes/pretrain_backbone_for_polygons.py
--arch maskrcnn``. It exists because that script pretrains through DeepForest/torchvision,
and the polygon leaderboard model is native Detectron2: running both stages in one stack
is what makes the downstream three-arm comparison attributable to the weak labels rather
than to a cross-stack conversion.

Two defects from the box round are fixed here by construction, because they are what made
the earlier stage-1 runs uninterpretable:

* **Gradient clipping is on** (``SOLVER.CLIP_GRADIENTS``, norm 1.0). Stage 1 had none, so
  lr 1e-2 went NaN from epoch 0 and lr 1e-3 diverged late.
* **Checkpoint selection is by box recall, not loss.** The loss is dominated by
  classification over ~150-boxes-per-tile pseudo-labels and does not track detection
  quality: the val_loss-best epoch of the earlier sweep exported a near-dead detector
  (recall 0.003) while a worse-loss epoch detected ten times better.

Runs under the CanopyRS uv venv (Detectron2 compiled from source); see
``training/slurm/pretrain_detectron2.sbatch``.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.logger import setup_logger

import d2_weak_supervision as d2ws

_COCO_MASKRCNN = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
_DEFAULT_COCO_PKL = str(
    Path(__file__).with_name("detectron2_assets") / "model_final_f10217.pkl"
)


# --------------------------------------------------------------------------- #
# Supervised holdout box recall -- the checkpoint-selection signal
# --------------------------------------------------------------------------- #
def gt_boxes_by_image_id(records):
    """``image_id`` -> ground-truth boxes in original pixel coordinates.

    The inference loader runs its mapper with ``is_train=False``, which strips
    ``annotations``, so the ground truth has to come from the records themselves.
    Detectron2 rescales predictions back to the ``height``/``width`` it was handed, so
    the two are already in the same coordinate space.
    """
    return {
        rec["image_id"]: torch.tensor(
            [a["bbox"] for a in rec["annotations"]], dtype=torch.float32
        ).reshape(-1, 4)
        for rec in records
    }


@torch.no_grad()
def holdout_box_recall(model, loader, gt_by_id, iou_threshold=0.4, score_threshold=0.05):
    """Fraction of held-out *human* boxes matched by a prediction at ``iou_threshold``.

    Deliberately measured against supervised annotations, not the pseudo-labels being
    trained on: the question stage 1 has to answer is whether the weak boxes teach the
    network to find real trees, which the training loss cannot say.
    """
    was_training = model.training
    model.eval()
    matched = total = n_pred = n_images = 0
    for batch in loader:
        outputs = model(batch)
        for datum, output in zip(batch, outputs):
            gt_boxes = gt_by_id[datum["image_id"]]
            total += len(gt_boxes)
            n_images += 1
            inst = output["instances"].to("cpu")
            keep = inst.scores >= score_threshold
            pred_boxes = inst.pred_boxes.tensor[keep]
            n_pred += len(pred_boxes)
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue
            ious = pairwise_iou(Boxes(gt_boxes), Boxes(pred_boxes.float()))
            matched += int((ious.max(dim=1).values >= iou_threshold).sum())
    if was_training:
        model.train()
    return {
        "box_recall": matched / total if total else 0.0,
        "mean_dets_per_img": n_pred / n_images if n_images else 0.0,
        "n_gt_boxes": total,
        "n_images": n_images,
    }


class BoxRecallHook(HookBase):
    """Evaluate holdout box recall periodically and keep the best-scoring weights."""

    def __init__(self, cfg, period, loader, gt_by_id, output_dir, comet_exp=None):
        self.period = period
        self.loader = loader
        self.gt_by_id = gt_by_id
        self.output_dir = Path(output_dir)
        self.comet_exp = comet_exp
        self.best_score = -1.0
        self.best_iter = -1
        self.history = []
        self._cfg = cfg

    def _evaluate(self):
        stats = holdout_box_recall(self.trainer.model, self.loader, self.gt_by_id)
        it = self.trainer.iter + 1
        stats["iter"] = it
        self.history.append(stats)
        print(f"[stage1] iter {it}: box_recall={stats['box_recall']:.4f} "
              f"mean_dets/img={stats['mean_dets_per_img']:.1f} "
              f"(n_gt={stats['n_gt_boxes']}, n_img={stats['n_images']})", flush=True)
        if self.comet_exp is not None:
            self.comet_exp.log_metrics(
                {"holdout_box_recall": stats["box_recall"],
                 "holdout_mean_dets_per_img": stats["mean_dets_per_img"]}, step=it)
        if stats["box_recall"] > self.best_score:
            self.best_score = stats["box_recall"]
            self.best_iter = it
            torch.save(self.trainer.model.state_dict(),
                       self.output_dir / "stage1_best.pth")
            print(f"[stage1] new best box_recall={self.best_score:.4f} at iter {it} "
                  f"-> stage1_best.pth", flush=True)

    def after_step(self):
        it = self.trainer.iter + 1
        if it % self.period == 0 and it != self.trainer.max_iter:
            self._evaluate()

    def after_train(self):
        self._evaluate()


# --------------------------------------------------------------------------- #
class Stage1Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        from detectron2.data import build_detection_train_loader
        from detectron2.data.dataset_mapper import DatasetMapper
        # MASK_ON is False here, so the mapper drops "segmentation" itself and the
        # rectangle stand-ins never reach the model.
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True))


def build_cfg(args, train_name, n_train_images, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_COCO_MASKRCNN))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = (
        args.coco_pkl if os.path.exists(args.coco_pkl)
        else model_zoo.get_checkpoint_url(_COCO_MASKRCNN)
    )
    # The whole point of stage 1: box supervision only, mask branch never built.
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    # Weak tiles carry ~150 boxes and the supervised holdout ~55, so Detectron2's default
    # 100-detection cap binds and silently ceilings the box_recall that selects the
    # checkpoint. 300 matches the polygon driver's test-time cap.
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    iters_per_epoch = max(1, math.ceil(n_train_images / args.batch_size))
    cfg.SOLVER.MAX_ITER = args.max_iter or iters_per_epoch * args.max_epochs
    cfg.SOLVER.STEPS = (int(0.7 * cfg.SOLVER.MAX_ITER), int(0.9 * cfg.SOLVER.MAX_ITER))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = min(1000, iters_per_epoch)
    cfg.SOLVER.CHECKPOINT_PERIOD = 10 ** 9  # selection is by box recall, not periodicity
    # Fix 1 from the box round: stage 1 previously had no clipping and diverged.
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = args.grad_clip
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    cfg.INPUT.MIN_SIZE_TRAIN = tuple(args.min_size_train)
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.INPUT.MAX_SIZE_TEST = args.max_size

    cfg.OUTPUT_DIR = str(output_dir)
    cfg.SEED = args.seed
    os.makedirs(output_dir, exist_ok=True)
    print(f"[cfg] {n_train_images} weak imgs | {iters_per_epoch} iters/epoch | "
          f"MAX_ITER {cfg.SOLVER.MAX_ITER} | lr {args.lr} | batch {args.batch_size} | "
          f"grad-clip {args.grad_clip} | steps {cfg.SOLVER.STEPS}")
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-dir", type=str,
                   default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    p.add_argument("--boxes-dataset", type=str, default="TreeBoxes_v0.23",
                   help="TreeBoxes release directory holding the weak (unsupervised) rows.")
    p.add_argument("--split-scheme", type=str, default="within-distribution",
                   choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    p.add_argument("--weak-sources", type=str, nargs="+", default=["*unsupervised*"],
                   help="Source glob(s) counted as weak supervision.")
    p.add_argument("--exclude-sources", type=str, nargs="*", default=[],
                   help="Source glob(s) to hold out of the weak set.")
    p.add_argument("--holdout-images", type=int, default=512,
                   help="Supervised images reserved for the box-recall selection signal.")
    p.add_argument("--holdout-exclude-sources", type=str, nargs="*",
                   default=["*unsupervised*", "*weak supervised*"],
                   help="Sources kept OUT of the selection holdout. Young et al. 2025 is "
                        "weak supervised, i.e. pseudo-labels: scoring against it would "
                        "measure agreement with another model rather than recall of real "
                        "trees, which is the whole point of the holdout.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--max-iter", type=int, default=None,
                   help="Override the epoch-derived schedule with an explicit iteration budget.")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--eval-period", type=int, default=2000,
                   help="Iterations between holdout box-recall evaluations.")
    p.add_argument("--min-size-train", type=int, nargs="+",
                   default=[640, 672, 704, 736, 768, 800])
    p.add_argument("--min-size-test", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--coco-pkl", type=str, default=_DEFAULT_COCO_PKL)
    p.add_argument("--output-dir", type=str,
                   default="training/weak_supervision/outputs/pretrain_detectron2")
    p.add_argument("--limit-weak-images", type=int, default=None,
                   help="Smoke-test knob: cap the weak training set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--comet", action="store_true")
    p.add_argument("--comet-name", type=str, default=None)
    args = p.parse_args()

    setup_logger()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.root_dir) / args.boxes_dataset

    print(f"=== Detectron2 stage 1: weak-box pretraining ({args.split_scheme}) ===")
    print(f"[data] {data_dir}")
    weak = d2ws.load_weak_box_dicts(
        data_dir, args.split_scheme, split="train",
        source_patterns=tuple(args.weak_sources),
        exclude_patterns=tuple(args.exclude_sources),
        limit_images=args.limit_weak_images,
    )
    n_weak_boxes = sum(len(r["annotations"]) for r in weak)
    print(f"[data] weak train: {len(weak)} images / {n_weak_boxes} boxes "
          f"({n_weak_boxes / max(1, len(weak)):.0f} boxes/img)")

    holdout = d2ws.load_weak_box_dicts(
        data_dir, args.split_scheme, split="train",
        source_patterns=("*",),
        exclude_patterns=(tuple(args.weak_sources) + tuple(args.exclude_sources)
                          + tuple(args.holdout_exclude_sources)),
        image_id_offset=10 ** 7, limit_images=args.holdout_images,
        stratify_by_source=True,
    )
    n_hold_boxes = sum(len(r["annotations"]) for r in holdout)
    print(f"[data] supervised holdout: {len(holdout)} images / {n_hold_boxes} human boxes "
          f"(excluding {list(args.holdout_exclude_sources)})")

    train_name = f"mt_stage1_weak_{args.split_scheme}"
    hold_name = f"mt_stage1_holdout_{args.split_scheme}"
    for name, recs in ((train_name, weak), (hold_name, holdout)):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        DatasetCatalog.register(name, lambda r=recs: r)
        MetadataCatalog.get(name).set(thing_classes=["Tree"])

    cfg = build_cfg(args, train_name, len(weak), output_dir)

    comet_exp = None
    if args.comet:
        try:
            import comet_ml
            comet_exp = comet_ml.Experiment(project_name="milliontrees-polygons",
                                            auto_metric_logging=False)
            comet_exp.set_name(args.comet_name or
                               f"stage1-{args.split_scheme}-coco-lr{args.lr:g}-detectron2")
            comet_exp.add_tags([f"split-{args.split_scheme}", "geometry-boxes",
                                "stack-detectron2", "stage-1", "weak-supervision"])
            comet_exp.log_parameters(vars(args))
        except Exception as e:  # noqa: BLE001
            print(f"Comet ML logging disabled: {e}")

    trainer = Stage1Trainer(cfg)
    trainer.resume_or_load(resume=False)
    # Verify AFTER the checkpointer has loaded, so this describes the actual init.
    d2ws.verify_coco_init(trainer.model, args.coco_pkl)

    hold_loader = build_detection_test_loader(cfg, hold_name)
    gt_by_id = gt_boxes_by_image_id(holdout)
    recall_hook = BoxRecallHook(cfg, args.eval_period, hold_loader, gt_by_id,
                                output_dir, comet_exp)
    trainer.register_hooks([recall_hook])

    print("=== pre-training BASELINE (COCO init, before any weak boxes) ===")
    baseline = holdout_box_recall(trainer.model, hold_loader, gt_by_id)
    print(f"[stage1] BEFORE: box_recall={baseline['box_recall']:.4f} "
          f"mean_dets/img={baseline['mean_dets_per_img']:.1f}")

    trainer.train()

    print("\n=== checkpoint selection (box_recall, mode=max) ===")
    best_path = output_dir / "stage1_best.pth"
    print(f"  best : {best_path}\n  score: {recall_hook.best_score} "
          f"(iter {recall_hook.best_iter})")
    if not best_path.exists():
        raise RuntimeError("no best checkpoint was written; refusing to export")
    trainer.model.load_state_dict(torch.load(best_path, map_location="cpu"))

    export_path = output_dir / f"d2_network_{args.split_scheme}.pkl"
    n_exported = d2ws.export_transferable(trainer.model, export_path)
    merged_path = output_dir / f"d2_network_{args.split_scheme}_merged.pkl"
    merge_report = d2ws.merge_transferable_into_coco(
        export_path, args.coco_pkl, merged_path)

    meta = {
        "split_scheme": args.split_scheme,
        "stack": "detectron2",
        "arch": "mask_rcnn_R_50_FPN_3x (MASK_ON=False)",
        "weak_images": len(weak),
        "weak_boxes": n_weak_boxes,
        "holdout_images": len(holdout),
        "holdout_boxes": n_hold_boxes,
        "holdout_exclude_sources": args.holdout_exclude_sources,
        "baseline_box_recall": baseline["box_recall"],
        "best_box_recall": recall_hook.best_score,
        "best_iter": recall_hook.best_iter,
        "max_iter": cfg.SOLVER.MAX_ITER,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "exported_tensors": n_exported,
        "merge_report": merge_report,
        "history": recall_hook.history,
        "export": str(export_path),
        "merged_export": str(merged_path),
    }
    meta_path = output_dir / f"results_{args.split_scheme}.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Metadata written to {meta_path}")

    if comet_exp is not None:
        comet_exp.log_metrics({"best_box_recall": recall_hook.best_score,
                               "baseline_box_recall": baseline["box_recall"]})
        comet_exp.end()

    print("STAGE-1 DONE")


if __name__ == "__main__":
    main()
