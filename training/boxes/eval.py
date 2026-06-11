"""Evaluate a trained DeepForestBoxTrainer checkpoint on MillionTrees TreeBoxes."""

import argparse
import os

import sys
import torch
from deepforest import main as df_main
from training.boxes.train import _AdaptCollate, evaluate
from milliontrees import get_dataset

# The checkpoint was saved when train.py ran as __main__, so pickle stored the
# batch adapter under __main__.  Inject it here so deserialization can resolve
# that reference. The class was renamed _AdaptCollate; alias the old name too so
# checkpoints saved before the rename still load.
sys.modules['__main__']._AdaptCollate = _AdaptCollate
sys.modules['__main__'].MillionTreesBatchAdapter = _AdaptCollate


def _load_model(checkpoint_path):
    """Load deepforest from checkpoint, remapping wrapper prefixes if needed.

    Older checkpoints were saved from a wrapper class that stored the deepforest
    model as self.df_model (keys: df_model.model.*) and kept a second reference
    as self.retinanet.  Strip df_model. and skip retinanet.* duplicates so the
    weights load cleanly into the stock deepforest class (self.model.*).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", {})
    needs_remap = any(k.startswith("df_model.") for k in state)
    if needs_remap:
        remapped = {}
        for k, v in state.items():
            if k.startswith("df_model."):
                remapped[k[len("df_model."):]] = v
            elif not k.startswith("retinanet."):
                remapped[k] = v
        model = df_main.deepforest()
        model.load_state_dict(remapped)
    else:
        # Build a fresh deepforest with devices=1 and load weights manually. Avoids
        # deepforest.load_from_checkpoint, which would re-read the saved config
        # (devices=2 from multi-GPU training) and crash on a single-GPU eval node.
        cfg = ckpt.get("hyper_parameters", {}).get("config", None)
        if isinstance(cfg, dict):
            cfg = {**cfg, "devices": 1}
            model = df_main.deepforest(config=cfg)
        else:
            model = df_main.deepforest()
        model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser(description="Eval a trained TreeBoxes checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt file")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="zeroshot",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Override the model's internal score_thresh "
                             "(standardized to 0.1 for the leaderboard).")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs "
                             "(default: <output-dir>/viz, else ./eval_viz; pass '' to disable)")
    args = parser.parse_args()

    # Visualization on by default: 10 overlays per source (dataset.eval viz_n_per_source=10).
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.output_dir, "viz") if args.output_dir else "eval_viz"
    elif args.viz_dir == "":
        args.viz_dir = None

    print(f"Loading checkpoint: {args.checkpoint}")
    model = _load_model(args.checkpoint)
    if args.score_threshold is not None:
        model.config["score_thresh"] = args.score_threshold
        if getattr(model, "model", None) is not None:
            model.model.score_thresh = args.score_threshold
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = get_dataset(
        "TreeBoxes",
        root_dir=args.root_dir,
        mini=args.mini,
        split_scheme=args.split_scheme,
    )
    test_subset = dataset.get_subset("test")

    results, results_str = evaluate(model, dataset, test_subset, batch_size=args.batch_size,
                                    viz_dir=args.viz_dir)
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
        with open(out_path, "w") as f:
            f.write(results_str)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
