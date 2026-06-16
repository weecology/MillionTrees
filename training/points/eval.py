"""Evaluate a fine-tuned TreeFormer checkpoint on MillionTrees TreePoints."""

import argparse
import json
import os

import torch

from deepforest import main as df_main

from milliontrees import get_dataset
from training.points.train import evaluate


def main():
    parser = argparse.ArgumentParser(description="Eval a TreeFormer TreePoints checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Lightning .ckpt file, or a HuggingFace repo id / local "
                             "HF checkpoint dir (config.json + model.safetensors)")
    parser.add_argument("--revision", type=str, default="main",
                        help="HF revision/tag (ignored for .ckpt files and local dirs)")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="zeroshot",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--small", action="store_true",
                        help="Use the SmallTreePoints release for fast iteration.")
    parser.add_argument("--image-size", type=int, default=448,
                        help="Square resize for images/points at eval; also sets "
                             "the KeypointAccuracy pixel threshold.")
    parser.add_argument("--eval-score-threshold", type=float, default=None,
                        help="Override the dataset eval_score_threshold (hard score "
                             "filter). Default (None) keeps the dataset default 0.4.")
    parser.add_argument("--score-thresh", type=float, default=0.1,
                        help="Relative peak threshold in [0, 1] for density_to_points "
                             "(standardized to 0.10 for the leaderboard). TreeFormer's "
                             "analog of a detection score_threshold.")
    parser.add_argument("--score-integration-radius", type=int, default=2,
                        help="peak_local_max min_distance in density-map px "
                             "(~4x in image px). Standard default 2 (tuned from "
                             "the deepforest default of 5); lower => more recall "
                             "in dense canopy.")
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

    # Two checkpoint formats: a Lightning .ckpt (load_from_checkpoint), or a
    # HuggingFace-format TreeFormer checkpoint (repo id, or local dir with
    # config.json + model.safetensors). The latter must be loaded the same way
    # the polygon eval and points training do: build a treeformer deepforest,
    # then load_model() the weights.
    if args.checkpoint.endswith(".ckpt") and os.path.isfile(args.checkpoint):
        print(f"Loading Lightning checkpoint: {args.checkpoint}")
        # Checkpoints trained on multiple GPUs save devices=[0,1,..] in their
        # config; deepforest.__init__ builds a trainer from it at load time and
        # crashes on a single-GPU eval node. Merge a single-device override over
        # the saved config (main.py merges config_args onto the checkpoint dict).
        model = df_main.deepforest.load_from_checkpoint(
            args.checkpoint,
            weights_only=False,
            config_args={"devices": 1, "accelerator": "auto",
                         "workers": args.num_workers},
        )
    else:
        print(f"Loading TreeFormer weights: {args.checkpoint} (revision={args.revision})")
        model = df_main.deepforest(config_args={
            "architecture": "treeformer",
            "model": {"name": args.checkpoint, "revision": args.revision},
        })
        model.load_model(args.checkpoint, revision=args.revision)
    if args.score_thresh is not None:
        model.model.score_thresh = args.score_thresh
    if args.score_integration_radius is not None:
        model.model.score_integration_radius = args.score_integration_radius
    print(f"score_thresh: {model.model.score_thresh} "
          f"score_integration_radius: {model.model.score_integration_radius}")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset_kwargs = dict(
        root_dir=args.root_dir,
        mini=args.mini,
        small=args.small,
        image_size=args.image_size,
        split_scheme=args.split_scheme,
    )
    if args.eval_score_threshold is not None:
        dataset_kwargs["eval_score_threshold"] = args.eval_score_threshold
    dataset = get_dataset("TreePoints", **dataset_kwargs)
    test_subset = dataset.get_subset("test")

    results, results_str = evaluate(
        model, dataset, test_subset,
        batch_size=args.batch_size,
        viz_dir=args.viz_dir,
    )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
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
                "metrics": flat,
            }, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
