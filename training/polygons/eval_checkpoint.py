"""Evaluate a trained MaskRCNNPolygonTrainer checkpoint on MillionTrees TreePolygons."""

import argparse
import json
import os
import sys

import torch

from training.polygons.train_polygons import MaskRCNNPolygonTrainer, evaluate
from milliontrees import get_dataset

sys.modules['__main__'].MaskRCNNPolygonTrainer = MaskRCNNPolygonTrainer


def main():
    parser = argparse.ArgumentParser(description="Eval a trained TreePolygons checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt file")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--init-mode", type=str, default="unknown")
    parser.add_argument("--data-scope", type=str, default="unknown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-unsupervised", action="store_true")
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs")
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="stream",
        choices=["stream", "legacy"],
        help="stream = low-memory per-batch metrics; legacy = accumulate then dataset.eval()",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    model = MaskRCNNPolygonTrainer.load_from_checkpoint(args.checkpoint, weights_only=False)
    model.eval()
    model = model.to(device)

    dataset = get_dataset(
        "TreePolygons",
        root_dir=args.root_dir,
        mini=args.mini,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
        include_unsupervised=args.include_unsupervised,
    )
    test_subset = dataset.get_subset("test")

    results, results_str = evaluate(
        model,
        dataset,
        test_subset,
        batch_size=args.batch_size,
        device=device,
        viz_dir=args.viz_dir,
        eval_mode=args.eval_mode,
    )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
        with open(out_path, "w") as f:
            f.write(results_str)
        json_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.json")
        flat = {k: float(v) if hasattr(v, "item") else v
                for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))}
        with open(json_path, "w") as f:
            json.dump(
                {
                    "model": "trained-polygons",
                    "task": "TreePolygons",
                    "split": args.split_scheme,
                    "metrics": flat,
                    "run_metadata": {
                        "init_mode": args.init_mode,
                        "data_scope": args.data_scope,
                        "seed": args.seed,
                        "include_unsupervised": args.include_unsupervised,
                        "checkpoint": args.checkpoint,
                        "eval_mode": args.eval_mode,
                    },
                },
                f,
                indent=2,
            )
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
