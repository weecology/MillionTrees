"""Evaluate a trained DeepForestBoxTrainer checkpoint on MillionTrees TreeBoxes."""

import argparse
import os

import torch

from training.boxes.train_boxes import DeepForestBoxTrainer, evaluate
from milliontrees import get_dataset


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
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DeepForestBoxTrainer.load_from_checkpoint(args.checkpoint)
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

    results, results_str = evaluate(model, dataset, test_subset, batch_size=args.batch_size)
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
        with open(out_path, "w") as f:
            f.write(results_str)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
