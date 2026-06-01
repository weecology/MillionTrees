"""Zero-shot TreeFormer (DeepForest point model) on MillionTrees TreePoints.

Requires ``uv sync --extra treeformer`` (DeepForest treeformer-training branch).
"""

import argparse
import json
import os
import warnings

import torch

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from training.points.train_treeformer import predict_batch


def main():
    parser = argparse.ArgumentParser(
        description="Pretrained TreeFormer on TreePoints (native point predictions)."
    )
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weecology/deepforest-tree-point",
        help="Hugging Face repo or path for pretrained TreeFormer weights",
    )
    args = parser.parse_args()

    model = df_main.deepforest(config="point")
    model.load_model(args.checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = get_dataset(
        "TreePoints",
        root_dir=args.root_dir,
        download=args.download,
        mini=args.mini,
        split_scheme=args.split_scheme,
    )
    test_subset = dataset.get_subset("test")
    test_loader = get_eval_loader(
        "standard", test_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Batches: {len(test_loader)}")

    all_y_pred, all_y_true = [], []
    for b_idx, (_, images, targets) in enumerate(test_loader):
        preds = predict_batch(model, images)
        all_y_pred.extend(preds)
        all_y_true.extend(targets)
        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    results, results_str = dataset.eval(
        all_y_pred,
        all_y_true,
        test_subset.metadata_array[:len(all_y_true)],
        viz_dir=args.viz_dir,
    )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        txt_path = os.path.join(args.output_dir, f"results_treeformer_{args.split_scheme}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(results_str)
        json_path = os.path.join(args.output_dir, f"results_treeformer_{args.split_scheme}.json")
        flat = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in results.items()
            if isinstance(v, (int, float, torch.Tensor))
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "TreeFormer-pretrained",
                "task": "TreePoints",
                "split": args.split_scheme,
                "checkpoint": args.checkpoint,
                "metrics": flat,
            }, f, indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
