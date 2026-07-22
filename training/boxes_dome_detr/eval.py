"""Evaluate a trained Dome-DETR checkpoint on MillionTrees TreeBoxes."""

import argparse
import os
import torch

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.common.eval_sweep import add_sweep_args, maybe_run_sweep, maybe_subsample


def predict_batch(model, images, device):
    """Run Dome-DETR inference; returns MillionTrees-format prediction dicts."""
    if isinstance(images, list):
        images = torch.stack(images)
    images = images.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)

    predictions = []
    for output in outputs:
        boxes = output.get("bboxes", torch.zeros((0, 4)))
        scores = output.get("scores", torch.zeros((0,)))
        if len(boxes) == 0:
            predictions.append({
                "y": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
        else:
            predictions.append({
                "y": boxes.detach().float().cpu(),
                "labels": torch.ones(len(boxes), dtype=torch.int64),
                "scores": scores.detach().float().cpu(),
            })

    return predictions


def collect_predictions(model, test_subset, batch_size=12, max_batches=None, device="cuda"):
    """Run inference over test_subset, returning (all_y_pred, all_y_true)."""
    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    all_y_pred, all_y_true = [], []

    for i, batch in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        _, images, targets = batch
        preds = predict_batch(model, images, device)
        all_y_pred.extend(preds)
        all_y_true.extend(targets)

    return all_y_pred, all_y_true


def evaluate(model, dataset, test_subset, batch_size=12, viz_dir=None, max_batches=None,
             device="cuda", score_threshold=None):
    """Evaluate model on test_subset and return MillionTrees eval results."""
    if score_threshold is not None:
        # TODO: Apply score_threshold to model predictions
        pass

    all_y_pred, all_y_true = collect_predictions(
        model, test_subset, batch_size=batch_size, max_batches=max_batches, device=device
    )

    results, results_str = dataset.eval(
        all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)],
        viz_dir=viz_dir,
    )
    return results, results_str


def load_model(checkpoint_path, device="cuda"):
    """Load Dome-DETR model from checkpoint.

    TODO: Implement full model loading from Dome-DETR checkpoint.
    For now, this is a stub that shows the interface.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    # In practice, this would:
    # 1. Load the config from the checkpoint's parent directory
    # 2. Build the Dome-DETR model
    # 3. Load the saved state_dict
    # For now, returning None as a placeholder
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Dome-DETR checkpoint on MillionTrees TreeBoxes."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"),
        help="Root directory of MillionTrees data.",
    )
    parser.add_argument(
        "--split-scheme",
        type=str,
        default="out-of-distribution",
        choices=["within-distribution", "out-of-distribution", "crossgeometry"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override model's score_thresh (default 0.1 for leaderboard).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Directory for prediction overlays (default: <output-dir>/viz)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    add_sweep_args(parser)
    args = parser.parse_args()

    # Visualization on by default: 10 overlays per source
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.output_dir, "viz") if args.output_dir else "eval_viz"
    elif args.viz_dir == "":
        args.viz_dir = None

    os.makedirs(args.output_dir, exist_ok=True) if args.output_dir else None
    os.makedirs(args.viz_dir, exist_ok=True) if args.viz_dir else None

    print(f"Loading dataset ({args.split_scheme})...")
    dataset = get_dataset(
        "TreeBoxes",
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        mini=args.mini,
    )

    test_subset = dataset.get_subset(args.eval_split)
    print(f"Evaluating {len(test_subset)} samples from {args.eval_split} split")

    # Handle --per-source subsampling
    test_subset = maybe_subsample(dataset, test_subset, args)

    # Load model and run inference
    model = load_model(args.checkpoint, device=args.device)

    if model is None:
        print("ERROR: Model loading is not yet implemented. This is a stub for integration testing.")
        print("Full implementation pending: model architecture, checkpoint format, etc.")
        return

    results, results_str = evaluate(
        model, dataset, test_subset,
        batch_size=args.batch_size,
        viz_dir=args.viz_dir,
        max_batches=2 if args.mini else None,
        device=args.device,
        score_threshold=args.score_threshold,
    )

    # Write results
    if args.output_dir:
        output_file = os.path.join(args.output_dir, f"results_{args.split_scheme}.txt")
        with open(output_file, "w") as f:
            f.write(results_str)
        print(f"\nResults written to {output_file}")

    # Run threshold sweep if requested
    if args.sweep:
        maybe_run_sweep(
            args, dataset, test_subset,
            model=None,  # Would use this in full implementation
            task="boxes",
        )

    print(results_str)


if __name__ == "__main__":
    main()
