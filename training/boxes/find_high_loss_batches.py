"""Find high-loss training images from a trained DeepForest checkpoint.

Runs inference-time loss over the training split and writes a CSV of the
top-k or threshold-exceeding images for inspection.
"""

import argparse
import csv
import os

import torch

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader

from train_boxes import DeepForestBoxTrainer


def compute_loss(model, images, targets_list):
    device = next(model.parameters()).device
    images = images.to(device)
    rt = model._prepare_targets(targets_list, device)
    model.retinanet.train()
    with torch.no_grad():
        loss_dict = model.retinanet(images, rt)
    model.retinanet.eval()
    return sum(v.item() for v in loss_dict.values())


def main():
    parser = argparse.ArgumentParser(description="Find high-loss images in training split")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", type=str, default="random")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--loss-threshold", type=float, default=None,
                        help="Include images with loss >= this value (use with or instead of --top-k)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Also report the top-k highest-loss images regardless of threshold")
    parser.add_argument("--output-dir", type=str, default="training/boxes/outputs/high_loss_images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DeepForestBoxTrainer.load_from_checkpoint(args.checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = get_dataset(
        "TreeBoxes",
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
    )
    train_subset = dataset.get_subset("train")
    loader = get_train_loader(
        "standard", train_subset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
    )

    records = []
    for batch_idx, batch in enumerate(loader):
        metadata, images, targets_list = batch
        loss = compute_loss(model, images, targets_list)
        for i in range(len(images)):
            meta = metadata[i] if isinstance(metadata, list) else metadata
            # Extract filename from metadata if available, fall back to index
            img_id = int(train_subset.indices[batch_idx * args.batch_size + i]) \
                if hasattr(train_subset, "indices") else batch_idx * args.batch_size + i
            records.append({"index": img_id, "loss": loss})
        if batch_idx % 100 == 0:
            print(f"  batch {batch_idx}/{len(loader)}  loss={loss:.4f}")

    records.sort(key=lambda r: r["loss"], reverse=True)

    # Apply threshold and/or top-k
    if args.loss_threshold is not None:
        above = [r for r in records if r["loss"] >= args.loss_threshold]
    else:
        above = []
    top_k = records[: args.top_k]

    # Union, preserving order
    seen = set()
    combined = []
    for r in records:
        key = r["index"]
        if key in seen:
            continue
        if r in above or r in top_k:
            combined.append(r)
            seen.add(key)

    out_path = os.path.join(args.output_dir, f"high_loss_{args.split_scheme}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "loss"])
        writer.writeheader()
        writer.writerows(combined)

    print(f"\nWrote {len(combined)} high-loss records to {out_path}")
    print(f"Top-5 losses:")
    for r in combined[:5]:
        print(f"  index={r['index']}  loss={r['loss']:.4f}")


if __name__ == "__main__":
    main()
