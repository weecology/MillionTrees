"""
Scan the training set for high-loss batches and save the offending images.

Usage:
    python training/boxes/find_high_loss_batches.py \
        --checkpoint training/boxes/outputs/random/checkpoints/boxes-epoch=06-val_loss=0.6541.ckpt \
        --output-dir training/boxes/outputs/high_loss_images \
        --top-k 20 \
        --loss-threshold 5.0
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_train_loader

import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_boxes import DeepForestBoxTrainer


def draw_boxes(image_tensor, boxes, color=(0, 255, 0)):
    t = image_tensor.cpu()
    img_np = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    boxes = boxes.cpu()
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    for b in boxes:
        x1, y1, x2, y2 = b.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return pil


def resolve_image_path(dataset, metadata_row):
    """Resolve filename from metadata tensor row (filename_id is metadata[:, 0])."""
    filename_id = metadata_row[0].item()
    return dataset._filename_id_to_code.get(filename_id, f"unknown_id_{filename_id}")


def save_image(images, targets_list, metadata, dataset, output_dir, prefix, loss):
    """Save annotated images with filename info embedded in output name and stdout."""
    saved = []
    for i in range(len(images)):
        boxes = targets_list[i]["y"]
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)
        img_filename = resolve_image_path(dataset, metadata[i])
        safe_name = os.path.basename(img_filename).replace("/", "_")
        pil = draw_boxes(images[i], boxes)
        fname = os.path.join(output_dir, f"{prefix}_loss{loss:.2f}_{safe_name}")
        pil.save(fname)
        print(f"    Saved: {fname}")
        print(f"    Source file: {img_filename}")
        print(f"    Boxes ({len(boxes)}): {boxes[:3].tolist()}")
        saved.append(img_filename)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root-dir", default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--split-scheme", default="random")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Use 1 to isolate individual images")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="training/boxes/outputs/high_loss_images")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Save the top-K highest loss images")
    parser.add_argument("--loss-threshold", type=float, default=5.0,
                        help="Also immediately save any batch with loss above this value")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop after this many batches (default: full dataset)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DeepForestBoxTrainer.load_from_checkpoint(args.checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.retinanet.train()  # need train mode to get losses

    dataset = get_dataset(
        "TreeBoxes",
        download=False,
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
    )
    train_subset = dataset.get_subset("train")
    loader = get_train_loader("standard", train_subset, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"Scanning {len(loader)} batches...")

    heap = []  # (loss_val, batch_idx, images, targets, metadata)

    for batch_idx, batch in enumerate(loader):
        if args.max_batches and batch_idx >= args.max_batches:
            break

        metadata, images, targets_list = batch
        images = images.to(device)

        rt = []
        for t in targets_list:
            boxes = t["y"]
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels = torch.zeros(len(boxes), dtype=torch.int64, device=device)
            rt.append({"boxes": boxes.to(device), "labels": labels})

        try:
            with torch.no_grad():
                loss_dict = model.retinanet(images, rt)
            loss = sum(l.item() for l in loss_dict.values())
        except Exception as e:
            print(f"  Batch {batch_idx}: ERROR — {e}")
            save_image(images.cpu(), targets_list, metadata, dataset, args.output_dir,
                       f"error_batch{batch_idx:06d}", loss=0.0)
            continue

        if batch_idx % 500 == 0:
            print(f"  batch {batch_idx:6d}  loss={loss:.4f}")

        # Immediately save anything above threshold
        if loss > args.loss_threshold:
            print(f"  *** HIGH LOSS batch {batch_idx}: {loss:.4f} ***")
            save_image(images.cpu(), targets_list, metadata, dataset, args.output_dir,
                       f"highloss_batch{batch_idx:06d}", loss=loss)

        heap.append((loss, batch_idx, images.cpu(), targets_list, metadata))
        heap.sort(key=lambda x: x[0], reverse=True)
        if len(heap) > args.top_k:
            heap.pop()

    print(f"\nTop-{args.top_k} highest loss batches:")
    for rank, (loss, batch_idx, images, targets_list, metadata) in enumerate(heap):
        print(f"  #{rank+1:2d}  batch={batch_idx:6d}  loss={loss:.4f}")
        save_image(images, targets_list, metadata, dataset, args.output_dir,
                   f"top{rank+1:02d}_batch{batch_idx:06d}", loss=loss)

    print(f"\nDone. Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
