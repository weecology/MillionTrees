"""Convert MillionTrees TreeBoxes dataset to COCO JSON format for Dome-DETR training.

Dome-DETR natively expects COCO-format annotations (single JSON file per split with
standard 'images' and 'annotations' arrays, each image linked to its box annotations).
This script converts TreeBoxes (split CSVs with per-image metadata) into that format.

Usage:
  python convert_to_coco.py --split-scheme within-distribution --root-dir /path/to/MillionTrees
  # Outputs: coco_annotations/{within-distribution,out-of-distribution}_train.json, etc.
"""

import argparse
import json
import os
from pathlib import Path

from milliontrees import get_dataset


def convert_split_to_coco(dataset, split_name, output_dir, img_folder):
    """Convert a single TreeBoxes subset (train/test) to COCO format.

    Args:
      dataset: TreeBoxesDataset instance
      split_name: "train" or "test"
      output_dir: directory to write coco_*.json to
      img_folder: path to images directory (for validation/documentation)

    Returns:
      dict with COCO format: { "images": [...], "annotations": [...], "categories": [...] }
    """
    subset = dataset.get_subset(split_name)
    images = []
    annotations = []
    annotation_id = 1

    for idx, (metadata, image, targets) in enumerate(subset):
        filename_id = int(metadata[0])
        img_height, img_width = image.shape[1], image.shape[2]

        images.append({
            "id": idx,
            "file_name": f"{filename_id}.png",
            "height": img_height,
            "width": img_width,
        })

        boxes = targets["y"]
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)

        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1
            annotations.append({
                "id": annotation_id,
                "image_id": idx,
                "category_id": 1,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tree"}],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert MillionTrees TreeBoxes to COCO format for Dome-DETR."
    )
    parser.add_argument(
        "--split-scheme",
        type=str,
        default="out-of-distribution",
        choices=["within-distribution", "out-of-distribution", "crossgeometry"],
        help="Which split scheme to use.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"),
        help="Root directory of MillionTrees data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/boxes_dome_detr/coco_annotations",
        help="Output directory for COCO JSON files.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Use the mini dataset (for testing).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading TreeBoxes dataset ({args.split_scheme})...")
    dataset = get_dataset(
        "TreeBoxes",
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        mini=args.mini,
    )

    img_folder = str(dataset._data_dir / "images")
    print(f"Image folder: {img_folder}")

    for split_name in ["train", "test"]:
        print(f"Converting {split_name} split...")
        coco_data = convert_split_to_coco(dataset, split_name, args.output_dir, img_folder)
        output_file = os.path.join(args.output_dir, f"{args.split_scheme}_{split_name}.json")
        with open(output_file, "w") as f:
            json.dump(coco_data, f)
        print(f"  Wrote {output_file} ({len(coco_data['images'])} images, "
              f"{len(coco_data['annotations'])} annotations)")


if __name__ == "__main__":
    main()
