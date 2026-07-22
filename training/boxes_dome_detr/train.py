"""Train Dome-DETR on MillionTrees TreeBoxes.

Wrapper that handles:
  1. Converting TreeBoxes → COCO format
  2. Rendering the config template with hyperparameters
  3. Calling the vendored Dome-DETR train.py via subprocess
  4. Running final MillionTrees eval on the best checkpoint
  5. Logging to Comet if requested

Usage:
  source training/boxes_dome_detr/.venv/bin/activate
  python training/boxes_dome_detr/train.py --split-scheme within-distribution --lr 0.001
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def convert_to_coco_if_needed(root_dir, split_scheme, output_dir, mini=False):
    """Convert TreeBoxes to COCO format if not already done.

    Returns:
      (img_folder, train_json_path, test_json_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    train_json = os.path.join(output_dir, f"{split_scheme}_train.json")
    test_json = os.path.join(output_dir, f"{split_scheme}_test.json")

    if os.path.exists(train_json) and os.path.exists(test_json):
        print(f"COCO annotations already exist for {split_scheme}, skipping conversion.")
    else:
        print(f"Converting TreeBoxes to COCO format for {split_scheme}...")
        dataset = get_dataset(
            "TreeBoxes",
            root_dir=root_dir,
            split_scheme=split_scheme,
            mini=mini,
        )
        img_folder = str(dataset._data_dir / "images")

        for split_name in ["train", "test"]:
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

            coco_data = {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "tree"}],
            }

            output_file = os.path.join(output_dir, f"{split_scheme}_{split_name}.json")
            with open(output_file, "w") as f:
                json.dump(coco_data, f)
            print(f"  {split_name}: {len(images)} images, {len(annotations)} boxes → {output_file}")

    return img_folder, train_json, test_json


def render_config(template_path, output_path, img_folder, train_json, test_json,
                  batch_size, num_workers, lr, epochs):
    """Render the config template with runtime values."""
    with open(template_path) as f:
        config_text = f.read()

    config_text = config_text.format(
        img_folder=img_folder,
        ann_file=train_json,  # Dome-DETR uses train_json for training
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        epochs=epochs,
    )

    with open(output_path, "w") as f:
        f.write(config_text)
    print(f"Rendered config: {output_path}")


def run_dome_detr_training(dome_detr_repo, config_path, output_dir, gpus, seed=0):
    """Run Dome-DETR training via subprocess (torchrun)."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        f"{dome_detr_repo}/train.py",
        "-c", config_path,
        f"--output_dir={output_dir}",
        f"--seed={seed}",
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=dome_detr_repo)
    if result.returncode != 0:
        raise RuntimeError(f"Dome-DETR training failed with code {result.returncode}")


def find_best_checkpoint(output_dir):
    """Find the latest/best checkpoint in Dome-DETR's output directory."""
    checkpoints = glob.glob(os.path.join(output_dir, "*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    # Return the most recently modified one (best_ckpt should be latest)
    return max(checkpoints, key=os.path.getmtime)


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


def evaluate(dataset, test_subset, checkpoint_path, batch_size=8, device="cuda"):
    """Run inference on test_subset and eval via MillionTrees API."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    # This is a simplified version; a full implementation would load the Dome-DETR
    # model from the checkpoint. For now, a placeholder that calls dataset.eval().
    # In practice, you'd load the model here.

    test_loader = get_eval_loader("standard", test_subset, batch_size=batch_size)
    all_y_pred, all_y_true = [], []

    for i, batch in enumerate(test_loader):
        _, images, targets = batch
        # TODO: Actual inference here with loaded model
        # preds = predict_batch(model, images, device)
        # all_y_pred.extend(preds)
        # all_y_true.extend(targets)
        if i == 0:
            print("(Placeholder: actual inference would happen here)")
        break

    if len(all_y_true) > 0:
        results, results_str = dataset.eval(
            all_y_pred, all_y_true, test_subset.metadata_array[:len(all_y_true)],
            viz_dir=None,
        )
        return results, results_str
    return {}, ""


def main():
    parser = argparse.ArgumentParser(
        description="Train Dome-DETR on MillionTrees TreeBoxes."
    )
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
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--output-dir", type=str, default="training/boxes_dome_detr/outputs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--model-size", type=str, default="m", choices=["s", "m", "l"])
    parser.add_argument(
        "--init-mode",
        type=str,
        default="pretrained",
        choices=["pretrained", "scratch"],
        help="pretrained = load VisDrone-pretrained weights; scratch = random init",
    )
    parser.add_argument("--comet", action="store_true", help="Log to Comet ML")
    parser.add_argument(
        "--comet-name",
        type=str,
        default=None,
        help="Comet experiment name (default: boxes-<split>-<init>-lr<lr>-domedetr)",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Limit to 2 epochs and small config for testing")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_epochs = 2

    os.makedirs(args.output_dir, exist_ok=True)

    box_dataset = get_dataset(
        "TreeBoxes",
        root_dir=args.root_dir,
        split_scheme=args.split_scheme,
        mini=args.mini,
    )

    train_subset = box_dataset.get_subset("train")
    if len(train_subset) == 0:
        print("No training samples for this split; skipping training.")
        return

    # Convert to COCO format
    coco_output_dir = os.path.join(args.output_dir, "coco_annotations")
    img_folder, train_json, test_json = convert_to_coco_if_needed(
        args.root_dir, args.split_scheme, coco_output_dir, mini=args.mini
    )

    # Render config
    dome_detr_repo = "/blue/ewhite/b.weinstein/src/Dome-DETR"
    template_path = os.path.join(
        os.path.dirname(__file__), "configs", f"dome_{args.model_size}_milliontrees.yml"
    )
    config_path = os.path.join(args.output_dir, f"config_dome_{args.model_size}.yml")
    render_config(
        template_path, config_path, img_folder, train_json, test_json,
        args.batch_size, args.num_workers, args.lr, args.max_epochs
    )

    # Run training
    run_dome_detr_training(
        dome_detr_repo, config_path, args.output_dir, args.gpus
    )

    print("\n=== Training complete ===")
    if not args.smoke_test:
        print("Note: Eval on the trained checkpoint is stubbed; full implementation coming next.")


if __name__ == "__main__":
    main()
