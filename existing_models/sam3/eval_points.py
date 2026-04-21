"""Evaluate SAM3 on MillionTrees TreePoints (text-prompted segmentation -> centroids)."""

import argparse
import json
import os
from typing import List

import numpy as np
import torch
from PIL import Image

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def to_pil_list(images: torch.Tensor) -> List[Image.Image]:
    return [Image.fromarray((img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
            for img in images]


def masks_to_centroids(masks: torch.Tensor) -> torch.Tensor:
    """Convert (N, H, W) boolean masks to (N, 2) centroid coordinates."""
    device = masks.device
    masks_u8 = masks.to(torch.uint8)
    H, W = masks_u8.shape[1], masks_u8.shape[2]
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    centroids = []
    for m in masks_u8:
        count = m.sum()
        if count == 0:
            continue
        centroids.append(torch.stack([(xs * m).sum() / count, (ys * m).sum() / count]))
    if not centroids:
        return torch.zeros((0, 2), dtype=torch.float32)
    return torch.stack(centroids).float()


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 on TreePoints (mask centroids).")
    parser.add_argument("--root-dir", type=str,
                        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-scheme", type=str, default="random",
                        choices=["random", "zeroshot", "crossgeometry"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--text-prompt", type=str, default="tree")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source prediction overlay PNGs")
    args = parser.parse_args()

    device = select_device(args.device)

    try:
        from transformers import Sam3Processor, Sam3Model  # type: ignore
    except Exception as exc:
        raise SystemExit("transformers with SAM3 support required.") from exc

    try:
        model = Sam3Model.from_pretrained("facebook/sam3", token=args.hf_token).to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=args.hf_token)
    except Exception as exc:
        raise SystemExit(f"Failed to load SAM3: {exc}") from exc

    dataset = get_dataset("TreePoints", root_dir=args.root_dir, download=args.download,
                          mini=args.mini, split_scheme=args.split_scheme)
    test_subset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    print(f"Batches: {len(test_loader)}")

    all_y_pred, all_y_true = [], []
    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        pil_images = to_pil_list(images)
        inputs = processor(images=pil_images, text=[args.text_prompt] * len(pil_images),
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=args.score_threshold, mask_threshold=args.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        for res, target in zip(results, targets):
            masks = res.get("masks", None)
            scores = res.get("scores", None)
            if masks is None or len(masks) == 0:
                y_pred = {"y": torch.zeros((0, 2), dtype=torch.float32),
                          "labels": torch.zeros((0,), dtype=torch.int64),
                          "scores": torch.zeros((0,), dtype=torch.float32)}
            else:
                masks_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
                centroids = masks_to_centroids(masks_t).cpu()
                scores_t = (torch.as_tensor(scores, dtype=torch.float32).cpu()
                            if scores is not None else torch.zeros(centroids.shape[0]))
                y_pred = {"y": centroids, "labels": torch.zeros(centroids.shape[0], dtype=torch.int64),
                          "scores": scores_t}

            all_y_pred.append(y_pred)
            all_y_true.append(target)

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    results, results_str = dataset.eval(
        all_y_pred, all_y_true, metadata=test_subset.metadata_array[:len(all_y_true)],
        viz_dir=args.viz_dir,
    )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"results_points_{args.split_scheme}.txt"), "w") as f:
            f.write(results_str)
        flat = {k: float(v) if hasattr(v, "item") else v
                for k, v in results.items() if isinstance(v, (int, float, torch.Tensor))}
        with open(os.path.join(args.output_dir, f"results_points_{args.split_scheme}.json"), "w") as f:
            json.dump({"model": "SAM3", "task": "TreePoints",
                       "split": args.split_scheme, "metrics": flat}, f, indent=2)


if __name__ == "__main__":
    main()
