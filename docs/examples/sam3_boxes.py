import argparse
import os
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import masks_to_boxes

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 -> Boxes on MillionTrees TreeBoxes (mask->box per instance).")
    parser.add_argument("--root-dir", type=str, default=os.environ.get("MT_ROOT", "data"), help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--mini", action="store_true", help="Use mini datasets for fast dev")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--text-prompt", type=str, default="tree", help="Open-vocabulary text prompt")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold for masks")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face token for gated model")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def to_pil_list(images: torch.Tensor) -> List[Image.Image]:
    pil_images: List[Image.Image] = []
    for img in images:
        im = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(im))
    return pil_images


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    try:
        from transformers import Sam3Processor, Sam3Model  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Transformers with SAM3 support is required. Install with `pip install -e .[sam3]`."
        ) from exc

    dataset = get_dataset("TreeBoxes",
                          root_dir=args.root_dir,
                          download=args.download,
                          mini=args.mini)
    test_dataset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard",
                                  test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    try:
        model = Sam3Model.from_pretrained("facebook/sam3", token=args.hf_token).to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=args.hf_token)
    except Exception as exc:
        raise SystemExit(
            "Unable to load facebook/sam3. Accept the terms and set HF_TOKEN. See https://huggingface.co/facebook/sam3"
        ) from exc

    all_y_pred: List[Dict[str, Any]] = []
    all_y_true: List[Dict[str, Any]] = []

    for b_idx, batch in enumerate(test_loader):
        metadata, images, targets = batch
        pil_images = to_pil_list(images)
        texts = [args.text_prompt] * len(pil_images)

        inputs = processor(images=pil_images, text=texts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        for res, target in zip(results, targets):
            masks = res.get("masks", None)
            scores = res.get("scores", None)
            if masks is None or len(masks) == 0:
                y_pred = {
                    "y": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }
            else:
                masks_t = torch.as_tensor(masks, dtype=torch.bool)
                boxes = masks_to_boxes(masks_t.to(torch.uint8))  # Nx4
                scores_t = torch.as_tensor(scores, dtype=torch.float32)
                labels_t = torch.zeros((boxes.shape[0],), dtype=torch.int64)
                y_pred = {"y": boxes, "labels": labels_t, "scores": scores_t}

            all_y_pred.append(y_pred)
            all_y_true.append(target)

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    results, results_str = dataset.eval(all_y_pred, all_y_true, metadata=test_dataset.metadata_array)
    print(results_str)


if __name__ == "__main__":
    main()


