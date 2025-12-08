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
    parser.add_argument("--split-scheme",
                        type=str,
                        default="random",
                        choices=["random", "zeroshot", "crossgeometry"],
                        help="Dataset split scheme")
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

    # Always use Transformers backend
    try:
        from transformers import Sam3Processor, Sam3Model  # type: ignore
    except Exception as exc:
        raise SystemExit("Transformers with SAM3 is required. Install extras and try again.") from exc

    dataset = get_dataset("TreeBoxes",
                          root_dir=args.root_dir,
                          download=args.download,
                          mini=args.mini,
                          split_scheme=args.split_scheme)
    test_dataset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard",
                                  test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    # Load model and processor from HF
    try:
        try:
            model = Sam3Model.from_pretrained("facebook/sam3", token=args.hf_token).to(device)
            processor = Sam3Processor.from_pretrained("facebook/sam3", token=args.hf_token)
        except Exception:
            model = Sam3Model.from_pretrained("facebook/sam3", use_auth_token=args.hf_token).to(device)
            processor = Sam3Processor.from_pretrained("facebook/sam3", use_auth_token=args.hf_token)
    except Exception as exc:
        raise SystemExit(
            f"Unable to initialize SAM3 (transformers): {exc}. "
            "Accept the model terms and ensure HF_TOKEN is valid: https://huggingface.co/facebook/sam3"
        )

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
            boxes_out = res.get("boxes", None)
            scores = res.get("scores", None)
            if (boxes_out is None or len(boxes_out) == 0) and (masks is None or len(masks) == 0):
                y_pred = {
                    "y": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }
            else:
                if boxes_out is not None and len(boxes_out) > 0:
                    boxes_t = torch.as_tensor(boxes_out, dtype=torch.float32).detach().to("cpu")
                else:
                    masks_t = torch.as_tensor(masks, dtype=torch.uint8, device=device)
                    # Normalize mask shape to [N, H, W]
                    if masks_t.dim() == 2:
                        masks_t = masks_t.unsqueeze(0)
                    if masks_t.dim() == 4 and masks_t.shape[1] == 1:
                        masks_t = masks_t[:, 0]
                    boxes_t = masks_to_boxes(masks_t).detach().to("cpu")  # Nx4
                scores_t = torch.as_tensor(scores, dtype=torch.float32).detach().to("cpu") if scores is not None else torch.zeros((boxes_t.shape[0],), dtype=torch.float32)
                labels_t = torch.zeros((boxes_t.shape[0],), dtype=torch.int64)
                y_pred = {"y": boxes_t, "labels": labels_t, "scores": scores_t}

            all_y_pred.append(y_pred)
            all_y_true.append(target)

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    results, results_str = dataset.eval(
        all_y_pred, all_y_true, metadata=test_dataset.metadata_array[:len(all_y_true)]
    )
    print(results_str)


if __name__ == "__main__":
    main()


