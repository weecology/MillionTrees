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
    parser = argparse.ArgumentParser(description="SAM3 on MillionTrees TreePolygons (PCS with text prompt).")
    parser.add_argument("--root-dir", type=str, default=os.environ.get("MT_ROOT", "data"), help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--backend", type=str, choices=["native", "transformers"], default="native", help="SAM3 backend (default: native)")
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

    # Lazy import with selectable backend
    use_transformers = (args.backend == "transformers")
    if use_transformers:
        try:
            from transformers import Sam3Processor, Sam3Model  # type: ignore
        except Exception:
            use_transformers = False
    if not use_transformers:
        try:
            from sam3.model_builder import build_sam3_image_model  # type: ignore
            from sam3.model.sam3_image_processor import Sam3Processor as NativeSam3Processor  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "SAM3 is required (either Transformers integration or native package). "
                "Install with `pip install -e .[sam3]`."
            ) from exc

    # Load dataset and test split (mini recommended)
    dataset = get_dataset("TreePolygons",
                          root_dir=args.root_dir,
                          download=args.download,
                          mini=args.mini)
    test_dataset = dataset.get_subset("test")
    test_loader = get_eval_loader("standard",
                                  test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    # Load model and processor
    try:
        if use_transformers:
            try:
                model = Sam3Model.from_pretrained("facebook/sam3", token=args.hf_token).to(device)
                processor = Sam3Processor.from_pretrained("facebook/sam3", token=args.hf_token)
            except Exception:
                model = Sam3Model.from_pretrained("facebook/sam3", use_auth_token=args.hf_token).to(device)
                processor = Sam3Processor.from_pretrained("facebook/sam3", use_auth_token=args.hf_token)
        else:
            model = build_sam3_image_model()
            processor = NativeSam3Processor(model)
    except Exception as exc:
        raise SystemExit(
            f"Unable to initialize SAM3 backend ({'transformers' if use_transformers else 'native'}): {exc}. "
            "If using Transformers, accept the model terms and ensure HF_TOKEN is valid: https://huggingface.co/facebook/sam3"
        )

    all_y_pred: List[Dict[str, Any]] = []
    all_y_true: List[Dict[str, Any]] = []

    for b_idx, batch in enumerate(test_loader):
        metadata, images, targets = batch
        pil_images = to_pil_list(images)
        texts = [args.text_prompt] * len(pil_images)

        if use_transformers:
            inputs = processor(images=pil_images, text=texts, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )
        else:
            results = []
            for im in pil_images:
                state = processor.set_image(im)
                out = processor.set_text_prompt(state=state, prompt=args.text_prompt)
                results.append({
                    "masks": out.get("masks", []),
                    "scores": out.get("scores", []),
                })

        for res, target in zip(results, targets):
            masks = res.get("masks", None)
            scores = res.get("scores", None)
            if masks is None or len(masks) == 0:
                y_pred = {
                    "y": torch.zeros((0, 0, 0), dtype=torch.bool),  # empty
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                }
            else:
                # masks: list of HxW arrays or a tensor shaped (N, H, W)
                masks_t = torch.as_tensor(masks, dtype=torch.uint8, device=device)
                if masks_t.dim() == 2:
                    masks_t = masks_t.unsqueeze(0)
                if masks_t.dim() == 4 and masks_t.shape[1] == 1:
                    masks_t = masks_t[:, 0]
                masks_t = masks_t.bool().detach().to("cpu")
                scores_t = torch.as_tensor(scores, dtype=torch.float32).detach().to("cpu")
                labels_t = torch.zeros((masks_t.shape[0],), dtype=torch.int64)
                y_pred = {"y": masks_t, "labels": labels_t, "scores": scores_t}

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


