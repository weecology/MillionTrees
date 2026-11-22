"""Evaluate CanopyGIS v1.8.1 on MillionTrees TreePolygons.

Pure zero-shot planar baseline using ``facebook/sam3``:
- Shadow-crevice geodesic watershed and 1D LUT RAG merging.
- Voronoi planar partition (``overlap_px == 0``, ``Euler == 1``).
- Pure-RGB zero-metadata mode (``gsd_m=None``).

Gated access to ``facebook/sam3`` is required; see README.md.
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.common.eval_sweep import (add_sweep_args, maybe_run_sweep,
                                            maybe_subsample)
from milliontrees.common.prediction_dump import add_dump_args, make_dumper
from milliontrees.datasets.polygon_stream_eval import (
    TreePolygonsStreamingEvalState, merge_viz_samples)

MODEL_ID = "facebook/sam3"
MODEL_NAME = "CanopyGIS-v1.8.1"


def select_device(device_arg: str) -> str:
    """Resolve 'auto' to 'cuda' when available, else 'cpu'."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_predictor(device: str, hf_token: Optional[str]):
    """Load the pretrained CanopyGIS v1.8.1 zero-shot SAM 3 predictor."""
    from canopygis_v18_eval import CanopyGISv18Predictor

    return CanopyGISv18Predictor.from_pretrained(
        model_id=MODEL_ID,
        device=device,
        hf_token=hf_token,
    )


def tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a MillionTrees CHW float [0, 1] RGB tensor to HWC uint8."""
    arr = image.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    return np.ascontiguousarray(np.rint(arr * 255.0).astype(np.uint8))


def load_native_rgb(dataset, filename: str) -> np.ndarray:
    """Read a raw RGB uint8 tile from disk at its native resolution."""
    img_path = dataset._data_dir / "images" / filename
    with Image.open(img_path) as pil_img:
        return np.asarray(pil_img.convert("RGB"), dtype=np.uint8)


def main() -> None:
    """Run CanopyGIS v1.8.1 evaluation on MillionTrees TreePolygons."""
    parser = argparse.ArgumentParser(
        description="CanopyGIS v1.8.1 pure zero-shot planar on TreePolygons.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=os.environ.get("MT_ROOT",
                               "/orange/ewhite/web/public/MillionTrees"),
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="MillionTrees TreePolygons dataset version (default: latest).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0 if os.name == "nt" else 2,
    )
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--split-scheme",
        type=str,
        default="within-distribution",
        choices=[
            "within-distribution",
            "out-of-distribution",
            "crossgeometry",
        ],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--resolution-mode",
        type=str,
        default="direct",
        choices=["direct", "native"],
        help=("'direct' runs on the resized loader tile; 'native' reads the "
              "raw tile from disk and resizes masks back to --image-size."),
    )
    parser.add_argument(
        "--no-remap-ws-scores",
        action="store_true",
        help="Disable watershed confidence score remapping.",
    )
    parser.add_argument("--hf-token",
                        type=str,
                        default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Directory for per-source prediction overlay PNGs.",
    )
    parser.add_argument(
        "--viz-n-per-source",
        type=int,
        default=10,
        help="Number of overlay PNGs to write per source.",
    )
    add_sweep_args(parser)
    add_dump_args(parser)
    args = parser.parse_args()

    device = select_device(args.device)
    predictor = load_predictor(device=device, hf_token=args.hf_token)

    dataset = get_dataset(
        "TreePolygons",
        version=args.version,
        root_dir=args.root_dir,
        download=args.download,
        mini=args.mini,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
        complete_tiles_only=args.complete_tiles_only,
    )
    test_subset = maybe_subsample(dataset, dataset.get_subset(args.eval_split),
                                  args)
    test_loader = get_eval_loader(
        "standard",
        test_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Batches: {len(test_loader)}")

    # Sweeping thresholds needs every prediction held in memory; a plain full
    # eval streams metrics batch-by-batch so peak RAM stays bounded to one
    # batch of dense masks.
    sweep_mode = getattr(args, "sweep", False)
    stream_eval = None if sweep_mode else TreePolygonsStreamingEvalState(
        dataset)
    all_y_pred: List[dict] = []
    all_y_true: List[dict] = []

    dumper = make_dumper(args, dataset, model=MODEL_NAME, task="TreePolygons")

    viz_cap: dict = {}
    viz_y_pred: List[dict] = []
    viz_y_true: List[dict] = []
    viz_rows: List[torch.Tensor] = []

    remap_ws_scores = not args.no_remap_ws_scores
    filename_idx = (dataset.metadata_fields.index("filename_id")
                    if "filename_id" in dataset.metadata_fields else 0)

    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        batch_y_pred: List[dict] = []
        batch_y_true: List[dict] = []

        for meta_row, img, target in zip(metadata, images, targets):
            output_hw: Optional[Tuple[int, int]]
            if args.resolution_mode == "native":
                filename_id = int(meta_row[filename_idx].item())
                filename = dataset._filename_id_to_code[filename_id]
                img_rgb = load_native_rgb(dataset, filename)
                output_hw = (args.image_size, args.image_size)
            else:
                img_rgb = tensor_to_rgb_uint8(img)
                output_hw = None

            pred = predictor.predict_tile(
                img_rgb,
                gsd_m=None,
                score_threshold=args.score_threshold,
                remap_ws_scores=remap_ws_scores,
                output_hw=output_hw,
            )
            batch_y_pred.append(pred)
            batch_y_true.append(target)

        batch_meta = metadata[:len(batch_y_pred)]
        if dumper is not None:
            dumper.update(batch_y_pred, batch_y_true, batch_meta)
        if sweep_mode:
            all_y_pred.extend(batch_y_pred)
            all_y_true.extend(batch_y_true)
        else:
            stream_eval.update(batch_y_pred, batch_y_true, batch_meta)
            if args.viz_dir:
                merge_viz_samples(
                    viz_cap,
                    batch_meta,
                    batch_y_pred,
                    batch_y_true,
                    viz_y_pred=viz_y_pred,
                    viz_y_true=viz_y_true,
                    viz_rows=viz_rows,
                    n_per_source=args.viz_n_per_source,
                )

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    if dumper is not None:
        dumper.close()

    if sweep_mode:
        if maybe_run_sweep(
                args,
                dataset,
                test_subset,
                all_y_pred,
                all_y_true,
                model=MODEL_NAME,
                task="TreePolygons",
        ):
            return
        results, results_str = dataset.eval(
            all_y_pred,
            all_y_true,
            metadata=test_subset.metadata_array[:len(all_y_true)],
            viz_dir=args.viz_dir,
        )
    else:
        results, results_str = stream_eval.finalize(
            viz_dir=args.viz_dir,
            viz_y_pred=viz_y_pred or None,
            viz_y_true=viz_y_true or None,
            viz_metadata=torch.stack(viz_rows) if viz_rows else None,
            viz_n_per_source=args.viz_n_per_source,
        )
    print(results_str)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        split_tag = (args.split_scheme
                     if args.eval_split == "test" else args.eval_split)
        txt_path = os.path.join(args.output_dir,
                                f"results_polygons_{split_tag}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(results_str)
        flat = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in results.items()
            if isinstance(v, (int, float, torch.Tensor))
        }
        json_path = os.path.join(args.output_dir,
                                 f"results_polygons_{split_tag}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": MODEL_NAME,
                    "task": "TreePolygons",
                    "split": args.split_scheme,
                    "eval_split": args.eval_split,
                    "resolution_mode": args.resolution_mode,
                    "metrics": flat,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
