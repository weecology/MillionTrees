import argparse
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detectree2 on MillionTrees TreeBoxes (envelope of polygons).")
    parser.add_argument("--root-dir",
                        type=str,
                        default=os.environ.get("MT_ROOT", "data"),
                        help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--mini", action="store_true", help="Use mini datasets for fast dev")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument("--split-scheme",
                        type=str,
                        default="random",
                        choices=["random", "zeroshot", "crossgeometry"],
                        help="Dataset split scheme")
    parser.add_argument("--model-weights",
                        type=str,
                        required=True,
                        help="Path to Detectree2 model weights (.pth). See model_garden in Detectree2.")
    parser.add_argument("--score-threshold",
                        type=float,
                        default=0.25,
                        help="Confidence threshold for polygons before boxing")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches")
    return parser.parse_args()


def _try_import_detectree2():
    try:
        import detectree2  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Detectree2 is required. Install with `uv sync --dev --extra detectree2` or `pip install -e .[detectree2]`."
        ) from exc


def _predict_polygons_for_image(
    image_path: str,
    weights_path: str,
    score_threshold: float = 0.25,
) -> "geopandas.GeoDataFrame":
    _try_import_detectree2()
    import importlib
    import geopandas as gpd

    candidates = [
        ("detectree2.models.predict", "predict_image", {"image_path": image_path, "model_path": weights_path}),
        ("detectree2.models.predict", "predict", {"input_path": image_path, "model_path": weights_path}),
        ("detectree2.models.infer", "predict_image", {"image_path": image_path, "model_path": weights_path}),
        ("detectree2.models.infer", "predict", {"input_path": image_path, "model_path": weights_path}),
    ]

    last_err: Optional[Exception] = None
    for module_name, func_name, base_kwargs in candidates:
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name, None)
            if fn is None:
                continue
            for kw in (
                base_kwargs,
                {**base_kwargs, "threshold": score_threshold},
                {**base_kwargs, "score_threshold": score_threshold},
                {**base_kwargs, "weights": weights_path},
            ):
                try:
                    out = fn(**kw)
                    if isinstance(out, str) and os.path.exists(out):
                        try:
                            gdf = gpd.read_file(out)
                            if "score" not in gdf.columns:
                                gdf["score"] = 1.0
                            return gdf
                        except Exception:
                            pass
                    if hasattr(out, "geometry"):
                        gdf2 = out  # type: ignore
                        if "score" not in gdf2.columns:
                            gdf2["score"] = 1.0
                        return gdf2
                except Exception as e2:
                    last_err = e2
                    continue
        except Exception as e:
            last_err = e
            continue

    raise SystemExit(
        "Unable to call Detectree2 prediction API automatically. Please adjust the adapter "
        "in docs/examples/detectree2_boxes.py:_predict_polygons_for_image to match your Detectree2 version."
    ) from last_err


def main() -> None:
    args = parse_args()

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

    all_y_pred: List[Dict[str, Any]] = []
    all_y_true: List[Dict[str, Any]] = []

    for b_idx, batch in enumerate(test_loader):
        metadata, images, targets = batch
        basenames: List[str] = [dataset._filename_id_to_code[int(m[0])] for m in metadata]  # type: ignore
        image_dir = os.path.join(dataset._data_dir._str, "images")  # type: ignore
        image_paths = [os.path.join(image_dir, b) for b in basenames]

        for img_path, target in zip(image_paths, targets):
            gdf = _predict_polygons_for_image(img_path, args.model_weights, score_threshold=args.score_threshold)
            if len(gdf) == 0:
                y_pred = {
                    "y": torch.empty((0, 4), dtype=torch.float32),
                    "labels": torch.empty((0, ), dtype=torch.int64),
                    "scores": torch.empty((0, ), dtype=torch.float32),
                }
            else:
                # Envelope to xyxy
                envelopes = gdf.geometry.envelope
                # bounds returns minx, miny, maxx, maxy
                xyxy = np.vstack([geom.bounds for geom in envelopes]).astype(np.float32)
                scores = gdf["score"].values.astype(np.float32) if "score" in gdf.columns else np.ones(
                    (len(gdf), ), dtype=np.float32)
                labels = np.zeros((len(gdf), ), dtype=np.int64)
                y_pred = {
                    "y": torch.from_numpy(xyxy),
                    "labels": torch.from_numpy(labels),
                    "scores": torch.from_numpy(scores),
                }
            all_y_pred.append(y_pred)
            all_y_true.append(target)

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    results, results_str = dataset.eval(all_y_pred,
                                        all_y_true,
                                        metadata=test_dataset.metadata_array[:len(all_y_true)])
    print(results_str)


if __name__ == "__main__":
    main()


