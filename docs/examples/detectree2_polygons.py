import argparse
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detectree2 on MillionTrees TreePolygons.")
    parser.add_argument("--root-dir",
                        type=str,
                        default=os.environ.get("MT_ROOT", "data"),
                        help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Eval batch size")
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
                        help="Confidence threshold for polygons")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches")
    return parser.parse_args()


def to_pil_list(images: torch.Tensor) -> List[Image.Image]:
    pil_images: List[Image.Image] = []
    for img in images:
        im = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(im))
    return pil_images


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
    """
    Best-effort adapter to call Detectree2 for a single image path and return a GeoDataFrame
    of polygons with a 'score' column. This tries a few plausible internal APIs.
    """
    _try_import_detectree2()
    import importlib
    import geopandas as gpd

    # Try a few likely modules/functions from detectree2
    candidates: List[Tuple[str, str, Dict[str, Any]]] = [
        # (module, function, default kwargs)
        ("detectree2.models.predict", "predict_image", {"image_path": image_path, "model_path": weights_path}),
        ("detectree2.models.predict", "predict", {
            "input_path": image_path,
            "model_path": weights_path
        }),  # may return file
        ("detectree2.models.infer", "predict_image", {"image_path": image_path, "model_path": weights_path}),
        ("detectree2.models.infer", "predict", {
            "input_path": image_path,
            "model_path": weights_path
        }),
    ]

    last_err: Optional[Exception] = None
    for module_name, func_name, base_kwargs in candidates:
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name, None)
            if fn is None:
                continue
            # Try several reasonable kwarg shapes
            for kw in (
                base_kwargs,
                {**base_kwargs, "threshold": score_threshold},
                {**base_kwargs, "score_threshold": score_threshold},
                {**base_kwargs, "weights": weights_path},
            ):
                try:
                    out = fn(**kw)
                    # If the function returns a path to a vector file, read it
                    if isinstance(out, str) and os.path.exists(out):
                        try:
                            gdf = gpd.read_file(out)
                            if "score" not in gdf.columns:
                                gdf["score"] = 1.0
                            return gdf
                        except Exception:
                            pass
                    # If it returns a GeoDataFrame directly
                    try:
                        import geopandas as _  # noqa: F401
                        if hasattr(out, "geometry"):
                            gdf2 = out  # type: ignore
                            if "score" not in gdf2.columns:
                                gdf2["score"] = 1.0
                            return gdf2
                    except Exception:
                        pass
                except Exception as e2:  # try next kw form
                    last_err = e2
                    continue
        except Exception as e:
            last_err = e
            continue

    raise SystemExit(
        "Unable to call Detectree2 prediction API automatically. Please adjust the adapter "
        "in docs/examples/detectree2_polygons.py:_predict_polygons_for_image to match your installed Detectree2 version."
    ) from last_err


def rasterize_polygons_to_mask(
    polygons_gdf: "geopandas.GeoDataFrame",
    image_size_hw: Tuple[int, int],
) -> torch.Tensor:
    import numpy as np
    import rasterio.features
    H, W = image_size_hw
    masks: List[np.ndarray] = []
    geoms = list(polygons_gdf.geometry) if len(polygons_gdf) else []
    if len(geoms) == 0:
        return torch.zeros((0, H, W), dtype=torch.bool)
    for geom in geoms:
        mask = rasterio.features.rasterize([(geom, 1)],
                                           out_shape=(H, W),
                                           fill=0,
                                           all_touched=False,
                                           dtype=np.uint8)
        masks.append(mask.astype(np.bool_))
    if len(masks) == 0:
        return torch.zeros((0, H, W), dtype=torch.bool)
    return torch.from_numpy(np.stack(masks, axis=0))


def main() -> None:
    args = parse_args()

    dataset = get_dataset("TreePolygons",
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
        # For Detectree2, prefer using file paths; build absolute paths via dataset internals
        # metadata[:,0] is filename_id; map to code/basename
        basenames: List[str] = [dataset._filename_id_to_code[int(m[0])] for m in metadata]  # type: ignore
        image_dir = os.path.join(dataset._data_dir._str, "images")  # type: ignore
        image_paths = [os.path.join(image_dir, b) for b in basenames]

        for img_path, img_tensor, target in zip(image_paths, images, targets):
            # Predict polygons for this image
            gdf = _predict_polygons_for_image(img_path, args.model_weights,
                                              score_threshold=args.score_threshold)
            # Rasterize into instance masks
            H, W = int(img_tensor.shape[1]), int(img_tensor.shape[2])
            masks = rasterize_polygons_to_mask(gdf, (H, W))
            scores = torch.as_tensor(gdf["score"].values, dtype=torch.float32) if len(gdf) else torch.zeros(
                (0, ), dtype=torch.float32)
            labels = torch.zeros((masks.shape[0], ), dtype=torch.int64) if masks.ndim == 3 else torch.zeros(
                (0, ), dtype=torch.int64)
            y_pred = {"y": masks, "labels": labels, "scores": scores}
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


