from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from milliontrees import get_dataset

from annotation_loop import LABEL_CONFIGS
from label_studio_utils import create_sftp_client, upload_to_label_studio


SUPPORTED_DATASETS = ("TreeBoxes", "TreePoints")


def _as_numpy(array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _prepare_tree_mask(tree_mask, image_shape: tuple[int, int]) -> np.ndarray:
    mask = _as_numpy(tree_mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0)
    if mask.shape != image_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image_shape}")
    return mask.astype(bool)


def _box_tree_fraction(box: np.ndarray, tree_mask: np.ndarray) -> float:
    height, width = tree_mask.shape
    x1 = int(np.floor(box[0]))
    y1 = int(np.floor(box[1]))
    x2 = int(np.ceil(box[2]))
    y2 = int(np.ceil(box[3]))
    x1 = min(max(x1, 0), width)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height)
    y2 = min(max(y2, 0), height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = tree_mask[y1:y2, x1:x2]
    return float(crop.mean())


def _point_tree_fraction(point: np.ndarray, tree_mask: np.ndarray) -> float:
    height, width = tree_mask.shape
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    return float(tree_mask[y, x])


def _flagged_preannotations(dataset_type: str,
                            filename: str,
                            flagged_geometries: list[np.ndarray]) -> pd.DataFrame:
    if dataset_type == "TreeBoxes":
        boxes = np.asarray(flagged_geometries, dtype=np.float32)
        return pd.DataFrame({
            "image_path": filename,
            "xmin": boxes[:, 0],
            "ymin": boxes[:, 1],
            "xmax": boxes[:, 2],
            "ymax": boxes[:, 3],
            "label": "Tree",
        })
    if dataset_type == "TreePoints":
        points = np.asarray(flagged_geometries, dtype=np.float32)
        return pd.DataFrame({
            "image_path": filename,
            "x": points[:, 0],
            "y": points[:, 1],
            "label": "Tree",
        })
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def collect_low_tree_coverage_cases(dataset_type: str,
                                    version: str | None,
                                    root_dir: str,
                                    split_scheme: str,
                                    threshold: float,
                                    include_unsupervised: bool,
                                    download: bool) -> tuple[object, pd.DataFrame, dict[str, list[np.ndarray]]]:
    dataset = get_dataset(
        dataset_type,
        version=version,
        root_dir=root_dir,
        split_scheme=split_scheme,
        include_unsupervised=include_unsupervised,
        download=download,
        verbose=False,
    )

    records: list[dict] = []
    flagged_by_image: dict[str, list[np.ndarray]] = {}

    for idx in range(len(dataset)):
        metadata, image, targets = dataset[idx]
        tree_mask_raw = targets.get("tree_coverage_mask")
        if tree_mask_raw is None:
            raise ValueError(
                f"{dataset_type} sample {idx} is missing tree_coverage_mask. "
                "Run against a packaged dataset that includes masks.")

        tree_mask = _prepare_tree_mask(tree_mask_raw, image.shape[:2])
        gt_geometries = _as_numpy(targets["y"])

        if dataset_type == "TreeBoxes":
            gt_geometries = np.asarray(gt_geometries, dtype=np.float32).reshape(
                -1, 4)
        elif dataset_type == "TreePoints":
            gt_geometries = np.asarray(gt_geometries, dtype=np.float32).reshape(
                -1, 2)

        filename_id = int(metadata[0])
        source_id = int(metadata[1])
        filename = dataset._filename_id_to_code[filename_id]
        source = dataset._source_id_to_code[source_id]

        for gt_index, geometry in enumerate(gt_geometries):
            if dataset_type == "TreeBoxes":
                tree_fraction = _box_tree_fraction(geometry, tree_mask)
                flagged = tree_fraction < threshold
            else:
                tree_fraction = _point_tree_fraction(geometry, tree_mask)
                flagged = tree_fraction == 0.0

            if not flagged:
                continue

            flagged_by_image.setdefault(filename, []).append(geometry)
            records.append({
                "dataset_type": dataset_type,
                "split_scheme": split_scheme,
                "source": source,
                "filename": filename,
                "gt_index": gt_index,
                "tree_fraction": tree_fraction,
                "threshold": threshold,
            })

    results = pd.DataFrame(records)
    return dataset, results, flagged_by_image


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Find GT annotations with low tree-mask overlap and upload flagged images to Label Studio."
    )
    parser.add_argument("--dataset-types",
                        nargs="+",
                        choices=SUPPORTED_DATASETS,
                        default=list(SUPPORTED_DATASETS))
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--root-dir", type=str, default="data")
    parser.add_argument("--split-scheme", type=str, default="within-distribution")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--include-unsupervised", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--project-prefix",
        type=str,
        default="MillionTrees-LowMaskCoverage",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data_prep") / "annotations",
    )
    parser.add_argument(
        "--label-studio-url",
        type=str,
        default=os.getenv("LABEL_STUDIO_URL"),
    )
    parser.add_argument(
        "--label-studio-folder",
        type=str,
        default=os.getenv("LABEL_STUDIO_DATA_DIR"),
    )
    parser.add_argument("--sftp-user", type=str, default=os.getenv("USER"))
    parser.add_argument("--sftp-host", type=str, default=os.getenv("HOST"))
    parser.add_argument("--sftp-key",
                        type=str,
                        default=os.getenv("KEY_FILENAME"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    all_reports: list[pd.DataFrame] = []
    upload_plan: list[tuple[str, list[Path], list[pd.DataFrame], Path]] = []

    for dataset_type in args.dataset_types:
        dataset, report_df, flagged_by_image = collect_low_tree_coverage_cases(
            dataset_type=dataset_type,
            version=args.version,
            root_dir=args.root_dir,
            split_scheme=args.split_scheme,
            threshold=args.threshold,
            include_unsupervised=args.include_unsupervised,
            download=args.download,
        )

        report_path = args.report_dir / (
            f"{dataset_type}_{args.split_scheme}_low_tree_coverage_{_timestamp()}.csv"
        )
        report_df.to_csv(report_path, index=False)
        print(
            f"{dataset_type}: {len(report_df)} flagged annotations across "
            f"{len(flagged_by_image)} images. Report: {report_path}")

        all_reports.append(report_df)
        if report_df.empty:
            continue

        image_names = sorted(flagged_by_image.keys())
        if args.max_images is not None:
            image_names = image_names[:args.max_images]

        images = [dataset._data_dir / "images" / name for name in image_names]
        preannotations = [
            _flagged_preannotations(dataset_type, name, flagged_by_image[name])
            for name in image_names
        ]
        upload_plan.append((dataset_type, images, preannotations,
                            dataset._data_dir / "images"))

    if all_reports:
        combined = pd.concat(all_reports, ignore_index=True)
        summary_path = args.report_dir / (
            f"all_datasets_low_tree_coverage_{_timestamp()}.csv")
        combined.to_csv(summary_path, index=False)
        print(f"Combined report: {summary_path}")

    if args.dry_run:
        print("Dry run enabled; skipping Label Studio upload.")
        return

    if not upload_plan:
        print("No flagged images to upload.")
        return

    if args.label_studio_url is None:
        raise ValueError("Missing Label Studio URL. Set --label-studio-url.")
    if args.label_studio_folder is None:
        raise ValueError(
            "Missing Label Studio data folder. Set --label-studio-folder.")
    if args.sftp_key is None:
        raise ValueError("Missing SSH key path. Set --sftp-key.")

    sftp_client = create_sftp_client(
        user=args.sftp_user,
        host=args.sftp_host,
        key_filename=os.path.expanduser(args.sftp_key),
    )

    for dataset_type, images, preannotations, image_dir in upload_plan:
        project_name = (
            f"{args.project_prefix}-{dataset_type}-{args.split_scheme}")
        upload_to_label_studio(
            images=images,
            sftp_client=sftp_client,
            dataset_type=dataset_type,
            url=args.label_studio_url,
            project_name=project_name,
            images_to_annotate_dir=image_dir,
            folder_name=args.label_studio_folder,
            preannotations=preannotations,
            batch_size=args.batch_size,
        )
        print(f"Uploaded {len(images)} images to {project_name}")


if __name__ == "__main__":
    main()
