import fnmatch
import os
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from shapely import from_wkt
import torch
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.ops import masks_to_boxes

from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.eval_visualization import save_eval_visualizations
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import (
    CountingError,
    MaskAccuracy,
    DetectionMAP,
    MaskAwareMaskPrecision,
    MergeCommissionMetric,
)
from milliontrees.common.onboarding import print_dataset_summary


class TreePolygonsDataset(MillionTreesDataset):
    """The TreePolygons dataset is a collection of tree annotations annotated as multi-point polygon
    locations.

    The dataset is comprised of many sources from across the world.

    Dataset Splits:
        - Within-distribution: For each source, 80% of the data is used for training and 20% for testing.
        - crossgeometry: Boxes and Points are used to predict polygons.
        - out-of-distribution: Selected sources are entirely held out for testing.

    Input (x):
        RGB aerial images.

    Label (y):
        y is an n x 2-dimensional vector where each line represents a point coordinate (x, y).

    Metadata:
        Each image is annotated with the following metadata:
            - location (int): location id
            - source (int): source id

    License:
        This dataset is distributed under the Creative Commons Attribution License.
    """

    _dataset_name = 'TreePolygons'
    _versions_dict = {
        '0.0': {
            'download_url': '',
            'supervised_download_url': '',
            'compressed_size': 105525592
        },
        "0.17": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.17.zip",
            'supervised_download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_supervised_v0.17.zip",
            'compressed_size':
                118069539057
        },
        "0.18": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.18.zip",
            'supervised_download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_supervised_v0.18.zip",
            'compressed_size':
                120747553994
        },
        "0.19": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.19.zip",
            'supervised_download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_supervised_v0.19.zip",
            # TODO: refresh with the real zip size once v0.19 zips are built;
            # unused for local download=False training/eval runs.
            'compressed_size':
                120747553994
        }
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='within-distribution',
                 geometry_name='y',
                 eval_score_threshold=0.5,
                 image_size=448,
                 remove_incomplete=False,
                 include_sources=None,
                 exclude_sources=None,
                 mini=False,
                 small=False,
                 verbose=True,
                 include_unsupervised=False):

        if mini and small:
            raise ValueError(
                'At most one of mini=True and small=True may be set.')

        self._version = version
        self._split_scheme = split_scheme
        self.geometry_name = geometry_name
        self.image_size = image_size
        self.eval_score_threshold = eval_score_threshold
        self.mini = mini
        self.small = small
        self.verbose = verbose
        self.include_unsupervised = include_unsupervised

        self._collate = TreePolygonsDataset._collate_fn

        if self._split_scheme not in [
                'within-distribution', 'crossgeometry', 'out-of-distribution'
        ]:
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')

        if mini:
            self._versions_dict = self._get_mini_versions_dict()
        elif small:
            self._versions_dict = self._get_small_versions_dict()

        # Select supervised-only dataset by default (smaller download).
        # Users must opt in with include_unsupervised=True to get the full dataset.
        if not include_unsupervised:
            modified_versions = {}
            for v, info in self._versions_dict.items():
                modified_info = dict(info)
                if info.get('supervised_download_url') is not None:
                    modified_info['download_url'] = info[
                        'supervised_download_url']
                modified_versions[v] = modified_info
            self._versions_dict = modified_versions
            if small:
                self._dataset_name = 'SmallTreePolygons'
            else:
                self._dataset_name = 'TreePolygons_supervised'

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Restore dataset name for proper operation after directory setup
        self._dataset_name = 'TreePolygons'

        # Load splits
        df = pd.read_csv(self._data_dir / '{}.csv'.format(split_scheme))

        # Cache available sources for convenience
        self.sources = df['source'].unique()
        available_source_count = len(self.sources)

        # Remove incomplete data based on flag
        if remove_incomplete:
            df = df[df['complete'] == True]

        # Filter by include/exclude source names with wildcard support
        # Default: exclude sources containing 'unsupervised' unless include_unsupervised=True
        include_patterns = None
        if include_sources is not None and include_sources != []:
            include_patterns = include_sources if isinstance(
                include_sources, (list, tuple)) else [include_sources]
        exclude_patterns = exclude_sources
        if exclude_patterns is None:
            exclude_patterns = [] if include_unsupervised else [
                '*unsupervised*'
            ]
        elif not isinstance(exclude_patterns, (list, tuple)):
            exclude_patterns = [exclude_patterns]

        source_str = df['source'].astype(str).str.lower()

        if include_patterns is not None:
            patterns_lower = [p.lower() for p in include_patterns]
            mask_include = source_str.apply(
                lambda s: any(fnmatch.fnmatch(s, p) for p in patterns_lower))
            df = df[mask_include]

        patterns_exclude_lower = [p.lower() for p in exclude_patterns]
        if len(patterns_exclude_lower) > 0:
            mask_exclude = source_str.apply(lambda s: any(
                fnmatch.fnmatch(s, p) for p in patterns_exclude_lower))
            df = df[~mask_exclude]
        selected_source_count = df['source'].nunique()
        df = df.reset_index(drop=True)

        # Splits
        self._split_dict = {
            'train': 0,
            'validation': 1,
            'test': 2,
        }
        self._split_names = {
            'train': 'Train',
            'validation': 'Validation',
            'test': 'Test',
        }

        unique_files = df.drop_duplicates(subset=['filename'],
                                          inplace=False).reset_index(drop=True)
        unique_files['split_id'] = unique_files['split'].apply(
            lambda x: self._split_dict[x])
        self._split_array = unique_files['split_id'].values

        # Filenames
        self._input_array = unique_files.filename

        # Create lookup table for which index to select for each filename
        self._input_lookup = df.groupby('filename').apply(
            lambda x: x.index.values, include_groups=False).to_dict()

        # Convert each polygon to shapely objects
        df['polygon'] = df['polygon'].apply(from_wkt)

        self._y_array = list(df['polygon'].values)

        # Labels -> just 'Tree'
        self._n_classes = 1

        # Not clear what this is, since we have a polygon, unknown size
        self._y_size = 4

        # Create source locations with a numeric ID
        df["source_id"] = df.source.astype('category').cat.codes

        # Create filename numeric ID
        df["filename_id"] = df.filename.astype('category').cat.codes

        # Create dictionary for codes to names
        self._source_id_to_code = df.set_index('source_id')['source'].to_dict()
        self._filename_id_to_code = df.set_index(
            'filename_id')['filename'].to_dict()

        # Expose source names to the grouper so per-source eval lines print the
        # source name instead of the numeric source_id. Ordered by source_id
        # (contiguous 0..n-1 from pandas category codes).
        self._metadata_map = {
            'source_id': [
                self._source_id_to_code[i]
                for i in sorted(self._source_id_to_code)
            ]
        }

        # Location/group info
        n_groups = max(df['source_id']) + 1
        self._n_groups = n_groups
        assert len(np.unique(df['source_id'])) == self._n_groups

        # Metadata is at the image level
        unique_sources = df[['filename_id', 'source_id']].drop_duplicates(
            subset="filename_id", inplace=False).reset_index(drop=True)
        self._metadata_array = torch.tensor(unique_sources.values.astype('int'))
        self._metadata_fields = ['filename_id', 'source_id']

        if 'complete' in df.columns:
            source_complete = df.groupby('source_id')['complete'].first()
            self._source_id_complete = {
                int(k): bool(v) for k, v in source_complete.items()
            }
        else:
            self._source_id_complete = {}

        # eval grouper
        self.metrics = self.build_metrics(self.eval_score_threshold)
        self._eval_grouper = CombinatorialGrouper(dataset=self,
                                                  groupby_fields=(['source_id'
                                                                  ]))

        if self.verbose:
            n_train_images = int(
                (self._split_array == self._split_dict['train']).sum())
            n_test_images = int(
                (self._split_array == self._split_dict['test']).sum())
            print_dataset_summary(
                dataset_name=self._dataset_name,
                version=self.version,
                data_dir=self._data_dir,
                split_scheme=self._split_scheme,
                n_annotations=len(df),
                n_total_images=len(unique_files),
                n_train_images=n_train_images,
                n_test_images=n_test_images,
                n_available_sources=available_source_count,
                n_selected_sources=selected_source_count,
                mini=self.mini,
                small=self.small,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )

        super().__init__(root_dir, download, split_scheme)

    def __getitem__(self, idx):
        """
        Args:
            - idx (int): Index of a data point

        Output:
            - metadata (Tensor): Metadata of the idx-th data point
            - x (np.ndarray): Input features of the idx-th data point
            - targets (dict): Dictionary containing:
                - "y" (Tensor): Masks of the polygons
                - "bboxes" (BoundingBoxes): Bounding boxes of the polygons
                - "labels" (np.ndarray): Labels for each mask (all zeros in this case)
        """
        x_full = self.get_input(idx)
        orig_h, orig_w = x_full.shape[:2]

        # Load tree coverage mask at original resolution first (it validates shape).
        tree_coverage_mask = self.get_tree_coverage_mask(idx, (orig_h, orig_w))

        # Pre-resize image and any auxiliary masks to target resolution so per-polygon
        # rasterization happens at 448x448 instead of full image resolution.
        target_size = self.image_size
        if orig_h != target_size or orig_w != target_size:
            x = cv2.resize(x_full, (target_size, target_size),
                           interpolation=cv2.INTER_LINEAR)
            if tree_coverage_mask is not None:
                tree_coverage_mask = cv2.resize(tree_coverage_mask,
                                                (target_size, target_size),
                                                interpolation=cv2.INTER_NEAREST)
            scale_x = target_size / orig_w
            scale_y = target_size / orig_h
        else:
            x = x_full
            scale_x = 1.0
            scale_y = 1.0

        y_indices = self._input_lookup[self._input_array[idx]]
        y_polygons = [self._y_array[i] for i in y_indices]

        # Handle empty case (no polygons for this image)
        if len(y_polygons) == 0:
            bboxes = BoundingBoxes(data=torch.zeros(0, 4, dtype=torch.float32),
                                   format='xyxy',
                                   canvas_size=x.shape[:2])
            masks = np.zeros((0, x.shape[0], x.shape[1]), dtype=np.uint8)
        else:
            mask_imgs = [
                self.create_polygon_mask(width=x.shape[1],
                                         height=x.shape[0],
                                         vertices=y_polygon,
                                         scale_x=scale_x,
                                         scale_y=scale_y)
                for y_polygon in y_polygons
            ]
            masks = torch.stack([Mask(mask_img) for mask_img in mask_imgs])

            # Filter out empty masks (all zeros) before calling masks_to_boxes
            # This can happen if polygons are outside image bounds or degenerate
            if len(masks) > 0:
                # Check which masks are non-empty (have at least one non-zero pixel)
                mask_sums = masks.sum(dim=(1, 2))  # Sum over H and W dimensions
                non_empty_mask = mask_sums > 0
                masks = masks[non_empty_mask]

            # Handle empty case (no valid masks)
            if len(masks) == 0:
                bboxes = BoundingBoxes(data=torch.zeros(0,
                                                        4,
                                                        dtype=torch.float32),
                                       format='xyxy',
                                       canvas_size=x.shape[:2])
                masks = np.zeros((0, x.shape[0], x.shape[1]), dtype=np.uint8)
            else:
                bbox_data = masks_to_boxes(masks)

                # Filter out degenerate bboxes (zero width or height)
                # This can happen when polygons are very thin and round to same pixel
                valid_mask = (bbox_data[:, 2] > bbox_data[:, 0]) & (
                    bbox_data[:, 3] > bbox_data[:, 1])

                if not valid_mask.all():
                    # Filter out invalid bboxes and corresponding masks/labels
                    bbox_data = bbox_data[valid_mask]
                    masks = masks[valid_mask]

                # Handle empty case (all masks filtered out)
                if len(bbox_data) == 0:
                    bboxes = BoundingBoxes(data=torch.zeros(
                        0, 4, dtype=torch.float32),
                                           format='xyxy',
                                           canvas_size=x.shape[:2])
                    masks = np.zeros((0, x.shape[0], x.shape[1]),
                                     dtype=np.uint8)
                else:
                    bboxes = BoundingBoxes(data=bbox_data,
                                           format='xyxy',
                                           canvas_size=x.shape[:2])
                    masks = np.stack([mask.numpy() for mask in masks])

        metadata = self._metadata_array[idx]
        targets = {
            "y": masks,
            "bboxes": bboxes,
            "labels": np.zeros(len(masks), dtype=int)
        }
        if tree_coverage_mask is not None:
            targets["tree_coverage_mask"] = tree_coverage_mask

        return metadata, x, targets

    def create_polygon_mask(self,
                            width,
                            height,
                            vertices,
                            scale_x=1.0,
                            scale_y=1.0):
        """Rasterize a shapely polygon to a binary mask at the given (width, height).

        Vertex coordinates are multiplied by (scale_x, scale_y) so a polygon defined on the original
        image can be drawn directly at a downscaled target size, avoiding allocation of a full-
        resolution mask.
        """
        mask_img = np.zeros((height, width), dtype=np.uint8)
        coords = vertices.exterior.coords._coords
        if scale_x == 1.0 and scale_y == 1.0:
            pts = np.asarray(coords, dtype=np.int32)
        else:
            pts = np.asarray(coords, dtype=np.float64)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(np.int32)
        cv2.fillPoly(mask_img, [pts], 255)
        return mask_img

    def build_metrics(self, score_threshold):
        """Construct the evaluation metric objects at a given score threshold.

        Each metric filters predictions by ``scores >= score_threshold``, so the threshold is baked
        in at construction. Factored out so callers (e.g. a threshold sweep) can build independent
        metric sets per threshold without reconstructing the whole dataset.
        """
        return {
            "accuracy":
                MaskAccuracy(geometry_name=self.geometry_name,
                             score_threshold=score_threshold,
                             metric="accuracy"),
            "recall":
                MaskAccuracy(geometry_name=self.geometry_name,
                             score_threshold=score_threshold,
                             metric="recall"),
            "maskaware_precision":
                MaskAwareMaskPrecision(geometry_name=self.geometry_name,
                                       score_threshold=score_threshold),
            "AP50":
                DetectionMAP(geometry_name=self.geometry_name,
                             score_threshold=score_threshold,
                             iou_type="segm",
                             iou_thresholds=[0.5],
                             max_detection_thresholds=[1, 10, 1000]),
            "merge_commission":
                MergeCommissionMetric(
                    geometry_name=self.geometry_name,
                    score_threshold=score_threshold,
                    modality="mask",
                ),
            "counting_mae":
                CountingError(
                    score_threshold=score_threshold,
                    geometry_name=self.geometry_name,
                ),
        }

    def eval(self,
             y_pred,
             y_true,
             metadata,
             *,
             viz_dir=None,
             viz_n_per_source=10):
        """The main evaluation metric, detection_acc_avg_dom, measures the simple average of the
        detection accuracies of each domain.

        Optional ``viz_dir`` / ``viz_n_per_source`` write qualitative overlays (purple = GT masks,
        orange = predicted masks above the eval score threshold).
        """

        results = {}
        results_str = ''
        for metric in self.metrics:
            result, result_str = self.standard_group_eval(
                self.metrics[metric], self._eval_grouper, y_pred, y_true,
                metadata)
            results[metric] = result
            results_str += result_str

        detection_accs = []
        for k, v in results["accuracy"].items():
            if k.startswith('detection_acc_source:'):
                d = k.split(':')[1]
                count = results["accuracy"][f'source:{d}']
                if count > 0:
                    detection_accs.append(v)
        detection_acc_avg_dom = np.array(detection_accs).mean()
        results['detection_acc_avg_dom'] = detection_acc_avg_dom
        results_str = f'Average detection_acc across source: {detection_acc_avg_dom:.3f}\n' + results_str

        # Format results with tables
        from milliontrees.common.utils import format_eval_results
        formatted_results = format_eval_results(results, self)
        results_str = formatted_results + '\n' + results_str

        if viz_dir is not None:
            paths = save_eval_visualizations(
                self,
                y_pred,
                y_true,
                metadata,
                viz_dir,
                n_per_source=viz_n_per_source,
                score_threshold=self.eval_score_threshold,
            )
            results["eval_visualization_paths"] = [str(p) for p in paths]

        return results, results_str

    def _get_mini_versions_dict(self):
        from milliontrees.common.release_sizes import subset_versions_dict
        return subset_versions_dict(self._versions_dict, "TreePolygons", "Mini")

    def _get_small_versions_dict(self):
        from milliontrees.common.release_sizes import subset_versions_dict
        return subset_versions_dict(self._versions_dict, "TreePolygons",
                                    "Small")

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (np.ndarray): Input features of the idx-th data point
        """
        # All images are in the images folder
        img_path = os.path.join(self._data_dir / 'images' /
                                self._input_array[idx])
        img = Image.open(img_path)
        img = np.asarray(img.convert('RGB'), dtype=np.float32)
        img /= 255.0

        return img

    def _transform_(self):
        transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
            ToTensorV2()
        ],
                              bbox_params=A.BboxParams(format='pascal_voc',
                                                       label_fields=['labels'],
                                                       clip=True))

        return transform

    def _train_transform_(self):
        """Train-time transform: simple augmentation on top of the resize.

        Overhead imagery is flip/rotation invariant, so horizontal/vertical flips and 90-degree
        rotations are label-preserving; Albumentations transforms the image, per-instance masks, and
        bboxes jointly so they stay aligned. A mild brightness/contrast jitter adds photometric
        variation. The default ``_transform_`` (resize only) stays the eval path so test-time inputs
        are deterministic.
        """
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
            ToTensorV2()
        ],
                              bbox_params=A.BboxParams(format='pascal_voc',
                                                       label_fields=['labels'],
                                                       clip=True))

        return transform

    @staticmethod
    def _collate_fn(batch):
        """Custom collate function to handle batching of metadata, inputs, and targets."""
        batch = list(zip(*batch))
        batch[0] = torch.stack(batch[0])
        batch[1] = torch.stack(batch[1])
        batch[2] = list(batch[2])

        return tuple(batch)
