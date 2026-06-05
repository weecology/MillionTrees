from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch

from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.eval_visualization import save_eval_visualizations
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import (
    KeypointAccuracy,
    CountingError,
    MaskAwareKeypointPrecision,
    KeypointMergeCommissionMetric,
)
from milliontrees.common.utils import format_eval_results
from milliontrees.common.onboarding import print_dataset_summary

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import fnmatch


class TreePointsDataset(MillionTreesDataset):
    """The TreePoints dataset is a collection of tree annotations annotated as x,y locations.

    Dataset Splits:
        - random: For each source, a portion of images is in train and a portion in test.
        - crossgeometry: Boxes and Points are used to predict polygons.
        - zeroshot: Selected sources are entirely held out for testing.
    Input (x):
        RGB aerial images
    Label (y):
        y is an n x 2 matrix where each row represents a keypoint (x, y)
    Metadata:
        Each image is annotated with the following metadata
            - location (int): location id

    License:
        This dataset is distributed under Creative Commons Attribution License
    """

    # Ground sample distance (meters/pixel) for each source at native resolution.
    # Used to compute a physically-meaningful matching threshold at eval time.
    # Sources not listed here fall back to the dataset-level distance_threshold.
    SOURCE_GSD = {
        'Amirkolaee et al. 2023': 0.20,
        'Beery et al. 2022': 0.05,
        'Chen & Shang (2022)': 0.12,
        'Dubrovin et al. 2024': 0.07,
        'NEON MultiTemporal': 0.10,
        'NEON_points': 0.10,
        'Ventura et al. 2022': 0.60,
        'Young et al. 2025 unsupervised': 0.10,
    }

    _dataset_name = 'TreePoints'
    _versions_dict = {
        '0.0': {
            'download_url': '',
            'supervised_download_url': '',
            'compressed_size': 160938856
        },
        "0.13": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.13.zip",
            'supervised_download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_supervised_v0.13.zip",
            'compressed_size':
                164722449837
        },
        "0.15": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.15.zip",
            'supervised_download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_supervised_v0.15.zip",
            'compressed_size':
                164722449837
        },
        "0.16": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.16.zip",
            'supervised_download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_supervised_v0.16.zip",
            'compressed_size':
                164722449837
        }
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='random',
                 geometry_name='y',
                 remove_incomplete=False,
                 distance_threshold=0.02,
                 include_sources=None,
                 exclude_sources=None,
                 mini=False,
                 small=False,
                 image_size=448,
                 verbose=True,
                 include_unsupervised=False,
                 eval_score_threshold=0.4,
                 real_world_threshold_m=3.0):

        if mini and small:
            raise ValueError(
                'At most one of mini=True and small=True may be set.')

        self._version = version
        self._split_scheme = split_scheme
        self.geometry_name = geometry_name
        self.eval_score_threshold = eval_score_threshold
        self.real_world_threshold_m = real_world_threshold_m
        self.distance_threshold = distance_threshold
        self.mini = mini
        self.small = small
        self.image_size = image_size
        self.verbose = verbose
        self.include_unsupervised = include_unsupervised

        if self._split_scheme not in ['random', 'crossgeometry', 'zeroshot']:
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
                self._dataset_name = 'SmallTreePoints'
            else:
                self._dataset_name = 'TreePoints_supervised'

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Restore dataset name for proper operation after directory setup
        self._dataset_name = 'TreePoints'

        # Load splits
        self.df = pd.read_csv(self._data_dir / f"{self._split_scheme}.csv")

        # Cache available sources for convenience
        self.sources = self.df['source'].unique()
        available_source_count = len(self.sources)

        if remove_incomplete:
            self.df = self.df[self.df['complete'] == True]

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

        source_str = self.df['source'].astype(str).str.lower()

        if include_patterns is not None:
            patterns_lower = [p.lower() for p in include_patterns]
            mask_include = source_str.apply(
                lambda s: any(fnmatch.fnmatch(s, p) for p in patterns_lower))
            self.df = self.df[mask_include]

        patterns_exclude_lower = [p.lower() for p in exclude_patterns]
        if len(patterns_exclude_lower) > 0:
            mask_exclude = source_str.apply(lambda s: any(
                fnmatch.fnmatch(s, p) for p in patterns_exclude_lower))
            self.df = self.df[~mask_exclude]
        selected_source_count = self.df['source'].nunique()
        self.df = self.df.reset_index(drop=True)

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

        unique_files = self.df.drop_duplicates(
            subset=['filename'], inplace=False).reset_index(drop=True)
        unique_files['split_id'] = unique_files['split'].apply(
            lambda x: self._split_dict[x])
        self._split_array = unique_files['split_id'].values

        # Filenames
        self._input_array = unique_files.filename

        # Create lookup table for which index to select for each filename
        self._input_lookup = self.df.groupby('filename').apply(
            lambda x: x.index.values, include_groups=False).to_dict()

        # Point labels
        self._y_array = self.df[["x", "y"]].values.astype(int)

        # Labels -> just 'Tree'
        self._n_classes = 1

        # Length of targets
        self._y_size = 4

        # Class labels
        self.labels = np.zeros(self.df.shape[0])

        # Create dictionary for codes to names
        # Create source locations with a numeric ID
        self.df["source_id"] = self.df.source.astype('category').cat.codes

        # Create filename numeric ID
        self.df["filename_id"] = self.df.filename.astype('category').cat.codes
        self._source_id_to_code = self.df.set_index(
            'source_id')['source'].to_dict()
        self._filename_id_to_code = self.df.set_index(
            'filename_id')['filename'].to_dict()

        # Location/group info
        n_groups = max(self.df['source_id']) + 1
        self._n_groups = n_groups
        assert len(np.unique(self.df['source_id'])) == self._n_groups

        # Metadata is at the image level
        unique_sources = self.df[['filename_id', 'source_id']].drop_duplicates(
            subset="filename_id", inplace=False).reset_index(drop=True)
        self._metadata_array = torch.tensor(unique_sources.values.astype('int'))
        self._metadata_fields = ['filename_id', 'source_id']

        if 'complete' in self.df.columns:
            source_complete = self.df.groupby('source_id')['complete'].first()
            self._source_id_complete = {
                int(k): bool(v) for k, v in source_complete.items()
            }
        else:
            self._source_id_complete = {}

        self.metrics = {
            "KeypointAccuracy":
                KeypointAccuracy(
                    distance_threshold=distance_threshold,
                    image_size=self.image_size,
                    score_threshold=self.eval_score_threshold,
                ),
            "maskaware_precision":
                MaskAwareKeypointPrecision(
                    distance_threshold=distance_threshold,
                    image_size=self.image_size,
                    geometry_name=self.geometry_name,
                    score_threshold=self.eval_score_threshold,
                ),
            "counting_mae":
                CountingError(geometry_name=self.geometry_name,),
            "merge_commission":
                KeypointMergeCommissionMetric(
                    distance_threshold=distance_threshold,
                    image_size=self.image_size,
                    geometry_name=self.geometry_name,
                    score_threshold=self.eval_score_threshold,
                ),
        }

        # Per-source normalized distance thresholds derived from GSD and real_world_threshold_m.
        self._source_thresholds = self._compute_source_thresholds()

        self._collate = TreePointsDataset._collate_fn

        # eval grouper
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
                n_annotations=len(self.df),
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

        super().__init__(root_dir, download, self._split_scheme)

    def _get_mini_versions_dict(self):
        from milliontrees.common.release_sizes import subset_versions_dict
        return subset_versions_dict(self._versions_dict, "TreePoints", "Mini")

    def _get_small_versions_dict(self):
        from milliontrees.common.release_sizes import subset_versions_dict
        return subset_versions_dict(self._versions_dict, "TreePoints", "Small")

    def get_annotation_from_filename(self, filename):
        indices = self._input_lookup[filename]
        return self._y_array[indices]

    def _compute_source_thresholds(self):
        """Return a dict mapping source name → normalized distance threshold.

        For sources in SOURCE_GSD the threshold is derived from real_world_threshold_m
        and the native image resolution:  threshold = real_world_m / (gsd * orig_px_size).
        Sources without a known GSD fall back to self.distance_threshold.
        """
        thresholds = {}
        images_dir = self._data_dir / 'images'
        for source_name in self.df['source'].unique():
            gsd = self.SOURCE_GSD.get(source_name)
            if gsd is None:
                thresholds[source_name] = self.distance_threshold
                continue
            fname = self.df[self.df['source'] ==
                            source_name]['filename'].iloc[0]
            try:
                img = Image.open(images_dir / fname)
                orig_size = img.size[0]  # assume square crops
            except Exception:
                thresholds[source_name] = self.distance_threshold
                continue
            thresholds[source_name] = self.real_world_threshold_m / (gsd *
                                                                     orig_size)
        return thresholds

    def eval(self,
             y_pred,
             y_true,
             metadata,
             *,
             viz_dir=None,
             viz_n_per_source=4):
        """Evaluate predictions.

        KeypointAccuracy (recall) uses a per-source distance threshold derived from each source's
        GSD so that the matching radius is always ``real_world_threshold_m`` metres regardless of
        image resolution. All other metrics use the dataset-level ``distance_threshold``.

        Optional ``viz_dir`` / ``viz_n_per_source`` write qualitative overlays.
        """
        results = {}
        results_str = ''

        # --- KeypointAccuracy with per-source GSD-aware thresholds ---
        g = self._eval_grouper.metadata_to_group(metadata)
        kp_results = {}
        kp_accs = []
        kp_results_str = ''
        for source_id in range(self._n_groups):
            indices = (g == source_id).nonzero(as_tuple=True)[0].tolist()
            kp_results[f'count_source:{source_id}'] = len(indices)
            if not indices:
                continue
            source_name = self._source_id_to_code[source_id]
            threshold = self._source_thresholds.get(source_name,
                                                    self.distance_threshold)
            metric = KeypointAccuracy(
                distance_threshold=threshold,
                image_size=self.image_size,
                score_threshold=self.eval_score_threshold,
            )
            y_pred_src = [y_pred[i] for i in indices]
            y_true_src = [y_true[i] for i in indices]
            result = metric.compute(y_pred_src, y_true_src)
            acc = float(result[metric.agg_metric_field])
            kp_results[f'keypoint_acc_source:{source_id}'] = acc
            kp_accs.append(acc)
            kp_results_str += (
                f'  source:{source_id}  [{source_name}]'
                f'  [n = {len(indices):6d}]'
                f'  [threshold = {threshold:.4f} ({self.real_world_threshold_m:.1f} m)]:'
                f'\tkeypoint_acc = {acc:.3f}\n')

        if kp_accs:
            kp_results['worst_group_keypoint_acc'] = float(min(kp_accs))
            avg = float(np.mean(kp_accs))
            results['keypoint_acc_avg_dom'] = avg
            results_str += f'Average keypoint_acc across source: {avg:.3f}\n'
            results_str += f'Worst-group keypoint_acc: {min(kp_accs):.3f}\n'
            results_str += kp_results_str
        results['KeypointAccuracy'] = kp_results

        # --- All other metrics via standard_group_eval ---
        for metric_name in ('maskaware_precision', 'counting_mae',
                            'merge_commission'):
            result, result_str = self.standard_group_eval(
                self.metrics[metric_name], self._eval_grouper, y_pred, y_true,
                metadata)
            results[metric_name] = result
            results_str += result_str

        # Format results with tables
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
        img = np.array(img.convert('RGB')) / 255
        img = np.array(img, dtype=np.float32)

        return img

    @staticmethod
    def _collate_fn(batch):
        """Stack x (batch[1]) and metadata (batch[0]), but not y.

        originally, batch = (item1, item2, item3, item4) after zip, batch = [(item1[0], item2[0],
        ..), ..]
        """
        batch = list(zip(*batch))
        batch[1] = torch.stack(batch[1])
        batch[0] = torch.stack(batch[0])
        batch[2] = list(batch[2])

        return tuple(batch)

    def _transform_(self):
        self.transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
            ToTensorV2()
        ],
                                   keypoint_params=A.KeypointParams(
                                       format='xy',
                                       label_fields=['labels'],
                                       remove_invisible=False))

        return self.transform
