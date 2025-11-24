from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch

from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import KeypointAccuracy, CountingError
from milliontrees.common.utils import format_eval_results

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import fnmatch


class TreePointsDataset(MillionTreesDataset):
    """The TreePoints dataset is a collection of tree annotations annotated as x,y locations.

    Dataset Splits:
        - random: For each source, 80% of the data is used for training and 20% for testing.
        - crossgeometry: Boxes and Points are used to predict polygons.
        - zeroshot: Selected sources are entirely held out for testing.
    Input (x):
        RGB images from camera traps
    Label (y):
        y is an n x 2 matrix where each row represents a keypoint (x, y)
    Metadata:
        Each image is annotated with the following metadata
            - location (int): location id

    License:
        This dataset is distributed under Creative Commons Attribution License
    """
    _dataset_name = 'TreePoints'
    _versions_dict = {
        '0.0': {
            'download_url': '',
            'compressed_size': 105525592
        },
        "0.8": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.8.zip",
            'compressed_size':
                160910816
        }
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='random',
                 geometry_name='y',
                 remove_incomplete=False,
                 distance_threshold=0.1,
                 include_sources=None,
                 exclude_sources=None,
                 mini=False):

        self._version = version
        self._split_scheme = split_scheme
        self.geometry_name = geometry_name
        self.distance_threshold = distance_threshold
        self.mini = mini

        if self._split_scheme not in ['random', 'crossgeometry', 'zeroshot']:
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')

        # Modify download URLs for mini datasets
        if mini:
            self._versions_dict = self._get_mini_versions_dict()

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load splits
        self.df = pd.read_csv(self._data_dir / '{}.csv'.format(split_scheme))

        # Cache available sources for convenience
        self.sources = self.df['source'].unique()

        if remove_incomplete:
            self.df = self.df[self.df['complete'] == True]

        # Filter by include/exclude source names with wildcard support
        # Default: exclude sources containing 'unsupervised'
        include_patterns = None
        if include_sources is not None and include_sources != []:
            include_patterns = include_sources if isinstance(
                include_sources, (list, tuple)) else [include_sources]
        exclude_patterns = exclude_sources
        if exclude_patterns is None:
            exclude_patterns = ['*unsupervised*']
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

        # Splits
        self._split_dict = {
            'train': 0,
            'test': 1,
        }
        self._split_names = {
            'train': 'Train',
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
            lambda x: x.index.values).to_dict()

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

        self.metrics = {
            "KeypointAccuracy":
                KeypointAccuracy(distance_threshold=distance_threshold),
            "CountingAccuracy":
                CountingError()
        }

        self._collate = TreePointsDataset._collate_fn

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(dataset=self,
                                                  groupby_fields=(['source_id'
                                                                  ]))

        super().__init__(root_dir, download, split_scheme)

    def _get_mini_versions_dict(self):
        """Generate mini versions dict with modified URLs for smaller datasets."""
        mini_versions = {}
        for version, info in self._versions_dict.items():
            mini_info = info.copy()
            if info['download_url']:
                # Replace dataset name with Mini version in URL
                original_filename = f"TreePoints_v{version}.zip"
                mini_filename = f"MiniTreePoints_v{version}.zip"
                mini_info['download_url'] = info['download_url'].replace(
                    original_filename, mini_filename)
            mini_versions[version] = mini_info
        return mini_versions

    def get_annotation_from_filename(self, filename):
        indices = self._input_lookup[filename]
        return self._y_array[indices]

    def eval(self, y_pred, y_true, metadata):
        """The main evaluation metric, detection_acc_avg_dom, measures the simple average of the
        detection accuracies of each domain."""

        results = {}
        results_str = ''
        for metric in self.metrics:
            result, result_str = self.standard_group_eval(
                self.metrics[metric], self._eval_grouper, y_pred, y_true,
                metadata)
            results[metric] = result
            results_str += result_str

        # Compute average keypoint accuracy across sources (domains)
        kp_accs = []
        kp_results = results.get("KeypointAccuracy", {})
        for k, v in kp_results.items():
            if k.startswith('keypoint_acc_source:'):
                d = k.split(':')[1]
                count = kp_results.get(f'count_source:{d}', 0)
                if count > 0:
                    kp_accs.append(v)
        if len(kp_accs) > 0:
            keypoint_acc_avg_dom = np.array(kp_accs).mean()
            results['keypoint_acc_avg_dom'] = keypoint_acc_avg_dom
            results_str = f'Average keypoint_acc across source: {keypoint_acc_avg_dom:.3f}\n' + results_str

        # Format results with tables
        formatted_results = format_eval_results(results, self)
        results_str = formatted_results + '\n' + results_str

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
        self.transform = A.Compose(
            [A.Resize(height=448, width=448, p=1.0),
             ToTensorV2()],
            keypoint_params=A.KeypointParams(format='xy',
                                             label_fields=['labels'],
                                             remove_invisible=False))

        return self.transform
