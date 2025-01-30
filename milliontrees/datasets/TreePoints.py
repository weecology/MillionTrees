from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch

from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import KeypointAccuracy
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TreePointsDataset(MillionTreesDataset):
    """The TreePoints dataset is a collection of tree annotations annotated as
    x,y locations.

    The dataset is comprised of many sources from across the world. There are 2 splits:
        - Official: 80% of the data randomly split into train and 20% in test
        - Random: 80% of the locations randomly split into train and 20% in test
    Supported `split_scheme`:
        - 'Official'
        - 'Random'
    Input (x):
        RGB images from camera traps
    Label (y):
        y is a n x 4-dimensional vector where each line represents a box coordinate (x_min, y_min, x_max, y_max)
    Metadata:
        Each image is annotated with the following metadata
            - location (int): location id

    License:
        This dataset is distributed under Creative Commons Attribution License
    """
    _dataset_name = 'TreePoints'
    _versions_dict = {
        '0.0': {
            'download_url':
                'https://github.com/weecology/MillionTrees/releases/latest/download/TreePoints_v0.0.zip',
            'compressed_size':
                523312564
        },
        "0.1": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/TreePoints_v0.1.zip",
            'compressed_size':
                170340
        }
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='official',
                 geometry_name='y',
                 distance_threshold=0.1):

        self._version = version
        self._split_scheme = split_scheme
        self.geometry_name = geometry_name
        self.distance_threshold = distance_threshold

        if self._split_scheme not in ['official', 'random']:
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load splits
        df = pd.read_csv(self._data_dir / '{}.csv'.format(split_scheme))

        # Splits
        self._split_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
            'id_val': 3,
            'id_test': 4
        }
        self._split_names = {
            'train': 'Train',
            'val': 'Validation (OOD/Trans)',
            'test': 'Test (OOD/Trans)',
            'id_val': 'Validation (ID/Cis)',
            'id_test': 'Test (ID/Cis)'
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
            lambda x: x.index.values).to_dict()

        # Point labels
        self._y_array = df[["x", "y"]].values.astype(int)

        # Labels -> just 'Tree'
        self._n_classes = 1

        # Length of targets
        self._y_size = 4

        # Class labels
        self.labels = np.zeros(df.shape[0])

        # Create dictionary for codes to names
        # Create source locations with a numeric ID
        df["source_id"] = df.source.astype('category').cat.codes

        # Create filename numeric ID
        df["filename_id"] = df.filename.astype('category').cat.codes
        self._source_id_to_code = df.set_index('source_id')['source'].to_dict()
        self._filename_id_to_code = df.set_index(
            'filename_id')['filename'].to_dict()

        # Location/group info
        n_groups = max(df['source_id']) + 1
        self._n_groups = n_groups
        assert len(np.unique(df['source_id'])) == self._n_groups

        # Metadata is at the image level
        unique_sources = df[['filename_id', 'source_id']].drop_duplicates(
            subset="filename_id", inplace=False).reset_index(drop=True)
        self._metadata_array = torch.tensor(unique_sources.values.astype('int'))
        self._metadata_fields = ['filename_id', 'source_id']

        self._metric = KeypointAccuracy(distance_threshold=distance_threshold)
        self._collate = TreePointsDataset._collate_fn

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(dataset=self,
                                                  groupby_fields=(['source_id'
                                                                  ]))

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata):
        """The main evaluation metric, detection_acc_avg_dom, measures the
        simple average of the detection accuracies of each domain."""
        results, results_str = self.standard_group_eval(self._metric,
                                                        self._eval_grouper,
                                                        y_pred, y_true,
                                                        metadata)

        detection_accs = []
        for k, v in results.items():
            if k.startswith('detection_acc_source:'):
                d = k.split(':')[1]
                count = results[f'source:{d}']
                if count > 0:
                    detection_accs.append(v)
        detection_acc_avg_dom = np.array(detection_accs).mean()
        results['detection_acc_avg_dom'] = detection_acc_avg_dom
        results_str = f'Average detection_acc across source: {detection_acc_avg_dom:.3f}\n' + results_str

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

        originally, batch = (item1, item2, item3, item4) after zip,
        batch = [(item1[0], item2[0], ..), ..]
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
