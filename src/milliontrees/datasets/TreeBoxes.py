from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch
import albumentations as A
import torchvision.transforms as T

from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import DetectionAccuracy
from PIL import Image
from albumentations.pytorch import ToTensorV2


class TreeBoxesDataset(MillionTreesDataset):
    """A dataset of tree annotations with bounding box coordinates from multiple global sources.

    The dataset contains aerial imagery of trees with their corresponding bounding box annotations.
    Each tree is annotated with a 4-point bounding box (x_min, y_min, x_max, y_max).

    Dataset Splits:
        - Official: For each source, 80% of the data is used for training and 20% for testing.
        - crossgeometry: Boxes and Points are used to predict polygons.
        - zeroshot: Selected sources are entirely held out for testing.

    Data Format:
        Input (x): RGB aerial imagery
        Labels (y): Nx4 array of bounding box coordinates
        Metadata: Location identifiers for each image

    References:
        Website: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009180

        Citation:
            @article{Weinstein2020,
            title={A benchmark dataset for canopy crown detection and delineation in co-registered airborne RGB, LiDAR and hyperspectral imagery from the National Ecological Observation Network.},
            author={Weinstein BG, Graves SJ, Marconi S, Singh A, Zare A, Stewart D, et al.},
            journal={PLoS Comput Biol},
            year={2021},
            doi={10.1371/journal.pcbi.1009180}
            }

    License: Creative Commons Attribution License
    """
    _dataset_name = 'TreeBoxes'
    _versions_dict = {
        '0.0': {
            'download_url':
                'https://github.com/weecology/MillionTrees/releases/download/0.0.0-dev1/TreeBoxes_v0.0.zip',
            'compressed_size':
                5940337
        },
        "0.1": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/TreeBoxes_v0.1.zip",
            'compressed_size':
                3476300
        },
        "0.2": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/TreeBoxes_v0.2.zip",
            'compressed_size':
                6717977561
        }
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='official',
                 geometry_name='y',
                 eval_score_threshold=0.1,
                 image_size=448):

        self._version = version
        self._split_scheme = split_scheme
        self.geometry_name = geometry_name
        self.eval_score_threshold = eval_score_threshold
        self.image_size = image_size

        if self._split_scheme not in ['official', 'zeroshot', 'crossgeometry']:
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
        }
        self._split_names = {
            'train': 'Train',
            'val': 'Validation (OOD/Trans)',
            'test': 'Test (OOD/Trans)',
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
        self._y_array = df[["xmin", "ymin", "xmax",
                            "ymax"]].values.astype("float32")

        # Labels -> just 'Tree'
        self._n_classes = 1

        # Length of targets
        self._y_size = 4

        # Class labels
        self.labels = torch.zeros(df.shape[0])

        # Create source locations with a numeric ID
        df["source_id"] = df.source.astype('category').cat.codes

        # Create filename numeric ID
        df["filename_id"] = df.filename.astype('category').cat.codes

        # Create dictionary for codes to names
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

        self._collate = TreeBoxesDataset._collate_fn

        self.metrics = {
            "accuracy":
                DetectionAccuracy(geometry_name=self.geometry_name,
                                  score_threshold=self.eval_score_threshold,
                                  metric="accuracy"),
            "recall":
                DetectionAccuracy(geometry_name=self.geometry_name,
                                  score_threshold=self.eval_score_threshold,
                                  metric="recall"),
        }

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(dataset=self,
                                                  groupby_fields=(['source_id'
                                                                  ]))

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata):
        """Performs evaluation on the given predictions.

        The main evaluation metric, detection_acc_avg_dom, measures the simple average of the
        detection accuracies of each domain.
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

        return results, results_str

    def get_input(self, idx):
        """Retrieves the input features (image) for a given data point.

        Args:
            idx (int): Index of a data point

        Returns:
            np.ndarray: Input features of the idx-th data point (image) as a normalized numpy array.
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
        """Collates a batch by stacking `x` (features) and `metadata`, but not `y` (targets).

        The batch is initially a tuple of individual data points: (item1, item2, item3, ...).
        After zipping, it transforms into a list of tuples:
        [(item1[0], item2[0], ...), (item1[1], item2[1], ...), ...].

        Args:
            batch (list): A batch of data points, where each data point is a tuple (metadata, x, y).

        Returns:
            tuple: A tuple containing:
                - Stacked `x` (features).
                - Stacked `metadata`.
        """
        batch = list(zip(*batch))
        batch[1] = torch.stack(batch[1])
        batch[0] = torch.stack(batch[0])
        batch[2] = list(batch[2])

        return tuple(batch)

    def _transform_(self):
        transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
            ToTensorV2()
        ],
                              bbox_params=A.BboxParams(format='pascal_voc',
                                                       label_fields=['labels'],
                                                       clip=True))

        return transform
