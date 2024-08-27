from datetime import datetime
from pathlib import Path
import os

from PIL import Image
import pandas as pd
import numpy as np
import torch

from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import Accuracy, Recall, F1


class TreePointsDataset(MillionTreesDataset):
    """The TreePoints dataset is a collection of tree annotations annotated as
    x,y locations.

    The dataset is comprised of many sources from across the world. There are 5 splits:
        - Random: 80% of the data randomly split into train and 20% in test
        - location: 80% of the locations randomly split into train and 20% in test
    Supported `split_scheme`:
        - 'Random'
        - 'location'
    Input (x):
        RGB images from camera traps
    Label (y):
        y is a n x 2-dimensional vector where each line represents a point coordinate (x, y)
    Metadata:
        Each image is annotated with the following metadata
            - location (int): location id
            - source (int): source id
            - resolution (int): resolution of image
            - focal view (int): focal view of image

    Website:
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009180
    Original publication:
        # Ventura et al. 2022
        @article{ventura2022individual,
        title={Individual tree detection in large-scale urban environments using high-resolution multispectral imagery},
        author={Ventura, Jonathan and Pawlak, Camille and Honsberger, Milo and Gonsalves, Cameron and Rice, Julian and Love, Natalie LR and Han, Skyler and Nguyen, Viet and Sugano, Keilana and Doremus, Jacqueline and others},
        journal={arXiv preprint arXiv:2208.10607},
        year={2022}
        }
        # TreeFormer
        #etc....


    License:
        This dataset is distributed under Creative Commons Attribution License
    """
    _dataset_name = 'TreePoints'
    _versions_dict = {
        '0.0': {
            'download_url':
                'https://www.dropbox.com/scl/fi/csqdtsps3thltrmbc2amx/TreePoints_v0.0.zip?rlkey=s8ycx5ssh14u2a5amiz0dx3ks&dl=0',
            'compressed_size':
                523312564
        }
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='official'):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
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

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df['split_id'].values

        # Filenames
        self._input_array = df['filename'].values

        # Point labels
        self._y_array = torch.tensor(df[["x", "y"]].values.astype(float))

        # Labels -> just 'Tree'
        self._n_classes = 1

        # Length of targets
        self._y_size = 2

        # Location/group info
        df["source_id"] = df.source.astype('category').cat.codes
        n_groups = max(df['source_id']) + 1
        self._n_groups = n_groups
        assert len(np.unique(df['source_id'])) == self._n_groups

        self._metadata_array = torch.tensor(
            np.stack([df['source_id'].values], axis=1))
        self._metadata_fields = ['source_id']

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(dataset=self,
                                                  groupby_fields=(['source_id'
                                                                  ]))

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """Computes all evaluation metrics.

        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
            })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n")

        return results, results_str

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        # All images are in the images folder
        img_path = os.path.join(self.data_dir / 'images' /
                                self._input_array[idx])
        img = Image.open(img_path)
        # Channels first input
        img = torch.tensor(np.array(img)).permute(2, 0, 1)

        return img
