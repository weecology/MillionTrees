"""Module for retrieving MillionTrees dataset instances."""
from typing import Optional
import milliontrees


def get_dataset(dataset: str,
                version: Optional[str] = None,
                unlabeled: bool = False,
                **dataset_kwargs):
    """Brief description of the function.

    Args:
        dataset: Description of dataset.
        version: Description of version.
        unlabeled: Description of unlabeled.
    """
    if version is not None:
        version = str(version)
    if dataset not in milliontrees.supported_datasets:
        raise ValueError(
            f'Dataset {dataset} not recognized. Must be one of {milliontrees.supported_datasets}.'
        )
    if unlabeled and dataset not in milliontrees.unlabeled_datasets:
        raise ValueError(
            f'Unlabeled data not available for {dataset}. Must be one of {milliontrees.unlabeled_datasets}.'
        )
    dataset_classes = {
        'TreePoints': {
            'labeled':
                'milliontrees.datasets.TreePoints.TreePointsDataset',
            'unlabeled':
                'milliontrees.datasets.unlabeled.TreePointsUnlabeled.TreePoints_Unlabeled_Dataset'
        },
        'TreePolygons': {
            'labeled':
                'milliontrees.datasets.TreePolygons.TreePolygonsDataset',
            'unlabeled':
                'milliontrees.datasets.unlabeled.TreePolygonsUnlabeled.TreePolygons_Unlabeled_Dataset'
        },
        'TreeBoxes': {
            'labeled':
                'milliontrees.datasets.TreeBoxes.TreeBoxesDataset',
            'unlabeled':
                'milliontrees.datasets.unlabeled.TreeBoxesUnlabeled.TreeBoxes_Unlabeled_Dataset'
        }
    }
    if dataset in dataset_classes:
        module_path = dataset_classes[dataset][
            'unlabeled' if unlabeled else 'labeled']
        module_name, class_name = module_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        dataset_class = getattr(module, class_name)
        return dataset_class(version=version, **dataset_kwargs)
    raise ValueError(f'Dataset {dataset} is not supported.')
