"""Module for retrieving MillionTrees dataset instances."""
from typing import Optional
import milliontrees


def get_dataset(dataset: str, version: Optional[str] = None, **dataset_kwargs):
    """Brief description of the function.

    Args:
        dataset: Description of dataset.
        version: Description of version.
    """
    if version is not None:
        version = str(version)
    if dataset not in milliontrees.supported_datasets:
        raise ValueError(
            f'Dataset {dataset} not recognized. Must be one of {milliontrees.supported_datasets}.'
        )
    dataset_classes = {
        'TreePoints':
            'milliontrees.datasets.TreePoints.TreePointsDataset',
        'TreePolygons':
            'milliontrees.datasets.TreePolygons.TreePolygonsDataset',
        'TreeBoxes':
            'milliontrees.datasets.TreeBoxes.TreeBoxesDataset',
    }
    if dataset in dataset_classes:
        module_path = dataset_classes[dataset]
        module_name, class_name = module_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        dataset_class = getattr(module, class_name)
        return dataset_class(version=version, **dataset_kwargs)
    raise ValueError(f'Dataset {dataset} is not supported.')
