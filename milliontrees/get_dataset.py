from typing import Optional

import milliontrees

def get_dataset(dataset: str, version: Optional[str] = None, unlabeled: bool = False, **dataset_kwargs):
    """
    Returns the appropriate milliontrees dataset class.
    Input:
        dataset (str): Name of the dataset
        version (Union[str, None]): Dataset version number, e.g., '1.0'.
                                    Defaults to the latest version.
        unlabeled (bool): If true, use the unlabeled version of the dataset.
        dataset_kwargs: Other keyword arguments to pass to the dataset constructors.
    Output:
        The specified milliontreesDataset class.
    """
    if version is not None:
        version = str(version)

    if dataset not in milliontrees.supported_datasets:
        raise ValueError(f'The dataset {dataset} is not recognized. Must be one of {milliontrees.supported_datasets}.')

    if unlabeled and dataset not in milliontrees.unlabeled_datasets:
        raise ValueError(f'Unlabeled data is not available for {dataset}. Must be one of {milliontrees.unlabeled_datasets}.')

    elif dataset == 'TreePoints':
        if unlabeled:
            from milliontrees.datasets.unlabeled.TreePointsUnlabeled import TreePoints_Unlabeled_Dataset
            return TreePoints_Unlabeled_Dataset(version=version, **dataset_kwargs)
        else:
            from milliontrees.datasets.TreePoints import TreePointsDataset # type:ignore
            return TreePointsDataset(version=version, **dataset_kwargs)

    elif dataset == 'TreePolygons':
        if unlabeled:
            from milliontrees.datasets.unlabeled.TreePolygonsUnlabeled import TreePolygons_Unlabeled_Dataset
            return TreePolygons_Unlabeled_Dataset(version=version, **dataset_kwargs)
        else:
            from milliontrees.datasets.TreePolygons import TreePolygonsDataset # type:ignore
            return TreePolygonsDataset(version=version, **dataset_kwargs)
    elif dataset == 'TreeBoxes':
        if unlabeled:
            from milliontrees.datasets.unlabeled.TreeBoxesUnlabeled import TreeBoxes_Unlabeled_Dataset
            return TreeBoxes_Unlabeled_Dataset(version=version, **dataset_kwargs)
        else:
            from milliontrees.datasets.TreeBoxes import TreeBoxesDataset # type:ignore
            return TreeBoxesDataset(version=version, **dataset_kwargs)
