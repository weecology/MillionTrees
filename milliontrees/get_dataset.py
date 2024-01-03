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

    elif dataset == 'iwildcam':
        if unlabeled:
            from milliontrees.datasets.unlabeled.iwildcam_unlabeled_dataset import IWildCamUnlabeledDataset
            return IWildCamUnlabeledDataset(version=version, **dataset_kwargs)
        else:
            from milliontrees.datasets.iwildcam_dataset import IWildCamDataset # type:ignore
            return IWildCamDataset(version=version, **dataset_kwargs)

