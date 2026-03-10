import os, sys
import argparse
import milliontrees


def main():
    """Downloads the latest versions of all specified datasets, if they do not already exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        required=True,
        help=
        'The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=None,
        help=
        f'Specify a space-separated list of dataset names to download. If left unspecified, the script will download all of the benchmark datasets. Available choices are {milliontrees.supported_datasets}.'
    )
    config = parser.parse_args()

    if config.datasets is None:
        config.datasets = milliontrees.benchmark_datasets

    for dataset in config.datasets:
        if dataset not in milliontrees.supported_datasets:
            raise ValueError(
                f'{dataset} not recognized; must be one of {milliontrees.supported_datasets}.'
            )

    print(f'Downloading the following datasets: {config.datasets}')
    for dataset in config.datasets:
        print(f'=== {dataset} ===')
        milliontrees.get_dataset(dataset=dataset,
                                 root_dir=config.root_dir,
                                 download=True)


if __name__ == '__main__':
    main()
