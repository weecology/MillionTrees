[![Github Actions](https://github.com/weecology/MillionTrees/actions/workflows/python-package.yml/badge.svg)](https://github.com/weecology/MillionTrees/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/milliontrees/badge/?version=latest)](https://milliontrees.readthedocs.io/en/latest/?badge=latest)
[![Version](https://img.shields.io/pypi/v/MillionTrees.svg)](https://pypi.python.org/pypi/MillionTrees)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/MillionTrees)](https://pypi.python.org/pypi/MillionTrees)

# Overview

The MillionTrees benchmark is designed to provide *open*, *reproducible*, and *rigorous* evaluation of tree detection algorithms. This repo is the Python package for rapid data sharing and evaluation.

# Current status

We have released a beta version of *public* data, these are datasets that have previously been published and have a DOI. We will followup this release, likely with a 1.0 tag, of the previously unpublished parts of the dataset along with a scientific manuscript.

📊 **Current Dataset Status**: See our comprehensive [Dataset Release Report](https://milliontrees.idtrees.org/en/latest/dataset_release_report.html) for up-to-date information on dataset versions, sizes, and download links.


# Dataloaders

There are three data loaders based on annotation geometry. TreeBoxes are bounding boxes for individual tree detection. TreePoints are centroids for tree counting and detection, and TreePolygons are for finer crown segmentation. 

## Why MillionTrees?

There has been a tremendous number of tree crown detection benchmarks, but a lack of progress towards a single algorithm that can be used globally across aquisition sensors, forest type and annotation geometry. Our view is that the hundreds of tree detection algorithms for RGB data published in the last 10 years are all data starved. There are many good models, but they can only be so useful with the small datasets any research team can collect. The result is years of effort in model development, but ultimately a lacking solution for a large audience. The MillionTrees dataset seeks to collect a million annotations across point, polygon and box geometries at a global scale.

The MillionTrees dataset represents where we are as a community. Many datasets are incompletely annotated, and there is varying degrees of annotation accuracy. This is by design, we aim to reflect the real, not idealized, status of tree detection algorithms and applications. By incluing these data that are normally excluded from benchmarks we can both dramatically increase the diversity of tree presentations and backgrounds, as well as engage the community in solving common computer vision challenges for applied machine learning.

## Installation

```
pip install milliontrees
```

Hugging Face dataset loading and sharing functionality is included in the main package.

### Dev Requirements

To build from the GitHub source and install the required dependencies, follow these instructions:

1. Clone the GitHub repository:
    ```
    git clone https://github.com/weecology/MillionTrees.git
    ```

2. Change to the repository directory:
    ```
    cd MillionTrees
    ```

3. (Recommended) Create and activate a virtual environment, then install dev extras:
    ```
    python -m venv .venv && source .venv/bin/activate
    pip install -e .[dev,docs]
    ```

4. (Optional) Build distributions:
    ```
    python -m build
    ```

Once the installation is complete, you can use the MillionTrees package in your Python projects.

# Datasets

Datasets are documented on ReadTheDocs with sample images overlayed with annotations.
https://milliontrees.idtrees.org/en/latest/datasets.html

# Leaderboard

See the latest results on the leaderboard.
https://milliontrees.idtrees.org/en/latest/leaderboard.html

# Citing MillionTrees

## Acknowledgements
The design of the MillionTrees benchmark was inspired by the [WILDS benchmark](https://github.com/p-lambda/wilds), and we are grateful to their work, as well as Sara Beery for suggesting the use of this template.
