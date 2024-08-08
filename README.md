[![Github Actions](https://github.com/weecology/MillionTrees/actions/workflows/python-package.yml/badge.svg)](https://github.com/weecology/MillionTrees/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/milliontrees/badge/?version=latest)](https://milliontrees.readthedocs.io/en/latest/?badge=latest)
[![Version](https://img.shields.io/pypi/v/MillionTrees.svg)](https://pypi.python.org/pypi/MillionTrees)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/MillionTrees)](https://pypi.python.org/pypi/MillionTrees)


# Overview

The MillionTrees benchmark is designed to provie *open*, *reproducible* and *rigorous* evaluation of tree detection algorithms. The dataset currently holds X images, X annotations and X train/test evaluation splits. This repo is the python package for rapid data sharing and evaluation.

## Why MillionTrees?

There has been a tremendous number of tree crown detection benchmarks, but a lack of progress towards a single algorithm that can be used globally across aquisition sensors, forest type and annotation geometry. Our view is that the hundreds of tree detection algorithms for RGB data published in the last 10 years are all data starved. There are many good models, but they can only be so useful with the small datasets any research team can collect. The result is years of effort in model development, but ultimately a lacking solution for a large audience. The MillionTrees dataset seeks to collect a million annotations across point, polygon and box geometries at a global scale. For information on the MillionTrees Benchmark see: https://milliontrees.idtrees.org/

## Installation

```
pip install MillionTrees
```

### Dev Requirements

To build from the GitHub source and install the required dependencies, follow these instructions:

1. Clone the GitHub repository:
    ```
    git clone https://github.com/username/repo.git
    ```

2. Change to the repository directory:
    ```
    cd repo
    ```

3. Install the required dependencies using pip:
    ```
    pip install -r requirements.txt
    ```

4. (Optional) Build and install the package:
    ```
    python setup.py install
    ```

Once the installation is complete, you can use the MillionTrees package in your Python projects.

# Datasets

The MillionTrees package has ingested many contributed datasets and formatted them in a consistant fashion.

| Geometry | Training Images | Training Annotations | Eval Images | Eval Annotations |
|----------|-----------------|----------------------|-------------|------------------|
| Point    |                 |                      |             |                  |
| Polygon  |                 |                      |             |                  |
| Box      |                 |                      |             |                  |

## Underlying data contributions

Many datasets have been cleaned or altered to fit the desired data format. Here is an incomplete list of the current contributions.

# Using the MillionTrees package

## Downloading and training on the MillionTrees datasets

## Algorithms

## Evaluation

### Reproducibility

# Citing MillionTrees


## Acknowledgements
The design of the MillionTrees benchmark was inspired by the [WILDS benchmark](https://github.com/p-lambda/wilds), and we are grateful to their work, as well as Sara Beery for suggesting the use of this template.
