# MillionTrees

The MillionTreesBenchmark seeks to collect a million tree locations to create a global benchmark for machine learning models for airborne tree data.

# How to contribute

* RGB airborne data

* Tree locations in stem, box, and polygon datasets.

# Why do we need this benchmark?

Forests underlie many ecosystem services, agricultural systems and urban planning programs. Monitoring, mapping and measuring forests is an essential task globally.
The growth in acquisition tools has led to many attempts to capture tree information from airborne perspectives. At the core of many of these methods is the need to identify individual tree crowns to further process for species, trait, or health information. The quality of the tree segmentations substantially impacts downstream analysis.
There have been dozens, if not hundreds, of articles predicting tree locations from airborne sensors. Here we focus on RGB images due to their low cost, uniform calibration across manufacturers and high resolution. There have been multiple recent reviews on RGB tree detection. The vast majority of articles assess proposed algorithms at one or two locations, often in similar habitat types and with little understanding about generalization across environments, sensors, resolutions, focal views and taxa. To move beyond the duplication and static constriction of the field, we need to tackle a unified concept of tree detection.
To develop a fully global sense of tree detection we need detectors that can be useful across a range of ecosystems, tree densities and taxonomy
Handle data from many input types, sensors and focal views. Robust in urban settings to non-tree backgrounds. Can be quickly customized to new datasets
To achieve these goals, we need data that covers these conditions.

There has been a myopic view of the task that has been overly constrained by off-the-shelf architectures, rather than the essential nature of the task. Tree localization, counting, and crown detection are all interrelated tasks using geometric representations of trees. We should not design benchmarks around current model architectures, we should put the problem first and build architectures that meet that need.

# How to contribute

* The best way to contribute is to make data available on Zenodo, and then make an issue in this repo documenting the location of the data.

* We are sensitive to the contributions and efforts of the hundreds of field researchers that make data collection possible. Authorship will be extended to any team with unpublished data. 

* The spatial location of the points will be destroyed, such that the point locations will only be relative to the image crop. This will prevent any user from being able to use the data for analysis outside of the benchmark. All species, DBH and other metadata will be removed.


# What does a successful dataset look like?

## Point

## Polygon

## Boxes


The working document is [here](https://docs.google.com/document/d/1K6G1tcdTuAv3FgGiDWq5QhO-kSoBrxzTiic5jH1CZF4/edit?usp=sharing)

# Who has been contacted?
