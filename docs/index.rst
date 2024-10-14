The MillionTrees Benchmark for Airborne Tree Prediction
=======================================================

The MillionTrees seeks to collect a million tree locations to create a global benchmark for machine learning models for airborne tree prediction. Machine learning models need large amounts of data to generate realistic predictions. Existing benchmarks often have small amounts of data, often less than 10,000 trees, from single geographic locations and resolutions. The MillionTrees will cover a range of backgrounds, taxa, focal views and resolutions. To make this possible, we need your help!

.. figure:: public/open_drone_example.png
  :alt: Image Placeholder
  :width: 50%

Current Status
--------------

There are 3 datasets available for the MillionTrees benchmark:
 
* TreeBoxes: A dataset of 282,288 tree crowns from 9 sources.

* TreePolygons: A dataset of 362,751 tree crowns from 8 sources.

* TreePoints: A dataset of 191,614 tree stems from 2 sources.

Contact
-------

* Ben Weinstein, Research Scientist, Weecology Lab, University of Florida.
  ben.weinstein@weecology.org or make an issue on the `repo <https://github.com/weecology/MillionTrees>`_

Why do we need this benchmark?
------------------------------

Forests underlie many ecosystem services, agricultural systems and urban planning programs. Monitoring, mapping and measuring forests is an essential task globally. The growth in airborne acquisition tools has led to many attempts to capture tree information from airborne perspectives. At the core of many of these methods is the need to identify individual tree crowns to further process for species, trait, or health information. There have been dozens, if not hundreds, of articles predicting tree locations from airborne sensors. Here we focus on *RGB images* due to their low cost, uniform calibration across manufacturers and high resolution. The vast majority of research articles assess proposed algorithms at one or two locations, often in similar habitat types and with little understanding about generalization across environments, sensors, resolutions, focal views and taxa. To move beyond the duplication and static constriction of the field, we need to tackle a unified concept of tree detection that can be useful across a range of ecosystems, tree densities and taxonomy, as well as handle data from many input types, sensors and focal views. Models should be robust in urban settings to non-tree backgrounds and be quickly customized to new datasets.

.. automodule:: milliontrees
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   getting_started
   datasets
   dataset_structure
   leaderboard
   contributing
   developer
   source/modules.rst

