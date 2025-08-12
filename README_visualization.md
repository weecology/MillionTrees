# MillionTrees Dataset Visualization Scripts

This repository contains scripts to visualize all MillionTrees datasets with annotation overlays and create organized splits for reviewers.

## Overview

The scripts loop through all images in all datasets (TreeBoxes, TreePoints, TreePolygons), use `plot_results` to overlay annotations on images, and save PNG files with filename format: `source_imagename.png`. The data is then split into two organizations:

1. **Miniature dataset**: 10 images from each source, split into 4 equal parts for reviewers
2. **Full dataset**: All data split into 4 equal parts for reviewers

## Scripts

### `visualize_all_datasets.py`
Full-featured script with detailed error handling, progress reporting, and comprehensive output organization.

### `visualize_datasets_concise.py` ⭐ 
**Recommended**: Concise version with minimal flow statements, optimized for performance and readability.

## Features

- **Comprehensive dataset coverage**: Processes all TreeBoxes, TreePoints, and TreePolygons datasets
- **Annotation overlay**: Uses DeepForest's `plot_results` to overlay annotations on images
- **Smart file naming**: Output files named as `{source}_{image_name}.png`
- **Reviewer organization**: Creates 4 equal splits for both mini and full datasets
- **Error handling**: Gracefully handles missing files and processing errors
- **Summary reporting**: Generates JSON summary with dataset statistics

## Usage

```bash
# Run the concise version (recommended)
python visualize_datasets_concise.py

# Or run the full-featured version
python visualize_all_datasets.py
```

## Output Structure

```
/tmp/milliontrees_viz/
├── mini_dataset/
│   ├── mini_reviewer_1/
│   │   ├── annotations.csv
│   │   └── visualizations/
│   │       └── *.png files
│   ├── mini_reviewer_2/
│   ├── mini_reviewer_3/
│   └── mini_reviewer_4/
├── full_dataset/
│   ├── full_reviewer_1/
│   ├── full_reviewer_2/
│   ├── full_reviewer_3/
│   └── full_reviewer_4/
└── summary.json
```

## Dataset Sources

The scripts process annotations from:

**TreeBoxes (12 sources)**:
- Ryoungseob 2023, Velasquez urban trees, Individual urban tree crown detection
- Radogoshi Sweden, WRI, Guangzhou 2022, NEON benchmark
- University of Florida, ReForestTree, Santos 2019, Zenodo, SelvaBox

**TreePoints (6 sources)**:
- TreeFormer, Ventura 2022, NEON points, Tonga
- BohlmanBCI, AutoArborist

**TreePolygons (20 sources)**:
- Jansen 2023, Troles Bamberg, Cloutier 2023, Firoze 2023
- Wagner Australia, Alejandro Chile, Urban London, Olive Trees Spain
- Araujo 2020, Justdiggit, BCI (2020 & 2022), Harz Mountains
- SPREAD, Kaggle Palm, Kattenborn, Quebec Lefebvre
- BohlmanBCI crowns, TreeCountSegHeight, Takeshige 2025

## Requirements

```python
pandas
numpy
opencv-python
matplotlib
deepforest
```

## Key Functions

- `load_all_data()`: Loads and combines all datasets
- `visualize_annotations()`: Creates annotation overlays using plot_results
- `create_splits()`: Splits data into reviewer assignments
- `main()`: Orchestrates the entire process

## Performance

The concise version is optimized for performance with:
- Minimal flow control statements
- Efficient pandas operations
- Streamlined error handling
- Reduced memory usage

## Notes

- Images must be accessible at the paths specified in the CSV files
- TreePolygons datasets require image dimensions for proper visualization
- All outputs are saved to `/tmp/milliontrees_viz/` by default
- Random shuffling ensures fair distribution across reviewer splits