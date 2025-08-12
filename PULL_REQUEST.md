# Add Dataset Visualization Scripts for Comprehensive Review

## Overview

This PR introduces comprehensive visualization scripts that loop through all images in all MillionTrees datasets, overlay annotations using `plot_results`, and organize the data for systematic review.

## Changes

### New Files
- `visualize_all_datasets.py` - Full-featured visualization script
- `visualize_datasets_concise.py` - Optimized concise version with minimal flow statements
- `README_visualization.md` - Comprehensive documentation

### Features Implemented

1. **Complete Dataset Coverage**
   - Processes all TreeBoxes (12 sources), TreePoints (6 sources), and TreePolygons (20 sources)
   - Handles different annotation types (boxes, points, polygons) appropriately

2. **Annotation Overlay Visualization**
   - Uses DeepForest's `plot_results` function to overlay annotations on images
   - Saves PNG files with format: `{source}_{image_name}.png`
   - Handles image dimensions correctly for polygon datasets

3. **Reviewer Organization**
   - **Miniature dataset**: 10 best-annotated images from each source → split into 4 equal parts
   - **Full dataset**: All data → split into 4 equal parts
   - Each reviewer gets balanced representation across sources

4. **Code Quality**
   - Concise implementation with minimal flow statements (as requested)
   - Efficient pandas operations and list comprehensions
   - Graceful error handling for missing files
   - Comprehensive documentation

## Usage

```bash
# Recommended: Run concise version
python visualize_datasets_concise.py

# Alternative: Full-featured version
python visualize_all_datasets.py
```

## Output Structure

```
/tmp/milliontrees_viz/
├── mini_dataset/
│   └── mini_reviewer_{1-4}/
│       ├── annotations.csv
│       └── visualizations/*.png
├── full_dataset/
│   └── full_reviewer_{1-4}/
│       ├── annotations.csv  
│       └── visualizations/*.png
└── summary.json
```

## Key Benefits

- **Systematic Review**: Organized splits enable efficient parallel review by multiple reviewers
- **Visual Verification**: PNG overlays allow quick visual validation of annotation quality
- **Comprehensive Coverage**: Processes all 38 dataset sources across 3 geometry types
- **Balanced Distribution**: Each reviewer gets representative samples from all sources
- **Automated Process**: Single command generates complete review-ready organization

## Performance Optimizations

The concise version uses:
- List comprehensions instead of loops where possible
- Pandas vectorized operations
- Minimal conditional statements
- Efficient memory usage patterns

## Testing

- [x] Validates against existing dataset paths from `count_annotations.py`
- [x] Handles missing files gracefully
- [x] Generates proper PNG files with annotation overlays
- [x] Creates balanced reviewer splits
- [x] Produces comprehensive summary statistics

## Related Work

This builds on existing visualization patterns from:
- `package_datasets.py` (mini dataset creation)
- `AutoArborist.py` (sample plotting)
- Various dataset processing scripts (annotation overlay patterns)

## Future Enhancements

- Support for custom output directories
- Configurable number of reviewer splits
- Integration with MillionTrees dataset loaders
- Batch processing optimizations for very large datasets