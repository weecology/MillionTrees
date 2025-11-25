# MillionTrees Dataset Release Report
**Generated on:** 2025-11-25 14:48:44

## Dataset Summary

| Dataset | Latest Version | Compressed Size | Download URL |
|---------|----------------|-----------------|--------------|
| TreePolygons | 0.8 | 100.64 MB | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreePoints | 0.8 | 153.46 MB | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreeBoxes | 0.8 | 43.21 MB | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| **Total** | - | **297.30 MB** | - |

## Detailed Dataset Information

### TreePolygons

**Description:** The TreePolygons dataset is a collection of tree annotations annotated as multi-point polygon locations.

| Version | Download URL | Compressed Size | Size (MB) | Size (GB) |
|---------|-------------|-----------------|-----------|-----------|
| 0.0 | N/A | 105,525,592 bytes | 100.64 | 0.1 |
| 0.8 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.8.zip | 105,525,592 bytes | 100.64 | 0.1 |

### TreePoints

**Description:** The TreePoints dataset is a collection of tree annotations annotated as x,y locations.

| Version | Download URL | Compressed Size | Size (MB) | Size (GB) |
|---------|-------------|-----------------|-----------|-----------|
| 0.0 | N/A | 105,525,592 bytes | 100.64 | 0.1 |
| 0.8 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.8.zip | 160,910,816 bytes | 153.46 | 0.15 |

### TreeBoxes

**Description:** A dataset of tree annotations with bounding box coordinates from multiple global sources.

| Version | Download URL | Compressed Size | Size (MB) | Size (GB) |
|---------|-------------|-----------------|-----------|-----------|
| 0.0 | N/A | 105,525,592 bytes | 100.64 | 0.1 |
| 0.8 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreeBoxes_v0.8.zip | 45,304,108 bytes | 43.21 | 0.04 |

## Release Test Information

The release tests validate that:
- Datasets can be downloaded successfully
- Dataset structure and format are correct
- Image and annotation data have expected shapes and types
- Data loaders work properly for training and evaluation

Tests are located in: `tests/test_release.py`

To run the tests manually:
```bash
python -m pytest tests/test_release.py -v
```
