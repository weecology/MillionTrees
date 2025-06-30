# MillionTrees Dataset Release Report
**Generated on:** 2025-06-30 15:24:53

## Dataset Summary

| Dataset | Latest Version | Compressed Size | Download URL |
|---------|----------------|-----------------|--------------|
| TreePolygons | 0.2 | 70.24 GB | https://data.rc.ufl.edu/pub/ewhite/TreePolygons... |
| TreePoints | 0.2 | 1.36 GB | https://data.rc.ufl.edu/pub/ewhite/TreePoints_v... |
| TreeBoxes | 0.2 | 6.26 GB | https://data.rc.ufl.edu/pub/ewhite/TreeBoxes_v0... |
| **Total** | - | **77.86 GB** | - |

## Detailed Dataset Information

### TreePolygons

**Description:** The TreePolygons dataset is a collection of tree annotations annotated as multi-point polygon locations.

| Version | Download URL | Compressed Size | Size (MB) | Size (GB) |
|---------|-------------|-----------------|-----------|-----------|
| 0.0 | https://github.com/weecology/MillionTrees/releases/download/0.0.0-dev1/TreePolygons_v0.0.zip | 17,112,645 bytes | 16.32 | 0.02 |
| 0.1 | https://data.rc.ufl.edu/pub/ewhite/TreePolygons_v0.1.zip | 40,277,152 bytes | 38.41 | 0.04 |
| 0.2 | https://data.rc.ufl.edu/pub/ewhite/TreePolygons_v0.2.zip | 75,419,767,345 bytes | 71925.9 | 70.24 |

### TreePoints

**Description:** The TreePoints dataset is a collection of tree annotations annotated as x,y locations.

| Version | Download URL | Compressed Size | Size (MB) | Size (GB) |
|---------|-------------|-----------------|-----------|-----------|
| 0.0 | https://github.com/weecology/MillionTrees/releases/download/0.0.0-dev1/TreePoints_v0.0.zip | 523,312,564 bytes | 499.07 | 0.49 |
| 0.1 | https://data.rc.ufl.edu/pub/ewhite/TreePoints_v0.1.zip | 170,340 bytes | 0.16 | 0.0 |
| 0.2 | https://data.rc.ufl.edu/pub/ewhite/TreePoints_v0.2.zip | 1,459,676,926 bytes | 1392.06 | 1.36 |

### TreeBoxes

**Description:** A dataset of tree annotations with bounding box coordinates from multiple global sources.

| Version | Download URL | Compressed Size | Size (MB) | Size (GB) |
|---------|-------------|-----------------|-----------|-----------|
| 0.0 | https://github.com/weecology/MillionTrees/releases/download/0.0.0-dev1/TreeBoxes_v0.0.zip | 5,940,337 bytes | 5.67 | 0.01 |
| 0.1 | https://data.rc.ufl.edu/pub/ewhite/TreeBoxes_v0.1.zip | 3,476,300 bytes | 3.32 | 0.0 |
| 0.2 | https://data.rc.ufl.edu/pub/ewhite/TreeBoxes_v0.2.zip | 6,717,977,561 bytes | 6406.76 | 6.26 |

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
