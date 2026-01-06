# MillionTrees Dataset Release Report
**Generated on:** 2026-01-06 13:56:54

## Dataset Summary

| Dataset | Latest Version | Size (GB) | Images | Annotations | Sources | Download URL |
|---------|----------------|-----------|--------|-------------|---------|--------------|
| TreePolygons | 0.9 | 98.81 GB | 21007 | 844872 | 15 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreePoints | 0.9 | 153.48 GB | 33543 | 1256100 | 8 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreeBoxes | 0.9 | 39.98 GB | 19894 | 1615798 | 11 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| **Total** | - | **292.27 GB** | - | - | - | - |

## Detailed Dataset Information

### TreePolygons

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.10 GB |
| 0.9 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.9.zip | 98.81 GB |

**Latest Version Dataset Stats (random split):**

- Images: 21007
- Annotations: 844872
- Sources: 15

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 18801 | 2232 |
| zeroshot | 17232 | 3775 |

### TreePoints

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.15 GB |
| 0.9 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.9.zip | 153.48 GB |

**Latest Version Dataset Stats (random split):**

- Images: 33543
- Annotations: 1256100
- Sources: 8

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 30297 | 3254 |
| zeroshot | 31828 | 1715 |

### TreeBoxes

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.10 GB |
| 0.9 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreeBoxes_v0.9.zip | 39.98 GB |

**Latest Version Dataset Stats (random split):**

- Images: 19894
- Annotations: 1615798
- Sources: 11

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 17469 | 3119 |
| zeroshot | 17426 | 2468 |

## Test Results Summary

✅ **All tests passed successfully**

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
