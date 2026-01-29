# MillionTrees Dataset Release Report
**Generated on:** 2026-01-20 13:14:19

## Dataset Summary

| Dataset | Latest Version | Size (GB) | Images | Annotations | Sources | Download URL |
|---------|----------------|-----------|--------|-------------|---------|--------------|
| TreePolygons | 0.10 | 98.80 GB | 19948 | 805818 | 15 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreePoints | 0.10 | 153.41 GB | 33369 | 1219904 | 7 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreeBoxes | 0.10 | 39.97 GB | 19894 | 1615798 | 10 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| **Total** | - | **292.18 GB** | - | - | - | - |

## Detailed Dataset Information

### TreePolygons

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.10 GB |
| 0.10 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.10.zip | 98.80 GB |

**Latest Version Dataset Stats (random split):**

- Images: 19948
- Annotations: 805818
- Sources: 15

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 18801 | 1172 |
| zeroshot | 17232 | 3775 |

### TreePoints

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.15 GB |
| 0.10 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.10.zip | 153.41 GB |

**Latest Version Dataset Stats (random split):**

- Images: 33369
- Annotations: 1219904
- Sources: 7

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 30139 | 3238 |
| zeroshot | 31639 | 1730 |

### TreeBoxes

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.10 GB |
| 0.10 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreeBoxes_v0.10.zip | 39.97 GB |

**Latest Version Dataset Stats (random split):**

- Images: 19894
- Annotations: 1615798
- Sources: 10

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 17484 | 3137 |
| zeroshot | 16904 | 2990 |

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
