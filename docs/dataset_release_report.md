# MillionTrees Dataset Release Report
**Generated on:** 2025-11-26 16:37:23

## Dataset Summary

| Dataset | Latest Version | Size (GB) | Images | Annotations | Sources | Download URL |
|---------|----------------|-----------|--------|-------------|---------|--------------|
| TreePolygons | 0.8 | 100.64 GB | 20207 | 825203 | 13 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreePoints | 0.8 | 153.46 GB | 33468 | 1240309 | 6 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreeBoxes | 0.8 | 43.21 GB | 18887 | 1507949 | 10 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| **Total** | - | **297.30 GB** | - | - | - | - |

## Detailed Dataset Information

### TreePolygons

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.8 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.8.zip | 100.64 GB |

**Latest Version Dataset Stats (random split):**

- Images: 20207
- Annotations: 825203
- Sources: 13

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 20120 | 87 |
| zeroshot | 17754 | 2453 |

### TreePoints

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.8 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.8.zip | 153.46 GB |

**Latest Version Dataset Stats (random split):**

- Images: 33468
- Annotations: 1240309
- Sources: 6

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 33424 | 44 |
| zeroshot | 32857 | 611 |

### TreeBoxes

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.8 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreeBoxes_v0.8.zip | 43.21 GB |

**Latest Version Dataset Stats (random split):**

- Images: 18887
- Annotations: 1507949
- Sources: 10

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 18816 | 71 |
| zeroshot | 16915 | 2990 |

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
