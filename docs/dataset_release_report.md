# MillionTrees Dataset Release Report
**Generated on:** 2026-03-16 13:35:04

## Dataset Summary (Supervised Sources Only)

| Dataset | Latest Version | Size (GB) | Images | Annotations | Sources | Download URL |
|---------|----------------|-----------|--------|-------------|---------|--------------|
| TreePolygons | 0.11 | 98.83 GB | 20429 | 791991 | 16 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreePoints | 0.11 | 148.47 GB | 27447 | 818619 | 8 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| TreeBoxes | 0.11 | 41.67 GB | 10612 | 713347 | 10 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees... |
| **Total** | - | **288.97 GB** | - | - | - | - |

## Unsupervised Sources (Train Split Only)

Sources with "unsupervised" or "weak supervised" in their name are excluded from test splits and counted separately here. Size is estimated as full zip minus supervised zip.

| Dataset | Size (GB) | Images | Annotations | Sources |
|---------|-----------|--------|-------------|---------|
| TreePoints | 148.42 GB | 1 | 138 | 1 |
| TreeBoxes | 41.52 GB | 10049 | 966952 | 2 |

## Detailed Dataset Information

### TreePolygons

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.10 GB |
| 0.11 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.11.zip | 98.83 GB |

**Latest Version Dataset Stats (random split, supervised sources only):**

- Images: 20429
- Annotations: 791991
- Sources: 16

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 19230 | 1225 |
| zeroshot | 17562 | 3926 |

### TreePoints

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.15 GB |
| 0.11 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePoints_v0.11.zip | 148.47 GB |

**Latest Version Dataset Stats (random split, supervised sources only):**

- Images: 27447
- Annotations: 818619
- Sources: 8

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 24674 | 2784 |
| zeroshot | 25392 | 2056 |

### TreeBoxes

| Version | Download URL | Size (GB) |
|---------|-------------|-----------|
| 0.0 | N/A | 0.10 GB |
| 0.11 | https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreeBoxes_v0.11.zip | 41.67 GB |

**Latest Version Dataset Stats (random split, supervised sources only):**

- Images: 10612
- Annotations: 713347
- Sources: 10

**Split counts (images):**

| Split | Train | Test |
|-------|-------|------|
| random | 18170 | 3225 |
| zeroshot | 17524 | 3137 |

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
