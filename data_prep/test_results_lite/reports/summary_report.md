# MillionTrees Image Sources Test Report (Lite Version)

Generated: 2025-06-19 21:29:16

*Note: This report was generated without DeepForest tree detection. For full functionality including tree detection, install DeepForest and use test_image_sources.py*

## Executive Summary

- **Total data sources tested**: 2
- **Functional image servers**: 1/2
- **Successful image downloads**: 1/2

## Server Status

| Source | Server Status | Image Download | Ground Truth Trees |
|--------|---------------|----------------|--------------------|
| calgary | functional | ✓ | 50 |
| charlottesville | error | ✗ | 50 |

## Detailed Results

### calgary

**Server Status**: functional
**Response Time**: 0.22s
**Response Size**: 778 bytes

### charlottesville

**Server Status**: error

**Recommendations**:
- Image server appears to be down or misconfigured

## Next Steps

1. Install DeepForest for tree detection functionality: `pip install deepforest`
2. Run the full test script: `python test_image_sources.py`
3. Review server configurations in `image_server_config.json`
4. Add missing tree location data for cities without ground truth
