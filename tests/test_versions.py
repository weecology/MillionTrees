import copy
import os

import pytest
import urllib.request

from milliontrees.datasets import TreeBoxes, TreePoints, TreePolygons

DATASET_CLASSES = [
    TreePolygons.TreePolygonsDataset, TreeBoxes.TreeBoxesDataset,
    TreePoints.TreePointsDataset
]


def _subset_default_download_url(dataset_class, subset_prefix):
    """Effective download_url after subset + default (supervised) URL swap in __init__."""
    obj = dataset_class.__new__(dataset_class)
    obj._versions_dict = copy.deepcopy(dataset_class._versions_dict)
    if subset_prefix == "Mini":
        subset_versions = dataset_class._get_mini_versions_dict(obj)
    else:
        subset_versions = dataset_class._get_small_versions_dict(obj)
    modified_versions = {}
    for v, info in subset_versions.items():
        modified_info = dict(info)
        if info.get('supervised_download_url') is not None:
            modified_info['download_url'] = info['supervised_download_url']
        modified_versions[v] = modified_info
    return modified_versions


@pytest.mark.parametrize("dataset_class,mini_basename", [
    (TreePolygons.TreePolygonsDataset, "MiniTreePolygons_v0.23.zip"),
    (TreeBoxes.TreeBoxesDataset, "MiniTreeBoxes_v0.23.zip"),
    (TreePoints.TreePointsDataset, "MiniTreePoints_v0.23.zip"),
])
def test_mini_supervised_default_uses_full_mini_zip(dataset_class,
                                                    mini_basename):
    """Regression: published mini archives are full splits only; supervised mini zips 404."""
    urls = _subset_default_download_url(dataset_class, "Mini")
    assert urls["0.23"]["download_url"].endswith(mini_basename)
    assert "supervised" not in urls["0.23"]["download_url"]


@pytest.mark.parametrize("dataset_class,small_basename", [
    (TreePolygons.TreePolygonsDataset, "SmallTreePolygons_v0.23.zip"),
    (TreeBoxes.TreeBoxesDataset, "SmallTreeBoxes_v0.23.zip"),
    (TreePoints.TreePointsDataset, "SmallTreePoints_v0.23.zip"),
])
def test_small_default_uses_small_zip(dataset_class, small_basename):
    urls = _subset_default_download_url(dataset_class, "Small")
    assert urls["0.23"]["download_url"].endswith(small_basename)
    assert "supervised" not in urls["0.23"]["download_url"]


def test_mini_and_small_mutually_exclusive():
    with pytest.raises(ValueError, match="mini=True and small=True"):
        TreePoints.TreePointsDataset(mini=True, small=True)


@pytest.mark.parametrize("dataset_class", DATASET_CLASSES)
def test_dataset_url_exists(dataset_class, tmpdir):
    """Every registered version must still be downloadable, in both variants.

    The host keeps only the most recent releases, so this fails when a version stays in
    ``_versions_dict`` after its archives are deleted -- the state that made every
    ``download=True`` call for v0.18-v0.21 return a 404. The supervised URL is checked
    too because it, not ``download_url``, is what the default (``include_unsupervised=False``)
    constructor actually fetches.
    """
    versions = dataset_class._versions_dict
    for version, info in versions.items():
        for key in ("download_url", "supervised_download_url"):
            url = info.get(key) or ""
            if url == "":
                continue
            request = urllib.request.Request(url, method="HEAD")
            try:
                with urllib.request.urlopen(request) as response:
                    assert response.status == 200
            except urllib.error.URLError as e:
                pytest.fail(f"{dataset_class.__name__} v{version} {key} "
                            f"{url} is not accessible: {e}")
