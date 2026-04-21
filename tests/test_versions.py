import copy
import os

import pytest
import urllib.request

from milliontrees.datasets import TreeBoxes, TreePoints, TreePolygons

DATASET_CLASSES = [
    TreePolygons.TreePolygonsDataset, TreeBoxes.TreeBoxesDataset,
    TreePoints.TreePointsDataset
]


def _mini_default_download_url(dataset_class):
    """Effective download_url after mini + default (supervised) URL swap in __init__."""
    obj = dataset_class.__new__(dataset_class)
    obj._versions_dict = copy.deepcopy(dataset_class._versions_dict)
    mini_versions = dataset_class._get_mini_versions_dict(obj)
    modified_versions = {}
    for v, info in mini_versions.items():
        modified_info = dict(info)
        if info.get('supervised_download_url') is not None:
            modified_info['download_url'] = info['supervised_download_url']
        modified_versions[v] = modified_info
    return modified_versions


@pytest.mark.parametrize("dataset_class,mini_basename", [
    (TreePolygons.TreePolygonsDataset, "MiniTreePolygons_v0.12.zip"),
    (TreeBoxes.TreeBoxesDataset, "MiniTreeBoxes_v0.12.zip"),
    (TreePoints.TreePointsDataset, "MiniTreePoints_v0.12.zip"),
])
def test_mini_supervised_default_uses_full_mini_zip(dataset_class,
                                                    mini_basename):
    """Regression: published mini archives are full splits only; supervised mini zips 404."""
    urls = _mini_default_download_url(dataset_class)
    assert urls["0.12"]["download_url"].endswith(mini_basename)
    assert "supervised" not in urls["0.12"]["download_url"]


@pytest.mark.parametrize("dataset_class", DATASET_CLASSES)
def test_dataset_url_exists(dataset_class, tmpdir):
    """Test that dataset URLs exist but don't actually download."""
    versions = dataset_class._versions_dict
    for version in versions:
        url = versions[version]['download_url']
        if url == "":
            continue
        try:
            with urllib.request.urlopen(url) as response:
                assert response.status == 200
        except urllib.error.URLError as e:
            pytest.fail(f"URL {url} is not accessible: {e}")
