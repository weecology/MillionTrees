import os
from glob import glob

import pandas as pd
import pytest
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

@pytest.fixture
def neon_token_txt(tmp_path):
    token_path = tmp_path / "neon_token.txt"
    token_path.write_text("FAKE_TOKEN")
    return str(token_path)


def test_parse_tile_names():
    import milliontrees.download_unsupervised as dl
    # HARV tile name
    e, n = dl.parse_tile_easting_northing("2018_HARV_5_733000_4698000_image.tif")
    assert isinstance(e, int) and isinstance(n, int)
    # BART tile name
    e, n = dl.parse_tile_easting_northing("2020_BART_4_322000_4882000_image.tif")
    assert isinstance(e, int) and isinstance(n, int)


def test_unsupervised_downloads(monkeypatch, dataset, unsupervised_annotations, neon_token_txt):
    # Arrange: point data_dir to the TreeBoxes dir created by dataset fixture
    data_dir = os.path.join(dataset, "TreeBoxes_v0.0")
    images_dir = os.path.join(data_dir, "images")
    download_dir = os.path.join(dataset, "neon_downloads")
    os.makedirs(download_dir, exist_ok=True)

    # Create dummy downloaded .tif files to simulate NEON download
    dummy_tiles = [
        "2018_HARV_5_733000_4698000_image.tif",
        "2019_HARV_6_733000_4698000_image.tif",
    ]
    for name in dummy_tiles:
        open(os.path.join(download_dir, name), "a").close()

    # Monkeypatch the downloader to skip network calls and only copy from download_dir
    import milliontrees.download_unsupervised as dl

    def fake_download_tile_rgb(site, easting, northing, year, savepath, token, data_product="DP3.30010.001"):
        # Already created dummy files above; no-op
        return

    monkeypatch.setattr(dl, "download_tile_rgb", fake_download_tile_rgb)

    # Act: simulate main's download loop using annotations
    ann = pd.read_parquet(unsupervised_annotations)
    to_download = ann[["siteID", "tile_name"]].drop_duplicates()
    for _, row in to_download.iterrows():
        site = row["siteID"]
        tile_name = str(row["tile_name"])
        e, n = dl.parse_tile_easting_northing(tile_name)
        year = int(tile_name.split("_")[0])
        dl.download_tile_rgb(site=site, easting=e, northing=n, year=year, savepath=download_dir, token="FAKE")

    # Run copy step and append annotations
    # Copy
    dl.copy_downloads_to_images(download_dir, images_dir)

    # Verify copies
    for name in dummy_tiles:
        assert os.path.exists(os.path.join(images_dir, name))

    # Append annotations to random.csv
    df_before = pd.read_csv(os.path.join(data_dir, "random.csv"))
    ann["filename"] = ann["filename"].apply(os.path.basename)
    ann["source"] = "Weinstein et al. 2018"
    ann["split"] = "train"
    keep_cols = [c for c in ["xmin", "ymin", "xmax", "ymax", "filename", "source", "split"] if c in ann.columns]
    updated = pd.concat([df_before, ann[keep_cols]], ignore_index=True)
    updated.to_csv(os.path.join(data_dir, "random.csv"), index=False)

    # Assert rows appended
    df_after = pd.read_csv(os.path.join(data_dir, "random.csv"))
    for name in dummy_tiles:
        assert os.path.basename(name) in set(df_after["filename"]) 

# Add a test and a pytest not run for downloading files -- actually perform real download
import pytest
import milliontrees.download_unsupervised as dl
import shutil

token_present = os.path.exists("neon_token.txt") or bool(os.environ.get("NEON_TOKEN"))
@pytest.mark.skipif(
    not token_present,
    reason="Provide NEON_TOKEN or neon_token.txt to enable real NEON download test."
)
def test_real_download(tmp_path):
    # Arrange
    real_download_dir = tmp_path / "real_downloads"
    real_download_dir.mkdir()
    real_images_dir = tmp_path / "real_images"
    real_images_dir.mkdir()
    real_data_dir = tmp_path / "real_data"
    real_data_dir.mkdir()

    # Use a real dataset and token for the test
    real_site = "HARV"
    real_tile_name = "2018_HARV_5_733000_4698000_image.tif"
    real_e, real_n = dl.parse_tile_easting_northing(real_tile_name)
    real_year = 2018

    # Read token from neon_token.txt
    with open("neon_token.txt", "r") as f:
        real_token = f.read().strip()

    # Act: perform the real download
    dl.download_tile_rgb(site=real_site, easting=real_e, northing=real_n, year=real_year, savepath=str(real_download_dir), token=real_token)

    # Run copy step
    dl.copy_downloads_to_images(real_download_dir, real_images_dir)

    # Verify the image was copied
    assert os.path.exists(os.path.join(real_images_dir, real_tile_name))

    # Clean up
    shutil.rmtree(real_download_dir)
    shutil.rmtree(real_images_dir)
    shutil.rmtree(real_data_dir)

def test_TreeBoxes_unsupervised_download(dataset, unsupervised_annotations):
    ds = TreeBoxesDataset(download=False, root_dir=dataset, version="0.0", unsupervised=True, unsupervised_args={'annotations_parquet': unsupervised_annotations,'token_path': 'neon_token.txt'})
    assert os.path.exists(ds._data_dir / 'unsupervised/unsupervised_annotations_tiled.parquet')
    assert len(ds) == 8