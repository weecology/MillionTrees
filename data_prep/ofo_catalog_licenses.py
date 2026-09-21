"""Licenses attached to the Open Forest Observatory catalog.

OFO is the one place in MillionTrees where the image and the annotation come from different
upstream datasets with different owners:

* the image tile is cut from an OFO drone mission orthomosaic -- every mission in the catalog
  is CC-BY 4.0;
* the points are a contributor's ground-reference stem map, and those plots span CC-BY 4.0,
  CC-BY-SA 4.0, CC-BY-NC-SA 4.0, CC0 1.0 and U.S. Forest Service public domain.

Using a packaged row means using both, so its license is the most restrictive of the two.
This module holds that combination table and the catalog reads behind it, shared by
``process_ofo_field.py`` (which stamps the license onto each annotation as it tiles) and
``ofo_license_overrides.py`` (which derives the filename-keyed fallback rules).

The catalog is Zenodo record 22731179, "Open Forest Observatory Data Catalog".
"""

import os
import urllib.request

import geopandas as gpd

ZENODO_RECORD = "https://zenodo.org/records/22731179"
ZENODO_FILE = "https://zenodo.org/api/records/22731179/files/{name}/content"

PLOTS_GPKG = "ofo_ground-reference_plots.gpkg"
MISSIONS_GPKG = "ofo_drone-missions_metadata.gpkg"

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "ofo_catalog")

#: Most restrictive first. A tile covering several plots takes the earliest license here.
RESTRICTIVENESS = [
    "CC BY NC SA 4.0",
    "CC BY SA 4.0",
    "CC BY 4.0",
    "Public domain",
    "CC0 1.0",
]

#: Effective license of a packaged row: the plot license combined with the CC-BY 4.0 imagery
#: every OFO mission carries. The CC0 and public-domain plots still land on CC-BY-4.0 --
#: that attribution obligation comes from the imagery, not from the stem map.
COMBINED_WITH_IMAGERY = {
    "CC BY NC SA 4.0": "CC-BY-NC-SA-4.0",
    "CC BY SA 4.0": "CC-BY-SA-4.0",
    "CC BY 4.0": "CC-BY-4.0",
    "Public domain": "CC-BY-4.0",
    "CC0 1.0": "CC-BY-4.0",
}


def download_catalog(cache_dir=DEFAULT_CACHE_DIR):
    """Fetch the OFO catalog geopackages from Zenodo, caching them in ``cache_dir``."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for name in (PLOTS_GPKG, MISSIONS_GPKG):
        dest = os.path.join(cache_dir, name)
        if not os.path.exists(dest):
            print(f"Downloading {name} from {ZENODO_RECORD}")
            urllib.request.urlretrieve(ZENODO_FILE.format(name=name), dest)
        paths[name] = dest
    return paths


def plot_licenses(plots_gpkg):
    """``{plot_id: license_short}`` from the OFO ground-reference plot catalog."""
    plots = gpd.read_file(plots_gpkg)
    plots["plot_id"] = plots["plot_id"].astype(str).str.zfill(4)
    unknown = sorted(set(plots["license_short"]) - set(RESTRICTIVENESS))
    if unknown:
        raise ValueError(
            f"OFO plot catalog carries license(s) {unknown} that this module does not know "
            "how to rank. Add them to RESTRICTIVENESS and COMBINED_WITH_IMAGERY."
        )
    return dict(zip(plots["plot_id"], plots["license_short"]))


def check_mission_imagery_license(missions_gpkg):
    """Fail loudly if the drone imagery is no longer uniformly CC-BY 4.0.

    The whole combination table assumes it is; a mission published under other terms would
    change the effective license of every tile cut from it.
    """
    missions = gpd.read_file(missions_gpkg)
    found = {
        str(value).replace("CC-BY", "CC BY")
        for value in missions["license"].unique()
    }
    if found != {"CC BY 4.0"}:
        raise ValueError(
            f"OFO drone missions are no longer uniformly CC-BY 4.0 (found {sorted(found)}). "
            "COMBINED_WITH_IMAGERY needs to be recomputed per mission.")


def load_combined_licenses(cache_dir=DEFAULT_CACHE_DIR):
    """``{plot_id: license id}`` already combined with the CC-BY 4.0 imagery."""
    catalog = download_catalog(cache_dir)
    check_mission_imagery_license(catalog[MISSIONS_GPKG])
    return {
        plot_id: COMBINED_WITH_IMAGERY[license_short] for plot_id, license_short
        in plot_licenses(catalog[PLOTS_GPKG]).items()
    }
