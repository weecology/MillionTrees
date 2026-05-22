"""Published dataset release size limits (images per source)."""

MINI_IMAGES_PER_SOURCE = 3
SMALL_IMAGES_PER_SOURCE = 50


def subset_versions_dict(versions_dict, dataset_basename, subset_prefix):
    """Return a versions dict pointing at Mini/Small zip archives."""
    subset_versions = {}
    for version, info in versions_dict.items():
        subset_info = info.copy()
        if info['download_url']:
            subset_info['download_url'] = info['download_url'].replace(
                f"{dataset_basename}_v{version}.zip",
                f"{subset_prefix}{dataset_basename}_v{version}.zip",
            )
            subset_info['compressed_size'] = None
        if info.get('supervised_download_url'):
            subset_info['supervised_download_url'] = None
        subset_versions[version] = subset_info
    return subset_versions
