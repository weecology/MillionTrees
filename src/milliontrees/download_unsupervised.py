from typing import Optional, Tuple

from .download_neon_unsupervised import run as run_neon
from .download_ofo_unsupervised import run as run_ofo


def run(
    data_dir: str,
    annotations_parquet: Optional[str] = None,
    max_tiles_per_site: Optional[int] = None,
    patch_size: int = 400,
    allow_empty: bool = False,
    num_workers: int = 4,
    token_path: str = 'neon_token.txt',
    data_product: str = 'DP3.30010.001',
    download_dir: str = 'neon_downloads',
    provider: str = 'neon',
    ofo_root: Optional[str] = None,
    mission_ids: Optional[list] = None,
    output_parquet_name: str = 'TreePoints_OFO_unsupervised.parquet',
    split: str = 'train',
    photogrammetry_glob: str = 'processed_*',
):
    """Unified entrypoint for unsupervised data generation.

    provider='neon' uses annotations_parquet of boxes and NEON tile downloads.
    provider='ofo' uses local OFO missions tree tops to tile orthomosaics into point parquet.
    """
    if provider.lower() == 'ofo':
        if not ofo_root:
            raise ValueError("ofo_root must be provided when provider='ofo'")
        return run_ofo(
            data_dir=data_dir,
            ofo_root=ofo_root,
            patch_size=patch_size,
            allow_empty=allow_empty,
            mission_ids=mission_ids,
            photogrammetry_glob=photogrammetry_glob,
            output_parquet_name=output_parquet_name,
            split=split,
        )

    # default NEON path for backward compatibility
    if annotations_parquet is None:
        raise ValueError(
            "annotations_parquet is required for provider='neon' (NEON unsupervised boxes)"
        )
    return run_neon(
        data_dir=data_dir,
        annotations_parquet=annotations_parquet,
        max_tiles_per_site=max_tiles_per_site,
        patch_size=patch_size,
        allow_empty=allow_empty,
        num_workers=num_workers,
        token_path=token_path,
        data_product=data_product,
        download_dir=download_dir,
    )

