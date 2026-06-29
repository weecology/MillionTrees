"""Two-row figure: zoomed LiDAR subset and full ortho with crown annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely import contains_xy
from shapely.geometry import box, Polygon

DEFAULT_BASE = Path(
    "/Users/benweinstein/Dropbox/Weecology/MillionTrees/TLS/ortho_3919")
LAZ_NAME = "2024-10-15_ecosense_forest_mamba2.laz"
ORTHO_NAME = "ortho_3919.tif"
CROWN_NAME = "crown_polygons_mamba2.gpkg"
OUTPUT_NAME = "ecosense_tls_ortho_crown_panel.png"

UNASSIGNED = -1
CHUNK_SIZE = 5_000_000
MAX_ZOOM_POINTS = 120_000
CROWN_BUFFER_M = 0.4
OUTSIDE_GREY = np.array([200, 200, 200], dtype=np.uint8)
OUTSIDE_BLEND = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--search-size-m", type=float, default=55.0,
                        help="Area used to pick the focal instance subset")
    parser.add_argument("--n-instances", type=int, default=7,
                        help="Number of nearby tree instances in the zoom row")
    parser.add_argument("--zoom-padding-m", type=float, default=2.5,
                        help="Padding around selected crowns for zoom bounds")
    parser.add_argument("--view-elev", type=float, default=18.0)
    parser.add_argument("--view-azim", type=float, default=-62.0)
    parser.add_argument("--z-exaggeration", type=float, default=2.5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def pick_search_bounds(crowns: gpd.GeoDataFrame,
                       search_size_m: float) -> tuple[float, float, float, float]:
    b = crowns.total_bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half = search_size_m / 2
    return cx - half, cy - half, cx + half, cy + half


def full_plot_bounds(crowns: gpd.GeoDataFrame,
                     padding_m: float = 1.0) -> tuple[float, float, float, float]:
    b = crowns.total_bounds
    return b[0] - padding_m, b[1] - padding_m, b[2] + padding_m, b[3] + padding_m


def select_instances(
    crowns: gpd.GeoDataFrame,
    search_bounds: tuple[float, float, float, float],
    n_instances: int,
) -> list[int]:
    x0, y0, x1, y1 = search_bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    patch = gpd.GeoDataFrame(geometry=[box(x0, y0, x1, y1)], crs=crowns.crs)
    candidates = gpd.clip(crowns[crowns["V1"] >= 0], patch).copy()
    centroids = candidates.geometry.centroid
    candidates["dist"] = np.hypot(centroids.x - cx, centroids.y - cy)
    selected = candidates.nsmallest(n_instances, "dist")
    return sorted(int(v) for v in selected["V1"])


def bounds_for_instances(
    crowns: gpd.GeoDataFrame,
    instance_ids: list[int],
    padding_m: float,
) -> tuple[float, float, float, float]:
    subset = crowns[crowns["V1"].isin(instance_ids)]
    b = subset.total_bounds
    return b[0] - padding_m, b[1] - padding_m, b[2] + padding_m, b[3] + padding_m


def read_lidar_patch(
    laz_path: Path,
    bounds: tuple[float, float, float, float],
    instance_ids: list[int],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bounds
    id_set = set(instance_ids)
    xs, ys, zs, inst = [], [], [], []
    with laspy.open(laz_path) as las:
        for chunk in las.chunk_iterator(CHUNK_SIZE):
            mask = ((chunk.x >= x0) & (chunk.x <= x1) & (chunk.y >= y0)
                    & (chunk.y <= y1))
            if not mask.any():
                continue
            chunk_inst = np.asarray(chunk["PredInstance_FM"][mask], dtype=np.int32)
            keep = np.isin(chunk_inst, list(id_set))
            if not keep.any():
                continue
            xs.append(np.asarray(chunk.x[mask][keep], dtype=np.float64))
            ys.append(np.asarray(chunk.y[mask][keep], dtype=np.float64))
            zs.append(np.asarray(chunk.z[mask][keep], dtype=np.float64))
            inst.append(chunk_inst[keep])

    if not xs:
        raise RuntimeError(
            f"No LiDAR points found for instances {instance_ids} in {bounds}")

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)
    inst = np.concatenate(inst)
    return stratified_subsample(xs, ys, zs, inst, max_points)


def stratified_subsample(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    inst: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(xs) <= max_points:
        return xs, ys, zs, inst

    rng = np.random.default_rng(0)
    ids = np.unique(inst)
    per_id = max(max_points // len(ids), 1)
    keep_idx = []
    for inst_id in ids:
        idx = np.flatnonzero(inst == inst_id)
        n = min(len(idx), per_id)
        keep_idx.append(rng.choice(idx, n, replace=False))
    keep_idx = np.concatenate(keep_idx)
    if len(keep_idx) > max_points:
        keep_idx = rng.choice(keep_idx, max_points, replace=False)
    return xs[keep_idx], ys[keep_idx], zs[keep_idx], inst[keep_idx]


def clip_points_to_crowns(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    inst: np.ndarray,
    crowns: gpd.GeoDataFrame,
    buffer_m: float = CROWN_BUFFER_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.zeros(len(xs), dtype=bool)
    for _, row in crowns.iterrows():
        inst_id = int(row["V1"])
        geom = row.geometry.buffer(buffer_m)
        mask = (inst == inst_id) & contains_xy(geom, xs, ys)
        keep |= mask
    return xs[keep], ys[keep], zs[keep], inst[keep]


def read_ortho_patch(
    ortho_path: Path,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, rasterio.Affine]:
    x0, y0, x1, y1 = bounds
    with rasterio.open(ortho_path) as src:
        window = from_bounds(x0, y0, x1, y1, transform=src.transform)
        rgb = src.read(indexes=(1, 2, 3), window=window, boundless=True, fill_value=0)
        transform = src.window_transform(window)
        rgb = np.transpose(rgb, (1, 2, 0))
        return np.clip(rgb, 0, 255).astype(np.uint8), transform


def grey_outside_crowns(
    ortho_rgb: np.ndarray,
    transform: rasterio.Affine,
    crowns: gpd.GeoDataFrame,
) -> np.ndarray:
    shapes = [geom for geom in crowns.geometry if not geom.is_empty]
    if not shapes:
        return ortho_rgb
    inside = ~geometry_mask(shapes,
                            out_shape=ortho_rgb.shape[:2],
                            transform=transform,
                            invert=False)
    out = ortho_rgb.copy()
    outside = ~inside
    out[outside] = (
        (1 - OUTSIDE_BLEND) * out[outside] + OUTSIDE_BLEND * OUTSIDE_GREY).astype(
            np.uint8)
    return out


def instance_colors(
    instance_ids: list[int],
) -> dict[int, tuple[float, float, float, float]]:
    rng = np.random.default_rng(42)
    colors = rng.uniform(0.15, 0.95, size=(len(instance_ids), 3))
    return {inst_id: (*color, 1.0) for inst_id, color in zip(instance_ids, colors)}


def format_axes_2d(ax, xlabel: str = "Easting (m)", ylabel: str = "Northing (m)") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)


def format_axes_3d(ax) -> None:
    ax.set_xlabel("Easting (m)", labelpad=6, fontsize=9)
    ax.set_ylabel("Northing (m)", labelpad=6, fontsize=9)
    ax.set_zlabel("Elevation (m)", labelpad=6, fontsize=9)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="z", labelleft=False)


def color_array(inst: np.ndarray,
                color_map: dict[int, tuple[float, ...]]) -> np.ndarray:
    rgba = np.empty((len(inst), 4), dtype=np.float32)
    for inst_id, color in color_map.items():
        rgba[inst == inst_id] = color
    return rgba


def plot_plan_lidar(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    inst: np.ndarray,
    color_map: dict[int, tuple[float, ...]],
    bounds: tuple[float, float, float, float],
    *,
    point_size: float,
    alpha: float,
    title: str,
) -> None:
    x0, y0, x1, y1 = bounds
    rgba = color_array(inst, color_map)
    ax.scatter(xs,
               ys,
               s=point_size,
               c=rgba,
               alpha=alpha,
               linewidths=0,
               rasterized=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    format_axes_2d(ax)


def plot_oblique_lidar(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    inst: np.ndarray,
    color_map: dict[int, tuple[float, ...]],
    bounds: tuple[float, float, float, float],
    view_elev: float,
    view_azim: float,
    z_exaggeration: float,
    *,
    point_size: float,
    alpha: float,
    title: str,
) -> None:
    x0, y0, x1, y1 = bounds
    z0 = float(zs.min())
    z_plot = z0 + (zs - z0) * z_exaggeration
    rgba = color_array(inst, color_map)
    rgba[:, 3] = alpha
    ax.scatter(xs, ys, z_plot, c=rgba, s=point_size, depthshade=True, linewidths=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z0 + (zs.max() - z0) * z_exaggeration)
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    format_axes_3d(ax)


def iter_polygons(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        polys = []
        for part in geom.geoms:
            polys.extend(iter_polygons(part))
        return polys
    return []


def add_polygon_outlines(
    ax,
    gdf: gpd.GeoDataFrame,
    color_map: dict[int, tuple[float, ...]],
    *,
    linewidth: float = 2.0,
    swap_xy: bool = False,
    highlight_ids: list[int] | None = None,
) -> None:
    for _, row in gdf.iterrows():
        inst_id = int(row["V1"])
        color = color_map.get(inst_id, (0.55, 0.55, 0.55, 0.5))
        lw = (linewidth if highlight_ids is None or inst_id in highlight_ids else
              max(linewidth * 0.6, 0.8))
        for poly in iter_polygons(row.geometry):
            x, y = poly.exterior.xy
            if swap_xy:
                x, y = y, x
            ax.add_patch(
                MplPolygon(np.column_stack([x, y]),
                           closed=True,
                           fill=False,
                           edgecolor=color,
                           linewidth=lw))


def plot_crown_panel(
    ax,
    ortho_rgb: np.ndarray,
    bounds: tuple[float, float, float, float],
    crowns: gpd.GeoDataFrame,
    color_map: dict[int, tuple[float, ...]],
    focal_ids: list[int],
    *,
    title: str,
    grey_outside: bool,
    transform: rasterio.Affine | None = None,
) -> None:
    x0, y0, x1, y1 = bounds
    extent = (x0, x1, y0, y1)
    display = ortho_rgb
    if grey_outside and transform is not None:
        display = grey_outside_crowns(ortho_rgb, transform, crowns)
    ax.imshow(display, extent=extent, origin="upper")
    add_polygon_outlines(ax, crowns, color_map, highlight_ids=focal_ids)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    format_axes_2d(ax)


def plot_full_ortho_rotated(
    ax,
    ortho_rgb: np.ndarray,
    bounds: tuple[float, float, float, float],
    crowns: gpd.GeoDataFrame,
    color_map: dict[int, tuple[float, ...]],
) -> None:
    """Show full RGB ortho rotated 90° CCW so the tall plot reads horizontally."""
    x0, y0, x1, y1 = bounds
    rgb = np.rot90(ortho_rgb, k=1)
    ax.imshow(rgb, extent=(y0, y1, x0, x1), origin="lower", aspect="equal")
    add_polygon_outlines(ax,
                         crowns,
                         color_map,
                         linewidth=1.0,
                         swap_xy=True)
    ax.set_xlim(y0, y1)
    ax.set_ylim(x0, x1)
    ax.set_title("Full plot · ortho with crown polygons", fontsize=11)
    format_axes_2d(ax, xlabel="Northing (m)", ylabel="Easting (m)")


def main() -> None:
    args = parse_args()
    base = args.base_dir
    laz_path = base / LAZ_NAME
    ortho_path = base / ORTHO_NAME
    crown_path = base / CROWN_NAME
    output_path = args.output or (base / OUTPUT_NAME)

    crowns = gpd.read_file(crown_path)
    search_bounds = pick_search_bounds(crowns, args.search_size_m)
    full_bounds = full_plot_bounds(crowns)
    focal_ids = select_instances(crowns, search_bounds, args.n_instances)
    zoom_bounds = bounds_for_instances(crowns, focal_ids, args.zoom_padding_m)
    focal_crowns = crowns[crowns["V1"].isin(focal_ids)]

    xs, ys, zs, inst = read_lidar_patch(laz_path, zoom_bounds, focal_ids,
                                        MAX_ZOOM_POINTS)
    xs, ys, zs, inst = clip_points_to_crowns(xs, ys, zs, inst, focal_crowns)
    if len(xs) == 0:
        raise RuntimeError("No LiDAR points remain after clipping to focal crowns")

    zoom_ortho, zoom_transform = read_ortho_patch(ortho_path, zoom_bounds)
    focal_colors = instance_colors(focal_ids)

    full_crowns = gpd.clip(crowns, gpd.GeoDataFrame(geometry=[box(*full_bounds)],
                                                    crs=crowns.crs))
    full_ortho, _ = read_ortho_patch(ortho_path, full_bounds)
    all_crown_ids = sorted(int(v) for v in full_crowns["V1"].unique() if v >= 0)
    full_color_map = instance_colors(all_crown_ids)

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax_plan_z = fig.add_subplot(gs[0, 0])
    ax_obl_z = fig.add_subplot(gs[0, 1], projection="3d")
    ax_crn_z = fig.add_subplot(gs[1, 0])
    ax_full = fig.add_subplot(gs[1, 1])

    plot_plan_lidar(
        ax_plan_z, xs, ys, inst, focal_colors, zoom_bounds,
        point_size=0.8,
        alpha=0.95,
        title="Zoom · TLS LiDAR plan view",
    )
    plot_oblique_lidar(
        ax_obl_z, xs, ys, zs, inst, focal_colors, zoom_bounds,
        args.view_elev, args.view_azim, args.z_exaggeration,
        point_size=1.2,
        alpha=0.95,
        title="Zoom · TLS LiDAR oblique",
    )
    plot_crown_panel(
        ax_crn_z,
        zoom_ortho,
        zoom_bounds,
        focal_crowns,
        focal_colors,
        focal_ids,
        title="Zoom · crown polygons on ortho",
        grey_outside=True,
        transform=zoom_transform,
    )

    plot_full_ortho_rotated(ax_full, full_ortho, full_bounds, full_crowns,
                            full_color_map)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved {output_path}")
    print(f"Focal instance IDs: {focal_ids}")
    print(f"Zoom bounds (UTM 32N): {zoom_bounds}")
    print(f"Full bounds (UTM 32N): {full_bounds}")
    print("Zoom LiDAR points after crown clip:", f"{len(xs):,}")


if __name__ == "__main__":
    main()
