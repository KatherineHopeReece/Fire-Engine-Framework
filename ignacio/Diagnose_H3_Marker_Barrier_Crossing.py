
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from rasterio.features import shapes
from shapely.geometry import shape, Point
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import translate


# ============================================================
# H3 MARKER BARRIER-CROSSING DIAGNOSTIC
# ============================================================

BASE = Path(
    "/Users/Martyn/GitRepos/Fire-Engine-Framework-dev/"
    "ignacio/LaughTests/Bova_1km_Experiments/"
    "H3_Checkerboard_Barriers/2m"
)

FUEL_PATH = BASE / "fuels_Bova_Spaced.tif"

# IMPORTANT:
# Point this to the Marker Euler GeoPackage used to
# produce your latest spaced-checkerboard comparison.
MARKER_GPKG = (
    BASE / "Marker_Euler_Output"
    / "all_perimeters_history.gpkg"
)

OUTPUT_DIR = BASE / "H3_Marker_Barrier_Diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NONBURNABLE = 101
PLOT_TIME = 40.0


# ============================================================
# 1. READ FUEL RASTER
# ============================================================

with rasterio.open(FUEL_PATH) as src:
    fuel = src.read(1)
    transform = src.transform
    crs = src.crs
    bounds = src.bounds

barrier_mask = fuel == NONBURNABLE

barrier_polygons = [
    shape(geom)
    for geom, value in shapes(
        barrier_mask.astype("uint8"),
        mask=barrier_mask,
        transform=transform,
    )
    if int(value) == 1
]

barrier_union = unary_union(barrier_polygons)

print("\nNON-BURNABLE LANDSCAPE")
print("Patches:", len(barrier_polygons))
print("Barrier area (ha):", barrier_union.area / 10000)


# ============================================================
# 2. READ MARKER HISTORY
# ============================================================

if not MARKER_GPKG.exists():
    raise FileNotFoundError(MARKER_GPKG)

gdf = gpd.read_file(MARKER_GPKG)

if gdf.crs is None:
    raise ValueError("Marker GeoPackage has no CRS.")

gdf = gdf.to_crs(crs)

gdf["t_min"] = pd.to_numeric(
    gdf["t_min"], errors="coerce"
).round(5)

gdf = gdf.dropna(subset=["t_min"])

print("\nMARKER HISTORY")
print("Source:", MARKER_GPKG)
print("Records:", len(gdf))


# ============================================================
# 3. GEOMETRY HELPERS
# ============================================================

def polygon_parts(geom):

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    return []


def get_vertices(geom):

    vertices = []

    for poly in polygon_parts(geom):
        coords = np.asarray(
            poly.exterior.coords
        )[:-1]

        vertices.extend(coords)

    return np.asarray(vertices)


def local_coordinates(geom):

    return translate(
        geom,
        xoff=-bounds.left,
        yoff=-bounds.bottom,
    )


# ============================================================
# 4. ANALYZE ALL SAVED SNAPSHOTS
# ============================================================

rows = []
geometries = {}

for time, group in gdf.groupby(
    "t_min", sort=True
):

    geom = unary_union(
        list(group.geometry)
    )

    if geom.is_empty:
        continue

    if not geom.is_valid:
        print(
            f"WARNING: Invalid perimeter at "
            f"{time:g} minutes"
        )
        continue

    geometries[float(time)] = geom

    # Area of non-burnable patches enclosed
    # by the exported marker polygon.
    intersection = geom.intersection(
        barrier_union
    )

    overlap_m2 = intersection.area

    # Length of marker boundary passing
    # through non-burnable patches.
    boundary_inside = (
        geom.boundary.intersection(barrier_union)
    )

    crossing_length = boundary_inside.length

    # Check individual exported marker vertices.
    vertices = get_vertices(geom)

    inside_count = sum(
        barrier_union.contains(
            Point(float(x), float(y))
        )
        for x, y in vertices
    )

    # Count complete obstacle patches that
    # the exported polygon represents as burned.
    enclosed_patches = sum(
        geom.covers(patch)
        for patch in barrier_polygons
    )

    # Check whether true interior holes are stored.
    hole_count = sum(
        len(poly.interiors)
        for poly in polygon_parts(geom)
    )

    rows.append({
        "time_min": float(time),
        "burned_area_ha": geom.area / 10000,
        "barrier_overlap_ha": overlap_m2 / 10000,
        "barrier_overlap_pct": (
            100 * overlap_m2 / geom.area
            if geom.area > 0 else np.nan
        ),
        "vertices_total": len(vertices),
        "vertices_inside_barrier": inside_count,
        "front_length_inside_barrier_m":
            crossing_length,
        "fully_enclosed_barrier_patches":
            enclosed_patches,
        "exported_interior_holes": hole_count,
    })


df = pd.DataFrame(rows).sort_values(
    "time_min"
)

csv_path = OUTPUT_DIR / "Barrier_Diagnostics.csv"
df.to_csv(csv_path, index=False)

print("\nBARRIER DIAGNOSTICS")
print(
    df[df["time_min"].isin(
        [5, 10, 20, 30, 40]
    )].to_string(index=False)
)

violations = df[
    df["barrier_overlap_ha"] > 1e-8
]

if not violations.empty:
    first = violations.iloc[0]

    print(
        "\nFirst observed barrier overlap:",
        first["time_min"],
        "minutes"
    )

print("\nSaved:", csv_path)


# ============================================================
# 5. VISUALIZE BARRIER CROSSING
# ============================================================

times = sorted(geometries)

plot_time = min(
    times,
    key=lambda t: abs(t - PLOT_TIME)
)

geom = geometries[plot_time]

overlap = geom.intersection(
    barrier_union
)

boundary_in_barrier = (
    geom.boundary.intersection(barrier_union)
)

geom_local = local_coordinates(geom)
overlap_local = local_coordinates(overlap)
crossing_local = local_coordinates(
    boundary_in_barrier
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 7),
    constrained_layout=True,
)

background = np.where(
    barrier_mask, 1, 0
)

cmap = ListedColormap([
    "#F1E5C4",
    "#666666",
])

for ax in axes:

    ax.imshow(
        background,
        extent=[0, 1000, 0, 1000],
        origin="upper",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    gpd.GeoSeries([
        geom_local.boundary
    ]).plot(
        ax=ax,
        color="#0072B2",
        linewidth=1.5,
        label="Marker perimeter",
    )

    if not overlap_local.is_empty:
        gpd.GeoSeries([
            overlap_local
        ]).plot(
            ax=ax,
            color="#D55E00",
            alpha=0.55,
        )

    if not crossing_local.is_empty:
        gpd.GeoSeries([
            crossing_local
        ]).plot(
            ax=ax,
            color="#B42318",
            linewidth=2.8,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("Distance east (m)")
    ax.set_ylabel("Distance north (m)")
    ax.grid(alpha=0.15)

axes[0].set_xlim(0, 1000)
axes[0].set_ylim(0, 1000)
axes[0].set_title(
    "Full H3 domain"
)

bx0, by0, bx1, by1 = (
    local_coordinates(barrier_union).bounds
)

axes[1].set_xlim(
    max(0, bx0 - 100),
    min(1000, bx1 + 100),
)

axes[1].set_ylim(
    max(0, by0 - 100),
    min(1000, by1 + 100),
)

axes[1].set_title(
    "Zoom: non-burnable patches"
)

fig.suptitle(
    f"H3 Marker Euler: barrier interaction at "
    f"t = {plot_time:g} min\n"
    "Orange = non-burnable area enclosed by "
    "the exported fire polygon",
    fontsize=14,
    fontweight="bold",
)

png_path = (
    OUTPUT_DIR / "H3_Marker_Barrier_Crossings.png"
)

fig.savefig(
    png_path,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", png_path)
print("\nDiagnostic complete.")
