from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.affinity import translate
from pyproj import CRS

# ============================================================
# 1. INPUT AND OUTPUT FILES
# ============================================================

input_shp = Path(
    "/Users/Martyn/GitRepos/Fire-Engine-Framework-dev/"
    "ignacio/LaughTests/Constant_Euler_Output/ignitions.shp"
)

output_shp = input_shp.with_name("ignitions_two.shp")

# Second ignition offset in metres
OFFSET_EAST = 80.0
OFFSET_NORTH = 60.0

# ============================================================
# 2. READ ORIGINAL IGNITION
# ============================================================

if not input_shp.exists():
    raise FileNotFoundError(f"File not found: {input_shp}")

gdf = gpd.read_file(input_shp)

print("\nOriginal ignition file:")
print(input_shp)
print(f"CRS: {gdf.crs}")
print(f"Number of ignitions: {len(gdf)}")

if len(gdf) != 1:
    raise ValueError("Expected exactly one original ignition.")

if gdf.crs is None:
    raise ValueError("Ignition shapefile has no defined CRS.")

if gdf.geometry.iloc[0].geom_type != "Point":
    raise ValueError("Ignition geometry must be a Point.")

original_crs = gdf.crs

# ============================================================
# 3. CONVERT TO METRE-BASED COORDINATES
# ============================================================

crs = CRS.from_user_input(original_crs)

if (
    crs.is_projected
    and crs.axis_info
    and abs(crs.axis_info[0].unit_conversion_factor - 1.0) < 1e-6
):
    gdf_metric = gdf.copy()
else:
    metric_crs = gdf.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError("Could not determine a suitable metric CRS.")

    gdf_metric = gdf.to_crs(metric_crs)

# ============================================================
# 4. CREATE SECOND IGNITION
# ============================================================

ignition1 = gdf_metric.geometry.iloc[0]

ignition2 = translate(
    ignition1,
    xoff=OFFSET_EAST,
    yoff=OFFSET_NORTH
)

print("\nIgnition coordinates (metres):")
print(f"Ignition 1: ({ignition1.x:.3f}, {ignition1.y:.3f})")
print(f"Ignition 2: ({ignition2.x:.3f}, {ignition2.y:.3f})")

# Duplicate original ignition attributes
second = gdf_metric.iloc[[0]].copy()
second.geometry = [ignition2]

# Update numeric ID if present
for field in ["ignition_id", "ID", "id"]:
    if field in second.columns:
        if pd.api.types.is_numeric_dtype(gdf_metric[field]):
            second[field] = gdf_metric[field].max() + 1
        break

# Combine both points
combined = gpd.GeoDataFrame(
    pd.concat(
        [gdf_metric, second],
        ignore_index=True
    ),
    geometry="geometry",
    crs=gdf_metric.crs
)

# ============================================================
# 5. RETURN TO ORIGINAL CRS
# ============================================================

combined = combined.to_crs(original_crs)

# ============================================================
# 6. SAVE NEW SHAPEFILE
# ============================================================

combined.to_file(output_shp)

# Verify saved file
check = gpd.read_file(output_shp)

assert len(check) == 2, "Output should contain 2 ignitions."

print("\n============================================")
print("TWO-IGNITION SHAPEFILE CREATED SUCCESSFULLY")
print("============================================")

print(f"\nOutput: {output_shp}")
print(f"CRS: {check.crs}")
print(f"Number of ignitions: {len(check)}")
print(f"Separation: {(OFFSET_EAST**2 + OFFSET_NORTH**2)**0.5:.2f} m")
print("\nBoth ignition points created.")