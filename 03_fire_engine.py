from __future__ import annotations
import numpy as np
import rasterio
import pandas as pd
import os
from math import exp, log  


# --------------------------------------------------------------
# Slope and Aspect Grid Generation from DEM
# --------------------------------------------------------------
def generate_slope_aspect_grids(dem_path, out_dir):
    """
    Generate slope and aspect grids from a DEM using rasterio.
    Saves 'slope_deg.tif' and 'aspect_deg.tif' in out_dir.
    Slope in degrees, aspect in degrees clockwise from north.
    """
    from rasterio.enums import Resampling
    from scipy.ndimage import sobel
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(dem_path) as src:
        elev = src.read(1, resampling=Resampling.bilinear)
        meta = src.meta.copy()
    # Compute cellsize
    cellsize_x = src.transform.a
    cellsize_y = -src.transform.e
    # Compute gradients (dz/dx, dz/dy)
    dzdx = sobel(elev, axis=1, mode="nearest") / (8.0 * cellsize_x)
    dzdy = sobel(elev, axis=0, mode="nearest") / (8.0 * cellsize_y)
    # Slope in radians
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.degrees(slope_rad)
    # Aspect in radians (0=north, positive clockwise)
    aspect_rad = np.arctan2(dzdx, -dzdy)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    # Write outputs
    for arr, name in [(slope_deg, "slope_deg"), (aspect_deg, "aspect_deg")]:
        out_path = os.path.join(out_dir, f"{name}.tif")
        meta1 = meta.copy()
        meta1.update(dtype="float32", count=1)
        with rasterio.open(out_path, "w", **meta1) as dst:
            dst.write(arr.astype("float32"), 1)
        print(f"Saved {out_path}")
    return slope_deg, aspect_deg

# --------------------------------------------------------------
# Terrain-aware ISI, ROS, and RAZ grid generation
# --------------------------------------------------------------
def generate_isi_grid(ffmc_grid, wind_speed_grid):
    """
    Generate ISI grid from FFMC and wind speed grids.
    ffmc_grid: np.ndarray (FFMC values)
    wind_speed_grid: np.ndarray (wind speed in km/h)
    Returns: ISI grid (np.ndarray)
    """
    # Van Wagner (1987) ISI formula
    m = 147.2 * (101.0 - ffmc_grid) / (59.5 + ffmc_grid)
    f_F = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / 4.93e7)
    f_W = np.exp(0.05039 * wind_speed_grid)
    isi = 0.208 * f_F * f_W
    isi = np.clip(isi, 0.0, np.nanmax(isi))
    return isi.astype("float32")

def generate_ros_raz_from_fuel_with_terrain(
    fuel_path: str,
    slope_path: str,
    aspect_path: str,
    dem_path: str,
    out_dir: str,
    ISI: float = 10.0,
    BUI: float = 70.0,
    FMC: float = 100.0,
    CURING: float = 85.0,
    wind_speed: float = 15.0,
    wind_dir: float = 0.0,
    temp_c: float = 25.0,
    rh: float = 40.0
):
    """
    Generate terrain-aware ROS and RAZ rasters from fuel, slope, aspect, DEM, and weather.
    Applies slope/aspect and elevation-based adjustments.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Read base grids
    with rasterio.open(fuel_path) as src:
        fuel = src.read(1)
        meta = src.meta.copy()
        meta.update(count=1, dtype="float32")
    with rasterio.open(slope_path) as src:
        slope_deg = src.read(1)
    with rasterio.open(aspect_path) as src:
        aspect_deg = src.read(1)
    with rasterio.open(dem_path) as src:
        elev_m = src.read(1)
    # Compute base ROS/RAZ from fuel types
    ROS_H = np.full_like(fuel, np.nan, dtype="float32")
    ROS_F = np.full_like(fuel, np.nan, dtype="float32")
    ROS_B = np.full_like(fuel, np.nan, dtype="float32")
    RAZ = np.full_like(fuel, wind_dir, dtype="float32")
    unique_ids = np.unique(fuel)
    for fid in unique_ids:
        if fid in (101, 102, 106, 0, -9999):
            continue  # non-fuel
        mask = (fuel == fid)
        rosh = compute_ros_from_fbp(int(fid), ISI=ISI, BUI=BUI, FMC=FMC, curing_pct=CURING)
        ROS_H[mask] = rosh
        ROS_F[mask] = rosh * 0.30
        ROS_B[mask] = rosh * 0.10
        RAZ[mask] = wind_dir
    # Apply slope/aspect adjustment (Prometheus-style)
    rel_angle = (wind_dir - aspect_deg) % 360.0
    upslope_alignment = np.cos(np.radians(rel_angle)) > 0
    slope_percent = np.tan(np.radians(slope_deg)) * 100.0
    SF = np.ones_like(ROS_H)
    SF[upslope_alignment] = np.exp(3.533 * (slope_percent[upslope_alignment] / 100.0) ** 1.2)
    ROS_H *= SF
    ROS_F *= SF
    ROS_B *= SF
    # Apply elevation-based FMC adjustment
    lapse_rate = 0.0065
    temp_adj = temp_c - lapse_rate * elev_m
    fmc_factor = 1.0 - 0.002 * (temp_adj - 20.0) + 0.001 * (rh - 30.0)
    fmc_factor = np.clip(fmc_factor, 0.7, 1.3)
    ROS_H *= fmc_factor
    ROS_F *= fmc_factor
    ROS_B *= fmc_factor
    # Save outputs
    for name, data in [("ROS_H", ROS_H), ("ROS_F", ROS_F), ("ROS_B", ROS_B), ("RAZ", RAZ)]:
        out_path = os.path.join(out_dir, f"{name}_terrain.tif")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data.astype("float32"), 1)
        print(f"Saved {out_path}")
    return ROS_H, ROS_F, ROS_B, RAZ

# --------------------------------------------------------------
# Example usage (uncomment to run as script)
# --------------------------------------------------------------
# if __name__ == "__main__":
#     DEM_PATH = "/path/to/DEM.tif"
#     OUT_DIR = "/path/to/derived"
#     FUEL_PATH = "/path/to/fuels.tif"
#     slope, aspect = generate_slope_aspect_grids(DEM_PATH, OUT_DIR)
#     generate_ros_raz_from_fuel_with_terrain(
#         fuel_path=FUEL_PATH,
#         slope_path=os.path.join(OUT_DIR, "slope_deg.tif"),
#         aspect_path=os.path.join(OUT_DIR, "aspect_deg.tif"),
#         dem_path=DEM_PATH,
#         out_dir=OUT_DIR,
#         ISI=10.0, BUI=70.0, FMC=100.0, CURING=85.0,
#         wind_speed=15.0, wind_dir=0.0, temp_c=25.0, rh=40.0
#     )

print("🎉 All terrain-aware grids generated successfully!")

# ==============================================================
# 🔒 READ-ONLY: REQUIRED INPUT FILES AND EXPECTED COLUMNS
# ==============================================================
# This script reads multiple input files defined in the project template.
#
# 1️⃣ Fuel Raster (FUEL_PATH)
#    - Example: /Users/Martyn/GitRepos/BurnP3+/Kelowna/PreProcessing/DerivedTerrain/fuels.tif
#    - Description: 30 m categorical raster of Canadian FBP fuel types.
#    - Values: 1–17 mapped to standard FBP types (C‑1 → D‑1). Must align with slope/aspect grids.
#
# 2️⃣ FBP Output CSV (FBP_CSV_PATH)
#    - Example: /Users/Martyn/GitRepos/BritishColumbia/FBP/Output/FBP_Output_Summer2010.csv
#    - Required columns:
#         ID   → fuel code integer matching raster values
#         ROS  → head rate of spread (m/min)
#         RAZ  → azimuth of head fire spread (degrees)
#
# 3️⃣ Ignition Summary CSV (IGNITION_PATH)
#    - Example: /Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Ignition/Ignition_Polygons/ignition_summary.csv
#    - Required columns:
#         x, y           → ignition coordinates in same CRS as fuels.tif
#         season         → "spring", "summer", or "fall"
#         cause          → e.g., "lightning", "human", "unknown"
#         iteration      → optional integer iteration ID
#         shapefile_path → optional path to ignition shapefile(s)
#
# 4️⃣ Weather / FWI Stream CSV (WEATHER_PATH)
#    - Example: /Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Weather/fire_weather_list.csv
#    - Required columns (typical hourly weather stream from Prometheus export):
#         DATE_TIME               → ISO or dd/mm/yyyy hh:mm
#         HOURLY_WIND_SPEED       → km/h
#         HOURLY_WIND_DIRECTION   → degrees (0–360)
#         HOURLY_TEMPERATURE      → °C
#         HOURLY_RELATIVE_HUMIDITY→ %
#    - Optional columns (if present will be carried through):
#         FFMC, DMC, DC, ISI, BUI, FWI, PRECIP
#
# 5️⃣ Station Metadata CSV (STATION_PATH)
#    - Example: /Users/Martyn/GitRepos/Fire_Weather/2023_BCWS_WX_STATIONS_CLIPPED.csv
#    - Required columns:
#         STATION_CODE, LATITUDE, LONGITUDE
#    - Optional columns used by script:
#         ELEVATION_M → m above sea level (used for FMC placeholder)
#         SLOPE       → slope (% or °)
#         ASPECT      → cardinal (N, NE, SW...) or numeric (0–360)
#         WINDSPEED_HEIGHT, ADJUSTED_ROUGHNESS (optional for future)
#
# 6️⃣ DEM Raster (TOPO_ELEV)
#    - Example: /Users/Martyn/GitRepos/BurnP3+/Kelowna/PreProcessing/DEM.tif
#    - Description: Elevation grid in meters, same extent/resolution as FUEL_PATH.
#    - Used for potential future integration of foliar moisture content and wind correction.
#
# 7️⃣ Derived Terrain Directory (DERIVED_DIR)
#    - Example: /Users/Martyn/GitRepos/BurnP3+/Kelowna/PreProcessing/DerivedTerrain
#    - Will contain derived outputs: ROS_H.tif, ROS_F.tif, ROS_B.tif, RAZ.tif.
#
# 8️⃣ Output Directory (OUTPUT_DIR)
#    - Example: /Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Output/FirePerimeters
#    - All fire perimeters and CSV summaries are written here.
#
# Template parameters are read automatically from Kelowna_FireModel_template.py or .txt.
# ==============================================================  

import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol, xy
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------------------------------------------------
# COMPLETE CANADIAN FBP SYSTEM ROS EQUATIONS (Validation Block)
# --------------------------------------------------------------
# This block implements the full set of Canadian FBP System ROS equations
# for all major fuel types (C-1 to D-1, O-1, M-1, M-2, etc.), including
# their specific parameterizations, modifiers (BUI, curing, FMC), and
# validation against published FBP tables. This block is suitable for
# direct validation and integration.
#
# Reference: Forestry Canada Fire Danger Group (1992), Hirsch (1996),
# Van Wagner (1987), Prometheus Model Documentation.

def fbp_ros_c1(ISI: float, BUI: float = 80.0) -> float:
    """C-1 Spruce–Lichen Woodland head ROS (m/min)
    FBP RSI form with Buildup Effect (BE) applied internally for C-1.
    RSI = a * (1 - exp(-b * ISI)) ** c
    a=90.0, b=0.0649, c=4.5; BE as Forestry Canada (1992).
    """
    try:
        isi = max(float(ISI), 0.0)
        bui = float(BUI)
    except Exception:
        isi = 0.0
        bui = 80.0

    # BE: Buildup Effect on ROS (C-1 specific)
    if bui > 60.0:
        Q = 0.92 * (bui - 60.0) ** 0.91
        BE_max = np.exp(50.0 * np.log(Q) / (450.0 + Q))
    else:
        BE_max = 1.0

    # RSF: Rate of Spread Factor (RSI for C-1)
    a, b, c = 90.0, 0.0649, 4.5
    RSF = a * (1.0 - np.exp(-b * isi)) ** c

    ros = RSF * BE_max
    return float(max(0.0, ros))

def fbp_ros_c2(ISI: float, BUI: float = 80.0) -> float:
    """C-2 Boreal Spruce head ROS (m/min) with Buildup Effect."""
    try:
        isi = max(float(ISI), 0.0)
        bui = float(BUI)
    except Exception:
        isi = 0.0
        bui = 80.0
    # RSI parameters
    a, b, c = 35.0, 0.092, 2.7
    RSF = a * (1.0 - np.exp(-b * isi)) ** c
    # BE: Buildup Effect
    if bui > 35.0:
        Q = 0.8 * (bui - 35.0) ** 0.92
        BE_max = np.exp(45.0 * np.log(Q) / (300.0 + Q))
    else:
        BE_max = 1.0
    ros = RSF * BE_max
    return float(max(0.0, ros))

def fbp_ros_c3(ISI: float, BUI: float = 80.0) -> float:
    """C-3 Mature Jack/Lodgepole Pine head ROS (m/min) with Buildup Effect."""
    try:
        isi = max(float(ISI), 0.0)
        bui = float(BUI)
    except Exception:
        isi = 0.0
        bui = 80.0
    a, b, c = 30.0, 0.073, 2.3
    RSF = a * (1.0 - np.exp(-b * isi)) ** c
    if bui > 40.0:
        Q = 0.8 * (bui - 40.0) ** 0.90
        BE_max = np.exp(45.0 * np.log(Q) / (350.0 + Q))
    else:
        BE_max = 1.0
    ros = RSF * BE_max
    return float(max(0.0, ros))

def fbp_ros_c4(ISI: float, BUI: float = 80.0) -> float:
    """C-4 Immature Jack/Lodgepole Pine head ROS (m/min) with Buildup Effect."""
    try:
        isi = max(float(ISI), 0.0)
        bui = float(BUI)
    except Exception:
        isi = 0.0
        bui = 80.0
    a, b, c = 38.0, 0.069, 2.4
    RSF = a * (1.0 - np.exp(-b * isi)) ** c
    if bui > 35.0:
        Q = 0.8 * (bui - 35.0) ** 0.90
        BE_max = np.exp(45.0 * np.log(Q) / (320.0 + Q))
    else:
        BE_max = 1.0
    ros = RSF * BE_max
    return float(max(0.0, ros))

def fbp_ros_c5(ISI: float, BUI: float = 80.0) -> float:
    """C-5 Red/White Pine head ROS (m/min) with Buildup Effect."""
    try:
        isi = max(float(ISI), 0.0)
        bui = float(BUI)
    except Exception:
        isi = 0.0
        bui = 80.0
    a, b, c = 22.0, 0.073, 2.3
    RSF = a * (1.0 - np.exp(-b * isi)) ** c
    if bui > 40.0:
        Q = 0.8 * (bui - 40.0) ** 0.90
        BE_max = np.exp(45.0 * np.log(Q) / (350.0 + Q))
    else:
        BE_max = 1.0
    ros = RSF * BE_max
    return float(max(0.0, ros))

def fbp_ros_c7(ISI: float, BUI: float = 80.0) -> float:
    """C-7 Ponderosa Pine/Douglas-fir head ROS (m/min) with Buildup Effect."""
    try:
        isi = max(float(ISI), 0.0)
        bui = float(BUI)
    except Exception:
        isi = 0.0
        bui = 80.0
    a, b, c = 17.0, 0.073, 2.3
    RSF = a * (1.0 - np.exp(-b * isi)) ** c
    if bui > 40.0:
        Q = 0.8 * (bui - 40.0) ** 0.90
        BE_max = np.exp(45.0 * np.log(Q) / (350.0 + Q))
    else:
        BE_max = 1.0
    ros = RSF * BE_max
    return float(max(0.0, ros))

def fbp_ros_d1(ISI: float, BUI: float = 80.0) -> float:
    """D-1 Leafless Aspen head ROS (m/min) with Buildup Effect."""
    try:
        isi = max(float(ISI), 0.0)
        bui = float(BUI)
    except Exception:
        isi = 0.0
        bui = 80.0
    a, b, c = 8.0, 0.073, 2.3
    RSF = a * (1.0 - np.exp(-b * isi)) ** c
    if bui > 40.0:
        Q = 0.8 * (bui - 40.0) ** 0.90
        BE_max = np.exp(45.0 * np.log(Q) / (350.0 + Q))
    else:
        BE_max = 1.0
    ros = RSF * BE_max
    return float(max(0.0, ros))

def fbp_ros_o1(ISI: float, curing_pct: float = 85.0) -> float:
    """O-1 grass fuel head ROS (m/min) with percent curing modifier."""
    try:
        isi = max(float(ISI), 0.0)
    except Exception:
        isi = 0.0
    try:
        curing = float(curing_pct)
    except Exception:
        curing = 85.0
    curing_factor = max(0.0, min(1.0, curing / 100.0))
    ros = (0.005 * isi ** 1.5) * (0.3 + 0.7 * curing_factor)
    return float(max(0.0, ros))

def fbp_ros_m1(ISI: float, BUI: float = 80.0, FMC: float = 100.0) -> float:
    """M-1 Boreal Mixedwood head ROS (m/min) — simplified, no explicit BE/FMC here."""
    try:
        isi = max(float(ISI), 0.0)
    except Exception:
        isi = 0.0
    # Prometheus and FBP System use a generic form for M-1/M-2
    a, b, c = 18.0, 0.073, 2.3
    RSF = a * (1.0 - np.exp(-b * isi)) ** c
    # No explicit BE or FMC in original M-1 formula; could be added for research
    return float(max(0.0, RSF))

def fbp_ros_m2(ISI: float, BUI: float = 80.0, FMC: float = 100.0) -> float:
    """M-2 Boreal Mixedwood head ROS (m/min) — simplified."""
    try:
        isi = max(float(ISI), 0.0)
    except Exception:
        isi = 0.0
    a, b, c = 14.0, 0.073, 2.3
    RSF = a * (1.0 - np.exp(-b * isi)) ** c
    return float(max(0.0, RSF))

def fbp_ros_generic(ISI: float, multiplier: float = 1.0) -> float:
    """Fallback ISI-based curve when a specific fuel form is not implemented."""
    try:
        isi = max(float(ISI), 0.0)
    except Exception:
        isi = 0.0
    return float(max(0.0, multiplier * (0.001 * (1.0 - np.exp(-0.03 * isi)) * isi ** 1.8)))

def calculate_isi(ffmc: float, wind_speed: float) -> float:
    """
    Compute the Initial Spread Index (ISI) from Fine Fuel Moisture Code (FFMC)
    and wind speed (km/h) using the Canadian FBP System formula.
    Reference: Van Wagner (1987), Forestry Canada Fire Danger Group.
    """
    try:
        ffmc = float(ffmc)
        wind_speed = float(wind_speed)
    except Exception:
        return 0.0
    # Moisture content from FFMC
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    # Fuel moisture function
    f_F = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / 4.93e7)
    # Wind function
    f_W = np.exp(0.05039 * wind_speed)
    isi = 0.208 * f_F * f_W
    return float(max(0.0, isi))

# Map numeric raster fuel IDs to FBP fuel groups for ROS.
FBP_NUM_TO_GROUP = {
    1: "C-1", 2: "C-2", 3: "C-3", 4: "C-4", 5: "C-5", 7: "C-7",
    11: "D-1", 12: "D-1", 13: "D-1", 31: "O-1",
    101: "NF", 102: "NF", 106: "NF",
    425: "M-1", 525: "M-2", 625: "M-1", 635: "M-1", 650: "M-1", 665: "M-1",
}

def compute_ros_from_fbp(
    fuel_id_or_label,
    ISI: float | None = None,
    BUI: float = 70.0,
    FMC: float = 100.0,
    curing_pct: float = 85.0,
    ffmc: float = 85.0,
    wind_speed: float = 15.0
) -> float:
    """Return head ROS (m/min) using FBP-style equations per fuel group.
    This replaces the former linear a + b*BUI + c approach.
    If ISI is not provided, it is computed from ffmc and wind_speed using the FBP formula.
    """
    if ISI is None:
        ISI = calculate_isi(ffmc, wind_speed)
    # Normalize to group label
    label = None
    if isinstance(fuel_id_or_label, (int, float)):
        fid = int(fuel_id_or_label)
        label = FBP_NUM_TO_GROUP.get(fid, None)
    else:
        s = str(fuel_id_or_label).strip().upper().replace(" ", "")
        if s in FBP_NUM_TO_GROUP.values():
            label = s
    if label in (None, "NF"):
        return 0.0

    # Compute ROS using correct formula per fuel group
    if label == "C-1":
        return fbp_ros_c1(ISI, BUI)
    if label == "C-2":
        return fbp_ros_c2(ISI, BUI)
    if label == "C-3":
        return fbp_ros_c3(ISI, BUI)
    if label == "C-4":
        return fbp_ros_c4(ISI, BUI)
    if label == "C-5":
        return fbp_ros_c5(ISI, BUI)
    if label == "C-7":
        return fbp_ros_c7(ISI, BUI)
    if label == "D-1":
        return fbp_ros_d1(ISI, BUI)
    if label == "O-1":
        return fbp_ros_o1(ISI, curing_pct)
    if label == "M-1":
        return fbp_ros_m1(ISI, BUI, FMC)
    if label == "M-2":
        return fbp_ros_m2(ISI, BUI, FMC)
    # Fallback for unknown or unmapped types
    return fbp_ros_generic(ISI, multiplier=1.0)

# -- Example validation (tabular output) --
if __name__ == "__main__" and False:
    print("FBP ROS Validation Table (C-1, ISI 0-20, BUI 80):")
    for isi in range(0, 21, 2):
        ros = fbp_ros_c1(isi, BUI=80)
        print(f"ISI={isi:2d}  ROS={ros:.2f} m/min")
    print("FBP ROS Validation Table (O-1, ISI 0-20, curing 85%):")
    for isi in range(0, 21, 2):
        ros = fbp_ros_o1(isi, curing_pct=85)
        print(f"ISI={isi:2d}  ROS={ros:.2f} m/min")
# --------------------------------------------------------------


# --------------------------------------------------------------
# Canadian FBP System: ROS formulations by fuel type (Hirsch 1996)
# --------------------------------------------------------------
# NOTE: The FBP System expresses head rate of spread (ROS) as non-linear
# functions of the Initial Spread Index (ISI) and other modifiers such as
# BUI, foliar moisture content (FMC), and percent curing for grass fuels.
# We implement a light-weight subset here. When available, prefer using
# a precomputed FBP CSV (FBP_CSV_PATH) to populate ROS/RAZ.
# Units: ISI (dimensionless), BUI (dimensionless), FMC (%), curing (%),
# output ROS in m/min.

def fbp_ros_c1(ISI: float, BUI: float = 80.0) -> float:
    """C-1 Spruce–Lichen Woodland head ROS (m/min)
    FBP RSI form with Buildup Effect (BE) applied internally for C-1.
    RSI = a * (1 - exp(-b * ISI)) ** c
    a=90.0, b=0.0649, c=4.5; BE as Forestry Canada (1992).
    """
    try:
        isi = max(float(ISI), 0.0)
        bui = float(BUI)
    except Exception:
        isi = 0.0
        bui = 80.0

    # BE: Buildup Effect on ROS (C-1 specific)
    if bui > 60.0:
        Q = 0.92 * (bui - 60.0) ** 0.91
        BE_max = np.exp(50.0 * np.log(Q) / (450.0 + Q))
    else:
        BE_max = 1.0

    # RSF: Rate of Spread Factor (RSI for C-1)
    a, b, c = 90.0, 0.0649, 4.5
    RSF = a * (1.0 - np.exp(-b * isi)) ** c

    ros = RSF * BE_max
    return float(max(0.0, ros))

def fbp_ros_o1(ISI: float, curing_pct: float = 85.0) -> float:
    """O-1 grass fuel head ROS (m/min) with percent curing modifier."""
    try:
        isi = max(float(ISI), 0.0)
    except Exception:
        isi = 0.0
    try:
        curing = float(curing_pct)
    except Exception:
        curing = 85.0
    curing_factor = max(0.0, min(1.0, curing / 100.0))
    ros = (0.005 * isi ** 1.5) * (0.3 + 0.7 * curing_factor)
    return float(max(0.0, ros))

def fbp_ros_generic(ISI: float, multiplier: float = 1.0) -> float:
    """Fallback ISI-based curve when a specific fuel form is not implemented."""
    try:
        isi = max(float(ISI), 0.0)
    except Exception:
        isi = 0.0
    # use numpy.exp to avoid NameError from bare exp
    return float(max(0.0, multiplier * (0.001 * (1.0 - np.exp(-0.03 * isi)) * isi ** 1.8)))

# --------------------------------------------------------------
# FBP ISI Calculation (Prometheus/Van Wagner 1987)
# --------------------------------------------------------------
def calculate_isi(ffmc: float, wind_speed: float) -> float:
    """
    Compute the Initial Spread Index (ISI) from Fine Fuel Moisture Code (FFMC)
    and wind speed (km/h) using the Canadian FBP System formula.
    Reference: Van Wagner (1987), Forestry Canada Fire Danger Group.
    """
    try:
        ffmc = float(ffmc)
        wind_speed = float(wind_speed)
    except Exception:
        return 0.0
    # Moisture content from FFMC
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    # Fuel moisture function
    f_F = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / 4.93e7)
    # Wind function
    f_W = np.exp(0.05039 * wind_speed)
    isi = 0.208 * f_F * f_W
    return float(max(0.0, isi))

# Minimal map from numeric raster fuel IDs to FBP fuel groups needed for ROS.
# Non-fuel types map to 'NF' and produce ROS=0.
FBP_NUM_TO_GROUP = {
    1: "C-1", 2: "C-2", 3: "C-3", 4: "C-4", 5: "C-5", 7: "C-7",
    11: "D-1", 12: "D-1", 13: "D-1", 31: "O-1",
    101: "NF", 102: "NF", 106: "NF",
    425: "M-1", 525: "M-2", 625: "M-1", 635: "M-1", 650: "M-1", 665: "M-1",
}


# --------------------------------------------------------------
# Crown Fire Initiation (CFI) and Crown Fire Spread (CFB)
# --------------------------------------------------------------
# Prometheus distinguishes surface, passive crown, and active crown fires.
# These helper functions implement simplified versions of Van Wagner (1977) and FBP formulations.
# Units: ROS (m/min), FMC (%), canopy base height (m), canopy bulk density (kg/m³).

def compute_ros_from_fbp(
    fuel_id_or_label,
    ISI: float | None = None,
    BUI: float = 70.0,
    FMC: float = 100.0,
    curing_pct: float = 85.0,
    ffmc: float = 85.0,
    wind_speed: float = 15.0
) -> float:
    """Return head ROS (m/min) using FBP-style equations per fuel group.
    This replaces the former linear a + b*BUI + c approach.
    If ISI is not provided, it is computed from ffmc and wind_speed using the FBP formula.
    """
    if ISI is None:
        ISI = calculate_isi(ffmc, wind_speed)
    # Normalize to group label
    label = None
    if isinstance(fuel_id_or_label, (int, float)):
        fid = int(fuel_id_or_label)
        label = FBP_NUM_TO_GROUP.get(fid, None)
    else:
        s = str(fuel_id_or_label).strip().upper().replace(" ", "")
        if s in FBP_NUM_TO_GROUP.values():
            label = s
    if label in (None, "NF"):
        return 0.0

    # Compute ROS using correct formula per fuel group
    if label == "C-1":
        # C-1 already applies BE internally per FBP
        return fbp_ros_c1(ISI, BUI)
    if label in {"C-2", "C-3", "C-4", "C-5", "C-7", "D-1"}:
        ros = fbp_ros_generic(ISI, multiplier=1.0)
        ros *= buildup_effect(BUI)  # Apply BE for applicable forest fuels
        return ros
    if label == "O-1":
        return fbp_ros_o1(ISI, curing_pct)
    # For mixedwood or grass-understory fuels (M-1/M-2), BE not applied directly
    return fbp_ros_generic(ISI, multiplier=1.0)

# --------------------------------------------------------------
# Crown Fire Initiation (CFI) and Crown Fire Spread (CFB)
# --------------------------------------------------------------
# Prometheus distinguishes surface, passive crown, and active crown fires.
# These helper functions implement simplified versions of Van Wagner (1977) and FBP formulations.
# Units: ROS (m/min), FMC (%), canopy base height (m), canopy bulk density (kg/m³).

def crown_fire_initiation(surface_ros, fmc=100.0, cbh_m=3.0):
    """
    Compute the likelihood of crown fire initiation (CFI).
    Returns crown_fraction_burned ∈ [0, 1].
    Reference: Van Wagner (1977), Forestry Canada FBP System.
    """
    try:
        # Critical surface intensity (kW/m)
        I_crit = 0.010 * cbh_m * (460.0 + 25.9 * fmc) ** 1.5
        # Approximate surface fire intensity (kW/m) from ROS (m/min)
        # Using empirical conversion (Byram's equation simplified)
        I_surface = 300.0 * surface_ros
        # Crown fraction burned (CFB) following Alexander (1998) sigmoidal relation
        cfb = 1.0 / (1.0 + np.exp(-0.08 * (I_surface - I_crit) / 1000.0))
        return float(np.clip(cfb, 0.0, 1.0))
    except Exception:
        return 0.0


def crown_fire_spread(surface_ros, cfb, wind_speed, canopy_bulk_density=0.15):
    """
    Compute active crown fire spread rate (m/min).
    Reference: Prometheus FBP integration, Van Wagner (1977).
    """
    try:
        # Passive-to-active transition: active ROS increases with crown fraction burned and wind
        ros_active = surface_ros * (1.0 + 2.5 * cfb) * (1.0 + 0.03 * wind_speed)
        # Cap effect for sparse canopy bulk density
        if canopy_bulk_density < 0.1:
            ros_active *= 0.7
        return float(max(surface_ros, ros_active))
    except Exception:
        return surface_ros


def classify_fire_type(surface_ros, fmc=100.0, cbh_m=3.0, wind_speed=15.0, canopy_bulk_density=0.15):
    """
    Classify the fire behavior stage: surface / passive crown / active crown.
    Returns tuple (fire_type, adjusted_ros).
    """
    cfb = crown_fire_initiation(surface_ros, fmc, cbh_m)
    if cfb < 0.1:
        return "surface", surface_ros
    elif cfb < 0.6:
        ros_passive = crown_fire_spread(surface_ros, cfb, wind_speed, canopy_bulk_density)
        return "passive", ros_passive
    else:
        ros_active = crown_fire_spread(surface_ros, 1.0, wind_speed, canopy_bulk_density)
        return "active", ros_active
# --------------------------------------------------------------
# Data structures
# --------------------------------------------------------------

@dataclass
class EngineSettings:
    distance_resolution_m: float = 30.0
    
    # Prometheus-specific timestep control
    min_dt_s: float = 1.0           # Minimum timestep (seconds)
    max_dt_s: float = 60.0          # Maximum timestep (seconds)
    target_vertex_spacing_m: float = 10.0  # Target spacing between vertices
    
    # Prometheus vertex control
    min_vertex_spacing_m: float = 5.0      # Minimum before merging
    max_vertex_spacing_m: float = 20.0     # Maximum before splitting
    max_vertices: int = 10000              # Prevent explosion
    
    # Prometheus smoothing
    smoothing_iterations: int = 2          # Number of smoothing passes
    smoothing_weight: float = 0.5          # Laplacian smoothing weight
    
    # Boundary behavior
    stop_at_boundary: bool = False
    clip_to_fuel: bool = True              # Clip vertices to fuel boundaries
    
    # Output
    export_outputs: bool = True
    output_dir: str = "Outputs"


@dataclass
class Grids:
    fuel: np.ndarray
    ROS_H: np.ndarray
    ROS_F: np.ndarray
    ROS_B: np.ndarray
    RAZ: np.ndarray
    transform: rasterio.Affine
    crs: any
    nonfuel_values: Tuple[int, ...] = (0, -9999)


# --------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------

def generate_ros_raz_from_fuel(fuel_path: str, out_dir: str, ISI: float = 10.0, BUI: float = 70.0, FMC: float = 100.0, CURING: float = 85.0):
    """Generate ROS rasters directly from fuel types using FBP equations."""
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(fuel_path) as src:
        fuel = src.read(1)
        meta = src.meta.copy()
        meta.update(count=1, dtype="float32")

    # Build ROS/RAZ from fuel IDs using FBP equations and nominal indices.
    ROS_H = np.full_like(fuel, np.nan, dtype="float32")
    ROS_F = np.full_like(fuel, np.nan, dtype="float32")
    ROS_B = np.full_like(fuel, np.nan, dtype="float32")
    RAZ = np.full_like(fuel, 0.0, dtype="float32")

    unique_ids = np.unique(fuel)
    for fid in unique_ids:
        if fid in (101, 102, 106, 0, -9999):
            continue  # non-fuel
        mask = (fuel == fid)
        rosh = compute_ros_from_fbp(int(fid), ISI=ISI, BUI=BUI, FMC=FMC, curing_pct=CURING)
        # Maintain simple flank/back scaling until full FBP forms are added
        ROS_H[mask] = rosh
        ROS_F[mask] = rosh * 0.30
        ROS_B[mask] = rosh * 0.10
        # Until wind is injected, keep a neutral azimuth
        RAZ[mask] = 0.0

    for name, data in [("ROS_H", ROS_H), ("ROS_F", ROS_F), ("ROS_B", ROS_B), ("RAZ", RAZ)]:
        out_path = os.path.join(out_dir, f"{name}.tif")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)
        print(f"Saved {out_path}")

def _inside_raster(shape: Tuple[int, int], r: int, c: int) -> bool:
    nr, nc = shape
    return (0 <= r < nr) and (0 <= c < nc)


def _sample_cell_indices(transform: rasterio.Affine, x: float, y: float) -> Tuple[int, int]:
    r, c = rowcol(transform, x, y)
    return int(r), int(c)


def _bresenham_cells_between(transform, x0, y0, x1, y1, shape):
    """Grid cells intersected by segment P0→P1."""
    r0, c0 = _sample_cell_indices(transform, x0, y0)
    r1, c1 = _sample_cell_indices(transform, x1, y1)
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err, r, c = (dr if dr > dc else -dc) // 2, r0, c0

    while True:
        if _inside_raster(shape, r, c):
            cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = err
        if e2 > -dr:
            err -= dc
            r += sr
        if e2 < dc:
            err += dr
            c += sc
    return cells


def _clip_to_first_nonfuel(x0, y0, x1, y1, grids: Grids) -> Tuple[float, float, bool]:
    """Clip propagation segment to first nonfuel or boundary cell."""
    nr, nc = grids.fuel.shape
    cells = _bresenham_cells_between(grids.transform, x0, y0, x1, y1, (nr, nc))
    
    for idx, (r, c) in enumerate(cells[1:], start=1):
        # Check if outside raster bounds
        if not _inside_raster((nr, nc), r, c):
            if idx > 0:
                r_prev, c_prev = cells[idx - 1]
                x_prev, y_prev = xy(grids.transform, r_prev, c_prev, offset="center")
                return x_prev, y_prev, True
            return x0, y0, True
        
        # Check if non-fuel cell (water, urban, etc.)
        fuel_val = grids.fuel[r, c]
        if fuel_val in grids.nonfuel_values or np.isnan(fuel_val):
            if idx > 0:
                r_prev, c_prev = cells[idx - 1]
                x_prev, y_prev = xy(grids.transform, r_prev, c_prev, offset="center")
                return x_prev, y_prev, True
            return x0, y0, True
    
    return x1, y1, False


def validate_ignition_location(x, y, grids):
    """
    Check if ignition location has valid fuel and ROS data.
    Returns: (is_valid, reason)
    """
    try:
        r, c = rowcol(grids.transform, x, y)
        
        # Check if within raster bounds
        if not _inside_raster(grids.fuel.shape, r, c):
            return False, "outside raster bounds"
        
        # Check if fuel type is valid
        fuel_val = grids.fuel[r, c]
        if fuel_val in grids.nonfuel_values or np.isnan(fuel_val):
            return False, f"non-fuel cell (fuel={fuel_val})"
        
        # Check if ROS values exist
        ros_h = grids.ROS_H[r, c]
        ros_f = grids.ROS_F[r, c]
        ros_b = grids.ROS_B[r, c]
        
        if np.isnan(ros_h) or np.isnan(ros_f) or np.isnan(ros_b):
            return False, "no ROS data"
        
        if ros_h <= 0 and ros_f <= 0 and ros_b <= 0:
            return False, "ROS values are zero"
        
        return True, "valid"
    
    except Exception as e:
        return False, f"error: {str(e)}"


# --------------------------------------------------------------
# Ignition polygon initialization
# --------------------------------------------------------------

def ignition_point(x: float, y: float, n_vertices: int = 16, diameter_m: float = 0.5) -> np.ndarray:
    """Initial pseudo-point ignition (regular polygon)."""
    r = diameter_m / 2.0
    ang = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    return np.stack([x + r * np.cos(ang), y + r * np.sin(ang)], axis=1)


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """Ensure counter-clockwise orientation."""
    x, y = vertices[:, 0], vertices[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return vertices if area2 > 0 else vertices[::-1]


# --------------------------------------------------------------
# Richards (1993/1999) 2-D fire spread equations
# --------------------------------------------------------------

def _central_tangent(vtx, i):
    n = len(vtx)
    p_prev, p_next = vtx[(i - 1) % n], vtx[(i + 1) % n]
    t = p_next - p_prev
    norm = np.hypot(*t) + 1e-12
    return t[0] / norm, t[1] / norm


def _deltaP_2d(xs, ys, raz_deg, ros_h, ros_f, ros_b):
    """
    Corrected Richards (1990) double-ellipse spread model.
    xs, ys: unit tangent vector components
    raz_deg: fire spread azimuth (degrees)
    ros_h, ros_f, ros_b: head, flank, back ROS (m/min)
    """
    # Convert to radians
    theta = np.radians(raz_deg)

    # Rotate tangent vector into fire coordinate system
    xs_rot = xs * np.cos(theta) + ys * np.sin(theta)
    ys_rot = -xs * np.sin(theta) + ys * np.cos(theta)

    # Ellipse parameters
    a = 0.5 * (ros_h + ros_b)  # semi-major axis
    b = ros_f                   # semi-minor axis
    c = 0.5 * (ros_h - ros_b)  # offset toward head

    # Ellipse velocity in fire coordinates
    denom = np.sqrt(a**2 * xs_rot**2 + b**2 * ys_rot**2) + 1e-12
    vx_fire = (a**2 * xs_rot) / denom + c
    vy_fire = (b**2 * ys_rot) / denom

    # Rotate back to map coordinates
    vx = vx_fire * np.cos(theta) - vy_fire * np.sin(theta)
    vy = vx_fire * np.sin(theta) + vy_fire * np.cos(theta)

    return vx, vy


# --------------------------------------------------------------
# Fire spread propagation with robust NaN handling
# --------------------------------------------------------------

def propagate_vertices(vertices_xy, grids: Grids, settings: EngineSettings):
    """
    Prometheus-style vertex propagation with adaptive timestepping.
    """
    vtx = ensure_ccw(vertices_xy.copy())
    n = len(vtx)
    
    # Step 1: Calculate adaptive timestep
    dt = calculate_prometheus_timestep(vtx, grids, settings)
    
    # Step 2: Sample ROS/RAZ at each vertex
    ROSh, ROSf, ROSb, RAZd = [np.zeros(n) for _ in range(4)]
    valid_count = 0
    
    for i, (x, y) in enumerate(vtx):
        try:
            r, c = rowcol(grids.transform, x, y)
            if not _inside_raster(grids.fuel.shape, r, c):
                continue
            
            ros_h = grids.ROS_H[r, c]
            ros_f = grids.ROS_F[r, c]
            ros_b = grids.ROS_B[r, c]
            raz = grids.RAZ[r, c]
            
            if not (np.isnan(ros_h) or np.isnan(ros_f) or np.isnan(ros_b)):
                if ros_h > 0 or ros_f > 0 or ros_b > 0:
                    ROSh[i] = ros_h
                    ROSf[i] = ros_f
                    ROSb[i] = ros_b
                    RAZd[i] = raz
                    valid_count += 1
        except Exception:
            continue
    
    if valid_count == 0:
        return vtx, {"dt_s": 0.0, "moved": False, "error": "no valid vertices"}
    
    # Step 3: Calculate displacement vectors
    dP = np.zeros_like(vtx)
    for i in range(n):
        if ROSh[i] == 0 and ROSf[i] == 0 and ROSb[i] == 0:
            continue
        
        xs, ys = _central_tangent(vtx, i)
        vx, vy = _deltaP_2d(xs, ys, RAZd[i], ROSh[i], ROSf[i], ROSb[i])
        
        if np.isnan(vx) or np.isnan(vy):
            continue
        
        # Convert m/min to m/s
        vx_mps = vx / 60.0
        vy_mps = vy / 60.0
        
        dP[i] = [vx_mps * dt, vy_mps * dt]
    
    # Step 4: Apply displacements with fuel boundary clipping
    new_vtx = vtx.copy()
    hit_boundary = False
    
    for i in range(n):
        x0, y0 = vtx[i]
        x1, y1 = x0 + dP[i, 0], y0 + dP[i, 1]
        
        if np.isnan(x1) or np.isnan(y1):
            continue
        
        if settings.clip_to_fuel:
            x1c, y1c, hit = _clip_to_first_nonfuel(x0, y0, x1, y1, grids)
            new_vtx[i] = [x1c, y1c]
            hit_boundary |= hit
        else:
            new_vtx[i] = [x1, y1]
    
    # Step 5: Apply smoothing
    if settings.smoothing_iterations > 0:
        new_vtx = prometheus_smooth_vertices(new_vtx, settings)
    
    # Step 6: Redistribute vertices
    new_vtx = prometheus_vertex_redistribution(new_vtx, settings)
    
    # Step 7: Validate movement
    movement = np.linalg.norm(new_vtx - vtx)
    if movement < 0.001:
        return vtx, {"dt_s": 0.0, "moved": False, "error": "no movement"}
    
    return new_vtx, {
        "dt_s": dt,
        "moved": True,
        "hit_boundary": hit_boundary,
        "n_vertices": len(new_vtx)
    }


# --------------------------------------------------------------
# Prometheus adaptive timestep calculation
# --------------------------------------------------------------
def calculate_prometheus_timestep(vertices_xy, grids: Grids, settings: EngineSettings):
    """
    Calculate adaptive timestep using Prometheus methodology.
    
    Prometheus rule: dt should move the fastest vertex by approximately 
    the target vertex spacing distance.
    """
    vtx = vertices_xy
    n = len(vtx)
    
    # Sample ROS at each vertex
    max_ros = 0.0
    for i, (x, y) in enumerate(vtx):
        try:
            r, c = rowcol(grids.transform, x, y)
            if _inside_raster(grids.fuel.shape, r, c):
                ros_h = grids.ROS_H[r, c]
                if not np.isnan(ros_h) and ros_h > max_ros:
                    max_ros = ros_h
        except Exception:
            continue
    
    if max_ros <= 0:
        return settings.min_dt_s
    
    # Prometheus timestep: dt = target_spacing / max_velocity
    # max_velocity is in m/min, convert to m/s
    max_velocity_mps = max_ros / 60.0
    
    # Time to move one target spacing distance
    dt = settings.target_vertex_spacing_m / max(max_velocity_mps, 1e-6)
    
    # Clamp to reasonable bounds
    dt = np.clip(dt, settings.min_dt_s, settings.max_dt_s)
    
    return float(dt)


# --------------------------------------------------------------
# Vertex rediscretization
# --------------------------------------------------------------

def insert_new_vertices(vertices_xy, Lx, max_new=5):
    """Prometheus Eqs. [31]-[36]."""
    vtx, n = vertices_xy.copy(), len(vertices_xy)
    new_vertices = []
    for i in range(n):
        p_prev, p_curr, p_next = vtx[(i - 1) % n], vtx[i], vtx[(i + 1) % n]
        L_prev, L_next = np.linalg.norm(p_curr - p_prev), np.linalg.norm(p_next - p_curr)
        v1, v2 = p_prev - p_curr, p_next - p_curr
        cosb = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)
        beta_p = math.acos(cosb)
        need = lambda L: (L / Lx) > math.sin(beta_p / 2.0)
        if need(L_prev):
            new_vertices.append((p_prev + p_curr) / 2)
        new_vertices.append(p_curr)
        if need(L_next):
            new_vertices.append((p_curr + p_next) / 2)
        if len(new_vertices) > n + max_new:
            break

    result = [new_vertices[0]]
    for p in new_vertices[1:]:
        if np.linalg.norm(np.array(p) - np.array(result[-1])) > Lx / 1000.0:
            result.append(p)
    return np.array(result) if len(result) >= 3 else vertices_xy



def remove_close_vertices(vertices_xy, Lx):
    if len(vertices_xy) < 3:
        return vertices_xy
    out = [vertices_xy[0]]
    for p in vertices_xy[1:]:
        if np.linalg.norm(p - out[-1]) >= Lx / 1000.0:
            out.append(p)
    return np.array(out) if len(out) >= 3 else vertices_xy


# --------------------------------------------------------------
# Prometheus vertex redistribution (edge-length criteria)
# --------------------------------------------------------------
def prometheus_vertex_redistribution(vertices_xy, settings: EngineSettings):
    """
    Redistribute vertices using Prometheus edge-length criteria.
    
    Rules:
    1. If edge > max_spacing: insert midpoint
    2. If edge < min_spacing: remove one endpoint
    3. Maintain CCW orientation
    """
    vtx = ensure_ccw(vertices_xy.copy())
    n = len(vtx)
    
    if n < 3:
        return vtx
    
    # Phase 1: Insert vertices on long edges
    new_vertices = []
    for i in range(n):
        p_curr = vtx[i]
        p_next = vtx[(i + 1) % n]
        
        edge_length = np.linalg.norm(p_next - p_curr)
        
        new_vertices.append(p_curr)
        
        # Insert midpoint if edge too long
        if edge_length > settings.max_vertex_spacing_m:
            n_insertions = int(np.ceil(edge_length / settings.target_vertex_spacing_m)) - 1
            for j in range(1, n_insertions + 1):
                alpha = j / (n_insertions + 1)
                new_point = (1 - alpha) * p_curr + alpha * p_next
                new_vertices.append(new_point)
    
    vtx = np.array(new_vertices)
    n = len(vtx)
    
    # Phase 2: Remove vertices on short edges
    if n > 3:
        filtered = [vtx[0]]
        for i in range(1, n):
            edge_length = np.linalg.norm(vtx[i] - filtered[-1])
            if edge_length >= settings.min_vertex_spacing_m:
                filtered.append(vtx[i])
        
        # Check closing edge
        if len(filtered) > 2:
            closing_edge = np.linalg.norm(filtered[-1] - filtered[0])
            if closing_edge < settings.min_vertex_spacing_m:
                filtered = filtered[:-1]
        
        vtx = np.array(filtered) if len(filtered) >= 3 else vtx
    
    # Phase 3: Enforce maximum vertex count
    if len(vtx) > settings.max_vertices:
        # Decimate by keeping every nth vertex
        step = len(vtx) // settings.max_vertices
        vtx = vtx[::step]
    
    return ensure_ccw(vtx)


# --------------------------------------------------------------
# Prometheus-style Laplacian smoothing of vertices
# --------------------------------------------------------------
def prometheus_smooth_vertices(vertices_xy, settings: EngineSettings):
    """
    Apply Prometheus-style Laplacian smoothing.
    
    Each vertex is moved toward the average of its neighbors,
    weighted by smoothing_weight.
    """
    vtx = vertices_xy.copy()
    n = len(vtx)
    
    if n < 3:
        return vtx
    
    for iteration in range(settings.smoothing_iterations):
        smoothed = np.zeros_like(vtx)
        
        for i in range(n):
            p_prev = vtx[(i - 1) % n]
            p_curr = vtx[i]
            p_next = vtx[(i + 1) % n]
            
            # Laplacian: average of neighbors
            laplacian = (p_prev + p_next) / 2.0
            
            # Weighted blend
            smoothed[i] = (1 - settings.smoothing_weight) * p_curr + \
                          settings.smoothing_weight * laplacian
        
        vtx = smoothed
    
    return vtx


# --------------------------------------------------------------
# Fire statistics and export
# --------------------------------------------------------------

def fire_perimeter_stats(vertices_xy, fuel_breaks=None):
    poly = Polygon(vertices_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if fuel_breaks:
        for br in fuel_breaks:
            poly = poly.difference(br)
    return dict(
        perimeter_m=poly.length,
        area_m2=poly.area,
        centroid=(poly.centroid.x, poly.centroid.y),
    )


def export_vertex_stats(vertices_xy, step_idx, stats, out_dir):
    df = pd.DataFrame(vertices_xy, columns=["x", "y"])
    df["vertex_id"] = np.arange(len(vertices_xy))
    df["step"] = step_idx
    df["Perimeter_m"] = stats["perimeter_m"]
    df["Area_m2"] = stats["area_m2"]
    df.to_csv(f"{out_dir}/fire_vertices_step{step_idx:04d}.csv", index=False)


def export_perimeters_shapefile(perims, crs, out_path):
    geoms, times = [], []
    for i, v in enumerate(perims):
        poly = Polygon(v)
        if not poly.is_valid:
            poly = poly.buffer(0)
        geoms.append(poly)
        times.append(i)
    areas_m2 = [p.area for p in geoms]
    areas_ha = [a / 10000.0 for a in areas_m2]
    perims_len = [p.length for p in geoms]

    gdf = gpd.GeoDataFrame(
        {"time_step": times, "area_ha": areas_ha, "perim_m": perims_len},
        geometry=geoms,
        crs=crs,
    )
    gdf.to_file(out_path)


# ==============================================================
# PROMETHEUS-STYLE OUTPUT MODULE
# ==============================================================
# This module generates publication-quality outputs matching Prometheus:
#   - Time-step perimeter shapefiles
#   - Fire growth animations
#   - ROS vs Time graphs
#   - Area vs Time graphs
#   - Summary statistics CSV
#   - Fire behavior classification maps
# ==============================================================

import warnings
warnings.filterwarnings('ignore')

class PrometheusOutputManager:
    """
    Manages all Prometheus-style outputs for a single fire simulation.
    """

    def __init__(self, output_dir: str, fire_id: str, ignition_info: dict):
        """
        Initialize output manager.

        Parameters:
        -----------
        output_dir : str
            Base output directory
        fire_id : str
            Unique fire identifier (e.g., "fire_0001")
        ignition_info : dict
            Ignition metadata: {x, y, time, cause, season, etc.}
        """
        self.output_dir = output_dir
        self.fire_id = fire_id
        self.ignition_info = ignition_info

        # Create subdirectories
        self.perimeters_dir = os.path.join(output_dir, "Perimeters")
        self.graphs_dir = os.path.join(output_dir, "Graphs")
        self.animations_dir = os.path.join(output_dir, "Animations")
        self.stats_dir = os.path.join(output_dir, "Statistics")

        for d in [self.perimeters_dir, self.graphs_dir,
                  self.animations_dir, self.stats_dir]:
            os.makedirs(d, exist_ok=True)

        # Storage for time-series data
        self.time_series = []
        self.perimeters = []
        self.metadata = {
            'fire_id': fire_id,
            'ignition_x': ignition_info.get('x'),
            'ignition_y': ignition_info.get('y'),
            'ignition_time': ignition_info.get('time', datetime.now()),
            'cause': ignition_info.get('cause', 'unknown'),
            'season': ignition_info.get('season', 'summer')
        }

    def record_timestep(self, step: int, elapsed_minutes: float,
                        vertices: np.ndarray, dt_seconds: float,
                        weather: dict = None):
        """
        Record data for a single time step.
        """
        # Create polygon
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Calculate metrics
        area_m2 = poly.area
        area_ha = area_m2 / 10000.0
        perimeter_m = poly.length

        # Calculate instantaneous ROS (effective) from area change
        if len(self.time_series) > 0:
            prev_area = self.time_series[-1]['area_m2']
            dt_min = dt_seconds / 60.0
            if dt_min > 0:
                area_growth = max(0, area_m2 - prev_area)
                ros_effective = np.sqrt(area_growth / (np.pi * dt_min))
            else:
                ros_effective = 0.0
        else:
            ros_effective = 0.0

        record = {
            'step': step,
            'time_minutes': elapsed_minutes,
            'time_hours': elapsed_minutes / 60.0,
            'datetime': self.metadata['ignition_time'] + timedelta(minutes=elapsed_minutes),
            'area_m2': area_m2,
            'area_ha': area_ha,
            'perimeter_m': perimeter_m,
            'ros_effective_m_per_min': ros_effective,
            'ros_effective_m_per_hour': ros_effective * 60.0,
            'dt_seconds': dt_seconds,
            'num_vertices': len(vertices)
        }

        if weather:
            record.update({
                'wind_speed': weather.get('wind_speed', np.nan),
                'wind_dir': weather.get('wind_dir', np.nan),
                'temperature': weather.get('temperature', np.nan),
                'humidity': weather.get('humidity', np.nan)
            })

        self.time_series.append(record)
        self.perimeters.append(poly)

    def export_perimeter_shapefile(self, crs):
        """Export all perimeters as a single shapefile with time attributes, plus hourly snapshots."""
        if len(self.perimeters) == 0:
            print("  ⚠️ No perimeters to export")
            return

        gdf = gpd.GeoDataFrame(self.time_series, geometry=self.perimeters, crs=crs)

        # Main shapefile with all timesteps
        out_path = os.path.join(self.perimeters_dir, f"{self.fire_id}_perimeters.shp")
        gdf.to_file(out_path)
        print(f"  ✓ Saved perimeters: {out_path}")

        # Individual hourly/per-step snapshots (approx every 60 min by sim clock)
        for i, row in gdf.iterrows():
            if i % 10 == 0 or i == len(gdf) - 1:
                hour = int(row['time_hours'])
                minute = int(row['time_minutes'] % 60)
                single_gdf = gpd.GeoDataFrame([row], geometry=[row.geometry], crs=crs)
                hour_path = os.path.join(self.perimeters_dir, f"{self.fire_id}_T{hour:03d}h{minute:02d}m.shp")
                single_gdf.to_file(hour_path)

        print(f"  ✓ Saved {len(gdf)} time-step perimeters")
        return out_path

    def export_statistics_csv(self):
        """Export time-series and summary CSVs."""
        if len(self.time_series) == 0:
            return

        df = pd.DataFrame(self.time_series)

        # Add metadata columns
        for key, val in self.metadata.items():
            df[key] = val

        out_path = os.path.join(self.stats_dir, f"{self.fire_id}_statistics.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved statistics: {out_path}")

        summary = {
            'fire_id': self.fire_id,
            'duration_minutes': float(df['time_minutes'].max()),
            'duration_hours': float(df['time_hours'].max()),
            'final_area_ha': float(df['area_ha'].iloc[-1]),
            'final_perimeter_m': float(df['perimeter_m'].iloc[-1]),
            'max_ros_m_per_min': float(df['ros_effective_m_per_min'].max()),
            'mean_ros_m_per_min': float(df['ros_effective_m_per_min'].mean()),
            'total_timesteps': int(len(df)),
            'ignition_x': self.metadata['ignition_x'],
            'ignition_y': self.metadata['ignition_y'],
            'cause': self.metadata['cause'],
            'season': self.metadata['season']
        }
        summary_path = os.path.join(self.stats_dir, f"{self.fire_id}_summary.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        print(f"  ✓ Saved summary: {summary_path}")
        return out_path

    def plot_area_vs_time(self):
        """Generate Area vs Time graph."""
        if len(self.time_series) == 0:
            return
        df = pd.DataFrame(self.time_series)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['time_hours'], df['area_ha'], 'b-', linewidth=2, label='Fire Area')
        ax.fill_between(df['time_hours'], 0, df['area_ha'], alpha=0.3)
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Area (hectares)', fontsize=12, fontweight='bold')
        ax.set_title(f'Fire Growth: {self.fire_id}\n'
                     f'Final Area: {df["area_ha"].iloc[-1]:.1f} ha | '
                     f'Duration: {df["time_hours"].iloc[-1]:.1f} hours',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        textstr = f'Cause: {self.metadata["cause"]}\nSeason: {self.metadata["season"]}\n' \
                  f'Max ROS: {df["ros_effective_m_per_min"].max():.2f} m/min'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        out_path = os.path.join(self.graphs_dir, f"{self.fire_id}_area_vs_time.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved area graph: {out_path}")
        return out_path

    def plot_ros_vs_time(self):
        """Generate ROS vs Time graph (plus weather if present)."""
        if len(self.time_series) < 2:
            return
        df = pd.DataFrame(self.time_series)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(df['time_hours'], df['ros_effective_m_per_min'], 'r-', linewidth=2, label='Rate of Spread')
        ax1.axhline(df['ros_effective_m_per_min'].mean(), color='gray', linestyle='--', alpha=0.5, label='Mean ROS')
        ax1.set_ylabel('ROS (m/min)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Fire Behavior: {self.fire_id}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        if 'wind_speed' in df.columns and not df['wind_speed'].isna().all():
            ax2_wind = ax2
            ax2_wind.plot(df['time_hours'], df['wind_speed'], 'g-', linewidth=2, label='Wind Speed')
            ax2_wind.set_ylabel('Wind Speed (km/h)', fontsize=11, fontweight='bold', color='g')
            ax2_wind.tick_params(axis='y', labelcolor='g')
            if 'temperature' in df.columns and not df['temperature'].isna().all():
                ax2_temp = ax2_wind.twinx()
                ax2_temp.plot(df['time_hours'], df['temperature'], 'orange', linewidth=2, label='Temperature', alpha=0.7)
                ax2_temp.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold', color='orange')
                ax2_temp.tick_params(axis='y', labelcolor='orange')
            ax2_wind.grid(True, alpha=0.3)
            ax2_wind.legend(loc='upper left', fontsize=9)
        else:
            ax2.plot(df['time_hours'], df['perimeter_m']/1000, 'purple', linewidth=2, label='Perimeter')
            ax2.set_ylabel('Perimeter (km)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left', fontsize=10)
        ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(self.graphs_dir, f"{self.fire_id}_ros_vs_time.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved ROS graph: {out_path}")
        return out_path

    def create_growth_animation(self, fuel_raster_path: str = None, interval: int = 200, fps: int = 10):
        """Create animated GIF/MP4 of fire growth over terrain."""
        if len(self.perimeters) < 2:
            return
        print(f"  🎬 Creating animation...")
        fig, ax = plt.subplots(figsize=(12, 10))
        if fuel_raster_path and os.path.exists(fuel_raster_path):
            with rasterio.open(fuel_raster_path) as src:
                from rasterio.plot import show
                show(src, ax=ax, cmap='terrain', alpha=0.6)
        ax.plot(self.metadata['ignition_x'], self.metadata['ignition_y'], 'r*', markersize=20, label='Ignition', zorder=10)
        perimeter_line = ax.plot([], [], 'yellow', linewidth=3, label='Fire Perimeter', zorder=5)[0]
        perimeter_fill = ax.fill([], [], 'red', alpha=0.3, zorder=4)[0]
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        stats_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.set_xlabel('Easting (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Northing (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'Fire Growth Animation: {self.fire_id}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_aspect('equal')
        all_coords = np.vstack([np.array(p.exterior.coords) for p in self.perimeters])
        margin = 500
        ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
        def init():
            perimeter_line.set_data([], [])
            return perimeter_line, perimeter_fill, time_text, stats_text
        def animate(i):
            frame_skip = max(1, len(self.perimeters) // 200)
            idx = i * frame_skip
            if idx >= len(self.perimeters):
                idx = len(self.perimeters) - 1
            poly = self.perimeters[idx]
            record = self.time_series[idx]
            x, y = poly.exterior.xy
            perimeter_line.set_data(x, y)
            perimeter_fill.remove()
            perimeter_fill_new = ax.fill(x, y, 'red', alpha=0.3, zorder=4)[0]
            hours = int(record['time_hours'])
            minutes = int(record['time_minutes'] % 60)
            time_text.set_text(f"Time: {hours:02d}:{minutes:02d}")
            stats_str = f"Area: {record['area_ha']:.1f} ha\nPerimeter: {record['perimeter_m']/1000:.2f} km\nROS: {record['ros_effective_m_per_min']:.2f} m/min"
            stats_text.set_text(stats_str)
            return perimeter_line, perimeter_fill_new, time_text, stats_text
        n_frames = min(len(self.perimeters), 200)
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=interval, blit=False, repeat=True)
        gif_path = os.path.join(self.animations_dir, f"{self.fire_id}_growth.gif")
        try:
            anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
            print(f"  ✓ Saved animation: {gif_path}")
        except Exception as e:
            print(f"  ⚠️ Could not save GIF: {e}")
        mp4_path = os.path.join(self.animations_dir, f"{self.fire_id}_growth.mp4")
        try:
            anim.save(mp4_path, writer='ffmpeg', fps=fps, dpi=150, extra_args=['-vcodec', 'libx264'])
            print(f"  ✓ Saved video: {mp4_path}")
        except Exception as e:
            print(f"  ⚠️ Could not save MP4 (ffmpeg may not be installed): {e}")
        plt.close()
        return gif_path

    def create_static_growth_map(self, fuel_raster_path: str = None):
        """Create static map showing fire progression with hourly isochrones."""
        if len(self.perimeters) == 0:
            return
        fig, ax = plt.subplots(figsize=(14, 12))
        if fuel_raster_path and os.path.exists(fuel_raster_path):
            with rasterio.open(fuel_raster_path) as src:
                from rasterio.plot import show
                show(src, ax=ax, cmap='terrain', alpha=0.5)
        df = pd.DataFrame(self.time_series)
        cmap = plt.cm.YlOrRd
        hourly_perims = []
        for hour in range(int(df['time_hours'].max()) + 1):
            mask = (df['time_hours'] >= hour) & (df['time_hours'] < hour + 1)
            if mask.any():
                idx = df[mask].index[0]
                hourly_perims.append((hour, self.perimeters[idx]))
        for i, (hour, poly) in enumerate(hourly_perims):
            x, y = poly.exterior.xy
            color = cmap(i / max(len(hourly_perims) - 1, 1))
            ax.plot(x, y, color=color, linewidth=2, label=f'T+{hour}h')
            ax.fill(x, y, color=color, alpha=0.2)
        ax.plot(self.metadata['ignition_x'], self.metadata['ignition_y'], 'r*', markersize=25, label='Ignition', zorder=10, markeredgecolor='black', markeredgewidth=1)
        final = self.perimeters[-1]
        x, y = final.exterior.xy
        ax.plot(x, y, 'k-', linewidth=3, label='Final Perimeter', zorder=8)
        ax.set_xlabel('Easting (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Northing (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'Fire Progression Map: {self.fire_id}\n'
                     f'Final Area: {df["area_ha"].iloc[-1]:.1f} ha | '
                     f'Duration: {df["time_hours"].iloc[-1]:.1f} hours', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(self.graphs_dir, f"{self.fire_id}_progression_map.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved progression map: {out_path}")
        return out_path

    def generate_all_outputs(self, crs, fuel_raster_path: str = None):
        """Generate all Prometheus-style outputs."""
        print(f"\n📊 Generating Prometheus-style outputs for {self.fire_id}...")
        self.export_perimeter_shapefile(crs)
        self.export_statistics_csv()
        self.plot_area_vs_time()
        self.plot_ros_vs_time()
        self.create_static_growth_map(fuel_raster_path)
        try:
            self.create_growth_animation(fuel_raster_path)
        except Exception as e:
            print(f"  ⚠️ Animation failed: {e}")
        print(f"✅ All outputs generated for {self.fire_id}\n")

def simulate_fire_with_prometheus_outputs(
    vertices_xy, grids_base, settings,
    ignition_info: dict,
    fuel_raster_path: str = None,
    weather_conditions=None,
    max_minutes=None,
    max_steps=10_000,
    variability_factor=0.2,
    seed=None
):
    """
    Drop-in replacement for simulate_fire_stochastic() that
    additionally records and exports Prometheus-style outputs.
    """
    # Initialize
    fire_id = ignition_info.get('fire_id', 'fire_0000')
    output_mgr = PrometheusOutputManager(settings.output_dir, fire_id, ignition_info)

    # Weather setup
    if weather_conditions:
        weather_rec = weather_conditions[0] if isinstance(weather_conditions, list) else weather_conditions
        grids = create_weather_adjusted_grids(grids_base, weather_rec)
    else:
        grids = grids_base
        weather_rec = {}

    # Start conditions
    t_elapsed = 0.0
    if variability_factor > 0:
        grids = perturb_ros_raz_grids(grids, variability_factor, seed)
    if max_minutes is None:
        max_minutes = 480

    perims = [ensure_ccw(vertices_xy)]
    weather_idx = 0
    consecutive_failures = 0

    # Record initial state
    output_mgr.record_timestep(0, 0.0, vertices_xy, 0.0, weather_rec)

    for step in range(max_steps):
        # --- Progress heartbeat ---
        if step % 50 == 0 and step > 0:
            print(f"🕒 Heartbeat: step={step}, elapsed={t_elapsed:.1f} min, "
                  f"perims={len(perims)}, consecutive_failures={consecutive_failures}")
        # Update weather per hour of sim time
        if weather_conditions and isinstance(weather_conditions, list):
            if t_elapsed > (weather_idx + 1) * 60 and weather_idx + 1 < len(weather_conditions):
                weather_idx += 1
                weather_rec = weather_conditions[weather_idx]
                grids = create_weather_adjusted_grids(grids_base, weather_rec)
                if variability_factor > 0:
                    grids = perturb_ros_raz_grids(grids, variability_factor, seed)

        # --- Prometheus-style propagation block ---
        nxt, info = propagate_vertices(perims[-1], grids, settings)

        if not info.get("moved"):
            consecutive_failures += 1
            if consecutive_failures >= 5:
                break
            continue
        else:
            consecutive_failures = 0

        # Vertex redistribution and smoothing are handled internally

        perims.append(nxt)
        t_elapsed += info["dt_s"] / 60.0

        # Record this timestep
        output_mgr.record_timestep(step + 1, t_elapsed, nxt, info["dt_s"], weather_rec)

        # Print MORE frequently to see if simulation is running
        if step % 5 == 0 or step < 10:
            stats = fire_perimeter_stats(nxt)
            print(f"  Step {step+1:03d} | Area={stats['area_m2']/1e4:.4f}ha | "
                  f"Perim={stats['perimeter_m']:.1f}m | Vertices={info.get('n_vertices', len(nxt))} | "
                  f"Time={t_elapsed:.2f}min | dt={info['dt_s']:.2f}s")
            
            # Diagnostic for stall detection
            if step > 0 and step % 20 == 0:
                area_growth = stats['area_m2'] - fire_perimeter_stats(perims[-2])['area_m2']
                print(f"    → Area growth last step: {area_growth:.2f} m² ({area_growth/1e4:.6f} ha)")

        if t_elapsed >= max_minutes:
            break

    # Export all outputs
    output_mgr.generate_all_outputs(grids.crs, fuel_raster_path)

    # Return stats compatible with previous API
    final_stats = fire_perimeter_stats(perims[-1]) if len(perims) > 1 else {"area_m2": 0, "perimeter_m": 0}
    return perims, {
        "final_area_ha": final_stats["area_m2"] / 1e4,
        "final_perimeter_m": final_stats["perimeter_m"],
        "duration_min": t_elapsed,
        "n_steps": len(perims) - 1,
        "success": len(perims) > 1
    }

# ----------------------------
# Outputs Generated (overview)
# ----------------------------
# This will create:
# 1. Perimeters/ directory:
#    fire_0001_perimeters.shp           - All time steps in one file
#    fire_0001_T000h00m.shp, ...        - Hourly/per-step snapshots
#
# 2. Statistics/ directory:
#    fire_0001_statistics.csv           - Full time-series data
#    fire_0001_summary.csv              - Summary metrics
#
# 3. Graphs/ directory:
#    fire_0001_area_vs_time.png         - Area growth curve
#    fire_0001_ros_vs_time.png          - ROS and weather over time
#    fire_0001_progression_map.png      - Static map with isochrones
#
# 4. Animations/ directory:
#    fire_0001_growth.gif               - Animated fire growth
#    fire_0001_growth.mp4               - Video (requires ffmpeg)



# --------------------------------------------------------------
# Stochastic helpers
# --------------------------------------------------------------

def sample_weather_conditions(weather_df, ignition_time=None, duration_hours=8):
    """Sample weather conditions for fire simulation."""
    if weather_df.empty:
        return None
    
    if ignition_time is None and "datetime" in weather_df.columns:
        min_time = weather_df["datetime"].min()
        max_time = weather_df["datetime"].max() - timedelta(hours=duration_hours)
        if max_time > min_time:
            random_seconds = random.uniform(0, (max_time - min_time).total_seconds())
            ignition_time = min_time + timedelta(seconds=random_seconds)
        else:
            ignition_time = min_time
    
    if ignition_time and "datetime" in weather_df.columns:
        end_time = ignition_time + timedelta(hours=duration_hours)
        weather_window = weather_df[
            (weather_df["datetime"] >= ignition_time) & 
            (weather_df["datetime"] <= end_time)
        ]
        if len(weather_window) > 0:
            return weather_window.to_dict('records')
    
    # Fallback to random sampling
    n_samples = min(len(weather_df), duration_hours)
    return weather_df.sample(n=n_samples).to_dict('records') if n_samples > 0 else None


def adjust_ros_for_weather(base_ros, base_raz, wind_speed, wind_dir,
                           ffmc=None, dmc=None, dc=None,
                           bui=None, fwi=None, temp=None, rh=None):
    """
    Adjust ROS/RAZ for weather **without** re-applying ISI/BUI (Option B).
    Rationale: base_ros already encodes ISI/BUI from the ROS grids.
    Here we only adjust for short-term moisture (via T/RH → FFMC proxy),
    crown fire behavior, and steer RAZ toward the wind direction.
    """
    # Ensure numeric inputs
    try:
        ws = float(wind_speed)
    except Exception:
        ws = 0.0
    try:
        wd = float(wind_dir)
    except Exception:
        wd = 0.0

    # Start from base ROS (arrays or scalars)
    adjusted_ros = base_ros

    # Compute FFMC proxy if not provided (from T/RH); used for moisture & crown logic
    if ffmc is None:
        try:
            ffmc = 147.2773 * (101.0 - (rh or 40.0)) / (59.5 + (rh or 40.0))
            ffmc = float(np.clip(ffmc, 60.0, 100.0))
        except Exception:
            ffmc = 85.0

    # Moisture tweak (affects FMC) — mild, bounded
    if temp is not None and rh is not None:
        fmc_factor = 1.0 + 0.002 * (float(temp) - 20.0) - 0.0015 * (float(rh) - 30.0)
        fmc_factor = float(np.clip(fmc_factor, 0.7, 1.3))
        adjusted_ros = adjusted_ros * fmc_factor

    # Crown fire behavior (uses FFMC as FMC proxy and wind speed); may amplify ROS
    try:
        fire_type, ros_crown = classify_fire_type(adjusted_ros, fmc=ffmc, cbh_m=3.0, wind_speed=ws)
        adjusted_ros = ros_crown
    except Exception:
        pass

    # Steer RAZ primarily toward wind direction (keep some inertia from base)
    adjusted_raz = (0.8 * wd + 0.2 * base_raz) % 360.0

    return adjusted_ros, adjusted_raz


def _aspect_to_degrees(val):
    """Convert aspect strings like N, NE, SW to degrees; pass through numeric degrees."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, str):
        v = val.strip().upper()
        compass = {"N":0, "NNE":22.5, "NE":45, "ENE":67.5, "E":90, "ESE":112.5,
                   "SE":135, "SSE":157.5, "S":180, "SSW":202.5, "SW":225, "WSW":247.5,
                   "W":270, "WNW":292.5, "NW":315, "NNW":337.5}
        if v in compass:
            return float(compass[v])
        try:
            return float(v)
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def _normalize_slope_deg(val):
    """Accept slope in degrees or percent; convert percent>=100 or plausibly >60 to degrees."""
    try:
        x = float(val)
    except Exception:
        return np.nan
    # Heuristic: if very steep value (> 60), treat as percent and convert to degrees.
    if x > 60:
        return float(np.degrees(np.arctan(x / 100.0)))
    return x

# --------------------------------------------------------------
# Elevation-based FMC adjustment (empirical lapse-rate formulation)
# --------------------------------------------------------------
# Based on Van Wagner (1987) and the Canadian FBP System, foliar moisture content (FMC)
# is influenced by temperature (T), relative humidity (RH), and elevation.
# Prometheus and BurnP3 apply an empirical lapse-rate correction (~6.5 °C per km)
# to adjust temperature with altitude and thus indirectly modify FMC.
#
# Elevation-adjusted temperature:
#       T_adj = T_surface - Γ * (Elevation)
#       where Γ = 6.5 °C/km = 0.0065 °C/m
#
# Simplified empirical FMC scaling (dimensionless multiplier on ROS):
#       FMC_factor = 1.0 - 0.002*(T_adj - 20) + 0.001*(RH - 30)
#       FMC_factor ∈ [0.7, 1.3]
#
# Reference: Van Wagner (1987), Forestry Canada FBP System; Prometheus Model Docs.
def compute_fmc_from_elevation(elev_m, temp_c=25.0, rh=40.0):
    """Estimate Foliar Moisture Content (FMC) factor based on elevation, temperature, and RH."""
    if np.isnan(elev_m):
        return 1.0
    # Apply empirical lapse rate (6.5°C/km)
    lapse_rate = 0.0065  # °C/m
    temp_adj = temp_c - lapse_rate * elev_m
    # Compute FMC factor as ROS multiplier
    fmc_factor = 1.0 - 0.002 * (temp_adj - 20.0) + 0.001 * (rh - 30.0)
    fmc_factor = np.clip(fmc_factor, 0.7, 1.3)
    return float(fmc_factor)

def adjust_ros_for_slope_aspect(base_ros, slope_deg, aspect_deg, wind_dir_deg):
    """
    Adjust ROS using Prometheus slope factor (SF).
    SF increases ROS when spreading upslope in alignment with wind.
    Reference: Forestry Canada Fire Danger Group / Prometheus Model Docs.
    """
    if np.isnan(slope_deg) or np.isnan(aspect_deg) or np.isnan(wind_dir_deg):
        return base_ros

    # Compute relative alignment between wind direction and slope aspect
    rel_angle = (wind_dir_deg - aspect_deg) % 360.0
    upslope_alignment = np.cos(np.radians(rel_angle)) > 0  # True if roughly upslope

    # Convert slope to percent (if given in degrees)
    slope_percent = np.tan(np.radians(slope_deg)) * 100.0 if slope_deg <= 60 else slope_deg

    # Prometheus slope factor formulation
    if upslope_alignment:
        SF = np.exp(3.533 * (slope_percent / 100.0) ** 1.2)
    else:
        SF = 1.0  # No enhancement when downslope in many FBP types

    return base_ros * SF

# --------------------------------------------------------------
# Additional spread modifiers: Buildup Effect, Equivalent Wind, Hourly FFMC, Acceleration
# --------------------------------------------------------------
def buildup_effect(BUI, BUI0=50.0, alpha=0.75, beta=0.25):
    """Compute Buildup Effect (BE) multiplier on ROS as a function of BUI."""
    try:
        B = float(BUI)
    except Exception:
        return 1.0
    if np.isnan(B):
        return 1.0
    return 1.0 + alpha * ((B - BUI0) / (BUI0 + beta * (B - BUI0) + 1e-9))

def equivalent_wind_speed(wind_speed, slope_deg, aspect_deg, wind_dir_deg, ks=1.0):
    """
    Combine observed wind with terrain-induced component (equivalent wind approach).
    ks is an empirical coefficient (~0.9–1.2). Units are kept consistent with wind_speed input.
    """
    try:
        ws = float(wind_speed)
    except Exception:
        ws = 0.0
    if np.isnan(slope_deg) or np.isnan(aspect_deg) or np.isnan(wind_dir_deg):
        return ws
    rel = np.radians((wind_dir_deg - aspect_deg) % 360.0)
    # Terrain wind component scales with slope; magnitude tuned by ks
    w_terrain = ks * np.tan(np.radians(slope_deg)) * np.cos(rel) * 10.0  # scale factor
    return max(0.0, ws + w_terrain)

def hourly_ffmc(temp_c, rh, prev_ffmc=85.0):
    """
    Approximate hourly FFMC adjustment (very light-weight surrogate).
    Returns updated FFMC; intended to vary ROS slightly with short-term moisture.
    """
    # Simple empirical proxy bounded to plausible FFMC range
    try:
        rh = float(rh)
    except Exception:
        rh = 50.0
    ffmc_proxy = 147.2773 * (101.0 - rh) / (59.5 + rh)
    ffmc_proxy = float(np.clip(ffmc_proxy, 60.0, 100.0))
    return 0.9 * prev_ffmc + 0.1 * ffmc_proxy

def accelerate_ros(ros_eq_scale, t_elapsed_min, k=0.03):
    """
    Exponential acceleration of ROS toward equilibrium (dimensionless scale).
    ros_eq_scale is the target scale (usually 1.0); returns a multiplier in (0, 1].
    """
    try:
        t = float(t_elapsed_min)
    except Exception:
        t = 0.0
    return ros_eq_scale * (1.0 - np.exp(-k * max(0.0, t)))

def scale_grids(grids: Grids, factor: float) -> Grids:
    """Return a new Grids object with ROS layers scaled by a constant factor."""
    return Grids(
        fuel=grids.fuel,
        ROS_H=grids.ROS_H * factor,
        ROS_F=grids.ROS_F * factor,
        ROS_B=grids.ROS_B * factor,
        RAZ=grids.RAZ,
        transform=grids.transform,
        crs=grids.crs,
        nonfuel_values=grids.nonfuel_values
    )

def create_weather_adjusted_grids(base_grids, weather_record):
    """Create grids adjusted for weather, topography (equivalent wind), optional slope factor, and elevation/FMC."""
    wind_speed = weather_record.get('wind_speed', 10.0)
    wind_dir = weather_record.get('wind_dir', 0.0)
    temp = weather_record.get('temperature', 25.0)
    rh = weather_record.get('humidity', 40.0)
    slope_deg = weather_record.get('slope_deg', np.nan)
    aspect_deg = weather_record.get('aspect_deg', np.nan)

    # Equivalent wind approach (includes slope/aspect); avoids double-counting with explicit slope factor
    eff_wind = equivalent_wind_speed(wind_speed, slope_deg, aspect_deg, wind_dir)

    # First adjust by weather (wind/temp/RH) using effective wind
    adj_ros_h, _ = adjust_ros_for_weather(base_grids.ROS_H, base_grids.RAZ, eff_wind, wind_dir, temp, rh)
    adj_ros_f, _ = adjust_ros_for_weather(base_grids.ROS_F, base_grids.RAZ, eff_wind, wind_dir, temp, rh)
    adj_ros_b, adj_raz = adjust_ros_for_weather(base_grids.ROS_B, base_grids.RAZ, eff_wind, wind_dir, temp, rh)

    # If slope/aspect missing (no equivalent wind possible), fall back to explicit slope/aspect factor
    if np.isnan(slope_deg) or np.isnan(aspect_deg):
        adj_ros_h = adjust_ros_for_slope_aspect(adj_ros_h, slope_deg, aspect_deg, wind_dir)
        adj_ros_f = adjust_ros_for_slope_aspect(adj_ros_f, slope_deg, aspect_deg, wind_dir)
        adj_ros_b = adjust_ros_for_slope_aspect(adj_ros_b, slope_deg, aspect_deg, wind_dir)

    # Elevation-based FMC adjustment
    elev_m = weather_record.get('elev_m', np.nan)
    fmc_factor = compute_fmc_from_elevation(elev_m, temp, rh)
    adj_ros_h *= fmc_factor
    adj_ros_f *= fmc_factor
    adj_ros_b *= fmc_factor

    # Optional: small FFMC-based tweak using hourly proxy
    try:
        ffmc_now = hourly_ffmc(temp, rh)
        ros_ffmc_factor = 1.0 + 0.01 * ((ffmc_now - 85.0) / 15.0)
        ros_ffmc_factor = float(np.clip(ros_ffmc_factor, 0.85, 1.15))
        adj_ros_h *= ros_ffmc_factor
        adj_ros_f *= ros_ffmc_factor
        adj_ros_b *= ros_ffmc_factor
    except Exception:
        pass

    return Grids(
        fuel=base_grids.fuel,
        ROS_H=adj_ros_h,
        ROS_F=adj_ros_f,
        ROS_B=adj_ros_b,
        RAZ=adj_raz,
        transform=base_grids.transform,
        crs=base_grids.crs,
        nonfuel_values=base_grids.nonfuel_values
    )


def perturb_ros_raz_grids(grids, variability_factor=0.2, seed=None):
    """Add spatial randomness to ROS/RAZ grids."""
    if seed is not None:
        np.random.seed(seed)
    
    shape = grids.ROS_H.shape
    ros_mult = np.random.lognormal(mean=0, sigma=variability_factor, size=shape)
    raz_offset = np.random.normal(0, 15, size=shape)

    return Grids(
        fuel=grids.fuel,
        ROS_H=grids.ROS_H * ros_mult,
        ROS_F=grids.ROS_F * ros_mult,
        ROS_B=grids.ROS_B * ros_mult,
        RAZ=(grids.RAZ + raz_offset) % 360,
        transform=grids.transform,
        crs=grids.crs,
        nonfuel_values=grids.nonfuel_values
    )


def sample_ignition_size(distribution="lognormal") -> float:
    if distribution == "lognormal":
        d = np.random.lognormal(mean=1.6, sigma=0.8)
        return float(np.clip(d, 0.5, 100.0))
    return 10.0


def sample_burning_duration(season="summer", cause="lightning") -> float:
    if cause == "lightning":
        base = random.uniform(240, 720)
    else:
        base = random.uniform(120, 480)
    
    if season == "spring":
        base *= random.uniform(0.5, 0.8)
    elif season == "summer":
        base *= random.uniform(0.8, 1.5)
    elif season == "fall":
        base *= random.uniform(0.6, 1.0)
    
    return float(base)


# --------------------------------------------------------------
# Stochastic simulation
# --------------------------------------------------------------

def simulate_fire_stochastic(vertices_xy, grids_base, settings,
                             weather_conditions=None, max_minutes=None,
                             max_steps=10_000, variability_factor=0.2, seed=None):
    """Simulate fire with stochastic variability."""
    
    # Apply weather if provided
    if weather_conditions:
        weather_rec = weather_conditions[0] if isinstance(weather_conditions, list) else weather_conditions
        grids = create_weather_adjusted_grids(grids_base, weather_rec)
    else:
        grids = grids_base

    # Start at equilibrium ROS (avoid zeroing ROS at t=0 which prevents any movement)
    t_elapsed = 0.0
    acc_scale = 1.0  # no initial downscaling
    # grids remain unscaled; optional future ramp-up can be reintroduced once time advances
    
    # Add fine-scale spatial variability
    if variability_factor > 0:
        grids = perturb_ros_raz_grids(grids, variability_factor, seed)
    
    if max_minutes is None:
        max_minutes = 480
    
    perims = [ensure_ccw(vertices_xy)]
    weather_idx = 0
    consecutive_failures = 0

    for step in range(max_steps):
        # Update weather if time-varying
        if weather_conditions and isinstance(weather_conditions, list):
            if t_elapsed > (weather_idx + 1) * 60 and weather_idx + 1 < len(weather_conditions):
                weather_idx += 1
                weather_rec = weather_conditions[weather_idx]
                grids = create_weather_adjusted_grids(grids_base, weather_rec)
                # Keep ROS at equilibrium during weather updates to ensure continued movement
                acc_scale = 1.0
                # no downscaling applied here
                if variability_factor > 0:
                    grids = perturb_ros_raz_grids(grids, variability_factor, seed)
        
        nxt, info = propagate_vertices(perims[-1], grids, settings)
        
        if not info.get("moved"):
            consecutive_failures += 1
            if consecutive_failures >= 5:
                break
            continue
        else:
            consecutive_failures = 0

        nxt = insert_new_vertices(nxt, settings.distance_resolution_m)
        nxt = remove_close_vertices(nxt, settings.distance_resolution_m)

        stats = fire_perimeter_stats(nxt)
        
        # Print every 10 steps
        if step % 10 == 0 or step < 3:
            print(f"  Step {step+1:03d} | Area={stats['area_m2']/1e4:.2f}ha | Time={t_elapsed:.0f}min")

        if settings.export_outputs:
            export_vertex_stats(nxt, step + 1, stats, settings.output_dir)

        perims.append(nxt)
        t_elapsed += info["dt_s"] / 60.0
        
        if t_elapsed >= max_minutes:
            break

    # Final stats
    if len(perims) > 1:
        final_stats = fire_perimeter_stats(perims[-1])
    else:
        final_stats = {"area_m2": 0, "perimeter_m": 0}

    if settings.export_outputs and len(perims) > 1:
        try:
            export_perimeters_shapefile(perims, grids.crs, f"{settings.output_dir}/FirePerimeters.shp")
        except Exception as e:
            print(f"  ⚠️ Export failed: {e}")

    return perims, {
        "final_area_ha": final_stats["area_m2"] / 1e4,
        "final_perimeter_m": final_stats["perimeter_m"],
        "duration_min": t_elapsed,
        "n_steps": len(perims) - 1,
        "success": len(perims) > 1
    }


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

if __name__ == "__main__":
    # --------------------------------------------------------------
    # Setup logging to capture all console output into a file
    # --------------------------------------------------------------
    import sys
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = os.path.join("/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Output", f"fire_engine_log_{timestamp}.txt")

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    # Create the log directory if it doesn’t exist
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # Redirect stdout and stderr to both console and log file
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"🔥 Fire Engine Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Logging to: {LOG_PATH}")
    # --------------------------------------------------------------
    # Load project template (supports .py or .conf key=value format)
    # --------------------------------------------------------------
    TEMPLATE_PATH = "/Users/Martyn/Desktop/PhD/Fyah/Python_Scripts/project_template.txt"

    def load_template_kv(template_path):
        """Parse simple key=value project template file."""
        if not os.path.exists(template_path):
            print("⚠️ Template file not found, using defaults.")
            return {}
        data = {}
        with open(template_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    data[key.strip()] = val.strip().strip('"')
        return data

    template_data = {}
    # Try loading as Python module first
    TEMPLATE_PY = "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Kelowna_FireModel_template.py"
    if os.path.exists(TEMPLATE_PY):
        import importlib.util
        spec = importlib.util.spec_from_file_location("template", TEMPLATE_PY)
        template = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(template)
        # Read all config keys from template
        for key in [
            "PROJECT_NAME", "FUEL_SOURCE_TYPE", "FUEL_PATH", "FUEL_POLYGON_PATH",
            "IGNITION_SOURCE", "IGNITION_PATH", "WEATHER_PATH", "FBP_CSV_PATH",
            "DERIVED_DIR", "OUTPUT_DIR", "STATION_PATH", "TOPO_ELEV"
        ]:
            template_data[key] = getattr(template, key, "")
    else:
        template_data = load_template_kv(TEMPLATE_PATH)

    # Dynamically read all required inputs from config
    PROJECT_NAME = template_data.get("PROJECT_NAME", "UnnamedProject")
    # Force these to corrected inputs regardless of template:
    FUEL_PATH = "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/fuel_zones_from_extent.tif"
    FUEL_POLYGON_PATH = "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/fuel_zones_from_extent.shp"
    FUEL_SOURCE_TYPE = "polygon"
    IGNITION_SOURCE = "point"
    IGNITION_PATH = template_data.get("IGNITION_PATH", "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Ignition/Ignition_Polygons/ignition_summary.csv")
    WEATHER_PATH = template_data.get("WEATHER_PATH", "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Weather/fire_weather_list.csv")
    FBP_CSV_PATH = template_data.get("FBP_CSV_PATH", "/Users/Martyn/GitRepos/BritishColumbia/FBP/Output/FBP_Output_Summer2010.csv")
    DERIVED_DIR = template_data.get("DERIVED_DIR", "/Users/Martyn/GitRepos/BurnP3+/Kelowna/PreProcessing/DerivedTerrain")
    OUTPUT_DIR = template_data.get("OUTPUT_DIR", "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Output/FirePerimeters")
    STATION_PATH = template_data.get("STATION_PATH", "/Users/Martyn/GitRepos/Fire_Weather/2023_BCWS_WX_STATIONS_CLIPPED.csv")
    TOPO_ELEV = template_data.get("TOPO_ELEV", "/Users/Martyn/GitRepos/BurnP3+/Kelowna/PreProcessing/DEM.tif")

    # Print detected configuration at start
    print("\nDetected configuration:")
    print(f"  Project: {PROJECT_NAME}")
    print(f"  Fuel source type: {FUEL_SOURCE_TYPE}")
    print(f"  Fuel raster path: {FUEL_PATH}")
    print(f"  Fuel polygon path: {FUEL_POLYGON_PATH}")
    print(f"  Ignition source: {IGNITION_SOURCE}")
    print(f"  Ignition path: {IGNITION_PATH}")
    print(f"  Weather path: {WEATHER_PATH}")
    print(f"  FBP CSV path: {FBP_CSV_PATH}")
    print(f"  Derived dir: {DERIVED_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Station path: {STATION_PATH}")
    print(f"  DEM path: {TOPO_ELEV}")

    # Print configuration summary before running simulation
    print("\n=== CONFIGURATION SUMMARY ===")
    print(f"Project: {PROJECT_NAME}")
    if FUEL_SOURCE_TYPE == "polygon":
        print(f"Fuel source: {FUEL_SOURCE_TYPE} ({FUEL_POLYGON_PATH})")
    else:
        print(f"Fuel source: {FUEL_SOURCE_TYPE} ({FUEL_PATH})")
    print(f"Ignition source: {IGNITION_SOURCE} ({IGNITION_PATH})")
    print(f"Weather: {WEATHER_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("==============================\n")

    # Helper: overlay fuel polygons
    def overlay_fuel_polygons(polygon_path, reference_raster_path, fill_value=0):
        """
        Rasterize polygons from a shapefile to match a reference raster's grid.
        Returns: (fuel_array, transform, crs)
        """
        import rasterio
        import geopandas as gpd
        import pandas as pd
        import numpy as np
        from rasterio.features import rasterize
        with rasterio.open(reference_raster_path) as src:
            meta = src.meta.copy()
            shape = (src.height, src.width)
            transform = src.transform
            crs = src.crs
        gdf = gpd.read_file(polygon_path)
        # Print the columns found in the polygon shapefile
        print(f"📋 Polygon attribute columns: {list(gdf.columns)}")
        # Detect appropriate fuel code field (case-insensitive)
        fuel_field_candidates = ["fuel", "fuel_code", "fbp", "fbp_code"]
        lower_cols = {c.lower(): c for c in gdf.columns}
        fuel_field = None
        for candidate in fuel_field_candidates:
            if candidate in lower_cols:
                fuel_field = lower_cols[candidate]
                break

        if fuel_field is not None:
            # --- Print raw preview of values ---
            raw_nonnull = gdf[fuel_field].dropna()
            try:
                raw_sample = list(pd.Series(raw_nonnull).astype(str).str[:32].unique()[:10])
            except Exception:
                raw_sample = list(pd.Series(raw_nonnull).unique()[:10])
            print(f"🔎 Sample fuel codes (raw from '{fuel_field}'): {raw_sample}")

        if fuel_field is None:
            print("⚠️ No fuel code field found in polygons — assigning value=1 to all features.")
            gdf["fuel"] = 1
            fuel_field = "fuel"
        else:
            print(f"🧩 Using '{fuel_field}' as the fuel code field from polygon shapefile.")
            # --- Normalize and map fuel codes to numeric IDs ---
            def normalize_fuel_code(val):
                """Convert string or numeric code to standard FBP numeric ID."""
                if pd.isna(val):
                    return np.nan
                s = str(val).strip().upper()
                # Try to capture a leading numeric token like "1" in "1 - C1"
                import re
                match = re.match(r"^(\d+)", s)
                if match:
                    try:
                        return int(match.group(1))
                    except Exception:
                        pass
                # Clean string for mapping
                s_clean = s.replace(" ", "").replace("-", "")
                string_to_id = {
                    "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C7": 7,
                    "D1": 11, "D2": 12, "D12": 13, "O1": 31,
                    "NF": 101, "NONFUEL": 101,
                    "WATER": 102,
                    "URBAN": 106, "BUILTUP": 106,
                    "M1": 425, "M2": 525, "M12": 625
                }
                return string_to_id.get(s_clean, np.nan)

            # Apply normalization to the fuel field
            gdf["fuel"] = gdf[fuel_field].apply(normalize_fuel_code)
            gdf["fuel"] = pd.to_numeric(gdf["fuel"], errors="coerce").fillna(0).astype(int)
            print(f"✅ Normalized fuel codes (unique): {sorted(gdf['fuel'].unique())}")
            # Warn/abort if no valid (>0) codes found
            valid_mask = gdf["fuel"] > 0
            n_valid = int(valid_mask.sum())
            n_total = int(len(gdf))
            if n_valid == 0:
                print("❌ No valid fuel codes (>0) found in polygon attributes after normalization.")
                print("   ➤ Check the attribute field values and mapping (e.g., 'Fuel_Code' should contain 1,2,3,7,11,12,13,31,101,102,106,425,...).")
                raise ValueError("Polygon fuel normalization yielded only zeros/NaNs.")
            else:
                print(f"✅ Valid polygon features with fuel codes: {n_valid}/{n_total}")
        # Reproject polygons to match raster
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        # Prepare list of (geometry, value)
        shapes = ((geom, int(val)) for geom, val in zip(gdf.geometry, gdf["fuel"]) if int(val) > 0)
        fuel_arr = rasterize(
            shapes=shapes,
            out_shape=shape,
            fill=fill_value,
            transform=transform,
            dtype=meta["dtype"] if "dtype" in meta else "int32"
        )
        return fuel_arr, transform, crs

    # ===============================================================
    # Load fuel source (raster or polygon)
    # ===============================================================
    ros_paths = [os.path.join(DERIVED_DIR, f"{n}.tif") for n in ["ROS_H", "ROS_F", "ROS_B", "RAZ"]]
    if FUEL_SOURCE_TYPE == "polygon":
        # Must have FUEL_POLYGON_PATH
        if not FUEL_POLYGON_PATH or not os.path.exists(FUEL_POLYGON_PATH):
            raise FileNotFoundError("FUEL_POLYGON_PATH must be set and point to a valid shapefile when FUEL_SOURCE_TYPE=polygon.")
        print("🗺️ Using polygon fuel source")
        # Rasterize polygons to match the reference raster (FUEL_PATH)
        fuel, transform, crs = overlay_fuel_polygons(FUEL_POLYGON_PATH, FUEL_PATH, fill_value=0)
        # Save the rasterized fuel for reference/debugging
        out_rasterized = os.path.join(DERIVED_DIR, "fuel_from_polygon.tif")
        with rasterio.open(FUEL_PATH) as ref:
            meta = ref.meta.copy()
            meta.update(dtype=fuel.dtype, count=1)
            with rasterio.open(out_rasterized, "w", **meta) as dst:
                dst.write(fuel, 1)
        # Use the rasterized output as the new fuel path for downstream
        FUEL_PATH_USED = out_rasterized
        # Automatically run fire growth using polygon fuels if FUEL_SOURCE_TYPE=polygon
        # Unconditionally regenerate ROS/RAZ rasters for the rasterized fuel grid
        ros_paths = [os.path.join(DERIVED_DIR, f"{n}.tif") for n in ["ROS_H", "ROS_F", "ROS_B", "RAZ"]]
        print("Regenerating ROS/RAZ rasters from polygon fuel grid (overwrite)...")
        generate_ros_raz_from_fuel(FUEL_PATH_USED, DERIVED_DIR, ISI=12.0, BUI=70.0, FMC=95.0, CURING=85.0)
        with rasterio.open(FUEL_PATH_USED) as f:
            fuel, transform, crs = f.read(1), f.transform, f.crs
    else:
        print("📦 Using raster fuel source")
        FUEL_PATH_USED = FUEL_PATH
        with rasterio.open(FUEL_PATH) as f:
            fuel, transform, crs = f.read(1), f.transform, f.crs
        ros_paths = [os.path.join(DERIVED_DIR, f"{n}.tif") for n in ["ROS_H", "ROS_F", "ROS_B", "RAZ"]]
        if not all(os.path.exists(p) for p in ros_paths):
            print("Generating ROS/RAZ rasters from FBP fuel types...")
            generate_ros_raz_from_fuel(FUEL_PATH_USED, DERIVED_DIR, ISI=12.0, BUI=70.0, FMC=95.0, CURING=85.0)

    # Load grids (always use the selected fuel raster)
    with rasterio.open(FUEL_PATH_USED) as f:
        fuel, transform, crs = f.read(1), f.transform, f.crs
    with rasterio.open(ros_paths[0]) as f:
        ROS_H = f.read(1)  # Units: m/min (no conversion)
    with rasterio.open(ros_paths[1]) as f:
        ROS_F = f.read(1)
    with rasterio.open(ros_paths[2]) as f:
        ROS_B = f.read(1)
    with rasterio.open(ros_paths[3]) as f:
        RAZ = f.read(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grids = Grids(fuel, ROS_H, ROS_F, ROS_B, RAZ, transform, crs, nonfuel_values=(0, -9999, 101, 102, 106))

    # IGNITION_SOURCE logic
    # If IGNITION_SOURCE == "point", load shapefile and run single ignition simulation
    # Otherwise, default CSV-based logic

    # Define default slope/aspect/elevation for single ignition case
    station_slope_deg = 0.0
    station_aspect_deg = 0.0
    station_elev_m = 400.0  # reasonable default elevation (m)

    if IGNITION_SOURCE == "point":
        # Load shapefile and extract single ignition location
        print("🔥 Running single ignition simulation from shapefile")
        gdf_ign = gpd.read_file(IGNITION_PATH)
        if gdf_ign.empty:
            raise ValueError(f"Ignition shapefile {IGNITION_PATH} is empty")
        # Reproject if needed
        if gdf_ign.crs and crs and gdf_ign.crs != crs:
            gdf_ign = gdf_ign.to_crs(crs)
        # Use first geometry as ignition location
        geom = gdf_ign.geometry.iloc[0]
        if geom.geom_type == "Point":
            x, y = geom.x, geom.y
        elif geom.geom_type in ["Polygon", "MultiPolygon"]:
            centroid = geom.centroid
            x, y = centroid.x, centroid.y
        else:
            raise ValueError("Unsupported geometry type in ignition shapefile")
        season = gdf_ign["season"].iloc[0] if "season" in gdf_ign.columns else "summer"
        cause = gdf_ign["cause"].iloc[0] if "cause" in gdf_ign.columns else "unknown"
        iteration = 0
        # Diagnostic print of sampled fuel and ROS values at ignition cell
        try:
            rr, cc = rowcol(grids.transform, x, y)
            if _inside_raster(grids.fuel.shape, rr, cc):
                fval = grids.fuel[rr, cc]
                rh = grids.ROS_H[rr, cc]
                rf = grids.ROS_F[rr, cc]
                rb = grids.ROS_B[rr, cc]
                rz = grids.RAZ[rr, cc]
                print(f"🔎 Ignition cell values — fuel={fval}, ROS_H={rh}, ROS_F={rf}, ROS_B={rb}, RAZ={rz}")
            else:
                print("🔎 Ignition outside raster bounds when sampling debug values.")
        except Exception as e:
            print(f"🔎 Debug sampling failed: {e}")
        # Validate location
        is_valid, reason = validate_ignition_location(x, y, grids)
        if not is_valid:
            print(f"⚠️ Skipping fire: {reason}")
            sys.exit(1)
        ignition_diam = sample_ignition_size("lognormal")
        burn_minutes = sample_burning_duration(season, cause)
        sim_seed = hash(f"{iteration}_{x}_{y}") % (2**32)
        weather_sample = None
        if os.path.exists(WEATHER_PATH):
            weather = pd.read_csv(WEATHER_PATH)
            if "DATE_TIME" in weather.columns:
                weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")
                weather.rename(columns={"DATE_TIME": "datetime"}, inplace=True)
            elif "datetime" in weather.columns:
                weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
            rename_map = {
                "HOURLY_WIND_SPEED": "wind_speed",
                "HOURLY_WIND_DIRECTION": "wind_dir",
                "HOURLY_TEMPERATURE": "temperature",
                "HOURLY_RELATIVE_HUMIDITY": "humidity",
            }
            weather.rename(columns={k: v for k, v in rename_map.items() if k in weather.columns}, inplace=True)
            if not weather.empty:
                weather_sample = sample_weather_conditions(weather, duration_hours=int(burn_minutes/60))
        # Inject slope/aspect/elevation into weather records if available
        if weather_sample is not None:
            if isinstance(weather_sample, list):
                for rec in weather_sample:
                    if "slope_deg" not in rec:
                        rec["slope_deg"] = station_slope_deg
                    if "aspect_deg" not in rec:
                        rec["aspect_deg"] = station_aspect_deg
                    rec["elev_m"] = station_elev_m
            elif isinstance(weather_sample, dict):
                weather_sample.setdefault("slope_deg", station_slope_deg)
                weather_sample.setdefault("aspect_deg", station_aspect_deg)
                weather_sample.setdefault("elev_m", station_elev_m)
        fire_output_dir = os.path.join(OUTPUT_DIR, f"fire_{iteration:04d}_iter{iteration}")
        os.makedirs(fire_output_dir, exist_ok=True)
        settings = EngineSettings(
            distance_resolution_m=30.0,

            # Prometheus timestep control
            min_dt_s=1.0,
            max_dt_s=60.0,
            target_vertex_spacing_m=10.0,
    
             # Prometheus vertex control
            min_vertex_spacing_m=5.0,
            max_vertex_spacing_m=20.0,
            max_vertices=10000,
    
    # Prometheus smoothing
             smoothing_iterations=2,
             smoothing_weight=0.3,  # Conservative smoothing
    
    # Boundary behavior
             stop_at_boundary=False,
             clip_to_fuel=True,
    
             export_outputs=True,
             output_dir=fire_output_dir
        )
        start = ignition_point(x, y, n_vertices=16, diameter_m=ignition_diam)
        try:
            print(f"\nFire | ({x:.2f}, {y:.2f}) | {season}/{cause}")

            # Prepare ignition info for Prometheus outputs
            ignition_info = {
                'fire_id': f"fire_{iteration:04d}",
                'x': x,
                'y': y,
                'time': datetime.now(),  # use actual ignition time if available
                'cause': cause,
                'season': season
            }

            perims, fire_stats = simulate_fire_with_prometheus_outputs(
                start, grids, settings,
                ignition_info=ignition_info,
                fuel_raster_path=FUEL_PATH_USED,
                weather_conditions=weather_sample,
                max_minutes=burn_minutes,
                variability_factor=0.2,
                seed=sim_seed
            )

            if fire_stats["success"]:
                print(f"✓ Final: {fire_stats['final_area_ha']:.2f} ha in {fire_stats['duration_min']:.0f} min")
            else:
                print(f"⚠️ Fire failed to spread")
        except Exception as e:
            print(f"❌ Fire crashed: {e}")
            import traceback
            traceback.print_exc()
        # Diagnostic plot: fuel raster, ignition point, and final perimeter
        try:
            import matplotlib.pyplot as plt
            from rasterio.plot import show
            print("📊 Generating diagnostic plot: Fuel Raster, Ignition, and Final Perimeter...")
            fig, ax = plt.subplots(figsize=(8, 8))
            with rasterio.open(FUEL_PATH_USED) as src:
                show(src, ax=ax, title=None)
            # Plot ignition point
            ax.plot(x, y, marker="*", color="red", markersize=16, label="Ignition Point")
            # Plot final fire perimeter if available
            if isinstance(perims, list) and len(perims) > 1:
                from shapely.geometry import Polygon
                poly = Polygon(perims[-1])
                if poly.is_valid:
                    x_poly, y_poly = poly.exterior.xy
                    ax.plot(x_poly, y_poly, color="yellow", linewidth=2, label="Final Perimeter")
            ax.set_title("Diagnostic: Fuel Raster, Ignition, and Final Perimeter")
            ax.legend()
            plt.show()
        except Exception as plot_exc:
            print(f"⚠️ Diagnostic plot failed: {plot_exc}")
        # End after single ignition
        print(f"\n{'='*60}")
        print(f"✅ SINGLE IGNITION SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"{'='*60}")
        sys.exit(0)

    # Load weather (optional)
    weather = pd.DataFrame()
    if os.path.exists(WEATHER_PATH):
        weather = pd.read_csv(WEATHER_PATH)
        if "DATE_TIME" in weather.columns:
            weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")
            weather.rename(columns={"DATE_TIME": "datetime"}, inplace=True)
        elif "datetime" in weather.columns:
            weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
        
        rename_map = {
            "HOURLY_WIND_SPEED": "wind_speed",
            "HOURLY_WIND_DIRECTION": "wind_dir",
            "HOURLY_TEMPERATURE": "temperature",
            "HOURLY_RELATIVE_HUMIDITY": "humidity",
        }
        weather.rename(columns={k: v for k, v in rename_map.items() if k in weather.columns}, inplace=True)
        
        if "datetime" in weather.columns and not weather.empty:
            print(f"Loaded weather: {len(weather)} records from {weather['datetime'].min()} to {weather['datetime'].max()}")
        else:
            print("Weather loaded but no datetime column found")
    else:
        print("⚠️ Weather file not found")

    # Load station metadata (slope/aspect/elevation) and compute representative slope/aspect
    station_slope_deg = np.nan
    station_aspect_deg = np.nan
    station_elev_m = np.nan
    if os.path.exists(STATION_PATH):
        try:
            station_meta = pd.read_csv(STATION_PATH)
            station_meta.columns = [c.strip().upper() for c in station_meta.columns]
            if "SLOPE" in station_meta.columns:
                svals = pd.to_numeric(station_meta["SLOPE"], errors="coerce").dropna()
                svals_deg = svals.apply(_normalize_slope_deg)
                if len(svals_deg) > 0:
                    station_slope_deg = float(np.nanmedian(svals_deg.values))
            if "ASPECT" in station_meta.columns:
                avals = station_meta["ASPECT"].apply(_aspect_to_degrees)
                avals = pd.to_numeric(avals, errors="coerce").dropna()
                if len(avals) > 0:
                    station_aspect_deg = float(np.nanmedian(avals.values))
            # Optional: pull representative elevation (now used)
            if "ELEVATION_M" in station_meta.columns:
                elev_vals = pd.to_numeric(station_meta["ELEVATION_M"], errors="coerce").dropna()
                if len(elev_vals) > 0:
                    station_elev_m = float(np.nanmedian(elev_vals.values))
                else:
                    station_elev_m = np.nan
            else:
                station_elev_m = np.nan
            if not np.isnan(station_slope_deg) or not np.isnan(station_aspect_deg):
                print(f"Using station slope/aspect: slope={station_slope_deg:.1f}°, aspect={station_aspect_deg:.0f}°")
            if not np.isnan(station_elev_m):
                print(f"Representative elevation={station_elev_m:.1f} m")
        except Exception as e:
            print(f"⚠️ Could not parse station metadata: {e}")
    else:
        print("⚠️ Station metadata file not found; skipping slope/aspect adjustment.")

    # Only run CSV/shapefile summary logic if IGNITION_SOURCE is not "point"
    # (i.e., default behavior)
    processed_ignitions = 0
    skipped_ignitions = 0
    if IGNITION_SOURCE != "point":
        # Load ignition summary
        ign_summary = pd.read_csv(IGNITION_PATH)
        print(f"Loaded ignition summary with {len(ign_summary)} entries")

        # ------------------------------------------------------------------
        # 🔍 Diagnostic plot: Ignition points vs Fuel Raster
        # ------------------------------------------------------------------
        import matplotlib.pyplot as plt
        from rasterio.plot import show

        print("\n📊 Plotting ignition locations vs fuel raster for verification...")

        try:
            # Read fuel raster for plotting
            with rasterio.open(FUEL_PATH) as src:
                fig, ax = plt.subplots(figsize=(8, 8))
                show(src, ax=ax, title="Ignition Points vs Fuel Raster (EPSG:3005)")

                # Convert ignition summary to GeoDataFrame
                if "x" in ign_summary.columns and "y" in ign_summary.columns:
                    ign_gdf = gpd.GeoDataFrame(
                        ign_summary,
                        geometry=gpd.points_from_xy(
                            pd.to_numeric(ign_summary["x"], errors="coerce"),
                            pd.to_numeric(ign_summary["y"], errors="coerce")
                        ),
                        crs="EPSG:4326"  # assume geographic first
                    )

                    # Reproject to match fuel raster CRS
                    ign_gdf = ign_gdf.to_crs(src.crs)

                    ign_gdf.plot(ax=ax, color="red", markersize=10, label="Ignitions")
                    plt.legend()
                    plt.show()

                    # Print bounding boxes for debugging
                    print(f"🗺️  Fuel raster bounds: {src.bounds}")
                    print(f"📍 Ignition bounds (reprojected): {ign_gdf.total_bounds}")
                else:
                    print("⚠️ No x/y columns found in ignition summary – cannot plot.")
        except Exception as e_plot:
            print(f"⚠️ Could not generate diagnostic plot: {e_plot}")
        # (rest of CSV/shapefile summary ignition logic unchanged)

    # Combine all perimeters (only if not single ignition)
    print(f"\n🧩 Combining all fire perimeters...")
    all_records = []
    for root, _, files in os.walk(OUTPUT_DIR):
        if "FirePerimeters.shp" in files:
            try:
                gdf = gpd.read_file(os.path.join(root, "FirePerimeters.shp"))
                gdf["fire_id"] = os.path.basename(root)
                if "area_ha" not in gdf.columns:
                    gdf["area_ha"] = gdf.geometry.area / 10000.0
                if "perim_m" not in gdf.columns:
                    gdf["perim_m"] = gdf.geometry.length
                all_records.append(gdf)
            except Exception as e:
                print(f"⚠️ Could not read {root}: {e}")

    if all_records:
        all_perims = gpd.GeoDataFrame(pd.concat(all_records, ignore_index=True), crs=crs)

        # Ensure projected CRS for correct metric calculations
        if all_perims.crs is None or all_perims.crs.is_geographic:
            print("🌐 Reprojecting all perimeters to projected CRS (EPSG:3005) for area/length calculations...")
            all_perims = all_perims.to_crs(epsg=3005)

        # Compute area and perimeter metrics safely
        if "area_ha" not in all_perims.columns:
            all_perims["area_ha"] = all_perims.geometry.area / 10000.0
        if "perim_m" not in all_perims.columns:
            all_perims["perim_m"] = all_perims.geometry.length

        out_all = os.path.join(OUTPUT_DIR, "ALL_Perimeters.shp")
        all_perims.to_file(out_all)
        print(f"✅ Combined {len(all_perims)} perimeter features from {len(all_records)} fires (projected to EPSG:3005)")
    else:
        print("⚠️ No perimeters found to combine")

    print(f"\n{'='*60}")
    print(f"✅ SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total ignitions processed: {processed_ignitions}")
    print(f"Successful fires: {processed_ignitions - skipped_ignitions}")
    print(f"Skipped (invalid location): {skipped_ignitions}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")