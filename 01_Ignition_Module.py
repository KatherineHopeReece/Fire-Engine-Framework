# ==============================================================
# Ignitions Module — Burn-P3 Style Implementation
# ==============================================================
# Reads project paths and settings from project_template.txt.
# Implements ignition grid adjustment (Eq. [2–4]), ignition rules,
# and probabilistic selection of escaped fire ignition locations.
# ==============================================================

import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.features import rasterize
from shapely.geometry import Point
import geopandas as gpd
import random
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. Read Project Template
# --------------------------------------------------------------

def load_project_template(template_path):
    """Reads key=value pairs from project_template.txt into a dictionary."""
    config = {}
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^(\w+)\s*=\s*(.+)$", line)
            if match:
                key, value = match.groups()
                config[key.strip()] = value.strip()
    return config


# Path to your template file
TEMPLATE_FILE = "/Users/Martyn/Desktop/PhD/Fyah/Python_Scripts/project_template.txt"
CONFIG = load_project_template(TEMPLATE_FILE)

# Find all ignition grid keys (including backward compatibility for IGNITION_GRID)
IGNITION_GRID_KEYS = [k for k in CONFIG if k.startswith("IGNITION_GRID")]
if not IGNITION_GRID_KEYS and "IGNITION_GRID" in CONFIG:
    IGNITION_GRID_KEYS = ["IGNITION_GRID"]

FUEL_GRID = CONFIG.get("FUEL_GRID")
ECOREGION_GRID = CONFIG.get("ECOREGION_GRID")

print("Loaded project paths from template:")
for key in IGNITION_GRID_KEYS:
    print(f"{key} = {CONFIG.get(key)}")
print(f"FUEL_GRID      = {FUEL_GRID}")
print(f"ECOREGION_GRID = {ECOREGION_GRID}")


# --------------------------------------------------------------
# 2. Escaped Fire Rate Tables (Example values)
# --------------------------------------------------------------

ESCAPED_FIRE_RATES = {
    "1": {"Lightning_Spring": 0.5, "Lightning_Summer": 0.9, "Human_Spring": 6.4, "Human_Summer": 0.5},
    "2": {"Lightning_Spring": 1.8, "Lightning_Summer": 5.1, "Human_Spring": 6.5, "Human_Summer": 0.0},
    "3": {"Lightning_Spring": 13.4, "Lightning_Summer": 23.0, "Human_Spring": 14.8, "Human_Summer": 3.2},
    "4": {"Lightning_Spring": 6.0, "Lightning_Summer": 15.2, "Human_Spring": 0.0, "Human_Summer": 2.8}
}

FIRE_OCCURRENCE_COUNTS = {
    "1": 0.12,
    "2": 0.24,
    "3": 0.43,
    "4": 0.21
}


# --------------------------------------------------------------
# 3. Utility Functions
# --------------------------------------------------------------

def load_raster(path):
    """Load raster and return array, transform, and CRS."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
    return arr, transform, crs


# Helper: load fuel layer, can be raster or rasterized polygons
def load_fuel_layer(config, reference_grid_path):
    """
    Loads the fuel grid, which may be a raster or polygon shapefile.
    If FUEL_SOURCE_TYPE == 'polygon', rasterizes the polygons to match reference grid.
    Otherwise, loads raster using load_raster.
    """
    fuel_source_type = config.get("FUEL_SOURCE_TYPE", "").lower()
    if fuel_source_type == "polygon":
        # Polygon shapefile path
        fuel_poly_path = config.get("FUEL_POLYGON_PATH")
        if not fuel_poly_path or not os.path.exists(fuel_poly_path):
            raise FileNotFoundError(f"FUEL_POLYGON_PATH not found: {fuel_poly_path}")
        # Get shape and transform from reference grid
        with rasterio.open(reference_grid_path) as ref:
            out_shape = (ref.height, ref.width)
            out_transform = ref.transform
            out_crs = ref.crs
        gdf = gpd.read_file(fuel_poly_path)
        # Use the first available integer column as the value, or fallback to 1
        value_col = None
        for col in gdf.columns:
            if pd.api.types.is_integer_dtype(gdf[col]):
                value_col = col
                break
        if value_col is None:
            values = np.ones(len(gdf), dtype=int)
        else:
            values = gdf[value_col].astype(int)
        shapes = zip(gdf.geometry, values)
        arr = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=out_transform,
            fill=0,
            dtype="int32"
        )
        return arr, out_transform, out_crs
    else:
        # Default: load raster
        fuel_grid_path = config.get("FUEL_GRID")
        if not fuel_grid_path or not os.path.exists(fuel_grid_path):
            raise FileNotFoundError(f"FUEL_GRID not found: {fuel_grid_path}")
        return load_raster(fuel_grid_path)


def adjust_ignition_grid(initial_grid, ecoregion_grid, escaped_rates, occurrence_rates, cause, season):
    """Eq. [4]: AI_ij = I_ij * (E_j / F_j)"""
    adjusted = np.zeros_like(initial_grid, dtype=float)
    unique_ecos = np.unique(ecoregion_grid[~np.isnan(ecoregion_grid)])
    
    for eco_id in unique_ecos:
        eco_mask = ecoregion_grid == eco_id
        eco_name = str(int(eco_id))
        key = f"{cause}_{season}"

        Ej = escaped_rates.get(eco_name, {}).get(key, 1.0)
        Fj = occurrence_rates.get(eco_name, 1.0)
        scale = Ej / Fj if Fj != 0 else 1.0
        adjusted[eco_mask] = initial_grid[eco_mask] * scale

    adjusted[adjusted < 0] = 0
    return adjusted


def get_fuel_codes(fuel_type_names):
    """Simplified lookup for FBP fuel types."""
    fuel_lut = {
        "D-1": 1, "O-1a": 2, "O-1b": 3,
        "C-1": 4, "C-2": 5, "C-3": 6, "C-6": 7
    }
    return [fuel_lut.get(f) for f in fuel_type_names if f in fuel_lut]


def apply_ignition_rules(adjusted_grid, fuel_grid, cause, season):
    """Zeroes cells that violate ignition rules."""
    rules = [
        {"fuel_types": ["D-1"], "cause": "Lightning", "season": None},
        {"fuel_types": ["O-1a", "O-1b"], "cause": "Lightning", "season": None},
        {"fuel_types": ["O-1b"], "cause": "Human", "season": "Summer"},
        {"fuel_types": ["O-1a"], "cause": "Human", "season": "Spring"},
        {"fuel_types": ["O-1a"], "cause": "Human", "season": "Summer"},
    ]

    filtered = adjusted_grid.copy()
    for rule in rules:
        if rule["cause"] == cause and (rule["season"] is None or rule["season"] == season):
            for code in get_fuel_codes(rule["fuel_types"]):
                if code is not None:
                    filtered[fuel_grid == code] = 0
    return filtered


def sample_ignitions(grid, transform, n_ignitions=10):
    """Weighted sampling of ignition coordinates from grid."""
    flat = grid.flatten()
    flat[flat < 0] = 0
    total = np.nansum(flat)
    if total == 0:
        raise ValueError("All ignition probabilities are zero after filtering.")
    probs = flat / total

    indices = np.random.choice(len(flat), size=n_ignitions, replace=False, p=probs)
    rows, cols = np.unravel_index(indices, grid.shape)
    coords = [xy(transform, r, c, offset="center") for r, c in zip(rows, cols)]
    return pd.DataFrame(coords, columns=["x", "y"])


def draw_number_of_escaped_fires(distribution):
    """Draws number of escaped fires from empirical distribution."""
    values, weights = zip(*distribution.items())
    return random.choices(values, weights=weights, k=1)[0]

def compute_escaped_fire_distribution(historical_fire_path, output_dir):
    """
    Loads historical fire shapefile, computes number of fires per year,
    plots and saves histogram, returns normalized distribution dict.
    """
    if not os.path.exists(historical_fire_path):
        raise FileNotFoundError(f"Historical fire data not found: {historical_fire_path}")

    gdf = gpd.read_file(historical_fire_path)
    if "YEAR" not in gdf.columns:
        raise KeyError("Historical fire shapefile must contain 'YEAR' field.")

    fire_counts = gdf.groupby("YEAR").size()
    # Sort by year ascending
    fire_counts = fire_counts.sort_index()

    years_since_start = range(1, len(fire_counts) + 1)

    plt.figure(figsize=(8,6))
    plt.bar(years_since_start, fire_counts.values, color='orange', edgecolor='black')
    plt.xlabel("Number of Years")
    plt.ylabel("Number of Escaped Fires")
    plt.title("Number of Escaped Fires per Year")
    plt.xticks(years_since_start, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "escaped_fire_distribution.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    # Return normalized distribution dict
    distribution = (fire_counts / fire_counts.sum()).to_dict()
    return distribution


# --------------------------------------------------------------
# 4. Main Ignition Workflow
# --------------------------------------------------------------


def generate_ignitions(ignition_grid_path, fuel_grid_path, ecoregion_grid_path, cause="Lightning", season="Summer", n_iterations=5, output_dir="./Ignitions", freq_dist=None):
    init_grid, transform, crs = load_raster(ignition_grid_path)
    fuel_grid, _, _ = load_fuel_layer(CONFIG, ignition_grid_path)
    if ecoregion_grid_path and os.path.exists(ecoregion_grid_path):
        eco_grid, _, _ = load_raster(ecoregion_grid_path)
        adjusted = adjust_ignition_grid(init_grid, eco_grid, ESCAPED_FIRE_RATES, FIRE_OCCURRENCE_COUNTS, cause, season)
    else:
        print("ℹ️ No ecoregion grid found — skipping adjustment (using base ignition probabilities).")
        adjusted = init_grid.copy()
    filtered = apply_ignition_rules(adjusted, fuel_grid, cause, season)

    # Normalize probabilities
    filtered[filtered < 0] = 0
    filtered /= np.nansum(filtered)

    if freq_dist is None:
        freq_dist = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2}  # Default

    os.makedirs(output_dir, exist_ok=True)
    all_points = []
    summary_records = []

    n_years = len(freq_dist)

    for i in range(n_iterations):
        n_fires = draw_number_of_escaped_fires(freq_dist)
        df = sample_ignitions(filtered, transform, n_fires)
        df["iteration"] = i + 1
        df["cause"] = cause
        df["season"] = season
        all_points.append(df)
        # Record summary info for this iteration
        shapefile_path = os.path.join(output_dir, f"ignitions_{cause}_{season}.shp")
        summary_records.append({
            "iteration": i + 1,
            "cause": cause,
            "season": season,
            "n_escaped_fires": n_fires,
            "n_years": n_years,
            "shapefile_path": shapefile_path
        })

    ign = pd.concat(all_points)
    gdf = gpd.GeoDataFrame(ign, geometry=gpd.points_from_xy(ign.x, ign.y), crs=crs)
    out_path = os.path.join(output_dir, f"ignitions_{cause}_{season}.shp")
    gdf.to_file(out_path)
    print(f"✅ Generated {len(gdf)} ignition points. Saved to {out_path}")

    # Create summary DataFrame and append ignition points coordinates
    summary_df = pd.DataFrame(summary_records)
    # Add ignition points coordinates with iteration info to summary CSV
    points_df = ign[["iteration", "x", "y", "cause", "season"]].copy()
    # Save summary and points to CSV
    summary_csv_path = os.path.join(output_dir, "ignition_summary.csv")
    # Write summary and points to the same CSV with a blank line in between
    with open(summary_csv_path, "w") as f:
        summary_df.to_csv(f, index=False)
        f.write("\n")
        points_df.to_csv(f, index=False)
    print(f"✅ Saved ignition summary and points to {summary_csv_path}")

    # Plot ignition points
    plt.figure(figsize=(8,6))
    plt.scatter(gdf.geometry.x, gdf.geometry.y, s=10, c='red', alpha=0.6)
    plt.title(f"Ignition Points - Cause: {cause}, Season: {season}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"ignition_points_{cause}_{season}.png")
    plt.savefig(plot_path)
    plt.close()

    return gdf


# --------------------------------------------------------------
# 4b. Batch Ignition Workflow for Multiple Grids
# --------------------------------------------------------------

def batch_generate_ignitions(config, ignition_grid_keys=None, fuel_grid=None, ecoregion_grid=None, n_iterations=5):
    """
    Loops through all ignition grids in the config and runs the ignition workflow for each.
    Tries to extract cause and season from key or filename if possible.
    """
    if ignition_grid_keys is None:
        ignition_grid_keys = [k for k in config if k.startswith("IGNITION_GRID")]
        if not ignition_grid_keys and "IGNITION_GRID" in config:
            ignition_grid_keys = ["IGNITION_GRID"]
    if fuel_grid is None:
        fuel_grid = config.get("FUEL_GRID")
    if ecoregion_grid is None:
        ecoregion_grid = config.get("ECOREGION_GRID")
    if not ecoregion_grid or not os.path.exists(ecoregion_grid):
        print("⚠️ ECOREGION_GRID missing or invalid — continuing without it.")
        ecoregion_grid = None

    base_dir = config.get("BASE_DIR", "")
    project_name = config.get("PROJECT_NAME", "")
    output_dir = os.path.join(base_dir, project_name, "Ignition", "Ignition_Polygons")
    os.makedirs(output_dir, exist_ok=True)

    # Check for single-point ignition mode
    ignition_source = config.get("IGNITION_SOURCE", "").lower()
    if ignition_source == "point":
        ignition_point_path = config.get("IGNITION_PATH")
        if ignition_point_path and os.path.exists(ignition_point_path):
            print(f"ℹ️ Using ignition point shapefile: {ignition_point_path}")
            gdf = gpd.read_file(ignition_point_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:3005")
            # Copy to output directory under set name
            out_point_path = os.path.join(output_dir, "ignition_single_point.shp")
            gdf.to_file(out_point_path)
            print(f"✅ Ignition point shapefile copied to {out_point_path}")
            return [("IGNITION_POINT", gdf)]
        else:
            print("⚠️ IGNITION_PATH not found or does not exist, cannot use point-based ignition.")
            # Fall through to raster ignition generation

    freq_dist = None
    historical_fire_path = config.get("HISTORICAL_FIRE_DATA")
    if historical_fire_path and os.path.exists(historical_fire_path):
        try:
            freq_dist = compute_escaped_fire_distribution(historical_fire_path, output_dir)
            print(f"Loaded empirical frequency distribution from {historical_fire_path}")
        except Exception as e:
            print(f"⚠️ Failed to compute empirical distribution: {e}")
            freq_dist = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2}
    else:
        freq_dist = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2}

    summary = []
    for key in ignition_grid_keys:
        ignition_grid_path = config[key]
        # Try to extract cause and season from key or filename
        cause = "Lightning"
        season = "Summer"
        # Try to parse cause and season from key name
        match = re.match(r"IGNITION_GRID[_\-]?([A-Za-z]+)?[_\-]?([A-Za-z]+)?", key)
        if match:
            groups = [g for g in match.groups() if g]
            if len(groups) == 2:
                cause, season = groups
            elif len(groups) == 1:
                cause = groups[0]
        else:
            # Try filename parsing
            fname = os.path.basename(ignition_grid_path)
            found = re.findall(r"(Lightning|Human|Spring|Summer)", fname, re.IGNORECASE)
            found = [f.title() for f in found]
            if "Lightning" in found or "Human" in found:
                cause = [f for f in found if f in ("Lightning", "Human")][0]
            if "Spring" in found or "Summer" in found:
                season = [f for f in found if f in ("Spring", "Summer")][0]
        print(f"\nProcessing ignition grid: {ignition_grid_path} (cause={cause}, season={season})")
        gdf = generate_ignitions(
            ignition_grid_path=ignition_grid_path,
            fuel_grid_path=fuel_grid,
            ecoregion_grid_path=ecoregion_grid,
            cause=cause,
            season=season,
            n_iterations=n_iterations,
            output_dir=output_dir,
            freq_dist=freq_dist
        )
        summary.append((key, gdf))
    return summary


# --------------------------------------------------------------
# 5. Example Run
# --------------------------------------------------------------

if __name__ == "__main__":
    # Batch process all ignition grids found in the template
    batch_results = batch_generate_ignitions(
        config=CONFIG,
        ignition_grid_keys=IGNITION_GRID_KEYS,
        fuel_grid=FUEL_GRID,
        ecoregion_grid=ECOREGION_GRID,
        n_iterations=10,
    )
    # Print the head of each result
    for key, gdf in batch_results:
        print(f"\nFirst few ignition points for {key}:")
        print(gdf.head())