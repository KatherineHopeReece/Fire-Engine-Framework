
"""
IGNACIO: Burned Area Difference Analysis
========================================

Compare Euler marker and level-set fire growth.

Experiment:
    Flat DEM
    Constant C-2 fuel
    Constant weather
    Grid resolution: 2 m
    Initial ignition radius: 10 m
    Duration: 60 minutes

Calculations:
    - Burned area from polygon geometries
    - Absolute burned-area difference (ha)
    - Standard percentage difference (%)
    - Signed difference relative to Euler (%)
    - Percentage difference at every saved timestep

Outputs:
    1. area_difference_analysis.csv
    2. burned_area_comparison.png
    3. area_percentage_difference.png
    4. absolute_area_difference.png
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. INPUT AND OUTPUT PATHS
# ============================================================

BASE = Path(
    "/Users/Martyn/GitRepos/Fire-Engine-Framework-dev/"
    "ignacio/LaughTests"
)

EULER_FILE = (
    BASE
    / "Constant_Euler_Output"
    / "all_perimeters_history.gpkg"
)

LEVELSET_FILE = (
    BASE
    / "Constant_Levelset_Output"
    / "all_perimeters_history.gpkg"
)

OUTPUT_DIR = BASE / "Constant_Comparison_2m"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_CRS = "EPSG:32611"

GRID_RESOLUTION = 2.0
INITIAL_RADIUS = 10.0
SIMULATION_DURATION = 60.0


# ============================================================
# 2. PLOT SETTINGS
# ============================================================

EULER_COLOR = "tab:blue"
LEVELSET_COLOR = "tab:orange"
DIFFERENCE_COLOR = "tab:purple"

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 3. LOAD AND CALCULATE BURNED AREAS
# ============================================================

def load_area_history(filepath, method_name):
    """
    Read a GeoPackage containing fire perimeter history.

    Burned areas are calculated directly from polygon
    geometries rather than from area_ha attributes.

    Returns:
        DataFrame containing:
            t_min
            method_name (burned area in hectares)
    """

    print("\n" + "=" * 60)
    print(f"LOADING {method_name.upper()}")
    print("=" * 60)

    if not filepath.exists():
        raise FileNotFoundError(
            f"GeoPackage not found:\n{filepath}"
        )

    # Read GeoPackage
    gdf = gpd.read_file(filepath)

    if gdf.empty:
        raise ValueError(
            f"{method_name}: GeoPackage is empty."
        )

    print(f"File: {filepath}")
    print(f"Records: {len(gdf)}")
    print(f"Columns: {list(gdf.columns)}")

    # --------------------------------------------------------
    # Check required fields
    # --------------------------------------------------------

    if "t_min" not in gdf.columns:
        raise ValueError(
            f"{method_name}: Missing 't_min' column."
        )

    if gdf.crs is None:
        raise ValueError(
            f"{method_name}: GeoPackage has no CRS."
        )

    # --------------------------------------------------------
    # Ensure projected metric coordinates
    # --------------------------------------------------------

    gdf = gdf.to_crs(TARGET_CRS)

    print(f"Working CRS: {gdf.crs}")

    # --------------------------------------------------------
    # Check geometry
    # --------------------------------------------------------

    if gdf.geometry.isna().any():
        raise ValueError(
            f"{method_name}: Missing polygon geometries."
        )

    if gdf.geometry.is_empty.any():
        raise ValueError(
            f"{method_name}: Empty polygon geometries."
        )

    valid_types = ["Polygon", "MultiPolygon"]

    if not gdf.geom_type.isin(valid_types).all():
        raise ValueError(
            f"{method_name}: Expected polygon geometries. "
            f"Found {gdf.geom_type.unique().tolist()}"
        )

    if not gdf.geometry.is_valid.all():
        raise ValueError(
            f"{method_name}: Invalid polygon geometries "
            "detected. Inspect the GeoPackage."
        )

    # --------------------------------------------------------
    # Extract simulation times
    # --------------------------------------------------------

    times = pd.to_numeric(
        gdf["t_min"],
        errors="coerce"
    )

    if times.isna().any():
        raise ValueError(
            f"{method_name}: Missing or invalid times."
        )

    # --------------------------------------------------------
    # Calculate area from polygon geometry
    # --------------------------------------------------------

    # EPSG:32611 has metre coordinates.
    # Polygon area is therefore in square metres.

    area_m2 = gdf.geometry.area.to_numpy()

    area_ha = area_m2 / 10000.0

    # --------------------------------------------------------
    # Build dataframe
    # --------------------------------------------------------

    df = pd.DataFrame({
        "t_min": times.to_numpy(dtype=float),
        method_name: area_ha
    })

    # Round time for numerical comparison
    df["t_min"] = df["t_min"].round(6)

    # --------------------------------------------------------
    # Validate values
    # --------------------------------------------------------

    if df.isna().any().any():
        raise ValueError(
            f"{method_name}: Missing calculated values."
        )

    if not np.isfinite(
        df[method_name].to_numpy()
    ).all():
        raise ValueError(
            f"{method_name}: Non-finite areas detected."
        )

    if (df[method_name] <= 0).any():
        raise ValueError(
            f"{method_name}: Zero or negative polygon area."
        )

    if df["t_min"].duplicated().any():
        raise ValueError(
            f"{method_name}: Duplicate simulation times. "
            "Check for multiple fires in the GeoPackage."
        )

    # Sort by time
    df = df.sort_values(
        "t_min"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(f"\nValidated records: {len(df)}")
    print(f"Initial time: {df['t_min'].iloc[0]:.1f} min")
    print(f"Final time: {df['t_min'].iloc[-1]:.1f} min")

    print(
        f"Initial burned area: "
        f"{df[method_name].iloc[0]:.6f} ha"
    )

    print(
        f"Final burned area: "
        f"{df[method_name].iloc[-1]:.6f} ha"
    )

    return df


# ============================================================
# 4. LOAD BOTH SIMULATION RESULTS
# ============================================================

euler = load_area_history(
    EULER_FILE,
    "Euler"
)

levelset = load_area_history(
    LEVELSET_FILE,
    "Level_Set"
)


# ============================================================
# 5. MATCH SIMULATION TIMES
# ============================================================

comparison = pd.merge(
    euler,
    levelset,
    on="t_min",
    how="inner",
    validate="one_to_one"
)

comparison = comparison.sort_values(
    "t_min"
).reset_index(drop=True)

if comparison.empty:
    raise ValueError(
        "No matching simulation times were found."
    )

print("\n" + "=" * 60)
print("MATCHED SIMULATION TIMES")
print("=" * 60)

print(f"Matching records: {len(comparison)}")
print(f"First time: {comparison['t_min'].iloc[0]:.1f} min")
print(f"Last time: {comparison['t_min'].iloc[-1]:.1f} min")

if len(comparison) != len(euler):
    print(
        "WARNING: Some Euler simulation times "
        "were not matched."
    )

if len(comparison) != len(levelset):
    print(
        "WARNING: Some level-set simulation times "
        "were not matched."
    )


# ============================================================
# 6. CALCULATE AREA DIFFERENCES
# ============================================================

# Signed difference in hectares:
#
# Positive = Euler has larger burned area
# Negative = Level set has larger burned area

comparison["difference_ha"] = (
    comparison["Euler"]
    - comparison["Level_Set"]
)

# Absolute difference in hectares

comparison["absolute_difference_ha"] = (
    comparison["difference_ha"].abs()
)

# ------------------------------------------------------------
# Standard percentage difference
# ------------------------------------------------------------

#             |Euler - LevelSet|
# Difference = ------------------ x 100
#             (Euler + LevelSet)/2

comparison["mean_area_ha"] = (
    comparison["Euler"]
    + comparison["Level_Set"]
) / 2.0

comparison["percentage_difference"] = np.where(
    comparison["mean_area_ha"] > 0,
    (
        comparison["absolute_difference_ha"]
        / comparison["mean_area_ha"]
    ) * 100.0,
    np.nan
)

# ------------------------------------------------------------
# Relative difference using Euler as reference
# ------------------------------------------------------------

#                   Euler - LevelSet
# Relative change = ---------------- x 100
#                        Euler

comparison["relative_difference_euler_pct"] = np.where(
    comparison["Euler"] > 0,
    (
        comparison["difference_ha"]
        / comparison["Euler"]
    ) * 100.0,
    np.nan
)

# ------------------------------------------------------------
# Convert areas to square metres
# ------------------------------------------------------------

comparison["Euler_m2"] = (
    comparison["Euler"] * 10000.0
)

comparison["Level_Set_m2"] = (
    comparison["Level_Set"] * 10000.0
)


# ============================================================
# 7. PRINT RESULTS AT EVERY TIMESTEP
# ============================================================

print("\n" + "=" * 85)
print("BURNED-AREA DIFFERENCE AT EVERY TIMESTEP")
print("=" * 85)

display_columns = [
    "t_min",
    "Euler",
    "Level_Set",
    "difference_ha",
    "percentage_difference",
    "relative_difference_euler_pct"
]

print(
    comparison[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. FINAL RESULTS
# ============================================================

final = comparison.iloc[-1]

initial = comparison.iloc[0]

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(f"Simulation time: {final['t_min']:.1f} min")

print(
    f"Euler burned area: "
    f"{final['Euler']:.6f} ha"
)

print(
    f"Level-set burned area: "
    f"{final['Level_Set']:.6f} ha"
)

print(
    f"Signed area difference: "
    f"{final['difference_ha']:.6f} ha"
)

print(
    f"Absolute area difference: "
    f"{final['absolute_difference_ha']:.6f} ha"
)

print(
    f"Percentage difference: "
    f"{final['percentage_difference']:.3f}%"
)

print(
    f"Relative difference (Euler reference): "
    f"{final['relative_difference_euler_pct']:.3f}%"
)

print("\nINITIAL COMPARISON")

print(
    f"Initial Euler area: "
    f"{initial['Euler']:.6f} ha"
)

print(
    f"Initial level-set area: "
    f"{initial['Level_Set']:.6f} ha"
)

print(
    f"Initial percentage difference: "
    f"{initial['percentage_difference']:.3f}%"
)

# Analytical initial circular area
analytical_initial_area = (
    np.pi * INITIAL_RADIUS**2 / 10000.0
)

print(
    f"Analytical initial circular area: "
    f"{analytical_initial_area:.6f} ha"
)


# ============================================================
# 9. SAVE RESULTS TO CSV
# ============================================================

csv_path = (
    OUTPUT_DIR / "area_difference_analysis.csv"
)

comparison.to_csv(
    csv_path,
    index=False,
    float_format="%.10f"
)

print(f"\nSaved CSV: {csv_path}")


# ============================================================
# 10. PLOT BURNED AREA OVER TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    comparison["t_min"],
    comparison["Euler"],
    color=EULER_COLOR,
    linewidth=2.5,
    label="Euler marker method"
)

ax.plot(
    comparison["t_min"],
    comparison["Level_Set"],
    color=LEVELSET_COLOR,
    linewidth=2.5,
    label="Level-set method"
)

ax.set_xlabel(
    "Simulation time (minutes)"
)

ax.set_ylabel(
    "Burned area (ha)"
)

ax.set_title(
    "IGNACIO: Burned Area Over Time\n"
    "Euler Marker vs Level Set | 2 m Grid"
)

ax.legend(
    loc="upper left"
)

ax.set_xlim(
    comparison["t_min"].min(),
    comparison["t_min"].max()
)

ax.set_ylim(bottom=0)

fig.tight_layout()

area_plot_path = (
    OUTPUT_DIR / "burned_area_comparison.png"
)

fig.savefig(
    area_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {area_plot_path}")


# ============================================================
# 11. PLOT PERCENTAGE DIFFERENCE OVER TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    comparison["t_min"],
    comparison["percentage_difference"],
    color=DIFFERENCE_COLOR,
    linewidth=2.5,
    label="Percentage difference"
)

ax.axhline(
    0,
    linestyle="--",
    color="black",
    linewidth=1
)

ax.scatter(
    final["t_min"],
    final["percentage_difference"],
    color=DIFFERENCE_COLOR,
    s=75,
    zorder=5
)

ax.annotate(
    f"{final['percentage_difference']:.2f}%",
    xy=(
        final["t_min"],
        final["percentage_difference"]
    ),
    xytext=(-65, 18),
    textcoords="offset points",
    fontsize=12,
    fontweight="bold",
    color=DIFFERENCE_COLOR,
    arrowprops={
        "arrowstyle": "->",
        "color": DIFFERENCE_COLOR
    }
)

ax.set_xlabel(
    "Simulation time (minutes)"
)

ax.set_ylabel(
    "Burned-area percentage difference (%)"
)

ax.set_title(
    "IGNACIO: Percentage Difference Over Time\n"
    "Euler Marker vs Level Set | 2 m Grid"
)

ax.set_xlim(
    comparison["t_min"].min(),
    comparison["t_min"].max() + 2
)

ax.set_ylim(bottom=0)

ax.legend(
    loc="upper right"
)

fig.tight_layout()

percentage_plot_path = (
    OUTPUT_DIR / "area_percentage_difference.png"
)

fig.savefig(
    percentage_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {percentage_plot_path}")


# ============================================================
# 12. PLOT ABSOLUTE AREA DIFFERENCE OVER TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    comparison["t_min"],
    comparison["absolute_difference_ha"],
    color="tab:red",
    linewidth=2.5,
    label="Absolute area difference"
)

ax.fill_between(
    comparison["t_min"],
    comparison["absolute_difference_ha"],
    0,
    alpha=0.15,
    color="tab:red"
)

ax.scatter(
    final["t_min"],
    final["absolute_difference_ha"],
    color="tab:red",
    s=75,
    zorder=5
)

ax.annotate(
    f"{final['absolute_difference_ha']:.3f} ha",
    xy=(
        final["t_min"],
        final["absolute_difference_ha"]
    ),
    xytext=(-95, 18),
    textcoords="offset points",
    fontweight="bold",
    arrowprops={
        "arrowstyle": "->",
        "color": "tab:red"
    }
)

ax.set_xlabel(
    "Simulation time (minutes)"
)

ax.set_ylabel(
    "Absolute burned-area difference (ha)"
)

ax.set_title(
    "IGNACIO: Absolute Area Difference Over Time\n"
    "Euler Marker vs Level Set | 2 m Grid"
)

ax.set_xlim(
    comparison["t_min"].min(),
    comparison["t_min"].max() + 2
)

ax.set_ylim(bottom=0)

ax.legend(
    loc="upper left"
)

fig.tight_layout()

absolute_plot_path = (
    OUTPUT_DIR / "absolute_area_difference.png"
)

fig.savefig(
    absolute_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {absolute_plot_path}")


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("IGNACIO AREA DIFFERENCE ANALYSIS COMPLETE")
print("=" * 60)

print(f"Spatial resolution: {GRID_RESOLUTION} m")
print(f"Initial ignition radius: {INITIAL_RADIUS} m")
print(f"Matched timesteps: {len(comparison)}")
print(f"Final time: {final['t_min']:.1f} minutes")

print("\nOutput files:")

print(f"1. {csv_path}")
print(f"2. {area_plot_path}")
print(f"3. {percentage_plot_path}")
print(f"4. {absolute_plot_path}")

print("=" * 60)
