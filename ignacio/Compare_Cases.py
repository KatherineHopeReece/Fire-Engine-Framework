
"""
IGNACIO - BOVA ET AL. (2016) NUMERICAL COMPARISON
=================================================

Compare Level-set vs Marker Euler for all H experiments.

Metrics:
    1. Burned area (ha)
    2. Fire-front perimeter (m)
    3. Signed area difference (%)
    4. Signed perimeter difference (%)
    5. Absolute percentage differences
    6. Intersection over Union (IoU)
    7. Maximum percentage difference over time
    8. Last available simulation time for each method
    9. Domain boundary / incomplete-run warnings

Sign convention:
    100 * (Levelset - Marker) / Marker

Positive = Level-set larger
Negative = Marker Euler larger

Outputs:
    Bova_IGNACIO_Case_Summary.csv
    Bova_IGNACIO_Time_Series.csv
    Bova_IGNACIO_Percent_Differences.png
    Bova_IGNACIO_Percent_Differences.pdf

Published benchmarks are approximate values taken from
Bova, Mell & Hoffman (2016), IJWF 25, 229-241.

Important:
    IGNACIO uses Canadian FBP fuel models, unlike Bova's
    WFDS-LS/FARSITE comparison. These benchmarks are
    contextual comparisons, not exact reproduction targets.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from shapely.geometry import box
from shapely.ops import unary_union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(
    "/Users/Martyn/GitRepos/Fire-Engine-Framework-dev"
)

BASE = (
    ROOT
    / "ignacio/LaughTests/Bova_1km_Experiments"
)

RESOLUTION_FOLDER = "2m"

OUTPUT_DIR = BASE / "Bova_Percent_Difference_Analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "Bova_IGNACIO_Case_Summary.csv"
TIMESERIES_CSV = OUTPUT_DIR / "Bova_IGNACIO_Time_Series.csv"

OUTPUT_PNG = OUTPUT_DIR / "Bova_IGNACIO_Percent_Differences.png"
OUTPUT_PDF = OUTPUT_DIR / "Bova_IGNACIO_Percent_Differences.pdf"

# Minimum reference area for time-series percentage statistics.
# Avoid excessively large percentages from tiny initial fires.
MIN_REFERENCE_AREA_HA = 0.1

TIME_DECIMALS = 4
TIME_TOL = 1e-4

# Select alternate output folders here.
# Particularly useful for the revised H3 experiment.
#
# Set to None for automatic selection.
#
# Example:
# "H3": {
#     "marker": "Marker_Euler_FuelBreaks_Output",
#     "levelset": "Levelset_Output",
# }

FOLDER_CHOICES = {
    "H3": {
        "marker": "Marker_Euler_Output",
        "levelset": "Levelset_Output",
    },
}

# ============================================================
# 2. BOVA ET AL. (2016) REFERENCE VALUES
# ============================================================

BOVA_REFERENCES = {
    "H0a": {
        "case": "Uniform O-1b",
        "reference": "Fig. 1a",
        "area": "Approximately 0%; nearly identical",
        "perimeter": "Nearly identical",
    },
    "H0b": {
        "case": "Uniform C-2",
        "reference": "Fig. 1; chamise comparison",
        "area": "Nearly identical (different fuel model)",
        "perimeter": "Nearly identical",
    },
    "H1": {
        "case": "Heterogeneous fuel patches",
        "reference": "Figs. 2-3",
        "area": "Level-set approximately 2% larger",
        "perimeter": "Level-set approximately 2% larger",
    },
    "H2": {
        "case": "Single non-burnable obstacle",
        "reference": "Fig. 4",
        "area": "Final absolute difference below ~5%",
        "perimeter": "Not numerically specified",
    },
    "H3": {
        "case": "Multiple checkerboard fuel breaks",
        "reference": "Figs. 5-6",
        "area": "At 2 m, final absolute difference below 3%",
        "perimeter": "Not numerically specified",
    },
    "H4": {
        "case": "Complex terrain",
        "reference": "Figs. 7-10",
        "area": (
            "Uniform grass: LS ~2% smaller; "
            "mixed fuels: ~16% smaller at peak "
            "with minmod, ~3% larger final "
            "with superbee"
        ),
        "perimeter": (
            "Uniform grass: LS ~6% larger"
        ),
    },
    "H5": {
        "case": "Two simultaneous ignitions",
        "reference": "Fig. 1b",
        "area": "Nearly identical; no numerical % reported",
        "perimeter": "Nearly identical; no numerical % reported",
    },
}

CASE_ORDER = list(BOVA_REFERENCES)


# ============================================================
# 3. DISCOVER EXPERIMENT DIRECTORIES
# ============================================================

def extract_case_id(folder_name):
    """Recognize H0a, H0b, H1, H2, ... H5."""

    match = re.match(
        r"^(H0[a-b]|H[1-9])(?:_|-|$)",
        folder_name,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    code = match.group(1)

    if code.lower().startswith("h0"):
        return "H0" + code[-1].lower()

    return code.upper()


def discover_cases():

    discovered = {}

    if not BASE.exists():
        raise FileNotFoundError(
            f"Experiment directory not found:\n{BASE}"
        )

    for folder in BASE.iterdir():

        if not folder.is_dir():
            continue

        case_id = extract_case_id(folder.name)

        if case_id is None:
            continue

        resolution_folder = folder / RESOLUTION_FOLDER

        if not resolution_folder.is_dir():
            continue

        discovered.setdefault(
            case_id, []
        ).append(resolution_folder)

    return discovered


# ============================================================
# 4. FIND PERIMETER HISTORY
# ============================================================

def find_history_path(case_id, case_dir, method):

    selected = FOLDER_CHOICES.get(
        case_id, {}
    ).get(method)

    if selected is not None:

        path = (
            case_dir
            / selected
            / "all_perimeters_history.gpkg"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Selected {method} history not found:\n{path}"
            )

        return path

    if method == "marker":
        preferred = [
            "Marker_Euler_Output",
            "Marker_Output",
            "Euler_Output",
        ]
    else:
        preferred = [
            "Levelset_Output",
            "Level_Set_Output",
        ]

    for folder in preferred:

        path = (
            case_dir / folder
            / "all_perimeters_history.gpkg"
        )

        if path.exists():
            return path

    # Limited fallback for differently named output folders.
    candidates = []

    for path in case_dir.glob(
        "*/all_perimeters_history.gpkg"
    ):

        name = path.parent.name.lower()

        if method == "marker":
            matches = (
                "marker" in name
                or "euler" in name
            )
        else:
            matches = (
                "levelset" in name
                or "level_set" in name
            )

        if matches:
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise ValueError(
            f"Multiple {method} histories for {case_id}. "
            "Set FOLDER_CHOICES explicitly:\n"
            + "\n".join(map(str, candidates))
        )

    raise FileNotFoundError(
        f"No {method} history for {case_id}"
    )


# ============================================================
# 5. LOAD AND VALIDATE POLYGON HISTORIES
# ============================================================

def load_history(path, target_crs=None):

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(
            f"Empty GeoPackage: {path}"
        )

    if "t_min" not in gdf.columns:
        raise ValueError(
            f"Missing t_min column: {path}"
        )

    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError(
            f"Projected metric CRS required: {path}"
        )

    units = gdf.crs.axis_info[0].unit_name.lower()

    if units not in (
        "metre", "meter", "metres", "meters"
    ):
        raise ValueError(
            f"CRS units must be metres, got {units}: {path}"
        )

    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)

    gdf["t_min"] = pd.to_numeric(
        gdf["t_min"],
        errors="coerce",
    )

    gdf = gdf.dropna(
        subset=["t_min"]
    ).copy()

    if gdf.empty:
        raise ValueError(
            f"No valid timestamps in {path}"
        )

    last_saved_time = float(
        gdf["t_min"].max()
    )

    gdf["_time"] = (
        gdf["t_min"].round(TIME_DECIMALS)
    )

    geometries_by_time = {}
    skipped_times = []

    for time, group in gdf.groupby("_time"):

        geometries = [
            geom
            for geom in group.geometry
            if geom is not None
            and not geom.is_empty
        ]

        if not geometries:
            skipped_times.append(
                (float(time), "empty geometry")
            )
            continue

        # Do not silently repair invalid geometry.
        # A corrected polygon might have a different area
        # and perimeter from the numerical solution.
        if any(
            not geom.is_valid
            for geom in geometries
        ):
            skipped_times.append(
                (float(time), "invalid polygon")
            )
            continue

        if any(
            geom.geom_type not in (
                "Polygon", "MultiPolygon"
            )
            for geom in geometries
        ):
            skipped_times.append(
                (float(time), "non-polygon geometry")
            )
            continue

        try:

            combined = unary_union(
                geometries
            )

            if (
                combined.is_empty
                or not combined.is_valid
                or combined.geom_type not in (
                    "Polygon", "MultiPolygon"
                )
            ):
                raise ValueError(
                    "Union is not a valid polygon"
                )

        except Exception as exc:

            skipped_times.append(
                (float(time), str(exc))
            )
            continue

        geometries_by_time[
            float(time)
        ] = combined

    return (
        geometries_by_time,
        last_saved_time,
        skipped_times,
        gdf.crs,
    )


# ============================================================
# 6. METRICS
# ============================================================

def percentage_difference(
    levelset_value,
    marker_value,
):

    if abs(marker_value) < 1e-12:
        return np.nan

    return (
        100.0
        * (levelset_value - marker_value)
        / marker_value
    )


def compute_iou(marker_geom, levelset_geom):

    intersection = (
        marker_geom.intersection(levelset_geom).area
    )

    union = (
        marker_geom.union(levelset_geom).area
    )

    if union <= 0:
        return np.nan

    return intersection / union


def get_domain(case_dir):

    dem_path = case_dir / "DEM.tif"

    if not dem_path.exists():
        return None, None

    with rasterio.open(dem_path) as src:

        bounds = src.bounds

        domain = box(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )

        cell_size = max(
            abs(src.res[0]),
            abs(src.res[1]),
        )

    return domain, cell_size


def touches_domain_edge(
    geom,
    domain,
    tolerance,
):

    if domain is None:
        return False

    # Detect fire reaching or exceeding the true
    # physical DEM boundary.
    if not domain.buffer(1e-6).covers(geom):
        return True

    return (
        geom.boundary.distance(
            domain.boundary
        ) <= tolerance
    )


# ============================================================
# 7. COMPARE ONE EXPERIMENT
# ============================================================

def compare_case(case_id, case_dir):

    marker_path = find_history_path(
        case_id, case_dir, "marker"
    )

    levelset_path = find_history_path(
        case_id, case_dir, "levelset"
    )

    marker, marker_end, marker_skips, crs = (
        load_history(marker_path)
    )

    levelset, levelset_end, levelset_skips, _ = (
        load_history(
            levelset_path,
            target_crs=crs,
        )
    )

    common_times = sorted(
        set(marker) & set(levelset)
    )

    if not common_times:
        raise ValueError(
            "No valid matching simulation times."
        )

    domain, cell_size = get_domain(case_dir)

    records = []

    for time in common_times:

        gm = marker[time]
        gl = levelset[time]

        marker_area = gm.area / 10000.0
        levelset_area = gl.area / 10000.0

        # Shapely's polygon boundary includes interior
        # holes, when present.
        marker_perimeter = gm.boundary.length
        levelset_perimeter = gl.boundary.length

        area_diff = percentage_difference(
            levelset_area,
            marker_area,
        )

        perimeter_diff = percentage_difference(
            levelset_perimeter,
            marker_perimeter,
        )

        iou = compute_iou(
            gm, gl
        )

        boundary_affected = (
            touches_domain_edge(
                gm, domain,
                cell_size or 0,
            )
            or touches_domain_edge(
                gl, domain,
                cell_size or 0,
            )
        )

        records.append({
            "case": case_id,
            "experiment_folder": case_dir.parent.name,
            "time_min": time,

            "marker_area_ha": marker_area,
            "levelset_area_ha": levelset_area,
            "area_diff_ha": levelset_area - marker_area,
            "area_diff_pct": area_diff,
            "area_abs_diff_pct": abs(area_diff),

            "marker_perimeter_m": marker_perimeter,
            "levelset_perimeter_m": levelset_perimeter,
            "perimeter_diff_m": (
                levelset_perimeter - marker_perimeter
            ),
            "perimeter_diff_pct": perimeter_diff,
            "perimeter_abs_diff_pct": abs(
                perimeter_diff
            ),

            "iou": iou,
            "domain_edge_affected": boundary_affected,
        })

    detail = pd.DataFrame(records)

    final = detail.iloc[-1]
    comparable = detail[
        detail["marker_area_ha"]
        >= MIN_REFERENCE_AREA_HA
    ]

    if comparable.empty:
        comparable = detail

    different_ends = (
        abs(marker_end - levelset_end)
        > TIME_TOL
    )

    any_invalid = bool(
        marker_skips or levelset_skips
    )

    status = []

    if different_ends:
        status.append("DIFFERENT_END_TIMES")

    if bool(final["domain_edge_affected"]):
        status.append("DOMAIN_EDGE")

    if any_invalid:
        status.append("INVALID_SNAPSHOTS_SKIPPED")

    if not status:
        status = ["OK"]

    reference = BOVA_REFERENCES[case_id]

    summary = {
        "case": case_id,
        "landscape": reference["case"],
        "experiment_folder": case_dir.parent.name,

        "marker_source": str(marker_path),
        "levelset_source": str(levelset_path),

        "marker_last_saved_min": marker_end,
        "levelset_last_saved_min": levelset_end,
        "comparison_time_min": float(final["time_min"]),
        "matched_snapshots": len(detail),

        "marker_area_ha": float(final["marker_area_ha"]),
        "levelset_area_ha": float(final["levelset_area_ha"]),
        "area_diff_pct": float(final["area_diff_pct"]),
        "area_abs_diff_pct": float(final["area_abs_diff_pct"]),

        "marker_perimeter_m": float(
            final["marker_perimeter_m"]
        ),
        "levelset_perimeter_m": float(
            final["levelset_perimeter_m"]
        ),
        "perimeter_diff_pct": float(
            final["perimeter_diff_pct"]
        ),
        "perimeter_abs_diff_pct": float(
            final["perimeter_abs_diff_pct"]
        ),

        "final_iou": float(final["iou"]),

        "mean_abs_area_diff_pct": float(
            comparable["area_abs_diff_pct"].mean()
        ),
        "max_abs_area_diff_pct": float(
            comparable["area_abs_diff_pct"].max()
        ),
        "mean_abs_perimeter_diff_pct": float(
            comparable["perimeter_abs_diff_pct"].mean()
        ),
        "max_abs_perimeter_diff_pct": float(
            comparable["perimeter_abs_diff_pct"].max()
        ),

        "marker_skipped_times": len(marker_skips),
        "levelset_skipped_times": len(levelset_skips),
        "status": "; ".join(status),

        "bova_reference": reference["reference"],
        "bova_area_result": reference["area"],
        "bova_perimeter_result": reference["perimeter"],
    }

    print(f"\n{case_id}: {reference['case']}")
    print("-" * 58)
    print(
        f"Compared at t = {final['time_min']:.2f} min"
    )
    print(
        f"Marker last saved = {marker_end:.2f} min"
    )
    print(
        f"Level-set last saved = {levelset_end:.2f} min"
    )
    print(
        f"Area difference = {final['area_diff_pct']:+.3f}%"
    )
    print(
        "Perimeter difference = "
        f"{final['perimeter_diff_pct']:+.3f}%"
    )
    print(
        f"IoU = {final['iou']:.4f}"
    )
    print(
        f"Status = {'; '.join(status)}"
    )

    if marker_skips:
        print(
            "Marker invalid snapshot examples:",
            marker_skips[:3],
        )

    if levelset_skips:
        print(
            "Level-set invalid snapshot examples:",
            levelset_skips[:3],
        )

    return summary, detail


# ============================================================
# 8. PLOT SUMMARY
# ============================================================

def plot_summary(summary_df):

    valid = summary_df[
        summary_df["comparison_time_min"].notna()
    ].copy()

    if valid.empty:
        return

    x = np.arange(len(valid))

    labels = [
        f"{row['case']}\nt={row['comparison_time_min']:g}"
        for _, row in valid.iterrows()
    ]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 9),
        constrained_layout=True,
    )

    metrics = [
        (
            "area_diff_pct",
            "Burned-area difference (%)",
        ),
        (
            "perimeter_diff_pct",
            "Fire-front perimeter difference (%)",
        ),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):

        values = valid[metric].to_numpy(
            dtype=float
        )

        bars = ax.bar(
            x,
            values,
            width=0.65,
            alpha=0.9,
        )

        ax.axhline(
            0,
            color="black",
            linewidth=1,
        )

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.25,
        )

        ax.set_axisbelow(True)

        for bar, value in zip(bars, values):

            if not np.isfinite(value):
                continue

            ax.annotate(
                f"{value:+.2f}%",
                (
                    bar.get_x()
                    + bar.get_width() / 2,
                    value,
                ),
                xytext=(
                    0,
                    5 if value >= 0 else -5
                ),
                textcoords="offset points",
                ha="center",
                va=(
                    "bottom"
                    if value >= 0
                    else "top"
                ),
                fontsize=9,
            )

        # Provide room for labels.
        max_abs = np.nanmax(
            np.abs(values)
        )

        limit = max(
            3.0,
            max_abs * 1.35,
        )

        ax.set_ylim(-limit, limit)

    fig.suptitle(
        "IGNACIO: Level-set vs Marker Euler\n"
        "Percentage differences by experimental landscape",
        fontsize=16,
        fontweight="bold",
    )

    fig.supxlabel(
        "Case and latest common comparison time (min)\n"
        "Positive = Level-set larger; "
        "negative = Marker Euler larger",
        fontsize=10,
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 72)
    print("IGNACIO - BOVA NUMERICAL PERCENT DIFFERENCES")
    print("=" * 72)

    discovered = discover_cases()

    summaries = []
    details = []

    for case_id in CASE_ORDER:

        directories = discovered.get(
            case_id, []
        )

        reference = BOVA_REFERENCES[case_id]

        if not directories:

            print(
                f"\n{case_id}: no 2 m experiment found"
            )

            summaries.append({
                "case": case_id,
                "landscape": reference["case"],
                "status": "NOT_FOUND",
                "bova_reference": reference["reference"],
                "bova_area_result": reference["area"],
                "bova_perimeter_result": reference["perimeter"],
            })

            continue

        if len(directories) != 1:

            print(
                f"\n{case_id}: ambiguous directories:",
                directories,
            )

            summaries.append({
                "case": case_id,
                "landscape": reference["case"],
                "status": "AMBIGUOUS_DIRECTORIES",
            })

            continue

        try:

            summary, detail = compare_case(
                case_id,
                directories[0],
            )

            summaries.append(summary)
            details.append(detail)

        except Exception as exc:

            print(
                f"\n{case_id}: could not compare: {exc}"
            )

            summaries.append({
                "case": case_id,
                "landscape": reference["case"],
                "status": f"ERROR: {exc}",
                "bova_reference": reference["reference"],
                "bova_area_result": reference["area"],
                "bova_perimeter_result": reference["perimeter"],
            })

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    if details:

        detail_df = pd.concat(
            details,
            ignore_index=True,
        )

        detail_df.to_csv(
            TIMESERIES_CSV,
            index=False,
        )

    plot_summary(summary_df)

    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)

    columns = [
        "case",
        "comparison_time_min",
        "area_diff_pct",
        "perimeter_diff_pct",
        "final_iou",
        "status",
    ]

    print(
        summary_df.reindex(
            columns=columns
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nOUTPUTS")
    print(SUMMARY_CSV)

    if details:
        print(TIMESERIES_CSV)
        print(OUTPUT_PNG)
        print(OUTPUT_PDF)

    print(
        "\nInterpretation: positive percentages mean "
        "the Level-set result is larger."
    )

    print(
        "For manuscript comparisons, review the status "
        "flags and compare only matching simulation times."
    )


if __name__ == "__main__":
    main()
