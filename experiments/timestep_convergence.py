"""Run a marker-advection timestep convergence experiment.

Submit the SLURM job from the repository root with:

    sbatch experiments/timestep_convergence.sbatch

For a local run, use:

    python -m experiments.timestep_convergence --config ignacio.yaml

The script compares Euler, Heun, and RK4 marker simulations with a fine-step
RK4 reference. It writes case metrics, convergence orders, and plots to the
configured output directory (or ``<project.output_dir>/timestep_convergence``).
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import make_valid
from shapely.geometry import Polygon
from shapely.ops import transform as transform_geometry

from ignacio.config import IgnacioConfig, load_config
from ignacio.ignition import generate_ignitions
from ignacio.io import read_raster_int
from ignacio.simulation import build_parameter_grid
from ignacio.spread import FireParameterGrid, FirePerimeterHistory, simulate_fire_spread
from ignacio.terrain import build_terrain_grids
from ignacio.weather import FireWeatherList, load_weather_data, process_fire_weather

logger = logging.getLogger("timestep_convergence")

DEFAULT_TIMESTEPS = (60.0, 30.0, 15.0, 7.5, 3.75)
DEFAULT_INTEGRATORS = ("euler", "heun", "rk4")
VALID_INTEGRATORS = frozenset(DEFAULT_INTEGRATORS)


def _build_history_geometry(
    history: FirePerimeterHistory,
    transformer: Transformer | None,
) -> Polygon:
    if not history.perimeters:
        raise ValueError("Simulation produced no perimeters to compare.")
    x, y = history.perimeters[-1]
    if len(x) < 3 or len(y) < 3:
        raise ValueError("Final perimeter has fewer than three markers.")
    polygon = Polygon(np.column_stack((x, y)))
    if transformer is not None:
        polygon = transform_geometry(transformer.transform, polygon)
    if not polygon.is_valid:
        polygon = make_valid(polygon)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("Final perimeter is empty or has zero area.")
    return polygon


def _metric_transformer(crs_value: str | None, center_x: float, center_y: float) -> Transformer | None:
    if crs_value is None:
        return None
    source_crs = CRS.from_user_input(crs_value)
    geographic_crs = CRS.from_epsg(4326)
    to_geographic = Transformer.from_crs(source_crs, geographic_crs, always_xy=True)
    center_lon, center_lat = to_geographic.transform(center_x, center_y)
    metric_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(source_crs, metric_crs, always_xy=True)


def _compare_geometries(simulated: Polygon, reference: Polygon) -> dict[str, float]:
    union_area = simulated.union(reference).area
    intersection_area = simulated.intersection(reference).area
    reference_area = reference.area
    reference_perimeter = reference.length
    area_error = abs(simulated.area - reference_area)
    perimeter_error = abs(simulated.length - reference_perimeter)
    return {
        "iou": float(intersection_area / union_area) if union_area > 0 else 1.0,
        "hausdorff_distance_m": float(simulated.boundary.hausdorff_distance(reference.boundary)),
        "area_error": float(area_error),
        "area_error_percent": float(100.0 * area_error / reference_area) if reference_area > 0 else 0.0,
        "perimeter_error": float(perimeter_error),
        "perimeter_error_percent": (
            float(100.0 * perimeter_error / reference_perimeter)
            if reference_perimeter > 0
            else 0.0
        ),
    }


def _enrich_hourly_weather(
    hourly_data: pd.DataFrame | None,
    weather: FireWeatherList,
) -> pd.DataFrame | None:
    if hourly_data is None or "DATE" not in hourly_data.columns:
        return hourly_data
    daily = weather.records
    if "FFMC" not in daily.columns or "BUI" not in daily.columns:
        return hourly_data

    from ignacio.fwi import calculate_isi

    date_keys = pd.to_datetime(daily["DATE"]).dt.date
    date_to_ffmc = dict(zip(date_keys, daily["FFMC"]))
    date_to_bui = dict(zip(date_keys, daily["BUI"]))
    hourly_dates = pd.to_datetime(hourly_data["DATE"]).dt.date
    wind_column = "WIND_SPEED" if "WIND_SPEED" in hourly_data.columns else "WS"
    hourly_data = hourly_data.copy()
    hourly_data["BUI"] = [date_to_bui.get(day, np.nan) for day in hourly_dates]
    hourly_ffmc = np.array([date_to_ffmc.get(day, np.nan) for day in hourly_dates])
    hourly_wind = hourly_data[wind_column].to_numpy(dtype=float)
    hourly_data["ISI"] = [
        calculate_isi(ffmc, wind) if np.isfinite(ffmc) and np.isfinite(wind) else np.nan
        for ffmc, wind in zip(hourly_ffmc, hourly_wind)
    ]
    hourly_data["FFMC"] = hourly_ffmc
    return hourly_data


def _run_case(
    grid: FireParameterGrid,
    config: IgnacioConfig,
    ignition_x: float,
    ignition_y: float,
    dt: float,
    n_steps: int,
    integrator: str,
    is_geographic: bool,
    center_latitude: float | None,
) -> tuple[FirePerimeterHistory, float]:
    marker = config.simulation.marker_method
    resample_spacing = marker.resample_spacing
    if resample_spacing is not None and resample_spacing > 0 and is_geographic:
        latitude = center_latitude if center_latitude is not None else ignition_y
        meters_per_degree_lat = 111_320.0
        meters_per_degree_lon = 111_320.0 * math.cos(math.radians(latitude))
        resample_spacing /= math.sqrt(meters_per_degree_lat * meters_per_degree_lon)

    start = time.perf_counter()
    history = simulate_fire_spread(
        param_grid=grid,
        x_ignition=ignition_x,
        y_ignition=ignition_y,
        dt=dt,
        n_vertices=config.simulation.n_vertices,
        initial_radius=config.simulation.initial_radius,
        store_every=n_steps,
        max_steps=n_steps,
        use_markers=marker.enabled,
        marker_epsilon=marker.epsilon,
        min_ros=config.simulation.min_ros,
        is_geographic=is_geographic,
        center_latitude=center_latitude,
        advection_integrator=integrator,
        resample_spacing=resample_spacing,
        insert_factor=marker.insert_factor,
        delete_factor=marker.delete_factor,
        smoothing_weight=marker.smooth_weight,
        smoothing_iters=marker.smooth_iters,
        redistribute_every=marker.redistribute_every if marker.redistribute_markers else 0,
        max_substeps=marker.max_substeps,
        max_move_fraction=marker.max_move_fraction,
    )
    return history, time.perf_counter() - start


def _convergence_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    error_fields = (
        "hausdorff_distance_m",
        "area_error_percent",
        "perimeter_error_percent",
        "iou_error",
    )
    by_integrator: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_integrator.setdefault(str(row["integrator"]), []).append(row)

    results: list[dict[str, object]] = []
    for integrator, cases in by_integrator.items():
        cases.sort(key=lambda case: float(case["dt"]), reverse=True)
        for coarse, fine in zip(cases, cases[1:]):
            dt_ratio = float(coarse["dt"]) / float(fine["dt"])
            result: dict[str, object] = {
                "integrator": integrator,
                "coarse_dt": coarse["dt"],
                "fine_dt": fine["dt"],
            }
            for field in error_fields:
                coarse_error = float(coarse[field])
                fine_error = float(fine[field])
                if coarse_error > 0 and fine_error > 0:
                    order = math.log(coarse_error / fine_error) / math.log(dt_ratio)
                else:
                    order = math.nan
                result[f"{field}_order"] = order
            results.append(result)
    return results


def _save_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    frames = pd.DataFrame(rows)
    plots = (
        ("hausdorff_distance_m", "Hausdorff distance (m)", "hausdorff_distance.png", True),
        ("area_error_percent", "Area error (%)", "area_error.png", True),
        ("perimeter_error_percent", "Perimeter error (%)", "perimeter_error.png", True),
        ("iou", "Intersection over union", "iou.png", False),
        ("runtime_seconds", "Simulation runtime (s)", "runtime.png", False),
    )
    for field, label, filename, log_y in plots:
        fig, ax = plt.subplots(figsize=(7, 5))
        for integrator, group in frames.groupby("integrator"):
            group = group.sort_values("dt", ascending=False)
            y_values = group[field].to_numpy(dtype=float)
            if log_y:
                y_values = np.where(y_values > 0, y_values, np.nan)
            ax.plot(group["dt"], y_values, marker="o", label=integrator)
        ax.set_xlabel("Timestep (minutes)")
        ax.set_ylabel(label)
        ax.set_title(label + " vs timestep")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("ignacio.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timesteps", type=float, nargs="+", default=DEFAULT_TIMESTEPS)
    parser.add_argument("--integrators", nargs="+", default=DEFAULT_INTEGRATORS)
    parser.add_argument("--reference-timestep", type=float)
    parser.add_argument("--reference-integrator", choices=sorted(VALID_INTEGRATORS), default="rk4")
    parser.add_argument("--duration-minutes", type=float)
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    config = load_config(args.config)
    if config.simulation.spread_method != "marker":
        raise ValueError("Timestep convergence experiment requires simulation.spread_method='marker'.")

    timesteps = sorted(set(float(dt) for dt in args.timesteps), reverse=True)
    if len(timesteps) < 2 or any(not np.isfinite(dt) or dt <= 0 for dt in timesteps):
        raise ValueError("Provide at least two distinct, positive timestep values.")
    integrators = tuple(dict.fromkeys(str(name).lower() for name in args.integrators))
    unknown_integrators = set(integrators) - VALID_INTEGRATORS
    if not integrators or unknown_integrators:
        raise ValueError(f"Unsupported integrator(s): {', '.join(sorted(unknown_integrators)) or '(none)'}")

    duration = float(
        config.simulation.max_duration
        if args.duration_minutes is None
        else args.duration_minutes
    )
    reference_dt = float(
        min(timesteps) if args.reference_timestep is None else args.reference_timestep
    )
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError("Duration and reference timestep must be positive finite values.")
    all_timesteps = set(timesteps) | {reference_dt}
    step_counts: dict[float, int] = {}
    for dt in all_timesteps:
        count = int(round(duration / dt))
        if count < 1 or not math.isclose(count * dt, duration, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(f"Duration {duration:g} minutes must be an integer multiple of timestep {dt:g}.")
        step_counts[dt] = count

    output_dir = args.output_dir or Path(config.project.output_dir) / "timestep_convergence"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.project.random_seed
    terrain = build_terrain_grids(config)
    rng = np.random.default_rng(seed)
    hourly_data = None
    if config.simulation.time_varying_weather:
        hourly_data = load_weather_data(config)
        if len(hourly_data) == 0 or "HOUR" not in hourly_data.columns:
            hourly_data = None
    weather = process_fire_weather(config, rng)
    hourly_data = _enrich_hourly_weather(hourly_data, weather)

    fuel_raster = read_raster_int(config.fuel.path)
    terrain_crs = str(terrain.crs) if terrain.crs is not None else None
    ignitions = generate_ignitions(
        config,
        fuel_raster,
        np.random.default_rng(seed),
        terrain_crs=terrain_crs,
    )
    if not ignitions.points:
        raise ValueError("Configuration produced no ignition points.")
    ignition = ignitions.points[0]

    is_geographic = False
    if terrain_crs is not None:
        is_geographic = CRS.from_user_input(terrain_crs).is_geographic
    _, y_coords = terrain.get_coordinate_arrays()
    center_latitude = (
        (float(np.min(y_coords)) + float(np.max(y_coords))) / 2.0
        if is_geographic
        else None
    )
    transformer = _metric_transformer(terrain_crs, ignition.x, ignition.y)

    grid_cache: dict[float, FireParameterGrid] = {}
    grid_build_seconds: dict[float, float] = {}

    def get_grid(dt: float) -> FireParameterGrid:
        if dt not in grid_cache:
            config.simulation.dt = dt
            start = time.perf_counter()
            grid_cache[dt] = build_parameter_grid(
                config,
                terrain,
                weather,
                n_timesteps=step_counts[dt],
                hourly_data=hourly_data,
                rng=np.random.default_rng(seed),
            )
            grid_build_seconds[dt] = time.perf_counter() - start
        return grid_cache[dt]

    reference_key = (reference_dt, args.reference_integrator)
    run_cache: dict[tuple[float, str], tuple[Polygon, float]] = {}

    def run_one(dt: float, integrator: str) -> tuple[Polygon, float]:
        key = (dt, integrator)
        if key not in run_cache:
            grid = get_grid(dt)
            history, runtime = _run_case(
                grid,
                config,
                ignition.x,
                ignition.y,
                dt,
                step_counts[dt],
                integrator,
                is_geographic,
                center_latitude,
            )
            run_cache[key] = (_build_history_geometry(history, transformer), runtime)
        return run_cache[key]

    logger.info("Running reference: dt=%g, integrator=%s", *reference_key)
    reference_geometry, _ = run_one(*reference_key)
    rows: list[dict[str, object]] = []
    for dt in timesteps:
        for integrator in integrators:
            logger.info("Running case: dt=%g, integrator=%s", dt, integrator)
            geometry, runtime = run_one(dt, integrator)
            metrics = _compare_geometries(geometry, reference_geometry)
            rows.append(
                {
                    "dt": dt,
                    "integrator": integrator,
                    "n_steps": step_counts[dt],
                    "duration_minutes": duration,
                    "runtime_seconds": runtime,
                    "grid_build_seconds": grid_build_seconds[dt],
                    "reference_dt": reference_dt,
                    "reference_integrator": args.reference_integrator,
                    "iou_error": 1.0 - metrics["iou"],
                    **metrics,
                }
            )

    cases_csv = output_dir / "timestep_convergence.csv"
    convergence_csv = output_dir / "convergence_orders.csv"
    pd.DataFrame(rows).to_csv(cases_csv, index=False)
    pd.DataFrame(_convergence_rows(rows)).to_csv(convergence_csv, index=False)
    _save_plots(rows, output_dir)
    logger.info("Saved convergence results to %s", output_dir)
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
