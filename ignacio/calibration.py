"""
Fire shape calibration and evaluation metrics.

Provides gradient-free evaluation metrics (Jaccard, Sorensen, Procrustes, etc.)
for assessing simulation accuracy, and differentiable level-set-based losses
for gradient-based parameter optimization.

References
----------
- Arca, B. et al. (2007). Evaluation of FARSITE simulator in Mediterranean
  maquis. Int. J. Wildland Fire.
- Duff, T.J. et al. (2012). Indices for the evaluation of wildfire spread
  simulations using contemporaneous reference data.
- Hao, Z. et al. (2018). Performance comparison of wildfire spread simulators.
- Zhou, S. et al. (2024). Evaluation metrics for wildfire simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

logger = logging.getLogger(__name__)


# =============================================================================
# Helper functions
# =============================================================================


def _rasterize_perimeter(
    xy_or_list,
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> np.ndarray:
    """
    Rasterize polygon perimeter(s) to a binary grid.

    Parameters
    ----------
    xy_or_list : np.ndarray or list of np.ndarray
        (N, 2) polygon vertices, or list of polygons for MultiPolygon.
    bounds : tuple
        (xmin, ymin, xmax, ymax) bounding box.
    resolution : float
        Grid cell size.

    Returns
    -------
    grid : np.ndarray
        2D boolean grid (ny, nx) with True inside any polygon.
    """
    from matplotlib.path import Path

    xmin, ymin, xmax, ymax = bounds
    nx = max(1, int(np.ceil((xmax - xmin) / resolution)))
    ny = max(1, int(np.ceil((ymax - ymin) / resolution)))

    # Build grid of cell centers
    gx = np.linspace(xmin + resolution / 2, xmin + (nx - 0.5) * resolution, nx)
    gy = np.linspace(ymin + resolution / 2, ymin + (ny - 0.5) * resolution, ny)
    GX, GY = np.meshgrid(gx, gy)
    points = np.column_stack([GX.ravel(), GY.ravel()])

    # Handle single polygon or list of polygons
    if isinstance(xy_or_list, list):
        mask = np.zeros(len(points), dtype=bool)
        for xy in xy_or_list:
            if len(xy) >= 3:
                mask |= Path(xy).contains_points(points)
        return mask.reshape(ny, nx)
    else:
        return Path(xy_or_list).contains_points(points).reshape(ny, nx)


def _sample_pseudo_landmarks(
    xy: np.ndarray,
    n_landmarks: int,
) -> np.ndarray:
    """
    Sample evenly-spaced pseudo-landmarks along a perimeter.

    Parameters
    ----------
    xy : np.ndarray
        (N, 2) array of perimeter vertices.
    n_landmarks : int
        Number of evenly-spaced points to sample.

    Returns
    -------
    landmarks : np.ndarray
        (n_landmarks, 2) evenly-spaced points.
    """
    # Compute cumulative arc length
    diffs = np.diff(xy, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len < 1e-12:
        return np.tile(xy[0], (n_landmarks, 1))

    # Evenly spaced parameter values
    t_vals = np.linspace(0, total_len, n_landmarks, endpoint=False)

    # Interpolate
    landmarks = np.zeros((n_landmarks, 2))
    landmarks[:, 0] = np.interp(t_vals, cum_len, xy[:, 0])
    landmarks[:, 1] = np.interp(t_vals, cum_len, xy[:, 1])
    return landmarks


def _kabsch_rotation_2d(
    P: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """
    Find optimal 2D rotation (Kabsch algorithm) to align P onto Q.

    Both P and Q should be centered (mean-subtracted) before calling.

    Parameters
    ----------
    P, Q : np.ndarray
        (N, 2) point sets, already centered.

    Returns
    -------
    P_rotated : np.ndarray
        P after optimal rotation.
    """
    H = P.T @ Q  # 2x2
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, np.sign(d)])
    R = Vt.T @ S @ U.T
    return P @ R.T


# =============================================================================
# Evaluation metrics (gradient-free)
# =============================================================================


def jaccard_index(
    sim_xy: np.ndarray,
    obs_xy: np.ndarray,
    bounds: tuple[float, float, float, float],
    resolution: float = 30.0,
) -> float:
    """
    Compute Jaccard index (IoU) between simulated and observed fire perimeters.

    Parameters
    ----------
    sim_xy, obs_xy : np.ndarray
        (N, 2) arrays of perimeter vertices.
    bounds : tuple
        (xmin, ymin, xmax, ymax) for rasterization.
    resolution : float
        Grid cell size for rasterization.

    Returns
    -------
    float
        Jaccard index in [0, 1]. Higher is better.
    """
    sim_mask = _rasterize_perimeter(sim_xy, bounds, resolution)
    obs_mask = _rasterize_perimeter(obs_xy, bounds, resolution)

    intersection = np.sum(sim_mask & obs_mask)
    union = np.sum(sim_mask | obs_mask)

    if union == 0:
        return 1.0
    return float(intersection / union)


def sorensen_coefficient(
    sim_xy: np.ndarray,
    obs_xy: np.ndarray,
    bounds: tuple[float, float, float, float],
    resolution: float = 30.0,
) -> float:
    """
    Compute Sorensen-Dice coefficient between simulated and observed perimeters.

    Returns
    -------
    float
        Sorensen coefficient in [0, 1]. Higher is better.
    """
    sim_mask = _rasterize_perimeter(sim_xy, bounds, resolution)
    obs_mask = _rasterize_perimeter(obs_xy, bounds, resolution)

    intersection = np.sum(sim_mask & obs_mask)
    total = np.sum(sim_mask) + np.sum(obs_mask)

    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def centroid_size_ratio(
    sim_xy: np.ndarray,
    obs_xy: np.ndarray,
    origin: np.ndarray | None = None,
) -> float:
    """
    Compute log centroid size ratio per Duff (2012).

    S(X) = sqrt(sum(||X_j - origin||^2) / n)

    Parameters
    ----------
    sim_xy, obs_xy : np.ndarray
        (N, 2) perimeter vertices.
    origin : np.ndarray, optional
        (2,) reference point (ignition). Defaults to centroid of obs_xy.

    Returns
    -------
    float
        ln(S_sim / S_obs). 0 = perfect size match.
    """
    if origin is None:
        origin = obs_xy.mean(axis=0)

    def _centroid_size(xy, o):
        d = xy - o
        return np.sqrt(np.mean(np.sum(d ** 2, axis=1)))

    s_sim = _centroid_size(sim_xy, origin)
    s_obs = _centroid_size(obs_xy, origin)

    if s_obs < 1e-12:
        return float("inf")
    return float(np.log(s_sim / s_obs))


def procrustes_shape_rmse(
    sim_xy: np.ndarray,
    obs_xy: np.ndarray,
    origin: np.ndarray | None = None,
    n_landmarks: int = 200,
) -> float:
    """
    Compute Procrustes shape RMSE per Duff (2012).

    Samples pseudo-landmarks on both perimeters, applies optimal rotation
    (shared origin at ignition), and returns normalized residual RMSE.

    Parameters
    ----------
    sim_xy, obs_xy : np.ndarray
        (N, 2) perimeter vertices.
    origin : np.ndarray, optional
        (2,) reference origin. Defaults to centroid of obs_xy.
    n_landmarks : int
        Number of pseudo-landmarks to sample.

    Returns
    -------
    float
        Procrustes RMSE normalized by S_obs. Lower is better.
    """
    if origin is None:
        origin = obs_xy.mean(axis=0)

    # Sample landmarks
    sim_lm = _sample_pseudo_landmarks(sim_xy, n_landmarks)
    obs_lm = _sample_pseudo_landmarks(obs_xy, n_landmarks)

    # Center on origin
    sim_c = sim_lm - origin
    obs_c = obs_lm - origin

    # Normalize by centroid size of observed
    s_obs = np.sqrt(np.mean(np.sum(obs_c ** 2, axis=1)))
    if s_obs < 1e-12:
        return float("inf")

    sim_n = sim_c / s_obs
    obs_n = obs_c / s_obs

    # Optimal rotation
    sim_rot = _kabsch_rotation_2d(sim_n, obs_n)

    # Residual RMSE
    residuals = sim_rot - obs_n
    rmse = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
    return float(rmse)


def orientation_error(
    sim_xy: np.ndarray,
    obs_xy: np.ndarray,
    origin: np.ndarray | None = None,
) -> float:
    """
    Compute orientation error in degrees per Duff (2012).

    Finds the furthest point from origin on each perimeter (primary spread
    axis) and returns the angular difference.

    Parameters
    ----------
    sim_xy, obs_xy : np.ndarray
        (N, 2) perimeter vertices.
    origin : np.ndarray, optional
        (2,) reference origin. Defaults to centroid of obs_xy.

    Returns
    -------
    float
        Angular difference in degrees in [0, 180].
    """
    if origin is None:
        origin = obs_xy.mean(axis=0)

    def _primary_angle(xy, o):
        d = xy - o
        dists = np.sqrt(np.sum(d ** 2, axis=1))
        idx = np.argmax(dists)
        return np.arctan2(d[idx, 1], d[idx, 0])

    a_sim = _primary_angle(sim_xy, origin)
    a_obs = _primary_angle(obs_xy, origin)

    diff = np.abs(a_sim - a_obs)
    # Wrap to [0, pi]
    diff = np.minimum(diff, 2 * np.pi - diff)
    return float(np.degrees(diff))


# =============================================================================
# Differentiable losses (JAX)
# =============================================================================


def _bilinear_sample_phi(
    phi: "jnp.ndarray",
    points_xy: "jnp.ndarray",
    x_coords: "jnp.ndarray",
    y_coords: "jnp.ndarray",
) -> "jnp.ndarray":
    """
    Bilinear interpolation of φ at arbitrary (x, y) points (JAX).

    Parameters
    ----------
    phi : jnp.ndarray
        (ny, nx) level set field.
    points_xy : jnp.ndarray
        (M, 2) query points in world coordinates.
    x_coords, y_coords : jnp.ndarray
        1D coordinate arrays.

    Returns
    -------
    values : jnp.ndarray
        (M,) interpolated φ values.
    """
    # Map world coords to fractional pixel coords
    x0, dx = x_coords[0], x_coords[1] - x_coords[0]
    y0, dy = y_coords[0], y_coords[1] - y_coords[0]

    col_f = (points_xy[:, 0] - x0) / dx
    row_f = (points_xy[:, 1] - y0) / dy

    ny, nx = phi.shape

    # Clamp to valid range
    col_f = jnp.clip(col_f, 0, nx - 1.001)
    row_f = jnp.clip(row_f, 0, ny - 1.001)

    c0 = jnp.floor(col_f).astype(jnp.int32)
    r0 = jnp.floor(row_f).astype(jnp.int32)
    c1 = jnp.minimum(c0 + 1, nx - 1)
    r1 = jnp.minimum(r0 + 1, ny - 1)

    wc = col_f - c0
    wr = row_f - r0

    # Bilinear
    val = (
        phi[r0, c0] * (1 - wr) * (1 - wc)
        + phi[r0, c1] * (1 - wr) * wc
        + phi[r1, c0] * wr * (1 - wc)
        + phi[r1, c1] * wr * wc
    )
    return val


def phi_perimeter_loss(
    phi: "jnp.ndarray",
    obs_points_xy: "jnp.ndarray",
    x_coords: "jnp.ndarray",
    y_coords: "jnp.ndarray",
) -> "jnp.ndarray":
    """
    Differentiable perimeter loss: observed boundary points should have φ ≈ 0.

    loss = mean(φ(x_obs, y_obs)²)

    Parameters
    ----------
    phi : jnp.ndarray
        (ny, nx) level set field.
    obs_points_xy : jnp.ndarray
        (M, 2) observed fire perimeter points.
    x_coords, y_coords : jnp.ndarray
        1D coordinate arrays.

    Returns
    -------
    jnp.ndarray
        Scalar loss value.
    """
    phi_at_obs = _bilinear_sample_phi(phi, obs_points_xy, x_coords, y_coords)
    return jnp.mean(phi_at_obs ** 2)


def phi_area_loss(
    phi: "jnp.ndarray",
    obs_area: float,
    dx: float,
    dy: float,
    epsilon: float = 1.0,
) -> "jnp.ndarray":
    """
    Differentiable area loss using smooth Heaviside on φ.

    H_ε(z) = 0.5 * (1 - tanh(z / ε))
    area_sim = Σ H_ε(-φ) · dx · dy
    loss = (area_sim - obs_area)²

    Parameters
    ----------
    phi : jnp.ndarray
        (ny, nx) level set field.
    obs_area : float
        Observed burned area in coordinate units².
    dx, dy : float
        Grid spacing.
    epsilon : float
        Smoothing width for Heaviside.

    Returns
    -------
    jnp.ndarray
        Scalar loss value.
    """
    H = 0.5 * (1.0 + jnp.tanh(-phi / epsilon))
    area_sim = jnp.sum(H) * dx * dy
    return (area_sim - obs_area) ** 2


def composite_loss(
    phi: "jnp.ndarray",
    obs_points_xy: "jnp.ndarray",
    obs_area: float,
    x_coords: "jnp.ndarray",
    y_coords: "jnp.ndarray",
    w_shape: float = 1.0,
    w_area: float = 0.5,
    epsilon: float = 1.0,
) -> "jnp.ndarray":
    """
    Weighted composite of perimeter and area losses.

    Parameters
    ----------
    w_shape : float
        Weight for phi_perimeter_loss.
    w_area : float
        Weight for phi_area_loss.

    Returns
    -------
    jnp.ndarray
        Scalar composite loss.
    """
    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])

    l_shape = phi_perimeter_loss(phi, obs_points_xy, x_coords, y_coords)
    l_area = phi_area_loss(phi, obs_area, dx, dy, epsilon=epsilon)

    return w_shape * l_shape + w_area * l_area


# =============================================================================
# Calibration runner
# =============================================================================


@dataclass
class CalibrationResult:
    """Result of a calibration run."""

    params: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    converged: bool = False


@dataclass
class _CalibrationContext:
    """Pre-computed data shared across calibration iterations."""

    param_grid: "FireParameterGrid"
    ignition_x: float
    ignition_y: float
    is_geographic: bool
    center_latitude: float | None
    sim_config: Any
    obs_xy: np.ndarray  # main polygon for shape metrics
    obs_polygons: list  # all polygons for rasterization
    obs_area_ha: float
    raster_bounds: tuple[float, float, float, float]
    # Spotting support
    fuel_grid: np.ndarray | None = None
    fuel_lookup: dict | None = None
    hourly_isi: np.ndarray | None = None  # (nt,) ISI per timestep
    hourly_ws: np.ndarray | None = None   # (nt,) wind speed per timestep


def _load_observed_perimeter(
    path: str,
    target_crs: str = "EPSG:4326",
    filter_main_complex: bool = False,
    min_area_fraction: float = 0.01,
) -> tuple[np.ndarray, float]:
    """
    Load observed fire perimeter from shapefile/GeoJSON.

    Handles MultiPolygon (dissolves), CRS reprojection, and area extraction.

    Parameters
    ----------
    path : str
        Path to shapefile/GeoJSON.
    target_crs : str
        Target CRS for reprojection.
    filter_main_complex : bool
        If True, discard small disconnected polygons that are less than
        ``min_area_fraction`` of the largest polygon's area. This removes
        tiny fragments and separate spot fires that a smooth level set
        solver cannot reproduce, improving Jaccard scores.
    min_area_fraction : float
        Minimum area as a fraction of the largest polygon (default 1%).
        Only used when ``filter_main_complex`` is True.

    Returns
    -------
    obs_xy : np.ndarray
        (N, 2) perimeter vertices in target_crs.
    area_ha : float
        Burned area in hectares (from shapefile attribute or computed).
    all_polygons : list of np.ndarray
        All polygon coordinate arrays for rasterization.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    gdf = gpd.read_file(path)

    # Extract area from attribute if available
    area_ha = None
    for col in ("SIZE_HA", "CALC_HA", "AREA_HA"):
        if col in gdf.columns and gdf[col].iloc[0] is not None:
            area_ha = float(gdf[col].iloc[0])
            break

    # Reproject to target CRS if needed
    if gdf.crs is not None and str(gdf.crs) != target_crs:
        logger.info(f"Reprojecting observed perimeter from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # Handle MultiPolygon — collect ALL polygons for rasterization
    geom = gdf.geometry.iloc[0]
    if geom.geom_type == "MultiPolygon":
        merged = unary_union(geom)
        if merged.geom_type == "MultiPolygon":
            polygons = list(merged.geoms)
            logger.info(f"Observed perimeter has {len(polygons)} polygon parts")
        else:
            polygons = [merged]
    else:
        polygons = [geom]

    # Filter to main fire complex: drop tiny fragments
    if filter_main_complex and len(polygons) > 1:
        largest_area = max(p.area for p in polygons)
        threshold = largest_area * min_area_fraction
        filtered = [p for p in polygons if p.area >= threshold]
        n_dropped = len(polygons) - len(filtered)
        if n_dropped > 0:
            dropped_area = sum(p.area for p in polygons if p.area < threshold)
            total_area = sum(p.area for p in polygons)
            logger.info(
                f"Filtered observed perimeter: kept {len(filtered)}/{len(polygons)} polygons "
                f"(dropped {n_dropped} fragments, {dropped_area/total_area*100:.1f}% of total area)"
            )
            polygons = filtered

    # Use largest polygon for shape metrics (centroid, orientation, Procrustes)
    largest = max(polygons, key=lambda g: g.area)
    main_coords = np.array(largest.exterior.coords)[:, :2]

    # Collect ALL polygon coords for rasterization (Jaccard/Sorensen)
    all_polygons = [np.array(p.exterior.coords)[:, :2] for p in polygons]

    # If no area from attribute, compute from all polygons
    if area_ha is None:
        area_ha = sum(_polygon_area(c) for c in all_polygons) * 1e-4

    return main_coords, area_ha, all_polygons


def _polygon_area(xy: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    x, y = xy[:, 0], xy[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _build_calibration_context(
    config_path: str,
    obs_perimeter_path: str,
    config_overrides: dict | None = None,
) -> _CalibrationContext:
    """
    Build everything needed for calibration: param_grid, ignition, observed data.

    This is the expensive step (terrain + weather + FBP). Done once.

    Parameters
    ----------
    config_overrides : dict, optional
        Nested dict of config overrides applied after loading from file.
        E.g. {"fbp": {"terrain_wind_deflection": 0.3}}
    """
    from ignacio.config import load_config
    from ignacio.simulation import build_parameter_grid
    from ignacio.terrain import build_terrain_grids
    from ignacio.weather import load_weather_data, process_fire_weather
    from ignacio.ignition import generate_ignitions

    config = load_config(config_path)

    # Apply overrides
    if config_overrides:
        for section, values in config_overrides.items():
            sub = getattr(config, section, None)
            if sub is not None:
                for key, val in values.items():
                    setattr(sub, key, val)

    # 1. Terrain
    logger.info("Building terrain grids...")
    terrain = build_terrain_grids(config)

    # 2. Weather
    logger.info("Processing weather...")
    rng = np.random.default_rng(config.project.random_seed)
    hourly_data = None
    if config.simulation.time_varying_weather:
        try:
            hourly_data = load_weather_data(config)
            if len(hourly_data) > 0 and "HOUR" in hourly_data.columns:
                logger.info(f"Loaded {len(hourly_data)} hourly records")
            else:
                hourly_data = None
        except Exception as e:
            logger.warning(f"Could not load hourly data: {e}")

    weather = process_fire_weather(config, rng)

    # Enrich hourly data with FWI
    if hourly_data is not None and "DATE" in hourly_data.columns:
        import pandas as pd
        daily_fwi = weather.records
        if "FFMC" in daily_fwi.columns and "BUI" in daily_fwi.columns:
            try:
                from ignacio.fwi import calculate_isi
                date_to_ffmc = dict(zip(
                    pd.to_datetime(daily_fwi["DATE"]).dt.date, daily_fwi["FFMC"],
                ))
                date_to_bui = dict(zip(
                    pd.to_datetime(daily_fwi["DATE"]).dt.date, daily_fwi["BUI"],
                ))
                h_dates = pd.to_datetime(hourly_data["DATE"]).dt.date
                hourly_data["BUI"] = [date_to_bui.get(d, np.nan) for d in h_dates]
                ws_col = "WIND_SPEED" if "WIND_SPEED" in hourly_data.columns else "WS"
                hourly_ffmc = np.array([date_to_ffmc.get(d, np.nan) for d in h_dates])
                hourly_ws = hourly_data[ws_col].values.astype(float)
                hourly_isi = np.array([
                    calculate_isi(f, w) if np.isfinite(f) and np.isfinite(w) else np.nan
                    for f, w in zip(hourly_ffmc, hourly_ws)
                ])
                hourly_data["ISI"] = hourly_isi
                hourly_data["FFMC"] = hourly_ffmc
            except Exception as e:
                logger.warning(f"Could not enrich hourly data: {e}")

    # 3. Build param_grid
    logger.info("Building parameter grid...")
    param_grid = build_parameter_grid(config, terrain, weather, hourly_data=hourly_data)

    # 4. Ignition + fuel grid for spotting
    from ignacio.io import read_raster_int
    fuel_raster = read_raster_int(config.fuel.path)
    ignitions = generate_ignitions(config, fuel_raster, rng, terrain_crs=str(terrain.crs))
    ignition = ignitions.points[0]

    # Resample fuel grid to terrain shape (needed for spotting)
    fuel_grid_raw = fuel_raster.data
    terrain_shape = terrain.slope_deg.shape
    if fuel_grid_raw.shape != terrain_shape:
        from scipy.ndimage import zoom as _zoom
        _zy = terrain_shape[0] / fuel_grid_raw.shape[0]
        _zx = terrain_shape[1] / fuel_grid_raw.shape[1]
        fuel_grid_resampled = _zoom(fuel_grid_raw, (_zy, _zx), order=0)
    else:
        fuel_grid_resampled = fuel_grid_raw

    # Build per-timestep ISI and wind speed arrays for spotting
    n_timesteps = param_grid.ros.shape[0]
    ts_isi = np.full(n_timesteps, np.nan)
    ts_ws = np.full(n_timesteps, np.nan)
    if hourly_data is not None and "ISI" in hourly_data.columns:
        import pandas as pd
        ws_col = "WIND_SPEED" if "WIND_SPEED" in hourly_data.columns else "WS"
        isi_vals = hourly_data["ISI"].values.astype(float)
        ws_vals = hourly_data[ws_col].values.astype(float)
        for t in range(min(n_timesteps, len(isi_vals))):
            ts_isi[t] = isi_vals[t] if np.isfinite(isi_vals[t]) else 0.0
            ts_ws[t] = ws_vals[t] if np.isfinite(ws_vals[t]) else 0.0
    # Fill remaining with representative weather
    rep_isi = weather.records["ISI"].median() if "ISI" in weather.records.columns else 5.0
    rep_ws_col = "WIND_SPEED" if "WIND_SPEED" in weather.records.columns else "WS"
    rep_ws = weather.records[rep_ws_col].median() if rep_ws_col in weather.records.columns else 15.0
    ts_isi[np.isnan(ts_isi)] = float(rep_isi)
    ts_ws[np.isnan(ts_ws)] = float(rep_ws)

    # 5. Geographic detection
    is_geographic = False
    center_latitude = None
    if terrain.crs is not None:
        try:
            import pyproj
            crs = pyproj.CRS.from_user_input(terrain.crs)
            is_geographic = crs.is_geographic
        except Exception:
            if -180 <= param_grid.x_min <= 180 and -180 <= param_grid.x_max <= 180:
                is_geographic = True
    if is_geographic:
        center_latitude = (param_grid.y_min + param_grid.y_max) / 2.0

    # 6. Load observed perimeter (reproject to simulation CRS)
    target_crs = str(terrain.crs) if terrain.crs else "EPSG:4326"
    obs_xy, obs_area_ha, obs_polygons = _load_observed_perimeter(
        obs_perimeter_path,
        target_crs=target_crs,
        filter_main_complex=config.calibration.filter_main_complex,
        min_area_fraction=config.calibration.min_area_fraction,
    )

    # Raster bounds for metric computation — tight around all observed polygons + buffer
    all_obs_pts = np.concatenate(obs_polygons, axis=0)
    _buf = 0.5  # degrees buffer around observed perimeter
    raster_bounds = (
        float(all_obs_pts[:, 0].min()) - _buf,
        float(all_obs_pts[:, 1].min()) - _buf,
        float(all_obs_pts[:, 0].max()) + _buf,
        float(all_obs_pts[:, 1].max()) + _buf,
    )

    logger.info(f"Observed area: {obs_area_ha:.0f} ha")
    logger.info(f"Ignition: ({ignition.x:.6f}, {ignition.y:.6f})")
    logger.info(f"Grid shape: {param_grid.ros.shape}")

    return _CalibrationContext(
        param_grid=param_grid,
        ignition_x=ignition.x,
        ignition_y=ignition.y,
        is_geographic=is_geographic,
        center_latitude=center_latitude,
        sim_config=config.simulation,
        obs_xy=obs_xy,
        obs_polygons=obs_polygons,
        obs_area_ha=obs_area_ha,
        raster_bounds=raster_bounds,
        fuel_grid=fuel_grid_resampled,
        fuel_lookup=config.fuel.fuel_lookup,
        hourly_isi=ts_isi,
        hourly_ws=ts_ws,
    )


def _run_scaled_simulation(
    ctx: _CalibrationContext,
    scales: dict[str, float],
    max_steps: int | None = None,
) -> "FirePerimeterHistory":
    """
    Run a single level set simulation with scaled ROS/BROS/FROS.

    Parameters
    ----------
    ctx : _CalibrationContext
        Pre-built context with param_grid, ignition, etc.
    scales : dict
        Calibration parameters. Supported keys:
        - ros_scale: multiplier on ROS (default 1.0)
        - backing_scale: multiplier on BROS (default 1.0)
        - flanking_scale: multiplier on FROS (default 1.0)
    max_steps : int, optional
        Override maximum simulation steps. If None, uses config max_duration/dt.

    Returns
    -------
    FirePerimeterHistory
    """
    from ignacio.levelset import simulate_fire_spread_levelset
    from ignacio.spread import FireParameterGrid

    pg = ctx.param_grid
    ros_scale = scales.get("ros_scale", 1.0)
    backing_scale = scales.get("backing_scale", ros_scale)
    flanking_scale = scales.get("flanking_scale", ros_scale)
    wind_direction_offset = scales.get("wind_direction_offset", 0.0)  # degrees

    # Apply scale factors
    scaled_ros = pg.ros * ros_scale
    scaled_bros = pg.bros * backing_scale
    scaled_fros = pg.fros * flanking_scale

    # Rotate RAZ by wind direction offset (convert degrees → radians)
    scaled_raz = pg.raz
    if abs(wind_direction_offset) > 0.01:
        scaled_raz = pg.raz + np.radians(wind_direction_offset)

    scaled_pg = FireParameterGrid(
        x_coords=pg.x_coords,
        y_coords=pg.y_coords,
        ros=scaled_ros,
        bros=scaled_bros,
        fros=scaled_fros,
        raz=scaled_raz,
    )

    sim = ctx.sim_config
    ls = sim.level_set

    if max_steps is None:
        max_steps = int(sim.max_duration / sim.dt)

    # Build spotting config if fuel data is available
    spotting_config = None
    fuel_grid = None
    fuel_lookup = None
    if ctx.fuel_grid is not None and ctx.fuel_lookup is not None:
        spotting_config = {
            "interval": 1,
            "seed": 42,
            "min_spot_distance": 500.0,
            "max_spot_distance": 3000.0,
            "isi_threshold": 6.0,
            "spot_probability": 0.10,
            # Per-timestep arrays for time-varying weather
            "isi_array": ctx.hourly_isi,
            "ws_array": ctx.hourly_ws,
        }
        fuel_grid = ctx.fuel_grid
        fuel_lookup = ctx.fuel_lookup

    history = simulate_fire_spread_levelset(
        param_grid=scaled_pg,
        x_ignition=ctx.ignition_x,
        y_ignition=ctx.ignition_y,
        dt=sim.dt,
        initial_radius=sim.initial_radius,
        store_every=sim.store_every,
        max_steps=max_steps,
        min_ros=sim.min_ros,
        is_geographic=ctx.is_geographic,
        center_latitude=ctx.center_latitude,
        cfl_factor=ls.cfl_factor,
        reinit_interval=ls.reinit_interval,
        reinit_iters=ls.reinit_iters,
        narrow_band_width=getattr(ls, "narrow_band_width", None),
        spotting_config=spotting_config,
        fuel_grid=fuel_grid,
        fuel_lookup=fuel_lookup,
    )
    return history


def evaluate_simulation(
    ctx: _CalibrationContext,
    scales: dict[str, float],
    resolution: float = 0.001,
    max_steps: int | None = None,
) -> dict[str, float]:
    """
    Run simulation with given scales and compute all evaluation metrics.

    Parameters
    ----------
    ctx : _CalibrationContext
    scales : dict
        Parameter scales (e.g. {"ros_scale": 0.5}).
    resolution : float
        Rasterization resolution for Jaccard/Sorensen (in CRS units).
    max_steps : int, optional
        Override maximum simulation steps.

    Returns
    -------
    dict
        Metric name → value.
    """
    history = _run_scaled_simulation(ctx, scales, max_steps=max_steps)

    if len(history.perimeters) < 2:
        logger.warning("Simulation produced fewer than 2 perimeters")
        return {"jaccard": 0.0, "sorensen": 0.0, "error": "no_spread"}

    x_final, y_final = history.get_final_perimeter()
    sim_xy = np.column_stack([x_final, y_final])
    obs_xy = ctx.obs_xy
    origin = np.array([ctx.ignition_x, ctx.ignition_y])

    # Rasterize both perimeters for Jaccard/Sorensen and area computation
    # Use all observed polygons (MultiPolygon) for accurate comparison
    sim_mask = _rasterize_perimeter(sim_xy, ctx.raster_bounds, resolution)
    obs_mask = _rasterize_perimeter(ctx.obs_polygons, ctx.raster_bounds, resolution)

    intersection = np.sum(sim_mask & obs_mask)
    union = np.sum(sim_mask | obs_mask)
    j_val = float(intersection / union) if union > 0 else 0.0
    s_val = float(2 * intersection / (np.sum(sim_mask) + np.sum(obs_mask))) if (np.sum(sim_mask) + np.sum(obs_mask)) > 0 else 0.0

    # Raster-based area (more accurate than contour polygon for complex shapes)
    import math
    if ctx.is_geographic and ctx.center_latitude is not None:
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(ctx.center_latitude))
        cell_area_ha = resolution * m_per_deg_lon * resolution * m_per_deg_lat / 1e4
    else:
        cell_area_ha = resolution * resolution / 1e4
    sim_area_ha = float(np.sum(sim_mask)) * cell_area_ha
    obs_area_ha_raster = float(np.sum(obs_mask)) * cell_area_ha

    metrics = {
        "jaccard": j_val,
        "sorensen": s_val,
        "centroid_size_ratio": centroid_size_ratio(sim_xy, obs_xy, origin=origin),
        "procrustes_rmse": procrustes_shape_rmse(sim_xy, obs_xy, origin=origin),
        "orientation_error": orientation_error(sim_xy, obs_xy, origin=origin),
        "sim_area_ha": sim_area_ha,
        "obs_area_ha": ctx.obs_area_ha,
        "obs_area_ha_raster": obs_area_ha_raster,
        "area_ratio": sim_area_ha / max(ctx.obs_area_ha, 1.0),
        "n_perimeters": len(history.perimeters),
    }
    return metrics


def run_ensemble_simulation(
    ctx: _CalibrationContext,
    base_scales: dict[str, float],
    n_members: int = 20,
    wd_perturbation: float = 20.0,
    ws_perturbation_frac: float = 0.2,
    resolution: float = 0.001,
    max_steps: int | None = None,
    burn_probability_threshold: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run an ensemble of simulations with perturbed wind parameters.

    Each ensemble member adds random perturbations to wind direction
    (±wd_perturbation degrees) and wind speed (±ws_perturbation_frac).
    The result is a burn probability map from which a thresholded
    perimeter can be extracted.

    Parameters
    ----------
    ctx : _CalibrationContext
        Pre-built calibration context.
    base_scales : dict
        Base parameter scales.
    n_members : int
        Number of ensemble members.
    wd_perturbation : float
        Wind direction perturbation std (degrees).
    ws_perturbation_frac : float
        Wind speed perturbation as fraction of base (e.g. 0.2 = ±20%).
    resolution : float
        Rasterization resolution for metrics.
    max_steps : int, optional
        Max simulation steps.
    burn_probability_threshold : float
        Threshold for burn probability map (0.5 = majority vote).
    seed : int
        Random seed.

    Returns
    -------
    dict
        'burn_probability': np.ndarray, 'metrics': dict, 'ensemble_jaccard': list
    """
    rng = np.random.default_rng(seed)

    # Accumulate burn counts
    burn_count = None
    ensemble_jaccards = []

    obs_mask = _rasterize_perimeter(ctx.obs_polygons, ctx.raster_bounds, resolution)

    for i in range(n_members):
        # Perturb wind direction offset
        scales = dict(base_scales)
        wd_offset = scales.get("wind_direction_offset", 0.0)
        scales["wind_direction_offset"] = wd_offset + rng.normal(0, wd_perturbation)

        # Perturb ROS scale (proxy for wind speed perturbation)
        ros_scale = scales.get("ros_scale", 1.0)
        ws_factor = 1.0 + rng.normal(0, ws_perturbation_frac)
        ws_factor = max(0.3, min(2.0, ws_factor))
        scales["ros_scale"] = ros_scale * ws_factor

        logger.info(f"Ensemble member {i+1}/{n_members}: WD_offset={scales['wind_direction_offset']:.1f}°, "
                     f"ros_scale={scales['ros_scale']:.3f}")

        try:
            history = _run_scaled_simulation(ctx, scales, max_steps=max_steps)
            if len(history.perimeters) < 2:
                continue

            x_final, y_final = history.get_final_perimeter()
            sim_xy = np.column_stack([x_final, y_final])
            sim_mask = _rasterize_perimeter(sim_xy, ctx.raster_bounds, resolution)

            if burn_count is None:
                burn_count = np.zeros_like(sim_mask, dtype=np.float32)
            burn_count += sim_mask.astype(np.float32)

            # Per-member Jaccard
            intersection = np.sum(sim_mask & obs_mask)
            union = np.sum(sim_mask | obs_mask)
            j = float(intersection / union) if union > 0 else 0.0
            ensemble_jaccards.append(j)
            logger.info(f"  Member {i+1} Jaccard: {j:.3f}")

        except Exception as e:
            logger.warning(f"Ensemble member {i+1} failed: {e}")
            continue

    if burn_count is None:
        return {"burn_probability": None, "metrics": {"jaccard": 0.0}, "ensemble_jaccards": []}

    n_valid = max(len(ensemble_jaccards), 1)
    burn_probability = burn_count / n_valid

    # Threshold to get ensemble perimeter
    ensemble_mask = burn_probability >= burn_probability_threshold

    intersection = np.sum(ensemble_mask & obs_mask)
    union = np.sum(ensemble_mask | obs_mask)
    j_ensemble = float(intersection / union) if union > 0 else 0.0
    s_ensemble = float(2 * intersection / (np.sum(ensemble_mask) + np.sum(obs_mask))) if (np.sum(ensemble_mask) + np.sum(obs_mask)) > 0 else 0.0

    import math
    if ctx.is_geographic and ctx.center_latitude is not None:
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(ctx.center_latitude))
        cell_area_ha = resolution * m_per_deg_lon * resolution * m_per_deg_lat / 1e4
    else:
        cell_area_ha = resolution * resolution / 1e4
    ensemble_area_ha = float(np.sum(ensemble_mask)) * cell_area_ha

    metrics = {
        "jaccard": j_ensemble,
        "sorensen": s_ensemble,
        "ensemble_area_ha": ensemble_area_ha,
        "obs_area_ha": ctx.obs_area_ha,
        "area_ratio": ensemble_area_ha / max(ctx.obs_area_ha, 1.0),
        "n_members": n_valid,
        "mean_member_jaccard": float(np.mean(ensemble_jaccards)) if ensemble_jaccards else 0.0,
        "std_member_jaccard": float(np.std(ensemble_jaccards)) if ensemble_jaccards else 0.0,
    }

    logger.info(
        f"Ensemble result: Jaccard={j_ensemble:.3f}, Sorensen={s_ensemble:.3f}, "
        f"area={ensemble_area_ha:.0f} ha (ratio={metrics['area_ratio']:.2f}), "
        f"mean member Jaccard={metrics['mean_member_jaccard']:.3f} ± {metrics['std_member_jaccard']:.3f}"
    )

    return {
        "burn_probability": burn_probability,
        "metrics": metrics,
        "ensemble_jaccards": ensemble_jaccards,
    }


def calibrate_parameters(
    config_path: str,
    obs_perimeter_path: str,
    params_to_calibrate: list[str] | None = None,
    method: str = "adam",
    n_iters: int = 30,
    learning_rate: float = 0.01,
    w_jaccard: float = 1.0,
    w_size: float = 0.5,
    w_shape: float = 0.3,
    w_area: float = 0.5,
    w_perimeter: float = 1.0,
    resolution: float = 0.001,
    max_steps: int | None = None,
) -> CalibrationResult:
    """
    Calibrate fire spread parameters against observed perimeter.

    Parameters
    ----------
    config_path : str
        Path to ignacio YAML config file.
    obs_perimeter_path : str
        Path to observed fire perimeter (shapefile/GeoJSON).
    params_to_calibrate : list of str, optional
        Parameter names. Default: ['ros_scale'].
        Supported: 'ros_scale', 'backing_scale', 'flanking_scale'.
    method : str
        'adam' for gradient-based (JAX), 'nelder-mead' for gradient-free.
    n_iters : int
        Max optimization iterations.
    learning_rate : float
        Adam learning rate (only for method='adam').
    w_jaccard : float
        Weight for (1 - Jaccard) in gradient-free loss.
    w_size : float
        Weight for |centroid_size_ratio| in gradient-free loss.
    w_shape : float
        Weight for Procrustes RMSE in gradient-free loss.
    w_area : float
        Weight for area mismatch in Adam loss.
    w_perimeter : float
        Weight for phi-perimeter loss in Adam loss.
    resolution : float
        Rasterization resolution for Jaccard/Sorensen.
    max_steps : int, optional
        Override maximum simulation steps.

    Returns
    -------
    CalibrationResult
    """
    if params_to_calibrate is None:
        params_to_calibrate = ["ros_scale"]

    # Build context (expensive — done once)
    ctx = _build_calibration_context(config_path, obs_perimeter_path)

    if method == "adam":
        if not _HAS_JAX:
            logger.warning("JAX not available, falling back to nelder-mead")
            method = "nelder-mead"
        else:
            return _calibrate_adam(
                ctx, params_to_calibrate, n_iters, learning_rate,
                w_perimeter, w_area, resolution,
                max_steps=max_steps,
            )

    if method == "two-stage":
        return _calibrate_two_stage(
            ctx, params_to_calibrate, n_iters,
            w_jaccard, w_size, w_shape, resolution,
            max_steps=max_steps,
        )

    return _calibrate_gradient_free(
        ctx, params_to_calibrate, n_iters,
        w_jaccard, w_size, w_shape, resolution,
        max_steps=max_steps,
    )


def _calibrate_gradient_free(
    ctx, param_names, n_iters,
    w_jaccard, w_size, w_shape, resolution,
    max_steps=None,
):
    """Gradient-free calibration using Nelder-Mead."""
    from scipy.optimize import minimize

    # Parameter-specific initial values and bounds
    _param_info = {
        "ros_scale":              {"x0": 1.0, "lo": 0.05, "hi": 5.0},
        "backing_scale":          {"x0": 1.0, "lo": 0.05, "hi": 5.0},
        "flanking_scale":         {"x0": 1.0, "lo": 0.05, "hi": 5.0},
        "wind_direction_offset":  {"x0": 0.0, "lo": -180.0, "hi": 180.0},
    }

    loss_history = []
    eval_count = [0]

    def _clamp(name, val):
        info = _param_info.get(name, {"lo": 0.05, "hi": 5.0})
        return float(np.clip(val, info["lo"], info["hi"]))

    def objective(x):
        scales = {k: _clamp(k, float(v)) for k, v in zip(param_names, x)}
        eval_count[0] += 1

        logger.info(f"[Eval {eval_count[0]}] params = {scales}")

        try:
            metrics = evaluate_simulation(ctx, scales, resolution=resolution,
                                          max_steps=max_steps)
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            return 1e6

        j = metrics.get("jaccard", 0.0)
        c = abs(metrics.get("centroid_size_ratio", 0.0))
        p = metrics.get("procrustes_rmse", 0.0)
        o = metrics.get("orientation_error", 0.0) / 180.0  # normalize to [0,1]
        area_ratio = metrics.get("area_ratio", 1.0)

        # Area mismatch: |ln(area_ratio)|² penalizes both over/under
        area_penalty = np.log(max(area_ratio, 0.01)) ** 2

        loss = (w_jaccard * (1.0 - j) + w_size * c + w_shape * p
                + 2.0 * area_penalty + 0.5 * o)
        loss_history.append(float(loss))

        logger.info(
            f"  → Jaccard={j:.3f}, area_ratio={area_ratio:.2f}, "
            f"orient={metrics.get('orientation_error', 0):.1f}°, "
            f"Procrustes={p:.3f}, loss={loss:.4f}"
        )
        return loss

    x0 = np.array([_param_info.get(k, {"x0": 1.0})["x0"] for k in param_names])
    bounds_opt = [(_param_info.get(k, {"lo": 0.05})["lo"],
                   _param_info.get(k, {"hi": 5.0})["hi"]) for k in param_names]

    if len(param_names) == 1:
        from scipy.optimize import minimize_scalar
        result_opt = minimize_scalar(
            lambda x: objective(np.array([x])),
            bounds=bounds_opt[0],
            method="bounded",
            options={"maxiter": n_iters, "xatol": 0.02},
        )
        params_out = {param_names[0]: _clamp(param_names[0], float(result_opt.x))}
    else:
        # Build initial simplex with parameter-appropriate step sizes.
        # Default Nelder-Mead uses 5% of x0, which is far too small for
        # wind_direction_offset (where steps of ~45-90° are needed).
        _step_sizes = {
            "ros_scale": 0.3,
            "backing_scale": 0.3,
            "flanking_scale": 0.3,
            "wind_direction_offset": 60.0,  # degrees
        }
        n = len(param_names)
        simplex = np.tile(x0, (n + 1, 1))
        for i, k in enumerate(param_names):
            simplex[i + 1, i] += _step_sizes.get(k, 0.3)

        result_opt = minimize(
            objective, x0,
            method="Nelder-Mead",
            options={"maxiter": n_iters, "fatol": 0.01,
                     "adaptive": True, "initial_simplex": simplex},
        )
        params_out = {k: _clamp(k, float(v)) for k, v in zip(param_names, result_opt.x)}

    # Final evaluation with best params
    logger.info(f"Optimal params: {params_out}")
    logger.info("Running final evaluation...")
    final_metrics = evaluate_simulation(ctx, params_out, resolution=resolution,
                                        max_steps=max_steps)

    return CalibrationResult(
        params=params_out,
        loss_history=loss_history,
        metrics=final_metrics,
        converged=bool(result_opt.success),
    )


def _calibrate_two_stage(
    ctx, param_names, n_iters,
    w_jaccard, w_size, w_shape, resolution,
    max_steps=None,
    wd_grid_step=15.0,
):
    """
    Two-stage calibration: grid search WD offset, then Nelder-Mead on remaining params.

    Stage 1: Test wind_direction_offset from -180° to 180° at ``wd_grid_step`` intervals.
             Pick the offset with the lowest loss.
    Stage 2: Fix WD offset at best value, run Nelder-Mead on the remaining parameters
             (ros_scale, backing_scale, flanking_scale).
    """
    from scipy.optimize import minimize

    has_wd = "wind_direction_offset" in param_names
    other_params = [p for p in param_names if p != "wind_direction_offset"]

    _param_info = {
        "ros_scale":              {"x0": 1.0, "lo": 0.05, "hi": 5.0},
        "backing_scale":          {"x0": 1.0, "lo": 0.05, "hi": 5.0},
        "flanking_scale":         {"x0": 1.0, "lo": 0.05, "hi": 5.0},
        "wind_direction_offset":  {"x0": 0.0, "lo": -180.0, "hi": 180.0},
    }

    loss_history = []
    eval_count = [0]

    def _clamp(name, val):
        info = _param_info.get(name, {"lo": 0.05, "hi": 5.0})
        return float(np.clip(val, info["lo"], info["hi"]))

    def _objective(x, fixed_params=None):
        if fixed_params is None:
            fixed_params = {}
        active_params = list(other_params) if has_wd and "wind_direction_offset" in fixed_params else list(param_names)
        scales = dict(fixed_params)
        for k, v in zip(active_params, x):
            scales[k] = _clamp(k, float(v))
        eval_count[0] += 1

        logger.info(f"[Eval {eval_count[0]}] params = {scales}")

        try:
            metrics = evaluate_simulation(ctx, scales, resolution=resolution,
                                          max_steps=max_steps)
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            return 1e6

        j = metrics.get("jaccard", 0.0)
        c = abs(metrics.get("centroid_size_ratio", 0.0))
        p = metrics.get("procrustes_rmse", 0.0)
        o = metrics.get("orientation_error", 0.0) / 180.0
        area_ratio = metrics.get("area_ratio", 1.0)
        area_penalty = np.log(max(area_ratio, 0.01)) ** 2

        loss = (w_jaccard * (1.0 - j) + w_size * c + w_shape * p
                + 2.0 * area_penalty + 0.5 * o)
        loss_history.append(float(loss))

        logger.info(
            f"  → Jaccard={j:.3f}, area_ratio={area_ratio:.2f}, "
            f"orient={metrics.get('orientation_error', 0):.1f}°, loss={loss:.4f}"
        )
        return loss

    # --- Stage 1: Grid search over wind_direction_offset ---
    best_wd = 0.0
    if has_wd:
        logger.info("=== Stage 1: Grid search over wind_direction_offset ===")
        wd_values = np.arange(-180.0, 180.0, wd_grid_step)
        default_other = {k: _param_info[k]["x0"] for k in other_params}
        best_loss = 1e9
        for wd in wd_values:
            scales = dict(default_other)
            scales["wind_direction_offset"] = float(wd)
            eval_count[0] += 1
            logger.info(f"[Grid {eval_count[0]}] WD offset = {wd:.0f}°")
            try:
                metrics = evaluate_simulation(ctx, scales, resolution=resolution,
                                              max_steps=max_steps)
            except Exception as e:
                logger.warning(f"Simulation failed at WD={wd:.0f}°: {e}")
                continue

            j = metrics.get("jaccard", 0.0)
            area_ratio = metrics.get("area_ratio", 1.0)
            o = metrics.get("orientation_error", 0.0) / 180.0
            area_penalty = np.log(max(area_ratio, 0.01)) ** 2
            loss = (w_jaccard * (1.0 - j) + 2.0 * area_penalty + 0.5 * o)
            loss_history.append(float(loss))
            logger.info(f"  → Jaccard={j:.3f}, area_ratio={area_ratio:.2f}, loss={loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                best_wd = float(wd)

        logger.info(f"Stage 1 result: best WD offset = {best_wd:.0f}° (loss={best_loss:.4f})")

    # --- Stage 2: Nelder-Mead on remaining params with WD fixed ---
    if other_params:
        logger.info(f"=== Stage 2: Nelder-Mead on {other_params} with WD={best_wd:.0f}° ===")
        fixed = {"wind_direction_offset": best_wd} if has_wd else {}
        x0_stage2 = np.array([_param_info.get(k, {"x0": 1.0})["x0"] for k in other_params])

        _step_sizes = {
            "ros_scale": 0.3, "backing_scale": 0.3, "flanking_scale": 0.3,
        }
        n = len(other_params)
        simplex = np.tile(x0_stage2, (n + 1, 1))
        for i, k in enumerate(other_params):
            simplex[i + 1, i] += _step_sizes.get(k, 0.3)

        result_opt = minimize(
            lambda x: _objective(x, fixed_params=fixed),
            x0_stage2,
            method="Nelder-Mead",
            options={"maxiter": n_iters, "fatol": 0.01,
                     "adaptive": True, "initial_simplex": simplex},
        )
        params_out = {k: _clamp(k, float(v)) for k, v in zip(other_params, result_opt.x)}
        if has_wd:
            params_out["wind_direction_offset"] = best_wd
    else:
        params_out = {"wind_direction_offset": best_wd} if has_wd else {}

    logger.info(f"Two-stage optimal params: {params_out}")
    logger.info("Running final evaluation...")
    final_metrics = evaluate_simulation(ctx, params_out, resolution=resolution,
                                        max_steps=max_steps)

    return CalibrationResult(
        params=params_out,
        loss_history=loss_history,
        metrics=final_metrics,
        converged=True,
    )


# =============================================================================
# Adam gradient-based calibration (JAX)
# =============================================================================


def _calibrate_adam(
    ctx: _CalibrationContext,
    param_names: list[str],
    n_iters: int,
    learning_rate: float,
    w_perimeter: float,
    w_area: float,
    resolution: float,
    max_steps: int | None = None,
) -> CalibrationResult:
    """
    Gradient-based calibration using Adam optimizer via JAX.

    Strategy
    --------
    1. Take a *truncated* level set simulation (N substeps via
       ``level_set_step_fixed``) — this keeps the computation graph in
       JAX and allows ``jax.grad`` to flow through.
    2. The loss is ``composite_loss`` (differentiable phi-perimeter +
       smooth-Heaviside area loss).
    3. Adam updates the ``ros_scale`` (and optionally backing_scale /
       flanking_scale) parameters.
    4. After Adam converges, run a full (non-differentiable) simulation
       and report evaluation metrics.
    """
    from ignacio.levelset import (
        initialize_phi,
        level_set_step_fixed,
    )

    pg = ctx.param_grid
    sim = ctx.sim_config
    ls = sim.level_set

    # Geographic unit conversion
    is_geo = ctx.is_geographic
    avg_m_per_deg = 1.0
    if is_geo and ctx.center_latitude is not None:
        import math
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(ctx.center_latitude))
        avg_m_per_deg = math.sqrt(m_per_deg_lat * m_per_deg_lon)

    # --- Crop grid to region around fire for JAX tractability ---
    # Full Jasper grid is 2468×3584 — too large for autodiff.
    # Crop to bounding box of observed perimeter + buffer.
    all_obs_pts = np.concatenate(ctx.obs_polygons, axis=0)
    buf = 0.15  # degrees buffer around observed fire
    crop_xmin = float(all_obs_pts[:, 0].min()) - buf
    crop_xmax = float(all_obs_pts[:, 0].max()) + buf
    crop_ymin = float(all_obs_pts[:, 1].min()) - buf
    crop_ymax = float(all_obs_pts[:, 1].max()) + buf

    # Include ignition point
    crop_xmin = min(crop_xmin, ctx.ignition_x - buf)
    crop_xmax = max(crop_xmax, ctx.ignition_x + buf)
    crop_ymin = min(crop_ymin, ctx.ignition_y - buf)
    crop_ymax = max(crop_ymax, ctx.ignition_y + buf)

    # Find index ranges
    ix_min = max(0, int(np.searchsorted(pg.x_coords, crop_xmin)) - 1)
    ix_max = min(len(pg.x_coords), int(np.searchsorted(pg.x_coords, crop_xmax)) + 1)
    # Handle y_coords that may be ascending or descending
    if pg.y_coords[0] < pg.y_coords[-1]:
        iy_min = max(0, int(np.searchsorted(pg.y_coords, crop_ymin)) - 1)
        iy_max = min(len(pg.y_coords), int(np.searchsorted(pg.y_coords, crop_ymax)) + 1)
    else:
        iy_min = max(0, int(np.searchsorted(-pg.y_coords, -crop_ymax)) - 1)
        iy_max = min(len(pg.y_coords), int(np.searchsorted(-pg.y_coords, -crop_ymin)) + 1)

    x_crop = pg.x_coords[ix_min:ix_max]
    y_crop = pg.y_coords[iy_min:iy_max]
    logger.info(
        f"Cropping grid for Adam: [{iy_min}:{iy_max}, {ix_min}:{ix_max}] "
        f"= {len(y_crop)}×{len(x_crop)} (from {pg.ny}×{pg.nx})"
    )

    # Downsample the cropped grid for tractable autodiff.
    # Target ~200 cells on longest side to keep memory manageable.
    max_side = max(len(y_crop), len(x_crop))
    ds = max(1, max_side // 200)  # downsample factor
    x_ds = x_crop[::ds]
    y_ds = y_crop[::ds]

    x_coords = jnp.asarray(x_ds)
    y_coords = jnp.asarray(y_ds)
    dx = float(x_coords[1] - x_coords[0])
    dy_raw = float(y_coords[1] - y_coords[0])
    y_ascending = dy_raw > 0
    dy = abs(dy_raw)  # level_set_step expects positive magnitude
    dt = float(sim.dt)

    logger.info(
        f"Downsampled grid: {len(y_ds)}×{len(x_ds)} (ds={ds}x from {len(y_crop)}×{len(x_crop)})"
    )

    # Initialize phi on downsampled grid
    init_radius = float(sim.initial_radius)
    if is_geo:
        init_radius = init_radius / avg_m_per_deg
    # Ensure radius spans at least 3 grid cells for a valid initial fire
    min_radius = 3.0 * max(dx, dy)
    if init_radius < min_radius:
        init_radius = min_radius
    phi0 = initialize_phi(
        x_coords, y_coords,
        ctx.ignition_x, ctx.ignition_y,
        init_radius,
    )

    # Observed perimeter as JAX array for differentiable loss
    obs_pts = jnp.asarray(ctx.obs_xy[:, :2])

    # Observed area in coordinate units²
    obs_area_coord = ctx.obs_area_ha * 1e4  # ha → m²
    if is_geo:
        obs_area_coord = obs_area_coord / (avg_m_per_deg ** 2)

    # Determine how many steps to run inside the differentiable loop.
    if max_steps is None:
        max_steps = int(sim.max_duration / sim.dt)

    # Compute CFL-safe substep count on the downsampled grid.
    _crop_ros = pg.ros[:, iy_min:iy_max, ix_min:ix_max]
    max_ros_mpm = float(np.nanmax(np.abs(_crop_ros)))  # m/min
    max_speed_grid = max_ros_mpm / avg_m_per_deg if is_geo else max_ros_mpm
    # Scale by max possible ros_scale (exp(log(5.0)) = 5.0)
    max_speed_grid *= 5.0
    dmin = min(abs(dx), abs(dy))
    if max_speed_grid > 1e-12:
        cfl_dt_max = ls.cfl_factor * dmin / max_speed_grid  # minutes
    else:
        cfl_dt_max = dt  # fallback
    del _crop_ros

    # Each diff step advances by 1 simulation timestep (dt minutes).
    # With the coarser grid, CFL dt_max is larger (∝ dx).
    n_substeps_cfl = max(1, int(np.ceil(dt / cfl_dt_max)))
    n_substeps_cfl = min(n_substeps_cfl, 50)

    # Truncated BPTT: sample n_diff_steps from the full simulation.
    # Keep total graph depth (n_diff_steps × n_substeps_cfl) under ~1500.
    max_graph_depth = 1500
    n_diff_steps = min(max_steps, max_graph_depth // max(n_substeps_cfl, 1))
    n_diff_steps = max(1, min(n_diff_steps, 80))

    # Space the diff steps evenly across the full simulation
    steps_per_diff = max(1, max_steps // n_diff_steps)
    t_indices = [min(i * steps_per_diff, pg.nt - 1) for i in range(n_diff_steps)]

    logger.info(
        f"Adam calibration: {n_iters} iters, lr={learning_rate}, "
        f"{n_diff_steps} diff steps (×{steps_per_diff} sim steps, "
        f"n_sub_cfl={n_substeps_cfl}), "
        f"grid={len(y_ds)}×{len(x_ds)}, params={param_names}"
    )

    # --- Adam state ---
    n_params = len(param_names)
    # Initialize as log-space for positivity: scale = exp(theta)
    theta = jnp.zeros(n_params)  # log(1.0) = 0
    m = jnp.zeros(n_params)
    v = jnp.zeros(n_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    loss_history = []
    best_loss = float("inf")
    best_theta = theta

    for iteration in range(n_iters):
        # Current scales
        scales_jax = jnp.exp(theta)
        ros_scale_val = float(scales_jax[0]) if "ros_scale" in param_names else 1.0

        def loss_fn(theta_vec):
            """Differentiable loss through truncated level set simulation."""
            s = jnp.exp(theta_vec)
            ros_s = s[param_names.index("ros_scale")] if "ros_scale" in param_names else 1.0
            back_s = s[param_names.index("backing_scale")] if "backing_scale" in param_names else ros_s
            flank_s = s[param_names.index("flanking_scale")] if "flanking_scale" in param_names else ros_s

            phi = phi0
            for step_i in range(n_diff_steps):
                t_idx = t_indices[step_i]
                # Convert current timestep CROPPED+DOWNSAMPLED slice to JAX (NaN → 0)
                ros_t = jnp.asarray(np.nan_to_num(pg.ros[t_idx, iy_min:iy_max, ix_min:ix_max][::ds, ::ds], nan=0.0)) * ros_s
                bros_t = jnp.asarray(np.nan_to_num(pg.bros[t_idx, iy_min:iy_max, ix_min:ix_max][::ds, ::ds], nan=0.0)) * back_s
                fros_t = jnp.asarray(np.nan_to_num(pg.fros[t_idx, iy_min:iy_max, ix_min:ix_max][::ds, ::ds], nan=0.0)) * flank_s
                raz_t = jnp.asarray(np.nan_to_num(pg.raz[t_idx, iy_min:iy_max, ix_min:ix_max][::ds, ::ds], nan=0.0))

                if is_geo:
                    ros_t = ros_t / avg_m_per_deg
                    bros_t = bros_t / avg_m_per_deg
                    fros_t = fros_t / avg_m_per_deg

                phi = level_set_step_fixed(
                    phi, ros_t, bros_t, fros_t, raz_t,
                    dx, dy, dt,
                    cfl_factor=ls.cfl_factor,
                    n_substeps=n_substeps_cfl,
                    y_ascending=y_ascending,
                )

            # Composite loss
            loss = composite_loss(
                phi, obs_pts, obs_area_coord,
                x_coords, y_coords,
                w_shape=w_perimeter,
                w_area=w_area,
            )
            return loss

        # Forward + backward
        loss_val, grad = jax.value_and_grad(loss_fn)(theta)
        loss_f = float(loss_val)
        loss_history.append(loss_f)

        if loss_f < best_loss:
            best_loss = loss_f
            best_theta = theta

        # Adam update
        t_adam = iteration + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t_adam)
        v_hat = v / (1 - beta2 ** t_adam)
        theta = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps_adam)

        # Clamp to reasonable range: exp(theta) in [0.01, 5.0]
        theta = jnp.clip(theta, jnp.log(0.01), jnp.log(5.0))

        scales_dict = {k: float(jnp.exp(theta[i])) for i, k in enumerate(param_names)}
        grad_norm = float(jnp.sqrt(jnp.sum(grad ** 2)))
        logger.info(
            f"  [Adam {t_adam:3d}/{n_iters}] loss={loss_f:.6f} "
            f"|grad|={grad_norm:.4f} scales={scales_dict}"
        )

    # Use best parameters
    best_scales = {k: float(jnp.exp(best_theta[i])) for i, k in enumerate(param_names)}
    logger.info(f"Best params (Adam): {best_scales} (loss={best_loss:.6f})")

    # Final full evaluation
    logger.info("Running final full evaluation...")
    final_metrics = evaluate_simulation(
        ctx, best_scales, resolution=resolution, max_steps=max_steps,
    )

    return CalibrationResult(
        params=best_scales,
        loss_history=loss_history,
        metrics=final_metrics,
        converged=True,
    )


def calibrate_from_config(config_path: str) -> CalibrationResult:
    """
    Run calibration using settings from the YAML config file.

    Reads ``calibration:`` section from the config and dispatches
    to ``calibrate_parameters()``.

    Parameters
    ----------
    config_path : str
        Path to ignacio YAML config.

    Returns
    -------
    CalibrationResult
    """
    from ignacio.config import load_config

    config = load_config(config_path)
    cal = config.calibration

    if not cal.enabled:
        raise ValueError("calibration.enabled is False in config")
    if cal.observed_perimeter_path is None:
        raise ValueError("calibration.observed_perimeter_path must be set")

    return calibrate_parameters(
        config_path=config_path,
        obs_perimeter_path=cal.observed_perimeter_path,
        params_to_calibrate=cal.params,
        method=cal.method,
        n_iters=cal.n_iters,
        learning_rate=cal.learning_rate,
        w_jaccard=cal.w_jaccard,
        w_size=cal.w_size,
        w_shape=cal.w_shape,
        w_area=cal.w_area,
        w_perimeter=cal.w_perimeter,
        resolution=cal.resolution,
        max_steps=cal.max_steps,
    )
