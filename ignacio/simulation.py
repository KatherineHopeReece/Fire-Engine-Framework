"""
Fire simulation orchestration for Ignacio.

This module provides the main entry points for running fire growth simulations,
coordinating the terrain, weather, ignition, FBP, and spread components.

Includes simultaneous multiple-ignition marker and level-set paths compatible
with ignacio.spread.FirePerimeterHistory (fronts_by_time/get_fronts).
Single-ignition behavior is preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace, is_dataclass
import copy
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import json
from rasterio.enums import Resampling

# Optional Shapely validity helpers (Shapely 2.0+)
try:
    from shapely.validation import make_valid  # type: ignore
except Exception:  # pragma: no cover
    make_valid = None  # type: ignore

from ignacio.config import IgnacioConfig
from ignacio.fbp import compute_ros, compute_ros_components
from ignacio.ignition import IgnitionPoint, IgnitionSet, generate_ignitions, save_ignitions
from ignacio.io import (
    RasterData,
    read_raster,
    read_raster_int,
    write_raster,
    write_vector,
    read_vector,
    rasterize_geometries,
    reproject_raster,
)
from ignacio.spread import (
    FireParameterGrid,
    FirePerimeterHistory,
    create_initial_perimeter,
    simulate_fire_spread,
)
from ignacio.terrain import TerrainGrids, build_terrain_grids, compute_slope_factor
from ignacio.weather import (
    FireWeatherList,
    get_representative_weather,
    load_weather_data,
    process_fire_weather,
    save_fire_weather_list,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Solver selection and topology-preserving history utilities
# =============================================================================


def _get_spread_method(sim_config) -> str:
    """Return a canonical solver name, compatible with both config versions.

    The modern selector is ``method``. Old configuration files use
    ``spread_method``. Raise on inconsistent choices rather than accidentally
    executing marker physics for a level-set experiment.
    """
    aliases = {
        "marker": "lagrangian",
        "markers": "lagrangian",
        "front": "lagrangian",
        "lagrangian": "lagrangian",
        "levelset": "level_set",
        "level_set": "level_set",
    }
    modern = getattr(sim_config, "method", None)
    legacy = getattr(sim_config, "spread_method", None)
    selected = modern if modern is not None else legacy
    if selected is None:
        raise ValueError("No simulation method configured (method/spread_method)")
    canonical = aliases.get(str(selected).strip().lower())
    if canonical is None:
        raise ValueError(f"Unsupported fire spread method: {selected!r}")
    if modern is not None and legacy is not None:
        other = aliases.get(str(legacy).strip().lower())
        if other != canonical:
            raise ValueError("simulation.method and simulation.spread_method conflict")
    return canonical


def _max_steps(config: IgnacioConfig) -> int:
    """Ceiling avoids inadvertently dropping a final partial timestep."""
    sim = config.simulation
    return max(1, int(np.ceil(float(sim.max_duration) / float(sim.dt))))


def _marker_kwargs(config: IgnacioConfig, *, geographic: bool = False,
                   latitude: float | None = None) -> dict[str, Any]:
    """Forward all H5 topology and numerical controls to ignacio.spread."""
    marker = config.simulation.marker_method
    spacing = marker.resample_spacing
    if geographic and spacing is not None:
        if latitude is None:
            raise ValueError("Latitude required for geographic marker spacing")
        import math
        spacing = float(spacing) / (
            111320.0 * math.sqrt(max(math.cos(math.radians(latitude)), 1e-8))
        )
    return {
        "use_markers": bool(marker.enabled),
        "marker_epsilon": float(marker.epsilon),
        "advection_integrator": str(marker.advection_integrator),
        "resample_spacing": spacing,
        "insert_factor": float(marker.insert_factor),
        "delete_factor": float(marker.delete_factor),
        "smoothing_weight": float(marker.smooth_weight),
        "smoothing_iters": int(marker.smooth_iters),
        "redistribute_every": (
            int(marker.redistribute_every) if marker.redistribute_markers else 0
        ),
        "max_substeps": int(marker.max_substeps),
        "max_move_fraction": float(marker.max_move_fraction),
        "topology_mode": str(getattr(marker, "topology_mode", "legacy")),
        "barrier_mode": str(getattr(marker, "barrier_mode", "off")),
        "barrier_debug_dir": str(getattr(marker, "barrier_debug_dir", "marker_barrier_debug")),
        "export_burned_geometry": bool(
            getattr(marker, "export_burned_geometry", False)
            or getattr(marker, "topology_mode", "legacy") == "outer_hull"
        ),
        "sim_config": config.simulation,
    }


def _levelset_kwargs(config: IgnacioConfig) -> dict[str, Any]:
    """Translate Pydantic option names to actual levelset.py arguments."""
    ls = config.simulation.level_set
    reinit = getattr(ls, "reinit_every", getattr(ls, "reinit_interval", 0))
    narrow_band_enabled = bool(getattr(ls, "narrow_band", True))
    return {
        "cfl_factor": float(ls.cfl_factor),
        "reinit_interval": int(reinit),
        "reinit_iters": int(ls.reinit_iters),
        "narrow_band_width": (
            int(ls.narrow_band_width) if narrow_band_enabled else None
        ),
    }


def _polygon_parts(geom: Any) -> list[Polygon]:
    """Retrieve every polygon component, including GeometryCollections."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon) or hasattr(geom, "geoms"):
        parts = []
        for piece in geom.geoms:
            parts.extend(_polygon_parts(piece))
        return parts
    return []


def _history_holes(history: Any, index: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read active inner fire fronts without assuming a specific solver."""
    getter = getattr(history, "get_holes", None)
    if callable(getter):
        return list(getter(index))
    holes = getattr(history, "holes_by_time", None)
    return list(holes[index]) if holes is not None else []


def _history_geometry(history: Any, index: int):
    """Read the exact Polygon/MultiPolygon first; never discard holes.

    If the solver supplies only exterior rings, we can reconstruct disconnected
    components, but cannot invent unknown level-set interior holes. Explicit
    holes_by_time, when available, are subtracted from those exterior polygons.
    """
    times = getattr(history, "times", None) or []
    if not times:
        return None
    exact = getattr(history, "burned_geometries", None)
    if exact is not None:
        if len(exact) != len(times):
            raise ValueError("Burned geometries are not aligned with times")
        geom = exact[index]
    else:
        getter = getattr(history, "get_burned_geometry", None)
        if callable(getter):
            geom = getter(index)
        else:
            geom = None
        if geom is None:
            geom = _multi_geometry(history, index)
        holes = _history_holes(history, index)
        if geom is not None and holes:
            hole_poly = unary_union([_front_polygon(x, y) for x, y in holes])
            geom = geom.difference(hole_poly)
    parts = _polygon_parts(geom)
    if not parts:
        return None
    result = parts[0] if len(parts) == 1 else unary_union(parts)
    if not result.is_valid:
        raise ValueError("Invalid burned geometry; repair in solver, not GIS export")
    return result


def _geometry_metrics(geom: Any, is_geographic: bool = False) -> tuple[float, float]:
    """Return burned area in hectares and full boundary length in metres."""
    if geom is None or geom.is_empty:
        return 0.0, 0.0
    if not is_geographic:
        return float(geom.area) / 10000.0, float(geom.length)
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    area_m2 = length_m = 0.0
    for part in _polygon_parts(geom):
        rings = [(part.exterior, 1)] + [(hole, -1) for hole in part.interiors]
        for ring, sign in rings:
            coords = np.asarray(ring.coords)
            area, length = geod.polygon_area_perimeter(coords[:, 0], coords[:, 1])
            area_m2 += sign * abs(area)
            length_m += length
    return max(0.0, area_m2) / 10000.0, length_m


def _history_statistics(history: Any, param_grid: FireParameterGrid,
                        config: IgnacioConfig, is_geographic: bool = False):
    """Metrics from full polygons, retaining signed area changes for QA."""
    times = getattr(history, "times", None) or []
    records = []
    previous_area = None
    for i, time_min in enumerate(times):
        geom = _history_geometry(history, i)
        area_ha, perim_m = _geometry_metrics(geom, is_geographic=is_geographic)
        fronts = _history_fronts(history, i)
        holes = _history_holes(history, i)
        ros_samples = []
        t_idx = max(0, min(int(round(float(time_min) / config.simulation.dt)), param_grid.nt - 1))
        for x, y in fronts:
            if len(x) == 0:
                continue
            ros, _, _, _ = param_grid.sample_at(t_idx, np.asarray(x), np.asarray(y))
            valid = np.asarray(ros, dtype=float)
            valid = valid[np.isfinite(valid) & (valid > 0)]
            if valid.size:
                ros_samples.append(valid)
        all_ros = np.concatenate(ros_samples) if ros_samples else np.array([], dtype=float)
        parts = _polygon_parts(geom)
        signed_delta = 0.0 if previous_area is None else area_ha - previous_area
        records.append({
            "area_ha": area_ha,
            "perim_m": perim_m,
            "darea_ha": signed_delta,
            "ros_mean": float(np.mean(all_ros)) if all_ros.size else 0.0,
            "ros_max": float(np.max(all_ros)) if all_ros.size else 0.0,
            "n_fronts": len(parts),
            "n_holes": sum(len(p.interiors) for p in parts),
            "n_active_fronts": len(fronts),
            "n_active_holes": len(holes),
            "geometry_valid": bool(geom is not None and geom.is_valid),
        })
        if previous_area is not None and signed_delta < -1e-8:
            logger.warning("Burned area fell by %.6f ha at t=%.4f min",
                           -signed_delta, float(time_min))
        previous_area = area_ha
    history.history_stats = records
    return records


def _check_metric_grid(terrain: TerrainGrids) -> None:
    """Multi-ignition front propagation is defined here in projected metres."""
    from pyproj import CRS
    if terrain.crs is None:
        raise ValueError("Multi-ignition runs require a defined CRS")
    crs = CRS.from_user_input(terrain.crs)
    if not crs.is_projected or any(
        abs(a.unit_conversion_factor - 1.0) > 1e-6 for a in crs.axis_info[:2]
    ):
        raise ValueError("Multi-ignition runs require projected metre CRS")


# =============================================================================
# Perimeter Untangling / Cleaning Utilities
# =============================================================================


def _ensure_ring_closed(coords: list[tuple[float, float]], eps: float = 0.0) -> list[tuple[float, float]]:
    """Ensure a coordinate ring is closed.

    Parameters
    ----------
    coords : list of (x, y)
        Ring coordinates.
    eps : float
        Optional tolerance: if the first/last points are within eps, snap closed.
    """
    if not coords:
        return coords

    x0, y0 = coords[0]
    x1, y1 = coords[-1]

    if (x0, y0) == (x1, y1):
        return coords

    if eps > 0.0:
        if (abs(x0 - x1) <= eps) and (abs(y0 - y1) <= eps):
            coords[-1] = (x0, y0)
            return coords

    coords.append((x0, y0))
    return coords


def _perim_to_ring(perim: Any) -> list[tuple[float, float]]:
    """Convert a perimeter representation to a coordinate ring.

    Supports:
      - (x_array, y_array)
      - Nx2 numpy arrays
      - list of (x, y)
    """
    if perim is None:
        return []

    # (x, y) tuple
    if isinstance(perim, tuple) and len(perim) == 2:
        x, y = perim
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            return []
        return list(zip(x.tolist(), y.tolist()))

    arr = np.asarray(perim)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [(float(a), float(b)) for a, b in arr]

    # list of tuples
    if isinstance(perim, list) and perim and isinstance(perim[0], (tuple, list)) and len(perim[0]) == 2:
        return [(float(a), float(b)) for a, b in perim]

    return []



def _ring_to_xy(ring: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not ring:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs = np.array([p[0] for p in ring], dtype=float)
    ys = np.array([p[1] for p in ring], dtype=float)
    return xs, ys


# ---------------------------------------------------------------------------
# Resampling and smoothing helpers for closed polygon rings
# ---------------------------------------------------------------------------
def _resample_ring_uniform(
    ring: list[tuple[float, float]],
    spacing: float,
) -> list[tuple[float, float]]:
    """Resample a closed ring to (approximately) uniform vertex spacing.

    Parameters
    ----------
    ring : list[(x, y)]
        Closed or open ring coordinates.
    spacing : float
        Target spacing in the same units as the coordinates.

    Returns
    -------
    list[(x, y)]
        Closed ring with resampled vertices.
    """
    if spacing <= 0 or not ring or len(ring) < 4:
        return ring

    pts = _ensure_ring_closed(list(ring), eps=0.0)
    if len(pts) < 4:
        return ring

    # Drop duplicate last point for distance traversal
    pts_open = pts[:-1]
    n = len(pts_open)
    if n < 3:
        return ring

    # Segment lengths
    xs = np.array([p[0] for p in pts_open] + [pts_open[0][0]], dtype=float)
    ys = np.array([p[1] for p in pts_open] + [pts_open[0][1]], dtype=float)
    dx = np.diff(xs)
    dy = np.diff(ys)
    seglens = np.hypot(dx, dy)
    perim = float(np.sum(seglens))

    if not np.isfinite(perim) or perim <= 0:
        return ring

    # Number of samples (at least original count, but not crazy)
    m = int(max(4, round(perim / spacing)))
    # Cap to avoid runaway memory when spacing is tiny
    m = min(m, 10000)

    # Cumulative distance along ring
    cum = np.concatenate([[0.0], np.cumsum(seglens)])
    # Sample locations (exclude the endpoint == perim)
    targets = np.linspace(0.0, perim, num=m, endpoint=False)

    out: list[tuple[float, float]] = []
    seg_i = 0
    for d in targets:
        while seg_i + 1 < len(cum) and cum[seg_i + 1] < d:
            seg_i += 1
        # Interpolate along segment seg_i
        d0 = cum[seg_i]
        d1 = cum[seg_i + 1]
        if d1 <= d0:
            t = 0.0
        else:
            t = (d - d0) / (d1 - d0)

        x0, y0 = xs[seg_i], ys[seg_i]
        x1, y1 = xs[seg_i + 1], ys[seg_i + 1]
        out.append((float(x0 + t * (x1 - x0)), float(y0 + t * (y1 - y0))))

    out = _ensure_ring_closed(out, eps=0.0)
    return out


def _chaikin_smooth_closed_ring(
    ring: list[tuple[float, float]],
    n_iter: int = 1,
) -> list[tuple[float, float]]:
    """Chaikin corner-cutting smoothing for a closed ring.

    This preserves overall shape but reduces jaggedness by creating new points
    along each edge. One iteration roughly doubles vertices.
    """
    if n_iter <= 0 or not ring or len(ring) < 4:
        return ring

    pts = _ensure_ring_closed(list(ring), eps=0.0)
    pts_open = pts[:-1]
    if len(pts_open) < 3:
        return ring

    for _ in range(n_iter):
        new_pts: list[tuple[float, float]] = []
        n = len(pts_open)
        for i in range(n):
            x0, y0 = pts_open[i]
            x1, y1 = pts_open[(i + 1) % n]
            # Q and R points
            qx = 0.75 * x0 + 0.25 * x1
            qy = 0.75 * y0 + 0.25 * y1
            rx = 0.25 * x0 + 0.75 * x1
            ry = 0.25 * y0 + 0.75 * y1
            new_pts.append((float(qx), float(qy)))
            new_pts.append((float(rx), float(ry)))
        pts_open = new_pts

    out = _ensure_ring_closed(pts_open, eps=0.0)
    return out


def _clean_polygon_from_ring(
    ring: list[tuple[float, float]],
    eps: float,
    simplify_tol: float | None = None,
    debug: dict[str, Any] | None = None,
    resample_spacing: float | None = None,
    chaikin_iters: int = 2,
) -> Polygon | None:
    """Attempt to remove self-intersections ("tangles") and jagged micro-edges.

    This is a pragmatic implementation inspired by Prometheus' "untangling":
    - enforce closure
    - drop duplicate consecutive points
    - repair self-intersections using make_valid() when available, otherwise buffer(0)
    - keep the largest resulting polygon (outer perimeter)
    - optional simplify() to reduce jaggedness

    Notes
    -----
    This is a post-processing step. The most faithful implementation would
    untangle after *each* marker advection step inside the spread solver.
    """
    if not ring or len(ring) < 4:
        return None

    if debug is not None:
        debug.clear()
        debug["n_vertices_in"] = len(ring)

    # Remove consecutive duplicates (within eps)
    cleaned: list[tuple[float, float]] = []
    for (x, y) in ring:
        if not cleaned:
            cleaned.append((x, y))
            continue
        px, py = cleaned[-1]
        if eps > 0.0 and (abs(x - px) <= eps) and (abs(y - py) <= eps):
            continue
        if (x, y) == (px, py):
            continue
        cleaned.append((x, y))

    cleaned = _ensure_ring_closed(cleaned, eps=eps)
    if len(cleaned) < 4:
        return None

    if debug is not None:
        debug["n_vertices_cleaned"] = len(cleaned)

    # -----------------------------------------------------------------
    # Resample + smooth (Fix B)
    # -----------------------------------------------------------------
    # Jagged perimeters are often due to uneven vertex spacing, even when
    # the polygon is topologically valid. Resampling to uniform spacing and
    # applying a light Chaikin smoothing tends to be more effective than
    # make_valid() for this case.
    if resample_spacing is None:
        # If caller didn't specify, use eps as a reasonable default when set.
        resample_spacing = float(eps) if eps and eps > 0 else None

    if resample_spacing is not None and resample_spacing > 0:
        cleaned = _resample_ring_uniform(cleaned, spacing=float(resample_spacing))
        if debug is not None:
            debug["resample_spacing"] = float(resample_spacing)
            debug["n_vertices_resampled"] = len(cleaned)

    if chaikin_iters and chaikin_iters > 0:
        cleaned = _chaikin_smooth_closed_ring(cleaned, n_iter=int(chaikin_iters))
        if debug is not None:
            debug["chaikin_iters"] = int(chaikin_iters)
            debug["n_vertices_smoothed"] = len(cleaned)

    poly = Polygon(cleaned)
    # Track area before any untangling/repair for diagnostics
    area_before = poly.area
    if debug is not None:
        debug["is_valid_before"] = bool(poly.is_valid)
        debug["used_make_valid"] = False
        debug["n_polygons_after_make_valid"] = 1
        debug["make_valid_result_type"] = None
    if poly.is_empty:
        return None

    # Repair invalid polygons (self-intersections, bow-ties)
    if not poly.is_valid:
        if make_valid is not None:
            fixed = make_valid(poly)
            if debug is not None:
                debug["used_make_valid"] = True
        else:
            # Classic Shapely trick that often resolves self-intersections
            fixed = poly.buffer(0)

        if fixed.is_empty:
            return None

        if debug is not None:
            debug["make_valid_result_type"] = type(fixed).__name__

        if isinstance(fixed, Polygon):
            poly = fixed
        elif isinstance(fixed, MultiPolygon):
            # Keep the largest component (outer perimeter)
            if debug is not None:
                debug["n_polygons_after_make_valid"] = len(list(fixed.geoms))
            poly = max(list(fixed.geoms), key=lambda g: g.area)
        else:
            # GeometryCollection: keep the largest polygonal part
            polys = [g for g in getattr(fixed, "geoms", []) if isinstance(g, Polygon)]
            if debug is not None:
                debug["n_polygons_after_make_valid"] = len(polys)
            if not polys:
                return None
            poly = max(polys, key=lambda g: g.area)

    # Log significant area change due to untangling/repair
    try:
        area_after = poly.area
        if area_before > 0:
            frac_change = (area_after - area_before) / area_before
            if abs(frac_change) > 0.1:
                logger.warning(
                    f"Untangling changed area by {100.0 * frac_change:.1f}% "
                    f"(before={area_before:.3f}, after={area_after:.3f})"
                )
    except Exception:
        pass

    # Optional simplification to reduce jaggedness (topology-preserving)
    if simplify_tol is not None and simplify_tol > 0.0:
        try:
            poly_s = poly.simplify(simplify_tol, preserve_topology=True)
            if isinstance(poly_s, Polygon) and (not poly_s.is_empty) and poly_s.area > 0:
                poly = poly_s
        except Exception:
            pass

    # Final sanity
    if poly.is_empty or poly.area <= 0:
        return None

    return poly


def _polygon_to_ring(poly: Polygon) -> list[tuple[float, float]]:
    """Extract exterior ring coordinates from a polygon."""
    coords = list(poly.exterior.coords)
    # Shapely returns a closed ring already
    return [(float(x), float(y)) for x, y in coords]


def untangle_history_perimeters(
    history: Any,
    eps: float,
    simplify_tol: float | None = None,
) -> None:
    """In-place untangling/cleaning of any perimeters stored on a FirePerimeterHistory.

    Works with common representations:
      - history.perimeters as list of (x, y) tuples
      - history.perimeters as list of Nx2 arrays

    If a perimeter cannot be repaired, it is left unchanged.
    """
    if history is None:
        return

    perims = getattr(history, "perimeters", None)
    if not perims or not isinstance(perims, list):
        return

    new_perims: list[Any] = []
    for step_idx, perim in enumerate(perims):
        ring = _perim_to_ring(perim)
        if not ring:
            new_perims.append(perim)
            continue

        dbg: dict[str, Any] = {}
        # Use eps both as a validity tolerance and a practical spacing scale.
        # If eps is 0, resampling/smoothing will be skipped.
        poly = _clean_polygon_from_ring(
            ring,
            eps=eps,
            simplify_tol=simplify_tol,
            debug=dbg,
            # Markersmethodchange: allow independent resampling + smoothing controls via config when available
            resample_spacing=(
                float(getattr(getattr(history, "marker_method", None), "resample_spacing", 0.0) or 0.0)
                if hasattr(history, "marker_method")
                else (eps if eps and eps > 0 else None)
            ),
            chaikin_iters=(
                int(getattr(getattr(history, "marker_method", None), "chaikin_iters", 0) or 0)
                if hasattr(history, "marker_method")
                else 1
            ),
        )

        # Prometheus-style debugging: step index, vertex counts, validity, make_valid output
        try:
            if dbg:
                logger.debug(
                    "Untangle step=%s n_vertices_in=%s n_vertices_cleaned=%s valid_before=%s "
                    "used_make_valid=%s make_valid_type=%s n_polys=%s",
                    step_idx,
                    dbg.get("n_vertices_in"),
                    dbg.get("n_vertices_cleaned"),
                    dbg.get("is_valid_before"),
                    dbg.get("used_make_valid"),
                    dbg.get("make_valid_result_type"),
                    dbg.get("n_polygons_after_make_valid"),
                )
        except Exception:
            pass

        if poly is None:
            new_perims.append(perim)
            continue

        out_ring = _polygon_to_ring(poly)
        x_out, y_out = _ring_to_xy(out_ring)
        new_perims.append((x_out, y_out))

    history.perimeters = new_perims


# =============================================================================
# Multiple-front geometry and ignition source helpers
# =============================================================================


def _history_fronts(history: FirePerimeterHistory, index: int):
    """Return all fronts, including a legacy single-front history."""
    if hasattr(history, "get_fronts"):
        return history.get_fronts(index)
    p = history.perimeters[index]
    return [] if p is None else [p]


def _front_polygon(x: np.ndarray, y: np.ndarray) -> Polygon:
    """Construct a valid projected-metre polygon without smoothing/area loss."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Invalid fire perimeter coordinates")
    # Work in a local coordinate frame to avoid numerical cancellation in UTM.
    x0, y0 = float(x[0]), float(y[0])
    poly = Polygon(np.column_stack((x - x0, y - y0)))
    if not poly.is_valid:
        raise ValueError("Invalid marker perimeter: solver should repair topology before export")
    if poly.is_empty or poly.area <= 0:
        raise ValueError("Empty marker perimeter")
    # Restore world coordinates and explicitly verify output geometry.
    world_polygon = Polygon(np.column_stack((x, y)))
    if not world_polygon.is_valid or world_polygon.is_empty:
        raise ValueError("Fire perimeter became invalid in projected coordinates")
    return world_polygon


def _multi_geometry(history: FirePerimeterHistory, index: int):
    """Exterior-only fallback for legacy histories without exact polygons."""
    polygons = [_front_polygon(x, y) for x, y in _history_fronts(history, index)]
    return unary_union(polygons) if polygons else None


def _file_ignition_positions(config: IgnacioConfig, terrain: TerrainGrids):
    """For a simultaneous-ignition shapefile experiment, read ALL source features.

    The legacy generate_ignitions() may sample input points according to
    occurrence/iteration logic; a simultaneous-front test must not do so.
    Returns None if not a multi-point shapefile.
    """
    source_type = str(getattr(config.ignition, "source_type", "")).lower()
    if not source_type.endswith("shapefile"):
        return None
    path_value = getattr(config.ignition, "point_path", None)
    if not path_value:
        raise ValueError("ignition.point_path is empty")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Ignition shapefile not found: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) < 2:
        return None  # The original one-ignition pipeline stays unchanged.
    if gdf.crs is None or terrain.crs is None:
        raise ValueError("Both ignition shapefile and terrain require a defined CRS")
    from pyproj import CRS
    crs_obj = CRS.from_user_input(terrain.crs)
    if not crs_obj.is_projected or not crs_obj.axis_info or not np.isclose(
        crs_obj.axis_info[0].unit_conversion_factor, 1.0, atol=1e-9
    ):
        raise ValueError("Multiple ignitions require a projected metre-based terrain grid")
    gdf = gdf.to_crs(terrain.crs)
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("Ignition shapefile contains empty geometries")
    if not all(geom.geom_type == "Point" for geom in gdf.geometry):
        raise ValueError("Simultaneous ignitions must be Point geometries")
    positions = [(float(p.x), float(p.y)) for p in gdf.geometry]
    if not np.isfinite(np.array(positions)).all() or len(set(positions)) != len(positions):
        raise ValueError("Ignition positions must be finite and distinct")
    logger.info("Read %d simultaneous ignitions directly from %s", len(positions), path)
    return positions


def _replace_ignitions_points(ignitions: IgnitionSet, positions):
    """Update the ignition record collection to reflect the actual input points."""
    points = list(getattr(ignitions, "points", []))
    if not points:
        # Fall back to the common IGNACIO ignition dataclass constructor.
        # Fail clearly rather than saving an empty or invented ignition layer.
        raise ValueError(
            "generate_ignitions() returned no ignition template. "
            "Inspect ignacio.ignition.IgnitionPoint before multi-front simulation."
        )
    template = points[0]
    new_points = []
    for x, y in positions:
        if is_dataclass(template):
            p = replace(template, x=x, y=y)
        else:
            p = copy.copy(template)
            p.x, p.y = x, y
        new_points.append(p)
    try:
        ignitions.points = new_points
    except (AttributeError, TypeError):
        if not is_dataclass(ignitions):
            raise ValueError("IgnitionSet is immutable and cannot be reconstructed")
        ignitions = replace(ignitions, points=new_points)
    if len(ignitions.points) != len(positions):
        raise ValueError("Failed to preserve all ignition points")
    return ignitions


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FireResult:
    """Result from a single fire simulation."""
    
    ignition: IgnitionPoint
    history: FirePerimeterHistory
    final_area_ha: float = 0.0
    final_perimeter_m: float = 0.0
    dur_min: float = 0.0
    ignition_positions: list[tuple[float, float]] | None = None
    
    def get_final_polygon(self, output_crs: str, source_crs: str | None = None) -> gpd.GeoDataFrame:
        """Final burned geometry, including all disconnected regions and holes."""
        geom = _history_geometry(self.history, -1)
        if geom is None:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=output_crs)
        gdf = gpd.GeoDataFrame({
            "area_ha": [self.final_area_ha],
            "perim_m": [self.final_perimeter_m],
            "dur_min": [self.dur_min],
            "cause": [self.ignition.cause],
            "season": [self.ignition.season],
            "iteration": [getattr(self.ignition, "iteration", 0)],
            "n_ignitions": [len(self.ignition_positions or []) or 1],
            "geometry": [geom],
        }, geometry="geometry", crs=source_crs or output_crs)
        if source_crs and source_crs != output_crs:
            gdf = gdf.to_crs(output_crs)
        return gdf




@dataclass
class SimulationResults:
    """Results from complete simulation run."""
    
    fires: list[FireResult] = field(default_factory=list)
    config: IgnacioConfig | None = None
    terrain: TerrainGrids | None = None
    weather: FireWeatherList | None = None
    ignitions: IgnitionSet | None = None
    
    @property
    def n_fires(self) -> int:
        """Number of simulated fires."""
        return len(self.fires)
    
    @property
    def total_area_ha(self) -> float:
        """Total burned area in hectares."""
        return sum(f.final_area_ha for f in self.fires)
    
    def get_all_perimeters(self, source_crs: str | None = None) -> gpd.GeoDataFrame:
        """Combine all fire perimeters into single GeoDataFrame.
        
        Parameters
        ----------
        source_crs : str, optional
            CRS of the source coordinates (terrain CRS).
        """
        if not self.fires:
            return gpd.GeoDataFrame()
        
        output_crs = self.config.crs.output_crs if self.config else "EPSG:4326"
        gdfs = [f.get_final_polygon(output_crs, source_crs=source_crs) for f in self.fires]
        gdfs = [gdf for gdf in gdfs if not gdf.empty]
        if not gdfs:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=output_crs)
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True),
                                geometry="geometry", crs=output_crs)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics for all fires."""
        data = []
        for i, fire in enumerate(self.fires):
            data.append({
                "fire_id": i,
                "n_ignitions": len(fire.ignition_positions) if fire.ignition_positions else 1,
                "x": fire.ignition.x,
                "y": fire.ignition.y,
                "cause": fire.ignition.cause,
                "season": fire.ignition.season,
                "iteration": fire.ignition.iteration,
                "area_ha": fire.final_area_ha,
                "perimeter_m": fire.final_perimeter_m,
                "dur_min": fire.dur_min,
            })
        return pd.DataFrame(data)


# =============================================================================
# Fuel-break rasterization aligned to the IGNACIO parameter grid
# =============================================================================


def _rasterize_fuel_break_geometries(
    fuel_break_gdf: gpd.GeoDataFrame,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    ny: int,
    nx: int,
    *,
    all_touched: bool = False,
) -> np.ndarray:
    """Rasterize a vector fuel-break layer to the terrain's cell-centre grid.

    Uses rasterio.features.rasterize directly rather than IGNACIO's legacy
    rasterize_geometries wrapper (whose accepted keyword arguments differ).
    The signed dy supports the north-up, descending-y DEM used by IGNACIO.
    Returns a uint8 grid containing 1 for covered cells and 0 otherwise.

    This function makes *no* changes to spread or fuel-break treatments.
    """
    from affine import Affine
    from rasterio.features import rasterize

    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    if nx < 2 or ny < 2 or x.shape != (nx,) or y.shape != (ny,):
        raise ValueError("Fuel-break rasterization requires 1-D x/y cell-centre coordinates")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Terrain grid coordinates contain non-finite values")

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if (dx == 0.0 or dy == 0.0
            or not np.allclose(np.diff(x), dx, rtol=1e-7, atol=1e-9)
            or not np.allclose(np.diff(y), dy, rtol=1e-7, atol=1e-9)):
        raise ValueError("Fuel-break rasterization requires a uniformly spaced grid")

    if fuel_break_gdf is None or fuel_break_gdf.empty:
        raise ValueError("Enabled fuel-break vector layer is empty")
    shapes = []
    for i, geom in enumerate(fuel_break_gdf.geometry):
        if geom is None or geom.is_empty:
            raise ValueError(f"Fuel-break feature {i} has missing/empty geometry")
        if not geom.is_valid:
            raise ValueError(f"Fuel-break feature {i} has invalid geometry")
        shapes.append((geom, 1))

    # x/y are cell centres; rasterio's transform locates the upper-left
    # *corner* of pixel (0, 0). Preserve the sign of dy for north-up rasters.
    transform = Affine.translation(
        float(x[0] - dx / 2.0), float(y[0] - dy / 2.0)
    ) * Affine.scale(dx, dy)

    return rasterize(
        shapes,
        out_shape=(int(ny), int(nx)),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=bool(all_touched),
    )


# =============================================================================
# Parameter Grid Building
# =============================================================================


def build_parameter_grid(
    config: IgnacioConfig,
    terrain: TerrainGrids,
    weather: FireWeatherList,
    n_timesteps: int | None = None,
    hourly_data: pd.DataFrame | None = None,
    start_datetime: pd.Timestamp | None = None,
    rng: np.random.Generator | None = None,
) -> FireParameterGrid:
    """
    Build spatially-varying fire parameter grid.

    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
    terrain : TerrainGrids
        Terrain data.
    weather : FireWeatherList
        Fire weather data (used for representative values if no hourly data).
    n_timesteps : int, optional
        Number of time steps. Defaults to max_duration / dt.
    hourly_data : DataFrame, optional
        Hourly weather data for time-varying simulation.
    start_datetime : Timestamp, optional
        Simulation start time for time-varying weather.
    rng : numpy.random.Generator, optional
        Random number generator (used for wind direction noise).

    Returns
    -------
    FireParameterGrid
        Grid of ROS parameters over space and time.
    """
    from ignacio.weather import HourlyWeatherInterpolator

    fbp_config = config.fbp
    sim_config = config.simulation

    if rng is None:
        rng = np.random.default_rng(config.project.random_seed)

    if n_timesteps is None:
        n_timesteps = _max_steps(config)
    
    # Determine if using time-varying weather
    use_time_varying = (
        sim_config.time_varying_weather 
        and hourly_data is not None 
        and len(hourly_data) > 0
    )
    
    if use_time_varying:
        logger.info("Using time-varying weather interpolation")
        interpolator = HourlyWeatherInterpolator(hourly_data=hourly_data)
        
        # Determine start time
        if start_datetime is None:
            if sim_config.start_datetime:
                start_datetime = pd.Timestamp(sim_config.start_datetime)
            else:
                # Use a representative date from the weather data with default hour
                if "DATE" in hourly_data.columns:
                    dates = pd.to_datetime(hourly_data["DATE"]).dropna()
                    if len(dates) > 0:
                        mid_date = dates.iloc[len(dates) // 2]
                        start_datetime = pd.Timestamp(mid_date) + pd.Timedelta(hours=sim_config.default_start_hour)
                        logger.info(f"Using mid-season start: {start_datetime}")
                
                if start_datetime is None:
                    start_datetime = pd.Timestamp("2024-07-15 12:00:00")
                    logger.warning(f"No date info found, using default: {start_datetime}")
    else:
        interpolator = None
        start_datetime = None
    
    # Get representative/initial weather
    weather_vals = get_representative_weather(weather, config)
    # Combined fuel break mask for zero_ros treatment (can come from multiple breaks)
    fuel_break_zero_ros_mask = None
    
    # Load fuel grid
    fuel_raster = reproject_raster(
        read_raster_int(config.fuel.path),
        reference=terrain.dem,
        resampling=Resampling.nearest,
        nodata=-9999.0,
    )
    fuel_grid = fuel_raster.data

    # Get coordinate arrays from terrain (this is our reference grid)
    x_coords, y_coords = terrain.get_coordinate_arrays()
    ny, nx = terrain.shape

    logger.info(f"Terrain grid shape: {ny} x {nx}")
    logger.info(f"Fuel grid shape: {fuel_grid.shape[0]} x {fuel_grid.shape[1]}")
    logger.debug(f"Terrain CRS: {terrain.dem.crs}, Fuel CRS: {fuel_raster.crs}")

    # Fuel matches the terrain CRS, affine transform and dimensions.

    # Load and process fuel breaks if enabled
    # Supports either:
    #   - config.fuel_break (single FuelBreakConfig)
    #   - config.fuel_breaks (list[FuelBreakConfig])
    #   - config.fuel_break being a list/tuple (for forward-compat)
    def _iter_fuel_breaks() -> list[Any]:
        """Return a list of fuel break config objects (backward compatible).

        NOTE: We intentionally return BOTH enabled and disabled entries so the
        logs can confirm what was parsed from YAML.
        """
        out: list[Any] = []

        # Backward-compatible single fuel_break (allow it to be a list/tuple too)
        fb_single = getattr(config, "fuel_break", None)
        if fb_single is not None:
            if isinstance(fb_single, (list, tuple)):
                out.extend(list(fb_single))
            else:
                out.append(fb_single)

        # Preferred new API: additional fuel_breaks list
        fb_list = getattr(config, "fuel_breaks", None)
        if fb_list:
            out.extend(list(fb_list))

        return out

    fuel_breaks = _iter_fuel_breaks()

    if fuel_breaks:
        enabled_ct = sum(1 for fb in fuel_breaks if getattr(fb, "enabled", False))
        logger.info(
            f"Found {len(fuel_breaks)} fuel break config(s) in config (enabled={enabled_ct}, disabled={len(fuel_breaks)-enabled_ct})"
        )
        for i, fb in enumerate(fuel_breaks, start=1):
            logger.info(
                f"  Fuel break {i}: enabled={getattr(fb, 'enabled', False)}, path={getattr(fb, 'path', None)}, treatment={getattr(fb, 'treatment_method', None)}, fuel_code={getattr(fb, 'fuel_code', None)}"
            )

    if fuel_breaks:
        logger.info(f"Processing {len([fb for fb in fuel_breaks if getattr(fb, 'enabled', False)])} enabled fuel break layer(s)")

        # Build a combined mask for zero_ros treatment across all breaks
        fuel_break_zero_ros_mask = np.zeros((ny, nx), dtype=bool)

        for fb_idx, fb_cfg in enumerate(fuel_breaks, start=1):
            if not getattr(fb_cfg, "enabled", False):
                logger.info(f"Skipping fuel break {fb_idx}: enabled=False")
                continue
            logger.info(f"Loading fuel break {fb_idx}/{len(fuel_breaks)} from: {fb_cfg.path}")

            try:
                fuel_break_path = Path(fb_cfg.path)

                # -----------------------------------------------------------------
                # Load break layer (raster or vector) and build mask on terrain grid
                # -----------------------------------------------------------------
                if fuel_break_path.suffix.lower() in [".tif", ".tiff"]:
                    # Align categorical fuel-break data to the terrain grid.
                    fuel_break_raster = reproject_raster(
                        read_raster_int(fuel_break_path),
                        reference=terrain.dem,
                        resampling=Resampling.nearest,
                        nodata=-9999.0,
                    )
                    fuel_break_grid = fuel_break_raster.data
                    fb_mask = fuel_break_grid > 0

                else:
                    # Vector fuel break (shapefile, GeoJSON, etc.)
                    fuel_break_gdf = read_vector(fuel_break_path, target_crs=terrain.dem.crs)
                    logger.info(f"Loaded {len(fuel_break_gdf)} fuel break features (break {fb_idx})")

                    # Buffer geometries if specified (buffer in meters; use projected CRS if needed)
                    if fb_cfg.buffer_m is not None and fb_cfg.buffer_m > 0:
                        logger.info(f"Buffering fuel break {fb_idx} by {fb_cfg.buffer_m} m")

                        if terrain.dem.crs is not None:
                            try:
                                import pyproj

                                crs_obj = pyproj.CRS.from_user_input(terrain.dem.crs)
                                if crs_obj.is_geographic:
                                    centroid = fuel_break_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
                                    lon, lat = float(centroid.x), float(centroid.y)
                                    utm_crs = pyproj.CRS.from_user_input(
                                        f"+proj=utm +zone={(int((lon + 180) / 6) + 1)} +datum=WGS84 +units=m +no_defs"
                                        + (" +south" if lat < 0 else "")
                                    )

                                    gdf_utm = fuel_break_gdf.to_crs(utm_crs)
                                    gdf_utm.geometry = gdf_utm.geometry.buffer(fb_cfg.buffer_m)
                                    fuel_break_gdf = gdf_utm.to_crs(terrain.dem.crs)
                                else:
                                    fuel_break_gdf.geometry = fuel_break_gdf.geometry.buffer(fb_cfg.buffer_m)
                            except Exception as e:
                                logger.warning(
                                    f"Fuel break {fb_idx} buffering CRS handling failed ({e}); buffering in layer CRS"
                                )
                                fuel_break_gdf.geometry = fuel_break_gdf.geometry.buffer(fb_cfg.buffer_m)
                        else:
                            fuel_break_gdf.geometry = fuel_break_gdf.geometry.buffer(fb_cfg.buffer_m)

                    # Rasterize to the *same* 2-D cell centres used by the
                    # ROS/BROS/FROS grids, using rasterio's supported API.
                    logger.info(f"Rasterizing {len(fuel_break_gdf)} fuel break features (break {fb_idx})")
                    fuel_break_grid = _rasterize_fuel_break_geometries(
                        fuel_break_gdf,
                        x_coords=x_coords,
                        y_coords=y_coords,
                        ny=ny,
                        nx=nx,
                        all_touched=fb_cfg.all_touched,
                    )
                    fb_mask = (fuel_break_grid == 1)
                    logger.info(
                        "Fuel break %d rasterization covered %d of %d cells (%.2f%%)",
                        fb_idx, int(fb_mask.sum()), ny * nx,
                        100.0 * float(fb_mask.sum()) / (ny * nx),
                    )

                # -------------------------------------------------------------
                # Apply preserve_non_fuel filter (per break)
                # -------------------------------------------------------------
                if getattr(fb_cfg, "preserve_non_fuel", False):
                    non_fuel_mask = np.isin(fuel_grid, config.fuel.non_fuel_codes)
                    fb_apply_mask = fb_mask & ~non_fuel_mask
                else:
                    fb_apply_mask = fb_mask

                n_cells = int(np.sum(fb_apply_mask))
                if n_cells <= 0:
                    logger.warning(
                        f"Fuel break {fb_idx}: no cells affected (check geometry/buffer and preserve_non_fuel setting)"
                    )
                    continue

                # -------------------------------------------------------------
                # Apply treatment
                # -------------------------------------------------------------
                treatment = getattr(fb_cfg, "treatment_method", "fuel_code")

                if treatment == "fuel_code":
                    # Overwrite fuel codes inside mask
                    fuel_grid[fb_apply_mask] = fb_cfg.fuel_code
                    logger.info(
                        f"Applied fuel break {fb_idx} (fuel_code): {n_cells} cells set to fuel code {fb_cfg.fuel_code}"
                    )

                elif treatment == "zero_ros":
                    # Accumulate into combined mask for later ROS forcing
                    fuel_break_zero_ros_mask |= fb_apply_mask
                    logger.info(
                        f"Applied fuel break {fb_idx} (zero_ros): {n_cells} cells will have ROS forced to 0"
                    )

                else:
                    logger.warning(
                        f"Fuel break {fb_idx}: unknown treatment_method='{treatment}'. No treatment applied."
                    )

            except Exception as e:
                # An enabled zero-ROS break is part of the physical experiment.
                # Never run the fire solver with that constraint silently absent.
                logger.exception(
                    "Failed to load/apply enabled fuel break %d from %s",
                    fb_idx, fb_cfg.path,
                )
                raise RuntimeError(
                    f"Required fuel break {fb_idx} ({fb_cfg.path}) could not be applied"
                ) from e

        # If no cells ended up in the zero_ros mask, disable it
        if fuel_break_zero_ros_mask is not None and not np.any(fuel_break_zero_ros_mask):
            fuel_break_zero_ros_mask = None
    
    # Constant weather only needs four 2-D slices.  A read-only broadcast
    # along time retains FireParameterGrid's (nt, ny, nx) API without holding
    # four separate 3,500 x 500 x 500 arrays (~14 GB) in physical RAM.
    # Dynamic weather still allocates independent arrays for every timestep.
    if use_time_varying:
        ros_arr = np.zeros((n_timesteps, ny, nx), dtype=np.float32)
        bros_arr = np.zeros_like(ros_arr)
        fros_arr = np.zeros_like(ros_arr)
        raz_arr = np.zeros_like(ros_arr)
    else:
        ros_arr = bros_arr = fros_arr = raz_arr = None
    
    logger.info(f"Building parameter grid: {n_timesteps} timesteps, {ny}x{nx} cells")
    
    # Get unique fuel types
    unique_fuels = np.unique(fuel_grid)
    non_fuel = config.fuel.non_fuel_codes
    
    logger.info(f"Unique fuel codes in grid: {sorted([int(f) for f in unique_fuels if np.isfinite(f)])}")
    
    # Build ROS grid for each timestep
    if use_time_varying:
        # Time-varying weather: compute ROS per timestep
        dt_minutes = sim_config.dt
        log_interval = max(1, n_timesteps // 5)  # Log ~5 times

        for t in range(n_timesteps):
            sim_time = start_datetime + pd.Timedelta(minutes=t * dt_minutes)
            weather_t = interpolator.interpolate_at(sim_time)

            # Get weather values for this timestep
            isi_t = weather_t.get("ISI", weather_vals["isi"])
            bui_t = weather_t.get("BUI", weather_vals["bui"])
            wind_dir_t = weather_t.get("WD", weather_t.get("WIND_DIRECTION", weather_vals["wind_direction"]))
            # Add optional per-timestep wind direction noise
            wd_noise_std = sim_config.wind_direction_noise_std
            if wd_noise_std > 0:
                wind_dir_t = (wind_dir_t + rng.normal(0, wd_noise_std)) % 360.0
            temp_t = weather_t.get("TEMP", weather_t.get("TEMPERATURE", weather_vals["temperature"]))
            rh_t = weather_t.get("RH", weather_t.get("RELATIVE_HUMIDITY", weather_vals["relative_humidity"]))

            # Compute base ROS for each fuel type
            base_ros = np.zeros((ny, nx), dtype=np.float32)
            for fuel_id in unique_fuels:
                if fuel_id in non_fuel or np.isnan(fuel_id):
                    continue

                ros = compute_ros(
                    fuel_type=int(fuel_id),
                    isi=isi_t,
                    bui=bui_t,
                    fmc=fbp_config.fmc,
                    curing=fbp_config.curing,
                    fuel_lookup=config.fuel.fuel_lookup,
                )
                mask = fuel_grid == fuel_id
                base_ros[mask] = ros

            # Apply terrain-aware wind deflection (spatially varying WD)
            terrain_deflection = float(getattr(fbp_config, "terrain_wind_deflection", 0.0))
            if terrain_deflection > 0:
                from ignacio.terrain import compute_terrain_wind_direction
                wind_dir_grid_t = compute_terrain_wind_direction(
                    wind_dir_t, terrain.aspect_deg, terrain.slope_deg,
                    deflection_weight=terrain_deflection,
                )
            else:
                wind_dir_grid_t = None

            # Compute slope factor for this wind direction
            slope_factor = compute_slope_factor(
                terrain.slope_deg,
                terrain.aspect_deg,
                wind_dir_t,
                fbp_config.slope_factor,
            )

            # Apply slope correction
            ros_corrected = base_ros * slope_factor

            # Apply fuel breaks (zero_ros method) - combined mask across all breaks
            if fuel_break_zero_ros_mask is not None:
                ros_corrected[fuel_break_zero_ros_mask] = 0.0

            # Elevation adjustment if enabled
            if fbp_config.elevation_adjustment.enabled:
                from ignacio.terrain import compute_elevation_adjustment

                elev_factor = compute_elevation_adjustment(
                    terrain.dem.data,
                    temp_t,
                    rh_t,
                    fbp_config.elevation_adjustment.lapse_rate,
                    fbp_config.elevation_adjustment.reference_temp,
                    fbp_config.elevation_adjustment.reference_rh,
                )
                ros_corrected *= elev_factor

            # Compute wind-dependent ellipse components (FBP Eq. 79-80)
            from ignacio.fbp import (
                compute_length_to_breadth_grid,
                compute_backing_ros,
                compute_flanking_ros,
            )
            wind_speed_t = weather_t.get("WS", weather_t.get("WIND_SPEED", weather_vals.get("wind_speed", 0.0)))
            lb_t = compute_length_to_breadth_grid(
                float(wind_speed_t),
                fuel_grid=fuel_grid,
                fuel_lookup=config.fuel.fuel_lookup,
            )
            bros_t = compute_backing_ros(ros_corrected, lb_t)
            fros_t = compute_flanking_ros(ros_corrected, bros_t, lb_t)

            ros_arr[t] = ros_corrected
            bros_arr[t] = bros_t
            fros_arr[t] = fros_t
            # Use spatially varying wind direction if terrain deflection is active
            if wind_dir_grid_t is not None:
                raz_arr[t] = np.radians((wind_dir_grid_t + 180.0) % 360.0)
            else:
                raz_arr[t] = np.radians((wind_dir_t + 180.0) % 360.0)

            if t % log_interval == 0:
                hour = sim_time.hour
                valid_ros = ros_corrected[ros_corrected > 0]
                if len(valid_ros) > 0:
                    logger.debug(
                        f"  t={t}: {sim_time.strftime('%H:%M')}, ISI={isi_t:.1f}, "
                        f"WD={wind_dir_t:.0f}°, mean ROS={np.mean(valid_ros):.2f} m/min"
                    )

        # Log summary
        logger.info(
            f"Time-varying weather: {start_datetime.strftime('%Y-%m-%d %H:%M')} to "
            f"{(start_datetime + pd.Timedelta(minutes=n_timesteps * dt_minutes)).strftime('%H:%M')}"
        )

    else:
        # Static weather: compute once and broadcast
        # Compute base ROS for each fuel type
        ros_by_fuel = {}
        for fuel_id in unique_fuels:
            if fuel_id in non_fuel or np.isnan(fuel_id):
                ros_by_fuel[fuel_id] = 0.0
                continue

            ros = compute_ros(
                fuel_type=int(fuel_id),
                isi=weather_vals["isi"],
                bui=weather_vals["bui"],
                fmc=fbp_config.fmc,
                curing=fbp_config.curing,
                fuel_lookup=config.fuel.fuel_lookup,
            )
            ros_by_fuel[fuel_id] = ros

            if ros > 0:
                fuel_name = config.fuel.fuel_lookup.get(int(fuel_id), f"ID-{int(fuel_id)}")
                logger.debug(f"Fuel {fuel_name} (code {int(fuel_id)}): ROS = {ros:.2f} m/min")

        # Compute slope factor
        wind_dir = weather_vals["wind_direction"]
        slope_factor = compute_slope_factor(
            terrain.slope_deg,
            terrain.aspect_deg,
            wind_dir,
            fbp_config.slope_factor,
        )

        # Build ROS grid
        base_ros = np.zeros((ny, nx), dtype=np.float32)
        for fuel_id, ros in ros_by_fuel.items():
            if np.isnan(fuel_id):
                continue
            mask = fuel_grid == fuel_id
            base_ros[mask] = ros

        # Apply slope correction
        ros_corrected = base_ros * slope_factor

        # Apply fuel breaks (zero_ros method) - combined mask across all breaks
        if fuel_break_zero_ros_mask is not None:
            ros_corrected[fuel_break_zero_ros_mask] = 0.0

        # Elevation adjustment if enabled
        if fbp_config.elevation_adjustment.enabled:
            from ignacio.terrain import compute_elevation_adjustment

            elev_factor = compute_elevation_adjustment(
                terrain.dem.data,
                weather_vals["temperature"],
                weather_vals["relative_humidity"],
                fbp_config.elevation_adjustment.lapse_rate,
                fbp_config.elevation_adjustment.reference_temp,
                fbp_config.elevation_adjustment.reference_rh,
            )
            ros_corrected *= elev_factor

        # Compute wind-dependent ellipse components (FBP Eq. 79-80)
        from ignacio.fbp import (
            compute_length_to_breadth_grid,
            compute_backing_ros,
            compute_flanking_ros,
        )
        wind_speed_val = weather_vals.get("wind_speed", 0.0)
        lb_static = compute_length_to_breadth_grid(
            float(wind_speed_val),
            fuel_grid=fuel_grid,
            fuel_lookup=config.fuel.fuel_lookup,
        )
        bros_base = compute_backing_ros(ros_corrected, lb_static)
        fros_base = compute_flanking_ros(ros_corrected, bros_base, lb_static)

        # Rate of spread azimuth (direction fire spreads TO)
        raz_deg = (wind_dir + 180.0) % 360.0
        raz_rad = np.radians(raz_deg)

        # Apply scaling before broadcasting: the views are deliberately
        # read-only, because downstream solvers must not mutate input grids.
        _ros_scale = float(getattr(config.fbp, "ros_scale", 1.0))
        if _ros_scale != 1.0:
            logger.info("Applying ros_scale=%.4f to static ROS components", _ros_scale)
            ros_corrected = ros_corrected * _ros_scale
            bros_base = bros_base * _ros_scale
            fros_base = fros_base * _ros_scale

        def _time_broadcast(slice_2d):
            values = np.asarray(slice_2d, dtype=np.float32)
            return np.broadcast_to(values[None, :, :], (n_timesteps, ny, nx))

        ros_arr = _time_broadcast(ros_corrected)
        bros_arr = _time_broadcast(bros_base)
        fros_arr = _time_broadcast(fros_base)
        raz_arr = _time_broadcast(np.full((ny, nx), raz_rad, dtype=np.float32))
        logger.info("Static-weather parameter fields use shared 2-D broadcast slices")

    # Time-varying fields are unique for each time, so scale in place.
    _ros_scale = float(getattr(config.fbp, "ros_scale", 1.0))
    if use_time_varying and _ros_scale != 1.0:
        logger.info("Applying ros_scale=%.4f to time-varying ROS components", _ros_scale)
        ros_arr *= _ros_scale
        bros_arr *= _ros_scale
        fros_arr *= _ros_scale

    # Log ROS statistics (from first timestep)
    ros_t0 = ros_arr[0]
    valid_ros = ros_t0[ros_t0 > 0]
    if len(valid_ros) > 0:
        logger.info(
            f"ROS statistics: min={np.min(valid_ros):.2f}, max={np.max(valid_ros):.2f}, "
            f"mean={np.mean(valid_ros):.2f} m/min"
        )
        logger.info(f"Cells with ROS > 0: {len(valid_ros)} ({100*len(valid_ros)/ros_t0.size:.1f}%)")
    else:
        logger.warning("No cells have ROS > 0! Check fuel type mapping.")
    
    nonfuel_mask = np.isin(fuel_grid, config.fuel.non_fuel_codes)
    if fuel_break_zero_ros_mask is not None:
        nonfuel_mask |= fuel_break_zero_ros_mask

    return FireParameterGrid(
        x_coords=x_coords,
        y_coords=y_coords,
        ros=ros_arr,
        bros=bros_arr,
        fros=fros_arr,
        raz=raz_arr,
        nonfuel_mask=nonfuel_mask,
    )


# =============================================================================
# Single Fire Simulation
# =============================================================================


def simulate_single_fire(
    ignition: IgnitionPoint,
    param_grid: FireParameterGrid,
    config: IgnacioConfig,
    is_geographic: bool = False,
    center_latitude: float | None = None,
) -> FireResult:
    """Simulate one ignition using its configured numerical method."""
    if not (param_grid.x_min <= ignition.x <= param_grid.x_max
            and param_grid.y_min <= ignition.y <= param_grid.y_max):
        logger.warning("Ignition (%s, %s) outside grid bounds", ignition.x, ignition.y)
        return FireResult(ignition=ignition,
                          history=FirePerimeterHistory(perimeters=[], times=[]))

    sim = config.simulation
    method = _get_spread_method(sim)
    logger.info("Single fire: %s, dt=%.4f min", method, sim.dt)

    if method == "level_set":
        from ignacio.levelset import simulate_fire_spread_levelset
        history = simulate_fire_spread_levelset(
            param_grid=param_grid,
            x_ignition=float(ignition.x),
            y_ignition=float(ignition.y),
            dt=sim.dt,
            initial_radius=sim.initial_radius,
            store_every=sim.store_every,
            max_steps=_max_steps(config),
            min_ros=sim.min_ros,
            is_geographic=is_geographic,
            center_latitude=center_latitude,
            **_levelset_kwargs(config),
        )
    else:
        history = simulate_fire_spread(
            param_grid=param_grid,
            x_ignition=float(ignition.x),
            y_ignition=float(ignition.y),
            dt=sim.dt,
            n_vertices=sim.n_vertices,
            initial_radius=sim.initial_radius,
            store_every=sim.store_every,
            max_steps=_max_steps(config),
            min_ros=sim.min_ros,
            is_geographic=is_geographic,
            center_latitude=center_latitude,
            **_marker_kwargs(config, geographic=is_geographic,
                             latitude=center_latitude),
        )
        history.marker_epsilon = float(sim.marker_method.epsilon)
        history.marker_method = sim.marker_method
        if sim.marker_method.enabled and sim.marker_method.topology_mode != "outer_hull":
            # Legacy cleanup is deliberately not used for exact polygons.
            if getattr(history, "burned_geometries", None) is None:
                eps = float(sim.marker_method.epsilon)
                untangle_history_perimeters(history, eps=eps,
                                            simplify_tol=(0.5 * eps if eps else None))

    stats = _history_statistics(history, param_grid, config,
                                is_geographic=is_geographic)
    final = stats[-1] if stats else {"area_ha": 0.0, "perim_m": 0.0}
    return FireResult(
        ignition=ignition,
        history=history,
        final_area_ha=float(final["area_ha"]),
        final_perimeter_m=float(final["perim_m"]),
        dur_min=float(history.times[-1]) if history.times else 0.0,
        ignition_positions=[(float(ignition.x), float(ignition.y))],
    )


# =============================================================================
# Simultaneous multi-ignition simulation (marker OR level set)
# =============================================================================


def simulate_merged_fire(
    ignitions: list[IgnitionPoint],
    param_grid: FireParameterGrid,
    config: IgnacioConfig,
) -> FireResult:
    """Evolve multiple ignitions together with the selected numerical method.

    Marker fronts are coupled through spread.py topology reconstruction.
    Level-set ignitions are passed together to a single phi evolution.
    """
    if len(ignitions) < 2:
        raise ValueError("simultaneous fire requires two or more ignitions")
    method = _get_spread_method(config.simulation)
    sim = config.simulation
    positions = [(float(p.x), float(p.y)) for p in ignitions]
    if not np.isfinite(np.asarray(positions)).all() or len(set(positions)) != len(positions):
        raise ValueError("Ignition locations must be finite and distinct")
    radius = float(sim.initial_radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Initial radius must be finite and positive")
    for x, y in positions:
        if not (param_grid.x_min + radius < x < param_grid.x_max - radius
                and param_grid.y_min + radius < y < param_grid.y_max - radius):
            raise ValueError(f"Ignition ({x}, {y}) lacks {radius} m grid clearance")

    logger.info("Combined %s run: %d ignitions, %d timesteps",
                method, len(positions), _max_steps(config))
    if method == "lagrangian":
        if not sim.marker_method.enabled:
            raise ValueError("Multiple marker ignitions require marker_method.enabled")
        history = simulate_fire_spread(
            param_grid=param_grid,
            x_ignition=positions[0][0],
            y_ignition=positions[0][1],
            ignition_points=positions,
            dt=sim.dt,
            n_vertices=sim.n_vertices,
            initial_radius=radius,
            store_every=sim.store_every,
            max_steps=_max_steps(config),
            min_ros=sim.min_ros,
            is_geographic=False,
            **_marker_kwargs(config),
        )
        history.marker_epsilon = float(sim.marker_method.epsilon)
        history.marker_method = sim.marker_method
    else:
        from ignacio.levelset import simulate_fire_spread_levelset
        history = simulate_fire_spread_levelset(
            param_grid=param_grid,
            x_ignition=positions[0][0],
            y_ignition=positions[0][1],
            ignition_points=positions,
            dt=sim.dt,
            initial_radius=radius,
            store_every=sim.store_every,
            max_steps=_max_steps(config),
            min_ros=sim.min_ros,
            is_geographic=False,
            **_levelset_kwargs(config),
        )

    times = getattr(history, "times", None) or []
    if not times:
        raise RuntimeError(f"{method} solver returned no snapshots")
    if getattr(history, "fronts_by_time", None) is not None:
        if len(history.fronts_by_time) != len(times):
            raise RuntimeError("Fronts and time histories are misaligned")
    if getattr(history, "burned_geometries", None) is not None:
        if len(history.burned_geometries) != len(times):
            raise RuntimeError("Burned geometries and time histories are misaligned")

    # A multi-ignition model must not silently return one empty/single ignition.
    init_geom = _history_geometry(history, 0)
    if init_geom is None:
        raise RuntimeError("No initial burned geometry from multi-ignition solver")
    grid_resolution = max(abs(float(param_grid.dx)), abs(float(param_grid.dy)))
    separated = all(np.hypot(positions[i][0] - positions[j][0],
                             positions[i][1] - positions[j][1]) >
                    2 * radius + 2 * grid_resolution
                    for i in range(len(positions)) for j in range(i + 1, len(positions)))
    if separated and len(_polygon_parts(init_geom)) != len(positions):
        raise RuntimeError(
            f"Expected {len(positions)} initial disconnected burned regions, "
            f"found {len(_polygon_parts(init_geom))}. "
            "Check level-set multi-seed initialization and contour extraction."
        )

    stats = _history_statistics(history, param_grid, config)
    final = stats[-1]
    logger.info(
        "%s: %d ignitions, %d snapshots, %.4f ha, %.2f m perimeter, "
        "%d mergers, %d topology repairs",
        method, len(positions), len(times), final["area_ha"], final["perim_m"],
        len(getattr(history, "merger_events", []) or []),
        len(getattr(history, "topology_repairs", []) or []),
    )
    return FireResult(
        ignition=ignitions[0],
        history=history,
        final_area_ha=float(final["area_ha"]),
        final_perimeter_m=float(final["perim_m"]),
        dur_min=float(times[-1]),
        ignition_positions=positions,
    )


# =============================================================================
# Main Simulation Entry Point
# =============================================================================


def run_simulation(config: IgnacioConfig) -> SimulationResults:
    """
    Run complete fire growth simulation.
    
    This is the main entry point for running simulations. It orchestrates:
    1. Terrain processing
    2. Weather processing
    3. Ignition generation
    4. Parameter grid building
    5. Fire spread simulation for each ignition
    6. Output generation
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
        
    Returns
    -------
    SimulationResults
        Complete simulation results.
    """
    # Set up random number generator
    rng = np.random.default_rng(config.project.random_seed)
    
    # Create output directory
    output_dir = Path(config.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process terrain
    logger.info("=" * 60)
    logger.info("Step 1: Processing terrain")
    logger.info("=" * 60)
    terrain = build_terrain_grids(config)
    
    # Step 2: Process weather
    logger.info("=" * 60)
    logger.info("Step 2: Processing fire weather")
    logger.info("=" * 60)
    
    # Load hourly data first (before process_fire_weather converts to daily)
    hourly_data = None
    if config.simulation.time_varying_weather:
        try:
            hourly_data = load_weather_data(config)
            if len(hourly_data) > 0 and "HOUR" in hourly_data.columns:
                logger.info(f"Loaded {len(hourly_data)} hourly records for time-varying simulation")
            else:
                logger.info("Hourly data not suitable for time-varying simulation, using static weather")
                hourly_data = None
        except Exception as e:
            logger.warning(f"Could not load hourly data: {e}, using static weather")
            hourly_data = None
    
    weather = process_fire_weather(config, rng)

    if config.output.save_weather_summary:
        weather_path = output_dir / "fire_weather_list.csv"
        save_fire_weather_list(weather, weather_path)

    # Enrich hourly data with FWI indices so time-varying simulation uses
    # per-hour ISI (from hourly WS + daily FFMC) instead of a static value
    if hourly_data is not None and "DATE" in hourly_data.columns:
        daily_fwi = weather.records
        if "FFMC" in daily_fwi.columns and "BUI" in daily_fwi.columns:
            try:
                from ignacio.fwi import calculate_isi
                # Map daily FFMC and BUI to each hour by date
                date_to_ffmc = dict(zip(
                    pd.to_datetime(daily_fwi["DATE"]).dt.date,
                    daily_fwi["FFMC"],
                ))
                date_to_bui = dict(zip(
                    pd.to_datetime(daily_fwi["DATE"]).dt.date,
                    daily_fwi["BUI"],
                ))
                h_dates = pd.to_datetime(hourly_data["DATE"]).dt.date if not hasattr(hourly_data["DATE"].iloc[0], 'date') else hourly_data["DATE"]
                hourly_data["BUI"] = [date_to_bui.get(d, np.nan) for d in h_dates]
                # Compute ISI from hourly wind speed + daily FFMC
                ws_col = "WIND_SPEED" if "WIND_SPEED" in hourly_data.columns else "WS"
                hourly_ffmc = np.array([date_to_ffmc.get(d, np.nan) for d in h_dates])
                hourly_ws = hourly_data[ws_col].values.astype(float)
                hourly_isi = np.array([
                    calculate_isi(f, w) if np.isfinite(f) and np.isfinite(w) else np.nan
                    for f, w in zip(hourly_ffmc, hourly_ws)
                ])
                hourly_data["ISI"] = hourly_isi
                hourly_data["FFMC"] = hourly_ffmc
                valid = np.isfinite(hourly_isi)
                if valid.any():
                    logger.info(
                        f"Enriched hourly data with FWI: ISI range {hourly_isi[valid].min():.1f}-{hourly_isi[valid].max():.1f}, "
                        f"mean {hourly_isi[valid].mean():.1f}"
                    )
            except Exception as e:
                logger.warning(f"Could not enrich hourly data with FWI: {e}")
    
    # Step 3: Generate ignitions
    logger.info("=" * 60)
    logger.info("Step 3: Generating ignitions")
    logger.info("=" * 60)
    fuel_raster = reproject_raster(
        read_raster_int(config.fuel.path),
        reference=terrain.dem,
        resampling=Resampling.nearest,
        nodata=-9999.0,
    )
    
    # Get terrain CRS to ensure ignitions are in the same coordinate system
    terrain_crs = terrain.crs
    if terrain_crs is not None:
        terrain_crs = str(terrain_crs)
    # Preserve every feature from a multi-point ignition shapefile BEFORE
    # saving the ignition layer. One run must evolve ALL fronts simultaneously
    # regardless of whether the selected solver is marker or levelset.
    simultaneous_points = (_file_ignition_positions(config, terrain)
                           if config.simulation.simultaneous_ignitions else None)
    if simultaneous_points:
        method = _get_spread_method(config.simulation)
        _check_metric_grid(terrain)
        logger.info("Using %d simultaneous ignition points with %s solver",
                    len(simultaneous_points), method)
    ignitions = generate_ignitions(config, fuel_raster, rng, terrain_crs=terrain_crs)
    if simultaneous_points:
        ignitions = _replace_ignitions_points(ignitions, simultaneous_points)
    
    if config.output.save_ignition_summary:
        ignition_path = output_dir / "ignitions.shp"
        save_ignitions(ignitions, ignition_path)
    
    # Step 4: Build parameter grid
    logger.info("=" * 60)
    logger.info("Step 4: Building parameter grid")
    logger.info("=" * 60)
    param_grid = build_parameter_grid(
        config,
        terrain,
        weather,
        hourly_data=hourly_data,
        rng=rng,
    )
    
    # Step 5: Run simulations
    logger.info("=" * 60)
    logger.info("Step 5: Running fire simulations")
    logger.info("=" * 60)
    
    results = SimulationResults(
        config=config,
        terrain=terrain,
        weather=weather,
        ignitions=ignitions,
    )
    
    n_ignitions = ignitions.n_points
    logger.info(f"Simulating {n_ignitions} fire(s)")
    logger.debug(
        f"Grid bounds: X=[{param_grid.x_min:.6f}, {param_grid.x_max:.6f}], "
        f"Y=[{param_grid.y_min:.6f}, {param_grid.y_max:.6f}]"
    )
    
    # Detect if terrain is in geographic CRS
    is_geographic = False
    center_latitude = None
    if terrain.crs is not None:
        try:
            import pyproj
            crs = pyproj.CRS.from_user_input(terrain.crs)
            is_geographic = crs.is_geographic
        except Exception:
            # Heuristic: if x bounds are in typical longitude range
            if -180 <= param_grid.x_min <= 180 and -180 <= param_grid.x_max <= 180:
                is_geographic = True
    
    if is_geographic:
        center_latitude = (param_grid.y_min + param_grid.y_max) / 2.0
        logger.info(f"Geographic CRS detected, center latitude: {center_latitude:.4f}°")
    
    # Get fuel grid for diagnostics (use the already-resampled param_grid data)
    # Re-load and resample fuel for diagnostics
    fuel_raster = reproject_raster(
        read_raster_int(config.fuel.path),
        reference=terrain.dem,
        resampling=Resampling.nearest,
        nodata=-9999.0,
    )
    fuel_grid = fuel_raster.data

    if simultaneous_points and len(ignitions.points) != len(simultaneous_points):
        raise ValueError("Ignition loader dropped one or more explicit points")

    if config.simulation.simultaneous_ignitions:
        _check_metric_grid(terrain)
        groups: dict[int, list[IgnitionPoint]] = {}
        for point in ignitions.points:
            iteration = int(getattr(point, "iteration", 0) or 0)
            groups.setdefault(iteration, []).append(point)
        for iteration, members in sorted(groups.items()):
            logger.info("Iteration %s: advancing %d ignition(s) as one system",
                        iteration, len(members))
            if len(members) == 1:
                fire_result = simulate_single_fire(
                    members[0], param_grid, config,
                    is_geographic=is_geographic,
                    center_latitude=center_latitude,
                )
            else:
                fire_result = simulate_merged_fire(members, param_grid, config)
            results.fires.append(fire_result)
    else:
        # Preserve original independent-fire behavior.
        for i, ignition in enumerate(ignitions.points):
            logger.info("Fire %d/%d: (%.6f, %.6f)",
                        i + 1, n_ignitions, ignition.x, ignition.y)
            fire_result = simulate_single_fire(
                ignition, param_grid, config,
                is_geographic=is_geographic,
                center_latitude=center_latitude,
            )
            results.fires.append(fire_result)

    # Save result metadata and histories, including all disjoint components.

    # Step 6: Save outputs


    # Step 6: Save outputs
    logger.info("=" * 60)
    logger.info("Step 6: Saving outputs")
    logger.info("=" * 60)
    
    if config.output.save_perimeters:
        _save_results(results, output_dir, config)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Simulation Complete")
    logger.info("=" * 60)
    logger.info(f"Total simulations: {results.n_fires}; ignition points: {ignitions.n_points}")
    logger.info(f"Total area burned: {results.total_area_ha:.2f} ha")
    logger.info(f"Output directory: {output_dir}")
    
    return results



# =============================================================================
# Perimeter History Export Helper
# =============================================================================

def perimeter_history_to_gdf(
    fire: FireResult,
    fire_index: int,
    output_crs: str,
    source_crs: str | None = None,
    start_datetime: str | None = None,
) -> gpd.GeoDataFrame:
    """One full hole-aware Polygon/MultiPolygon per stored simulation time."""
    when0 = pd.Timestamp(start_datetime) if start_datetime else None
    records = []
    stats = getattr(fire.history, "history_stats", []) or []
    for i, time_min in enumerate(fire.history.times):
        geom = _history_geometry(fire.history, i)
        if geom is None or geom.is_empty:
            continue
        row_stats = stats[i] if i < len(stats) else {}
        parts = _polygon_parts(geom)
        t_min = float(time_min)
        when = when0 + pd.Timedelta(minutes=t_min) if when0 is not None else None
        records.append({
            "fire_index": int(fire_index),
            "n_ignitions": len(fire.ignition_positions or []) or 1,
            "step": int(i),
            "t_min": t_min,
            "datetime": when.to_pydatetime() if when is not None else None,
            "cause": fire.ignition.cause,
            "season": fire.ignition.season,
            "iteration": int(getattr(fire.ignition, "iteration", 0) or 0),
            "area_ha": float(geom.area) / 10000.0,
            "perim_m": float(geom.length),
            "darea_ha": float(row_stats.get("darea_ha", np.nan)),
            "ros_mean": float(row_stats.get("ros_mean", np.nan)),
            "ros_max": float(row_stats.get("ros_max", np.nan)),
            "n_fronts": len(parts),
            "n_holes": sum(len(p.interiors) for p in parts),
            "n_active_fronts": len(_history_fronts(fire.history, i)),
            "n_active_holes": len(_history_holes(fire.history, i)),
            "exact_geom": getattr(fire.history, "burned_geometries", None) is not None,
            "geometry": geom,
        })
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs or output_crs)
    if source_crs and source_crs != output_crs and not gdf.empty:
        gdf = gdf.to_crs(output_crs)
    return gdf


def _save_results(
    results: SimulationResults,
    output_dir: Path,
    config: IgnacioConfig,
) -> None:
    """Export final perimeters, complete histories and active front boundaries."""
    source_crs = (
        str(results.terrain.crs)
        if results.terrain is not None and results.terrain.crs is not None
        else None
    )
    output_crs = config.crs.output_crs
    output_dir.mkdir(parents=True, exist_ok=True)
    perimeters_dir = output_dir / "perimeters"
    perimeters_dir.mkdir(parents=True, exist_ok=True)

    for i, fire in enumerate(results.fires):
        if fire.final_area_ha <= 0:
            continue
        gdf = fire.get_final_polygon(output_crs, source_crs=source_crs)
        if gdf.empty:
            continue
        fmt = config.output.perimeter_format
        suffix, driver = {
            "shapefile": (".shp", "ESRI Shapefile"),
            "geojson": (".geojson", "GeoJSON"),
            "geopackage": (".gpkg", "GPKG"),
        }[fmt]
        write_vector(gdf, perimeters_dir / f"fire_{i:04d}{suffix}", driver=driver)

    if config.output.combine_perimeters:
        all_fires = results.get_all_perimeters(source_crs=source_crs)
        if not all_fires.empty:
            fmt = config.output.perimeter_format
            suffix, driver = {
                "shapefile": (".shp", "ESRI Shapefile"),
                "geojson": (".geojson", "GeoJSON"),
                "geopackage": (".gpkg", "GPKG"),
            }[fmt]
            write_vector(all_fires, output_dir / f"all_perimeters{suffix}", driver=driver)

    all_frames = []
    exterior_records, interior_records = [], []
    active_exterior_records, active_interior_records = [], []
    event_records = []

    for idx, fire in enumerate(results.fires):
        history = fire.history
        if not getattr(history, "times", None):
            continue
        frame = perimeter_history_to_gdf(
            fire, idx, output_crs, source_crs,
            start_datetime=config.simulation.start_datetime,
        )
        if not frame.empty:
            all_frames.append(frame)

        for step, t_min in enumerate(history.times):
            geom = _history_geometry(history, step)
            if geom is None:
                continue
            # Solver-native propagating rings are distinct from static holes
            # or geometric interfaces obtained from the burned polygon.
            for role, native_rings, destination in (
                ("active_exterior", _history_fronts(history, step), active_exterior_records),
                ("active_interior", _history_holes(history, step), active_interior_records),
            ):
                for ring_idx, (x, y) in enumerate(native_rings):
                    coords = np.column_stack((x, y))
                    if len(coords) < 3:
                        continue
                    if not np.array_equal(coords[0], coords[-1]):
                        coords = np.vstack((coords, coords[0]))
                    destination.append({
                        "fire_index": idx, "step": step, "t_min": float(t_min),
                        "ring_id": ring_idx, "role": role,
                        "geometry": LineString(coords),
                    })
            # Geometric boundaries include closed unburned islands, whether
            # or not the solver exposes them as active marker rings.
            for poly_idx, polygon in enumerate(_polygon_parts(geom)):
                exterior_records.append({
                    "fire_index": idx, "step": step, "t_min": float(t_min),
                    "ring_id": poly_idx, "role": "exterior",
                    "geometry": LineString(polygon.exterior.coords),
                })
                for hole_idx, inner in enumerate(polygon.interiors):
                    interior_records.append({
                        "fire_index": idx, "step": step, "t_min": float(t_min),
                        "ring_id": hole_idx, "parent_id": poly_idx,
                        "role": "interior", "geometry": LineString(inner.coords),
                    })

        stats = getattr(history, "history_stats", []) or []
        if stats:
            pd.DataFrame([
                {"step": j, "t_min": float(history.times[j]), **st}
                for j, st in enumerate(stats)
            ]).to_csv(output_dir / f"fire_{idx:04d}_history_totals.csv", index=False)
        for event_type, attr, name in (
            ("merger", "merger_events", "merger_events"),
            ("topology_repair", "topology_repairs", "topology_repairs"),
        ):
            events = getattr(history, attr, []) or []
            if events:
                pd.DataFrame(events).to_csv(
                    output_dir / f"fire_{idx:04d}_{name}.csv", index=False
                )
            for event in events:
                event_records.append({
                    "fire_index": idx,
                    "event_type": event_type,
                    "detail_json": json.dumps(event, default=str),
                    "time_min": event.get("time_min") if isinstance(event, dict) else None,
                })

    if all_frames:
        history_gdf = gpd.GeoDataFrame(
            pd.concat(all_frames, ignore_index=True),
            geometry="geometry", crs=output_crs,
        )
        history_gdf.to_file(
            output_dir / "all_perimeters_history.gpkg",
            layer="burned_history", driver="GPKG",
        )
        attributes = history_gdf.drop(columns=["geometry"]).copy()
        if "datetime" in attributes:
            attributes["datetime"] = (
                pd.to_datetime(attributes["datetime"], errors="coerce")
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )
        attributes.to_csv(
            output_dir / "all_perimeters_history_attributes.csv", index=False
        )
        logger.info("Saved %d complete fire-history snapshots", len(history_gdf))
    else:
        logger.warning("No fire-history geometries available to export")

    for filename, layer_name, rows in (
        ("all_exterior_boundaries.gpkg", "exterior_boundaries", exterior_records),
        ("all_interior_boundaries.gpkg", "interior_boundaries", interior_records),
        ("all_active_exterior_fronts.gpkg", "active_exterior", active_exterior_records),
        ("all_active_interior_fronts.gpkg", "active_interior", active_interior_records),
    ):
        if rows:
            rings = gpd.GeoDataFrame(rows, geometry="geometry", crs=source_crs or output_crs)
            if source_crs and source_crs != output_crs:
                rings = rings.to_crs(output_crs)
            rings.to_file(output_dir / filename, layer=layer_name, driver="GPKG")

    if event_records:
        pd.DataFrame(event_records).to_csv(
            output_dir / "topology_events.csv", index=False
        )

    results.get_summary().to_csv(output_dir / "simulation_summary.csv", index=False)
    logger.info("Saved complete H5 results to %s", output_dir)
