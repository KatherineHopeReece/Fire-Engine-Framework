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
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
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
    """Dissolved burned geometry for a snapshot. Preserve disconnected fronts."""
    polygons = [_front_polygon(x, y) for x, y in _history_fronts(history, index)]
    if not polygons:
        return None
    return unary_union(polygons)


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
        """Get final fire perimeter as GeoDataFrame.
        
        Parameters
        ----------
        output_crs : str
            Desired output CRS.
        source_crs : str, optional
            CRS of the source coordinates. If different from output_crs,
            the polygon will be reprojected.
        """
        # Multiple ignition histories can have several disconnected exterior
        # rings. Never call the legacy get_final_perimeter() in that case.
        if getattr(self.history, "fronts_by_time", None) is not None:
            if not self.history.times:
                return gpd.GeoDataFrame({"geometry": []}, crs=output_crs)
            geom = _multi_geometry(self.history, -1)
            if geom is None or geom.is_empty:
                return gpd.GeoDataFrame({"geometry": []}, crs=output_crs)
            gdf = gpd.GeoDataFrame({
                "area_ha": [self.final_area_ha],
                "perim_m": [self.final_perimeter_m],
                "dur_min": [self.dur_min],
                "cause": [self.ignition.cause],
                "season": [self.ignition.season],
                "iteration": [self.ignition.iteration],
                "n_ignitions": [len(self.ignition_positions or [])],
                "geometry": [geom],
            }, crs=source_crs if source_crs else output_crs)
            if source_crs and source_crs != output_crs:
                gdf = gdf.to_crs(output_crs)
            return gdf

        x, y = self.history.get_final_perimeter()
        if len(x) == 0:
            return gpd.GeoDataFrame({"geometry": []}, crs=output_crs)

        # Build polygon from perimeter coordinates
        ring = list(zip(x.tolist(), y.tolist()))
        ring = _ensure_ring_closed(ring, eps=0.0)

        # Level set contours are topologically clean — skip cleaning/smoothing
        _is_level_set = (self.history is not None
                         and not hasattr(self.history, "marker_method")
                         and not hasattr(self.history, "marker_epsilon"))

        if _is_level_set:
            polygon = Polygon(ring)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                if isinstance(polygon, MultiPolygon):
                    polygon = max(polygon.geoms, key=lambda g: g.area)
        else:
            # Use marker epsilon as a reasonable default tolerance if available
            eps = 0.0
            try:
                if self.history is not None and hasattr(self.history, "marker_epsilon"):
                    eps = float(getattr(self.history, "marker_epsilon") or 0.0)
            except Exception:
                eps = 0.0

            # Simplify tolerance: small fraction of epsilon (or disabled)
            simplify_tol = (0.5 * eps) if eps and eps > 0 else None

            poly = _clean_polygon_from_ring(
                ring,
                eps=eps,
                simplify_tol=simplify_tol,
                resample_spacing=(
                    float(getattr(getattr(self.history, "marker_method", None), "resample_spacing", 0.0) or 0.0)
                    if hasattr(self.history, "marker_method")
                    else (eps if eps and eps > 0 else None)
                ),
                chaikin_iters=(
                    int(getattr(getattr(self.history, "marker_method", None), "chaikin_iters", 0) or 0)
                    if hasattr(self.history, "marker_method")
                    else 1
                ),
            )
            polygon = poly if poly is not None else Polygon(ring)

        # Create GeoDataFrame with source CRS
        gdf = gpd.GeoDataFrame(
            {
                "area_ha": [self.final_area_ha],
                "perim_m": [self.final_perimeter_m],
                "dur_min": [self.dur_min],
                "cause": [self.ignition.cause],
                "season": [self.ignition.season],
                "iteration": [self.ignition.iteration],
                "geometry": [polygon],
            },
            crs=source_crs if source_crs else output_crs,
        )

        # Reproject if needed
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
        
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=output_crs)
    
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
        n_timesteps = int(sim_config.max_duration / sim_config.dt)
    
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

                    # Rasterize to match terrain grid
                    logger.info(f"Rasterizing {len(fuel_break_gdf)} fuel break features (break {fb_idx})")
                    fuel_break_grid = rasterize_geometries(
                        fuel_break_gdf,
                        reference=terrain.dem,
                        value=1,
                        fill=0,
                        dtype="uint8",
                        all_touched=fb_cfg.all_touched,
                    )
                    fb_mask = (fuel_break_grid == 1)

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
                logger.error(f"Failed to load/apply fuel break {fb_idx}: {e}")
                logger.warning("Continuing simulation without this fuel break")
                import traceback

                logger.debug(traceback.format_exc())

        # If no cells ended up in the zero_ros mask, disable it
        if fuel_break_zero_ros_mask is not None and not np.any(fuel_break_zero_ros_mask):
            fuel_break_zero_ros_mask = None
    
    # Initialize arrays
    ros_arr = np.zeros((n_timesteps, ny, nx), dtype=np.float32)
    bros_arr = np.zeros_like(ros_arr)
    fros_arr = np.zeros_like(ros_arr)
    raz_arr = np.zeros_like(ros_arr)
    
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
            terrain_deflection = fbp_config.terrain_wind_deflection
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

        # Fill time dimension (constant weather)
        for t in range(n_timesteps):
            ros_arr[t] = ros_corrected
            bros_arr[t] = bros_base
            fros_arr[t] = fros_base
            raz_arr[t] = raz_rad
    
    # Apply global ROS scale factor
    _ros_scale = getattr(config.fbp, "ros_scale", 1.0)
    if _ros_scale != 1.0:
        logger.info(f"Applying ros_scale={_ros_scale:.4f} to all ROS components")
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
    
    return FireParameterGrid(
        x_coords=x_coords,
        y_coords=y_coords,
        ros=ros_arr,
        bros=bros_arr,
        fros=fros_arr,
        raz=raz_arr,
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
    """
    Simulate a single fire from ignition point.
    
    Parameters
    ----------
    ignition : IgnitionPoint
        Ignition location and metadata.
    param_grid : FireParameterGrid
        Fire behaviour parameter grid.
    config : IgnacioConfig
        Configuration object.
    is_geographic : bool
        If True, coordinates are in degrees (lat/lon).
    center_latitude : float, optional
        Center latitude for geographic conversion.
        
    Returns
    -------
    FireResult
        Simulation results for this fire.
    """
    sim_config = config.simulation
    
    # Check if ignition is within grid bounds
    if not (param_grid.x_min <= ignition.x <= param_grid.x_max and
            param_grid.y_min <= ignition.y <= param_grid.y_max):
        logger.warning(
            f"Ignition ({ignition.x}, {ignition.y}) outside grid bounds, skipping"
        )
        return FireResult(
            ignition=ignition,
            history=FirePerimeterHistory(perimeters=[], times=[]),
        )
    
    # Markersmethodchange: log chosen advection integrator from YAML.
    try:
        logger.info(
            f"Requested advection integrator: {getattr(sim_config.marker_method, 'advection_integrator', None)}"
        )
    except Exception:
        pass
    # Run spread simulation
    if getattr(sim_config, "spread_method", "marker") == "levelset":
        # ---- Level set solver (JAX) ----
        from ignacio.levelset import simulate_fire_spread_levelset
        ls = sim_config.level_set
        history = simulate_fire_spread_levelset(
            param_grid=param_grid,
            x_ignition=ignition.x,
            y_ignition=ignition.y,
            dt=sim_config.dt,
            initial_radius=sim_config.initial_radius,
            store_every=sim_config.store_every,
            min_ros=sim_config.min_ros,
            is_geographic=is_geographic,
            center_latitude=center_latitude,
            cfl_factor=ls.cfl_factor,
            reinit_interval=ls.reinit_interval,
            reinit_iters=ls.reinit_iters,
            narrow_band_width=ls.narrow_band_width,
        )
        # Level set contours are topologically clean — no untangling needed
    else:
        # ---- Marker method (legacy) ----
        mm = sim_config.marker_method
        # Resolve advection integrator: legacy flag overrides if set
        if mm.use_rk4_advection is not None:
            _integrator = "rk4" if mm.use_rk4_advection else "euler"
        else:
            _integrator = mm.advection_integrator or "euler"

        # Convert resample_spacing from meters to coordinate units (degrees) if geographic
        _resample_spacing = None
        if mm.resample_spacing is not None and mm.resample_spacing > 0:
            if is_geographic and center_latitude is not None:
                import math
                _meters_per_deg_lat = 111_320.0
                _meters_per_deg_lon = 111_320.0 * math.cos(math.radians(center_latitude))
                _avg_m_per_deg = math.sqrt(_meters_per_deg_lat * _meters_per_deg_lon)
                _resample_spacing = mm.resample_spacing / _avg_m_per_deg
            else:
                _resample_spacing = mm.resample_spacing

        history = simulate_fire_spread(
            param_grid=param_grid,
            x_ignition=ignition.x,
            y_ignition=ignition.y,
            dt=sim_config.dt,
            n_vertices=sim_config.n_vertices,
            initial_radius=sim_config.initial_radius,
            store_every=sim_config.store_every,
            use_markers=mm.enabled,
            marker_epsilon=mm.epsilon,
            min_ros=sim_config.min_ros,
            is_geographic=is_geographic,
            center_latitude=center_latitude,
            advection_integrator=_integrator,
            resample_spacing=_resample_spacing,
            redistribute_every=mm.redistribute_every if mm.redistribute_markers else 0,
        )

        # -----------------------------------------------------------------
        # Untangle / clean stored perimeters (post-processing)
        # -----------------------------------------------------------------
        if sim_config.marker_method.enabled:
            eps = float(sim_config.marker_method.epsilon or 0.0)
            simplify_tol = (0.5 * eps) if eps > 0 else None
            try:
                setattr(history, "marker_epsilon", eps)
                setattr(history, "marker_method", sim_config.marker_method)
            except Exception:
                pass

            try:
                untangle_history_perimeters(history, eps=eps, simplify_tol=simplify_tol)
            except Exception as e:
                logger.debug(f"Perimeter untangling failed (continuing without): {e}")

    # ---------------------------------------------------------------------
    # Compute and attach per-timestep history stats (area/perimeter/ROS)
    # ---------------------------------------------------------------------
    # These stats are exported alongside the perimeter history to CSV.
    try:
        perims = getattr(history, "perimeters", None) or []
        times = getattr(history, "times", None) or []
        n = min(len(perims), len(times))

        history_stats: list[dict[str, Any]] = []

        # Helper to compute area/perimeter in meters (and ha) for both projected and geographic CRSs
        def _area_perim_from_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
            if x.size < 3 or y.size < 3:
                return 0.0, 0.0

            if is_geographic:
                lat0 = float(center_latitude) if center_latitude is not None else float(np.nanmean(y))
                lat_rad = np.radians(lat0)
                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * np.cos(lat_rad)

                x_m = (x - float(np.nanmean(x))) * meters_per_deg_lon
                y_m = (y - float(np.nanmean(y))) * meters_per_deg_lat

                area_m2 = 0.5 * np.abs(np.sum(x_m * np.roll(y_m, -1) - np.roll(x_m, -1) * y_m))
                dx = np.diff(np.append(x_m, x_m[0]))
                dy = np.diff(np.append(y_m, y_m[0]))
                perim_m = float(np.sum(np.hypot(dx, dy)))
                return area_m2 / 10000.0, perim_m

            # Projected
            area_m2 = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            dx = np.diff(np.append(x, x[0]))
            dy = np.diff(np.append(y, y[0]))
            perim_m = float(np.sum(np.hypot(dx, dy)))
            return area_m2 / 10000.0, perim_m

        prev_area_ha = 0.0
        dt_minutes = float(sim_config.dt)

        for step in range(n):
            perim = perims[step]
            x_arr, y_arr = None, None

            # Stored perimeters in this pipeline are typically (x_array, y_array)
            if isinstance(perim, tuple) and len(perim) == 2:
                x_arr = np.asarray(perim[0], dtype=float)
                y_arr = np.asarray(perim[1], dtype=float)
            else:
                ring = _perim_to_ring(perim)
                if ring:
                    x_arr = np.asarray([p[0] for p in ring], dtype=float)
                    y_arr = np.asarray([p[1] for p in ring], dtype=float)

            if x_arr is None or y_arr is None or x_arr.size < 3:
                history_stats.append({"area_ha": 0.0, "perim_m": 0.0, "darea_ha": 0.0, "ros_mean": 0.0, "ros_max": 0.0})
                continue

            area_ha, perim_m = _area_perim_from_xy(x_arr, y_arr)
            darea_ha = max(0.0, area_ha - prev_area_ha)
            prev_area_ha = area_ha

            # Map stored time (minutes) -> parameter-grid timestep index
            t_min = float(times[step])
            t_idx = int(round(t_min / dt_minutes)) if dt_minutes > 0 else 0
            t_idx = max(0, min(t_idx, int(param_grid.n_timesteps) - 1))

            # Sample ROS along the perimeter vertices for this timestep
            ros_mean = 0.0
            ros_max = 0.0
            try:
                ros_s, _, _, _ = param_grid.sample_at(t_idx, x_arr, y_arr)
                ros_s = np.asarray(ros_s, dtype=float)
                ros_pos = ros_s[np.isfinite(ros_s) & (ros_s > 0)]
                if ros_pos.size > 0:
                    ros_mean = float(np.mean(ros_pos))
                    ros_max = float(np.max(ros_pos))
            except Exception:
                pass

            history_stats.append(
                {
                    "area_ha": float(area_ha),
                    "perim_m": float(perim_m),
                    "darea_ha": float(darea_ha),
                    "ros_mean": float(ros_mean),
                    "ros_max": float(ros_max),
                }
            )

        setattr(history, "history_stats", history_stats)
    except Exception as e:
        logger.debug(f"Failed computing history stats (continuing without): {e}")
    
    # Compute final metrics
    final_area_ha = 0.0
    final_perimeter_m = 0.0
    
    if history.perimeters:
        # Find the last perimeter with enough vertices for area calculation
        x, y = np.array([]), np.array([])
        for pi in reversed(range(len(history.perimeters))):
            _x, _y = history.perimeters[pi]
            if len(_x) > 2:
                x, y = _x, _y
                break
        if len(x) > 2:
            if is_geographic:
                # Convert from degrees to meters for area/perimeter calculation
                if center_latitude is None:
                    center_latitude = np.mean(y)
                lat_rad = np.radians(center_latitude)
                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
                
                # Convert coordinates to local meters
                x_m = (x - np.mean(x)) * meters_per_deg_lon
                y_m = (y - np.mean(y)) * meters_per_deg_lat
                
                # Area via shoelace formula (in m²)
                area_m2 = 0.5 * np.abs(np.sum(x_m * np.roll(y_m, -1) - np.roll(x_m, -1) * y_m))
                final_area_ha = area_m2 / 10000.0
                
                # Perimeter length (in m)
                dx = np.diff(np.append(x_m, x_m[0]))
                dy = np.diff(np.append(y_m, y_m[0]))
                final_perimeter_m = float(np.sum(np.hypot(dx, dy)))
            else:
                # Projected coordinates - direct calculation
                # Area via shoelace formula
                area_m2 = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
                final_area_ha = area_m2 / 10000.0
                
                # Perimeter length
                dx = np.diff(np.append(x, x[0]))
                dy = np.diff(np.append(y, y[0]))
                final_perimeter_m = float(np.sum(np.hypot(dx, dy)))
    
    spread_duration = history.times[-1] if history.times else 0.0
    
    return FireResult(
        ignition=ignition,
        history=history,
        final_area_ha=final_area_ha,
        final_perimeter_m=final_perimeter_m,
        dur_min=spread_duration,
    )


# =============================================================================
# Simultaneous multi-ignition simulation (marker OR level set)
# =============================================================================


def simulate_merged_fire(
    ignitions: list[IgnitionPoint],
    param_grid: FireParameterGrid,
    config: IgnacioConfig,
) -> FireResult:
    """Propagate two or more ignitions together in one fire simulation.

    Marker mode evolves separate Richards marker rings and resolves intersections
    with polygon union inside ``simulate_fire_spread``.

    Level-set mode initializes a *single* signed-distance field from all ignition
    locations and uses ``simulate_fire_spread_levelset``. Topology changes arise
    from the field evolution, without marker-style polygon union in the solver.

    Both modes return the same FirePerimeterHistory multi-front representation;
    each saved time contains one or more exterior rings. Geometry unions below
    are used ONLY for output statistics, not for advancing the level-set field.
    """
    if len(ignitions) < 2:
        raise ValueError("simulate_merged_fire requires two or more ignitions")

    sim_config = config.simulation
    method = str(getattr(sim_config, "spread_method", "marker")).strip().lower()
    if method not in {"marker", "levelset"}:
        raise ValueError(f"Unsupported multi-ignition spread method: {method!r}")

    positions = [(float(p.x), float(p.y)) for p in ignitions]
    if not np.isfinite(np.asarray(positions, dtype=float)).all():
        raise ValueError("Ignition coordinates must be finite")
    if len(set(positions)) != len(positions):
        raise ValueError("Simultaneous ignition coordinates must be distinct")

    radius = float(sim_config.initial_radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("simulation.initial_radius must be positive and finite")

    for i, (x, y) in enumerate(positions, start=1):
        if not (param_grid.x_min + radius < x < param_grid.x_max - radius
                and param_grid.y_min + radius < y < param_grid.y_max - radius):
            raise ValueError(
                f"Ignition {i}: ({x:.3f}, {y:.3f}) must lie within the "
                f"projected grid with at least {radius:g} m boundary clearance"
            )

    if method == "marker":
        mm = sim_config.marker_method
        if not mm.enabled:
            raise ValueError("Multi-front marker propagation requires marker_method.enabled=true")
        if getattr(mm, "use_rk4_advection", None) is not None:
            integrator = "rk4" if mm.use_rk4_advection else "euler"
        else:
            integrator = mm.advection_integrator or "euler"
        spacing = getattr(mm, "resample_spacing", None)
        if spacing is None or float(spacing) <= 0:
            raise ValueError(
                "For multiple ignitions using markers, set "
                "simulation.marker_method.resample_spacing > 0 in metres"
            )
        logger.info(
            "Simultaneous marker experiment: %d ignitions, spacing %.2f m, integrator %s",
            len(positions), float(spacing), integrator,
        )
        history = simulate_fire_spread(
            param_grid=param_grid,
            # Keep legacy coordinates for compatibility with an older caller;
            # all ignition locations are passed explicitly below.
            x_ignition=positions[0][0],
            y_ignition=positions[0][1],
            ignition_points=positions,
            dt=sim_config.dt,
            n_vertices=sim_config.n_vertices,
            initial_radius=radius,
            store_every=sim_config.store_every,
            use_markers=True,
            marker_epsilon=mm.epsilon,
            min_ros=sim_config.min_ros,
            is_geographic=False,
            advection_integrator=integrator,
            resample_spacing=float(spacing),
            insert_factor=getattr(mm, "insert_factor", 1.5),
            delete_factor=getattr(mm, "delete_factor", 0.5),
            smoothing_weight=getattr(mm, "smooth_weight", 0.0),
            smoothing_iters=getattr(mm, "smooth_iters", 0),
            redistribute_every=mm.redistribute_every if mm.redistribute_markers else 0,
            max_substeps=getattr(mm, "max_substeps", 50),
            max_move_fraction=getattr(mm, "max_move_fraction", 0.5),
        )
        history.marker_method = mm
        history.marker_epsilon = float(mm.epsilon)
        # Multi-marker topology is already resolved by the spread solver.
        # Do NOT run legacy single-ring untangling on these saved fronts.

    else:
        from ignacio.levelset import simulate_fire_spread_levelset

        ls = sim_config.level_set
        logger.info(
            "Simultaneous level-set experiment: %d ignitions, dt=%.4f min, CFL=%.3f",
            len(positions), float(sim_config.dt), float(ls.cfl_factor),
        )
        history = simulate_fire_spread_levelset(
            param_grid=param_grid,
            x_ignition=positions[0][0],
            y_ignition=positions[0][1],
            ignition_points=positions,
            dt=sim_config.dt,
            initial_radius=radius,
            store_every=sim_config.store_every,
            min_ros=sim_config.min_ros,
            is_geographic=False,
            cfl_factor=ls.cfl_factor,
            reinit_interval=ls.reinit_interval,
            reinit_iters=ls.reinit_iters,
            narrow_band_width=ls.narrow_band_width,
        )
        # No Shapely union or marker smoothing is applied to the evolving phi.
        # The solver extracts all phi=0 exterior fronts into fronts_by_time.

    if getattr(history, "fronts_by_time", None) is None:
        raise RuntimeError(
            f"The installed {method} solver returned a single-front history. "
            "Check that the revised ignacio/spread.py and ignacio/levelset.py "
            "are installed and provide multi-front support."
        )
    if not history.times or len(history.times) != len(history.fronts_by_time):
        raise RuntimeError("Multi-front history has no snapshots or misaligned timestamps")
    # Expect a distinct initial contour for each well-separated ignition.
    # A small grid-resolution allowance avoids misclassifying barely touching
    # circles that can connect under contour discretization.
    grid_resolution = max(abs(float(param_grid.dx)), abs(float(param_grid.dy)))
    if len(history.get_fronts(0)) != len(positions) and all(
        np.hypot(positions[i][0] - positions[j][0],
                 positions[i][1] - positions[j][1]) > 2 * radius + 2 * grid_resolution
        for i in range(len(positions)) for j in range(i + 1, len(positions))
    ):
        raise RuntimeError(
            f"Expected {len(positions)} initially separate fronts but found "
            f"{len(history.get_fronts(0))}. Check ignition and contour initialization."
        )

    # Compute statistics from ALL exterior rings at each saved time, avoiding
    # the invalid assumption that a disconnected fire is one closed ring.
    stats = []
    previous_area = 0.0
    for i, t in enumerate(history.times):
        geometry = _multi_geometry(history, i)
        area_ha = float(geometry.area) / 10000.0 if geometry is not None else 0.0
        perimeter_m = float(geometry.length) if geometry is not None else 0.0
        fronts = _history_fronts(history, i)
        ros_values = []
        t_idx = max(0, min(
            int(round(float(t) / float(sim_config.dt))), param_grid.nt - 1,
        ))
        for x, y in fronts:
            ros, _, _, _ = param_grid.sample_at(t_idx, x, y)
            ros = np.asarray(ros)
            ros_values.extend(ros[np.isfinite(ros) & (ros > 0)].tolist())
        stats.append({
            "area_ha": area_ha,
            "perim_m": perimeter_m,
            "darea_ha": max(0.0, area_ha - previous_area),
            "ros_mean": float(np.mean(ros_values)) if ros_values else 0.0,
            "ros_max": float(np.max(ros_values)) if ros_values else 0.0,
            "n_fronts": len(fronts),
        })
        previous_area = area_ha

    history.history_stats = stats
    final = stats[-1]
    logger.info(
        "%s combined fire: %d ignitions, %d snapshots, area %.4f ha, "
        "boundary %.2f m, %d merger event(s), %d topology repair(s)",
        method, len(positions), len(history.times), final["area_ha"],
        final["perim_m"], len(history.merger_events), len(history.topology_repairs),
    )
    return FireResult(
        ignition=ignitions[0],
        history=history,
        final_area_ha=float(final["area_ha"]),
        final_perimeter_m=float(final["perim_m"]),
        dur_min=float(history.times[-1]),
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
    simultaneous_points = _file_ignition_positions(config, terrain)
    if simultaneous_points:
        method = str(getattr(config.simulation, "spread_method", "marker")).strip().lower()
        if method not in {"marker", "levelset"}:
            raise ValueError(f"Unsupported multi-ignition spread method: {method!r}")
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

    if simultaneous_points:
        if len(ignitions.points) != len(simultaneous_points):
            raise ValueError("The ignition loader dropped one or more ignition points")
        fire_result = simulate_merged_fire(list(ignitions.points), param_grid, config)
        results.fires.append(fire_result)
        logger.info("Combined fire: %d ignitions, area %.4f ha, boundary %.1f m, "
                    "%d merger events, %d topology repairs",
                    len(simultaneous_points), fire_result.final_area_ha,
                    fire_result.final_perimeter_m,
                    len(fire_result.history.merger_events),
                    len(fire_result.history.topology_repairs))
    else:
        for i, ignition in enumerate(ignitions.points):
            logger.info(f"Fire {i+1}/{n_ignitions}: ({ignition.x:.6f}, {ignition.y:.6f})")
        
            # Diagnostic: what's the fuel and ROS at ignition location?
            try:
                ros_arr, _, _, _ = param_grid.sample_at(0, np.array([ignition.x]), np.array([ignition.y]))
                ros_at_ign = ros_arr[0]
            
                # Get grid indices for fuel lookup
                x_frac = (ignition.x - param_grid.x_min) / (param_grid.x_max - param_grid.x_min)
                x_idx = int(x_frac * (len(param_grid.x_coords) - 1))
                x_idx = max(0, min(x_idx, fuel_grid.shape[1] - 1))
            
                if param_grid.y_flipped:
                    y_frac = (param_grid.y_max - ignition.y) / (param_grid.y_max - param_grid.y_min)
                else:
                    y_frac = (ignition.y - param_grid.y_min) / (param_grid.y_max - param_grid.y_min)
            
                y_idx = int(y_frac * (len(param_grid.y_coords) - 1))
                y_idx = max(0, min(y_idx, fuel_grid.shape[0] - 1))
            
                fuel_at_ign = fuel_grid[y_idx, x_idx]
                fuel_name = config.fuel.fuel_lookup.get(int(fuel_at_ign), f"Unknown ({fuel_at_ign})")
                logger.info(f"  Fuel: {fuel_name}, ROS: {ros_at_ign:.2f} m/min")
            
                if ros_at_ign < 0.01:
                    logger.warning(f"  ROS at ignition is very low! Fire may not spread.")
            except Exception as e:
                logger.warning(f"  Could not get ignition diagnostics: {e}")
        
            fire_result = simulate_single_fire(
                ignition, param_grid, config,
                is_geographic=is_geographic,
                center_latitude=center_latitude,
            )
            results.fires.append(fire_result)
        
            if fire_result.final_area_ha > 0:
                logger.info(
                    f"  - Area: {fire_result.final_area_ha:.2f} ha, "
                    f"Perimeter: {fire_result.final_perimeter_m:.0f} m"
                )
    
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
    """Convert a single FireResult history into a GeoDataFrame.

    Produces one row per stored perimeter, with time metadata so the
    progression can be plotted/animated.

    Notes
    -----
    - `fire.history.times` is assumed to be minutes since ignition/start.
    - Use GeoPackage (GPKG) if you want robust datetime storage.
    """
    # In a multi-front run each time can have 2+ exterior rings. Emit one row
    # for each ring. All metadata remains in the projected CRS until GeoPandas
    # performs the requested reprojection for storage.
    if getattr(fire.history, "fronts_by_time", None) is not None:
        start_ts = pd.Timestamp(start_datetime) if start_datetime else None
        rows = []
        stats = getattr(fire.history, "history_stats", [])
        for step, t_min in enumerate(fire.history.times):
            fronts = _history_fronts(fire.history, step)
            st = stats[step] if step < len(stats) else {}
            for front_id, (x, y) in enumerate(fronts):
                poly = _front_polygon(x, y)
                when = (start_ts + pd.Timedelta(minutes=float(t_min))) if start_ts is not None else None
                rows.append({
                    "fire_index": int(fire_index),
                    "front_id": int(front_id),
                    "n_fronts": int(len(fronts)),
                    "n_ignitions": int(len(fire.ignition_positions or [])),
                    "step": int(step),
                    "t_min": float(t_min),
                    "cause": fire.ignition.cause,
                    "season": fire.ignition.season,
                    "iteration": int(fire.ignition.iteration),
                    "datetime": when.to_pydatetime() if when is not None else None,
                    "area_ha": float(poly.area / 10000.0),
                    "perim_m": float(poly.length),
                    "total_area_ha": float(st.get("area_ha", np.nan)),
                    "total_perim_m": float(st.get("perim_m", np.nan)),
                    "darea_ha": float(st.get("darea_ha", np.nan)),
                    "ros_mean": float(st.get("ros_mean", np.nan)),
                    "ros_max": float(st.get("ros_max", np.nan)),
                    "geometry": poly,
                })
        if not rows:
            return gpd.GeoDataFrame({"geometry": []}, crs=output_crs)
        gdf = gpd.GeoDataFrame(rows, geometry="geometry",
                               crs=source_crs if source_crs else output_crs)
        if source_crs and source_crs != output_crs:
            gdf = gdf.to_crs(output_crs)
        return gdf

    perims = getattr(fire.history, "perimeters", None) or []
    times = getattr(fire.history, "times", None) or []

    n = min(len(perims), len(times))
    if n <= 0:
        return gpd.GeoDataFrame({"geometry": []}, crs=output_crs)

    # Optional absolute datetime column
    start_ts = pd.Timestamp(start_datetime) if start_datetime else None

    rows: list[dict[str, Any]] = []
    for step in range(n):
        perim = perims[step]
        ring = _perim_to_ring(perim)
        if not ring or len(ring) < 4:
            continue

        ring = _ensure_ring_closed(list(ring), eps=0.0)

        # Level set contours are topologically clean — skip cleaning/smoothing
        _is_level_set = (fire.history is not None
                         and not hasattr(fire.history, "marker_method")
                         and not hasattr(fire.history, "marker_epsilon"))

        if _is_level_set:
            poly = Polygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda g: g.area)
        else:
            # Clean/repair polygon using same utilities as final-perimeter export
            eps = 0.0
            try:
                if fire.history is not None and hasattr(fire.history, "marker_epsilon"):
                    eps = float(getattr(fire.history, "marker_epsilon") or 0.0)
            except Exception:
                eps = 0.0

            simplify_tol = (0.5 * eps) if eps and eps > 0 else None

            poly = _clean_polygon_from_ring(
                ring,
                eps=eps,
                simplify_tol=simplify_tol,
                resample_spacing=(
                    float(getattr(getattr(fire.history, "marker_method", None), "resample_spacing", 0.0) or 0.0)
                    if hasattr(fire.history, "marker_method")
                    else (eps if eps and eps > 0 else None)
                ),
                chaikin_iters=(
                    int(getattr(getattr(fire.history, "marker_method", None), "chaikin_iters", 0) or 0)
                    if hasattr(fire.history, "marker_method")
                    else 1
                ),
            )
            if poly is None:
                poly = Polygon(ring)

        if poly.is_empty or poly.area <= 0:
            continue

        # Optional per-timestep stats computed during simulation
        area_ha = None
        perim_m = None
        darea_ha = None
        ros_mean = None
        ros_max = None
        try:
            stats_list = getattr(fire.history, "history_stats", None)
            if isinstance(stats_list, list) and step < len(stats_list):
                st = stats_list[step] or {}
                area_ha = st.get("area_ha")
                perim_m = st.get("perim_m")
                darea_ha = st.get("darea_ha")
                ros_mean = st.get("ros_mean")
                ros_max = st.get("ros_max")
        except Exception:
            pass

        t_min = float(times[step])
        dt_abs = (start_ts + pd.Timedelta(minutes=t_min)) if start_ts is not None else None

        rows.append(
            {
                "fire_index": int(fire_index),
                "step": int(step),
                "t_min": t_min,
                "cause": fire.ignition.cause,
                "season": fire.ignition.season,
                "iteration": int(fire.ignition.iteration),
                "datetime": dt_abs.to_pydatetime() if dt_abs is not None else None,
                "area_ha": float(area_ha) if area_ha is not None else None,
                "perim_m": float(perim_m) if perim_m is not None else None,
                "darea_ha": float(darea_ha) if darea_ha is not None else None,
                "ros_mean": float(ros_mean) if ros_mean is not None else None,
                "ros_max": float(ros_max) if ros_max is not None else None,
                "geometry": poly,
            }
        )

    gdf = gpd.GeoDataFrame(rows, crs=source_crs if source_crs else output_crs)
    if source_crs and source_crs != output_crs and len(gdf) > 0:
        gdf = gdf.to_crs(output_crs)

    return gdf


def _save_results(
    results: SimulationResults,
    output_dir: Path,
    config: IgnacioConfig,
) -> None:
    """Save simulation results to files."""
    # Get source CRS from terrain
    source_crs = None
    if results.terrain and results.terrain.crs:
        source_crs = str(results.terrain.crs)
    
    # Save individual perimeters
    perimeters_dir = output_dir / "perimeters"
    perimeters_dir.mkdir(exist_ok=True)
    
    for i, fire in enumerate(results.fires):
        if fire.final_area_ha > 0:
            gdf = fire.get_final_polygon(config.crs.output_crs, source_crs=source_crs)
            
            # Determine format
            fmt = config.output.perimeter_format
            if fmt == "shapefile":
                path = perimeters_dir / f"fire_{i:04d}.shp"
                driver = "ESRI Shapefile"
            elif fmt == "geojson":
                path = perimeters_dir / f"fire_{i:04d}.geojson"
                driver = "GeoJSON"
            else:
                path = perimeters_dir / f"fire_{i:04d}.gpkg"
                driver = "GPKG"
            
            write_vector(gdf, path, driver=driver)
    
    # Save combined perimeters
    if config.output.combine_perimeters:
        all_perimeters = results.get_all_perimeters(source_crs=source_crs)
        if len(all_perimeters) > 0:
            fmt = config.output.perimeter_format
            if fmt == "shapefile":
                combined_path = output_dir / "all_perimeters.shp"
            elif fmt == "geojson":
                combined_path = output_dir / "all_perimeters.geojson"
            else:
                combined_path = output_dir / "all_perimeters.gpkg"
            
            write_vector(all_perimeters, combined_path)
            logger.info(f"Saved combined perimeters to {combined_path}")

    # Save combined perimeter HISTORY (all fires, all stored timesteps)
    # This enables plotting/animation of fire progression over time.
    # NOTE: Shapefile has strict field limitations; prefer GPKG.
    history_frames: list[gpd.GeoDataFrame] = []
    for i, fire in enumerate(results.fires):
        # Include fires even if the final area is small, as long as there is history
        if not getattr(fire.history, "perimeters", None) or not getattr(fire.history, "times", None):
            continue

        gdf_h = perimeter_history_to_gdf(
            fire,
            fire_index=i,
            output_crs=config.crs.output_crs,
            source_crs=source_crs,
            start_datetime=getattr(config.simulation, "start_datetime", None),
        )
        if len(gdf_h) > 0:
            history_frames.append(gdf_h)

    if history_frames:
        gdf_hist_all = gpd.GeoDataFrame(pd.concat(history_frames, ignore_index=True), crs=config.crs.output_crs)

        # Force a robust container for time fields regardless of perimeter_format
        hist_path = output_dir / "all_perimeters_history.gpkg"
        write_vector(gdf_hist_all, hist_path, driver="GPKG")
        logger.info(f"Saved combined perimeter history to {hist_path}")

        # Also save an attribute-only table for the history (no geometry).
        # This is convenient when final perimeters are stored as Shapefiles.
        hist_attr = gdf_hist_all.drop(columns=["geometry"], errors="ignore").copy()

        # Make datetime CSV-friendly
        if "datetime" in hist_attr.columns:
            hist_attr["datetime"] = (
                pd.to_datetime(hist_attr["datetime"], errors="coerce")
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )

        hist_attr_path = output_dir / "all_perimeters_history_attributes.csv"
        hist_attr.to_csv(hist_attr_path, index=False)
        logger.info(f"Saved perimeter history attribute table to {hist_attr_path}")
    else:
        logger.info("No perimeter history to save (no stored perimeters/times found)")
    
    # Additional diagnostic tables for simultaneous ignition experiments.
    for i, fire in enumerate(results.fires):
        history = fire.history
        if getattr(history, "fronts_by_time", None) is None:
            continue
        stats_rows = []
        for step, t in enumerate(history.times):
            st = dict(history.history_stats[step]) if step < len(history.history_stats) else {}
            stats_rows.append({"step": step, "t_min": float(t), **st})
        pd.DataFrame(stats_rows).to_csv(
            output_dir / f"fire_{i:04d}_history_totals.csv", index=False)
        if history.merger_events:
            pd.DataFrame(history.merger_events).to_csv(
                output_dir / f"fire_{i:04d}_merger_events.csv", index=False)
        if history.topology_repairs:
            pd.DataFrame(history.topology_repairs).to_csv(
                output_dir / f"fire_{i:04d}_topology_repairs.csv", index=False)

    # Save summary CSV
    summary = results.get_summary()
    summary_path = output_dir / "simulation_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved simulation summary to {summary_path}")
