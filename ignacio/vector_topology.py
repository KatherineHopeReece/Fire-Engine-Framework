"""
Vector Topology Handler for Fire Perimeter Processing.

This module provides topology-aware processing of fire perimeters to:
1. Fix self-intersections in vector polygons
2. Ensure valid geometry for GIS operations
3. Provide Prometheus-compatible output

Uses Shapely for robust polygon operations.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

try:
    from shapely.geometry import Polygon, MultiPolygon, Point, LineString
    from shapely.ops import unary_union
    from shapely.validation import make_valid, explain_validity
    from shapely import BufferCapStyle, BufferJoinStyle
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FirePerimeter:
    """A fire perimeter with optional holes (islands)."""
    exterior: np.ndarray  # (N, 2) - exterior ring coordinates
    holes: List[np.ndarray] = None  # List of (M, 2) arrays for holes
    timestamp: float = 0.0  # Simulation time (minutes)
    area: float = 0.0  # Area in m²
    perimeter_length: float = 0.0  # Perimeter in m
    is_valid: bool = True
    
    def __post_init__(self):
        if self.holes is None:
            self.holes = []


@dataclass
class TopologyResult:
    """Result of topology operations."""
    perimeters: List[FirePerimeter]
    was_repaired: bool = False
    original_area: float = 0.0
    repaired_area: float = 0.0
    issues_found: List[str] = None
    
    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []


# =============================================================================
# Topology Methods
# =============================================================================

def coords_to_polygon(coords: np.ndarray) -> Optional[Polygon]:
    """Convert coordinate array to Shapely Polygon."""
    if not HAS_SHAPELY:
        raise ImportError("Shapely is required for vector topology operations")
    
    if coords is None or len(coords) < 3:
        return None
    
    # Ensure ring is closed
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    try:
        return Polygon(coords)
    except Exception:
        return None


def polygon_to_coords(polygon: Polygon) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Convert Shapely Polygon to coordinate arrays."""
    exterior = np.array(polygon.exterior.coords)
    holes = [np.array(hole.coords) for hole in polygon.interiors]
    return exterior, holes


def check_validity(coords: np.ndarray) -> Tuple[bool, str]:
    """Check if polygon coordinates form a valid geometry."""
    if not HAS_SHAPELY:
        return True, "Shapely not available for validation"
    
    poly = coords_to_polygon(coords)
    if poly is None:
        return False, "Could not create polygon"
    
    if poly.is_valid:
        return True, "Valid"
    else:
        return False, explain_validity(poly)


# =============================================================================
# Topology Repair Methods
# =============================================================================

def repair_buffer_zero(coords: np.ndarray) -> np.ndarray:
    """
    Repair self-intersections using buffer(0) trick.
    
    This is a fast method that works for simple self-intersections.
    """
    if not HAS_SHAPELY:
        return coords
    
    poly = coords_to_polygon(coords)
    if poly is None:
        return coords
    
    if poly.is_valid:
        return coords
    
    # Buffer(0) often fixes simple issues
    try:
        fixed = poly.buffer(0)
        if isinstance(fixed, MultiPolygon):
            # Take largest polygon
            fixed = max(fixed.geoms, key=lambda p: p.area)
        return np.array(fixed.exterior.coords)
    except Exception:
        return coords


def repair_make_valid(coords: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Repair using Shapely's make_valid function.
    
    This handles more complex topology issues.
    """
    if not HAS_SHAPELY:
        return coords, []
    
    poly = coords_to_polygon(coords)
    if poly is None:
        return coords, []
    
    if poly.is_valid:
        return coords, []
    
    try:
        fixed = make_valid(poly)
        
        if isinstance(fixed, Polygon):
            return polygon_to_coords(fixed)
        elif isinstance(fixed, MultiPolygon):
            # Take largest polygon
            largest = max(fixed.geoms, key=lambda p: p.area)
            return polygon_to_coords(largest)
        else:
            return coords, []
    except Exception:
        return coords, []


def repair_convex_hull(coords: np.ndarray) -> np.ndarray:
    """
    Replace with convex hull (extreme simplification).
    
    Use only when other methods fail.
    """
    if not HAS_SHAPELY:
        return coords
    
    try:
        poly = coords_to_polygon(coords)
        if poly is None:
            return coords
        hull = poly.convex_hull
        return np.array(hull.exterior.coords)
    except Exception:
        return coords


# =============================================================================
# Shapely Clipping (Prometheus-compatible)
# =============================================================================

def clip_perimeter(
    coords: np.ndarray,
    method: str = 'shapely_clip'
) -> FirePerimeter:
    """
    Process a fire perimeter with topology correction.
    
    Parameters
    ----------
    coords : array (N, 2)
        Perimeter coordinates
    method : str
        Topology correction method:
        - 'none': No correction
        - 'buffer_zero': Simple buffer(0) fix
        - 'shapely_clip': Full make_valid repair
        - 'convex_hull': Replace with convex hull
        
    Returns
    -------
    FirePerimeter
        Processed perimeter with valid topology
    """
    if method == 'none' or not HAS_SHAPELY:
        return FirePerimeter(
            exterior=coords,
            holes=[],
            is_valid=True,  # Assume valid if not checking
        )
    
    # Check validity
    is_valid, issue = check_validity(coords)
    
    if is_valid:
        poly = coords_to_polygon(coords)
        return FirePerimeter(
            exterior=coords,
            holes=[],
            area=float(poly.area) if poly else 0.0,
            perimeter_length=float(poly.length) if poly else 0.0,
            is_valid=True,
        )
    
    # Apply repair method
    if method == 'buffer_zero':
        repaired = repair_buffer_zero(coords)
        holes = []
    elif method == 'shapely_clip':
        repaired, holes = repair_make_valid(coords)
    elif method == 'convex_hull':
        repaired = repair_convex_hull(coords)
        holes = []
    elif method == 'vatti':
        # Vatti is essentially what Shapely uses internally
        repaired, holes = repair_make_valid(coords)
    else:
        repaired = coords
        holes = []
    
    # Compute metrics
    poly = coords_to_polygon(repaired)
    
    return FirePerimeter(
        exterior=repaired,
        holes=holes,
        area=float(poly.area) if poly else 0.0,
        perimeter_length=float(poly.length) if poly else 0.0,
        is_valid=poly.is_valid if poly else False,
    )


def merge_perimeters(perimeters: List[FirePerimeter]) -> FirePerimeter:
    """
    Merge multiple fire perimeters into one.
    
    Handles overlapping fires that have merged.
    """
    if not HAS_SHAPELY:
        # Without Shapely, just return the largest
        return max(perimeters, key=lambda p: len(p.exterior))
    
    polygons = []
    for p in perimeters:
        poly = coords_to_polygon(p.exterior)
        if poly and poly.is_valid:
            polygons.append(poly)
    
    if not polygons:
        return perimeters[0] if perimeters else FirePerimeter(np.array([]))
    
    # Union all polygons
    merged = unary_union(polygons)
    
    if isinstance(merged, MultiPolygon):
        # Keep all parts
        all_perimeters = []
        for geom in merged.geoms:
            ext, holes = polygon_to_coords(geom)
            all_perimeters.append(FirePerimeter(
                exterior=ext,
                holes=holes,
                area=float(geom.area),
                perimeter_length=float(geom.length),
                is_valid=True,
            ))
        # Return largest
        return max(all_perimeters, key=lambda p: p.area)
    else:
        ext, holes = polygon_to_coords(merged)
        return FirePerimeter(
            exterior=ext,
            holes=holes,
            area=float(merged.area),
            perimeter_length=float(merged.length),
            is_valid=True,
        )


# =============================================================================
# Level-Set to Vector Conversion
# =============================================================================

def levelset_to_perimeter(
    phi: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    level: float = 0.0,
    simplify_tolerance: float = 1.0,
    topology_method: str = 'shapely_clip',
) -> List[FirePerimeter]:
    """
    Extract fire perimeter(s) from level-set function.
    
    Parameters
    ----------
    phi : array (ny, nx)
        Level-set function (negative = burned)
    x_coords, y_coords : array
        Grid coordinates
    level : float
        Contour level (usually 0)
    simplify_tolerance : float
        Simplification tolerance in grid units
    topology_method : str
        Topology correction method
        
    Returns
    -------
    List[FirePerimeter]
        Extracted and cleaned perimeters
    """
    try:
        from skimage import measure
    except ImportError:
        # Fallback: simple boundary detection
        return _simple_boundary_extraction(phi, x_coords, y_coords, topology_method)
    
    # Find contours at zero level
    contours = measure.find_contours(phi, level)
    
    if not contours:
        return []
    
    perimeters = []
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0
    
    for contour in contours:
        # Convert to real coordinates
        # Contours are in (row, col) = (y, x) order
        real_coords = np.column_stack([
            x_coords[0] + contour[:, 1] * dx,
            y_coords[0] + contour[:, 0] * dy,
        ])
        
        # Close the polygon if needed
        if not np.allclose(real_coords[0], real_coords[-1]):
            real_coords = np.vstack([real_coords, real_coords[0]])
        
        # Simplify if requested
        if simplify_tolerance > 0 and HAS_SHAPELY:
            poly = coords_to_polygon(real_coords)
            if poly and poly.is_valid:
                simplified = poly.simplify(simplify_tolerance * dx)
                real_coords = np.array(simplified.exterior.coords)
        
        # Apply topology correction
        perim = clip_perimeter(real_coords, topology_method)
        if len(perim.exterior) >= 3:
            perimeters.append(perim)
    
    return perimeters


def _simple_boundary_extraction(
    phi: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    topology_method: str,
) -> List[FirePerimeter]:
    """Simple boundary extraction without skimage."""
    # Find boundary cells (where phi changes sign)
    burned = phi < 0
    
    # Create boundary mask
    boundary = np.zeros_like(burned, dtype=bool)
    boundary[1:, :] |= (burned[1:, :] != burned[:-1, :])
    boundary[:-1, :] |= (burned[:-1, :] != burned[1:, :])
    boundary[:, 1:] |= (burned[:, 1:] != burned[:, :-1])
    boundary[:, :-1] |= (burned[:, :-1] != burned[:, 1:])
    
    # Get boundary points
    y_idx, x_idx = np.where(boundary & burned)
    
    if len(x_idx) == 0:
        return []
    
    # Convert to coordinates
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0
    
    points = np.column_stack([
        x_coords[0] + x_idx * dx,
        y_coords[0] + y_idx * dy,
    ])
    
    # Order points to form perimeter (simple convex hull)
    if HAS_SHAPELY and len(points) >= 3:
        from shapely.geometry import MultiPoint
        mp = MultiPoint(points)
        hull = mp.convex_hull
        if isinstance(hull, Polygon):
            perim = clip_perimeter(np.array(hull.exterior.coords), topology_method)
            return [perim]
    
    return []


# =============================================================================
# Prometheus Format Export
# =============================================================================

def export_prometheus_format(
    perimeters: List[FirePerimeter],
    output_path: str,
    crs: str = "EPSG:32611",
) -> None:
    """
    Export perimeters in Prometheus-compatible format.
    
    Prometheus uses WKT or shapefile format.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        geometries = []
        for i, p in enumerate(perimeters):
            if len(p.exterior) >= 3:
                poly = Polygon(p.exterior, p.holes if p.holes else None)
                if poly.is_valid:
                    geometries.append({
                        'geometry': poly,
                        'id': i,
                        'time_min': p.timestamp,
                        'area_m2': p.area,
                        'perim_m': p.perimeter_length,
                    })
        
        if geometries:
            gdf = gpd.GeoDataFrame(geometries, crs=crs)
            
            if output_path.endswith('.shp'):
                gdf.to_file(output_path)
            elif output_path.endswith('.geojson'):
                gdf.to_file(output_path, driver='GeoJSON')
            else:
                gdf.to_file(output_path + '.shp')
                
    except ImportError:
        # Fallback: simple WKT export
        with open(output_path + '.wkt', 'w') as f:
            for i, p in enumerate(perimeters):
                wkt = f"POLYGON(({', '.join(f'{x} {y}' for x, y in p.exterior)}))"
                f.write(f"{i},{p.timestamp},{wkt}\n")


# =============================================================================
# Topology Processing Pipeline
# =============================================================================

def process_fire_perimeters(
    phi_history: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    times: np.ndarray,
    topology_method: str = 'shapely_clip',
    simplify_tolerance: float = 1.0,
) -> List[List[FirePerimeter]]:
    """
    Process a time series of level-set fields into clean perimeters.
    
    Parameters
    ----------
    phi_history : array (T, ny, nx)
        Time series of level-set function
    x_coords, y_coords : array
        Grid coordinates
    times : array (T,)
        Time values in minutes
    topology_method : str
        Topology correction method
    simplify_tolerance : float
        Simplification tolerance
        
    Returns
    -------
    List[List[FirePerimeter]]
        Perimeters for each time step
    """
    all_perimeters = []
    
    for t_idx, (phi, t) in enumerate(zip(phi_history, times)):
        perims = levelset_to_perimeter(
            phi, x_coords, y_coords,
            simplify_tolerance=simplify_tolerance,
            topology_method=topology_method,
        )
        
        # Add timestamp
        for p in perims:
            p.timestamp = float(t)
        
        all_perimeters.append(perims)
    
    return all_perimeters


def generate_isochrones(
    phi_history: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    times: np.ndarray,
    interval_minutes: float = 30.0,
    topology_method: str = 'shapely_clip',
) -> List[FirePerimeter]:
    """
    Generate fire isochrones (progression lines) at regular intervals.
    
    Parameters
    ----------
    phi_history : array (T, ny, nx)
        Time series of level-set function
    x_coords, y_coords : array
        Grid coordinates
    times : array (T,)
        Time values in minutes
    interval_minutes : float
        Interval between isochrones
    topology_method : str
        Topology correction method
        
    Returns
    -------
    List[FirePerimeter]
        Isochrone perimeters
    """
    isochrones = []
    
    # Find indices at regular intervals
    interval_times = np.arange(times[0], times[-1], interval_minutes)
    
    for target_time in interval_times:
        # Find closest time index
        idx = np.argmin(np.abs(times - target_time))
        
        perims = levelset_to_perimeter(
            phi_history[idx], x_coords, y_coords,
            topology_method=topology_method,
        )
        
        for p in perims:
            p.timestamp = float(times[idx])
            isochrones.append(p)
    
    return isochrones
