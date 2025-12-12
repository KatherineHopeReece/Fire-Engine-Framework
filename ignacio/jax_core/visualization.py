"""
Visualization utilities for fire spread comparison.

Provides functions to compare simulated fire perimeters/areas with
observed fire boundaries.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import transform
    import pyproj
    HAS_GEO = True
except ImportError:
    HAS_GEO = False


def load_observed_fire(
    shapefile_path: Union[str, Path],
    target_crs: Optional[str] = None,
) -> 'gpd.GeoDataFrame':
    """
    Load observed fire perimeter from shapefile.
    
    Parameters
    ----------
    shapefile_path : str or Path
        Path to shapefile containing fire perimeter
    target_crs : str, optional
        Target CRS to reproject to (e.g., 'EPSG:4326')
        
    Returns
    -------
    gdf : GeoDataFrame
        Fire perimeter geometry
    """
    if not HAS_GEO:
        raise ImportError("geopandas required for load_observed_fire")
    
    from shapely import force_2d
    from shapely.validation import make_valid
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon
    
    gdf = gpd.read_file(shapefile_path)
    
    def fix_geometry(geom):
        """Fix geometry issues: strip Z, validate, extract polygon."""
        if geom is None or geom.is_empty:
            return geom
        
        # Strip Z coordinates
        geom = force_2d(geom)
        
        # Fix invalid geometry
        if not geom.is_valid:
            geom = make_valid(geom)
        
        # Handle geometry collections - extract largest polygon
        if geom.geom_type == 'GeometryCollection':
            polys = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
            if polys:
                geom = max(polys, key=lambda p: p.area)
            else:
                return ShapelyPolygon()  # Return empty polygon
        
        # Handle MultiPolygon - take largest
        if geom.geom_type == 'MultiPolygon':
            geom = max(geom.geoms, key=lambda p: p.area)
        
        # Final buffer(0) to clean up any remaining issues
        if not geom.is_valid:
            geom = geom.buffer(0)
        
        return geom
    
    gdf['geometry'] = gdf['geometry'].apply(fix_geometry)
    
    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
        # Re-validate after reprojection
        gdf['geometry'] = gdf['geometry'].apply(fix_geometry)
    
    return gdf


def perimeter_to_polygon(x: np.ndarray, y: np.ndarray) -> 'Polygon':
    """Convert perimeter coordinates to Shapely polygon."""
    if not HAS_GEO:
        raise ImportError("shapely required for perimeter_to_polygon")
    
    from shapely.validation import make_valid
    
    coords = list(zip(x, y))
    if coords[0] != coords[-1]:
        coords.append(coords[0])  # Close polygon
    
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type == 'GeometryCollection':
                polys = [g for g in poly.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
                poly = max(polys, key=lambda p: p.area) if polys else Polygon()
            elif poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda p: p.area)
        return poly
    except Exception:
        return Polygon()


def levelset_to_polygon(
    phi: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> 'Polygon':
    """
    Convert level-set field to polygon.
    
    Parameters
    ----------
    phi : np.ndarray
        Signed distance field (ny, nx), negative = burned
    x_coords, y_coords : np.ndarray
        1D coordinate arrays
        
    Returns
    -------
    polygon : Polygon
        Burned area as polygon
    """
    if not HAS_GEO:
        raise ImportError("shapely required for levelset_to_polygon")
    
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("scikit-image required for levelset_to_polygon")
    
    from shapely.validation import make_valid
    
    # Find contours at phi = 0
    contours = measure.find_contours(phi, 0.0)
    
    if len(contours) == 0:
        return Polygon()
    
    # Get largest contour
    largest = max(contours, key=len)
    
    # Convert from array indices to coordinates
    rows, cols = largest[:, 0], largest[:, 1]
    
    # Interpolate to coordinates
    x = np.interp(cols, np.arange(len(x_coords)), x_coords)
    y = np.interp(rows, np.arange(len(y_coords)), y_coords)
    
    coords = list(zip(x, y))
    if len(coords) < 3:
        return Polygon()
    
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            # Try to fix invalid geometry
            poly = make_valid(poly)
            # make_valid can return GeometryCollection, extract polygon
            if poly.geom_type == 'GeometryCollection':
                polys = [g for g in poly.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
                if polys:
                    poly = max(polys, key=lambda p: p.area)
                else:
                    poly = Polygon()
            elif poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda p: p.area)
        return poly
    except Exception:
        return Polygon()


def compute_metrics(
    simulated: 'Polygon',
    observed: 'Polygon',
) -> dict:
    """
    Compute comparison metrics between simulated and observed fire.
    
    Parameters
    ----------
    simulated, observed : Polygon
        Fire perimeter polygons
        
    Returns
    -------
    metrics : dict
        Dictionary with:
        - area_simulated: Simulated area
        - area_observed: Observed area
        - area_diff: Absolute difference
        - area_diff_pct: Percentage difference
        - intersection: Area of intersection
        - union: Area of union
        - iou: Intersection over Union (Jaccard index)
        - precision: Intersection / Simulated (how much of sim is correct)
        - recall: Intersection / Observed (how much of obs is captured)
        - f1: Harmonic mean of precision and recall
    """
    if not HAS_GEO:
        raise ImportError("shapely required for compute_metrics")
    
    from shapely.validation import make_valid
    
    # Handle empty geometries
    if simulated.is_empty or observed.is_empty:
        return {
            'area_simulated': simulated.area if not simulated.is_empty else 0,
            'area_observed': observed.area if not observed.is_empty else 0,
            'area_diff': float('inf'),
            'area_diff_pct': float('inf'),
            'intersection': 0,
            'union': 0,
            'iou': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
        }
    
    # Ensure geometries are valid
    if not simulated.is_valid:
        simulated = make_valid(simulated)
        if simulated.geom_type == 'GeometryCollection':
            polys = [g for g in simulated.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
            simulated = max(polys, key=lambda p: p.area) if polys else Polygon()
        elif simulated.geom_type == 'MultiPolygon':
            simulated = max(simulated.geoms, key=lambda p: p.area)
    
    if not observed.is_valid:
        observed = make_valid(observed)
        if observed.geom_type == 'GeometryCollection':
            polys = [g for g in observed.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
            observed = max(polys, key=lambda p: p.area) if polys else Polygon()
        elif observed.geom_type == 'MultiPolygon':
            observed = max(observed.geoms, key=lambda p: p.area)
    
    try:
        intersection = simulated.intersection(observed).area
        union = simulated.union(observed).area
    except Exception as e:
        # If intersection/union fails, try buffer(0) as last resort
        try:
            simulated = simulated.buffer(0)
            observed = observed.buffer(0)
            intersection = simulated.intersection(observed).area
            union = simulated.union(observed).area
        except Exception:
            # Give up on spatial metrics, just report area
            return {
                'area_simulated': simulated.area,
                'area_observed': observed.area,
                'area_diff': abs(simulated.area - observed.area),
                'area_diff_pct': abs(simulated.area - observed.area) / observed.area * 100 if observed.area > 0 else float('inf'),
                'intersection': 0,
                'union': 0,
                'iou': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
            }
    
    area_sim = simulated.area
    area_obs = observed.area
    
    iou = intersection / union if union > 0 else 0
    precision = intersection / area_sim if area_sim > 0 else 0
    recall = intersection / area_obs if area_obs > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'area_simulated': area_sim,
        'area_observed': area_obs,
        'area_diff': abs(area_sim - area_obs),
        'area_diff_pct': abs(area_sim - area_obs) / area_obs * 100 if area_obs > 0 else float('inf'),
        'intersection': intersection,
        'union': union,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def plot_comparison(
    observed_gdf: 'gpd.GeoDataFrame',
    numpy_perimeter: Optional[tuple] = None,
    jax_levelset: Optional[tuple] = None,
    dem: Optional[np.ndarray] = None,
    dem_extent: Optional[tuple] = None,
    title: str = "Fire Spread Comparison",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 10),
    show_metrics: bool = True,
) -> 'plt.Figure':
    """
    Create comparison plot of observed vs simulated fires.
    
    Parameters
    ----------
    observed_gdf : GeoDataFrame
        Observed fire perimeter
    numpy_perimeter : tuple, optional
        (x, y) arrays for NumPy perimeter simulation
    jax_levelset : tuple, optional
        (phi, x_coords, y_coords) for JAX level-set simulation
    dem : np.ndarray, optional
        DEM for background hillshade
    dem_extent : tuple, optional
        (xmin, xmax, ymin, ymax) for DEM
    title : str
        Plot title
    output_path : str or Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    show_metrics : bool
        Whether to show comparison metrics
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_comparison")
    if not HAS_GEO:
        raise ImportError("geopandas required for plot_comparison")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get bounds from observed fire
    bounds = observed_gdf.total_bounds  # xmin, ymin, xmax, ymax
    pad = 0.1 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    
    # Plot DEM hillshade if provided
    if dem is not None and dem_extent is not None:
        # Simple hillshade
        from numpy import gradient
        dx, dy = gradient(dem)
        slope = np.sqrt(dx**2 + dy**2)
        aspect = np.arctan2(-dx, dy)
        
        # Sun from northwest
        azimuth = 315 * np.pi / 180
        altitude = 45 * np.pi / 180
        
        hillshade = (np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect) +
                     np.sin(altitude) * np.cos(slope))
        hillshade = np.clip(hillshade, 0, 1)
        
        ax.imshow(hillshade, extent=dem_extent, cmap='gray', alpha=0.5,
                  origin='upper', aspect='auto')
    
    # Plot observed fire
    observed_geom = observed_gdf.geometry.iloc[0]
    if isinstance(observed_geom, (Polygon, MultiPolygon)):
        observed_gdf.plot(ax=ax, facecolor='none', edgecolor='black', 
                         linewidth=2.5, label='Observed')
    
    metrics_text = []
    
    # Plot NumPy perimeter
    if numpy_perimeter is not None:
        x_np, y_np = numpy_perimeter
        ax.plot(x_np, y_np, 'b-', linewidth=2, label='NumPy (perimeter)')
        ax.plot([x_np[-1], x_np[0]], [y_np[-1], y_np[0]], 'b-', linewidth=2)
        
        if show_metrics:
            np_poly = perimeter_to_polygon(x_np, y_np)
            np_metrics = compute_metrics(np_poly, observed_geom)
            metrics_text.append(f"NumPy: IoU={np_metrics['iou']:.2%}, Area diff={np_metrics['area_diff_pct']:.1f}%")
    
    # Plot JAX level-set
    if jax_levelset is not None:
        phi, x_coords, y_coords = jax_levelset
        
        # Plot burned area as filled contour
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Create custom colormap (transparent to red)
        colors = [(1, 0, 0, 0), (1, 0, 0, 0.4)]  # transparent to semi-transparent red
        cmap = LinearSegmentedColormap.from_list('fire', colors)
        
        # Fill burned area
        ax.contourf(X, Y, -phi, levels=[0, 1e10], colors=['red'], alpha=0.3)
        
        # Draw contour line
        ax.contour(X, Y, phi, levels=[0], colors=['red'], linewidths=2)
        
        # Add to legend
        ax.plot([], [], 'r-', linewidth=2, label='JAX Level-set')
        
        if show_metrics:
            ls_poly = levelset_to_polygon(phi, x_coords, y_coords)
            ls_metrics = compute_metrics(ls_poly, observed_geom)
            metrics_text.append(f"Level-set: IoU={ls_metrics['iou']:.2%}, Area diff={ls_metrics['area_diff_pct']:.1f}%")
    
    # Set bounds
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    
    # Labels and legend
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    # Add metrics text box
    if show_metrics and metrics_text:
        textstr = '\n'.join(metrics_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    
    return fig


def plot_calibration_progress(
    losses: list,
    params_history: Optional[list] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> 'plt.Figure':
    """
    Plot calibration optimization progress.
    
    Parameters
    ----------
    losses : list
        Loss values at each iteration
    params_history : list, optional
        List of parameter dicts at each iteration
    output_path : str or Path, optional
        Path to save figure
        
    Returns
    -------
    fig : Figure
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_calibration_progress")
    
    n_plots = 1 + (1 if params_history else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Loss plot
    axes[0].plot(losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Calibration Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Parameters plot
    if params_history:
        ax = axes[1]
        param_names = list(params_history[0].keys())
        for name in param_names:
            values = [p[name] for p in params_history]
            ax.plot(values, label=name, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration plot to {output_path}")
    
    return fig
