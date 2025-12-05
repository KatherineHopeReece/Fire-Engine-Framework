"""
Visualization module for Ignacio.

This module provides functions for plotting fire perimeters, ROS grids,
and generating animations of fire spread.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon

if TYPE_CHECKING:
    from ignacio.spread import FirePerimeterHistory
    from ignacio.terrain import TerrainGrids

logger = logging.getLogger(__name__)


# =============================================================================
# Color Maps and Styles
# =============================================================================

# Fire perimeter colors by time
FIRE_CMAP = cm.hot_r

# Terrain color maps
DEM_CMAP = cm.terrain
SLOPE_CMAP = cm.YlOrRd
ASPECT_CMAP = cm.hsv

# ROS color map
ROS_CMAP = cm.inferno


# =============================================================================
# Static Plotting Functions
# =============================================================================


def plot_terrain(
    terrain: TerrainGrids,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (15, 5),
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot terrain grids (DEM, slope, aspect).
    
    Parameters
    ----------
    terrain : TerrainGrids
        Terrain data container.
    output_path : Path, optional
        If provided, save figure to this path.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Figure resolution.
        
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    x_coords, y_coords = terrain.get_coordinate_arrays()
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    
    # DEM
    ax = axes[0]
    dem_data = terrain.dem.data
    im = ax.imshow(
        dem_data,
        extent=extent,
        origin="upper",
        cmap=DEM_CMAP,
        aspect="equal",
    )
    ax.set_title("Elevation (m)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    
    # Slope
    ax = axes[1]
    im = ax.imshow(
        terrain.slope_deg,
        extent=extent,
        origin="upper",
        cmap=SLOPE_CMAP,
        aspect="equal",
        vmin=0,
        vmax=45,
    )
    ax.set_title("Slope (degrees)")
    ax.set_xlabel("Easting (m)")
    plt.colorbar(im, ax=ax, label="Slope (deg)")
    
    # Aspect
    ax = axes[2]
    im = ax.imshow(
        terrain.aspect_deg,
        extent=extent,
        origin="upper",
        cmap=ASPECT_CMAP,
        aspect="equal",
        vmin=0,
        vmax=360,
    )
    ax.set_title("Aspect (degrees from N)")
    ax.set_xlabel("Easting (m)")
    plt.colorbar(im, ax=ax, label="Aspect (deg)")
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved terrain plot to {output_path}")
    
    return fig


def plot_ros_grid(
    ros_grid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    output_path: Path | None = None,
    title: str = "Rate of Spread (m/min)",
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
    vmax: float | None = None,
) -> plt.Figure:
    """
    Plot rate of spread grid.
    
    Parameters
    ----------
    ros_grid : np.ndarray
        2D ROS grid.
    x_coords : np.ndarray
        X coordinates.
    y_coords : np.ndarray
        Y coordinates.
    output_path : Path, optional
        Output file path.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    vmax : float, optional
        Maximum value for color scale.
        
    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    
    if vmax is None:
        vmax = np.nanpercentile(ros_grid, 99)
    
    im = ax.imshow(
        ros_grid,
        extent=extent,
        origin="upper",
        cmap=ROS_CMAP,
        aspect="equal",
        vmin=0,
        vmax=vmax,
    )
    
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.colorbar(im, ax=ax, label="ROS (m/min)")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved ROS plot to {output_path}")
    
    return fig


def plot_fire_perimeter(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes | None = None,
    color: str = "red",
    linewidth: float = 2.0,
    fill: bool = True,
    fill_alpha: float = 0.3,
    label: str | None = None,
) -> plt.Axes:
    """
    Plot a single fire perimeter.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates of perimeter vertices.
    y : np.ndarray
        Y coordinates of perimeter vertices.
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure.
    color : str
        Line and fill color.
    linewidth : float
        Line width.
    fill : bool
        Whether to fill the polygon.
    fill_alpha : float
        Fill transparency.
    label : str, optional
        Legend label.
        
    Returns
    -------
    Axes
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Close polygon
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    
    # Plot line
    ax.plot(x_closed, y_closed, color=color, linewidth=linewidth, label=label)
    
    # Fill polygon
    if fill:
        polygon = MplPolygon(
            list(zip(x_closed, y_closed)),
            closed=True,
            facecolor=color,
            alpha=fill_alpha,
            edgecolor="none",
        )
        ax.add_patch(polygon)
    
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    return ax


def plot_perimeter_history(
    history: FirePerimeterHistory,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 150,
    show_every: int = 1,
    background_grid: np.ndarray | None = None,
    extent: list[float] | None = None,
) -> plt.Figure:
    """
    Plot fire perimeter evolution over time.
    
    Parameters
    ----------
    history : FirePerimeterHistory
        Perimeter history container.
    output_path : Path, optional
        Output file path.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    show_every : int
        Show every Nth perimeter.
    background_grid : np.ndarray, optional
        Background raster to display.
    extent : list, optional
        Extent for background grid [xmin, xmax, ymin, ymax].
        
    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Background
    if background_grid is not None and extent is not None:
        ax.imshow(
            background_grid,
            extent=extent,
            origin="upper",
            cmap="gray",
            alpha=0.5,
        )
    
    # Color normalization by time
    n_perimeters = len(history.perimeters)
    norm = Normalize(vmin=0, vmax=history.times[-1] if history.times else 1)
    
    # Plot perimeters
    for i, ((x, y), t) in enumerate(zip(history.perimeters, history.times)):
        if i % show_every != 0 and i != n_perimeters - 1:
            continue
        
        color = FIRE_CMAP(norm(t))
        alpha = 0.3 + 0.7 * (i / n_perimeters)  # Increasing opacity
        
        # Only fill the final perimeter
        fill = (i == n_perimeters - 1)
        
        plot_fire_perimeter(
            x, y, ax=ax,
            color=color,
            linewidth=1.5,
            fill=fill,
            fill_alpha=0.4,
        )
    
    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap=FIRE_CMAP, norm=norm)
    plt.colorbar(sm, ax=ax, label="Time (minutes)")
    
    ax.set_title("Fire Perimeter Evolution")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved perimeter history plot to {output_path}")
    
    return fig


def plot_ignition_points(
    x: np.ndarray | list,
    y: np.ndarray | list,
    ax: plt.Axes | None = None,
    color: str = "yellow",
    marker: str = "*",
    markersize: float = 15,
    edgecolor: str = "black",
    label: str = "Ignition",
) -> plt.Axes:
    """
    Plot ignition points.
    
    Parameters
    ----------
    x : array-like
        X coordinates.
    y : array-like
        Y coordinates.
    ax : Axes, optional
        Matplotlib axes.
    color : str
        Marker color.
    marker : str
        Marker style.
    markersize : float
        Marker size.
    edgecolor : str
        Marker edge color.
    label : str
        Legend label.
        
    Returns
    -------
    Axes
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(
        x, y,
        c=color,
        marker=marker,
        s=markersize**2,
        edgecolors=edgecolor,
        linewidths=1,
        label=label,
        zorder=10,
    )
    
    return ax


# =============================================================================
# Animation Functions
# =============================================================================


def create_fire_animation(
    history: FirePerimeterHistory,
    output_path: Path,
    fps: int = 10,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 100,
    background_grid: np.ndarray | None = None,
    extent: list[float] | None = None,
) -> None:
    """
    Create animated GIF of fire spread.
    
    Parameters
    ----------
    history : FirePerimeterHistory
        Perimeter history.
    output_path : Path
        Output GIF path.
    fps : int
        Frames per second.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    background_grid : np.ndarray, optional
        Background raster.
    extent : list, optional
        Background extent.
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        logger.warning("Animation requires pillow. Install with: pip install pillow")
        return
    
    if len(history.perimeters) == 0:
        logger.warning("No perimeters to animate")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Background
    if background_grid is not None and extent is not None:
        ax.imshow(
            background_grid,
            extent=extent,
            origin="upper",
            cmap="gray",
            alpha=0.5,
        )
    
    # Determine plot bounds
    all_x = np.concatenate([p[0] for p in history.perimeters])
    all_y = np.concatenate([p[1] for p in history.perimeters])
    
    margin = 0.1 * max(all_x.ptp(), all_y.ptp())
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    # Initialize line and polygon
    line, = ax.plot([], [], "r-", linewidth=2)
    polygon = MplPolygon(
        [(0, 0)],
        closed=True,
        facecolor="red",
        alpha=0.3,
        edgecolor="none",
    )
    ax.add_patch(polygon)
    
    time_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        fontweight="bold",
    )
    
    def init():
        line.set_data([], [])
        polygon.set_xy([(0, 0)])
        time_text.set_text("")
        return line, polygon, time_text
    
    def update(frame):
        x, y = history.perimeters[frame]
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        
        line.set_data(x_closed, y_closed)
        polygon.set_xy(list(zip(x_closed, y_closed)))
        
        t = history.times[frame]
        time_text.set_text(f"Time: {t:.1f} min")
        
        return line, polygon, time_text
    
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(history.perimeters),
        interval=1000 // fps,
        blit=True,
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    logger.info(f"Saved animation to {output_path}")


# =============================================================================
# Summary Plots
# =============================================================================


def plot_burn_probability(
    perimeters: list[tuple[np.ndarray, np.ndarray]],
    grid_resolution: float,
    bounds: tuple[float, float, float, float],
    output_path: Path | None = None,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 150,
) -> plt.Figure:
    """
    Create burn probability map from multiple fire simulations.
    
    Parameters
    ----------
    perimeters : list
        List of (x, y) perimeter tuples.
    grid_resolution : float
        Grid cell size in meters.
    bounds : tuple
        (xmin, ymin, xmax, ymax) bounds.
    output_path : Path, optional
        Output file path.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
        
    Returns
    -------
    Figure
        Matplotlib figure.
    """
    from shapely.geometry import Point, Polygon
    
    xmin, ymin, xmax, ymax = bounds
    
    # Create grid
    x_edges = np.arange(xmin, xmax + grid_resolution, grid_resolution)
    y_edges = np.arange(ymin, ymax + grid_resolution, grid_resolution)
    
    burn_count = np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    
    # Convert perimeters to polygons
    polygons = []
    for x, y in perimeters:
        if len(x) > 2:
            coords = list(zip(x, y))
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                poly = Polygon(coords)
                if poly.is_valid:
                    polygons.append(poly)
            except Exception:
                continue
    
    if not polygons:
        logger.warning("No valid polygons for burn probability")
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Burn Probability (no data)")
        return fig
    
    # Count burns per cell
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    for i, yc in enumerate(y_centers):
        for j, xc in enumerate(x_centers):
            point = Point(xc, yc)
            for poly in polygons:
                if poly.contains(point):
                    burn_count[i, j] += 1
    
    # Convert to probability
    n_fires = len(polygons)
    burn_prob = burn_count / n_fires
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    extent = [xmin, xmax, ymin, ymax]
    im = ax.imshow(
        burn_prob,
        extent=extent,
        origin="lower",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    
    ax.set_title(f"Burn Probability (n={n_fires} fires)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Probability")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved burn probability plot to {output_path}")
    
    return fig


def plot_area_distribution(
    areas_ha: list[float],
    output_path: Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 150,
    bins: int = 30,
) -> plt.Figure:
    """
    Plot histogram of fire sizes.
    
    Parameters
    ----------
    areas_ha : list
        Fire areas in hectares.
    output_path : Path, optional
        Output file path.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    bins : int
        Number of histogram bins.
        
    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    areas = np.array(areas_ha)
    areas = areas[areas > 0]  # Filter zero-area fires
    
    if len(areas) == 0:
        ax.set_title("Fire Size Distribution (no data)")
        return fig
    
    ax.hist(areas, bins=bins, edgecolor="black", alpha=0.7)
    
    # Statistics
    mean_area = np.mean(areas)
    median_area = np.median(areas)
    
    ax.axvline(mean_area, color="red", linestyle="--", label=f"Mean: {mean_area:.1f} ha")
    ax.axvline(median_area, color="blue", linestyle="--", label=f"Median: {median_area:.1f} ha")
    
    ax.set_title(f"Fire Size Distribution (n={len(areas)})")
    ax.set_xlabel("Fire Area (ha)")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved area distribution plot to {output_path}")
    
    return fig
