"""
Ignition module for Ignacio.

This module handles ignition probability grids, escaped fire rate adjustments,
and Monte Carlo sampling of ignition locations.

Based on the Burn-P3 ignition framework (Parisien et al., 2005).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ignacio.config import IgnacioConfig, IgnitionRule
from ignacio.io import RasterData, read_raster, read_raster_int, read_vector, write_vector

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class IgnitionPoint:
    """A single ignition point."""
    
    x: float
    y: float
    cause: str
    season: str
    iteration: int = 0
    
    def to_geometry(self) -> Point:
        """Convert to Shapely Point."""
        return Point(self.x, self.y)


@dataclass
class IgnitionSet:
    """Collection of ignition points from Monte Carlo sampling."""
    
    points: list[IgnitionPoint]
    crs: str
    
    @property
    def n_points(self) -> int:
        """Number of ignition points."""
        return len(self.points)
    
    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert to GeoDataFrame."""
        data = {
            "x": [p.x for p in self.points],
            "y": [p.y for p in self.points],
            "cause": [p.cause for p in self.points],
            "season": [p.season for p in self.points],
            "iteration": [p.iteration for p in self.points],
            "geometry": [p.to_geometry() for p in self.points],
        }
        return gpd.GeoDataFrame(data, crs=self.crs)
    
    def get_iteration(self, iteration: int) -> list[IgnitionPoint]:
        """Get all points for a specific iteration."""
        return [p for p in self.points if p.iteration == iteration]
    
    @property
    def iterations(self) -> list[int]:
        """Get unique iteration numbers."""
        return sorted(set(p.iteration for p in self.points))


# =============================================================================
# Fuel Type Lookup
# =============================================================================


def get_fuel_codes(fuel_type_names: list[str], fuel_lookup: dict[int, str]) -> list[int]:
    """
    Get numeric fuel codes for named fuel types.
    
    Parameters
    ----------
    fuel_type_names : list[str]
        List of FBP fuel type codes (e.g., ["D-1", "O-1a"]).
    fuel_lookup : dict
        Mapping of numeric codes to fuel type names.
        
    Returns
    -------
    list[int]
        Numeric codes corresponding to the fuel types.
    """
    # Reverse the lookup
    name_to_code = {v: k for k, v in fuel_lookup.items()}
    
    codes = []
    for name in fuel_type_names:
        name_upper = name.upper().strip()
        if name_upper in name_to_code:
            codes.append(name_to_code[name_upper])
    
    return codes


# =============================================================================
# Ignition Grid Processing
# =============================================================================


def adjust_ignition_grid(
    initial_grid: np.ndarray,
    ecoregion_grid: np.ndarray,
    escaped_rates: dict[str, dict[str, float]],
    occurrence_rates: dict[str, float],
    cause: str,
    season: str,
) -> np.ndarray:
    """
    Adjust ignition probability grid by escaped fire rates.
    
    Implements Equation [4] from Burn-P3:
        AI_ij = I_ij * (E_j / F_j)
        
    where:
        AI_ij = adjusted ignition probability
        I_ij = initial ignition probability
        E_j = escaped fire rate for ecoregion j
        F_j = fire occurrence rate for ecoregion j
    
    Parameters
    ----------
    initial_grid : np.ndarray
        Initial ignition probability grid.
    ecoregion_grid : np.ndarray
        Ecoregion classification grid.
    escaped_rates : dict
        Escaped fire rates by ecoregion and cause/season.
    occurrence_rates : dict
        Fire occurrence rates by ecoregion.
    cause : str
        Fire cause ("Lightning" or "Human").
    season : str
        Season ("Spring", "Summer", or "Fall").
        
    Returns
    -------
    np.ndarray
        Adjusted ignition probability grid.
    """
    adjusted = np.zeros_like(initial_grid, dtype=np.float64)
    
    # Get unique ecoregion IDs
    unique_ecos = np.unique(ecoregion_grid[np.isfinite(ecoregion_grid)])
    
    for eco_id in unique_ecos:
        eco_mask = ecoregion_grid == eco_id
        eco_key = str(int(eco_id))
        rate_key = f"{cause}_{season}"
        
        # Get rates
        E_j = escaped_rates.get(eco_key, {}).get(rate_key, 1.0)
        F_j = occurrence_rates.get(eco_key, 1.0)
        
        # Compute scaling factor
        scale = E_j / F_j if F_j != 0 else 1.0
        
        adjusted[eco_mask] = initial_grid[eco_mask] * scale
    
    # Ensure non-negative
    adjusted[adjusted < 0] = 0
    
    return adjusted


def apply_ignition_rules(
    adjusted_grid: np.ndarray,
    fuel_grid: np.ndarray,
    cause: str,
    season: str,
    rules: list[IgnitionRule],
    fuel_lookup: dict[int, str],
) -> np.ndarray:
    """
    Apply ignition rules to filter out invalid ignition locations.
    
    Parameters
    ----------
    adjusted_grid : np.ndarray
        Adjusted ignition probability grid.
    fuel_grid : np.ndarray
        Fuel type grid.
    cause : str
        Fire cause.
    season : str
        Season.
    rules : list[IgnitionRule]
        List of ignition restriction rules.
    fuel_lookup : dict
        Mapping of numeric codes to fuel types.
        
    Returns
    -------
    np.ndarray
        Filtered ignition probability grid.
    """
    filtered = adjusted_grid.copy()
    
    for rule in rules:
        # Check if rule applies
        if rule.cause != cause:
            continue
        if rule.season is not None and rule.season != season:
            continue
        
        # Get fuel codes to filter
        codes = get_fuel_codes(rule.fuel_types, fuel_lookup)
        
        # Zero out matching cells
        for code in codes:
            filtered[fuel_grid == code] = 0
    
    return filtered


def normalize_probability_grid(grid: np.ndarray) -> np.ndarray:
    """
    Normalize grid values to sum to 1.
    
    Parameters
    ----------
    grid : np.ndarray
        Probability grid.
        
    Returns
    -------
    np.ndarray
        Normalized probability grid.
    """
    grid = grid.copy()
    grid[grid < 0] = 0
    grid[~np.isfinite(grid)] = 0
    
    total = np.sum(grid)
    if total > 0:
        return grid / total
    else:
        return grid


# =============================================================================
# Monte Carlo Sampling
# =============================================================================


def draw_number_of_escaped_fires(
    distribution: dict[int, float],
    rng: np.random.Generator | None = None,
) -> int:
    """
    Draw number of escaped fires from frequency distribution.
    
    Parameters
    ----------
    distribution : dict
        Mapping of fire count to probability.
    rng : Generator, optional
        Random number generator.
        
    Returns
    -------
    int
        Number of escaped fires.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    counts = list(distribution.keys())
    probs = np.array(list(distribution.values()))
    probs = probs / np.sum(probs)  # Normalize
    
    return rng.choice(counts, p=probs)


def sample_ignition_locations(
    probability_grid: np.ndarray,
    transform,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float]]:
    """
    Sample ignition locations weighted by probability grid.
    
    Parameters
    ----------
    probability_grid : np.ndarray
        Normalized probability grid.
    transform : Affine
        Affine transform for the grid.
    n_samples : int
        Number of locations to sample.
    rng : Generator, optional
        Random number generator.
        
    Returns
    -------
    list[tuple[float, float]]
        List of (x, y) coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Flatten grid
    flat = probability_grid.flatten()
    flat[~np.isfinite(flat)] = 0
    flat[flat < 0] = 0
    
    total = np.sum(flat)
    if total == 0:
        raise ValueError("All ignition probabilities are zero after filtering")
    
    probs = flat / total
    
    # Sample indices
    n_samples = min(n_samples, np.sum(probs > 0))  # Can't sample more than non-zero cells
    indices = rng.choice(len(flat), size=n_samples, replace=False, p=probs)
    
    # Convert to coordinates
    rows, cols = np.unravel_index(indices, probability_grid.shape)
    
    coords = []
    for row, col in zip(rows, cols):
        # Cell center coordinates
        x = transform.c + (col + 0.5) * transform.a
        y = transform.f + (row + 0.5) * transform.e
        coords.append((x, y))
    
    return coords


# =============================================================================
# Main Ignition Generation
# =============================================================================


def generate_ignitions(
    config: IgnacioConfig,
    fuel_raster: RasterData | None = None,
    rng: np.random.Generator | None = None,
    terrain_crs: str | None = None,
) -> IgnitionSet:
    """
    Generate ignition points based on configuration.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
    fuel_raster : RasterData, optional
        Pre-loaded fuel raster. If None, loaded from config.
    rng : Generator, optional
        Random number generator for reproducibility.
    terrain_crs : str, optional
        CRS of the terrain grid. If provided, ignitions will be reprojected
        to match this CRS.
        
    Returns
    -------
    IgnitionSet
        Set of sampled ignition points.
    """
    if rng is None:
        seed = config.project.random_seed
        rng = np.random.default_rng(seed)
    
    ign_config = config.ignition
    crs = config.crs.working_crs
    
    # Handle point-based ignition
    if ign_config.source_type in ("point", "shapefile"):
        return _load_point_ignitions(config, terrain_crs=terrain_crs)
    
    # Grid-based ignition
    logger.info(f"Loading ignition grid from {ign_config.grid_path}")
    ign_raster = read_raster(ign_config.grid_path)
    
    # Load fuel grid for rules
    if fuel_raster is None:
        fuel_raster = read_raster_int(config.fuel.path)
    
    # Load ecoregion grid if available
    eco_grid = None
    if ign_config.ecoregion_path and Path(ign_config.ecoregion_path).exists():
        logger.info(f"Loading ecoregion grid from {ign_config.ecoregion_path}")
        eco_raster = read_raster(ign_config.ecoregion_path)
        eco_grid = eco_raster.data
    
    # Process ignition grid
    ign_grid = ign_raster.data.copy()
    
    # Apply escaped fire rate adjustment
    if eco_grid is not None:
        logger.info("Applying escaped fire rate adjustment")
        ign_grid = adjust_ignition_grid(
            ign_grid,
            eco_grid,
            ign_config.escaped_fire_rates,
            ign_config.fire_occurrence_rates,
            ign_config.cause,
            ign_config.season,
        )
    
    # Apply ignition rules
    logger.info("Applying ignition rules")
    ign_grid = apply_ignition_rules(
        ign_grid,
        fuel_raster.data,
        ign_config.cause,
        ign_config.season,
        ign_config.ignition_rules,
        config.fuel.fuel_lookup,
    )
    
    # Normalize
    ign_grid = normalize_probability_grid(ign_grid)
    
    # Get frequency distribution
    freq_dist = ign_config.escaped_fire_distribution
    
    # Sample ignitions for each iteration
    all_points = []
    
    for iteration in range(ign_config.n_iterations):
        # Draw number of fires
        n_fires = draw_number_of_escaped_fires(freq_dist, rng)
        
        # Sample locations
        try:
            coords = sample_ignition_locations(
                ign_grid,
                ign_raster.transform,
                n_fires,
                rng,
            )
        except ValueError as e:
            logger.warning(f"Iteration {iteration}: {e}")
            continue
        
        # Create ignition points
        for x, y in coords:
            point = IgnitionPoint(
                x=x,
                y=y,
                cause=ign_config.cause,
                season=ign_config.season,
                iteration=iteration,
            )
            all_points.append(point)
        
        logger.debug(f"Iteration {iteration}: sampled {len(coords)} ignition points")
    
    logger.info(f"Generated {len(all_points)} total ignition points across {ign_config.n_iterations} iterations")
    
    return IgnitionSet(points=all_points, crs=crs)


def _load_point_ignitions(config: IgnacioConfig, terrain_crs: str | None = None) -> IgnitionSet:
    """Load ignition points from shapefile or point file.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
    terrain_crs : str, optional
        CRS of the terrain grid. If provided, ignitions will be reprojected
        to match this CRS instead of the working_crs.
    """
    ign_config = config.ignition
    
    # Use terrain CRS if provided, otherwise fall back to working CRS
    if terrain_crs is not None:
        crs = terrain_crs
        logger.info(f"Using terrain CRS for ignitions: {crs}")
    else:
        crs = config.crs.working_crs
    
    logger.info(f"Loading ignition points from {ign_config.point_path}")
    gdf = read_vector(ign_config.point_path, target_crs=crs)
    
    # Log the reprojection info
    if gdf.crs is not None:
        logger.info(f"Ignitions reprojected to CRS: {crs}")
    
    if len(gdf) == 0:
        logger.warning("No features found in ignition file")
        return IgnitionSet(points=[], crs=crs)
    
    points = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        
        if geom is None or geom.is_empty:
            logger.warning(f"Skipping empty geometry at index {idx}")
            continue
        
        # Handle different geometry types
        if geom.geom_type == "Point":
            x, y = geom.x, geom.y
        elif geom.geom_type in ("Polygon", "MultiPolygon"):
            # Use centroid for polygons
            centroid = geom.centroid
            x, y = centroid.x, centroid.y
            logger.debug(f"Using centroid for polygon geometry at index {idx}")
        elif geom.geom_type in ("LineString", "MultiLineString"):
            # Use midpoint for lines
            centroid = geom.centroid
            x, y = centroid.x, centroid.y
            logger.debug(f"Using centroid for line geometry at index {idx}")
        else:
            # Try to get centroid for any other geometry type
            try:
                centroid = geom.centroid
                x, y = centroid.x, centroid.y
            except Exception as e:
                logger.warning(f"Could not extract point from geometry type {geom.geom_type}: {e}")
                continue
        
        point = IgnitionPoint(
            x=x,
            y=y,
            cause=row.get("cause", ign_config.cause),
            season=row.get("season", ign_config.season),
            iteration=row.get("iteration", 0),
        )
        points.append(point)
    
    logger.info(f"Loaded {len(points)} ignition points")
    if len(points) > 0:
        logger.info(f"First ignition point: ({points[0].x:.6f}, {points[0].y:.6f})")
    
    return IgnitionSet(points=points, crs=crs)


def save_ignitions(
    ignition_set: IgnitionSet,
    output_path: Path,
    driver: str = "ESRI Shapefile",
) -> None:
    """
    Save ignition points to file.
    
    Parameters
    ----------
    ignition_set : IgnitionSet
        Ignition points to save.
    output_path : Path
        Output file path.
    driver : str
        Output driver.
    """
    gdf = ignition_set.to_geodataframe()
    write_vector(gdf, output_path, driver=driver)
    logger.info(f"Saved {len(gdf)} ignition points to {output_path}")
