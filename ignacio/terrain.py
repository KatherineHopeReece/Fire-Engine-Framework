"""
Terrain processing for Ignacio.

This module provides functions for processing Digital Elevation Models (DEMs)
to derive slope and aspect grids used in fire behaviour calculations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import sobel

from ignacio.config import IgnacioConfig
from ignacio.io import RasterData, read_raster, write_raster

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TerrainGrids:
    """Container for terrain-derived grids."""
    
    dem: RasterData
    slope_deg: np.ndarray
    aspect_deg: np.ndarray
    
    @property
    def transform(self):
        """Return the affine transform."""
        return self.dem.transform
    
    @property
    def crs(self):
        """Return the coordinate reference system."""
        return self.dem.crs
    
    @property
    def shape(self) -> tuple[int, int]:
        """Return (height, width) of the grids."""
        return self.dem.shape
    
    def get_coordinate_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return 1D arrays of x and y coordinates."""
        return self.dem.get_coordinate_arrays()


# =============================================================================
# Slope and Aspect Calculation
# =============================================================================


def compute_slope_aspect(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and aspect from a DEM using finite differences.
    
    Uses the Sobel operator for gradient estimation, which provides
    better noise reduction than simple central differences.
    
    Parameters
    ----------
    dem : np.ndarray
        2D array of elevations in meters.
    dx : float
        Cell size in the x direction (positive).
    dy : float
        Cell size in the y direction (positive).
        
    Returns
    -------
    slope_deg : np.ndarray
        Slope in degrees from horizontal (0 = flat, 90 = vertical).
    aspect_deg : np.ndarray
        Aspect in degrees clockwise from north (0/360 = north, 90 = east,
        180 = south, 270 = west). NaN where slope is undefined.
        
    Notes
    -----
    The Sobel operator applies a 3x3 kernel that computes weighted gradients:
    
    For dz/dx:
        [-1  0  1]
        [-2  0  2] / (8 * dx)
        [-1  0  1]
        
    This provides smoothing while computing the gradient.
    """
    # Ensure float64 for numerical stability
    z = np.asarray(dem, dtype=np.float64)
    
    # Identify invalid cells (NaN or very extreme values)
    invalid = ~np.isfinite(z)
    
    # For gradient computation, we need to handle NaN values
    # Replace NaN temporarily with local mean to avoid edge effects
    if np.any(invalid):
        z_filled = z.copy()
        # Simple approach: fill with median of valid values
        valid_vals = z[~invalid]
        if len(valid_vals) > 0:
            fill_val = np.median(valid_vals)
            z_filled[invalid] = fill_val
        else:
            # All NaN - return NaN arrays
            return np.full_like(z, np.nan), np.full_like(z, np.nan)
    else:
        z_filled = z
    
    # Compute gradients using numpy gradient (more robust than Sobel for edges)
    # Note: gradient returns (dy, dx) for axis order
    dzdy, dzdx = np.gradient(z_filled, dy, dx)
    
    # Slope magnitude: tan(slope) = sqrt((dz/dx)^2 + (dz/dy)^2)
    grad_mag = np.hypot(dzdx, dzdy)
    slope_rad = np.arctan(grad_mag)
    slope_deg = np.degrees(slope_rad)
    
    # Aspect: direction of steepest descent
    # Using atan2 for proper quadrant handling
    # Convention: 0 = north, 90 = east, 180 = south, 270 = west
    aspect_rad = np.arctan2(dzdx, -dzdy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.mod(aspect_deg, 360.0)
    
    # Where slope is ~0 (flat), aspect is undefined
    flat_mask = grad_mag < 1e-6
    aspect_deg[flat_mask] = np.nan
    
    # Propagate invalid DEM cells back to output
    slope_deg[invalid] = np.nan
    aspect_deg[invalid] = np.nan
    
    # Clamp slope to valid range (numerical issues can cause >90)
    slope_deg = np.clip(slope_deg, 0.0, 90.0)
    
    return slope_deg, aspect_deg


def compute_slope_aspect_simple(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and aspect using simple central differences.
    
    This is faster but noisier than the Sobel-based method.
    
    Parameters
    ----------
    dem : np.ndarray
        2D array of elevations in meters.
    dx : float
        Cell size in x direction.
    dy : float
        Cell size in y direction.
        
    Returns
    -------
    slope_deg : np.ndarray
        Slope in degrees.
    aspect_deg : np.ndarray
        Aspect in degrees clockwise from north.
    """
    z = np.asarray(dem, dtype=np.float64)
    invalid = ~np.isfinite(z)
    
    # Central differences with numpy.gradient
    dzdy, dzdx = np.gradient(z, dy, dx)
    
    # Slope
    grad_mag = np.hypot(dzdx, dzdy)
    slope_deg = np.degrees(np.arctan(grad_mag))
    
    # Aspect
    aspect_deg = np.degrees(np.arctan2(dzdx, -dzdy))
    aspect_deg = np.mod(aspect_deg, 360.0)
    
    flat_mask = grad_mag < 1e-6
    aspect_deg[flat_mask] = np.nan
    
    slope_deg[invalid] = np.nan
    aspect_deg[invalid] = np.nan
    
    return slope_deg, aspect_deg


# =============================================================================
# Terrain Grid Building
# =============================================================================


def build_terrain_grids(config: IgnacioConfig) -> TerrainGrids:
    """
    Build terrain grids from DEM, computing slope and aspect if needed.
    
    If pre-computed slope and aspect grids are specified in the configuration
    and exist, they will be loaded. Otherwise, they will be computed from
    the DEM and optionally saved.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
        
    Returns
    -------
    TerrainGrids
        Container with DEM, slope, and aspect arrays.
    """
    logger.info(f"Loading DEM from {config.terrain.dem_path}")
    dem = read_raster(config.terrain.dem_path)
    
    # Log DEM statistics
    valid_dem = dem.data[np.isfinite(dem.data)]
    if len(valid_dem) > 0:
        logger.info(
            f"DEM statistics: min={np.nanmin(valid_dem):.1f}m, "
            f"max={np.nanmax(valid_dem):.1f}m, mean={np.nanmean(valid_dem):.1f}m"
        )
        logger.info(f"DEM valid cells: {len(valid_dem)} / {dem.data.size} ({100*len(valid_dem)/dem.data.size:.1f}%)")
    
    # Get cell sizes from transform
    # transform.a is the pixel width (x resolution)
    # transform.e is the pixel height (y resolution, typically negative)
    dx_raw = abs(dem.transform.a)
    dy_raw = abs(dem.transform.e)
    
    logger.info(f"DEM CRS: {dem.crs}")
    logger.info(f"DEM raw cell size: dx={dx_raw}, dy={dy_raw}")
    logger.info(f"DEM shape: {dem.data.shape[0]} rows x {dem.data.shape[1]} cols")
    
    # Check if CRS is geographic (lat/lon) - cell sizes would be in degrees
    is_geographic = False
    if dem.crs is not None:
        try:
            import pyproj
            crs = pyproj.CRS.from_user_input(dem.crs)
            is_geographic = crs.is_geographic
            logger.info(f"CRS is geographic: {is_geographic}")
        except Exception as e:
            logger.warning(f"Could not parse CRS: {e}")
            # Heuristic: if cell sizes are very small (<1), likely geographic
            if dx_raw < 1 and dy_raw < 1:
                is_geographic = True
                logger.info("Cell sizes < 1, assuming geographic CRS")
    else:
        # No CRS - use heuristic
        if dx_raw < 1 and dy_raw < 1:
            is_geographic = True
            logger.info("No CRS defined, cell sizes < 1, assuming geographic")
    
    if is_geographic:
        # Convert degrees to approximate meters
        # At the center latitude of the DEM
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        
        # Get center latitude from transform
        # transform.f is the y coordinate of the upper-left corner
        ny, nx = dem.data.shape
        center_y = dem.transform.f + dem.transform.e * (ny / 2)
        center_lat = center_y  # In geographic CRS, y = latitude
        
        lat_rad = np.radians(center_lat)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
        
        dx = dx_raw * meters_per_deg_lon
        dy = dy_raw * meters_per_deg_lat
        
        logger.info(f"Center latitude: {center_lat:.4f}°")
        logger.info(f"Converted cell size: dx={dx:.2f}m, dy={dy:.2f}m")
    else:
        dx = dx_raw
        dy = dy_raw
        logger.info(f"DEM cell size: dx={dx:.2f}m, dy={dy:.2f}m")
    
    # Check for pre-computed grids
    slope_path = config.terrain.slope_path
    aspect_path = config.terrain.aspect_path
    
    if slope_path and Path(slope_path).exists():
        logger.info(f"Loading pre-computed slope from {slope_path}")
        slope_raster = read_raster(slope_path)
        slope_deg = slope_raster.data
    else:
        slope_deg = None
    
    if aspect_path and Path(aspect_path).exists():
        logger.info(f"Loading pre-computed aspect from {aspect_path}")
        aspect_raster = read_raster(aspect_path)
        aspect_deg = aspect_raster.data
    else:
        aspect_deg = None
    
    # Compute if not loaded
    if slope_deg is None or aspect_deg is None:
        logger.info("Computing slope and aspect from DEM")
        computed_slope, computed_aspect = compute_slope_aspect(dem.data, dx, dy)
        
        if slope_deg is None:
            slope_deg = computed_slope
        if aspect_deg is None:
            aspect_deg = computed_aspect
        
        # Save computed grids if output is enabled
        if config.output.save_ros_grids:
            output_dir = Path(config.project.output_dir) / "terrain"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not (slope_path and Path(slope_path).exists()):
                slope_out = output_dir / "slope_deg.tif"
                write_raster(slope_out, slope_deg, dem.transform, dem.crs)
                logger.info(f"Saved slope grid to {slope_out}")
            
            if not (aspect_path and Path(aspect_path).exists()):
                aspect_out = output_dir / "aspect_deg.tif"
                write_raster(aspect_out, aspect_deg, dem.transform, dem.crs)
                logger.info(f"Saved aspect grid to {aspect_out}")
    
    # Log summary statistics
    valid_slope = slope_deg[np.isfinite(slope_deg)]
    if len(valid_slope) > 0:
        logger.info(
            f"Slope statistics: min={np.nanmin(valid_slope):.1f}, "
            f"max={np.nanmax(valid_slope):.1f}, mean={np.nanmean(valid_slope):.1f} degrees"
        )
        logger.info(f"Valid slope cells: {len(valid_slope)} / {slope_deg.size} ({100*len(valid_slope)/slope_deg.size:.1f}%)")
        
        # Log slope distribution
        flat = np.sum(valid_slope < 5)
        gentle = np.sum((valid_slope >= 5) & (valid_slope < 15))
        moderate = np.sum((valid_slope >= 15) & (valid_slope < 30))
        steep = np.sum((valid_slope >= 30) & (valid_slope < 45))
        very_steep = np.sum(valid_slope >= 45)
        logger.info(
            f"Slope distribution: <5°: {100*flat/len(valid_slope):.1f}%, "
            f"5-15°: {100*gentle/len(valid_slope):.1f}%, "
            f"15-30°: {100*moderate/len(valid_slope):.1f}%, "
            f"30-45°: {100*steep/len(valid_slope):.1f}%, "
            f">45°: {100*very_steep/len(valid_slope):.1f}%"
        )
    else:
        logger.warning("No valid slope values computed")
    
    return TerrainGrids(
        dem=dem,
        slope_deg=slope_deg,
        aspect_deg=aspect_deg,
    )


# =============================================================================
# Terrain Correction Factors
# =============================================================================


def compute_slope_factor(
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    wind_direction: float,
    slope_factor_strength: float = 0.5,
) -> np.ndarray:
    """
    Compute slope correction factor for rate of spread.
    
    Fire spreads faster upslope when aligned with wind direction.
    This implements a simplified version of the CFFDRS slope effect.
    
    Parameters
    ----------
    slope_deg : np.ndarray
        Slope in degrees.
    aspect_deg : np.ndarray
        Aspect in degrees clockwise from north.
    wind_direction : float
        Wind direction in degrees (direction wind is coming FROM).
    slope_factor_strength : float
        Multiplier for slope effect (0 = no effect, 1 = full effect).
        
    Returns
    -------
    np.ndarray
        Multiplicative factor for ROS adjustment (>1 = faster, <1 = slower).
        
    Notes
    -----
    The slope factor is computed as:
    
        SF = 1 + strength * tan(slope) * cos(wind_dir - upslope_dir)
        
    where upslope_dir = aspect + 180 (aspect points downslope).
    
    When wind is aligned with upslope direction, SF > 1 (fire spreads faster).
    When wind is aligned with downslope direction, SF < 1 (fire spreads slower).
    """
    slope_rad = np.radians(slope_deg)
    tan_slope = np.tan(slope_rad)
    
    # Upslope direction is opposite to aspect (aspect points downslope)
    upslope_deg = (aspect_deg + 180.0) % 360.0
    upslope_rad = np.radians(upslope_deg)
    
    # Wind direction in radians
    wind_rad = np.radians(wind_direction)
    
    # Angle difference between wind and upslope
    delta = wind_rad - upslope_rad
    cos_delta = np.cos(delta)
    
    # Slope factor
    sf = 1.0 + slope_factor_strength * tan_slope * cos_delta
    
    # Ensure non-negative
    sf = np.maximum(sf, 0.0)
    
    # Handle NaN from aspect
    sf = np.where(np.isfinite(aspect_deg), sf, 1.0)
    
    return sf


def compute_elevation_adjustment(
    elevation: np.ndarray,
    temperature: float,
    relative_humidity: float,
    lapse_rate: float = 0.0065,
    ref_temp: float = 20.0,
    ref_rh: float = 30.0,
) -> np.ndarray:
    """
    Compute foliar moisture adjustment based on elevation.
    
    Higher elevations are cooler and may have different moisture conditions.
    This provides a simple parameterization for elevation effects.
    
    Parameters
    ----------
    elevation : np.ndarray
        Elevation in meters.
    temperature : float
        Reference temperature at measurement station (degrees C).
    relative_humidity : float
        Reference relative humidity at measurement station (percent).
    lapse_rate : float
        Temperature lapse rate (degrees C per meter).
    ref_temp : float
        Reference temperature for adjustment calculation.
    ref_rh : float
        Reference relative humidity for adjustment calculation.
        
    Returns
    -------
    np.ndarray
        Multiplicative factor for FMC/ROS adjustment.
    """
    # Adjust temperature for elevation
    temp_adj = temperature - lapse_rate * elevation
    
    # Simple moisture factor based on temperature and humidity deviation
    # Higher temperatures = drier = faster spread
    # Higher humidity = wetter = slower spread
    factor = 1.0 - 0.002 * (temp_adj - ref_temp) + 0.001 * (relative_humidity - ref_rh)
    
    # Clamp to reasonable range
    factor = np.clip(factor, 0.7, 1.3)
    
    return factor
