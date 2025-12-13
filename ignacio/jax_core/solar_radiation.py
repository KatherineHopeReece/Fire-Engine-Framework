"""
Solar Radiation and Fuel Conditioning Model.

Computes spatially-varying solar exposure based on terrain and sun position,
then adjusts fuel moisture to account for aspect-driven drying effects.

This explains why:
- East-facing slopes dry first in morning
- South-facing slopes (N. hemisphere) are driest
- North-facing slopes retain moisture and often stop fires
- Fires "wake up" with the sun and calm at night

Physics:
- Sun position computed from datetime using astronomical equations
- Terrain shading computed via horizon angle comparison
- Solar flux depends on angle of incidence with terrain normal
- Fuel moisture adjusted based on cumulative solar exposure

References:
- Iqbal, M. (1983). An Introduction to Solar Radiation.
- Dozier, J. & Frew, J. (1990). Rapid calculation of terrain parameters
  for radiation modeling from DEMs.
- Nelson, R.M. (2000). Prediction of diurnal change in 10-h fuel moisture.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from datetime import datetime
import math


class SunPosition(NamedTuple):
    """Sun position in the sky."""
    azimuth: float      # Degrees from North (0=N, 90=E, 180=S, 270=W)
    elevation: float    # Degrees above horizon (0=horizon, 90=zenith)
    zenith: float       # Degrees from vertical (90 - elevation)
    
    @property
    def is_daytime(self) -> bool:
        return self.elevation > 0


def compute_sun_position(
    dt: datetime,
    latitude: float,
    longitude: float,
) -> SunPosition:
    """
    Compute sun position using astronomical equations.
    
    Based on NOAA Solar Calculator algorithms.
    
    Parameters
    ----------
    dt : datetime
        Date and time (should be in local solar time or UTC)
    latitude : float
        Latitude in degrees (positive = North)
    longitude : float
        Longitude in degrees (positive = East)
        
    Returns
    -------
    SunPosition
        Azimuth and elevation of sun
    """
    # Day of year
    doy = dt.timetuple().tm_yday
    
    # Fractional year (radians)
    gamma = 2 * math.pi / 365 * (doy - 1 + (dt.hour - 12) / 24)
    
    # Equation of time (minutes)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) 
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) 
                       - 0.040849 * math.sin(2 * gamma))
    
    # Solar declination (radians)
    decl = (0.006918 - 0.399912 * math.cos(gamma) 
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) 
            + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) 
            + 0.00148 * math.sin(3 * gamma))
    
    # Time offset (minutes)
    time_offset = eqtime + 4 * longitude
    
    # True solar time (minutes)
    tst = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
    
    # Hour angle (degrees)
    ha = (tst / 4) - 180
    
    # Convert to radians
    lat_rad = math.radians(latitude)
    ha_rad = math.radians(ha)
    
    # Solar zenith angle
    cos_zenith = (math.sin(lat_rad) * math.sin(decl) + 
                  math.cos(lat_rad) * math.cos(decl) * math.cos(ha_rad))
    cos_zenith = max(-1, min(1, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    elevation = 90 - zenith
    
    # Solar azimuth
    if cos_zenith != 0:
        cos_azimuth = ((math.sin(lat_rad) * cos_zenith - math.sin(decl)) / 
                       (math.cos(lat_rad) * math.sin(math.radians(zenith)) + 1e-10))
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth = math.degrees(math.acos(cos_azimuth))
        
        if ha > 0:
            azimuth = 360 - azimuth
    else:
        azimuth = 180 if latitude > decl else 0
    
    return SunPosition(azimuth=azimuth, elevation=elevation, zenith=zenith)


def compute_terrain_normal(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute terrain normal vectors from DEM.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid (ny, nx) in meters
    dx, dy : float
        Grid spacing in meters
        
    Returns
    -------
    nx, ny, nz : jnp.ndarray
        Components of unit normal vector at each cell
    """
    # Compute gradients (dz/dx, dz/dy)
    # Use central differences with boundary handling
    dzdx = jnp.zeros_like(dem)
    dzdy = jnp.zeros_like(dem)
    
    # Central differences for interior
    dzdx = dzdx.at[1:-1, 1:-1].set(
        (dem[1:-1, 2:] - dem[1:-1, :-2]) / (2 * dx)
    )
    dzdy = dzdy.at[1:-1, 1:-1].set(
        (dem[2:, 1:-1] - dem[:-2, 1:-1]) / (2 * dy)
    )
    
    # Forward/backward differences at boundaries
    dzdx = dzdx.at[:, 0].set((dem[:, 1] - dem[:, 0]) / dx)
    dzdx = dzdx.at[:, -1].set((dem[:, -1] - dem[:, -2]) / dx)
    dzdy = dzdy.at[0, :].set((dem[1, :] - dem[0, :]) / dy)
    dzdy = dzdy.at[-1, :].set((dem[-1, :] - dem[-2, :]) / dy)
    
    # Normal vector: n = (-dz/dx, -dz/dy, 1) normalized
    mag = jnp.sqrt(dzdx**2 + dzdy**2 + 1)
    
    nx = -dzdx / mag
    ny = -dzdy / mag
    nz = 1.0 / mag
    
    return nx, ny, nz


def compute_slope_aspect_from_dem(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute slope and aspect from DEM.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid
    dx, dy : float
        Grid spacing in meters
        
    Returns
    -------
    slope : jnp.ndarray
        Slope in degrees
    aspect : jnp.ndarray
        Aspect in degrees (0=N, 90=E, 180=S, 270=W)
    """
    nx, ny, nz = compute_terrain_normal(dem, dx, dy)
    
    # Slope = arccos(nz) in degrees
    slope = jnp.degrees(jnp.arccos(jnp.clip(nz, -1, 1)))
    
    # Aspect = atan2(nx, ny) converted to compass bearing
    aspect = jnp.degrees(jnp.arctan2(nx, ny))
    aspect = (aspect + 360) % 360  # Ensure 0-360
    
    return slope, aspect


def compute_cos_incidence_angle(
    slope: jnp.ndarray,
    aspect: jnp.ndarray,
    sun_elevation: float,
    sun_azimuth: float,
) -> jnp.ndarray:
    """
    Compute cosine of angle between sun rays and terrain normal.
    
    This determines how much solar radiation hits each cell.
    cos(i) = 1 means sun perpendicular to slope (max heating)
    cos(i) = 0 means sun parallel to slope (no direct heating)
    cos(i) < 0 means slope faces away from sun (self-shaded)
    
    Parameters
    ----------
    slope : jnp.ndarray
        Terrain slope (degrees)
    aspect : jnp.ndarray
        Terrain aspect (degrees, 0=N)
    sun_elevation : float
        Sun elevation angle (degrees above horizon)
    sun_azimuth : float
        Sun azimuth (degrees from North)
        
    Returns
    -------
    cos_i : jnp.ndarray
        Cosine of incidence angle (clipped to [0, 1])
    """
    # Convert to radians
    slope_rad = jnp.radians(slope)
    aspect_rad = jnp.radians(aspect)
    sun_elev_rad = jnp.radians(sun_elevation)
    sun_az_rad = jnp.radians(sun_azimuth)
    
    # Zenith angle of sun
    sun_zenith_rad = jnp.pi / 2 - sun_elev_rad
    
    # Cosine of incidence angle formula
    cos_i = (jnp.cos(sun_zenith_rad) * jnp.cos(slope_rad) +
             jnp.sin(sun_zenith_rad) * jnp.sin(slope_rad) * 
             jnp.cos(sun_az_rad - aspect_rad))
    
    # Clip to [0, 1] - negative means self-shaded
    cos_i = jnp.clip(cos_i, 0, 1)
    
    return cos_i


def compute_hillshade(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
    sun_elevation: float,
    sun_azimuth: float,
) -> jnp.ndarray:
    """
    Compute hillshade (illumination) map.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid
    dx, dy : float
        Grid spacing
    sun_elevation : float
        Sun elevation (degrees)
    sun_azimuth : float
        Sun azimuth (degrees from North)
        
    Returns
    -------
    hillshade : jnp.ndarray
        Illumination values 0-1
    """
    slope, aspect = compute_slope_aspect_from_dem(dem, dx, dy)
    cos_i = compute_cos_incidence_angle(slope, aspect, sun_elevation, sun_azimuth)
    return cos_i


def compute_cast_shadows(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
    sun_elevation: float,
    sun_azimuth: float,
    max_search_distance: int = 100,
) -> jnp.ndarray:
    """
    Compute cast shadow mask (simplified ray-tracing).
    
    This identifies cells that are in shadow due to terrain blocking
    the sun, not just self-shading from slope orientation.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid (ny, nx)
    dx, dy : float
        Grid spacing in meters
    sun_elevation : float
        Sun elevation (degrees)
    sun_azimuth : float
        Sun azimuth (degrees from North)
    max_search_distance : int
        Maximum cells to search for blocking terrain
        
    Returns
    -------
    shadow_mask : jnp.ndarray
        1.0 = illuminated, 0.0 = in shadow
    """
    if sun_elevation <= 0:
        # Sun below horizon - everything in shadow
        return jnp.zeros_like(dem)
    
    ny, nx = dem.shape
    
    # Direction toward sun (opposite of sun rays)
    sun_az_rad = jnp.radians(sun_azimuth)
    sun_elev_rad = jnp.radians(sun_elevation)
    
    # Step direction (in grid cells)
    step_x = -jnp.sin(sun_az_rad)  # Toward sun
    step_y = -jnp.cos(sun_az_rad)
    
    # Height gain per horizontal step
    tan_elev = jnp.tan(sun_elev_rad)
    
    # Grid cell size in direction of sun
    step_dist = jnp.sqrt((step_x * dx)**2 + (step_y * dy)**2)
    height_per_step = step_dist * tan_elev
    
    # Initialize shadow mask (1 = lit, 0 = shadow)
    shadow = jnp.ones((ny, nx))
    
    # Create coordinate grids
    y_idx, x_idx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
    
    # Trace rays - simplified using loop (could be optimized with scan)
    def trace_step(carry, step):
        shadow, current_height = carry
        
        # Position along ray
        ray_x = x_idx + step * step_x
        ray_y = y_idx + step * step_y
        
        # Check if within bounds
        valid = ((ray_x >= 0) & (ray_x < nx - 1) & 
                 (ray_y >= 0) & (ray_y < ny - 1))
        
        # Bilinear interpolation of terrain height
        rx_floor = jnp.floor(ray_x).astype(jnp.int32)
        ry_floor = jnp.floor(ray_y).astype(jnp.int32)
        rx_floor = jnp.clip(rx_floor, 0, nx - 2)
        ry_floor = jnp.clip(ry_floor, 0, ny - 2)
        
        fx = ray_x - rx_floor
        fy = ray_y - ry_floor
        
        # Get elevations at corners
        h00 = dem[ry_floor, rx_floor]
        h01 = dem[ry_floor, rx_floor + 1]
        h10 = dem[ry_floor + 1, rx_floor]
        h11 = dem[ry_floor + 1, rx_floor + 1]
        
        # Bilinear interpolation
        terrain_h = (h00 * (1 - fx) * (1 - fy) +
                     h01 * fx * (1 - fy) +
                     h10 * (1 - fx) * fy +
                     h11 * fx * fy)
        
        # Height the ray should be at
        ray_height = dem + step * height_per_step
        
        # If terrain is higher than ray, we're in shadow
        in_shadow = valid & (terrain_h > ray_height)
        shadow = jnp.where(in_shadow, 0.0, shadow)
        
        return (shadow, ray_height), None
    
    # Run shadow trace
    (shadow, _), _ = lax.scan(
        trace_step,
        (shadow, dem.astype(jnp.float32)),
        jnp.arange(1, max_search_distance + 1)
    )
    
    return shadow


def compute_solar_radiation(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
    sun_position: SunPosition,
    include_cast_shadows: bool = True,
    atmospheric_transmittance: float = 0.75,
    solar_constant: float = 1361.0,  # W/m²
) -> jnp.ndarray:
    """
    Compute incoming solar radiation at each cell.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid
    dx, dy : float
        Grid spacing in meters
    sun_position : SunPosition
        Current sun position
    include_cast_shadows : bool
        Whether to compute terrain shadows (slower but more accurate)
    atmospheric_transmittance : float
        Fraction of radiation reaching surface (0.6-0.8 typical)
    solar_constant : float
        Solar irradiance at top of atmosphere (W/m²)
        
    Returns
    -------
    radiation : jnp.ndarray
        Solar radiation (W/m²) at each cell
    """
    if sun_position.elevation <= 0:
        return jnp.zeros_like(dem)
    
    # Compute hillshade (self-shading from slope)
    slope, aspect = compute_slope_aspect_from_dem(dem, dx, dy)
    cos_incidence = compute_cos_incidence_angle(
        slope, aspect, sun_position.elevation, sun_position.azimuth
    )
    
    # Air mass (path length through atmosphere)
    # Simplified Kasten-Young formula
    zenith_rad = jnp.radians(sun_position.zenith)
    air_mass = 1.0 / (jnp.cos(zenith_rad) + 0.50572 * (96.07995 - sun_position.zenith)**(-1.6364))
    air_mass = jnp.minimum(air_mass, 38.0)  # Limit near horizon
    
    # Direct beam radiation on horizontal surface
    direct_horizontal = solar_constant * atmospheric_transmittance**air_mass * jnp.sin(jnp.radians(sun_position.elevation))
    
    # Radiation on sloped surface
    radiation = direct_horizontal * cos_incidence / jnp.maximum(jnp.sin(jnp.radians(sun_position.elevation)), 0.01)
    
    # Apply cast shadows if requested
    if include_cast_shadows:
        shadow_mask = compute_cast_shadows(
            dem, dx, dy, sun_position.elevation, sun_position.azimuth
        )
        radiation = radiation * shadow_mask
    
    # Add diffuse radiation (simplified - 15% of horizontal)
    diffuse = 0.15 * direct_horizontal
    radiation = radiation + diffuse
    
    return radiation


def compute_solar_exposure_index(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
    latitude: float,
    longitude: float,
    dt: datetime,
    hours_back: int = 6,
    time_step_hours: float = 1.0,
) -> jnp.ndarray:
    """
    Compute cumulative solar exposure index over recent hours.
    
    This represents how much solar heating each cell has received,
    which correlates with fuel drying.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid
    dx, dy : float
        Grid spacing
    latitude, longitude : float
        Location for sun position
    dt : datetime
        Current datetime
    hours_back : int
        Hours of history to consider
    time_step_hours : float
        Time step for integration
        
    Returns
    -------
    exposure : jnp.ndarray
        Cumulative solar exposure (dimensionless index, 0-1)
    """
    n_steps = int(hours_back / time_step_hours)
    exposure = jnp.zeros_like(dem)
    max_possible = 0.0
    
    for i in range(n_steps):
        hours_ago = hours_back - i * time_step_hours
        from datetime import timedelta
        past_dt = dt - timedelta(hours=hours_ago)
        
        sun_pos = compute_sun_position(past_dt, latitude, longitude)
        
        if sun_pos.elevation > 0:
            radiation = compute_solar_radiation(
                dem, dx, dy, sun_pos,
                include_cast_shadows=False  # Faster for integration
            )
            exposure = exposure + radiation * time_step_hours
            max_possible += 1361 * 0.75 * time_step_hours  # Max possible
    
    # Normalize to 0-1
    if max_possible > 0:
        exposure = exposure / max_possible
    
    return jnp.clip(exposure, 0, 1)


# =============================================================================
# Fuel Moisture Adjustment
# =============================================================================

class FuelConditioningParams(NamedTuple):
    """Parameters for solar-based fuel conditioning."""
    
    # Sensitivity of moisture to solar exposure
    # Higher = more drying on sunny slopes
    solar_moisture_sensitivity: float = 0.03  # Moisture % change per unit exposure
    
    # Reference exposure (average flat terrain gets this)
    reference_exposure: float = 0.5
    
    # Maximum moisture adjustment
    max_moisture_adjustment: float = 0.05  # ±5% moisture
    
    # Aspect-based moisture adjustment (independent of solar calc)
    # South-facing = drier in N. hemisphere
    aspect_moisture_sensitivity: float = 0.02


def adjust_fuel_moisture_solar(
    base_moisture: jnp.ndarray,
    solar_exposure: jnp.ndarray,
    aspect: jnp.ndarray,
    latitude: float,
    params: FuelConditioningParams = FuelConditioningParams(),
) -> jnp.ndarray:
    """
    Adjust fuel moisture based on solar exposure and aspect.
    
    Parameters
    ----------
    base_moisture : jnp.ndarray
        Base fuel moisture (fraction, e.g., 0.08 for 8%)
    solar_exposure : jnp.ndarray
        Solar exposure index (0-1)
    aspect : jnp.ndarray
        Terrain aspect (degrees, 0=N)
    latitude : float
        Latitude (determines which aspects are sunniest)
    params : FuelConditioningParams
        Adjustment parameters
        
    Returns
    -------
    adjusted_moisture : jnp.ndarray
        Adjusted fuel moisture (fraction)
    """
    # Solar exposure adjustment
    # More exposure = drier (lower moisture)
    exposure_anomaly = solar_exposure - params.reference_exposure
    solar_adjustment = -params.solar_moisture_sensitivity * exposure_anomaly
    
    # Aspect adjustment (N. hemisphere: S-facing drier, N-facing wetter)
    # Convert aspect to "southness" (-1 to 1)
    if latitude >= 0:  # Northern hemisphere
        # South (180°) = driest, North (0°/360°) = wettest
        southness = -jnp.cos(jnp.radians(aspect))
    else:  # Southern hemisphere
        # North (0°) = driest, South (180°) = wettest
        southness = jnp.cos(jnp.radians(aspect))
    
    aspect_adjustment = -params.aspect_moisture_sensitivity * southness
    
    # Combine adjustments
    total_adjustment = solar_adjustment + aspect_adjustment
    total_adjustment = jnp.clip(
        total_adjustment,
        -params.max_moisture_adjustment,
        params.max_moisture_adjustment
    )
    
    # Apply adjustment
    adjusted_moisture = base_moisture + total_adjustment
    
    # Ensure physical bounds (moisture can't go below ~3% or above 30%)
    adjusted_moisture = jnp.clip(adjusted_moisture, 0.03, 0.30)
    
    return adjusted_moisture


def compute_fuel_conditioning(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
    base_moisture: float,
    latitude: float,
    longitude: float,
    dt: datetime,
    params: FuelConditioningParams = FuelConditioningParams(),
) -> jnp.ndarray:
    """
    Compute spatially-varying fuel moisture based on terrain and sun.
    
    This is the main entry point for fuel conditioning.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid (meters)
    dx, dy : float
        Grid spacing (meters)
    base_moisture : float
        Base fuel moisture (fraction)
    latitude, longitude : float
        Location
    dt : datetime
        Current datetime
    params : FuelConditioningParams
        Conditioning parameters
        
    Returns
    -------
    moisture_grid : jnp.ndarray
        Spatially-varying fuel moisture (fraction)
    """
    # Compute terrain properties
    slope, aspect = compute_slope_aspect_from_dem(dem, dx, dy)
    
    # Compute solar exposure
    solar_exposure = compute_solar_exposure_index(
        dem, dx, dy, latitude, longitude, dt,
        hours_back=6, time_step_hours=1.0
    )
    
    # Create base moisture grid
    base_moisture_grid = jnp.full_like(dem, base_moisture)
    
    # Adjust for solar exposure and aspect
    adjusted_moisture = adjust_fuel_moisture_solar(
        base_moisture_grid, solar_exposure, aspect, latitude, params
    )
    
    return adjusted_moisture
