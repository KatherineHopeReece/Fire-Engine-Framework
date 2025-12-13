"""
Dynamic Phenology Module - Live Fuel Moisture and Curing

Implements spatially-varying vegetation phenology that affects fire behavior.
Standard models treat curing (grass dryness) as a global constant, but in 
reality it varies dramatically across the landscape.

Physical Background
-------------------
Plant phenology (green-up, curing, senescence) is driven by:
1. Accumulated solar radiation (Growing Degree Days analog)
2. Aspect: South-facing slopes cure weeks earlier than north-facing
3. Elevation: Higher elevations green up later, cure earlier
4. Soil moisture history (approximated by recent precip)

This creates critical fire behavior boundaries where a fire may:
- Rage up a cured (dry) valley
- Die instantly when it hits a green (moist) ridge
- Accelerate through a "cured corridor" between green patches

Implementation
--------------
We model a "Virtual Greenness Index" (VGI) based on:
- Cumulative solar exposure from the solar_radiation module
- Elevation relative to local mean
- Day of year (seasonal phenology curve)
- Optional: NDVI from satellite if available

This VGI then modifies:
- Grass curing percentage (O-1 fuels)
- Live fuel moisture (conifer and mixed fuels)
- ISI calculation through effective FFMC

References
----------
- Jolly, W.M. et al. (2014). Climate-induced variations in global wildfire
  danger from 1979 to 2013. Nature Communications.
- Dennison, P.E. et al. (2005). Use of NDVI for fire risk rating.
- Anderson, S.A.J. (2009). A method to calculate fine fuel moisture content
  based on temperature and relative humidity.
"""

from __future__ import annotations
from typing import NamedTuple, Optional
from datetime import datetime
import jax
import jax.numpy as jnp
from functools import partial


# =============================================================================
# Data Structures
# =============================================================================

class PhenologyParams(NamedTuple):
    """Parameters for phenology model."""
    
    # Growing degree day accumulation
    base_temperature: float = 5.0     # °C - GDD base temperature
    gdd_green_up: float = 150.0       # GDD for full green-up
    gdd_cure_start: float = 800.0     # GDD when curing begins
    gdd_full_cure: float = 1200.0     # GDD for full curing
    
    # Aspect effects (modification to GDD accumulation rate)
    south_aspect_speedup: float = 1.5    # South faces accumulate faster
    north_aspect_slowdown: float = 0.6   # North faces accumulate slower
    
    # Elevation effects
    elevation_lapse_rate: float = 6.5    # Days later per 100m elevation
    reference_elevation: float = 1500.0  # Reference elevation (m)
    
    # Solar exposure effects
    solar_curing_coef: float = 0.1       # How much solar speeds curing
    shade_moisture_boost: float = 0.2    # Extra moisture in shade
    
    # Live fuel moisture bounds
    lfm_cured: float = 30.0              # LFM when fully cured (%)
    lfm_green: float = 150.0             # LFM when fully green (%)
    
    # Grass curing bounds
    curing_min: float = 20.0             # Minimum curing (%)
    curing_max: float = 100.0            # Maximum curing (%)
    
    # Seasonal curve parameters (Northern Hemisphere)
    green_up_doy: int = 120              # Day of year for green-up start
    peak_green_doy: int = 180            # Day of year for peak greenness
    cure_start_doy: int = 220            # Day of year curing starts
    full_cure_doy: int = 280             # Day of year fully cured


class PhenologyState(NamedTuple):
    """State variables for phenology tracking."""
    
    # Accumulated values
    gdd_accumulated: jnp.ndarray         # Growing degree days
    solar_accumulated: jnp.ndarray       # Cumulative solar exposure (MJ/m²)
    precip_accumulated: jnp.ndarray      # Recent precipitation (mm)
    
    # Current phenology
    greenness_index: jnp.ndarray         # Virtual Greenness Index [0-1]
    curing_fraction: jnp.ndarray         # Grass curing [0-1]
    live_fuel_moisture: jnp.ndarray      # Live fuel moisture (%)


class PhenologyResult(NamedTuple):
    """Results from phenology calculation."""
    
    curing_grid: jnp.ndarray             # Spatially-varying curing (%)
    lfm_grid: jnp.ndarray                # Live fuel moisture (%)
    greenness_index: jnp.ndarray         # Greenness [0-1]
    ffmc_modifier: jnp.ndarray           # Modifier to apply to FFMC
    fuel_moisture_modifier: jnp.ndarray  # Modifier for fine fuel moisture


# =============================================================================
# Core Physics Functions
# =============================================================================

@jax.jit
def compute_aspect_modifier(
    aspect_degrees: jnp.ndarray,
    params: PhenologyParams,
) -> jnp.ndarray:
    """
    Compute aspect-based modifier for phenology rate.
    
    South-facing slopes receive more solar radiation, accumulate GDD faster,
    and thus cure earlier. North-facing slopes stay green longer.
    
    Parameters
    ----------
    aspect_degrees : array
        Terrain aspect (downslope direction, degrees from north)
    params : PhenologyParams
        Model parameters
        
    Returns
    -------
    array
        Modifier to GDD accumulation rate (>1 = faster, <1 = slower)
    """
    # Convert aspect to radians
    aspect_rad = jnp.deg2rad(aspect_degrees)
    
    # South = 180°, so cos(aspect - 180°) = -cos(aspect) gives south positive
    # We use a shifted cosine: max at south (180°), min at north (0°/360°)
    south_factor = -jnp.cos(aspect_rad)  # -1 (north) to +1 (south)
    
    # Map to speedup/slowdown range
    # south_factor = 1 -> speedup = south_aspect_speedup
    # south_factor = -1 -> speedup = north_aspect_slowdown
    modifier = jnp.where(
        south_factor > 0,
        1.0 + (params.south_aspect_speedup - 1.0) * south_factor,
        1.0 + (1.0 - params.north_aspect_slowdown) * south_factor
    )
    
    return modifier


@jax.jit
def compute_elevation_delay(
    dem: jnp.ndarray,
    params: PhenologyParams,
) -> jnp.ndarray:
    """
    Compute phenology delay due to elevation.
    
    Higher elevations have delayed green-up and earlier curing.
    
    Parameters
    ----------
    dem : array
        Elevation in meters
    params : PhenologyParams
        Model parameters
        
    Returns
    -------
    array
        Day offset (positive = delayed phenology)
    """
    elevation_diff = dem - params.reference_elevation
    
    # Days of delay per 100m elevation
    delay_days = elevation_diff / 100.0 * params.elevation_lapse_rate
    
    return delay_days


@jax.jit
def compute_seasonal_greenness(
    day_of_year: int,
    elevation_delay: jnp.ndarray,
    aspect_modifier: jnp.ndarray,
    params: PhenologyParams,
) -> jnp.ndarray:
    """
    Compute base seasonal greenness curve.
    
    This models the typical phenology cycle:
    1. Spring green-up (DOY 100-150)
    2. Peak greenness (DOY 150-220)
    3. Fall curing (DOY 220-280)
    4. Winter dormancy (DOY 280-100)
    
    Parameters
    ----------
    day_of_year : int
        Current day of year (1-366)
    elevation_delay : array
        Phenology delay due to elevation (days)
    aspect_modifier : array
        Aspect-based rate modifier
    params : PhenologyParams
        Model parameters
        
    Returns
    -------
    array
        Seasonal greenness [0-1]
    """
    # Effective DOY accounting for elevation and aspect
    # Higher elevation = earlier in season (delayed phenology)
    # South aspect = later in season (accelerated phenology)
    effective_doy = day_of_year - elevation_delay
    effective_doy = effective_doy / aspect_modifier
    
    # Green-up phase (DOY 100-180)
    greenup_progress = (effective_doy - params.green_up_doy) / \
                       (params.peak_green_doy - params.green_up_doy)
    greenup_progress = jnp.clip(greenup_progress, 0.0, 1.0)
    
    # Curing phase (DOY 220-280)
    curing_progress = (effective_doy - params.cure_start_doy) / \
                      (params.full_cure_doy - params.cure_start_doy)
    curing_progress = jnp.clip(curing_progress, 0.0, 1.0)
    
    # Greenness: rises during green-up, falls during curing
    greenness = jnp.where(
        effective_doy < params.peak_green_doy,
        greenup_progress,  # Rising
        1.0 - curing_progress  # Falling
    )
    
    # Winter: force low greenness
    winter_mask = (effective_doy < params.green_up_doy) | \
                  (effective_doy > params.full_cure_doy)
    greenness = jnp.where(winter_mask, 0.1, greenness)
    
    return jnp.clip(greenness, 0.0, 1.0)


@jax.jit
def compute_solar_greenness_modifier(
    solar_accumulated: jnp.ndarray,
    solar_mean: jnp.ndarray,
    params: PhenologyParams,
) -> jnp.ndarray:
    """
    Compute greenness modifier based on solar exposure.
    
    Areas with more solar exposure cure faster (lower greenness).
    Areas with less solar stay greener longer.
    
    Parameters
    ----------
    solar_accumulated : array
        Cumulative solar radiation (MJ/m²)
    solar_mean : array or scalar
        Domain mean solar accumulation for normalization
    params : PhenologyParams
        Model parameters
        
    Returns
    -------
    array
        Modifier to greenness (-0.3 to +0.2)
    """
    # Normalize solar exposure
    solar_anomaly = jnp.where(
        solar_mean > 0,
        (solar_accumulated - solar_mean) / jnp.maximum(solar_mean, 1e-6),
        jnp.zeros_like(solar_accumulated)
    )
    
    # More sun = lower greenness (faster curing)
    # Less sun = higher greenness (stays green)
    modifier = -params.solar_curing_coef * solar_anomaly
    modifier = modifier + params.shade_moisture_boost * jnp.where(
        solar_anomaly < -0.3, 1.0, 0.0
    )
    
    return jnp.clip(modifier, -0.3, 0.2)


@jax.jit
def greenness_to_curing(
    greenness: jnp.ndarray,
    params: PhenologyParams,
) -> jnp.ndarray:
    """
    Convert greenness index to grass curing percentage.
    
    Parameters
    ----------
    greenness : array
        Greenness index [0-1]
    params : PhenologyParams
        Model parameters
        
    Returns
    -------
    array
        Grass curing percentage
    """
    # Curing is inverse of greenness
    # Greenness 1.0 -> curing_min
    # Greenness 0.0 -> curing_max
    curing = params.curing_min + (1.0 - greenness) * \
             (params.curing_max - params.curing_min)
    
    return curing


@jax.jit
def greenness_to_lfm(
    greenness: jnp.ndarray,
    params: PhenologyParams,
) -> jnp.ndarray:
    """
    Convert greenness index to live fuel moisture.
    
    Parameters
    ----------
    greenness : array
        Greenness index [0-1]
    params : PhenologyParams
        Model parameters
        
    Returns
    -------
    array
        Live fuel moisture (%)
    """
    # LFM scales linearly with greenness
    lfm = params.lfm_cured + greenness * (params.lfm_green - params.lfm_cured)
    
    return lfm


@jax.jit
def compute_ffmc_modifier(
    greenness: jnp.ndarray,
    lfm: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute modifier to effective FFMC based on live fuel state.
    
    Green vegetation increases effective fine fuel moisture,
    reducing the effective FFMC. Cured vegetation has no effect.
    
    Parameters
    ----------
    greenness : array
        Greenness index [0-1]
    lfm : array
        Live fuel moisture (%)
        
    Returns
    -------
    array
        Additive modifier to FFMC (negative = wetter = slower spread)
    """
    # High greenness = high LFM = reduction in effective FFMC
    # This models the "green barrier" effect where fires die out
    
    # At peak greenness (LFM ~150%), reduce FFMC by up to 10 points
    # At full curing, no modification
    ffmc_reduction = -10.0 * greenness * jnp.minimum(lfm / 150.0, 1.0)
    
    return ffmc_reduction


# =============================================================================
# NDVI Integration (Optional Satellite Data)
# =============================================================================

@jax.jit
def integrate_ndvi(
    modeled_greenness: jnp.ndarray,
    ndvi: jnp.ndarray,
    ndvi_weight: float = 0.7,
) -> jnp.ndarray:
    """
    Integrate satellite NDVI with modeled greenness.
    
    If NDVI data is available, blend it with modeled values for
    improved accuracy.
    
    Parameters
    ----------
    modeled_greenness : array
        Greenness from phenology model [0-1]
    ndvi : array
        Normalized Difference Vegetation Index [-1 to 1]
    ndvi_weight : float
        Weight for NDVI (0-1), remainder goes to model
        
    Returns
    -------
    array
        Blended greenness index [0-1]
    """
    # Convert NDVI to greenness scale
    # NDVI typically ranges from -0.1 (bare/water) to 0.9 (dense veg)
    ndvi_greenness = jnp.clip((ndvi + 0.1) / 1.0, 0.0, 1.0)
    
    # Blend
    blended = ndvi_weight * ndvi_greenness + (1.0 - ndvi_weight) * modeled_greenness
    
    return blended


# =============================================================================
# Main Interface Functions
# =============================================================================

def initialize_phenology_state(
    shape: tuple[int, int],
    dem: jnp.ndarray,
    start_date: datetime,
    params: PhenologyParams = None,
) -> PhenologyState:
    """
    Initialize phenology state for a domain.
    
    Parameters
    ----------
    shape : tuple
        Grid shape (ny, nx)
    dem : array
        Digital elevation model
    start_date : datetime
        Simulation start date
    params : PhenologyParams, optional
        Model parameters
        
    Returns
    -------
    PhenologyState
        Initialized state with seasonal values
    """
    if params is None:
        params = PhenologyParams()
    
    ny, nx = shape
    
    # Initialize accumulators at zero
    gdd_acc = jnp.zeros((ny, nx))
    solar_acc = jnp.zeros((ny, nx))
    precip_acc = jnp.zeros((ny, nx))
    
    # Compute initial greenness from date
    doy = start_date.timetuple().tm_yday
    
    # Need aspect for initial calculation - use flat as placeholder
    # Real aspect will be computed in first update
    greenness = compute_seasonal_greenness(
        doy,
        compute_elevation_delay(dem, params),
        jnp.ones((ny, nx)),  # Placeholder aspect modifier
        params,
    )
    
    curing = greenness_to_curing(greenness, params)
    lfm = greenness_to_lfm(greenness, params)
    
    return PhenologyState(
        gdd_accumulated=gdd_acc,
        solar_accumulated=solar_acc,
        precip_accumulated=precip_acc,
        greenness_index=greenness,
        curing_fraction=curing / 100.0,
        live_fuel_moisture=lfm,
    )


@partial(jax.jit, static_argnums=(9,))
def update_phenology_state(
    state: PhenologyState,
    dem: jnp.ndarray,
    aspect: jnp.ndarray,
    temperature: float,
    solar_radiation: jnp.ndarray,
    precipitation: float,
    day_of_year: int,
    dt_hours: float,
    ndvi: Optional[jnp.ndarray],
    params: PhenologyParams,
) -> PhenologyState:
    """
    Update phenology state for one timestep.
    
    Parameters
    ----------
    state : PhenologyState
        Current state
    dem : array
        Digital elevation model
    aspect : array
        Terrain aspect (degrees)
    temperature : float
        Air temperature (°C)
    solar_radiation : array
        Solar radiation grid (W/m²)
    precipitation : float
        Precipitation (mm)
    day_of_year : int
        Current day of year
    dt_hours : float
        Timestep in hours
    ndvi : array, optional
        NDVI satellite data if available
    params : PhenologyParams
        Model parameters
        
    Returns
    -------
    PhenologyState
        Updated state
    """
    # Accumulate GDD (only positive contributions)
    gdd_increment = jnp.maximum(temperature - params.base_temperature, 0.0) * dt_hours / 24.0
    gdd_accumulated = state.gdd_accumulated + gdd_increment
    
    # Accumulate solar radiation (convert W/m² to MJ/m² for dt)
    solar_increment = solar_radiation * dt_hours * 3600.0 / 1e6
    solar_accumulated = state.solar_accumulated + solar_increment
    
    # Track recent precipitation (exponential decay)
    decay_rate = 0.1  # Per day
    precip_accumulated = state.precip_accumulated * (1.0 - decay_rate * dt_hours / 24.0)
    precip_accumulated = precip_accumulated + precipitation
    
    # Compute modifiers
    aspect_modifier = compute_aspect_modifier(aspect, params)
    elevation_delay = compute_elevation_delay(dem, params)
    
    # Compute seasonal greenness
    greenness = compute_seasonal_greenness(
        day_of_year, elevation_delay, aspect_modifier, params
    )
    
    # Apply solar exposure modifier
    solar_mean = jnp.mean(solar_accumulated)
    solar_modifier = compute_solar_greenness_modifier(
        solar_accumulated, solar_mean, params
    )
    greenness = jnp.clip(greenness + solar_modifier, 0.0, 1.0)
    
    # Integrate NDVI if available
    if ndvi is not None:
        greenness = integrate_ndvi(greenness, ndvi)
    
    # Convert to curing and LFM
    curing = greenness_to_curing(greenness, params)
    lfm = greenness_to_lfm(greenness, params)
    
    return PhenologyState(
        gdd_accumulated=gdd_accumulated,
        solar_accumulated=solar_accumulated,
        precip_accumulated=precip_accumulated,
        greenness_index=greenness,
        curing_fraction=curing / 100.0,
        live_fuel_moisture=lfm,
    )


@partial(jax.jit, static_argnums=(1,))
def compute_phenology_effects(
    state: PhenologyState,
    params: PhenologyParams,
) -> PhenologyResult:
    """
    Compute fire behavior effects from current phenology.
    
    Parameters
    ----------
    state : PhenologyState
        Current phenology state
    params : PhenologyParams
        Model parameters
        
    Returns
    -------
    PhenologyResult
        Results for modifying fire spread
    """
    # Grass curing percentage
    curing_grid = state.curing_fraction * 100.0
    
    # Live fuel moisture
    lfm_grid = state.live_fuel_moisture
    
    # FFMC modifier
    ffmc_modifier = compute_ffmc_modifier(
        state.greenness_index, state.live_fuel_moisture
    )
    
    # General fuel moisture modifier
    # High greenness = higher moisture content = slower spread
    fuel_moisture_modifier = 1.0 - 0.3 * state.greenness_index
    
    return PhenologyResult(
        curing_grid=curing_grid,
        lfm_grid=lfm_grid,
        greenness_index=state.greenness_index,
        ffmc_modifier=ffmc_modifier,
        fuel_moisture_modifier=fuel_moisture_modifier,
    )


# =============================================================================
# Fuel-Specific Functions
# =============================================================================

@jax.jit
def apply_curing_to_grass_fuels(
    base_ros: jnp.ndarray,
    fuel_type: jnp.ndarray,
    curing_grid: jnp.ndarray,
    standard_curing: float = 85.0,
) -> jnp.ndarray:
    """
    Apply spatially-varying curing to grass fuel ROS.
    
    For O-1 (grass) fuels, ROS is highly sensitive to curing.
    This replaces the constant curing value with spatial variation.
    
    Parameters
    ----------
    base_ros : array
        Base rate of spread computed with standard curing
    fuel_type : array
        FBP fuel type codes (17=O-1a, 18=O-1b)
    curing_grid : array
        Spatially-varying curing (%)
    standard_curing : float
        The curing value used to compute base_ros
        
    Returns
    -------
    array
        Adjusted rate of spread
    """
    # FBP grass ROS is approximately proportional to curing effect
    # CF = 0.005 * (curing - 58)² for curing > 58
    
    is_grass = (fuel_type == 17) | (fuel_type == 18)
    
    # Compute curing effect ratio
    def curing_factor(c):
        return jnp.where(
            c > 58.0,
            0.005 * (c - 58.0)**2,
            0.0
        )
    
    standard_cf = curing_factor(standard_curing)
    spatial_cf = curing_factor(curing_grid)
    
    # Ratio of effects
    cf_ratio = jnp.where(
        standard_cf > 0.01,
        spatial_cf / standard_cf,
        1.0
    )
    
    # Apply only to grass fuels
    adjusted_ros = jnp.where(is_grass, base_ros * cf_ratio, base_ros)
    
    return adjusted_ros


@jax.jit
def apply_lfm_to_conifer_fuels(
    base_ros: jnp.ndarray,
    fuel_type: jnp.ndarray,
    lfm_grid: jnp.ndarray,
    standard_fmc: float = 100.0,
) -> jnp.ndarray:
    """
    Apply spatially-varying live fuel moisture to conifer ROS.
    
    For C-6 and C-7 fuels, crown involvement depends on foliar moisture.
    This allows spatial variation in FMC effects.
    
    Parameters
    ----------
    base_ros : array
        Base rate of spread computed with standard FMC
    fuel_type : array
        FBP fuel type codes
    lfm_grid : array
        Spatially-varying live fuel moisture (%)
    standard_fmc : float
        The FMC value used to compute base_ros
        
    Returns
    -------
    array
        Adjusted rate of spread
    """
    # C-6 (Conifer plantation) and C-7 (Ponderosa) are most sensitive
    # FMC effect is approximately 1 - 0.003*(FMC - 100) for FMC > 100
    
    is_conifer = (fuel_type >= 1) & (fuel_type <= 7)
    
    # FMC effect ratio
    standard_effect = 1.0 - 0.003 * jnp.maximum(standard_fmc - 100.0, 0.0)
    spatial_effect = 1.0 - 0.003 * jnp.maximum(lfm_grid - 100.0, 0.0)
    
    # Ensure minimum effect
    standard_effect = jnp.maximum(standard_effect, 0.3)
    spatial_effect = jnp.maximum(spatial_effect, 0.3)
    
    effect_ratio = spatial_effect / standard_effect
    
    # Apply only to conifer fuels
    adjusted_ros = jnp.where(is_conifer, base_ros * effect_ratio, base_ros)
    
    return adjusted_ros


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_phenology_from_climate(
    latitude: float,
    elevation: float,
    mean_annual_temp: float,
) -> PhenologyParams:
    """
    Estimate phenology parameters from climate normals.
    
    Parameters
    ----------
    latitude : float
        Site latitude (degrees)
    elevation : float
        Site elevation (m)
    mean_annual_temp : float
        Mean annual temperature (°C)
        
    Returns
    -------
    PhenologyParams
        Estimated parameters
    """
    # Adjust phenology dates based on latitude
    # Higher latitude = later green-up, earlier curing
    lat_adjustment = (latitude - 45.0) * 1.5  # Days per degree latitude
    
    green_up_doy = int(120 + lat_adjustment)
    peak_green_doy = int(180 + lat_adjustment * 0.5)
    cure_start_doy = int(220 - lat_adjustment * 0.5)
    full_cure_doy = int(280 - lat_adjustment)
    
    # Adjust reference elevation based on typical treeline
    ref_elevation = 2500.0 - 100.0 * (latitude - 45.0)
    
    return PhenologyParams(
        green_up_doy=green_up_doy,
        peak_green_doy=peak_green_doy,
        cure_start_doy=cure_start_doy,
        full_cure_doy=full_cure_doy,
        reference_elevation=ref_elevation,
    )


def summarize_phenology(state: PhenologyState) -> dict:
    """
    Generate summary statistics of current phenology.
    
    Parameters
    ----------
    state : PhenologyState
        Current phenology state
        
    Returns
    -------
    dict
        Summary statistics
    """
    return {
        "mean_greenness": float(jnp.mean(state.greenness_index)),
        "min_greenness": float(jnp.min(state.greenness_index)),
        "max_greenness": float(jnp.max(state.greenness_index)),
        "mean_curing_pct": float(jnp.mean(state.curing_fraction) * 100.0),
        "mean_lfm_pct": float(jnp.mean(state.live_fuel_moisture)),
        "min_lfm_pct": float(jnp.min(state.live_fuel_moisture)),
        "gdd_mean": float(jnp.mean(state.gdd_accumulated)),
        "cured_fraction": float(jnp.mean(state.curing_fraction > 0.8)),
        "green_fraction": float(jnp.mean(state.greenness_index > 0.7)),
    }
