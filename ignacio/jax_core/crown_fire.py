"""
Crown Fire Transition Model.

Implements Van Wagner's (1977) criteria for crown fire initiation and
active crown fire spread. This completes the Rothermel model which is
strictly a surface fire model.

Physics:
1. Crown fire initiation: Does surface fire intensity exceed the critical
   intensity needed to ignite the canopy?
   - I_surface > I_critical = (0.010 * CBH * (460 + 26*FMC))^1.5
   
2. Active crown fire: Is the ROS sufficient to sustain a running crown fire?
   - ROS > ROS_critical = 3.0 / CBD
   
3. Crown fire ROS: If active crowning, ROS increases significantly
   - Various models: Cruz et al., Alexander & Cruz, etc.

Key Variables:
- CBH: Canopy Base Height (m) - height to bottom of canopy
- CBD: Canopy Bulk Density (kg/m³) - fuel density in canopy
- FMC: Foliar Moisture Content (%)
- CFB: Crown Fraction Burned (0-1)

References:
- Van Wagner, C.E. (1977). Conditions for the start and spread of crown fire.
- Cruz, M.G. et al. (2005). Development and testing of models for predicting
  crown fire rate of spread in conifer fuel stands.
- Alexander, M.E. & Cruz, M.G. (2012). Interdependencies between flame length
  and fireline intensity in predicting crown fire initiation and crown scorch height.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


class CrownFireParams(NamedTuple):
    """Parameters for crown fire model."""
    
    # Default canopy properties (used if no raster available)
    default_cbh: float = 3.0      # Canopy base height (m)
    default_cbd: float = 0.15     # Canopy bulk density (kg/m³)
    default_fmc: float = 100.0    # Foliar moisture content (%)
    
    # Crown fire spread parameters
    # Multiplier for crown ROS relative to surface ROS
    crown_ros_multiplier: float = 3.0
    
    # Cruz et al. model coefficients for crown ROS
    cruz_a: float = 11.02
    cruz_b: float = 0.90
    cruz_c: float = 0.19
    
    # Transition smoothness (for differentiable transition)
    # Higher = sharper transition at critical thresholds
    transition_temperature: float = 0.1
    
    # Minimum intensity for any crowning consideration (kW/m)
    min_crowning_intensity: float = 500.0


class CrownFireState(NamedTuple):
    """Crown fire state at each cell."""
    
    cfb: jnp.ndarray           # Crown Fraction Burned (0-1)
    is_crowning: jnp.ndarray   # Whether cell is actively crowning
    ros_crown: jnp.ndarray     # Crown fire ROS (m/min)
    ros_total: jnp.ndarray     # Total ROS (surface + crown)


def compute_byram_intensity(
    ros: jnp.ndarray,
    fuel_load: jnp.ndarray,
    heat_content: float = 18000.0,  # kJ/kg
) -> jnp.ndarray:
    """
    Compute Byram's fire line intensity.
    
    I = H * W * R
    
    Parameters
    ----------
    ros : jnp.ndarray
        Rate of spread (m/min)
    fuel_load : jnp.ndarray
        Fuel consumed (kg/m²)
    heat_content : float
        Heat of combustion (kJ/kg)
        
    Returns
    -------
    intensity : jnp.ndarray
        Fire line intensity (kW/m)
    """
    # Convert ROS from m/min to m/s
    ros_ms = ros / 60.0
    
    # Byram's intensity (kJ/s/m = kW/m)
    intensity = heat_content * fuel_load * ros_ms
    
    return intensity


def compute_critical_intensity(
    cbh: jnp.ndarray,
    fmc: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Van Wagner's critical surface intensity for crown ignition.
    
    I_0 = (0.010 * CBH * (460 + 26*FMC))^1.5
    
    Parameters
    ----------
    cbh : jnp.ndarray
        Canopy base height (m)
    fmc : jnp.ndarray
        Foliar moisture content (%)
        
    Returns
    -------
    I_critical : jnp.ndarray
        Critical intensity (kW/m)
    """
    # Van Wagner (1977) equation
    I_crit = jnp.power(0.010 * cbh * (460.0 + 26.0 * fmc), 1.5)
    
    return I_crit


def compute_critical_ros(
    cbd: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute critical ROS for active crown fire.
    
    R_0 = 3.0 / CBD
    
    Below this ROS, crown fire cannot sustain itself.
    
    Parameters
    ----------
    cbd : jnp.ndarray
        Canopy bulk density (kg/m³)
        
    Returns
    -------
    R_critical : jnp.ndarray
        Critical ROS (m/min)
    """
    # Avoid division by zero
    cbd_safe = jnp.maximum(cbd, 0.01)
    
    R_crit = 3.0 / cbd_safe
    
    return R_crit


def compute_crown_fraction_burned(
    surface_intensity: jnp.ndarray,
    critical_intensity: jnp.ndarray,
    ros: jnp.ndarray,
    critical_ros: jnp.ndarray,
    params: CrownFireParams = CrownFireParams(),
) -> jnp.ndarray:
    """
    Compute Crown Fraction Burned (CFB).
    
    CFB represents the fraction of canopy consumed, from 0 (surface fire)
    to 1 (fully active crown fire).
    
    Uses smooth transition for differentiability.
    
    Parameters
    ----------
    surface_intensity : jnp.ndarray
        Surface fire intensity (kW/m)
    critical_intensity : jnp.ndarray
        Critical intensity for crown ignition (kW/m)
    ros : jnp.ndarray
        Rate of spread (m/min)
    critical_ros : jnp.ndarray
        Critical ROS for active crowning (m/min)
    params : CrownFireParams
        Model parameters
        
    Returns
    -------
    cfb : jnp.ndarray
        Crown Fraction Burned (0-1)
    """
    # Crown ignition factor (0-1)
    # Using sigmoid for smooth transition
    intensity_ratio = surface_intensity / jnp.maximum(critical_intensity, 1.0)
    ignition_factor = jax.nn.sigmoid(
        (intensity_ratio - 1.0) / params.transition_temperature
    )
    
    # Active crowning factor (0-1)
    ros_ratio = ros / jnp.maximum(critical_ros, 0.1)
    active_factor = jax.nn.sigmoid(
        (ros_ratio - 1.0) / params.transition_temperature
    )
    
    # CFB is product of both conditions
    # Must have both crown ignition AND sufficient ROS
    cfb = ignition_factor * active_factor
    
    # Below minimum intensity, no crowning
    cfb = jnp.where(surface_intensity < params.min_crowning_intensity, 0.0, cfb)
    
    return cfb


def compute_crown_ros_cruz(
    surface_ros: jnp.ndarray,
    wind_speed: jnp.ndarray,
    cbd: jnp.ndarray,
    fmc: jnp.ndarray,
    cfb: jnp.ndarray,
    params: CrownFireParams = CrownFireParams(),
) -> jnp.ndarray:
    """
    Compute crown fire ROS using Cruz et al. (2005) model.
    
    R_crown = 11.02 * U^0.90 * CBD^0.19 * exp(-0.17*FMC)
    
    Parameters
    ----------
    surface_ros : jnp.ndarray
        Surface fire ROS (m/min)
    wind_speed : jnp.ndarray
        10-m wind speed (km/h)
    cbd : jnp.ndarray
        Canopy bulk density (kg/m³)
    fmc : jnp.ndarray
        Foliar moisture content (%)
    cfb : jnp.ndarray
        Crown Fraction Burned (0-1)
    params : CrownFireParams
        Model parameters
        
    Returns
    -------
    ros_crown : jnp.ndarray
        Crown fire ROS (m/min)
    """
    # Cruz et al. (2005) crown fire spread model
    # R = 11.02 * U^0.90 * CBD^0.19 * exp(-0.017*FMC)
    
    # Ensure positive values
    ws_safe = jnp.maximum(wind_speed, 0.1)
    cbd_safe = jnp.maximum(cbd, 0.01)
    
    R_crown = (params.cruz_a * 
               jnp.power(ws_safe, params.cruz_b) * 
               jnp.power(cbd_safe, params.cruz_c) * 
               jnp.exp(-0.017 * fmc))
    
    return R_crown


def compute_crown_ros_simple(
    surface_ros: jnp.ndarray,
    cfb: jnp.ndarray,
    params: CrownFireParams = CrownFireParams(),
) -> jnp.ndarray:
    """
    Simple crown fire ROS model (multiplier approach).
    
    When crown fire is active, ROS increases by a factor.
    
    Parameters
    ----------
    surface_ros : jnp.ndarray
        Surface fire ROS (m/min)
    cfb : jnp.ndarray
        Crown Fraction Burned (0-1)
    params : CrownFireParams
        Model parameters
        
    Returns
    -------
    ros_total : jnp.ndarray
        Total ROS including crown fire (m/min)
    """
    # Linear blend based on CFB
    crown_contribution = cfb * (params.crown_ros_multiplier - 1.0)
    ros_total = surface_ros * (1.0 + crown_contribution)
    
    return ros_total


def compute_total_ros_with_crown(
    surface_ros: jnp.ndarray,
    fuel_load: jnp.ndarray,
    cbh: jnp.ndarray,
    cbd: jnp.ndarray,
    fmc: jnp.ndarray,
    wind_speed: jnp.ndarray,
    params: CrownFireParams = CrownFireParams(),
    use_cruz_model: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute total ROS including crown fire transition.
    
    Main entry point for crown fire calculations.
    
    Parameters
    ----------
    surface_ros : jnp.ndarray
        Surface fire ROS from Rothermel (m/min)
    fuel_load : jnp.ndarray
        Surface fuel load (kg/m²)
    cbh : jnp.ndarray
        Canopy base height (m)
    cbd : jnp.ndarray
        Canopy bulk density (kg/m³)
    fmc : jnp.ndarray
        Foliar moisture content (%)
    wind_speed : jnp.ndarray
        Wind speed (km/h)
    params : CrownFireParams
        Model parameters
    use_cruz_model : bool
        Whether to use Cruz et al. crown ROS model
        
    Returns
    -------
    total_ros : jnp.ndarray
        Total ROS including crown fire (m/min)
    cfb : jnp.ndarray
        Crown Fraction Burned (0-1)
    """
    # Compute surface intensity
    surface_intensity = compute_byram_intensity(surface_ros, fuel_load)
    
    # Compute critical values
    I_critical = compute_critical_intensity(cbh, fmc)
    R_critical = compute_critical_ros(cbd)
    
    # Compute CFB
    cfb = compute_crown_fraction_burned(
        surface_intensity, I_critical, 
        surface_ros, R_critical,
        params
    )
    
    # Compute crown fire ROS
    if use_cruz_model:
        ros_crown = compute_crown_ros_cruz(
            surface_ros, wind_speed, cbd, fmc, cfb, params
        )
        # Blend surface and crown ROS based on CFB
        total_ros = surface_ros * (1 - cfb) + ros_crown * cfb
    else:
        total_ros = compute_crown_ros_simple(surface_ros, cfb, params)
    
    return total_ros, cfb


def apply_crown_fire_to_grids(
    ros_grid: jnp.ndarray,
    fuel_load_grid: jnp.ndarray,
    cbh_grid: Optional[jnp.ndarray] = None,
    cbd_grid: Optional[jnp.ndarray] = None,
    fmc: float = 100.0,
    wind_speed: float = 15.0,
    params: CrownFireParams = CrownFireParams(),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply crown fire model to ROS grids.
    
    Convenience function that handles default canopy values.
    
    Parameters
    ----------
    ros_grid : jnp.ndarray
        Surface ROS grid (m/min)
    fuel_load_grid : jnp.ndarray
        Fuel load grid (kg/m²)
    cbh_grid : jnp.ndarray, optional
        Canopy base height grid. Uses default if None.
    cbd_grid : jnp.ndarray, optional
        Canopy bulk density grid. Uses default if None.
    fmc : float
        Foliar moisture content (%)
    wind_speed : float
        Wind speed (km/h)
    params : CrownFireParams
        Model parameters
        
    Returns
    -------
    total_ros : jnp.ndarray
        Total ROS including crown fire
    cfb : jnp.ndarray
        Crown Fraction Burned
    """
    shape = ros_grid.shape
    
    # Use defaults if grids not provided
    if cbh_grid is None:
        cbh_grid = jnp.full(shape, params.default_cbh)
    if cbd_grid is None:
        cbd_grid = jnp.full(shape, params.default_cbd)
    
    fmc_grid = jnp.full(shape, fmc)
    wind_grid = jnp.full(shape, wind_speed)
    
    return compute_total_ros_with_crown(
        ros_grid, fuel_load_grid,
        cbh_grid, cbd_grid, fmc_grid, wind_grid,
        params
    )


# =============================================================================
# Crown Fire State Evolution (for level-set integration)
# =============================================================================

def update_crown_fire_state(
    phi: jnp.ndarray,
    surface_ros: jnp.ndarray,
    fuel_load: jnp.ndarray,
    cbh: jnp.ndarray,
    cbd: jnp.ndarray,
    fmc: float,
    wind_speed: float,
    params: CrownFireParams = CrownFireParams(),
) -> CrownFireState:
    """
    Update crown fire state for level-set simulation.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Level-set field
    surface_ros : jnp.ndarray
        Surface fire ROS
    fuel_load : jnp.ndarray
        Surface fuel load
    cbh, cbd : jnp.ndarray
        Canopy properties
    fmc : float
        Foliar moisture
    wind_speed : float
        Wind speed
    params : CrownFireParams
        Model parameters
        
    Returns
    -------
    state : CrownFireState
        Updated crown fire state
    """
    total_ros, cfb = apply_crown_fire_to_grids(
        surface_ros, fuel_load, cbh, cbd, fmc, wind_speed, params
    )
    
    # Determine if actively crowning (CFB > 0.5 threshold)
    is_crowning = cfb > 0.5
    
    # Crown ROS is the excess over surface
    ros_crown = total_ros - surface_ros
    ros_crown = jnp.maximum(ros_crown, 0.0)
    
    return CrownFireState(
        cfb=cfb,
        is_crowning=is_crowning,
        ros_crown=ros_crown,
        ros_total=total_ros,
    )


def get_default_fuel_load(fuel_code: int) -> float:
    """
    Get typical fuel load for a Rothermel fuel model.
    
    Parameters
    ----------
    fuel_code : int
        Rothermel fuel model number
        
    Returns
    -------
    fuel_load : float
        Typical fuel load (kg/m²)
    """
    # Approximate fuel loads for common models
    fuel_loads = {
        1: 0.17,   # Short grass
        2: 0.45,   # Timber grass
        3: 0.68,   # Tall grass
        4: 2.5,    # Chaparral
        5: 0.45,   # Brush
        6: 0.90,   # Dormant brush
        7: 0.80,   # Southern rough
        8: 0.90,   # Compact timber litter
        9: 0.78,   # Hardwood litter
        10: 2.25,  # Timber understory
        11: 2.58,  # Light slash
        12: 7.05,  # Medium slash
        13: 12.9,  # Heavy slash
    }
    return fuel_loads.get(fuel_code, 1.0)


def estimate_canopy_properties(
    fuel_code: int,
) -> Tuple[float, float]:
    """
    Estimate canopy properties from fuel model.
    
    For use when CBH/CBD rasters are not available.
    
    Parameters
    ----------
    fuel_code : int
        Rothermel fuel model number
        
    Returns
    -------
    cbh, cbd : float
        Estimated canopy base height (m) and bulk density (kg/m³)
    """
    # Estimates based on typical forest conditions
    # Grass/shrub models: no canopy
    # Timber models: typical conifer canopy
    
    if fuel_code in [1, 2, 3]:  # Grass
        return 99.0, 0.0  # No canopy (high CBH = no crowning)
    elif fuel_code in [4, 5, 6, 7]:  # Shrub
        return 1.0, 0.05  # Low brush canopy
    elif fuel_code in [8, 9]:  # Compact litter
        return 4.0, 0.10  # Moderate canopy
    elif fuel_code == 10:  # Timber understory
        return 3.0, 0.15  # Significant canopy
    elif fuel_code in [11, 12, 13]:  # Slash
        return 2.0, 0.20  # Dense debris/regeneration
    else:
        return 5.0, 0.10  # Default moderate canopy
