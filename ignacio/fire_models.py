"""
Unified Fire Model Interface.

Provides a common interface to compute rate of spread (ROS) using
either the Canadian FBP system or the US Rothermel model.

This module dispatches to the appropriate model based on configuration.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# FBP to Rothermel Fuel Code Mapping
# =============================================================================

# Default mapping from Canadian FBP fuel types to Rothermel fuel models
# This is approximate - actual mapping depends on local fuel characteristics
FBP_TO_ROTHERMEL = {
    # Canadian Boreal - FBP codes to closest Rothermel equivalents
    
    # Conifer fuels
    "C-1": 8,    # Spruce-Lichen -> Compact timber litter
    "C-2": 10,   # Boreal Spruce -> Timber understory  
    "C-3": 9,    # Mature Jack Pine -> Hardwood litter
    "C-4": 10,   # Immature Jack Pine -> Timber understory
    "C-5": 181,  # Red/White Pine -> TL1 Low load timber litter
    "C-6": 10,   # Conifer plantation -> Timber understory
    "C-7": 183,  # Ponderosa Pine -> TL3 Moderate conifer litter
    
    # Deciduous fuels
    "D-1": 186,  # Leafless aspen -> TL6 Broadleaf litter
    "D-2": 9,    # Green aspen -> Hardwood litter
    
    # Mixedwood fuels
    "M-1": 10,   # Boreal mixedwood (leafless) -> Timber understory
    "M-2": 10,   # Boreal mixedwood (green) -> Timber understory
    "M-3": 165,  # Dead balsam fir -> TU5 High load timber-shrub
    "M-4": 165,  # Dead balsam fir (green) -> TU5
    
    # Slash fuels  
    "S-1": 11,   # Jack pine slash -> Light slash
    "S-2": 12,   # White spruce slash -> Medium slash
    "S-3": 13,   # Coastal cedar slash -> Heavy slash
    
    # Open fuels
    "O-1a": 1,   # Matted grass -> Short grass
    "O-1b": 3,   # Standing grass -> Tall grass
    
    # Non-fuel
    "NF": 0,     # Non-fuel
    "WA": 0,     # Water
}

# Numeric FBP codes to Rothermel
FBP_CODE_TO_ROTHERMEL = {
    1: 8,    # C-1
    2: 10,   # C-2
    3: 9,    # C-3
    4: 10,   # C-4
    5: 181,  # C-5
    6: 10,   # C-6
    7: 183,  # C-7
    11: 186, # D-1
    12: 9,   # D-2
    21: 10,  # M-1
    22: 10,  # M-2
    23: 165, # M-3
    24: 165, # M-4
    31: 11,  # S-1
    32: 12,  # S-2
    33: 13,  # S-3
    41: 1,   # O-1a
    42: 3,   # O-1b
    # Non-fuel codes
    101: 0,  # NF
    102: 0,  # WA (Water)
    103: 0,  # Unknown
    # Additional non-fuel codes that might appear
    0: 0,
    -1: 0,
    255: 0,
    -9999: 0,
}


def map_fbp_to_rothermel(fbp_code: int, custom_mapping: Optional[Dict[int, int]] = None) -> int:
    """
    Map FBP fuel code to Rothermel fuel model.
    
    Parameters
    ----------
    fbp_code : int
        Canadian FBP fuel type code
    custom_mapping : dict, optional
        Custom mapping to override defaults
        
    Returns
    -------
    rothermel_code : int
        Rothermel fuel model number (0 = non-fuel)
    """
    if custom_mapping and fbp_code in custom_mapping:
        return custom_mapping[fbp_code]
    return FBP_CODE_TO_ROTHERMEL.get(fbp_code, 0)


def convert_fuel_grid_to_rothermel(
    fbp_fuel_grid: np.ndarray,
    custom_mapping: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """
    Convert FBP fuel code grid to Rothermel fuel model codes.
    
    Parameters
    ----------
    fbp_fuel_grid : ndarray
        Grid of FBP fuel type codes
    custom_mapping : dict, optional
        Custom mapping to override defaults
        
    Returns
    -------
    rothermel_grid : ndarray
        Grid of Rothermel fuel model codes
    """
    rothermel_grid = np.zeros_like(fbp_fuel_grid, dtype=np.int32)
    
    unique_codes = np.unique(fbp_fuel_grid)
    for code in unique_codes:
        if np.isnan(code):
            continue
        code_int = int(code)
        rothermel_code = map_fbp_to_rothermel(code_int, custom_mapping)
        rothermel_grid[fbp_fuel_grid == code] = rothermel_code
    
    return rothermel_grid


# =============================================================================
# Unified ROS Computation
# =============================================================================

def compute_ros_unified(
    fuel_code: int,
    model: str,
    isi: float = 5.0,
    bui: float = 50.0,
    fmc: float = 100.0,
    curing: float = 85.0,
    moisture_1hr: float = 0.08,
    moisture_10hr: float = 0.09,
    moisture_100hr: float = 0.10,
    moisture_live: float = 1.0,
    wind_speed: float = 15.0,
    slope: float = 0.0,
    fuel_lookup: Optional[Dict] = None,
    rothermel_fuel_models: Optional[Dict] = None,
) -> float:
    """
    Compute rate of spread using specified fire model.
    
    Parameters
    ----------
    fuel_code : int
        Fuel type code
    model : str
        Fire model to use: "fbp" or "rothermel"
    isi : float
        Initial Spread Index (for FBP)
    bui : float
        Buildup Index (for FBP)
    fmc : float
        Foliar moisture content (%) (for FBP)
    curing : float
        Grass curing (%) (for FBP)
    moisture_* : float
        Fuel moisture fractions (for Rothermel)
    wind_speed : float
        Wind speed (km/h for both)
    slope : float
        Slope (degrees)
    fuel_lookup : dict, optional
        FBP fuel name lookup
    rothermel_fuel_models : dict, optional
        Rothermel fuel model definitions
        
    Returns
    -------
    ros : float
        Rate of spread (m/min)
    """
    if model.lower() == "fbp":
        try:
            from ignacio.fbp import compute_ros
        except ImportError:
            from .fbp import compute_ros
        return compute_ros(
            fuel_type=fuel_code,
            isi=isi,
            bui=bui,
            fmc=fmc,
            curing=curing,
            fuel_lookup=fuel_lookup,
        )
    
    elif model.lower() == "rothermel":
        try:
            from ignacio.jax_core.rothermel import FUEL_MODELS, rothermel_ros
        except ImportError:
            from .jax_core.rothermel import FUEL_MODELS, rothermel_ros
        
        if rothermel_fuel_models is None:
            rothermel_fuel_models = FUEL_MODELS
        
        if fuel_code not in rothermel_fuel_models:
            return 0.0
        
        fuel_model = rothermel_fuel_models[fuel_code]
        return float(rothermel_ros(
            fuel_model,
            moisture_1hr,
            moisture_10hr,
            moisture_100hr,
            moisture_live,
            wind_speed,
            slope,
        ))
    
    else:
        raise ValueError(f"Unknown fire model: {model}. Use 'fbp' or 'rothermel'.")


def compute_ros_grid_unified(
    fuel_grid: np.ndarray,
    model: str,
    # FBP parameters
    isi: float = 5.0,
    bui: float = 50.0,
    fmc: float = 100.0,
    curing: float = 85.0,
    # Rothermel parameters  
    moisture_1hr: Union[float, np.ndarray] = 0.08,
    moisture_10hr: Union[float, np.ndarray] = 0.09,
    moisture_100hr: Union[float, np.ndarray] = 0.10,
    moisture_live: Union[float, np.ndarray] = 1.0,
    wind_speed: Union[float, np.ndarray] = 15.0,
    slope: Union[float, np.ndarray] = 0.0,
    # Shared
    non_fuel_codes: Optional[set] = None,
    fuel_lookup: Optional[Dict] = None,
    rothermel_fuel_models: Optional[Dict] = None,
    fbp_to_rothermel_map: Optional[Dict[int, int]] = None,
    midflame_wind_reduction: float = 0.4,
) -> np.ndarray:
    """
    Compute ROS grid using specified fire model.
    
    Parameters
    ----------
    fuel_grid : ndarray
        Grid of fuel type codes
    model : str
        Fire model: "fbp" or "rothermel"
    ... : various
        Model-specific parameters
    non_fuel_codes : set, optional
        Fuel codes to treat as non-fuel (ROS=0)
    fbp_to_rothermel_map : dict, optional
        Custom mapping from FBP codes to Rothermel models
    midflame_wind_reduction : float
        Factor to reduce 10m wind to midflame (Rothermel only)
        
    Returns
    -------
    ros_grid : ndarray
        Rate of spread at each cell (m/min)
    """
    ny, nx = fuel_grid.shape
    ros_grid = np.zeros((ny, nx), dtype=np.float32)
    
    if non_fuel_codes is None:
        non_fuel_codes = {0, 101, 102, 103, -1, 255}
    
    if model.lower() == "fbp":
        # FBP: compute per fuel type
        unique_fuels = np.unique(fuel_grid)
        
        for fuel_id in unique_fuels:
            if fuel_id in non_fuel_codes or np.isnan(fuel_id):
                continue
            
            ros = compute_ros_unified(
                fuel_code=int(fuel_id),
                model="fbp",
                isi=isi,
                bui=bui,
                fmc=fmc,
                curing=curing,
                fuel_lookup=fuel_lookup,
            )
            
            mask = fuel_grid == fuel_id
            ros_grid[mask] = ros
    
    elif model.lower() == "rothermel":
        try:
            from ignacio.jax_core.rothermel import FUEL_MODELS, compute_ros_rothermel_grid
        except ImportError:
            from .jax_core.rothermel import FUEL_MODELS, compute_ros_rothermel_grid
        import jax.numpy as jnp
        
        if rothermel_fuel_models is None:
            rothermel_fuel_models = FUEL_MODELS
        
        # Convert fuel codes if they're FBP codes
        # Check if any codes > 100 that look like Rothermel codes
        unique_codes = np.unique(fuel_grid[~np.isnan(fuel_grid)]).astype(int)
        max_fbp_code = 50  # FBP codes are typically < 50
        
        if all(c < max_fbp_code or c in non_fuel_codes for c in unique_codes):
            # Looks like FBP codes - convert to Rothermel
            logger.info("Converting FBP fuel codes to Rothermel fuel models")
            rothermel_fuel_grid = convert_fuel_grid_to_rothermel(
                fuel_grid, fbp_to_rothermel_map
            )
        else:
            # Assume already Rothermel codes
            rothermel_fuel_grid = fuel_grid.astype(np.int32)
        
        # Reduce wind speed to midflame height
        if isinstance(wind_speed, np.ndarray):
            midflame_wind = wind_speed * midflame_wind_reduction
        else:
            midflame_wind = wind_speed * midflame_wind_reduction
        
        # Broadcast scalars to arrays if needed
        def ensure_array(x, shape):
            if isinstance(x, (int, float)):
                return jnp.full(shape, x)
            return jnp.array(x)
        
        m1hr = ensure_array(moisture_1hr, (ny, nx))
        m10hr = ensure_array(moisture_10hr, (ny, nx))
        m100hr = ensure_array(moisture_100hr, (ny, nx))
        mlive = ensure_array(moisture_live, (ny, nx))
        ws = ensure_array(midflame_wind, (ny, nx))
        slp = ensure_array(slope, (ny, nx))
        
        # Use JAX vectorized computation
        ros_jax = compute_ros_rothermel_grid(
            jnp.array(rothermel_fuel_grid),
            m1hr, m10hr, m100hr, mlive,
            ws, slp,
            fuel_models=rothermel_fuel_models,
        )
        
        ros_grid = np.array(ros_jax)
    
    else:
        raise ValueError(f"Unknown fire model: {model}")
    
    return ros_grid


def compute_ros_components_unified(
    fuel_grid: np.ndarray,
    model: str,
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    wind_direction: float,
    # FBP parameters
    isi: float = 5.0,
    bui: float = 50.0,
    fmc: float = 100.0,
    curing: float = 85.0,
    backing_fraction: float = 0.2,
    length_to_breadth: float = 2.0,
    slope_factor_strength: float = 0.5,
    # Rothermel parameters
    moisture_1hr: Union[float, np.ndarray] = 0.08,
    wind_speed: float = 15.0,
    # Shared
    non_fuel_codes: Optional[set] = None,
    fuel_lookup: Optional[Dict] = None,
    fbp_to_rothermel_map: Optional[Dict[int, int]] = None,
    midflame_wind_reduction: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROS, BROS, FROS, and spread direction.
    
    Parameters
    ----------
    fuel_grid : ndarray
        Fuel type codes
    model : str
        "fbp" or "rothermel"
    slope_deg : ndarray
        Terrain slope (degrees)
    aspect_deg : ndarray
        Terrain aspect (degrees, downslope direction)
    wind_direction : float
        Wind direction (degrees, direction wind blows FROM)
    ... : various
        Model-specific parameters
        
    Returns
    -------
    ros : ndarray
        Head fire rate of spread (m/min)
    bros : ndarray
        Back fire rate of spread (m/min)
    fros : ndarray
        Flank fire rate of spread (m/min)
    raz : ndarray
        Spread direction (radians, direction fire spreads TO)
    """
    ny, nx = fuel_grid.shape
    
    if model.lower() == "fbp":
        # Compute base ROS
        ros = compute_ros_grid_unified(
            fuel_grid, "fbp",
            isi=isi, bui=bui, fmc=fmc, curing=curing,
            non_fuel_codes=non_fuel_codes,
            fuel_lookup=fuel_lookup,
        )
        
        # Apply slope correction
        try:
            from ignacio.spread import compute_slope_factor
        except ImportError:
            from .spread import compute_slope_factor
        slope_factor = compute_slope_factor(
            slope_deg, aspect_deg, wind_direction, slope_factor_strength
        )
        ros = ros * slope_factor
        
        # Back and flank fire
        bros = backing_fraction * ros
        fros = (ros + bros) / (2.0 * length_to_breadth)
        
        # Spread direction (fire spreads opposite to wind)
        raz = np.radians((wind_direction + 180.0) % 360.0)
        raz = np.full((ny, nx), raz, dtype=np.float32)
        
    elif model.lower() == "rothermel":
        try:
            from ignacio.jax_core.rothermel import compute_ros_components_rothermel
        except ImportError:
            from .jax_core.rothermel import compute_ros_components_rothermel
        import jax.numpy as jnp
        
        # Convert fuel codes if needed
        unique_codes = np.unique(fuel_grid[~np.isnan(fuel_grid)]).astype(int)
        if all(c < 50 or c in (non_fuel_codes or {0, 101, 102}) for c in unique_codes):
            rothermel_fuel_grid = convert_fuel_grid_to_rothermel(
                fuel_grid, fbp_to_rothermel_map
            )
        else:
            rothermel_fuel_grid = fuel_grid.astype(np.int32)
        
        # Reduce wind to midflame
        midflame_wind = wind_speed * midflame_wind_reduction
        
        ros_jax, bros_jax, fros_jax, raz_jax = compute_ros_components_rothermel(
            jnp.array(rothermel_fuel_grid),
            jnp.full((ny, nx), moisture_1hr),
            jnp.full((ny, nx), midflame_wind),
            jnp.full((ny, nx), wind_direction),
            jnp.array(slope_deg),
            jnp.array(aspect_deg),
            lb_ratio=length_to_breadth,
        )
        
        ros = np.array(ros_jax)
        bros = np.array(bros_jax)
        fros = np.array(fros_jax)
        raz = np.radians(np.array(raz_jax))  # Convert to radians
        
    else:
        raise ValueError(f"Unknown fire model: {model}")
    
    return ros, bros, fros, raz


# =============================================================================
# Moisture Conversion Utilities
# =============================================================================

def ffmc_to_moisture(ffmc: float) -> float:
    """
    Convert FFMC to fine fuel moisture content (fraction).
    
    Based on Van Wagner (1987) equations.
    
    Parameters
    ----------
    ffmc : float
        Fine Fuel Moisture Code (0-101)
        
    Returns
    -------
    moisture : float
        Moisture content as fraction (e.g., 0.08 for 8%)
    """
    # Invert FFMC equation: FFMC = 59.5 * (250 - m) / (147.2 + m)
    # Solving for m: m = 147.2 * (101 - FFMC) / (59.5 + FFMC)
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    return m / 100.0  # Convert from % to fraction


def dmc_to_moisture(dmc: float) -> float:
    """
    Convert DMC to 10-hour fuel moisture (fraction).
    
    Approximate relationship.
    
    Parameters
    ----------
    dmc : float
        Duff Moisture Code
        
    Returns
    -------
    moisture : float
        Approximate 10-hour moisture as fraction
    """
    # DMC relates to moisture in duff layer
    # Higher DMC = drier conditions
    # This is an approximation
    if dmc <= 0:
        return 0.25
    # Exponential decay relationship
    m = 20.0 * np.exp(-0.02 * dmc) + 5.0
    return m / 100.0


def dc_to_moisture(dc: float) -> float:
    """
    Convert DC to 100-hour fuel moisture (fraction).
    
    Approximate relationship.
    
    Parameters
    ----------
    dc : float
        Drought Code
        
    Returns
    -------
    moisture : float
        Approximate 100-hour moisture as fraction
    """
    # DC relates to deep organic layer moisture
    # Higher DC = drier conditions
    if dc <= 0:
        return 0.30
    # Exponential decay
    m = 25.0 * np.exp(-0.01 * dc) + 8.0
    return m / 100.0


def fwi_to_rothermel_moisture(
    ffmc: float = 88.0,
    dmc: float = 30.0,
    dc: float = 150.0,
) -> Tuple[float, float, float]:
    """
    Convert FWI indices to Rothermel fuel moisture values.
    
    Parameters
    ----------
    ffmc : float
        Fine Fuel Moisture Code
    dmc : float
        Duff Moisture Code
    dc : float
        Drought Code
        
    Returns
    -------
    m_1hr : float
        1-hour fuel moisture (fraction)
    m_10hr : float
        10-hour fuel moisture (fraction)
    m_100hr : float
        100-hour fuel moisture (fraction)
    """
    m_1hr = ffmc_to_moisture(ffmc)
    m_10hr = dmc_to_moisture(dmc)
    m_100hr = dc_to_moisture(dc)
    
    return m_1hr, m_10hr, m_100hr
