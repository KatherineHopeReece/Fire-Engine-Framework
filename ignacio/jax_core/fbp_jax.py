"""
Differentiable Fire Behaviour Prediction (FBP) System for JAX.

This module implements JAX-compatible versions of the Canadian FBP System
equations, enabling gradient-based optimization of fire behavior parameters.

Key calibratable parameters:
- wind_adj: Wind speed adjustment factor (multiplier)
- ffmc_adj: Fine Fuel Moisture Code adjustment (additive bias)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# Calibration Parameters
# =============================================================================


class FBPCalibrationParams(NamedTuple):
    """
    Calibratable parameters for FBP calculations.
    
    Attributes
    ----------
    wind_adj : float
        Wind speed adjustment factor (multiplicative). Default 1.0.
        Values > 1 increase effective wind speed and fire spread.
    ffmc_adj : float
        FFMC adjustment (additive). Default 0.0.
        Positive values decrease fuel moisture (drier conditions).
    ros_scale : float
        Overall ROS scaling factor. Default 1.0.
    backing_frac : float
        Back fire ROS as fraction of head fire. Default 0.2.
    lb_base : float
        Base length-to-breadth ratio. Default 1.0.
    lb_wind_coef : float
        Wind speed coefficient for L/B ratio. Default 0.0233.
    """
    wind_adj: float = 1.0
    ffmc_adj: float = 0.0
    ros_scale: float = 1.0
    backing_frac: float = 0.2
    lb_base: float = 1.0
    lb_wind_coef: float = 0.0233


def default_params() -> FBPCalibrationParams:
    """Return default calibration parameters."""
    return FBPCalibrationParams()


# =============================================================================
# Differentiable ISI Calculation
# =============================================================================


def calculate_isi_jax(
    ffmc: jnp.ndarray,
    wind_speed: jnp.ndarray,
    params: FBPCalibrationParams,
) -> jnp.ndarray:
    """
    Compute Initial Spread Index (ISI) with calibration adjustments.
    
    Differentiable JAX implementation of ISI calculation.
    
    Parameters
    ----------
    ffmc : jnp.ndarray
        Fine Fuel Moisture Code (0-101 scale).
    wind_speed : jnp.ndarray
        Wind speed in km/h.
    params : FBPCalibrationParams
        Calibration parameters.
        
    Returns
    -------
    jnp.ndarray
        Initial Spread Index values.
    """
    # Apply calibration adjustments
    ffmc_eff = jnp.clip(ffmc + params.ffmc_adj, 0.0, 101.0)
    ws_eff = jnp.maximum(wind_speed * params.wind_adj, 0.0)
    
    # Moisture content from FFMC (smooth approximation)
    # Original: m = 147.2 * (101 - FFMC) / (59.5 + FFMC)
    m = 147.2 * (101.0 - ffmc_eff) / (59.5 + ffmc_eff + 1e-8)
    
    # Fuel moisture function
    # f_F = 91.9 * exp(-0.1386 * m) * (1 + m^5.31 / 4.93e7)
    f_F = 91.9 * jnp.exp(-0.1386 * m) * (1.0 + jnp.power(m + 1e-8, 5.31) / 4.93e7)
    
    # Wind function
    f_W = jnp.exp(0.05039 * ws_eff)
    
    # ISI
    isi = 0.208 * f_F * f_W
    
    return jnp.maximum(isi, 0.0)


# =============================================================================
# Differentiable ROS Functions by Fuel Type
# =============================================================================


def _smooth_threshold(x: jnp.ndarray, threshold: float, sharpness: float = 10.0) -> jnp.ndarray:
    """Smooth step function approximation for threshold operations."""
    return jax.nn.sigmoid(sharpness * (x - threshold))


def _ros_conifer_generic(
    isi: jnp.ndarray,
    bui: jnp.ndarray,
    a: float,
    b: float,
    c: float,
    bui_thresh: float,
    q_coef: float,
    q_exp: float,
    be_num: float,
    be_denom: float,
) -> jnp.ndarray:
    """
    Generic conifer fuel type ROS calculation.
    
    RSI = a * (1 - exp(-b * ISI))^c
    BE = exp(be_num * ln(Q) / (be_denom + Q)) where Q = q_coef * (BUI - bui_thresh)^q_exp
    ROS = RSI * BE
    """
    # Rate of spread index
    rsi = a * jnp.power(1.0 - jnp.exp(-b * isi), c)
    
    # Buildup effect (smooth transition at threshold)
    bui_excess = jnp.maximum(bui - bui_thresh, 0.0)
    Q = q_coef * jnp.power(bui_excess + 1e-8, q_exp)
    
    # Smooth transition for BE activation
    be_active = _smooth_threshold(bui, bui_thresh)
    ln_Q = jnp.log(Q + 1e-8)
    be_raw = jnp.exp(be_num * ln_Q / (be_denom + Q + 1e-8))
    be = 1.0 + be_active * (jnp.maximum(be_raw, 1.0) - 1.0)
    
    return jnp.maximum(rsi * be, 0.0)


def ros_c1_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """C-1 Spruce-Lichen Woodland ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=90.0, b=0.0649, c=4.5,
        bui_thresh=60.0, q_coef=0.92, q_exp=0.91,
        be_num=50.0, be_denom=450.0,
    )


def ros_c2_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """C-2 Boreal Spruce ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=110.0, b=0.0282, c=1.5,
        bui_thresh=35.0, q_coef=0.8, q_exp=0.92,
        be_num=45.0, be_denom=300.0,
    )


def ros_c3_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """C-3 Mature Jack/Lodgepole Pine ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=110.0, b=0.0444, c=3.0,
        bui_thresh=40.0, q_coef=0.8, q_exp=0.90,
        be_num=45.0, be_denom=350.0,
    )


def ros_c4_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """C-4 Immature Jack/Lodgepole Pine ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=110.0, b=0.0293, c=1.5,
        bui_thresh=35.0, q_coef=0.8, q_exp=0.90,
        be_num=45.0, be_denom=320.0,
    )


def ros_c5_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """C-5 Red and White Pine ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=30.0, b=0.0697, c=4.0,
        bui_thresh=40.0, q_coef=0.8, q_exp=0.90,
        be_num=45.0, be_denom=350.0,
    )


def ros_c7_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """C-7 Ponderosa Pine / Douglas-fir ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=45.0, b=0.0305, c=2.0,
        bui_thresh=40.0, q_coef=0.8, q_exp=0.90,
        be_num=45.0, be_denom=350.0,
    )


def ros_d1_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """D-1 Leafless Aspen ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=30.0, b=0.0232, c=1.6,
        bui_thresh=32.0, q_coef=0.8, q_exp=0.90,
        be_num=45.0, be_denom=280.0,
    )


def ros_o1_jax(isi: jnp.ndarray, curing: jnp.ndarray) -> jnp.ndarray:
    """
    O-1a Standing grass ROS (m/min).
    
    Parameters
    ----------
    isi : jnp.ndarray
        Initial Spread Index.
    curing : jnp.ndarray
        Grass curing percentage (0-100).
    """
    # Curing factor
    cf = jnp.where(
        curing < 58.8,
        0.005 * (jnp.exp(0.061 * curing) - 1.0),
        0.176 + 0.02 * (curing - 58.8),
    )
    
    # Base ROS
    a, b, c = 190.0, 0.0310, 1.4
    rsi = a * jnp.power(1.0 - jnp.exp(-b * isi), c)
    
    return jnp.maximum(rsi * cf, 0.0)


def ros_o1b_jax(isi: jnp.ndarray, curing: jnp.ndarray) -> jnp.ndarray:
    """O-1b Matted grass ROS (m/min)."""
    # Curing factor (same as O-1a)
    cf = jnp.where(
        curing < 58.8,
        0.005 * (jnp.exp(0.061 * curing) - 1.0),
        0.176 + 0.02 * (curing - 58.8),
    )
    
    # Base ROS (different coefficients)
    a, b, c = 250.0, 0.0350, 1.7
    rsi = a * jnp.power(1.0 - jnp.exp(-b * isi), c)
    
    return jnp.maximum(rsi * cf, 0.0)


def ros_s1_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """S-1 Jack/Lodgepole Pine Slash ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=75.0, b=0.0297, c=1.3,
        bui_thresh=35.0, q_coef=0.8, q_exp=0.90,
        be_num=45.0, be_denom=300.0,
    )


def ros_s2_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """S-2 White Spruce/Balsam Slash ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=40.0, b=0.0438, c=1.7,
        bui_thresh=35.0, q_coef=0.8, q_exp=0.90,
        be_num=45.0, be_denom=300.0,
    )


def ros_s3_jax(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """S-3 Coastal Cedar/Hemlock/Douglas-fir Slash ROS (m/min)."""
    return _ros_conifer_generic(
        isi, bui,
        a=55.0, b=0.0829, c=3.2,
        bui_thresh=35.0, q_coef=0.8, q_exp=0.90,
        be_num=45.0, be_denom=300.0,
    )


# =============================================================================
# Fuel Type Dispatcher
# =============================================================================


# Fuel type ID to function mapping
FUEL_TYPE_REGISTRY = {
    1: ros_c1_jax,
    2: ros_c2_jax,
    3: ros_c3_jax,
    4: ros_c4_jax,
    5: ros_c5_jax,
    7: ros_c7_jax,
    11: ros_d1_jax,
    21: ros_s1_jax,
    22: ros_s2_jax,
    23: ros_s3_jax,
}

GRASS_FUEL_TYPES = {31, 32}  # O-1a, O-1b
NON_FUEL_TYPES = {0, 101, 102, 106, -9999}


def compute_ros_by_fuel_jax(
    fuel_id: int,
    isi: jnp.ndarray,
    bui: jnp.ndarray,
    curing: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Compute ROS for a single fuel type.
    
    Parameters
    ----------
    fuel_id : int
        Fuel type ID.
    isi : jnp.ndarray
        Initial Spread Index.
    bui : jnp.ndarray
        Buildup Index.
    curing : jnp.ndarray, optional
        Grass curing percentage (for O-1 fuels).
        
    Returns
    -------
    jnp.ndarray
        Rate of spread (m/min).
    """
    if fuel_id in NON_FUEL_TYPES:
        return jnp.zeros_like(isi)
    
    if fuel_id == 31:  # O-1a
        return ros_o1_jax(isi, curing if curing is not None else jnp.full_like(isi, 85.0))
    elif fuel_id == 32:  # O-1b
        return ros_o1b_jax(isi, curing if curing is not None else jnp.full_like(isi, 85.0))
    elif fuel_id in FUEL_TYPE_REGISTRY:
        return FUEL_TYPE_REGISTRY[fuel_id](isi, bui)
    else:
        # Default to C-2 for unknown fuel types
        return ros_c2_jax(isi, bui)


# =============================================================================
# Grid-Based ROS Computation
# =============================================================================


def compute_ros_grid_jax(
    fuel_grid: jnp.ndarray,
    isi: jnp.ndarray,
    bui: jnp.ndarray,
    params: FBPCalibrationParams,
    curing: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Compute ROS grid for multiple fuel types.
    
    This uses a loop over unique fuel types for now.
    For full differentiability through fuel type selection,
    consider using a continuous fuel type interpolation.
    
    Parameters
    ----------
    fuel_grid : jnp.ndarray
        2D grid of fuel type IDs.
    isi : jnp.ndarray
        ISI grid (same shape or broadcastable).
    bui : jnp.ndarray
        BUI grid (same shape or broadcastable).
    params : FBPCalibrationParams
        Calibration parameters.
    curing : jnp.ndarray, optional
        Grass curing grid.
        
    Returns
    -------
    jnp.ndarray
        ROS grid (m/min).
    """
    ros_grid = jnp.zeros_like(fuel_grid, dtype=jnp.float32)
    
    # Broadcast isi/bui to grid shape if scalar
    if isi.ndim == 0:
        isi = jnp.full_like(fuel_grid, isi, dtype=jnp.float32)
    if bui.ndim == 0:
        bui = jnp.full_like(fuel_grid, bui, dtype=jnp.float32)
    
    # Process each fuel type
    # Note: This loop breaks full differentiability through fuel type
    # For calibration, we assume fuel types are fixed
    unique_fuels = jnp.unique(fuel_grid)
    
    for fuel_id in unique_fuels:
        fuel_id_int = int(fuel_id)
        if fuel_id_int in NON_FUEL_TYPES:
            continue
        
        mask = fuel_grid == fuel_id
        ros_values = compute_ros_by_fuel_jax(fuel_id_int, isi, bui, curing)
        ros_grid = jnp.where(mask, ros_values, ros_grid)
    
    # Apply ROS scale factor from calibration
    return ros_grid * params.ros_scale


# =============================================================================
# ROS Components (Head, Flank, Back)
# =============================================================================


def compute_lb_ratio_jax(
    wind_speed: jnp.ndarray,
    params: FBPCalibrationParams,
) -> jnp.ndarray:
    """
    Compute length-to-breadth ratio from wind speed.
    
    L/B = 1.0 + 8.729 * (1 - exp(-0.030 * WS))^2.155
    Simplified: L/B = lb_base + lb_wind_coef * WS
    
    Parameters
    ----------
    wind_speed : jnp.ndarray
        Wind speed in km/h.
    params : FBPCalibrationParams
        Calibration parameters.
        
    Returns
    -------
    jnp.ndarray
        Length-to-breadth ratio.
    """
    # FBP formula (Alexander 1985)
    ws_eff = wind_speed * params.wind_adj
    lb = 1.0 + 8.729 * jnp.power(1.0 - jnp.exp(-0.030 * ws_eff), 2.155)
    return jnp.maximum(lb, 1.0)


def compute_ros_components_jax(
    ros_head: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_direction: jnp.ndarray,
    params: FBPCalibrationParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute all ROS components from head fire ROS.
    
    Parameters
    ----------
    ros_head : jnp.ndarray
        Head fire rate of spread (m/min).
    wind_speed : jnp.ndarray
        Wind speed in km/h.
    wind_direction : jnp.ndarray
        Wind direction in degrees (direction FROM).
    params : FBPCalibrationParams
        Calibration parameters.
        
    Returns
    -------
    ros : jnp.ndarray
        Head fire ROS.
    bros : jnp.ndarray
        Back fire ROS.
    fros : jnp.ndarray
        Flank fire ROS.
    raz : jnp.ndarray
        Rate of spread azimuth in radians (direction TO).
    """
    # Length-to-breadth ratio
    lb = compute_lb_ratio_jax(wind_speed, params)
    
    # Back fire ROS
    bros = params.backing_frac * ros_head
    
    # Flank fire ROS (from ellipse geometry)
    fros = (ros_head + bros) / (2.0 * lb)
    
    # RAZ is direction fire spreads TO (opposite of wind FROM)
    # Convert to radians
    raz_deg = (wind_direction + 180.0) % 360.0
    raz = jnp.deg2rad(raz_deg)
    
    return ros_head, bros, fros, raz


# =============================================================================
# Complete FBP Pipeline
# =============================================================================


def fbp_pipeline_jax(
    ffmc: jnp.ndarray,
    bui: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_direction: jnp.ndarray,
    fuel_grid: jnp.ndarray,
    params: FBPCalibrationParams,
    curing: jnp.ndarray = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Complete FBP calculation pipeline.
    
    FFMC, Wind -> ISI -> ROS (by fuel) -> ROS components
    
    Parameters
    ----------
    ffmc : jnp.ndarray
        Fine Fuel Moisture Code.
    bui : jnp.ndarray
        Buildup Index.
    wind_speed : jnp.ndarray
        Wind speed in km/h.
    wind_direction : jnp.ndarray
        Wind direction in degrees (FROM).
    fuel_grid : jnp.ndarray
        Fuel type grid.
    params : FBPCalibrationParams
        Calibration parameters.
    curing : jnp.ndarray, optional
        Grass curing percentage.
        
    Returns
    -------
    ros : jnp.ndarray
        Head fire ROS (m/min).
    bros : jnp.ndarray
        Back fire ROS (m/min).
    fros : jnp.ndarray
        Flank fire ROS (m/min).
    raz : jnp.ndarray
        Spread azimuth (radians).
    """
    # Calculate ISI with calibration adjustments
    isi = calculate_isi_jax(ffmc, wind_speed, params)
    
    # Compute ROS by fuel type
    ros_head = compute_ros_grid_jax(fuel_grid, isi, bui, params, curing)
    
    # Compute all components
    ros, bros, fros, raz = compute_ros_components_jax(
        ros_head, wind_speed, wind_direction, params
    )
    
    return ros, bros, fros, raz
