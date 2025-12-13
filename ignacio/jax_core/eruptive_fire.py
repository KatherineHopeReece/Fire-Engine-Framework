"""
Eruptive Fire Behavior Module - The "Canyon Effect"

Implements the physics of fire blow-up on steep terrain, where standard
fire spread models catastrophically underpredict because they assume
steady-state behavior.

Physical Background
-------------------
Standard FBP/Rothermel models assume fire spreads at a steady rate determined
by fuel, weather, and slope. In reality, on steep slopes (>25°) or in canyons,
the flame can "attach" to the slope surface (Coandă effect). When this happens:

1. Convective heating becomes dominant over radiant heating
2. The fire preheats fuel much faster than steady-state models predict
3. Rate of spread can accelerate exponentially (not linearly with slope)
4. This is the physics behind tragedies like Yarnell Hill (2013) and
   Storm King Mountain (1994)

Implemented Criteria
--------------------
1. Viegas (2004) eruptive fire criterion
2. Dold & Zinoviev (2009) flame attachment instability
3. Canyon geometry amplification factor

References
----------
- Viegas, D.X. (2004). A mathematical model for forest fires blowup.
  Combustion Science and Technology, 177(1), 27-51.
- Dold, J.W., Zinoviev, A. (2009). Fire eruption through intensity and
  spread rate interaction mediated by flow attachment. Combustion Theory
  and Modelling, 13(5), 763-793.
- Sharples, J.J. (2008). Review of formal methodologies for wind-slope
  correction of wildfire rate of spread. Int. J. Wildland Fire, 17, 179-193.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from functools import partial


# =============================================================================
# Data Structures
# =============================================================================

class EruptiveParams(NamedTuple):
    """Parameters for eruptive fire behavior model."""
    
    # Viegas criterion thresholds
    critical_slope: float = 25.0       # Degrees - onset of attachment risk
    blowup_slope: float = 35.0         # Degrees - high blowup probability
    
    # Flame geometry
    flame_tilt_base: float = 45.0      # Base flame tilt angle (degrees)
    flame_tilt_wind_coef: float = 2.0  # Wind effect on flame tilt
    
    # Acceleration parameters
    eruptive_multiplier: float = 5.0   # Max ROS multiplier during blowup
    acceleration_time: float = 5.0     # Minutes to reach peak acceleration
    decay_distance: float = 200.0      # Meters - eruptive pulse decay
    
    # Canyon geometry
    canyon_width_threshold: float = 100.0  # Meters - narrow canyon threshold
    canyon_amplification: float = 2.0      # Additional multiplier in canyons
    
    # Safety margins
    attachment_probability_threshold: float = 0.5  # Warn above this


class EruptiveState(NamedTuple):
    """State variables for eruptive fire tracking."""
    
    # Per-cell state
    attachment_probability: jnp.ndarray  # Probability of flame attachment [0-1]
    eruptive_potential: jnp.ndarray      # Eruptive potential index
    time_since_ignition: jnp.ndarray     # Time fire has been active at cell
    ros_history: jnp.ndarray             # Recent ROS for acceleration detection
    
    # Flags
    blowup_active: jnp.ndarray           # Boolean - is blowup occurring?
    canyon_cells: jnp.ndarray            # Boolean - is cell in canyon?


class EruptiveResult(NamedTuple):
    """Results from eruptive fire calculation."""
    
    ros_multiplier: jnp.ndarray          # Multiplier to apply to base ROS
    attachment_probability: jnp.ndarray  # Updated attachment probability
    eruptive_potential: jnp.ndarray      # Eruptive danger index
    warning_zones: jnp.ndarray           # High-danger zones for warnings


# =============================================================================
# Core Physics Functions
# =============================================================================

@jax.jit
def compute_flame_tilt_angle(
    wind_speed: jnp.ndarray,
    slope_degrees: jnp.ndarray,
    params: EruptiveParams,
) -> jnp.ndarray:
    """
    Compute flame tilt angle based on wind and slope.
    
    The flame tilts toward the slope due to:
    1. Wind pushing the flame
    2. Buoyancy-driven indraft up the slope
    3. Slope-induced flow attachment
    
    Parameters
    ----------
    wind_speed : array
        Wind speed in m/s
    slope_degrees : array
        Terrain slope in degrees
    params : EruptiveParams
        Model parameters
        
    Returns
    -------
    array
        Flame tilt angle in degrees from vertical
    """
    # Base tilt from wind (empirical relationship)
    # Higher wind = more horizontal flame
    wind_tilt = params.flame_tilt_base * (1.0 - jnp.exp(-wind_speed / 10.0))
    
    # Slope-induced tilt (flame tends to align with slope)
    slope_tilt = slope_degrees * 0.5
    
    # Combined tilt (capped at 85 degrees from vertical)
    total_tilt = jnp.minimum(wind_tilt + slope_tilt, 85.0)
    
    return total_tilt


@jax.jit
def compute_attachment_probability(
    slope_degrees: jnp.ndarray,
    flame_tilt: jnp.ndarray,
    wind_speed: jnp.ndarray,
    ros: jnp.ndarray,
    params: EruptiveParams,
) -> jnp.ndarray:
    """
    Compute probability of flame attachment to slope.
    
    Flame attachment occurs when the flame angle becomes parallel to or
    less than the slope angle, causing convective heating to dominate.
    
    Based on Viegas (2004) criterion:
    - Attachment likely when tan(slope) > tan(90 - flame_tilt)
    
    Parameters
    ----------
    slope_degrees : array
        Terrain slope in degrees
    flame_tilt : array
        Flame tilt angle from vertical (degrees)
    wind_speed : array
        Wind speed in m/s
    ros : array
        Current rate of spread (m/min)
    params : EruptiveParams
        Model parameters
        
    Returns
    -------
    array
        Probability of flame attachment [0-1]
    """
    # Critical angle: flame perpendicular to slope surface
    # When flame_tilt > (90 - slope), attachment can occur
    critical_difference = flame_tilt - (90.0 - slope_degrees)
    
    # Sigmoid probability based on how far past critical we are
    # P = 1 / (1 + exp(-k * (angle - threshold)))
    k = 0.2  # Steepness of transition
    base_probability = jax.nn.sigmoid(k * critical_difference)
    
    # Enhance probability with slope steepness
    slope_factor = jnp.where(
        slope_degrees > params.critical_slope,
        1.0 + (slope_degrees - params.critical_slope) / 20.0,
        1.0
    )
    
    # ROS feedback: faster fires more likely to attach
    ros_factor = jnp.minimum(1.0 + ros / 50.0, 2.0)
    
    # Combined probability (capped at 1.0)
    probability = jnp.minimum(base_probability * slope_factor * ros_factor, 1.0)
    
    # Zero probability on flat terrain
    probability = jnp.where(slope_degrees < 5.0, 0.0, probability)
    
    return probability


@jax.jit
def compute_eruptive_potential(
    slope_degrees: jnp.ndarray,
    aspect_degrees: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_direction: jnp.ndarray,
    attachment_prob: jnp.ndarray,
    params: EruptiveParams,
) -> jnp.ndarray:
    """
    Compute eruptive fire potential index.
    
    This combines multiple factors that increase blowup risk:
    1. Slope steepness (Viegas criterion)
    2. Wind-slope alignment (upslope wind is worst)
    3. Attachment probability
    
    Parameters
    ----------
    slope_degrees : array
        Terrain slope
    aspect_degrees : array
        Terrain aspect (downslope direction)
    wind_speed : array
        Wind speed in m/s
    wind_direction : array
        Wind direction (from) in degrees
    attachment_prob : array
        Flame attachment probability
    params : EruptiveParams
        Model parameters
        
    Returns
    -------
    array
        Eruptive potential index [0-1]
    """
    # Slope factor: exponential increase above critical slope
    slope_factor = jnp.where(
        slope_degrees > params.critical_slope,
        1.0 - jnp.exp(-(slope_degrees - params.critical_slope) / 15.0),
        0.0
    )
    
    # Wind-slope alignment
    # Upslope wind (wind from opposite direction of aspect) is most dangerous
    # Aspect is downslope direction, wind_direction is "from" direction
    upslope_direction = (aspect_degrees + 180.0) % 360.0
    wind_slope_diff = jnp.abs(wind_direction - upslope_direction)
    wind_slope_diff = jnp.minimum(wind_slope_diff, 360.0 - wind_slope_diff)
    
    # Aligned if difference < 45 degrees
    alignment_factor = jnp.maximum(0.0, 1.0 - wind_slope_diff / 90.0)
    
    # Wind speed contribution (stronger wind = higher potential if aligned)
    wind_factor = jnp.minimum(wind_speed / 20.0, 1.0)
    
    # Combine factors
    eruptive_potential = (
        0.4 * slope_factor +
        0.3 * alignment_factor * wind_factor +
        0.3 * attachment_prob
    )
    
    return jnp.clip(eruptive_potential, 0.0, 1.0)


def detect_canyon_geometry(
    dem: jnp.ndarray,
    dx: float,
    params: EruptiveParams,
) -> jnp.ndarray:
    """
    Detect canyon/chimney terrain configurations.
    
    Canyons amplify eruptive behavior because:
    1. Channeled airflow accelerates up canyon
    2. Radiant heat reflects off opposing walls
    3. Escape routes are limited
    
    Parameters
    ----------
    dem : array
        Digital elevation model
    dx : float
        Grid spacing in meters
    params : EruptiveParams
        Model parameters
        
    Returns
    -------
    array
        Boolean mask of canyon cells
    """
    # Use fixed kernel size instead of dynamic
    kernel_size = 5  # Fixed 5x5 kernel for stability
    
    # Use morphological operations to find valleys
    # A canyon cell is lower than surroundings on two opposing sides
    # but similar elevation along the canyon axis
    
    # Compute elevation percentile in local window
    from jax.scipy.signal import convolve2d
    
    kernel = jnp.ones((kernel_size, kernel_size), dtype=dem.dtype) / (kernel_size * kernel_size)
    
    local_mean = convolve2d(dem, kernel, mode='same')
    local_diff = dem - local_mean
    
    # Canyon cells are below local mean (in valley)
    in_valley = local_diff < -10.0  # 10m below local mean
    
    # Check for steep walls on opposing sides
    # Compute directional gradients
    dy_kernel = jnp.array([[-1], [0], [1]], dtype=dem.dtype) / (2 * dx)
    dx_kernel = jnp.array([[-1, 0, 1]], dtype=dem.dtype) / (2 * dx)
    
    grad_y = convolve2d(dem, dy_kernel, mode='same')
    grad_x = convolve2d(dem, dx_kernel, mode='same')
    
    # High gradient variance indicates canyon walls
    grad_mag = jnp.sqrt(grad_x**2 + grad_y**2)
    
    grad_mag_kernel = jnp.ones((3, 3), dtype=dem.dtype) / 9.0
    local_grad_var = convolve2d(grad_mag**2, grad_mag_kernel, mode='same') - \
                     convolve2d(grad_mag, grad_mag_kernel, mode='same')**2
    
    steep_sides = local_grad_var > 0.01  # High gradient variance
    
    # Canyon = valley + steep sides
    canyon_mask = in_valley & steep_sides
    
    return canyon_mask


@jax.jit
def compute_ros_multiplier(
    base_ros: jnp.ndarray,
    attachment_prob: jnp.ndarray,
    eruptive_potential: jnp.ndarray,
    canyon_mask: jnp.ndarray,
    time_burning: jnp.ndarray,
    params: EruptiveParams,
) -> jnp.ndarray:
    """
    Compute the ROS multiplier for eruptive fire behavior.
    
    When eruptive conditions are met, ROS can increase dramatically.
    The multiplier ramps up over time as the fire accelerates.
    
    Parameters
    ----------
    base_ros : array
        Base rate of spread from standard model
    attachment_prob : array
        Flame attachment probability
    eruptive_potential : array
        Eruptive potential index
    canyon_mask : array
        Boolean mask of canyon cells
    time_burning : array
        Time since fire reached cell (minutes)
    params : EruptiveParams
        Model parameters
        
    Returns
    -------
    array
        Multiplier to apply to base ROS
    """
    # Eruptive multiplier based on attachment probability and potential
    # Uses a threshold behavior: below threshold = no effect
    eruptive_trigger = jnp.maximum(
        attachment_prob - params.attachment_probability_threshold,
        0.0
    ) / (1.0 - params.attachment_probability_threshold)
    
    # Ramp up multiplier over time (fire accelerates)
    time_factor = 1.0 - jnp.exp(-time_burning / params.acceleration_time)
    
    # Base multiplier from eruptive behavior
    base_multiplier = 1.0 + (params.eruptive_multiplier - 1.0) * \
                      eruptive_trigger * eruptive_potential * time_factor
    
    # Canyon amplification
    canyon_amplification = jnp.where(
        canyon_mask,
        params.canyon_amplification,
        1.0
    )
    
    # Combined multiplier
    total_multiplier = base_multiplier * canyon_amplification
    
    # Cap at reasonable maximum
    max_multiplier = params.eruptive_multiplier * params.canyon_amplification
    total_multiplier = jnp.minimum(total_multiplier, max_multiplier)
    
    return total_multiplier


# =============================================================================
# Main Interface Functions
# =============================================================================

def initialize_eruptive_state(
    shape: tuple[int, int],
    dem: jnp.ndarray,
    dx: float,
    params: EruptiveParams = None,
) -> EruptiveState:
    """
    Initialize eruptive fire state for a domain.
    
    Parameters
    ----------
    shape : tuple
        Grid shape (ny, nx)
    dem : array
        Digital elevation model
    dx : float
        Grid spacing in meters
    params : EruptiveParams, optional
        Model parameters
        
    Returns
    -------
    EruptiveState
        Initialized state
    """
    if params is None:
        params = EruptiveParams()
    
    ny, nx = shape
    
    # Detect canyon geometry (static, computed once)
    canyon_mask = detect_canyon_geometry(dem, dx, params)
    
    return EruptiveState(
        attachment_probability=jnp.zeros((ny, nx)),
        eruptive_potential=jnp.zeros((ny, nx)),
        time_since_ignition=jnp.zeros((ny, nx)),
        ros_history=jnp.zeros((ny, nx, 5)),  # Last 5 timesteps
        blowup_active=jnp.zeros((ny, nx), dtype=bool),
        canyon_cells=canyon_mask,
    )


@partial(jax.jit, static_argnums=(7,))
def update_eruptive_state(
    state: EruptiveState,
    slope_degrees: jnp.ndarray,
    aspect_degrees: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_direction: jnp.ndarray,
    ros: jnp.ndarray,
    fire_mask: jnp.ndarray,
    params: EruptiveParams,
    dt: float = 1.0,
) -> EruptiveState:
    """
    Update eruptive fire state for one timestep.
    
    Parameters
    ----------
    state : EruptiveState
        Current state
    slope_degrees : array
        Terrain slope
    aspect_degrees : array
        Terrain aspect
    wind_speed : array
        Wind speed (m/s)
    wind_direction : array
        Wind direction (degrees)
    ros : array
        Current rate of spread (m/min)
    fire_mask : array
        Boolean mask of burning cells
    params : EruptiveParams
        Model parameters
    dt : float
        Timestep in minutes
        
    Returns
    -------
    EruptiveState
        Updated state
    """
    # Compute flame tilt
    flame_tilt = compute_flame_tilt_angle(wind_speed, slope_degrees, params)
    
    # Compute attachment probability
    attachment_prob = compute_attachment_probability(
        slope_degrees, flame_tilt, wind_speed, ros, params
    )
    
    # Update only in burning areas
    attachment_prob = jnp.where(fire_mask, attachment_prob, state.attachment_probability)
    
    # Compute eruptive potential
    eruptive_potential = compute_eruptive_potential(
        slope_degrees, aspect_degrees, wind_speed, wind_direction,
        attachment_prob, params
    )
    
    # Update time since ignition
    time_burning = jnp.where(
        fire_mask,
        state.time_since_ignition + dt,
        0.0
    )
    
    # Update ROS history (shift and add new)
    ros_history = jnp.roll(state.ros_history, 1, axis=-1)
    ros_history = ros_history.at[:, :, 0].set(ros)
    
    # Detect blowup (rapid acceleration)
    ros_acceleration = ros - state.ros_history[:, :, 0]
    blowup_active = (ros_acceleration > 5.0) & (attachment_prob > 0.5)  # >5 m/min/step
    
    return EruptiveState(
        attachment_probability=attachment_prob,
        eruptive_potential=eruptive_potential,
        time_since_ignition=time_burning,
        ros_history=ros_history,
        blowup_active=blowup_active,
        canyon_cells=state.canyon_cells,
    )


@partial(jax.jit, static_argnums=(6,))
def compute_eruptive_ros(
    base_ros: jnp.ndarray,
    state: EruptiveState,
    slope_degrees: jnp.ndarray,
    aspect_degrees: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_direction: jnp.ndarray,
    params: EruptiveParams,
) -> EruptiveResult:
    """
    Compute eruptive-adjusted rate of spread.
    
    Parameters
    ----------
    base_ros : array
        Base ROS from standard FBP/Rothermel model (m/min)
    state : EruptiveState
        Current eruptive state
    slope_degrees : array
        Terrain slope
    aspect_degrees : array
        Terrain aspect
    wind_speed : array
        Wind speed (m/s)
    wind_direction : array
        Wind direction (degrees)
    params : EruptiveParams
        Model parameters
        
    Returns
    -------
    EruptiveResult
        Results including ROS multiplier and warning zones
    """
    # Compute multiplier
    multiplier = compute_ros_multiplier(
        base_ros,
        state.attachment_probability,
        state.eruptive_potential,
        state.canyon_cells,
        state.time_since_ignition,
        params,
    )
    
    # Identify warning zones
    warning_zones = (
        (state.eruptive_potential > 0.5) |
        (state.attachment_probability > params.attachment_probability_threshold) |
        state.blowup_active
    )
    
    return EruptiveResult(
        ros_multiplier=multiplier,
        attachment_probability=state.attachment_probability,
        eruptive_potential=state.eruptive_potential,
        warning_zones=warning_zones,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def classify_eruptive_danger(
    eruptive_potential: jnp.ndarray,
    attachment_prob: jnp.ndarray,
) -> jnp.ndarray:
    """
    Classify eruptive fire danger into categories.
    
    Parameters
    ----------
    eruptive_potential : array
        Eruptive potential index
    attachment_prob : array
        Flame attachment probability
        
    Returns
    -------
    array
        Danger class: 0=Low, 1=Moderate, 2=High, 3=Extreme
    """
    combined = 0.5 * eruptive_potential + 0.5 * attachment_prob
    
    danger = jnp.zeros_like(combined, dtype=jnp.int32)
    danger = jnp.where(combined > 0.25, 1, danger)  # Moderate
    danger = jnp.where(combined > 0.50, 2, danger)  # High
    danger = jnp.where(combined > 0.75, 3, danger)  # Extreme
    
    return danger


def summarize_eruptive_risk(
    state: EruptiveState,
    fire_mask: jnp.ndarray,
) -> dict:
    """
    Generate summary statistics of eruptive fire risk.
    
    Parameters
    ----------
    state : EruptiveState
        Current eruptive state
    fire_mask : array
        Boolean mask of fire area
        
    Returns
    -------
    dict
        Summary statistics
    """
    # Only consider fire-adjacent cells (potential spread)
    import jax.scipy.ndimage as ndimage
    
    # Dilate fire mask to get spread zone
    # struct = jnp.ones((3, 3), dtype=bool)
    # spread_zone = ndimage.binary_dilation(fire_mask, struct)
    
    return {
        "mean_attachment_prob": float(jnp.mean(state.attachment_probability)),
        "max_attachment_prob": float(jnp.max(state.attachment_probability)),
        "mean_eruptive_potential": float(jnp.mean(state.eruptive_potential)),
        "max_eruptive_potential": float(jnp.max(state.eruptive_potential)),
        "canyon_cell_fraction": float(jnp.mean(state.canyon_cells.astype(jnp.float32))),
        "blowup_active_cells": int(jnp.sum(state.blowup_active)),
        "high_risk_cells": int(jnp.sum(state.eruptive_potential > 0.5)),
    }
