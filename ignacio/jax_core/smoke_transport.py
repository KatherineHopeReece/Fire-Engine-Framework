"""
Smoke Transport Module - Passive Tracer Advection-Diffusion

Implements simplified smoke plume transport for operational awareness.
While the fire module calculates where fire is, users also need to know
where smoke goes for evacuation planning and air quality warnings.

Physical Background
-------------------
Smoke transport is governed by the advection-diffusion equation:
    ∂C/∂t + u·∇C = D∇²C + S - λC

Where:
- C = smoke concentration (mass/volume or PM2.5)
- u = wind velocity field (from wind solver)
- D = turbulent diffusion coefficient
- S = source term (fire intensity)
- λ = deposition/decay rate

This module provides:
1. Real-time smoke concentration maps
2. Visibility estimates
3. PM2.5 concentration (air quality index)
4. Plume centerline and dispersion width

Limitations
-----------
This is a 2D ground-level model. For accurate 3D plume rise and
long-range transport, use dedicated atmospheric dispersion models
(HYSPLIT, BlueSky, etc.). This module is for near-field tactical use.

References
----------
- Briggs, G.A. (1975). Plume rise predictions. EPA report.
- Achtemeier, G.L. (2006). Measurements of moisture in smoldering smoke.
- Larkin, N.K. et al. (2009). The BlueSky smoke modeling framework.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from functools import partial


# =============================================================================
# Data Structures
# =============================================================================

class SmokeParams(NamedTuple):
    """Parameters for smoke transport model."""
    
    # Diffusion parameters
    diffusion_coef: float = 50.0        # m²/s - turbulent diffusion
    wind_diffusion_scaling: float = 0.1 # Diffusion scales with wind
    
    # Source term parameters
    emission_factor: float = 1.0        # kg PM2.5 per kg fuel consumed
    fuel_consumption_rate: float = 0.5  # kg/m²/min for active fire
    plume_rise_factor: float = 0.5      # Fraction staying near ground
    
    # Decay/deposition
    deposition_rate: float = 0.001      # 1/s - settling
    chemical_decay: float = 0.0001      # 1/s - chemical transformation
    rain_washout_rate: float = 0.01     # 1/s when precipitating
    
    # Visibility calculation
    visibility_extinction_coef: float = 4.0  # m²/g for PM2.5
    background_visibility: float = 100000.0  # meters - clean air
    
    # PM2.5 thresholds (μg/m³)
    aqi_good: float = 12.0
    aqi_moderate: float = 35.4
    aqi_unhealthy_sensitive: float = 55.4
    aqi_unhealthy: float = 150.4
    aqi_very_unhealthy: float = 250.4
    aqi_hazardous: float = 500.4
    
    # Numerical parameters
    cfl_safety: float = 0.8             # CFL condition multiplier
    min_concentration: float = 1e-10    # Numerical floor


class SmokeState(NamedTuple):
    """State variables for smoke transport."""
    
    concentration: jnp.ndarray          # Smoke concentration (g/m³)
    cumulative_exposure: jnp.ndarray    # Time-integrated exposure
    max_concentration: jnp.ndarray      # Maximum seen at each cell
    plume_age: jnp.ndarray              # Time since smoke arrived


class SmokeResult(NamedTuple):
    """Results from smoke transport calculation."""
    
    concentration: jnp.ndarray          # Current concentration (g/m³)
    pm25: jnp.ndarray                   # PM2.5 in μg/m³
    visibility: jnp.ndarray             # Visibility in meters
    aqi_category: jnp.ndarray           # AQI category (0-5)
    plume_mask: jnp.ndarray             # Boolean - significant smoke


# =============================================================================
# Core Physics Functions  
# =============================================================================

@jax.jit
def compute_emission_source(
    fire_ros: jnp.ndarray,
    fire_intensity: jnp.ndarray,
    fire_mask: jnp.ndarray,
    dx: float,
    params: SmokeParams,
) -> jnp.ndarray:
    """
    Compute smoke emission source term from fire activity.
    
    Parameters
    ----------
    fire_ros : array
        Rate of spread (m/min)
    fire_intensity : array
        Fire intensity (kW/m)
    fire_mask : array
        Boolean mask of active fire
    dx : float
        Grid spacing (m)
    params : SmokeParams
        Model parameters
        
    Returns
    -------
    array
        Source term (g/m³/s)
    """
    # Estimate fuel consumption from fire intensity
    # I = H * w * R where H=18000 kJ/kg, so w*R = I/H
    heat_content = 18000.0  # kJ/kg
    consumption_rate = fire_intensity / heat_content  # kg/m/s
    
    # Convert to areal rate (approximate fire width from ROS)
    fire_width = jnp.maximum(fire_ros / 60.0 * 10.0, 1.0)  # ~10 sec of spread
    areal_consumption = consumption_rate / fire_width  # kg/m²/s
    
    # Apply emission factor
    emission_rate = areal_consumption * params.emission_factor  # kg PM/m²/s
    
    # Convert to volumetric (assume well-mixed in lower atmosphere, ~100m)
    mixing_height = 100.0  # meters
    source = emission_rate * 1000.0 / mixing_height  # g/m³/s
    
    # Apply plume rise factor (some smoke goes up, not staying near ground)
    source = source * params.plume_rise_factor
    
    # Only emit from active fire cells
    source = jnp.where(fire_mask, source, 0.0)
    
    return source


@jax.jit
def compute_effective_diffusion(
    wind_speed: jnp.ndarray,
    params: SmokeParams,
) -> jnp.ndarray:
    """
    Compute effective turbulent diffusion coefficient.
    
    Diffusion increases with wind speed due to increased turbulence.
    
    Parameters
    ----------
    wind_speed : array
        Wind speed (m/s)
    params : SmokeParams
        Model parameters
        
    Returns
    -------
    array
        Effective diffusion coefficient (m²/s)
    """
    D_base = params.diffusion_coef
    D_wind = params.wind_diffusion_scaling * wind_speed * 100.0  # Scale factor
    
    return D_base + D_wind


@jax.jit  
def advection_step(
    C: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    dx: float,
    dt: float,
) -> jnp.ndarray:
    """
    Perform advection using upwind scheme.
    
    Uses first-order upwind differencing for stability.
    
    Parameters
    ----------
    C : array
        Concentration field
    u : array
        Wind velocity in x direction (m/s)
    v : array
        Wind velocity in y direction (m/s)
    dx : float
        Grid spacing (m)
    dt : float
        Time step (s)
        
    Returns
    -------
    array
        Concentration after advection
    """
    # Upwind scheme: use upstream value based on wind direction
    
    # X-direction
    C_left = jnp.roll(C, 1, axis=1)
    C_right = jnp.roll(C, -1, axis=1)
    
    dCdx_forward = (C_right - C) / dx
    dCdx_backward = (C - C_left) / dx
    
    # Use upstream difference
    dCdx = jnp.where(u > 0, dCdx_backward, dCdx_forward)
    
    # Y-direction  
    C_down = jnp.roll(C, 1, axis=0)
    C_up = jnp.roll(C, -1, axis=0)
    
    dCdy_forward = (C_up - C) / dx
    dCdy_backward = (C - C_down) / dx
    
    dCdy = jnp.where(v > 0, dCdy_backward, dCdy_forward)
    
    # Update concentration
    C_new = C - dt * (u * dCdx + v * dCdy)
    
    # Ensure non-negative and finite
    C_new = jnp.maximum(C_new, 0.0)
    C_new = jnp.where(jnp.isfinite(C_new), C_new, 0.0)
    
    return C_new


@jax.jit
def diffusion_step(
    C: jnp.ndarray,
    D: jnp.ndarray,
    dx: float,
    dt: float,
) -> jnp.ndarray:
    """
    Perform diffusion using explicit finite differences.
    
    Parameters
    ----------
    C : array
        Concentration field
    D : array
        Diffusion coefficient field (m²/s)
    dx : float
        Grid spacing (m)
    dt : float
        Time step (s)
        
    Returns
    -------
    array
        Concentration after diffusion
    """
    # Laplacian using 5-point stencil
    laplacian_kernel = jnp.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=C.dtype) / (dx * dx)
    
    laplacian = convolve2d(C, laplacian_kernel, mode='same')
    
    # Update with diffusion
    C_new = C + dt * D * laplacian
    
    # Ensure non-negative and finite
    C_new = jnp.maximum(C_new, 0.0)
    C_new = jnp.where(jnp.isfinite(C_new), C_new, 0.0)
    
    return C_new


@jax.jit
def apply_decay(
    C: jnp.ndarray,
    precipitation_rate: float,
    dt: float,
    params: SmokeParams,
) -> jnp.ndarray:
    """
    Apply decay and deposition to concentration.
    
    Parameters
    ----------
    C : array
        Concentration field
    precipitation_rate : float
        Precipitation rate (mm/hr)
    dt : float
        Time step (s)
    params : SmokeParams
        Model parameters
        
    Returns
    -------
    array
        Concentration after decay
    """
    # Combined decay rate
    decay_rate = params.deposition_rate + params.chemical_decay
    
    # Add rain washout if precipitating (JAX-compatible)
    rain_washout = params.rain_washout_rate * jnp.maximum(precipitation_rate, 0.0) / 10.0
    decay_rate = decay_rate + rain_washout
    
    # Exponential decay
    C_new = C * jnp.exp(-decay_rate * dt)
    
    return C_new


# =============================================================================
# Visualization Functions
# =============================================================================

@jax.jit
def concentration_to_pm25(
    concentration: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convert smoke concentration to PM2.5.
    
    Parameters
    ----------
    concentration : array
        Smoke concentration (g/m³)
        
    Returns
    -------
    array
        PM2.5 concentration (μg/m³)
    """
    # Assume smoke is ~90% PM2.5 by mass
    pm25_fraction = 0.9
    
    # Convert g/m³ to μg/m³
    pm25 = concentration * pm25_fraction * 1e6
    
    return pm25


@jax.jit
def compute_visibility(
    pm25: jnp.ndarray,
    params: SmokeParams,
) -> jnp.ndarray:
    """
    Compute visibility from PM2.5 concentration.
    
    Uses Beer-Lambert law: V = 3.912 / (β_ext * C + β_background)
    
    Parameters
    ----------
    pm25 : array
        PM2.5 concentration (μg/m³)
    params : SmokeParams
        Model parameters
        
    Returns
    -------
    array
        Visibility in meters
    """
    # Extinction coefficient: β_ext = k * PM2.5
    # k ≈ 4 m²/g for typical smoke
    beta_ext = params.visibility_extinction_coef * pm25 / 1e6  # Convert μg to g
    
    # Background extinction (clean air visibility)
    beta_background = 3.912 / params.background_visibility
    
    # Total extinction
    beta_total = beta_ext + beta_background
    
    # Visibility (Koschmieder formula)
    visibility = 3.912 / jnp.maximum(beta_total, 1e-10)
    
    # Cap at background visibility
    visibility = jnp.minimum(visibility, params.background_visibility)
    
    return visibility


@jax.jit
def compute_aqi_category(
    pm25: jnp.ndarray,
    params: SmokeParams,
) -> jnp.ndarray:
    """
    Compute AQI category from PM2.5.
    
    Categories:
    0: Good (0-12 μg/m³)
    1: Moderate (12.1-35.4)
    2: Unhealthy for Sensitive Groups (35.5-55.4)
    3: Unhealthy (55.5-150.4)
    4: Very Unhealthy (150.5-250.4)
    5: Hazardous (>250.5)
    
    Parameters
    ----------
    pm25 : array
        PM2.5 concentration (μg/m³)
    params : SmokeParams
        Model parameters
        
    Returns
    -------
    array
        AQI category (0-5)
    """
    category = jnp.zeros_like(pm25, dtype=jnp.int32)
    
    category = jnp.where(pm25 > params.aqi_good, 1, category)
    category = jnp.where(pm25 > params.aqi_moderate, 2, category)
    category = jnp.where(pm25 > params.aqi_unhealthy_sensitive, 3, category)
    category = jnp.where(pm25 > params.aqi_unhealthy, 4, category)
    category = jnp.where(pm25 > params.aqi_very_unhealthy, 5, category)
    
    return category


# =============================================================================
# Main Interface Functions
# =============================================================================

def initialize_smoke_state(
    shape: tuple[int, int],
) -> SmokeState:
    """
    Initialize smoke state for a domain.
    
    Parameters
    ----------
    shape : tuple
        Grid shape (ny, nx)
        
    Returns
    -------
    SmokeState
        Initialized state (all zeros)
    """
    ny, nx = shape
    
    return SmokeState(
        concentration=jnp.zeros((ny, nx)),
        cumulative_exposure=jnp.zeros((ny, nx)),
        max_concentration=jnp.zeros((ny, nx)),
        plume_age=jnp.zeros((ny, nx)),
    )


def compute_stable_timestep(
    wind_speed: jnp.ndarray,
    diffusion: jnp.ndarray,
    dx: float,
    params: SmokeParams,
) -> float:
    """
    Compute stable timestep for advection-diffusion.
    
    Parameters
    ----------
    wind_speed : array
        Wind speed field (m/s)
    diffusion : array
        Diffusion coefficient field (m²/s)
    dx : float
        Grid spacing (m)
    params : SmokeParams
        Model parameters
        
    Returns
    -------
    float
        Maximum stable timestep (s)
    """
    # CFL condition for advection
    max_wind = float(jnp.max(wind_speed))
    if max_wind > 0:
        dt_advection = dx / max_wind
    else:
        dt_advection = 1e10
    
    # Stability condition for diffusion
    max_D = float(jnp.max(diffusion))
    if max_D > 0:
        dt_diffusion = 0.25 * dx * dx / max_D
    else:
        dt_diffusion = 1e10
    
    # Take minimum with safety factor
    dt = params.cfl_safety * min(dt_advection, dt_diffusion)
    
    return max(dt, 0.1)  # At least 0.1 second


def update_smoke_state(
    state: SmokeState,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    fire_ros: jnp.ndarray,
    fire_intensity: jnp.ndarray,
    fire_mask: jnp.ndarray,
    dx: float,
    dt_seconds: float,
    precipitation: float,
    params: SmokeParams,
) -> SmokeState:
    """
    Update smoke transport for one timestep.
    
    Uses operator splitting: advection -> diffusion -> source -> decay
    
    Parameters
    ----------
    state : SmokeState
        Current smoke state
    wind_u : array
        Wind velocity x-component (m/s)
    wind_v : array
        Wind velocity y-component (m/s)
    fire_ros : array
        Fire rate of spread (m/min)
    fire_intensity : array
        Fire intensity (kW/m)
    fire_mask : array
        Boolean mask of active fire
    dx : float
        Grid spacing (m)
    dt_seconds : float
        Timestep in seconds
    precipitation : float
        Precipitation rate (mm/hr)
    params : SmokeParams
        Model parameters
        
    Returns
    -------
    SmokeState
        Updated state
    """
    C = state.concentration
    
    # Compute wind speed for diffusion
    wind_speed = jnp.sqrt(wind_u**2 + wind_v**2)
    
    # Effective diffusion coefficient
    D = compute_effective_diffusion(wind_speed, params)
    
    # Sub-cycle if needed for stability
    n_substeps = max(1, int(dt_seconds / 10.0))
    dt_sub = dt_seconds / n_substeps
    
    # Run substeps without jit scan for simplicity
    for _ in range(n_substeps):
        # Advection
        C = advection_step(C, wind_u, wind_v, dx, dt_sub)
        
        # Diffusion
        C = diffusion_step(C, D, dx, dt_sub)
        
        # Ensure non-negative
        C = jnp.maximum(C, 0.0)
    
    # Source term (emissions)
    source = compute_emission_source(
        fire_ros, fire_intensity, fire_mask, dx, params
    )
    C = C + source * dt_seconds
    
    # Decay and deposition
    C = apply_decay(C, precipitation, dt_seconds, params)
    
    # Apply minimum threshold
    C = jnp.where(C < params.min_concentration, 0.0, C)
    
    # Update tracking variables
    cumulative = state.cumulative_exposure + C * dt_seconds
    max_conc = jnp.maximum(state.max_concentration, C)
    
    # Plume age: increment where smoke present, reset where none
    plume_age = jnp.where(
        C > params.min_concentration * 100,
        state.plume_age + dt_seconds / 60.0,  # In minutes
        0.0
    )
    
    return SmokeState(
        concentration=C,
        cumulative_exposure=cumulative,
        max_concentration=max_conc,
        plume_age=plume_age,
    )


@partial(jax.jit, static_argnums=(1,))
def compute_smoke_impacts(
    state: SmokeState,
    params: SmokeParams,
) -> SmokeResult:
    """
    Compute smoke impacts from current concentration.
    
    Parameters
    ----------
    state : SmokeState
        Current smoke state
    params : SmokeParams
        Model parameters
        
    Returns
    -------
    SmokeResult
        Computed impacts
    """
    # Convert to PM2.5
    pm25 = concentration_to_pm25(state.concentration)
    
    # Visibility
    visibility = compute_visibility(pm25, params)
    
    # AQI category
    aqi_category = compute_aqi_category(pm25, params)
    
    # Plume mask (significant smoke)
    plume_mask = pm25 > 5.0  # 5 μg/m³ threshold
    
    return SmokeResult(
        concentration=state.concentration,
        pm25=pm25,
        visibility=visibility,
        aqi_category=aqi_category,
        plume_mask=plume_mask,
    )


# =============================================================================
# Plume Analysis Functions
# =============================================================================

@jax.jit
def compute_plume_centerline(
    concentration: jnp.ndarray,
    dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute smoke plume centerline (maximum concentration path).
    
    Parameters
    ----------
    concentration : array
        Concentration field
    dx : float
        Grid spacing (m)
        
    Returns
    -------
    tuple
        (x_coords, y_coords) of centerline in meters
    """
    ny, nx = concentration.shape
    
    # Find maximum concentration in each row
    max_idx = jnp.argmax(concentration, axis=1)
    
    # Convert to coordinates
    y_coords = jnp.arange(ny) * dx
    x_coords = max_idx * dx
    
    # Only include rows with significant concentration
    row_max = jnp.max(concentration, axis=1)
    threshold = jnp.max(concentration) * 0.1
    
    valid = row_max > threshold
    
    return x_coords * valid, y_coords * valid


@jax.jit
def compute_plume_width(
    concentration: jnp.ndarray,
    dx: float,
    threshold_fraction: float = 0.1,
) -> jnp.ndarray:
    """
    Compute plume width at each downwind distance.
    
    Parameters
    ----------
    concentration : array
        Concentration field
    dx : float
        Grid spacing (m)
    threshold_fraction : float
        Fraction of max to define plume edge
        
    Returns
    -------
    array
        Plume width at each row (m)
    """
    ny, nx = concentration.shape
    
    # Threshold for plume definition
    row_max = jnp.max(concentration, axis=1, keepdims=True)
    threshold = row_max * threshold_fraction
    
    # Find plume extent in each row
    above_threshold = concentration > threshold
    
    # Find first and last index above threshold in each row
    indices = jnp.arange(nx)
    
    # Masked indices
    masked = jnp.where(above_threshold, indices, -1)
    first_idx = jnp.min(jnp.where(masked >= 0, masked, nx), axis=1)
    last_idx = jnp.max(masked, axis=1)
    
    # Width in meters
    width = (last_idx - first_idx + 1) * dx
    width = jnp.where(last_idx >= first_idx, width, 0.0)
    
    return width


def summarize_smoke_impacts(
    result: SmokeResult,
    dx: float,
) -> dict:
    """
    Generate summary of smoke impacts.
    
    Parameters
    ----------
    result : SmokeResult
        Smoke calculation results
    dx : float
        Grid spacing (m)
        
    Returns
    -------
    dict
        Summary statistics
    """
    # Cell area in km²
    cell_area_km2 = (dx / 1000.0) ** 2
    
    # Count cells in each AQI category
    n_cells = result.aqi_category.size
    
    return {
        "pm25_mean": float(jnp.mean(result.pm25)),
        "pm25_max": float(jnp.max(result.pm25)),
        "pm25_p95": float(jnp.percentile(result.pm25, 95)),
        "visibility_min_km": float(jnp.min(result.visibility)) / 1000.0,
        "visibility_mean_km": float(jnp.mean(result.visibility)) / 1000.0,
        "plume_area_km2": float(jnp.sum(result.plume_mask)) * cell_area_km2,
        "aqi_good_pct": float(jnp.sum(result.aqi_category == 0)) / n_cells * 100,
        "aqi_moderate_pct": float(jnp.sum(result.aqi_category == 1)) / n_cells * 100,
        "aqi_unhealthy_sensitive_pct": float(jnp.sum(result.aqi_category == 2)) / n_cells * 100,
        "aqi_unhealthy_pct": float(jnp.sum(result.aqi_category >= 3)) / n_cells * 100,
    }
