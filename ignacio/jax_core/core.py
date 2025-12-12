"""
Ignacio JAX Core - Differentiable Fire Spread Model.

This module provides a numerically robust, differentiable implementation of
Richards' elliptical fire spread equations optimized for gradient-based
calibration using JAX.

Key improvements over base implementation:
- Numerically stable gradient computation (smooth clamps, safe divisions)
- Built-in parameter regularization and priors
- Multiple integration schemes (Euler, Heun)
- Soft boundary handling for topology robustness
- JIT-compiled forward model for performance

Primary calibration parameters:
- wind_adj: Wind speed multiplier (affects ISI → ROS)
- ffmc_adj: Fuel moisture code bias (drier conditions increase spread)

Example usage:
    >>> from ignacio.jax_core.core import (
    ...     calibrate_wind_and_moisture,
    ...     create_observation,
    ...     FireParams,
    ... )
    >>> obs = create_observation(area=50000, duration=60, ffmc=90, wind=20)
    >>> result = calibrate_wind_and_moisture([obs])
    >>> print(f"wind_adj={result.params.wind_adj:.3f}")

Author: Ignacio Team
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Callable
import logging

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, value_and_grad
import optax


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Numerical stability
EPS = 1e-8
GRAD_CLIP = 10.0

# Default calibration bounds
DEFAULT_BOUNDS = {
    "wind_adj": (0.3, 3.0),
    "ffmc_adj": (-15.0, 15.0),
    "ros_scale": (0.3, 3.0),
    "backing_frac": (0.05, 0.5),
}


# =============================================================================
# Parameter Containers
# =============================================================================


class FireParams(NamedTuple):
    """
    Calibratable fire behavior parameters.
    
    These parameters modify the FBP System calculations to better match
    observed fire behavior in a specific region or fuel type.
    
    Attributes
    ----------
    wind_adj : float
        Wind speed adjustment factor (multiplicative).
        > 1.0 increases effective wind → faster spread
        < 1.0 decreases effective wind → slower spread
        Default: 1.0
        
    ffmc_adj : float
        Fine Fuel Moisture Code adjustment (additive).
        > 0.0 increases FFMC → drier fuel → faster spread
        < 0.0 decreases FFMC → wetter fuel → slower spread
        Default: 0.0
        
    ros_scale : float
        Overall rate of spread scaling factor.
        Applied after FBP calculations.
        Default: 1.0
        
    backing_frac : float
        Back fire ROS as fraction of head fire ROS.
        Default: 0.2 (20%)
    """
    wind_adj: float = 1.0
    ffmc_adj: float = 0.0
    ros_scale: float = 1.0
    backing_frac: float = 0.2


class FireGrids(NamedTuple):
    """
    Spatial grids of fire behavior parameters.
    
    All arrays are (ny, nx) for 2D or (nt, ny, nx) for time-varying.
    
    Attributes
    ----------
    x_coords : jnp.ndarray
        1D array of x coordinates (column centers).
    y_coords : jnp.ndarray
        1D array of y coordinates (row centers).
    ros : jnp.ndarray
        Head fire rate of spread (m/min).
    bros : jnp.ndarray
        Back fire rate of spread (m/min).
    fros : jnp.ndarray
        Flank fire rate of spread (m/min).
    raz : jnp.ndarray
        Spread direction/azimuth (radians from north, clockwise).
    """
    x_coords: jnp.ndarray
    y_coords: jnp.ndarray
    ros: jnp.ndarray
    bros: jnp.ndarray
    fros: jnp.ndarray
    raz: jnp.ndarray


class Observation(NamedTuple):
    """
    Fire observation for calibration.
    
    Contains all inputs needed to simulate a fire and the observed
    outcome (area) for loss computation.
    
    Attributes
    ----------
    fire_id : str
        Unique identifier.
    x_ign : float
        Ignition x coordinate.
    y_ign : float
        Ignition y coordinate.
    observed_area : float
        Observed fire area (m²).
    duration : float
        Fire duration (minutes).
    ffmc : float
        Fine Fuel Moisture Code (0-101).
    bui : float
        Buildup Index.
    wind_speed : float
        Wind speed (km/h).
    wind_dir : float
        Wind direction (degrees FROM, meteorological convention).
    fuel_grid : jnp.ndarray
        2D grid of fuel type IDs.
    x_coords : jnp.ndarray
        X coordinates of grid.
    y_coords : jnp.ndarray
        Y coordinates of grid.
    """
    fire_id: str
    x_ign: float
    y_ign: float
    observed_area: float
    duration: float
    ffmc: float
    bui: float
    wind_speed: float
    wind_dir: float
    fuel_grid: jnp.ndarray
    x_coords: jnp.ndarray
    y_coords: jnp.ndarray


class CalibrationResult(NamedTuple):
    """
    Result from parameter calibration.
    
    Attributes
    ----------
    params : FireParams
        Optimized parameters.
    loss_history : jnp.ndarray
        Loss values at each iteration.
    final_loss : float
        Final loss value.
    n_iter : int
        Number of iterations run.
    converged : bool
        Whether optimization converged.
    """
    params: FireParams
    loss_history: jnp.ndarray
    final_loss: float
    n_iter: int
    converged: bool


# =============================================================================
# Numerical Utilities (Differentiable & Stable)
# =============================================================================


def safe_div(num: jnp.ndarray, denom: jnp.ndarray) -> jnp.ndarray:
    """Safe division with gradient-friendly epsilon."""
    return num / (denom + EPS)


def soft_clamp(x: jnp.ndarray, lo: float, hi: float, sharpness: float = 5.0) -> jnp.ndarray:
    """
    Soft clamp using sigmoid blending.
    
    Differentiable alternative to jnp.clip that avoids zero gradients
    at boundaries.
    """
    # Blend towards lo when x < lo
    blend_lo = jax.nn.sigmoid(sharpness * (lo - x))
    # Blend towards hi when x > hi  
    blend_hi = jax.nn.sigmoid(sharpness * (x - hi))
    
    return x * (1 - blend_lo - blend_hi) + lo * blend_lo + hi * blend_hi


def soft_relu(x: jnp.ndarray, sharpness: float = 10.0) -> jnp.ndarray:
    """Differentiable ReLU approximation: softplus."""
    return jax.nn.softplus(x * sharpness) / sharpness


# =============================================================================
# FBP System (Differentiable)
# =============================================================================


def compute_isi(
    ffmc: jnp.ndarray,
    wind_speed: jnp.ndarray,
    params: FireParams,
) -> jnp.ndarray:
    """
    Compute Initial Spread Index with calibration adjustments.
    
    ISI = f(FFMC, Wind) where both inputs are adjusted by calibration params.
    
    Parameters
    ----------
    ffmc : jnp.ndarray
        Fine Fuel Moisture Code (0-101 scale).
    wind_speed : jnp.ndarray
        Wind speed (km/h).
    params : FireParams
        Calibration parameters.
        
    Returns
    -------
    jnp.ndarray
        Initial Spread Index.
    """
    # Apply calibration adjustments
    ffmc_eff = soft_clamp(ffmc + params.ffmc_adj, 0.0, 101.0)
    wind_eff = soft_relu(wind_speed * params.wind_adj)
    
    # Moisture content from FFMC
    # m = 147.2 * (101 - FFMC) / (59.5 + FFMC)
    m = safe_div(147.2 * (101.0 - ffmc_eff), 59.5 + ffmc_eff)
    
    # Moisture function
    # f_F = 91.9 * exp(-0.1386 * m) * (1 + m^5.31 / 4.93e7)
    m_safe = soft_relu(m)  # Ensure non-negative for power
    f_F = 91.9 * jnp.exp(-0.1386 * m_safe) * (1.0 + jnp.power(m_safe + EPS, 5.31) / 4.93e7)
    
    # Wind function
    f_W = jnp.exp(0.05039 * wind_eff)
    
    # ISI
    isi = 0.208 * f_F * f_W
    
    return soft_relu(isi)


def compute_ros_c2(isi: jnp.ndarray, bui: jnp.ndarray) -> jnp.ndarray:
    """
    C-2 Boreal Spruce rate of spread.
    
    This is the default fuel type for calibration demos.
    ROS = RSI * BE (Rate of Spread Index × Buildup Effect)
    """
    # RSI = a * (1 - exp(-b * ISI))^c
    a, b, c = 110.0, 0.0282, 1.5
    rsi = a * jnp.power(soft_relu(1.0 - jnp.exp(-b * isi)), c)
    
    # Buildup Effect (smooth threshold at BUI=35)
    bui_thresh = 35.0
    bui_excess = soft_relu(bui - bui_thresh)
    
    # BE approximation: exponential ramp
    q = 0.8 * jnp.power(bui_excess + EPS, 0.92)
    be = 1.0 + soft_relu(jnp.exp(45.0 * jnp.log(q + EPS) / (300.0 + q + EPS)) - 1.0)
    
    return soft_relu(rsi * be)


def compute_ros_generic(
    isi: jnp.ndarray,
    bui: jnp.ndarray,
    fuel_type: int = 2,
) -> jnp.ndarray:
    """
    Generic ROS computation by fuel type.
    
    Currently supports C-2 as default. Extend for other fuel types.
    """
    # For now, default to C-2
    return compute_ros_c2(isi, bui)


def compute_lb_ratio(wind_speed: jnp.ndarray, params: FireParams) -> jnp.ndarray:
    """
    Compute length-to-breadth ratio from wind speed.
    
    L/B = 1 + 8.729 * (1 - exp(-0.030 * WS))^2.155
    
    Controls the ellipse shape - higher wind = more elongated.
    """
    ws_eff = soft_relu(wind_speed * params.wind_adj)
    term = soft_relu(1.0 - jnp.exp(-0.030 * ws_eff))
    lb = 1.0 + 8.729 * jnp.power(term + EPS, 2.155)
    return jnp.maximum(lb, 1.0)


def compute_ros_components(
    ros_head: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_dir: jnp.ndarray,
    params: FireParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute all ROS components from head fire ROS.
    
    Parameters
    ----------
    ros_head : jnp.ndarray
        Head fire rate of spread (m/min).
    wind_speed : jnp.ndarray
        Wind speed (km/h).
    wind_dir : jnp.ndarray
        Wind direction (degrees FROM).
    params : FireParams
        Calibration parameters.
        
    Returns
    -------
    ros : jnp.ndarray
        Head fire ROS (scaled).
    bros : jnp.ndarray
        Back fire ROS.
    fros : jnp.ndarray
        Flank fire ROS.
    raz : jnp.ndarray
        Spread direction (radians, mathematical convention).
    """
    # Apply overall ROS scaling
    ros = ros_head * params.ros_scale
    
    # Back fire ROS
    bros = params.backing_frac * ros
    
    # Length-to-breadth ratio
    lb = compute_lb_ratio(wind_speed, params)
    
    # Flank fire ROS (from ellipse geometry)
    fros = safe_div(ros + bros, 2.0 * lb)
    
    # Rate of spread azimuth (direction fire spreads TO)
    # Wind FROM → fire spreads TO (opposite direction)
    # Convert from meteorological (degrees from north) to mathematical (radians from east)
    raz_deg = (wind_dir + 180.0) % 360.0
    raz = jnp.deg2rad(90.0 - raz_deg)  # Convert to math convention
    
    return ros, bros, fros, raz


def fbp_pipeline(
    ffmc: jnp.ndarray,
    bui: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_dir: jnp.ndarray,
    params: FireParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Complete FBP pipeline: weather → ROS components.
    
    Parameters
    ----------
    ffmc : jnp.ndarray
        Fine Fuel Moisture Code.
    bui : jnp.ndarray
        Buildup Index.
    wind_speed : jnp.ndarray
        Wind speed (km/h).
    wind_dir : jnp.ndarray
        Wind direction (degrees FROM).
    params : FireParams
        Calibration parameters.
        
    Returns
    -------
    ros, bros, fros, raz : tuple of jnp.ndarray
        Rate of spread components.
    """
    # ISI with calibration adjustments
    isi = compute_isi(ffmc, wind_speed, params)
    
    # ROS from fuel type
    ros_head = compute_ros_generic(isi, bui)
    
    # All components
    return compute_ros_components(ros_head, wind_speed, wind_dir, params)


# =============================================================================
# Spatial Operations (Differentiable)
# =============================================================================


def bilinear_interp(
    field: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
) -> jnp.ndarray:
    """
    Differentiable bilinear interpolation.
    
    Parameters
    ----------
    field : jnp.ndarray
        2D field (ny, nx) to sample.
    x, y : jnp.ndarray
        Query coordinates.
    x_coords, y_coords : jnp.ndarray
        1D coordinate arrays.
        
    Returns
    -------
    jnp.ndarray
        Interpolated values.
    """
    ny, nx = field.shape
    
    # Grid spacing
    dx = x_coords[1] - x_coords[0] if nx > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if ny > 1 else 1.0
    
    # Handle both y-orientations using a unified formula
    # For normal (dy > 0): iy = (y - y_coords[0]) / dy
    # For flipped (dy < 0): iy = (y_coords[0] - y) / |dy|
    # Unified: iy = (y - y_coords[0]) / dy works for both!
    # When dy < 0, this naturally inverts the direction.
    
    # Fractional indices
    ix = (x - x_coords[0]) / dx
    iy = (y - y_coords[0]) / dy
    
    # For flipped grids (dy < 0), iy will be negative for y < y_coords[0]
    # We need to handle this by using absolute iy and flipped indexing
    # Simpler approach: just use the sign-aware formula
    # When dy < 0: y_coords[0] is max, y decreasing → need (y_max - y) / |dy|
    # This is equivalent to: (y - y_min) / |dy| where y_min = y_coords[-1]
    
    # Actually, the cleanest fix is to use where:
    dy_abs = jnp.abs(dy)
    y_flipped = dy < 0  # This is a concrete boolean at trace time if dy is concrete
    
    # Use jnp.where for JAX-compatible conditional
    iy = jnp.where(
        dy < 0,
        (y_coords[0] - y) / dy_abs,  # Flipped case
        (y - y_coords[0]) / dy_abs    # Normal case
    )
    
    # Soft clamp to valid range (gradient-friendly)
    ix = soft_clamp(ix, 0.0, nx - 1.001)
    iy = soft_clamp(iy, 0.0, ny - 1.001)
    
    # Integer indices
    ix0 = jnp.floor(ix).astype(jnp.int32)
    iy0 = jnp.floor(iy).astype(jnp.int32)
    ix1 = jnp.minimum(ix0 + 1, nx - 1)
    iy1 = jnp.minimum(iy0 + 1, ny - 1)
    
    # Fractional parts
    fx = ix - ix0
    fy = iy - iy0
    
    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    
    # Interpolate
    return (w00 * field[iy0, ix0] + w10 * field[iy0, ix1] +
            w01 * field[iy1, ix0] + w11 * field[iy1, ix1])


def sample_grids(
    grids: FireGrids,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t_idx: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample fire parameter grids at given positions.
    
    Parameters
    ----------
    grids : FireGrids
        Fire parameter grids.
    x, y : jnp.ndarray
        Query coordinates.
    t_idx : int
        Time index (for time-varying grids).
        
    Returns
    -------
    ros, bros, fros, raz : tuple of jnp.ndarray
        Sampled parameters at each (x, y) position.
    """
    # Handle 2D vs 3D arrays
    def get_field(arr, t_idx):
        if arr.ndim == 3:
            nt = arr.shape[0]
            t_idx = jnp.clip(t_idx, 0, nt - 1)
            return arr[t_idx]
        return arr
    
    ros = bilinear_interp(
        get_field(grids.ros, t_idx), x, y, grids.x_coords, grids.y_coords
    )
    bros = bilinear_interp(
        get_field(grids.bros, t_idx), x, y, grids.x_coords, grids.y_coords
    )
    fros = bilinear_interp(
        get_field(grids.fros, t_idx), x, y, grids.x_coords, grids.y_coords
    )
    raz = bilinear_interp(
        get_field(grids.raz, t_idx), x, y, grids.x_coords, grids.y_coords
    )
    
    return ros, bros, fros, raz


# =============================================================================
# Richards' Equations (Differentiable)
# =============================================================================


def compute_tangents(
    x: jnp.ndarray,
    y: jnp.ndarray,
    normalize: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute tangent vectors along closed curve using central differences.
    
    Parameters
    ----------
    x, y : jnp.ndarray
        Vertex coordinates of closed curve.
    normalize : bool
        Return unit tangent vectors.
        
    Returns
    -------
    tx, ty : jnp.ndarray
        Tangent vector components.
    """
    # Central differences (periodic boundary)
    tx = (jnp.roll(x, -1) - jnp.roll(x, 1)) / 2.0
    ty = (jnp.roll(y, -1) - jnp.roll(y, 1)) / 2.0
    
    if normalize:
        mag = jnp.sqrt(tx**2 + ty**2 + EPS)
        tx = tx / mag
        ty = ty / mag
    
    return tx, ty


def ros_to_ellipse(
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert ROS components to ellipse parameters.
    
    a = (ROS + BROS) / 2  (semi-major axis)
    b = FROS              (semi-minor axis)
    c = (ROS - BROS) / 2  (center offset)
    theta = RAZ           (orientation)
    
    Returns
    -------
    a, b, c, theta : tuple of jnp.ndarray
        Ellipse parameters.
    """
    a = 0.5 * (ros + bros)
    c = 0.5 * (ros - bros)
    b = fros
    theta = raz
    
    return a, b, c, theta


def richards_velocity(
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute fire front velocities using Richards' elliptical spread equations.
    
    Implements the component form of Richards' (1990) differential equations:
    
        dx/dt = [b² cos(θ)(x_s sin(θ) + y_s cos(θ)) 
                - a² sin(θ)(x_s cos(θ) - y_s sin(θ))] / D + c sin(θ)
                
        dy/dt = [-b² sin(θ)(x_s sin(θ) + y_s cos(θ))
                - a² cos(θ)(x_s cos(θ) - y_s sin(θ))] / D + c cos(θ)
    
    where D = sqrt(a²(x_s cos(θ) - y_s sin(θ))² + b²(x_s sin(θ) + y_s cos(θ))²)
    
    Parameters
    ----------
    x, y : jnp.ndarray
        Vertex coordinates.
    a : jnp.ndarray
        Semi-major axis at each vertex.
    b : jnp.ndarray
        Semi-minor axis at each vertex.
    c : jnp.ndarray
        Center offset at each vertex.
    theta : jnp.ndarray
        Ellipse orientation at each vertex.
        
    Returns
    -------
    vx, vy : jnp.ndarray
        Velocity components at each vertex.
    """
    # Unit tangent vectors
    tx, ty = compute_tangents(x, y, normalize=True)
    
    # Trig terms
    cos_th = jnp.cos(theta)
    sin_th = jnp.sin(theta)
    
    # Rotated tangent components
    term1 = tx * cos_th - ty * sin_th
    term2 = tx * sin_th + ty * cos_th
    
    # Denominator (with numerical stability)
    denom = jnp.sqrt(a**2 * term1**2 + b**2 * term2**2 + EPS)
    
    # Velocity components
    vx = safe_div(b**2 * cos_th * term2 - a**2 * sin_th * term1, denom) + c * sin_th
    vy = safe_div(-b**2 * sin_th * term2 - a**2 * cos_th * term1, denom) + c * cos_th
    
    return vx, vy


# =============================================================================
# Perimeter Evolution
# =============================================================================


def create_perimeter(
    x_center: float,
    y_center: float,
    radius: float,
    n_vertices: int = 200,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create initial circular fire perimeter.
    
    Parameters
    ----------
    x_center, y_center : float
        Center coordinates.
    radius : float
        Initial radius.
    n_vertices : int
        Number of vertices.
        
    Returns
    -------
    x, y : jnp.ndarray
        Vertex coordinates.
    """
    theta = jnp.linspace(0, 2 * jnp.pi, n_vertices, endpoint=False)
    x = x_center + radius * jnp.cos(theta)
    y = y_center + radius * jnp.sin(theta)
    return x, y


def compute_outward_normal(
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute outward normal vectors for each vertex.
    
    For a counter-clockwise oriented polygon, the outward normal
    is the tangent rotated 90 degrees clockwise (right).
    """
    # Tangent vectors (central difference)
    tx = (jnp.roll(x, -1) - jnp.roll(x, 1)) / 2.0
    ty = (jnp.roll(y, -1) - jnp.roll(y, 1)) / 2.0
    
    # Outward normal: rotate tangent 90° right (for CCW polygon)
    nx = ty
    ny = -tx
    
    # Normalize
    mag = jnp.sqrt(nx**2 + ny**2 + EPS)
    nx = nx / mag
    ny = ny / mag
    
    return nx, ny


def compute_active_vertices_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
) -> jnp.ndarray:
    """
    Determine active vertices using velocity direction check.
    
    A vertex is active if its velocity points generally outward
    (positive dot product with outward normal). This is a simplified
    version of the full marker method that's JAX-compatible.
    
    Returns
    -------
    active : jnp.ndarray
        Float array (0.0 to 1.0) indicating vertex activity.
        Uses soft threshold for differentiability.
    """
    # Get outward normals
    nx, ny = compute_outward_normal(x, y)
    
    # Dot product of velocity with outward normal
    # Positive = moving outward (active), Negative = moving inward (inactive)
    dot = vx * nx + vy * ny
    
    # Soft threshold: sigmoid for differentiability
    # Vertices with outward velocity get weight ~1, inward get ~0
    # Use small scale factor to make it fairly sharp but still differentiable
    scale = 100.0  # Sharpness of transition
    active = jax.nn.sigmoid(scale * dot)
    
    return active


def evolve_step_euler(
    x: jnp.ndarray,
    y: jnp.ndarray,
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
    dt: float,
    use_marker: bool = False,  # Disabled by default - needs proper implementation
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evolve perimeter by one time step using Forward Euler.
    
    Parameters
    ----------
    x, y : jnp.ndarray
        Current vertex coordinates.
    ros, bros, fros, raz : jnp.ndarray
        Fire behavior parameters at each vertex.
    dt : float
        Time step (same units as ROS, typically minutes).
    use_marker : bool
        If True, apply marker method to freeze inward-moving vertices.
        
    Returns
    -------
    x_new, y_new : jnp.ndarray
        Updated coordinates.
    """
    # Convert ROS to ellipse parameters
    a, b, c, theta = ros_to_ellipse(ros, bros, fros, raz)
    
    # Compute velocities
    vx, vy = richards_velocity(x, y, a, b, c, theta)
    
    # Apply marker method to freeze inward-moving vertices
    if use_marker:
        active = compute_active_vertices_jax(x, y, vx, vy)
        vx = vx * active
        vy = vy * active
    
    # Euler update
    x_new = x + dt * vx
    y_new = y + dt * vy
    
    return x_new, y_new


def evolve_step_heun(
    x: jnp.ndarray,
    y: jnp.ndarray,
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
    dt: float,
    use_marker: bool = False,  # Disabled by default - needs proper implementation
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evolve perimeter using Heun's method (2nd-order Runge-Kutta).
    
    More accurate than Euler, especially for curved fronts.
    """
    # Convert to ellipse params
    a, b, c, theta = ros_to_ellipse(ros, bros, fros, raz)
    
    # Stage 1: Euler predictor
    vx1, vy1 = richards_velocity(x, y, a, b, c, theta)
    
    # Apply marker method
    if use_marker:
        active1 = compute_active_vertices_jax(x, y, vx1, vy1)
        vx1 = vx1 * active1
        vy1 = vy1 * active1
    
    x_pred = x + dt * vx1
    y_pred = y + dt * vy1
    
    # Stage 2: Evaluate at predicted position
    vx2, vy2 = richards_velocity(x_pred, y_pred, a, b, c, theta)
    
    # Apply marker at predicted position
    if use_marker:
        active2 = compute_active_vertices_jax(x_pred, y_pred, vx2, vy2)
        vx2 = vx2 * active2
        vy2 = vy2 * active2
    
    # Corrector (average of velocities)
    x_new = x + 0.5 * dt * (vx1 + vx2)
    y_new = y + 0.5 * dt * (vy1 + vy2)
    
    return x_new, y_new


# =============================================================================
# Fire Simulation
# =============================================================================


def simulate_fire(
    grids: FireGrids,
    x_ign: float,
    y_ign: float,
    n_steps: int,
    dt: float = 1.0,
    n_vertices: int = 200,
    initial_radius: float = 1.0,
    method: str = "euler",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate fire spread from ignition point.
    
    Parameters
    ----------
    grids : FireGrids
        Spatial fire behavior parameters.
    x_ign, y_ign : float
        Ignition coordinates.
    n_steps : int
        Number of time steps.
    dt : float
        Time step (minutes).
    n_vertices : int
        Number of perimeter vertices.
    initial_radius : float
        Initial fire radius (same units as coordinates).
    method : str
        Integration method: "euler" or "heun".
        
    Returns
    -------
    x_final, y_final : jnp.ndarray
        Final perimeter coordinates.
    """
    # Initialize perimeter
    x, y = create_perimeter(x_ign, y_ign, initial_radius, n_vertices)
    
    # Select integration method
    step_fn = evolve_step_euler if method == "euler" else evolve_step_heun
    
    def body_fn(i, state):
        x, y = state
        
        # Sample parameters at current positions
        ros, bros, fros, raz = sample_grids(grids, x, y, i)
        
        # Evolve
        x_new, y_new = step_fn(x, y, ros, bros, fros, raz, dt)
        
        return (x_new, y_new)
    
    # Run simulation with fori_loop (JIT-compatible)
    x_final, y_final = lax.fori_loop(0, n_steps, body_fn, (x, y))
    
    return x_final, y_final


def compute_area(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute area enclosed by perimeter using Shoelace formula.
    
    Differentiable area computation.
    """
    x_next = jnp.roll(x, -1)
    y_next = jnp.roll(y, -1)
    return 0.5 * jnp.abs(jnp.sum(x * y_next - x_next * y))


# =============================================================================
# Forward Model (Complete Pipeline)
# =============================================================================


def forward_model(
    params: FireParams,
    obs: Observation,
    dt: float = 1.0,
    n_vertices: int = 200,
    initial_radius: float = 1.0,
) -> jnp.ndarray:
    """
    Run complete forward model: params + observation → predicted area.
    
    This is the main function for calibration loss computation.
    
    Parameters
    ----------
    params : FireParams
        Calibration parameters to evaluate.
    obs : Observation
        Fire observation with weather and ignition data.
    dt : float
        Time step (minutes).
    n_vertices : int
        Number of perimeter vertices.
    initial_radius : float
        Initial fire radius (m).
        
    Returns
    -------
    jnp.ndarray
        Predicted fire area (m²).
    """
    # Compute ROS components via FBP pipeline
    ros, bros, fros, raz = fbp_pipeline(
        jnp.array(obs.ffmc),
        jnp.array(obs.bui),
        jnp.array(obs.wind_speed),
        jnp.array(obs.wind_dir),
        params,
    )
    
    # Create spatial grids (uniform for now)
    ny, nx = obs.fuel_grid.shape
    
    # Broadcast to grid shape
    ros_grid = jnp.full((ny, nx), ros)
    bros_grid = jnp.full((ny, nx), bros)
    fros_grid = jnp.full((ny, nx), fros)
    raz_grid = jnp.full((ny, nx), raz)
    
    grids = FireGrids(
        x_coords=obs.x_coords,
        y_coords=obs.y_coords,
        ros=ros_grid,
        bros=bros_grid,
        fros=fros_grid,
        raz=raz_grid,
    )
    
    # Number of time steps
    n_steps = max(1, int(obs.duration / dt))
    
    # Simulate
    x_final, y_final = simulate_fire(
        grids, obs.x_ign, obs.y_ign,
        n_steps=n_steps,
        dt=dt,
        n_vertices=n_vertices,
        initial_radius=initial_radius,
    )
    
    # Compute area
    return compute_area(x_final, y_final)


# =============================================================================
# Loss Functions
# =============================================================================


def area_loss(predicted: jnp.ndarray, observed: jnp.ndarray) -> jnp.ndarray:
    """
    Relative squared error loss for fire area.
    
    L = ((pred - obs) / obs)²
    
    Scale-invariant: treats 10% error the same regardless of absolute size.
    """
    rel_error = safe_div(predicted - observed, observed)
    return rel_error ** 2


def calibration_loss(
    param_array: jnp.ndarray,
    observations: list[Observation],
    param_names: list[str],
    base_params: FireParams,
    reg_strength: float = 0.01,
    dt: float = 1.0,
    n_vertices: int = 200,
) -> jnp.ndarray:
    """
    Compute calibration loss over batch of observations.
    
    Loss = mean(area_loss) + regularization
    
    Parameters
    ----------
    param_array : jnp.ndarray
        Array of parameter values being optimized.
    observations : list[Observation]
        Training observations.
    param_names : list[str]
        Names of parameters in param_array.
    base_params : FireParams
        Base parameters for non-optimized values.
    reg_strength : float
        L2 regularization strength on parameter deviations.
    dt : float
        Time step.
    n_vertices : int
        Number of perimeter vertices.
        
    Returns
    -------
    jnp.ndarray
        Total loss value.
    """
    # Reconstruct full parameters
    params_dict = base_params._asdict()
    for i, name in enumerate(param_names):
        params_dict[name] = param_array[i]
    params = FireParams(**params_dict)
    
    # Compute loss for each observation
    total_loss = jnp.array(0.0)
    for obs in observations:
        pred_area = forward_model(params, obs, dt=dt, n_vertices=n_vertices)
        obs_area = jnp.array(obs.observed_area)
        total_loss = total_loss + area_loss(pred_area, obs_area)
    
    # Average over observations
    mean_loss = total_loss / len(observations)
    
    # Regularization: penalize deviation from defaults
    defaults = jnp.array([getattr(FireParams(), name) for name in param_names])
    reg_loss = reg_strength * jnp.sum((param_array - defaults) ** 2)
    
    return mean_loss + reg_loss


# =============================================================================
# Calibration
# =============================================================================


def calibrate(
    observations: list[Observation],
    param_names: list[str] = None,
    initial_values: dict = None,
    bounds: dict = None,
    learning_rate: float = 0.05,
    n_iterations: int = 100,
    convergence_tol: float = 1e-7,
    reg_strength: float = 0.01,
    dt: float = 1.0,
    n_vertices: int = 200,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Calibrate fire behavior parameters using gradient descent.
    
    Parameters
    ----------
    observations : list[Observation]
        Training data.
    param_names : list[str]
        Parameters to optimize. Default: ["wind_adj", "ffmc_adj"]
    initial_values : dict
        Starting values. Default: defaults from FireParams.
    bounds : dict
        Parameter bounds {name: (lo, hi)}.
    learning_rate : float
        Optimizer learning rate.
    n_iterations : int
        Maximum iterations.
    convergence_tol : float
        Stop when loss change < tol.
    reg_strength : float
        Regularization strength.
    dt : float
        Time step for simulation.
    n_vertices : int
        Number of perimeter vertices.
    verbose : bool
        Print progress.
        
    Returns
    -------
    CalibrationResult
        Optimized parameters and diagnostics.
    """
    # Defaults
    if param_names is None:
        param_names = ["wind_adj", "ffmc_adj"]
    
    if initial_values is None:
        initial_values = {}
    
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    
    base_params = FireParams()
    
    # Initialize parameter array
    param_array = jnp.array([
        initial_values.get(name, getattr(base_params, name))
        for name in param_names
    ])
    
    if verbose:
        logger.info(f"Starting calibration with {len(observations)} observations")
        logger.info(f"Optimizing: {param_names}")
        logger.info(f"Initial values: {dict(zip(param_names, param_array.tolist()))}")
    
    # Create loss function
    def loss_fn(p):
        return calibration_loss(
            p, observations, param_names, base_params,
            reg_strength=reg_strength, dt=dt, n_vertices=n_vertices
        )
    
    # JIT compile
    loss_and_grad_fn = jit(value_and_grad(loss_fn))
    
    # Setup optimizer (Adam with gradient clipping)
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(param_array)
    
    # Optimization loop
    loss_history = []
    prev_loss = jnp.inf
    
    for iteration in range(n_iterations):
        loss, grads = loss_and_grad_fn(param_array)
        loss_history.append(float(loss))
        
        # Progress logging
        if verbose and iteration % 10 == 0:
            param_str = ", ".join(f"{n}={param_array[i]:.4f}" 
                                  for i, n in enumerate(param_names))
            logger.info(f"Iter {iteration:3d}: loss={loss:.6f}, {param_str}")
        
        # Convergence check
        if jnp.abs(prev_loss - loss) < convergence_tol:
            if verbose:
                logger.info(f"Converged at iteration {iteration}")
            break
        prev_loss = loss
        
        # Update
        updates, opt_state = optimizer.update(grads, opt_state, param_array)
        param_array = optax.apply_updates(param_array, updates)
        
        # Apply bounds (hard clip after soft operations)
        for i, name in enumerate(param_names):
            lo, hi = bounds.get(name, (-jnp.inf, jnp.inf))
            param_array = param_array.at[i].set(jnp.clip(param_array[i], lo, hi))
    
    # Build final parameters
    final_dict = base_params._asdict()
    for i, name in enumerate(param_names):
        final_dict[name] = float(param_array[i])
    final_params = FireParams(**final_dict)
    
    if verbose:
        logger.info(f"Calibration complete. Final loss: {loss_history[-1]:.6f}")
        logger.info(f"Final parameters: {final_params}")
    
    return CalibrationResult(
        params=final_params,
        loss_history=jnp.array(loss_history),
        final_loss=loss_history[-1],
        n_iter=len(loss_history),
        converged=len(loss_history) < n_iterations,
    )


def calibrate_wind_and_moisture(
    observations: list[Observation],
    learning_rate: float = 0.1,
    n_iterations: int = 50,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Quick calibration of wind and fuel moisture adjustment factors.
    
    This is the most common calibration scenario - adjusting for local
    wind effects and fuel moisture conditions.
    
    Parameters
    ----------
    observations : list[Observation]
        Training observations with observed fire areas.
    learning_rate : float
        Optimizer learning rate. Higher = faster but less stable.
    n_iterations : int
        Maximum iterations.
    verbose : bool
        Print progress.
        
    Returns
    -------
    CalibrationResult
        Calibrated wind_adj and ffmc_adj.
    
    Example
    -------
    >>> obs = create_observation(area=50000, duration=60, ffmc=90, wind=20)
    >>> result = calibrate_wind_and_moisture([obs])
    >>> print(f"wind_adj = {result.params.wind_adj:.3f}")
    """
    return calibrate(
        observations,
        param_names=["wind_adj", "ffmc_adj"],
        initial_values={"wind_adj": 1.0, "ffmc_adj": 0.0},
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        verbose=verbose,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_observation(
    fire_id: str = "fire_001",
    x_ign: float = 500.0,
    y_ign: float = 500.0,
    area: float = 10000.0,
    duration: float = 60.0,
    ffmc: float = 90.0,
    bui: float = 80.0,
    wind: float = 20.0,
    wind_dir: float = 270.0,
    fuel_type: int = 2,
    grid_size: int = 100,
    cell_size: float = 10.0,
) -> Observation:
    """
    Create a fire observation for calibration.
    
    Parameters
    ----------
    fire_id : str
        Unique identifier.
    x_ign, y_ign : float
        Ignition coordinates.
    area : float
        Observed fire area (m²).
    duration : float
        Fire duration (minutes).
    ffmc : float
        Fine Fuel Moisture Code (0-101).
    bui : float
        Buildup Index.
    wind : float
        Wind speed (km/h).
    wind_dir : float
        Wind direction (degrees FROM).
    fuel_type : int
        Fuel type ID (2 = C-2 Boreal Spruce).
    grid_size : int
        Grid dimensions (cells).
    cell_size : float
        Cell size (m).
        
    Returns
    -------
    Observation
        Fire observation ready for calibration.
    """
    x_coords = jnp.arange(grid_size) * cell_size
    y_coords = jnp.arange(grid_size) * cell_size
    fuel_grid = jnp.full((grid_size, grid_size), fuel_type, dtype=jnp.int32)
    
    return Observation(
        fire_id=fire_id,
        x_ign=x_ign,
        y_ign=y_ign,
        observed_area=area,
        duration=duration,
        ffmc=ffmc,
        bui=bui,
        wind_speed=wind,
        wind_dir=wind_dir,
        fuel_grid=fuel_grid,
        x_coords=x_coords,
        y_coords=y_coords,
    )


def validate(
    params: FireParams,
    observations: list[Observation],
    dt: float = 1.0,
) -> dict:
    """
    Validate calibrated parameters on observations.
    
    Parameters
    ----------
    params : FireParams
        Calibrated parameters.
    observations : list[Observation]
        Validation set.
    dt : float
        Time step.
        
    Returns
    -------
    dict
        Validation metrics.
    """
    errors = []
    rel_errors = []
    
    for obs in observations:
        pred_area = forward_model(params, obs, dt=dt)
        error = float(pred_area) - obs.observed_area
        rel_error = error / (obs.observed_area + EPS)
        errors.append(error)
        rel_errors.append(rel_error)
    
    errors = jnp.array(errors)
    rel_errors = jnp.array(rel_errors)
    
    return {
        "mean_error": float(jnp.mean(errors)),
        "rmse": float(jnp.sqrt(jnp.mean(errors**2))),
        "mean_rel_error": float(jnp.mean(rel_errors)),
        "rmse_rel": float(jnp.sqrt(jnp.mean(rel_errors**2))),
        "max_rel_error": float(jnp.max(jnp.abs(rel_errors))),
        "n_obs": len(observations),
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Parameter containers
    "FireParams",
    "FireGrids",
    "Observation",
    "CalibrationResult",
    # FBP functions
    "compute_isi",
    "compute_ros_c2",
    "compute_lb_ratio",
    "compute_ros_components",
    "fbp_pipeline",
    # Spatial operations
    "bilinear_interp",
    "sample_grids",
    # Richards' equations
    "compute_tangents",
    "ros_to_ellipse",
    "richards_velocity",
    # Simulation
    "create_perimeter",
    "evolve_step_euler",
    "evolve_step_heun",
    "simulate_fire",
    "compute_area",
    # Forward model
    "forward_model",
    # Loss functions
    "area_loss",
    "calibration_loss",
    # Calibration
    "calibrate",
    "calibrate_wind_and_moisture",
    # Convenience
    "create_observation",
    "validate",
]