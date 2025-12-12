"""
Differentiable Fire Spread Equations for JAX.

This module implements JAX-compatible versions of Richards' differential
equations for elliptical fire spread, enabling gradient-based optimization.

Key features:
- Differentiable bilinear interpolation
- Smooth Richards' velocity computation  
- Vectorized perimeter evolution
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# Fire Parameter Grid (JAX Version)
# =============================================================================


class FireParamsJAX(NamedTuple):
    """
    Container for gridded fire behavior parameters.
    
    All arrays are (nt, ny, nx) for time-varying fields
    or (ny, nx) for static fields.
    
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
        Spread azimuth (radians).
    """
    x_coords: jnp.ndarray
    y_coords: jnp.ndarray
    ros: jnp.ndarray
    bros: jnp.ndarray
    fros: jnp.ndarray
    raz: jnp.ndarray


def create_fire_params(
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
) -> FireParamsJAX:
    """Create FireParamsJAX from arrays."""
    return FireParamsJAX(
        x_coords=jnp.asarray(x_coords),
        y_coords=jnp.asarray(y_coords),
        ros=jnp.asarray(ros),
        bros=jnp.asarray(bros),
        fros=jnp.asarray(fros),
        raz=jnp.asarray(raz),
    )


# =============================================================================
# Differentiable Bilinear Interpolation
# =============================================================================


def bilinear_interpolate_jax(
    field: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
) -> jnp.ndarray:
    """
    Differentiable bilinear interpolation of a 2D field.
    
    Parameters
    ----------
    field : jnp.ndarray
        2D array (ny, nx) to interpolate.
    x : jnp.ndarray
        X coordinates of query points.
    y : jnp.ndarray
        Y coordinates of query points.
    x_coords : jnp.ndarray
        1D array of x coordinate values.
    y_coords : jnp.ndarray
        1D array of y coordinate values.
        
    Returns
    -------
    jnp.ndarray
        Interpolated values at (x, y) positions.
    """
    ny, nx = field.shape
    
    x_min = x_coords[0]
    y_min = y_coords[0]
    dx = x_coords[1] - x_coords[0] if nx > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if ny > 1 else 1.0
    
    # Check if y is flipped (common for raster data)
    y_flipped = dy < 0
    dy_abs = jnp.abs(dy)
    
    # Convert to fractional indices
    ix = (x - x_min) / dx
    
    if y_flipped:
        y_max = y_coords[0]
        iy = (y_max - y) / dy_abs
    else:
        iy = (y - y_min) / dy_abs
    
    # Clamp to valid range (with small epsilon for edge cases)
    eps = 1e-6
    ix = jnp.clip(ix, 0, nx - 1 - eps)
    iy = jnp.clip(iy, 0, ny - 1 - eps)
    
    # Integer indices
    ix0 = jnp.floor(ix).astype(jnp.int32)
    iy0 = jnp.floor(iy).astype(jnp.int32)
    ix1 = jnp.clip(ix0 + 1, 0, nx - 1)
    iy1 = jnp.clip(iy0 + 1, 0, ny - 1)
    
    # Fractional parts
    fx = ix - ix0
    fy = iy - iy0
    
    # Corner values (using advanced indexing)
    f00 = field[iy0, ix0]
    f10 = field[iy0, ix1]
    f01 = field[iy1, ix0]
    f11 = field[iy1, ix1]
    
    # Bilinear interpolation
    f0 = f00 * (1 - fx) + f10 * fx
    f1 = f01 * (1 - fx) + f11 * fx
    result = f0 * (1 - fy) + f1 * fy
    
    return result


def sample_fire_params_jax(
    params: FireParamsJAX,
    t_idx: int,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample fire parameters at given positions for a time step.
    
    Parameters
    ----------
    params : FireParamsJAX
        Fire parameter grids.
    t_idx : int
        Time index.
    x : jnp.ndarray
        X coordinates of query points.
    y : jnp.ndarray
        Y coordinates of query points.
        
    Returns
    -------
    ros, bros, fros, raz : tuple of jnp.ndarray
        Interpolated fire parameters at each point.
    """
    # Handle 2D (static) vs 3D (time-varying) arrays
    if params.ros.ndim == 3:
        nt = params.ros.shape[0]
        t_idx = jnp.clip(t_idx, 0, nt - 1)
        ros_field = params.ros[t_idx]
        bros_field = params.bros[t_idx]
        fros_field = params.fros[t_idx]
        raz_field = params.raz[t_idx]
    else:
        ros_field = params.ros
        bros_field = params.bros
        fros_field = params.fros
        raz_field = params.raz
    
    ros = bilinear_interpolate_jax(ros_field, x, y, params.x_coords, params.y_coords)
    bros = bilinear_interpolate_jax(bros_field, x, y, params.x_coords, params.y_coords)
    fros = bilinear_interpolate_jax(fros_field, x, y, params.x_coords, params.y_coords)
    raz = bilinear_interpolate_jax(raz_field, x, y, params.x_coords, params.y_coords)
    
    return ros, bros, fros, raz


# =============================================================================
# Spatial Derivatives (Differentiable)
# =============================================================================


def compute_spatial_derivatives_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    normalize: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute spatial derivatives along the fire front curve.
    
    Uses periodic central differences for a closed curve.
    Fully differentiable via JAX.
    
    Parameters
    ----------
    x : jnp.ndarray
        X coordinates of front vertices.
    y : jnp.ndarray
        Y coordinates of front vertices.
    normalize : bool
        If True, return unit tangent vectors.
        
    Returns
    -------
    x_s : jnp.ndarray
        Derivative of x with respect to arc parameter.
    y_s : jnp.ndarray
        Derivative of y with respect to arc parameter.
    """
    # Forward and backward neighbors (periodic)
    x_forward = jnp.roll(x, -1)
    x_backward = jnp.roll(x, 1)
    y_forward = jnp.roll(y, -1)
    y_backward = jnp.roll(y, 1)
    
    # Central differences (ds = 2 for central diff)
    x_s = (x_forward - x_backward) / 2.0
    y_s = (y_forward - y_backward) / 2.0
    
    if normalize:
        # Normalize to unit tangent vectors
        mag = jnp.sqrt(x_s**2 + y_s**2 + 1e-12)
        x_s = x_s / mag
        y_s = y_s / mag
    
    return x_s, y_s


# =============================================================================
# Richards' Velocity Equations (Differentiable)
# =============================================================================


def ros_to_ellipse_params_jax(
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert ROS components to ellipse parameters for Richards' equations.
    
    Parameters
    ----------
    ros : jnp.ndarray
        Head fire rate of spread.
    bros : jnp.ndarray
        Back fire rate of spread.
    fros : jnp.ndarray
        Flank fire rate of spread.
    raz : jnp.ndarray
        Spread azimuth (radians).
        
    Returns
    -------
    a : jnp.ndarray
        Semi-major axis = (ROS + BROS) / 2.
    b : jnp.ndarray
        Semi-minor axis = FROS.
    c : jnp.ndarray
        Center offset = (ROS - BROS) / 2.
    theta : jnp.ndarray
        Orientation angle (radians).
    """
    a = 0.5 * (ros + bros)
    c = 0.5 * (ros - bros)
    b = fros
    theta = raz
    
    return a, b, c, theta


def richards_velocity_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    theta: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute fire front velocities using Richards' elliptical spread equations.
    
    Differentiable JAX implementation.
    
    Parameters
    ----------
    x : jnp.ndarray
        X coordinates of front vertices.
    y : jnp.ndarray
        Y coordinates of front vertices.
    a : jnp.ndarray
        Semi-major axis of ellipse.
    b : jnp.ndarray
        Semi-minor axis of ellipse.
    c : jnp.ndarray
        Center offset.
    theta : jnp.ndarray
        Ellipse orientation angle (radians).
        
    Returns
    -------
    x_t : jnp.ndarray
        Time derivative of x (velocity in x direction).
    y_t : jnp.ndarray
        Time derivative of y (velocity in y direction).
    """
    # Compute normalized spatial derivatives (unit tangent vectors)
    x_s, y_s = compute_spatial_derivatives_jax(x, y, normalize=True)
    
    # Broadcast parameters to vertex count
    a = jnp.broadcast_to(a, x.shape)
    b = jnp.broadcast_to(b, x.shape)
    c = jnp.broadcast_to(c, x.shape)
    theta = jnp.broadcast_to(theta, x.shape)
    
    cos_th = jnp.cos(theta)
    sin_th = jnp.sin(theta)
    
    # Rotated derivative terms
    term1 = x_s * cos_th - y_s * sin_th
    term2 = x_s * sin_th + y_s * cos_th
    
    # Denominator with smooth regularization
    eps = 1e-10
    denom = jnp.sqrt(a**2 * term1**2 + b**2 * term2**2 + eps)
    
    # Velocity components from Richards' equations
    x_t = (b**2 * cos_th * term2 - a**2 * sin_th * term1) / denom + c * sin_th
    y_t = (-b**2 * sin_th * term2 - a**2 * cos_th * term1) / denom + c * cos_th
    
    return x_t, y_t


# =============================================================================
# Perimeter Evolution (Differentiable)
# =============================================================================


def evolve_perimeter_step_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evolve fire perimeter by one time step (Euler integration).
    
    Differentiable version without marker method (for simplicity).
    
    Parameters
    ----------
    x : jnp.ndarray
        Current X coordinates of perimeter vertices.
    y : jnp.ndarray
        Current Y coordinates of perimeter vertices.
    ros : jnp.ndarray
        Head fire rate of spread at each vertex.
    bros : jnp.ndarray
        Back fire rate of spread at each vertex.
    fros : jnp.ndarray
        Flank fire rate of spread at each vertex.
    raz : jnp.ndarray
        Rate of spread azimuth at each vertex (radians).
    dt : float
        Time step (same units as ROS).
        
    Returns
    -------
    x_new : jnp.ndarray
        Updated X coordinates.
    y_new : jnp.ndarray
        Updated Y coordinates.
    """
    # Convert ROS to ellipse parameters
    a, b, c, theta = ros_to_ellipse_params_jax(ros, bros, fros, raz)
    
    # Compute velocities
    x_t, y_t = richards_velocity_jax(x, y, a, b, c, theta)
    
    # Explicit Euler update
    x_new = x + dt * x_t
    y_new = y + dt * y_t
    
    return x_new, y_new


def create_initial_perimeter_jax(
    x_center: float,
    y_center: float,
    radius: float,
    n_vertices: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create an initial circular fire perimeter.
    
    Parameters
    ----------
    x_center : float
        X coordinate of ignition point.
    y_center : float
        Y coordinate of ignition point.
    radius : float
        Initial fire radius.
    n_vertices : int
        Number of vertices on perimeter.
        
    Returns
    -------
    x : jnp.ndarray
        X coordinates of vertices.
    y : jnp.ndarray
        Y coordinates of vertices.
    """
    theta = jnp.linspace(0, 2 * jnp.pi, n_vertices, endpoint=False)
    x = x_center + radius * jnp.cos(theta)
    y = y_center + radius * jnp.sin(theta)
    
    return x, y


# =============================================================================
# Full Simulation (Differentiable)
# =============================================================================


class SimulationState(NamedTuple):
    """State during fire simulation."""
    x: jnp.ndarray
    y: jnp.ndarray
    step: int


def simulate_fire_jax(
    fire_params: FireParamsJAX,
    x_ignition: float,
    y_ignition: float,
    dt: float = 1.0,
    n_vertices: int = 200,
    initial_radius: float = 0.5,
    n_steps: int = 60,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate fire spread from ignition point.
    
    Differentiable simulation using lax.fori_loop.
    
    Parameters
    ----------
    fire_params : FireParamsJAX
        Gridded fire behavior parameters.
    x_ignition : float
        X coordinate of ignition point.
    y_ignition : float
        Y coordinate of ignition point.
    dt : float
        Time step in minutes.
    n_vertices : int
        Number of vertices on perimeter.
    initial_radius : float
        Initial fire radius in meters.
    n_steps : int
        Number of time steps to simulate.
        
    Returns
    -------
    x_final : jnp.ndarray
        Final X coordinates of perimeter.
    y_final : jnp.ndarray
        Final Y coordinates of perimeter.
    """
    # Initialize perimeter
    x, y = create_initial_perimeter_jax(x_ignition, y_ignition, initial_radius, n_vertices)
    
    def step_fn(i: int, state: tuple) -> tuple:
        x, y = state
        
        # Sample parameters at current vertex positions
        ros, bros, fros, raz = sample_fire_params_jax(fire_params, i, x, y)
        
        # Evolve perimeter
        x_new, y_new = evolve_perimeter_step_jax(x, y, ros, bros, fros, raz, dt)
        
        return (x_new, y_new)
    
    # Run simulation using fori_loop (JIT-compatible)
    x_final, y_final = lax.fori_loop(0, n_steps, step_fn, (x, y))
    
    return x_final, y_final


def compute_fire_area_jax(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute area enclosed by perimeter using Shoelace formula.
    
    Differentiable area calculation.
    
    Parameters
    ----------
    x : jnp.ndarray
        X coordinates of perimeter vertices.
    y : jnp.ndarray
        Y coordinates of perimeter vertices.
        
    Returns
    -------
    jnp.ndarray
        Area enclosed by perimeter.
    """
    # Shoelace formula
    x_roll = jnp.roll(x, -1)
    y_roll = jnp.roll(y, -1)
    area = 0.5 * jnp.abs(jnp.sum(x * y_roll - x_roll * y))
    
    return area


# =============================================================================
# Gradient-Friendly Loss Functions
# =============================================================================


def area_loss_jax(
    predicted_area: jnp.ndarray,
    observed_area: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute area-based loss for calibration.
    
    Uses relative squared error for scale-invariance.
    
    Parameters
    ----------
    predicted_area : jnp.ndarray
        Predicted fire area.
    observed_area : jnp.ndarray
        Observed fire area.
        
    Returns
    -------
    jnp.ndarray
        Loss value.
    """
    # Relative squared error
    relative_error = (predicted_area - observed_area) / (observed_area + 1e-6)
    return relative_error ** 2


def perimeter_iou_loss_jax(
    x_pred: jnp.ndarray,
    y_pred: jnp.ndarray,
    x_obs: jnp.ndarray,
    y_obs: jnp.ndarray,
    grid_resolution: float = 10.0,
    grid_size: int = 100,
) -> jnp.ndarray:
    """
    Compute IoU-based loss using soft rasterization.
    
    This is a differentiable approximation to Intersection over Union
    using soft membership functions on a grid.
    
    Parameters
    ----------
    x_pred : jnp.ndarray
        Predicted perimeter X coordinates.
    y_pred : jnp.ndarray
        Predicted perimeter Y coordinates.
    x_obs : jnp.ndarray
        Observed perimeter X coordinates.
    y_obs : jnp.ndarray
        Observed perimeter Y coordinates.
    grid_resolution : float
        Grid cell size.
    grid_size : int
        Number of grid cells in each dimension.
        
    Returns
    -------
    jnp.ndarray
        1 - IoU (so minimizing gives better overlap).
    """
    # Create grid
    x_min = jnp.minimum(jnp.min(x_pred), jnp.min(x_obs)) - grid_resolution
    y_min = jnp.minimum(jnp.min(y_pred), jnp.min(y_obs)) - grid_resolution
    
    grid_x = jnp.linspace(x_min, x_min + grid_size * grid_resolution, grid_size)
    grid_y = jnp.linspace(y_min, y_min + grid_size * grid_resolution, grid_size)
    
    gx, gy = jnp.meshgrid(grid_x, grid_y)
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()
    
    # Soft membership using signed distance (simplified)
    # This is an approximation - full SDF computation would be more accurate
    
    def soft_inside(px, py, x_perim, y_perim, sigma=1.0):
        """Soft indicator for points inside perimeter."""
        # Compute centroid
        cx = jnp.mean(x_perim)
        cy = jnp.mean(y_perim)
        
        # Distance from centroid (normalized)
        d_point = jnp.sqrt((px - cx)**2 + (py - cy)**2)
        
        # Average perimeter distance from centroid
        d_perim = jnp.mean(jnp.sqrt((x_perim - cx)**2 + (y_perim - cy)**2))
        
        # Soft membership
        return jax.nn.sigmoid(sigma * (d_perim - d_point))
    
    # Compute soft memberships
    sigma = 0.5  # Controls softness of boundary
    
    # Vectorize over grid points
    inside_pred = jax.vmap(lambda px, py: soft_inside(px, py, x_pred, y_pred, sigma))(gx_flat, gy_flat)
    inside_obs = jax.vmap(lambda px, py: soft_inside(px, py, x_obs, y_obs, sigma))(gx_flat, gy_flat)
    
    # Soft IoU
    intersection = jnp.sum(inside_pred * inside_obs)
    union = jnp.sum(inside_pred + inside_obs - inside_pred * inside_obs)
    
    iou = intersection / (union + 1e-6)
    
    return 1.0 - iou


def combined_loss_jax(
    x_pred: jnp.ndarray,
    y_pred: jnp.ndarray,
    observed_area: jnp.ndarray,
    area_weight: float = 1.0,
) -> jnp.ndarray:
    """
    Combined loss function for calibration.
    
    Parameters
    ----------
    x_pred : jnp.ndarray
        Predicted perimeter X coordinates.
    y_pred : jnp.ndarray
        Predicted perimeter Y coordinates.
    observed_area : jnp.ndarray
        Observed fire area.
    area_weight : float
        Weight for area loss component.
        
    Returns
    -------
    jnp.ndarray
        Total loss value.
    """
    predicted_area = compute_fire_area_jax(x_pred, y_pred)
    loss = area_weight * area_loss_jax(predicted_area, observed_area)
    
    return loss
