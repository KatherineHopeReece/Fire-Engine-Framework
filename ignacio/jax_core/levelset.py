"""
Level-Set Fire Spread Model.

Implements fire spread using the level-set method, which naturally handles
topology changes (merging fronts, burning around obstacles) and is fully
differentiable through JAX.

The fire is represented as a signed distance field φ where:
- φ < 0: burned area
- φ > 0: unburned area  
- φ = 0: fire front

The front evolves according to the Hamilton-Jacobi equation:
    ∂φ/∂t + F(x,y,n)|∇φ| = 0

where F is the speed function (derived from ROS) and n is the front normal.

References:
- Osher & Sethian (1988): Fronts propagating with curvature-dependent speed
- Mallet et al. (2009): Level set methods for fire simulation
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

# Numerical stability
EPS = 1e-8


class LevelSetGrids(NamedTuple):
    """
    Grid data for level-set fire simulation.
    
    Attributes
    ----------
    x_coords : jnp.ndarray
        1D array of x coordinates (nx,)
    y_coords : jnp.ndarray
        1D array of y coordinates (ny,)
    ros : jnp.ndarray
        Head fire rate of spread (nt, ny, nx) or (ny, nx)
    bros : jnp.ndarray
        Back fire rate of spread
    fros : jnp.ndarray
        Flank fire rate of spread
    raz : jnp.ndarray
        Spread azimuth in radians (direction fire spreads TO)
    """
    x_coords: jnp.ndarray
    y_coords: jnp.ndarray
    ros: jnp.ndarray
    bros: jnp.ndarray
    fros: jnp.ndarray
    raz: jnp.ndarray


def initialize_phi(
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    initial_radius: float,
) -> jnp.ndarray:
    """
    Initialize signed distance field for circular ignition.
    
    Parameters
    ----------
    x_coords, y_coords : jnp.ndarray
        1D coordinate arrays
    x_ign, y_ign : float
        Ignition point coordinates
    initial_radius : float
        Initial fire radius. If this is smaller than the grid spacing,
        it will be automatically increased to ensure at least one burned cell.
        
    Returns
    -------
    phi : jnp.ndarray
        Signed distance field (ny, nx)
        Negative inside fire, positive outside
    """
    # Create 2D coordinate grids
    X, Y = jnp.meshgrid(x_coords, y_coords)
    
    # Ensure initial radius is at least as large as the grid diagonal
    # so we have at least one burned cell
    dx = jnp.abs(x_coords[1] - x_coords[0])
    dy = jnp.abs(y_coords[1] - y_coords[0])
    min_radius = jnp.sqrt(dx**2 + dy**2)  # Grid cell diagonal
    effective_radius = jnp.maximum(initial_radius, min_radius)
    
    # Signed distance from ignition point
    # Negative inside the initial fire circle
    dist = jnp.sqrt((X - x_ign)**2 + (Y - y_ign)**2)
    phi = dist - effective_radius
    
    return phi


def compute_gradient(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    y_ascending: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute gradient of phi using central differences.
    
    Parameters
    ----------
    phi : jnp.ndarray
        2D field (ny, nx)
    dx, dy : float
        Grid spacing (should be positive magnitudes)
    y_ascending : bool
        If True, y increases with row index (typical math convention)
        If False, y decreases with row index (typical raster convention)
    
    Returns
    -------
    phi_x, phi_y : jnp.ndarray
        Gradient components in geographic coordinates
    """
    # Central differences
    # axis 1 = x direction (columns)
    # axis 0 = y direction (rows)
    phi_x = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2 * dx)
    phi_y = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2 * dy)
    
    # If y is descending (row 0 = north, row n = south), flip the y gradient
    # because (phi[row+1] - phi[row-1]) computes (phi_south - phi_north)
    # but we want ∂phi/∂y where y increases going north
    phi_y = jnp.where(y_ascending, phi_y, -phi_y)
    
    # Zero out boundary gradients (Neumann BC)
    phi_x = phi_x.at[:, 0].set(0)
    phi_x = phi_x.at[:, -1].set(0)
    phi_y = phi_y.at[0, :].set(0)
    phi_y = phi_y.at[-1, :].set(0)
    
    return phi_x, phi_y


def compute_gradient_magnitude_upwind(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    speed: jnp.ndarray,
    y_ascending: bool = True,
) -> jnp.ndarray:
    """
    Compute |∇φ| using Godunov upwind scheme.
    
    This is essential for stability of the level-set evolution.
    For fire spread (speed > 0, front moves outward), we use:
    |∇φ|² = max(D⁻x, 0)² + min(D⁺x, 0)² + max(D⁻y, 0)² + min(D⁺y, 0)²
    
    Parameters
    ----------
    y_ascending : bool
        If True, y increases with row index
        If False, y decreases with row index
    """
    # One-sided differences in ARRAY index space
    # D⁻ = backward (toward smaller index), D⁺ = forward (toward larger index)
    
    # X direction (columns) - same regardless of coordinate convention
    Dx_minus = (phi - jnp.roll(phi, 1, axis=1)) / dx  # phi[j] - phi[j-1]
    Dx_plus = (jnp.roll(phi, -1, axis=1) - phi) / dx   # phi[j+1] - phi[j]
    
    # Y direction (rows)
    # phi[i] - phi[i-1] and phi[i+1] - phi[i]
    Dy_minus_idx = (phi - jnp.roll(phi, 1, axis=0)) / dy  # toward smaller row
    Dy_plus_idx = (jnp.roll(phi, -1, axis=0) - phi) / dy   # toward larger row
    
    # For y_ascending=False, geographic y decreases with row index
    # So "toward smaller y" = "toward larger row index" = Dy_plus_idx
    # We need to flip the interpretation
    Dy_minus = jnp.where(y_ascending, Dy_minus_idx, -Dy_plus_idx)
    Dy_plus = jnp.where(y_ascending, Dy_plus_idx, -Dy_minus_idx)
    
    # Godunov scheme for F > 0 (front expanding outward)
    # Select upwind differences: information flows from burned to unburned
    grad_mag_sq = (
        jnp.maximum(Dx_minus, 0)**2 + jnp.minimum(Dx_plus, 0)**2 +
        jnp.maximum(Dy_minus, 0)**2 + jnp.minimum(Dy_plus, 0)**2
    )
    
    return jnp.sqrt(grad_mag_sq + EPS)


def compute_elliptical_speed(
    phi_x: jnp.ndarray,
    phi_y: jnp.ndarray,
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute speed in the normal direction for elliptical fire spread.
    
    Uses direct angular interpolation between ROS, BROS, and FROS.
    
    Parameters
    ----------
    phi_x, phi_y : jnp.ndarray
        Gradient components (proportional to outward normal)
    ros, bros, fros : jnp.ndarray
        Rate of spread components (head, back, flank)
    raz : jnp.ndarray
        Spread direction (radians, from north clockwise)
        
    Returns
    -------
    speed : jnp.ndarray
        Speed in the normal direction at each grid point
    """
    # Gradient magnitude (outward normal direction)
    grad_mag = jnp.sqrt(phi_x**2 + phi_y**2 + EPS)
    
    # Unit normal (pointing outward from fire, into unburned area)
    nx = phi_x / grad_mag
    ny = phi_y / grad_mag
    
    # Convert normal to angle (from positive x-axis, counter-clockwise)
    # This is the direction the fire is spreading TO at this point
    normal_angle = jnp.arctan2(ny, nx)
    
    # RAZ is the direction fire spreads TO (from north, clockwise)
    # Convert to math convention (from east, counter-clockwise)
    # RAZ: 0=N, 90=E, 180=S, 270=W
    # Math: 0=E, 90=N, 180=W, 270=S
    spread_angle = jnp.pi/2 - raz
    
    # Angle between normal and spread direction
    # 0 = head fire direction, π = back fire direction, ±π/2 = flanks
    angle_diff = normal_angle - spread_angle
    
    # Normalize to [-π, π]
    angle_diff = jnp.arctan2(jnp.sin(angle_diff), jnp.cos(angle_diff))
    
    # Use cos²/sin² interpolation for smooth transition
    # This gives: θ=0 → ROS, θ=π → BROS, θ=±π/2 → weighted average
    cos_half = jnp.cos(angle_diff / 2)
    sin_half = jnp.sin(angle_diff / 2)
    
    # Interpolate between head and back fire
    # cos²(θ/2) gives 1 at θ=0, 0 at θ=π
    # sin²(θ/2) gives 0 at θ=0, 1 at θ=π
    head_back_speed = ros * cos_half**2 + bros * sin_half**2
    
    # Blend with flank speed based on |sin(θ)|
    # |sin(θ)| is 0 at head/back, 1 at flanks
    flank_weight = jnp.abs(jnp.sin(angle_diff))
    
    # Final speed: blend head/back with flank
    speed = head_back_speed * (1 - flank_weight) + fros * flank_weight
    
    # Ensure non-negative speed
    speed = jnp.maximum(speed, 0.0)
    
    return speed


def get_ros_at_time(
    grids: LevelSetGrids,
    t_idx: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract ROS grids at a specific time index."""
    def get_field(arr, t):
        if arr.ndim == 3:
            nt = arr.shape[0]
            t = jnp.clip(t, 0, nt - 1)
            return arr[t]
        return arr
    
    return (
        get_field(grids.ros, t_idx),
        get_field(grids.bros, t_idx),
        get_field(grids.fros, t_idx),
        get_field(grids.raz, t_idx),
    )


def evolve_phi(
    phi: jnp.ndarray,
    grids: LevelSetGrids,
    t_idx: int,
    dt: float,
    use_ellipse: bool = True,
) -> jnp.ndarray:
    """
    Evolve level-set field by one time step.
    
    Uses first-order upwind scheme with Godunov flux.
    
    ∂φ/∂t + F|∇φ| = 0
    
    Parameters
    ----------
    phi : jnp.ndarray
        Current signed distance field (ny, nx)
    grids : LevelSetGrids
        Fire behavior parameter grids
    t_idx : int
        Current time index (for time-varying grids)
    dt : float
        Time step
    use_ellipse : bool
        If True, use elliptical speed. If False, use isotropic ROS.
        
    Returns
    -------
    phi_new : jnp.ndarray
        Updated signed distance field
    """
    # Grid spacing
    dx = grids.x_coords[1] - grids.x_coords[0]
    dy = grids.y_coords[1] - grids.y_coords[0]
    
    # Detect if y is ascending or descending
    y_ascending = dy > 0
    
    # Use absolute values for spacing magnitude
    dx = jnp.abs(dx)
    dy = jnp.abs(dy)
    
    # Get ROS at current time
    ros, bros, fros, raz = get_ros_at_time(grids, t_idx)
    
    if use_ellipse:
        # Compute gradient for normal direction
        phi_x, phi_y = compute_gradient(phi, dx, dy, y_ascending)
        
        # Compute direction-dependent speed
        speed = compute_elliptical_speed(phi_x, phi_y, ros, bros, fros, raz)
    else:
        # Use isotropic speed (just ROS)
        speed = ros
    
    # Compute gradient magnitude using upwind scheme
    grad_mag = compute_gradient_magnitude_upwind(phi, dx, dy, speed, y_ascending)
    
    # Hamilton-Jacobi update: φ_new = φ - dt * F * |∇φ|
    phi_new = phi - dt * speed * grad_mag
    
    return phi_new


def simulate_fire_levelset(
    grids: LevelSetGrids,
    x_ign: float,
    y_ign: float,
    n_steps: int,
    dt: float = 1.0,
    initial_radius: float = 1.0,
    use_ellipse: bool = True,
) -> jnp.ndarray:
    """
    Simulate fire spread using level-set method.
    
    Parameters
    ----------
    grids : LevelSetGrids
        Fire behavior parameter grids
    x_ign, y_ign : float
        Ignition coordinates
    n_steps : int
        Number of time steps
    dt : float
        Time step (same units as ROS)
    initial_radius : float
        Initial fire radius (same units as coordinates)
    use_ellipse : bool
        If True, use elliptical spread. If False, use isotropic ROS.
        
    Returns
    -------
    phi_final : jnp.ndarray
        Final signed distance field
        Burned area where phi < 0
    """
    # Initialize
    phi = initialize_phi(grids.x_coords, grids.y_coords, x_ign, y_ign, initial_radius)
    
    def body_fn(i, phi):
        return evolve_phi(phi, grids, i, dt, use_ellipse)
    
    # Run simulation
    phi_final = lax.fori_loop(0, n_steps, body_fn, phi)
    
    return phi_final


def compute_burned_area(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
) -> jnp.ndarray:
    """
    Compute burned area from level-set field.
    
    Uses smooth Heaviside for differentiability.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Signed distance field (negative = burned)
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    area : jnp.ndarray
        Burned area
    """
    # Smooth Heaviside: H(-φ) gives 1 inside fire, 0 outside
    # Using sigmoid for differentiability
    # Scale by typical grid spacing for smooth transition
    epsilon = jnp.sqrt(dx**2 + dy**2)
    burned_fraction = jax.nn.sigmoid(-phi / epsilon)
    
    # Sum up burned cells
    cell_area = jnp.abs(dx * dy)
    area = jnp.sum(burned_fraction) * cell_area
    
    return area


def compute_burned_area_hard(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
) -> jnp.ndarray:
    """
    Compute burned area using hard threshold (not differentiable).
    
    More accurate for evaluation, but use compute_burned_area for training.
    """
    burned = phi < 0
    cell_area = jnp.abs(dx * dy)
    return jnp.sum(burned) * cell_area


# =============================================================================
# Convenience function matching perimeter-based API
# =============================================================================


def simulate_fire_levelset_with_area(
    grids: LevelSetGrids,
    x_ign: float,
    y_ign: float,
    n_steps: int,
    dt: float = 1.0,
    initial_radius: float = 1.0,
    differentiable: bool = True,
    use_ellipse: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate fire and return both phi field and burned area.
    
    Parameters
    ----------
    grids : LevelSetGrids
        Fire behavior parameter grids
    x_ign, y_ign : float
        Ignition coordinates
    n_steps : int
        Number of time steps
    dt : float
        Time step
    initial_radius : float
        Initial fire radius
    differentiable : bool
        If True, use smooth area calculation (for gradients)
        If False, use hard threshold (for evaluation)
    use_ellipse : bool
        If True, use elliptical spread. If False, use isotropic ROS.
        
    Returns
    -------
    phi : jnp.ndarray
        Final level-set field
    area : jnp.ndarray
        Burned area
    """
    phi = simulate_fire_levelset(grids, x_ign, y_ign, n_steps, dt, initial_radius, use_ellipse)
    
    dx = grids.x_coords[1] - grids.x_coords[0]
    dy = grids.y_coords[1] - grids.y_coords[0]
    
    if differentiable:
        area = compute_burned_area(phi, dx, dy)
    else:
        area = compute_burned_area_hard(phi, dx, dy)
    
    return phi, area


# =============================================================================
# Conversion utilities
# =============================================================================


def phi_to_perimeter(
    phi: jnp.ndarray,
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
    n_points: int = 200,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract perimeter contour from level-set field.
    
    Uses marching squares to find the zero level set.
    This is primarily for visualization - use compute_burned_area
    for area calculations.
    
    Note: This function is not differentiable.
    """
    import numpy as np
    from skimage import measure
    
    # Convert to numpy for skimage
    phi_np = np.array(phi)
    
    # Find contours at phi = 0
    contours = measure.find_contours(phi_np, 0.0)
    
    if len(contours) == 0:
        # No contour found - return empty
        return jnp.array([]), jnp.array([])
    
    # Get largest contour (main fire perimeter)
    largest = max(contours, key=len)
    
    # Convert from array indices to coordinates
    # Contours are in (row, col) format
    rows, cols = largest[:, 0], largest[:, 1]
    
    # Interpolate to coordinates
    x = np.interp(cols, np.arange(len(x_coords)), np.array(x_coords))
    y = np.interp(rows, np.arange(len(y_coords)), np.array(y_coords))
    
    # Resample to fixed number of points
    if len(x) > n_points:
        indices = np.linspace(0, len(x) - 1, n_points).astype(int)
        x = x[indices]
        y = y[indices]
    
    return jnp.array(x), jnp.array(y)
