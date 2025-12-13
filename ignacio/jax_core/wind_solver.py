"""
Mass-Conserving Diagnostic Wind Solver.

Computes terrain-adjusted wind fields that respect mass conservation
(divergence-free) while following topography. Similar to WindNinja but
implemented in JAX for GPU acceleration and differentiability.

Physics:
- Wind accelerates over ridges (Venturi effect)
- Wind slows in valleys and lee sides
- Wind diverts around obstacles
- Mass must be conserved: ∇·(ρu) = 0

Method:
1. Initialize with background wind (interpolated from stations/ERA5)
2. Project to terrain-following coordinates
3. Solve Poisson equation for adjustment potential
4. Apply corrections to make field divergence-free

This transforms uniform "weather station" wind into realistic
terrain-adjusted wind that bends around topography.

References:
- Forthofer, J.M. et al. (2014). A comparison of three approaches for
  simulating fine-scale surface winds in support of wildland fire management.
- Wagenbrenner, N.S. et al. (2016). Downscaling surface wind predictions
  from numerical weather prediction models in complex terrain with WindNinja.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import ndimage as jnd
import numpy as np


class WindField(NamedTuple):
    """Wind field on terrain grid."""
    u: jnp.ndarray      # East-west component (m/s), positive = eastward
    v: jnp.ndarray      # North-south component (m/s), positive = northward
    speed: jnp.ndarray  # Wind speed (m/s)
    direction: jnp.ndarray  # Wind direction (degrees, direction FROM)


class WindSolverParams(NamedTuple):
    """Parameters for wind solver."""
    
    # Number of Jacobi iterations for Poisson solver
    n_iterations: int = 100
    
    # Relaxation parameter (0.5-1.0, higher = faster but less stable)
    relaxation: float = 0.8
    
    # Convergence tolerance
    tolerance: float = 1e-6
    
    # Height above ground for wind field (m)
    wind_height: float = 10.0
    
    # Terrain influence decay height (m)
    # Wind adjustments decay exponentially above this height
    terrain_influence_height: float = 100.0
    
    # Speed-up factor on ridges (multiplier)
    ridge_speedup: float = 1.3
    
    # Slow-down factor in valleys (multiplier)
    valley_slowdown: float = 0.7
    
    # Diverting strength (how much wind bends around terrain)
    diverting_strength: float = 1.0


def compute_terrain_parameters(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute terrain parameters needed for wind adjustment.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid (m)
    dx, dy : float
        Grid spacing (m)
        
    Returns
    -------
    slope : jnp.ndarray
        Terrain slope (radians)
    aspect : jnp.ndarray
        Terrain aspect (radians, 0=N, π/2=E)
    curvature : jnp.ndarray
        Terrain curvature (positive = ridge, negative = valley)
    tpi : jnp.ndarray
        Topographic Position Index (elevation relative to neighbors)
    """
    # Compute gradients
    dzdx = jnp.zeros_like(dem)
    dzdy = jnp.zeros_like(dem)
    
    # Central differences
    dzdx = dzdx.at[:, 1:-1].set((dem[:, 2:] - dem[:, :-2]) / (2 * dx))
    dzdy = dzdy.at[1:-1, :].set((dem[2:, :] - dem[:-2, :]) / (2 * dy))
    
    # Boundary handling
    dzdx = dzdx.at[:, 0].set((dem[:, 1] - dem[:, 0]) / dx)
    dzdx = dzdx.at[:, -1].set((dem[:, -1] - dem[:, -2]) / dx)
    dzdy = dzdy.at[0, :].set((dem[1, :] - dem[0, :]) / dy)
    dzdy = dzdy.at[-1, :].set((dem[-1, :] - dem[-2, :]) / dy)
    
    # Slope magnitude
    slope = jnp.arctan(jnp.sqrt(dzdx**2 + dzdy**2))
    
    # Aspect (direction of steepest descent)
    aspect = jnp.arctan2(-dzdx, -dzdy)  # 0 = N, π/2 = E
    
    # Curvature (Laplacian of elevation)
    d2zdx2 = jnp.zeros_like(dem)
    d2zdy2 = jnp.zeros_like(dem)
    
    d2zdx2 = d2zdx2.at[:, 1:-1].set(
        (dem[:, 2:] - 2*dem[:, 1:-1] + dem[:, :-2]) / (dx**2)
    )
    d2zdy2 = d2zdy2.at[1:-1, :].set(
        (dem[2:, :] - 2*dem[1:-1, :] + dem[:-2, :]) / (dy**2)
    )
    
    curvature = d2zdx2 + d2zdy2
    
    # Topographic Position Index
    # Compare elevation to mean of neighbors
    kernel = jnp.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=dem.dtype) / 8.0
    
    # Convolve to get mean neighbor elevation
    dem_4d = dem[None, :, :, None]
    kernel_4d = kernel[:, :, None, None]
    
    mean_neighbors = jax.lax.conv_general_dilated(
        dem_4d, kernel_4d,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )[0, :, :, 0]
    
    tpi = dem - mean_neighbors
    
    return slope, aspect, curvature, tpi


def initialize_wind_field(
    shape: Tuple[int, int],
    wind_speed: float,
    wind_direction: float,
) -> WindField:
    """
    Initialize uniform background wind field.
    
    Parameters
    ----------
    shape : tuple
        Grid shape (ny, nx)
    wind_speed : float
        Wind speed (m/s or km/h - specify in calling code)
    wind_direction : float
        Wind direction (degrees, direction wind blows FROM)
        
    Returns
    -------
    wind : WindField
        Uniform wind field
    """
    ny, nx = shape
    
    # Convert direction to radians (FROM direction)
    dir_rad = jnp.radians(wind_direction)
    
    # Wind components (direction wind blows TO is opposite of FROM)
    # u = eastward component, v = northward component
    u = -wind_speed * jnp.sin(dir_rad)  # Negative because FROM
    v = -wind_speed * jnp.cos(dir_rad)
    
    u_grid = jnp.full((ny, nx), u, dtype=jnp.float32)
    v_grid = jnp.full((ny, nx), v, dtype=jnp.float32)
    speed_grid = jnp.full((ny, nx), wind_speed, dtype=jnp.float32)
    dir_grid = jnp.full((ny, nx), wind_direction, dtype=jnp.float32)
    
    return WindField(u=u_grid, v=v_grid, speed=speed_grid, direction=dir_grid)


def compute_speed_adjustment(
    tpi: jnp.ndarray,
    curvature: jnp.ndarray,
    slope: jnp.ndarray,
    aspect: jnp.ndarray,
    wind_direction: float,
    params: WindSolverParams = WindSolverParams(),
) -> jnp.ndarray:
    """
    Compute wind speed adjustment factor based on terrain.
    
    - Ridges (positive TPI) speed up wind
    - Valleys (negative TPI) slow down wind
    - Windward slopes speed up, leeward slopes slow down
    
    Parameters
    ----------
    tpi : jnp.ndarray
        Topographic Position Index
    curvature : jnp.ndarray
        Terrain curvature
    slope : jnp.ndarray
        Terrain slope (radians)
    aspect : jnp.ndarray
        Terrain aspect (radians)
    wind_direction : float
        Background wind direction (degrees FROM)
    params : WindSolverParams
        Solver parameters
        
    Returns
    -------
    speed_factor : jnp.ndarray
        Multiplicative adjustment factor (>1 = speedup, <1 = slowdown)
    """
    # Normalize TPI to reasonable range
    tpi_normalized = tpi / (jnp.std(tpi) + 1e-6)
    tpi_normalized = jnp.clip(tpi_normalized, -3, 3)
    
    # Ridge/valley effect
    # Positive TPI = ridge = speedup
    # Negative TPI = valley = slowdown
    ridge_valley_factor = 1.0 + 0.1 * tpi_normalized
    
    # Windward/leeward effect
    # Wind direction is FROM, aspect is DOWN-slope direction
    wind_rad = jnp.radians(wind_direction)
    
    # Angle difference between wind direction and aspect
    # If wind is hitting slope head-on (windward), speed up
    # If wind is in lee, slow down
    angle_diff = jnp.abs(jnp.cos(wind_rad - aspect))
    slope_exposure = jnp.sin(slope) * angle_diff
    
    # Windward = wind hitting upslope
    # Check if wind is going upslope (aspect points downslope)
    upslope_component = -jnp.sin(wind_rad) * jnp.sin(aspect) - jnp.cos(wind_rad) * jnp.cos(aspect)
    windward_factor = 1.0 + 0.15 * slope_exposure * jnp.sign(upslope_component)
    
    # Combine factors
    speed_factor = ridge_valley_factor * windward_factor
    
    # Apply limits
    speed_factor = jnp.clip(
        speed_factor,
        params.valley_slowdown,
        params.ridge_speedup
    )
    
    return speed_factor


def compute_divergence(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dx: float,
    dy: float,
) -> jnp.ndarray:
    """
    Compute divergence of wind field.
    
    div(u,v) = ∂u/∂x + ∂v/∂y
    
    For mass conservation, divergence should be zero.
    """
    dudx = jnp.zeros_like(u)
    dvdy = jnp.zeros_like(v)
    
    # Central differences
    dudx = dudx.at[:, 1:-1].set((u[:, 2:] - u[:, :-2]) / (2 * dx))
    dvdy = dvdy.at[1:-1, :].set((v[2:, :] - v[:-2, :]) / (2 * dy))
    
    return dudx + dvdy


def solve_poisson_jacobi(
    rhs: jnp.ndarray,
    dx: float,
    dy: float,
    n_iterations: int = 100,
    relaxation: float = 0.8,
) -> jnp.ndarray:
    """
    Solve Poisson equation using Jacobi iteration.
    
    ∇²φ = rhs
    
    This finds the potential φ whose gradient will make the
    wind field divergence-free.
    
    Parameters
    ----------
    rhs : jnp.ndarray
        Right-hand side (divergence of initial field)
    dx, dy : float
        Grid spacing
    n_iterations : int
        Number of iterations
    relaxation : float
        Relaxation parameter
        
    Returns
    -------
    phi : jnp.ndarray
        Solution potential
    """
    ny, nx = rhs.shape
    phi = jnp.zeros_like(rhs)
    
    # Precompute coefficients
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (1.0/dx2 + 1.0/dy2)
    
    def jacobi_step(phi, _):
        # Interior update
        phi_new = jnp.zeros_like(phi)
        
        phi_xp = jnp.roll(phi, -1, axis=1)  # phi[i+1,j]
        phi_xm = jnp.roll(phi, 1, axis=1)   # phi[i-1,j]
        phi_yp = jnp.roll(phi, -1, axis=0)  # phi[i,j+1]
        phi_ym = jnp.roll(phi, 1, axis=0)   # phi[i,j-1]
        
        phi_new = ((phi_xp + phi_xm) / dx2 + 
                   (phi_yp + phi_ym) / dy2 - rhs) / denom
        
        # Apply relaxation
        phi_new = phi + relaxation * (phi_new - phi)
        
        # Boundary conditions (Neumann: zero gradient)
        phi_new = phi_new.at[0, :].set(phi_new[1, :])
        phi_new = phi_new.at[-1, :].set(phi_new[-2, :])
        phi_new = phi_new.at[:, 0].set(phi_new[:, 1])
        phi_new = phi_new.at[:, -1].set(phi_new[:, -2])
        
        return phi_new, None
    
    phi, _ = lax.scan(jacobi_step, phi, None, length=n_iterations)
    
    return phi


def solve_poisson_fft(
    rhs: jnp.ndarray,
    dx: float,
    dy: float,
) -> jnp.ndarray:
    """
    Solve Poisson equation using FFT (faster for large grids).
    
    ∇²φ = rhs
    
    Assumes periodic boundary conditions (less accurate at edges).
    """
    ny, nx = rhs.shape
    
    # Wavenumbers
    kx = jnp.fft.fftfreq(nx, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, dy) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky)
    
    # Laplacian in Fourier space: -k²
    k_squared = KX**2 + KY**2
    k_squared = k_squared.at[0, 0].set(1.0)  # Avoid division by zero
    
    # Transform RHS
    rhs_hat = jnp.fft.fft2(rhs)
    
    # Solve in Fourier space
    phi_hat = -rhs_hat / k_squared
    phi_hat = phi_hat.at[0, 0].set(0.0)  # Set mean to zero
    
    # Transform back
    phi = jnp.real(jnp.fft.ifft2(phi_hat))
    
    return phi


def apply_mass_conservation(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dx: float,
    dy: float,
    params: WindSolverParams = WindSolverParams(),
    use_fft: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Make wind field mass-conserving (divergence-free).
    
    Uses Helmholtz decomposition: any field can be split into
    divergence-free and curl-free parts. We subtract the curl-free
    (divergent) part.
    
    Parameters
    ----------
    u, v : jnp.ndarray
        Wind components
    dx, dy : float
        Grid spacing
    params : WindSolverParams
        Solver parameters
    use_fft : bool
        Use FFT solver (faster) or Jacobi (more stable at boundaries)
        
    Returns
    -------
    u_corrected, v_corrected : jnp.ndarray
        Divergence-free wind components
    """
    # Compute divergence
    div = compute_divergence(u, v, dx, dy)
    
    # Solve Poisson equation for correction potential
    if use_fft:
        phi = solve_poisson_fft(div, dx, dy)
    else:
        phi = solve_poisson_jacobi(div, dx, dy, params.n_iterations, params.relaxation)
    
    # Compute gradient of potential
    dphidx = jnp.zeros_like(phi)
    dphidy = jnp.zeros_like(phi)
    
    dphidx = dphidx.at[:, 1:-1].set((phi[:, 2:] - phi[:, :-2]) / (2 * dx))
    dphidy = dphidy.at[1:-1, :].set((phi[2:, :] - phi[:-2, :]) / (2 * dy))
    
    # Boundary handling
    dphidx = dphidx.at[:, 0].set((phi[:, 1] - phi[:, 0]) / dx)
    dphidx = dphidx.at[:, -1].set((phi[:, -1] - phi[:, -2]) / dx)
    dphidy = dphidy.at[0, :].set((phi[1, :] - phi[0, :]) / dy)
    dphidy = dphidy.at[-1, :].set((phi[-1, :] - phi[-2, :]) / dy)
    
    # Subtract gradient to make divergence-free
    u_corrected = u - dphidx
    v_corrected = v - dphidy
    
    return u_corrected, v_corrected


def compute_terrain_deflection(
    u: jnp.ndarray,
    v: jnp.ndarray,
    slope: jnp.ndarray,
    aspect: jnp.ndarray,
    params: WindSolverParams = WindSolverParams(),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deflect wind around steep terrain.
    
    Wind hitting a steep slope deflects laterally rather than
    going straight over.
    
    Parameters
    ----------
    u, v : jnp.ndarray
        Wind components
    slope : jnp.ndarray
        Terrain slope (radians)
    aspect : jnp.ndarray
        Terrain aspect (radians)
    params : WindSolverParams
        Solver parameters
        
    Returns
    -------
    u_deflected, v_deflected : jnp.ndarray
        Deflected wind components
    """
    # Wind direction
    wind_dir = jnp.arctan2(u, v)
    wind_speed = jnp.sqrt(u**2 + v**2)
    
    # Angle between wind and slope
    # aspect points downslope, so wind hitting upslope has angle diff ~π
    angle_to_slope = wind_dir - aspect
    
    # Component of wind hitting slope
    hitting_component = jnp.abs(jnp.sin(angle_to_slope))
    
    # Deflection magnitude (more deflection for steeper slopes)
    deflection_strength = params.diverting_strength * jnp.sin(slope) * hitting_component
    
    # Deflection direction (perpendicular to slope)
    # Deflect to the right (could be either way)
    deflection_angle = aspect + jnp.pi/2
    
    # Apply deflection
    u_deflected = u + deflection_strength * wind_speed * jnp.sin(deflection_angle)
    v_deflected = v + deflection_strength * wind_speed * jnp.cos(deflection_angle)
    
    return u_deflected, v_deflected


def solve_wind_field(
    dem: jnp.ndarray,
    dx: float,
    dy: float,
    background_speed: float,
    background_direction: float,
    params: WindSolverParams = WindSolverParams(),
) -> WindField:
    """
    Compute terrain-adjusted wind field.
    
    This is the main entry point for the wind solver.
    
    Parameters
    ----------
    dem : jnp.ndarray
        Elevation grid (m)
    dx, dy : float
        Grid spacing (m)
    background_speed : float
        Background wind speed (m/s)
    background_direction : float
        Background wind direction (degrees, direction FROM)
    params : WindSolverParams
        Solver parameters
        
    Returns
    -------
    wind : WindField
        Terrain-adjusted wind field
    """
    ny, nx = dem.shape
    
    # Compute terrain parameters
    slope, aspect, curvature, tpi = compute_terrain_parameters(dem, dx, dy)
    
    # Initialize with background wind
    wind = initialize_wind_field((ny, nx), background_speed, background_direction)
    u, v = wind.u, wind.v
    
    # Apply speed adjustment based on terrain
    speed_factor = compute_speed_adjustment(
        tpi, curvature, slope, aspect, background_direction, params
    )
    u = u * speed_factor
    v = v * speed_factor
    
    # Apply terrain deflection
    u, v = compute_terrain_deflection(u, v, slope, aspect, params)
    
    # Make field mass-conserving
    u, v = apply_mass_conservation(u, v, dx, dy, params)
    
    # Compute final speed and direction
    speed = jnp.sqrt(u**2 + v**2)
    direction = jnp.degrees(jnp.arctan2(-u, -v)) % 360  # Convert back to FROM
    
    return WindField(u=u, v=v, speed=speed, direction=direction)


def interpolate_wind_height(
    wind_surface: WindField,
    dem: jnp.ndarray,
    target_height: float,
    roughness_length: float = 0.1,
) -> WindField:
    """
    Interpolate wind from surface to specified height above ground.
    
    Uses logarithmic wind profile.
    
    Parameters
    ----------
    wind_surface : WindField
        Wind at surface (reference height)
    dem : jnp.ndarray
        Elevation grid
    target_height : float
        Height above ground (m)
    roughness_length : float
        Surface roughness length (m). ~0.1 for grassland, ~1.0 for forest
        
    Returns
    -------
    wind_height : WindField
        Wind at target height
    """
    # Logarithmic wind profile
    # u(z) = u* / k * ln(z / z0)
    # Ratio: u(z2) / u(z1) = ln(z2/z0) / ln(z1/z0)
    
    reference_height = 10.0  # Assume surface wind is at 10m
    
    ratio = (jnp.log(target_height / roughness_length) / 
             jnp.log(reference_height / roughness_length))
    
    return WindField(
        u=wind_surface.u * ratio,
        v=wind_surface.v * ratio,
        speed=wind_surface.speed * ratio,
        direction=wind_surface.direction,  # Direction unchanged
    )


# =============================================================================
# Convenience functions for fire simulation
# =============================================================================

def get_wind_at_cell(
    wind: WindField,
    i: int,
    j: int,
) -> Tuple[float, float, float]:
    """
    Get wind speed and direction at a specific cell.
    
    Returns
    -------
    speed, direction, (u, v) components
    """
    return (float(wind.speed[i, j]), 
            float(wind.direction[i, j]),
            float(wind.u[i, j]),
            float(wind.v[i, j]))


def wind_field_to_ros_direction(
    wind: WindField,
) -> jnp.ndarray:
    """
    Convert wind field to fire spread direction.
    
    Fire spreads in the direction wind blows TO
    (opposite of meteorological direction FROM).
    
    Returns
    -------
    raz : jnp.ndarray
        Rate-of-spread azimuth (degrees, direction fire spreads TO)
    """
    # Wind direction is FROM, fire spreads TO (opposite)
    raz = (wind.direction + 180.0) % 360.0
    return raz
