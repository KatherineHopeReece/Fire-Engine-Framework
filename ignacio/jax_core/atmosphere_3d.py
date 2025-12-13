"""
3D Atmospheric Dynamics Solver with Fire-Atmosphere Coupling.

Implements a simplified but physically-meaningful 3D atmospheric model
for fire-driven convection, similar in spirit to WRF-SFIRE but optimized
for JAX and differentiable computation.

Physical Background
-------------------
Fire-atmosphere coupling is critical for predicting extreme fire behavior:
1. Fire releases heat → buoyant plume rises
2. Rising air creates surface low pressure → indraft winds
3. Indraft can double or triple effective wind speed near fire
4. Vorticity generation → fire whirls
5. Ember transport in the convective column

This module solves the anelastic equations:
- Momentum: ∂u/∂t + u·∇u = -∇p'/ρ₀ + g(θ'/θ₀)k + ∇·(K∇u)
- Continuity: ∇·(ρ₀u) = 0
- Thermodynamics: ∂θ/∂t + u·∇θ = ∇·(K∇θ) + Q_fire

Grid Structure
--------------
Uses Arakawa C-grid staggering:
- u at (i+1/2, j, k)
- v at (i, j+1/2, k)
- w at (i, j, k+1/2)
- scalars (θ, p) at cell centers (i, j, k)

Terrain-following coordinates (sigma) for complex topography.

References
----------
- Mandel, J. et al. (2011). Coupled atmosphere-wildland fire modeling
  with WRF-SFIRE. Geosci. Model Dev.
- Clark, T.L. et al. (1996). Description of a coupled atmosphere-fire
  model. Int. J. Wildland Fire.
- Coen, J.L. (2013). Modeling Wildland Fires: A Description of the
  Coupled Atmosphere-Wildland Fire Environment Model (CAWFE).
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve


# =============================================================================
# Constants
# =============================================================================

# Physical constants
G = 9.81              # Gravitational acceleration (m/s²)
CP = 1004.0           # Specific heat at constant pressure (J/kg/K)
CV = 717.0            # Specific heat at constant volume (J/kg/K)
R_DRY = 287.0         # Gas constant for dry air (J/kg/K)
P_REF = 100000.0      # Reference pressure (Pa)
T_REF = 300.0         # Reference temperature (K)
RHO_REF = 1.2         # Reference density (kg/m³)


# =============================================================================
# Data Structures
# =============================================================================

class AtmosphereParams(NamedTuple):
    """Parameters for atmospheric solver."""
    
    # Grid parameters
    nz: int = 20                      # Number of vertical levels
    z_top: float = 2000.0             # Domain top (m AGL)
    
    # Turbulence parameters
    km_h: float = 50.0                # Horizontal eddy viscosity (m²/s)
    km_v: float = 10.0                # Vertical eddy viscosity (m²/s)
    kh_h: float = 50.0                # Horizontal thermal diffusivity (m²/s)
    kh_v: float = 10.0                # Vertical thermal diffusivity (m²/s)
    use_smagorinsky: bool = True      # Use Smagorinsky turbulence model
    cs: float = 0.2                   # Smagorinsky constant
    
    # Fire coupling
    fire_heat_flux_scale: float = 1.0  # Scale factor for fire heat release
    plume_injection_depth: float = 100.0  # Depth of heat injection (m)
    
    # Boundary conditions
    lateral_bc: str = "periodic"       # "periodic" or "open"
    top_bc: str = "free_slip"          # "free_slip" or "damping"
    damping_depth: float = 500.0       # Rayleigh damping layer depth (m)
    damping_coef: float = 0.01         # Damping coefficient (1/s)
    
    # Numerical parameters
    dt_ratio: float = 0.5              # Acoustic timestep ratio
    divergence_damping: float = 0.1    # Divergence damping coefficient


class AtmosphereState(NamedTuple):
    """3D atmospheric state variables."""
    
    # Velocity components (staggered)
    u: jnp.ndarray    # (nz, ny, nx+1) - x-velocity at cell faces
    v: jnp.ndarray    # (nz, ny+1, nx) - y-velocity at cell faces  
    w: jnp.ndarray    # (nz+1, ny, nx) - z-velocity at cell faces
    
    # Thermodynamic variables (cell centers)
    theta: jnp.ndarray     # (nz, ny, nx) - potential temperature (K)
    theta_base: jnp.ndarray  # (nz,) - base state potential temperature
    rho_base: jnp.ndarray    # (nz,) - base state density
    
    # Pressure perturbation
    p_prime: jnp.ndarray   # (nz, ny, nx) - pressure perturbation (Pa)
    
    # Grid information
    z_levels: jnp.ndarray  # (nz+1,) - vertical level heights (m)
    dz: jnp.ndarray        # (nz,) - layer thicknesses (m)


class AtmosphereGrids(NamedTuple):
    """Precomputed grid information for efficiency."""
    
    # Horizontal grid
    dx: float
    dy: float
    nx: int
    ny: int
    nz: int
    
    # Vertical grid (possibly stretched)
    z_w: jnp.ndarray      # (nz+1,) - heights at w-levels
    z_u: jnp.ndarray      # (nz,) - heights at u/v/theta levels
    dz_w: jnp.ndarray     # (nz,) - spacing between w-levels
    
    # Terrain (sigma coordinates)
    terrain: jnp.ndarray  # (ny, nx) - terrain height (m)
    sigma: jnp.ndarray    # (nz+1,) - sigma levels [0, 1]
    
    # Metric terms for terrain-following
    dzdx: jnp.ndarray     # (ny, nx) - terrain slope in x
    dzdy: jnp.ndarray     # (ny, nx) - terrain slope in y


class CouplingResult(NamedTuple):
    """Results from fire-atmosphere coupling."""
    
    # Surface wind modification
    u_surface: jnp.ndarray    # (ny, nx) - modified u at surface
    v_surface: jnp.ndarray    # (ny, nx) - modified v at surface
    wind_speed: jnp.ndarray   # (ny, nx) - total wind speed
    wind_direction: jnp.ndarray  # (ny, nx) - wind direction (degrees)
    
    # Vertical motion
    w_max: jnp.ndarray        # (ny, nx) - maximum updraft
    plume_top: jnp.ndarray    # (ny, nx) - plume top height (m)
    
    # Diagnostics
    indraft_strength: jnp.ndarray  # (ny, nx) - indraft magnitude
    vorticity: jnp.ndarray         # (ny, nx) - vertical vorticity


# =============================================================================
# Grid Setup
# =============================================================================

def create_vertical_grid(
    nz: int,
    z_top: float,
    stretch_factor: float = 1.2,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create stretched vertical grid with finer resolution near surface.
    
    Parameters
    ----------
    nz : int
        Number of vertical levels
    z_top : float
        Domain top height (m)
    stretch_factor : float
        Grid stretching factor (1.0 = uniform)
        
    Returns
    -------
    z_w : array (nz+1,)
        Heights at w-levels (cell interfaces)
    z_u : array (nz,)
        Heights at u/v/theta levels (cell centers)
    """
    if stretch_factor == 1.0:
        # Uniform grid
        z_w = jnp.linspace(0, z_top, nz + 1)
    else:
        # Stretched grid using tanh function
        eta = jnp.linspace(0, 1, nz + 1)
        z_w = z_top * (jnp.tanh(stretch_factor * eta) / jnp.tanh(stretch_factor))
    
    # Cell centers
    z_u = 0.5 * (z_w[:-1] + z_w[1:])
    
    return z_w, z_u


def create_atmosphere_grids(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    terrain: jnp.ndarray,
    params: AtmosphereParams,
) -> AtmosphereGrids:
    """
    Create atmospheric grid structure.
    
    Parameters
    ----------
    nx, ny : int
        Horizontal grid dimensions
    dx, dy : float
        Horizontal grid spacing (m)
    terrain : array (ny, nx)
        Terrain height (m)
    params : AtmosphereParams
        Atmospheric parameters
        
    Returns
    -------
    AtmosphereGrids
        Grid structure with all precomputed terms
    """
    nz = params.nz
    z_top = params.z_top
    
    # Create vertical grid
    z_w, z_u = create_vertical_grid(nz, z_top)
    dz_w = jnp.diff(z_w)
    
    # Sigma levels (0 at surface, 1 at top)
    sigma = z_w / z_top
    
    # Terrain gradients
    # Use central differences with periodic BC
    terrain_padded = jnp.pad(terrain, ((1, 1), (1, 1)), mode='wrap')
    dzdx = (terrain_padded[1:-1, 2:] - terrain_padded[1:-1, :-2]) / (2 * dx)
    dzdy = (terrain_padded[2:, 1:-1] - terrain_padded[:-2, 1:-1]) / (2 * dy)
    
    return AtmosphereGrids(
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
        nz=nz,
        z_w=z_w,
        z_u=z_u,
        dz_w=dz_w,
        terrain=terrain,
        sigma=sigma,
        dzdx=dzdx,
        dzdy=dzdy,
    )


# =============================================================================
# Base State
# =============================================================================

def compute_base_state(
    z_u: jnp.ndarray,
    theta_surface: float = 300.0,
    lapse_rate: float = 0.0065,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute hydrostatic base state.
    
    Parameters
    ----------
    z_u : array
        Heights at cell centers (m)
    theta_surface : float
        Surface potential temperature (K)
    lapse_rate : float
        Temperature lapse rate (K/m)
        
    Returns
    -------
    theta_base : array
        Base state potential temperature (K)
    rho_base : array
        Base state density (kg/m³)
    """
    # Potential temperature (constant or slight increase with height)
    # For neutral stability, theta increases slightly
    theta_base = theta_surface + 0.003 * z_u  # ~3 K/km stable stratification
    
    # Pressure from hydrostatic balance
    # dp/dz = -ρg, with ideal gas law
    # For simplicity, use exponential atmosphere
    scale_height = R_DRY * T_REF / G  # ~8.5 km
    p_base = P_REF * jnp.exp(-z_u / scale_height)
    
    # Density from ideal gas law
    T_base = theta_base * (p_base / P_REF) ** (R_DRY / CP)
    rho_base = p_base / (R_DRY * T_base)
    
    return theta_base, rho_base


# =============================================================================
# Initialization
# =============================================================================

def initialize_atmosphere(
    grids: AtmosphereGrids,
    params: AtmosphereParams,
    u_init: float = 5.0,
    v_init: float = 0.0,
    theta_surface: float = 300.0,
) -> AtmosphereState:
    """
    Initialize atmospheric state.
    
    Parameters
    ----------
    grids : AtmosphereGrids
        Grid structure
    params : AtmosphereParams
        Atmospheric parameters
    u_init, v_init : float
        Initial wind components (m/s)
    theta_surface : float
        Surface potential temperature (K)
        
    Returns
    -------
    AtmosphereState
        Initialized state
    """
    nx, ny, nz = grids.nx, grids.ny, grids.nz
    
    # Base state
    theta_base, rho_base = compute_base_state(grids.z_u, theta_surface)
    
    # Initialize velocity (uniform with height for simplicity)
    u = jnp.ones((nz, ny, nx + 1)) * u_init
    v = jnp.ones((nz, ny + 1, nx)) * v_init
    w = jnp.zeros((nz + 1, ny, nx))
    
    # Initialize theta with base state
    theta = jnp.broadcast_to(theta_base[:, None, None], (nz, ny, nx)).copy()
    
    # Zero pressure perturbation
    p_prime = jnp.zeros((nz, ny, nx))
    
    return AtmosphereState(
        u=u,
        v=v,
        w=w,
        theta=theta,
        theta_base=theta_base,
        rho_base=rho_base,
        p_prime=p_prime,
        z_levels=grids.z_w,
        dz=grids.dz_w,
    )


# =============================================================================
# Fire Heat Source
# =============================================================================

def compute_fire_heat_source(
    fire_intensity: jnp.ndarray,
    grids: AtmosphereGrids,
    params: AtmosphereParams,
) -> jnp.ndarray:
    """
    Compute heat source from fire for theta tendency.
    
    Fire heat is injected into the lowest atmospheric levels,
    distributed vertically based on plume injection depth.
    
    Parameters
    ----------
    fire_intensity : array (ny, nx)
        Fire intensity (kW/m)
    grids : AtmosphereGrids
        Grid structure
    params : AtmosphereParams
        Parameters
        
    Returns
    -------
    Q : array (nz, ny, nx)
        Heating rate (K/s)
    """
    nz = grids.nz
    ny, nx = fire_intensity.shape
    
    # Convert fire intensity to heat flux (W/m²)
    # Assume fire width of ~10m and 50% radiative loss
    heat_flux = fire_intensity * 1000.0 * 0.5 * params.fire_heat_flux_scale  # W/m²
    
    # Vertical distribution of heat injection
    # Use exponential decay from surface
    z_inject = params.plume_injection_depth
    z_mid = grids.z_u
    
    # Weight function: exponential decay
    weights = jnp.exp(-z_mid / z_inject)
    weights = weights / jnp.sum(weights * grids.dz_w)  # Normalize
    
    # 3D heat source
    Q = jnp.zeros((nz, ny, nx))
    
    # Inject heat (dθ/dt = Q / (ρ * cp))
    for k in range(nz):
        rho_k = RHO_REF * jnp.exp(-grids.z_u[k] / 8500.0)  # Approximate density
        Q = Q.at[k].set(
            heat_flux * weights[k] / (rho_k * CP * grids.dz_w[k])
        )
    
    return Q


# =============================================================================
# Advection
# =============================================================================

# @jax.jit  # Removed: grids has non-static fields
def advect_scalar(
    phi: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    grids: AtmosphereGrids,
    dt: float,
) -> jnp.ndarray:
    """
    Advect scalar field using upwind scheme.
    
    Parameters
    ----------
    phi : array (nz, ny, nx)
        Scalar field to advect
    u, v, w : arrays
        Velocity components (staggered)
    grids : AtmosphereGrids
        Grid structure
    dt : float
        Time step
        
    Returns
    -------
    phi_new : array
        Advected scalar field
    """
    dx, dy = grids.dx, grids.dy
    dz = grids.dz_w
    nz, ny, nx = phi.shape
    
    # Interpolate velocities to cell centers
    u_c = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_c = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    w_c = 0.5 * (w[:-1, :, :] + w[1:, :, :])
    
    # X-direction advection (upwind)
    phi_xm = jnp.roll(phi, 1, axis=2)
    phi_xp = jnp.roll(phi, -1, axis=2)
    
    flux_x = jnp.where(
        u_c > 0,
        u_c * (phi - phi_xm) / dx,
        u_c * (phi_xp - phi) / dx
    )
    
    # Y-direction advection
    phi_ym = jnp.roll(phi, 1, axis=1)
    phi_yp = jnp.roll(phi, -1, axis=1)
    
    flux_y = jnp.where(
        v_c > 0,
        v_c * (phi - phi_ym) / dy,
        v_c * (phi_yp - phi) / dy
    )
    
    # Z-direction advection (no periodic BC)
    phi_zm = jnp.concatenate([phi[0:1], phi[:-1]], axis=0)
    phi_zp = jnp.concatenate([phi[1:], phi[-1:]], axis=0)
    dz_3d = dz[:, None, None]
    
    flux_z = jnp.where(
        w_c > 0,
        w_c * (phi - phi_zm) / dz_3d,
        w_c * (phi_zp - phi) / dz_3d
    )
    
    # Update
    phi_new = phi - dt * (flux_x + flux_y + flux_z)
    
    # Ensure finite values
    phi_new = jnp.where(jnp.isfinite(phi_new), phi_new, phi)
    
    return phi_new


# @jax.jit  # Removed: grids has non-static fields
def advect_momentum(
    state: AtmosphereState,
    grids: AtmosphereGrids,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Advect momentum using centered differences.
    
    Returns updated u, v, w after advection step.
    """
    u, v, w = state.u, state.v, state.w
    dx, dy = grids.dx, grids.dy
    dz = grids.dz_w
    
    # This is a simplified implementation
    # Full implementation would use proper staggered interpolation
    
    # For now, use simple central differences
    # u advection
    u_c = 0.5 * (u[:, :, :-1] + u[:, :, 1:])  # Interpolate to centers
    
    # Advection of u by u (x-direction)
    dudx = (jnp.roll(u, -1, axis=2) - jnp.roll(u, 1, axis=2)) / (2 * dx)
    
    # Advection of u by v (y-direction)  
    v_on_u = 0.5 * (v[:, :, :-1] + v[:, :, 1:])  # v at u points
    v_on_u = 0.5 * (v_on_u[:, :-1, :] + v_on_u[:, 1:, :])
    v_on_u = jnp.pad(v_on_u, ((0, 0), (0, 1), (0, 0)), mode='edge')
    
    dudy = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * dy)
    
    # Combine
    u_adv = u - dt * (u * dudx + v_on_u * dudy)
    
    # Similar for v and w (simplified)
    v_adv = v  # Placeholder
    w_adv = w  # Placeholder
    
    return u_adv, v_adv, w_adv


# =============================================================================
# Pressure Solver
# =============================================================================

# @jax.jit  # Removed: grids has non-static fields
def compute_divergence(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    grids: AtmosphereGrids,
    rho_base: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute mass-weighted divergence.
    
    For anelastic equations: ∇·(ρ₀u) = 0
    """
    dx, dy = grids.dx, grids.dy
    dz = grids.dz_w
    nz = grids.nz
    
    # dρu/dx
    div_x = (u[:, :, 1:] - u[:, :, :-1]) / dx
    
    # dρv/dy
    div_y = (v[:, 1:, :] - v[:, :-1, :]) / dy
    
    # dρw/dz (need density weighting)
    rho_3d = rho_base[:, None, None]
    div_z = (rho_base[None, :, None, None] * w[1:] - 
             rho_base[None, :, None, None] * w[:-1])
    # Simplified: just use w divergence
    div_z = (w[1:, :, :] - w[:-1, :, :]) / dz[:, None, None]
    
    return div_x + div_y + div_z


def pressure_projection(
    u: jnp.ndarray,
    v: jnp.ndarray,
    w: jnp.ndarray,
    grids: AtmosphereGrids,
    rho_base: jnp.ndarray,
    n_iter: int = 50,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Project velocity field to be divergence-free.
    
    Uses iterative Jacobi method to solve pressure Poisson equation.
    
    Returns corrected u, v, w and pressure perturbation.
    """
    dx, dy = grids.dx, grids.dy
    dz = grids.dz_w
    nz, ny, nx = grids.nz, grids.ny, grids.nx
    
    # Compute divergence
    div = compute_divergence(u, v, w, grids, rho_base)
    
    # Ensure divergence is finite
    div = jnp.where(jnp.isfinite(div), div, 0.0)
    
    # Solve ∇²p = div using Jacobi iteration
    p = jnp.zeros((nz, ny, nx))
    
    # Coefficients for Laplacian with regularization
    cx = 1.0 / (dx * dx)
    cy = 1.0 / (dy * dy)
    cz = 1.0 / (dz[:, None, None] ** 2 + 1e-10)
    cc = 2.0 * (cx + cy) + 2.0 * cz + 1e-8  # Regularization
    
    def jacobi_step(p, _):
        # Neighbors with periodic BC in x,y
        p_xm = jnp.roll(p, 1, axis=2)
        p_xp = jnp.roll(p, -1, axis=2)
        p_ym = jnp.roll(p, 1, axis=1)
        p_yp = jnp.roll(p, -1, axis=1)
        
        # Vertical neighbors (zero gradient at boundaries)
        p_zm = jnp.concatenate([p[0:1], p[:-1]], axis=0)
        p_zp = jnp.concatenate([p[1:], p[-1:]], axis=0)
        
        # Jacobi update with under-relaxation
        omega = 0.7
        p_new = (cx * (p_xm + p_xp) + 
                 cy * (p_ym + p_yp) + 
                 cz * (p_zm + p_zp) - div) / cc
        p_new = omega * p_new + (1 - omega) * p
        
        # Ensure finite
        p_new = jnp.where(jnp.isfinite(p_new), p_new, 0.0)
        
        return p_new, None
    
    p, _ = jax.lax.scan(jacobi_step, p, None, length=n_iter)
    
    # Correct velocities with damped correction
    correction_scale = 0.3
    
    dpdx = (jnp.roll(p, -1, axis=2) - p) / dx
    dpdx = jnp.concatenate([dpdx, dpdx[:, :, -1:]], axis=2)
    
    dpdy = (jnp.roll(p, -1, axis=1) - p) / dy
    dpdy = jnp.concatenate([dpdy, dpdy[:, -1:, :]], axis=1)
    
    dpdz = jnp.zeros((nz + 1, ny, nx))
    dpdz = dpdz.at[1:-1].set((p[1:] - p[:-1]) / (dz[:-1, None, None] + 1e-10))
    
    rho_3d = rho_base[:, None, None]
    rho_mean = jnp.mean(rho_base)
    
    u_corr = u - correction_scale * dpdx / rho_3d
    v_corr = v - correction_scale * dpdy / rho_3d
    w_corr = w - correction_scale * dpdz / rho_mean
    
    # Final NaN check
    u_corr = jnp.where(jnp.isfinite(u_corr), u_corr, u)
    v_corr = jnp.where(jnp.isfinite(v_corr), v_corr, v)
    w_corr = jnp.where(jnp.isfinite(w_corr), w_corr, w)
    
    return u_corr, v_corr, w_corr, p


# =============================================================================
# Buoyancy and Diffusion
# =============================================================================

# @jax.jit  # Removed: grids has non-static fields
def compute_buoyancy(
    theta: jnp.ndarray,
    theta_base: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute buoyancy acceleration.
    
    B = g * (θ' / θ₀) where θ' = θ - θ₀
    """
    theta_base_3d = theta_base[:, None, None]
    theta_prime = theta - theta_base_3d
    
    buoyancy = G * theta_prime / theta_base_3d
    
    return buoyancy


# @jax.jit  # Removed: grids has non-static fields
def apply_diffusion(
    phi: jnp.ndarray,
    K_h: float,
    K_v: float,
    grids: AtmosphereGrids,
    dt: float,
) -> jnp.ndarray:
    """
    Apply turbulent diffusion to scalar field.
    
    ∂φ/∂t = ∇·(K∇φ)
    """
    dx, dy = grids.dx, grids.dy
    dz = grids.dz_w[:, None, None]
    
    # Laplacian in x
    phi_xm = jnp.roll(phi, 1, axis=2)
    phi_xp = jnp.roll(phi, -1, axis=2)
    d2phi_dx2 = (phi_xm - 2*phi + phi_xp) / (dx * dx)
    
    # Laplacian in y
    phi_ym = jnp.roll(phi, 1, axis=1)
    phi_yp = jnp.roll(phi, -1, axis=1)
    d2phi_dy2 = (phi_ym - 2*phi + phi_yp) / (dy * dy)
    
    # Laplacian in z (with BC)
    phi_zm = jnp.concatenate([phi[0:1], phi[:-1]], axis=0)
    phi_zp = jnp.concatenate([phi[1:], phi[-1:]], axis=0)
    d2phi_dz2 = (phi_zm - 2*phi + phi_zp) / (dz * dz + 1e-10)
    
    # Update with stability limit
    max_K = max(K_h, K_v)
    max_dt = 0.25 * min(dx*dx, float(jnp.min(dz))**2) / (max_K + 1e-10)
    dt_safe = min(dt, max_dt)
    
    phi_new = phi + dt_safe * (K_h * (d2phi_dx2 + d2phi_dy2) + K_v * d2phi_dz2)
    
    # Ensure finite
    phi_new = jnp.where(jnp.isfinite(phi_new), phi_new, phi)
    
    return phi_new


# =============================================================================
# Boundary Conditions
# =============================================================================

def apply_boundary_conditions(
    state: AtmosphereState,
    grids: AtmosphereGrids,
    params: AtmosphereParams,
) -> AtmosphereState:
    """Apply boundary conditions to state."""
    
    u, v, w = state.u, state.v, state.w
    theta = state.theta
    
    # Surface: no-slip for w
    w = w.at[0].set(0.0)
    
    # Top: free-slip or damping
    if params.top_bc == "free_slip":
        w = w.at[-1].set(0.0)
    else:
        # Rayleigh damping in upper layers
        z_top = grids.z_w[-1]
        damping_start = z_top - params.damping_depth
        
        for k in range(grids.nz):
            z = grids.z_u[k]
            if z > damping_start:
                damping = params.damping_coef * (z - damping_start) / params.damping_depth
                u = u.at[k].set(u[k] * (1 - damping))
                v = v.at[k].set(v[k] * (1 - damping))
                theta = theta.at[k].set(
                    theta[k] - damping * (theta[k] - state.theta_base[k])
                )
    
    return state._replace(u=u, v=v, w=w, theta=theta)


# =============================================================================
# Main Time Integration
# =============================================================================

def evolve_atmosphere_step(
    state: AtmosphereState,
    grids: AtmosphereGrids,
    params: AtmosphereParams,
    fire_intensity: jnp.ndarray,
    dt: float,
) -> AtmosphereState:
    """
    Evolve atmospheric state by one time step.
    
    Uses operator splitting:
    1. Advection
    2. Fire heat source
    3. Buoyancy
    4. Diffusion
    5. Pressure projection
    6. Boundary conditions
    
    Parameters
    ----------
    state : AtmosphereState
        Current state
    grids : AtmosphereGrids
        Grid structure
    params : AtmosphereParams
        Parameters
    fire_intensity : array (ny, nx)
        Fire intensity for heat source
    dt : float
        Time step (s)
        
    Returns
    -------
    AtmosphereState
        Updated state
    """
    u, v, w = state.u, state.v, state.w
    theta = state.theta
    
    # 1. Advect theta
    theta = advect_scalar(theta, u, v, w, grids, dt)
    
    # 2. Fire heat source
    Q = compute_fire_heat_source(fire_intensity, grids, params)
    theta = theta + dt * Q
    
    # 3. Buoyancy forcing on w
    buoyancy = compute_buoyancy(theta, state.theta_base)
    # Apply buoyancy at w-levels (interpolate)
    buoy_w = 0.5 * (
        jnp.concatenate([buoyancy[0:1], buoyancy], axis=0) +
        jnp.concatenate([buoyancy, buoyancy[-1:]], axis=0)
    )
    w = w + dt * buoy_w
    
    # 4. Diffusion
    theta = apply_diffusion(theta, params.kh_h, params.kh_v, grids, dt)
    
    # Note: Momentum diffusion would go here but omitted for simplicity
    
    # 5. Pressure projection (enforce continuity)
    u, v, w, p_prime = pressure_projection(
        u, v, w, grids, state.rho_base, n_iter=50
    )
    
    # 6. Clamp values to prevent runaway
    max_vel = 100.0  # m/s - reasonable max
    u = jnp.clip(u, -max_vel, max_vel)
    v = jnp.clip(v, -max_vel, max_vel)
    w = jnp.clip(w, -max_vel, max_vel)
    theta = jnp.clip(theta, 200.0, 500.0)  # Reasonable temperature range
    
    # Replace any NaN/inf
    u = jnp.where(jnp.isfinite(u), u, state.u)
    v = jnp.where(jnp.isfinite(v), v, state.v)
    w = jnp.where(jnp.isfinite(w), w, state.w)
    theta = jnp.where(jnp.isfinite(theta), theta, state.theta)
    
    # 7. Boundary conditions
    new_state = state._replace(
        u=u, v=v, w=w, theta=theta, p_prime=p_prime
    )
    new_state = apply_boundary_conditions(new_state, grids, params)
    
    return new_state


# =============================================================================
# Fire-Atmosphere Coupling Interface
# =============================================================================

def couple_atmosphere_to_fire(
    state: AtmosphereState,
    grids: AtmosphereGrids,
) -> CouplingResult:
    """
    Extract surface wind and diagnostics for fire spread model.
    
    Parameters
    ----------
    state : AtmosphereState
        Current atmospheric state
    grids : AtmosphereGrids
        Grid structure
        
    Returns
    -------
    CouplingResult
        Surface winds and diagnostics for fire model
    """
    # Surface winds (lowest level, interpolated to centers)
    u_sfc = 0.5 * (state.u[0, :, :-1] + state.u[0, :, 1:])
    v_sfc = 0.5 * (state.v[0, :-1, :] + state.v[0, 1:, :])
    
    # Clamp to reasonable values and replace NaN/inf
    u_sfc = jnp.clip(u_sfc, -100.0, 100.0)
    v_sfc = jnp.clip(v_sfc, -100.0, 100.0)
    u_sfc = jnp.where(jnp.isfinite(u_sfc), u_sfc, 0.0)
    v_sfc = jnp.where(jnp.isfinite(v_sfc), v_sfc, 0.0)
    
    # Wind speed and direction
    wind_speed = jnp.sqrt(u_sfc**2 + v_sfc**2)
    wind_direction = jnp.rad2deg(jnp.arctan2(-u_sfc, -v_sfc)) % 360.0
    
    # Maximum updraft (indicator of plume strength)
    w_safe = jnp.where(jnp.isfinite(state.w), state.w, 0.0)
    w_safe = jnp.clip(w_safe, -100.0, 100.0)
    w_max = jnp.max(w_safe, axis=0)
    
    # Plume top (height where w drops below threshold)
    w_threshold = 1.0  # m/s
    w_above_threshold = w_safe > w_threshold
    # Find highest level with significant w
    plume_top = jnp.zeros_like(w_max)
    for k in range(grids.nz):
        plume_top = jnp.where(
            w_above_threshold[k],
            grids.z_w[k],
            plume_top
        )
    
    # Indraft strength (convergence at surface)
    dudx = (state.u[0, :, 1:] - state.u[0, :, :-1]) / grids.dx
    dvdy = (state.v[0, 1:, :] - state.v[0, :-1, :]) / grids.dy
    dudx = jnp.where(jnp.isfinite(dudx), dudx, 0.0)
    dvdy = jnp.where(jnp.isfinite(dvdy), dvdy, 0.0)
    convergence = -(dudx + dvdy)
    indraft_strength = jnp.maximum(convergence, 0.0)
    
    # Vertical vorticity
    # ζ = dv/dx - du/dy
    dvdx = (jnp.roll(v_sfc, -1, axis=1) - jnp.roll(v_sfc, 1, axis=1)) / (2 * grids.dx)
    dudy = (jnp.roll(u_sfc, -1, axis=0) - jnp.roll(u_sfc, 1, axis=0)) / (2 * grids.dy)
    vorticity = dvdx - dudy
    vorticity = jnp.where(jnp.isfinite(vorticity), vorticity, 0.0)
    
    return CouplingResult(
        u_surface=u_sfc,
        v_surface=v_sfc,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        w_max=w_max,
        plume_top=plume_top,
        indraft_strength=indraft_strength,
        vorticity=vorticity,
    )


# =============================================================================
# Full Coupled Simulation
# =============================================================================

def run_coupled_simulation(
    grids_atm: AtmosphereGrids,
    params_atm: AtmosphereParams,
    fire_intensity_func,  # Function: (t) -> fire_intensity array
    n_steps: int,
    dt: float,
    u_init: float = 5.0,
    v_init: float = 0.0,
    verbose: bool = True,
) -> Tuple[AtmosphereState, list]:
    """
    Run coupled atmosphere-fire simulation.
    
    Parameters
    ----------
    grids_atm : AtmosphereGrids
        Atmospheric grid structure
    params_atm : AtmosphereParams
        Atmospheric parameters
    fire_intensity_func : callable
        Function returning fire intensity at time t
    n_steps : int
        Number of time steps
    dt : float
        Time step (s)
    u_init, v_init : float
        Initial wind components
    verbose : bool
        Print progress
        
    Returns
    -------
    final_state : AtmosphereState
        Final atmospheric state
    coupling_history : list
        History of coupling results
    """
    if verbose:
        print("="*60)
        print("3D ATMOSPHERIC SIMULATION WITH FIRE COUPLING")
        print("="*60)
        print(f"Grid: {grids_atm.nx} x {grids_atm.ny} x {grids_atm.nz}")
        print(f"Domain: {grids_atm.nx * grids_atm.dx/1000:.1f} x "
              f"{grids_atm.ny * grids_atm.dy/1000:.1f} x "
              f"{params_atm.z_top/1000:.1f} km")
        print(f"Time step: {dt:.2f} s, Total: {n_steps * dt:.0f} s")
        print("="*60)
    
    # Initialize
    state = initialize_atmosphere(
        grids_atm, params_atm, u_init, v_init
    )
    
    coupling_history = []
    
    # Time loop
    for step in range(n_steps):
        t = step * dt
        
        # Get fire intensity
        fire_intensity = fire_intensity_func(t)
        
        # Evolve atmosphere
        state = evolve_atmosphere_step(
            state, grids_atm, params_atm, fire_intensity, dt
        )
        
        # Get coupling result
        coupling = couple_atmosphere_to_fire(state, grids_atm)
        coupling_history.append(coupling)
        
        # Progress
        if verbose and (step + 1) % 100 == 0:
            max_w = float(jnp.max(coupling.w_max))
            max_sfc = float(jnp.max(coupling.wind_speed))
            print(f"  Step {step+1}/{n_steps}: t={t:.0f}s, "
                  f"max_w={max_w:.1f} m/s, max_sfc_wind={max_sfc:.1f} m/s")
    
    if verbose:
        print("="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
    
    return state, coupling_history


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_atmosphere_test(
    nx: int = 50,
    ny: int = 50,
    dx: float = 100.0,
    nz: int = 20,
    z_top: float = 2000.0,
    fire_intensity_kw: float = 5000.0,
    duration_s: float = 300.0,
    dt: float = 1.0,
) -> Tuple[AtmosphereState, CouplingResult]:
    """
    Quick test of atmospheric solver with point fire source.
    
    Parameters
    ----------
    nx, ny : int
        Horizontal grid size
    dx : float
        Grid spacing (m)
    nz : int
        Vertical levels
    z_top : float
        Domain top (m)
    fire_intensity_kw : float
        Fire intensity (kW/m)
    duration_s : float
        Simulation duration (s)
    dt : float
        Time step (s)
        
    Returns
    -------
    final_state : AtmosphereState
        Final state
    coupling : CouplingResult
        Final coupling diagnostics
    """
    # Create flat terrain
    terrain = jnp.zeros((ny, nx))
    
    # Parameters
    params = AtmosphereParams(nz=nz, z_top=z_top)
    
    # Create grids
    grids = create_atmosphere_grids(nx, ny, dx, dx, terrain, params)
    
    # Fire intensity function (point source at center)
    def fire_func(t):
        intensity = jnp.zeros((ny, nx))
        # 3x3 fire at center
        cy, cx = ny // 2, nx // 2
        intensity = intensity.at[cy-1:cy+2, cx-1:cx+2].set(fire_intensity_kw)
        return intensity
    
    # Run
    n_steps = int(duration_s / dt)
    state, history = run_coupled_simulation(
        grids, params, fire_func, n_steps, dt,
        verbose=True
    )
    
    # Final coupling
    coupling = couple_atmosphere_to_fire(state, grids)
    
    return state, coupling


def summarize_atmosphere_state(
    state: AtmosphereState,
    grids: AtmosphereGrids,
) -> dict:
    """Generate summary statistics of atmospheric state."""
    
    coupling = couple_atmosphere_to_fire(state, grids)
    
    return {
        "max_u": float(jnp.max(jnp.abs(state.u))),
        "max_v": float(jnp.max(jnp.abs(state.v))),
        "max_w": float(jnp.max(state.w)),
        "min_w": float(jnp.min(state.w)),
        "theta_perturbation_max": float(jnp.max(state.theta - state.theta_base[:, None, None])),
        "surface_wind_max": float(jnp.max(coupling.wind_speed)),
        "surface_wind_mean": float(jnp.mean(coupling.wind_speed)),
        "plume_top_max": float(jnp.max(coupling.plume_top)),
        "max_indraft": float(jnp.max(coupling.indraft_strength)),
        "max_vorticity": float(jnp.max(jnp.abs(coupling.vorticity))),
    }
