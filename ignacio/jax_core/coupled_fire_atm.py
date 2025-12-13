"""
WRF-SFIRE Style Coupled Fire-Atmosphere Simulation.

This module provides the full coupling between the 2D fire spread model
and the 3D atmospheric dynamics, similar to WRF-SFIRE but in pure JAX.

The coupling is two-way:
1. Fire → Atmosphere: Heat release creates buoyancy, driving convection
2. Atmosphere → Fire: Modified wind field affects fire spread

This enables simulation of:
- Plume-dominated fires
- Fire whirls and vortices  
- Blow-up fire conditions
- Ember transport in convective columns
- Indraft-enhanced fire spread

Architecture
------------
The fire model runs on a fine 2D mesh (e.g., 30m resolution)
The atmosphere model runs on a coarser 3D mesh (e.g., 100m horizontal, 20 levels)

Fire heat release is aggregated to the atmospheric grid.
Atmospheric winds are interpolated to the fire grid.

References
----------
- Mandel, J. et al. (2011). Coupled atmosphere-wildland fire modeling
  with WRF-SFIRE. Geosci. Model Dev., 4, 591-610.
- Coen, J.L. et al. (2013). WRF-Fire: Coupled Weather-Wildland Fire
  Modeling with the Weather Research and Forecasting Model.
- Clark, T.L. et al. (2004). Description of a coupled atmosphere-fire
  model. Int. J. Wildland Fire, 13, 49-63.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple, Callable
from datetime import datetime, timedelta
from functools import partial
import jax
import jax.numpy as jnp

# Import fire model components
from .levelset import (
    LevelSetGrids,
    initialize_phi,
    evolve_phi,
)

# Import atmosphere components
from .atmosphere_3d import (
    AtmosphereParams,
    AtmosphereState,
    AtmosphereGrids,
    CouplingResult,
    create_atmosphere_grids,
    initialize_atmosphere,
    evolve_atmosphere_step,
    couple_atmosphere_to_fire,
    compute_base_state,
)

# Import spotting for ember transport
from .spotting import SpottingParams, apply_spotting


# =============================================================================
# Configuration
# =============================================================================

class CoupledSimConfig(NamedTuple):
    """Configuration for coupled fire-atmosphere simulation."""
    
    # Fire grid parameters
    fire_dx: float = 30.0              # Fire grid spacing (m)
    
    # Atmosphere grid parameters  
    atm_dx: float = 100.0              # Atmosphere horizontal spacing (m)
    atm_nz: int = 20                   # Atmosphere vertical levels
    atm_z_top: float = 2000.0          # Atmosphere domain top (m)
    
    # Coupling parameters
    coupling_interval: int = 10        # Steps between atmosphere updates
    heat_flux_fraction: float = 0.5    # Fraction of fire heat entering atmosphere
    plume_injection_depth: float = 100.0  # Depth of heat injection (m)
    
    # Stability controls
    max_wind_speed: float = 50.0       # Maximum allowed wind (m/s)
    max_updraft: float = 30.0          # Maximum updraft (m/s)
    atmosphere_dt: float = 0.5         # Atmosphere time step (s)
    
    # Physics options
    enable_ember_transport: bool = True
    enable_fire_whirls: bool = True
    enable_indraft: bool = True
    
    # Ember transport parameters
    ember_release_threshold: float = 3000.0  # Fire intensity for ember release (kW/m)
    ember_lofting_height: float = 500.0      # Maximum lofting height (m)
    
    # Turbulence
    km_h: float = 50.0                 # Horizontal eddy viscosity
    km_v: float = 10.0                 # Vertical eddy viscosity


class CoupledSimState(NamedTuple):
    """State for coupled simulation."""
    
    # Fire state
    phi: jnp.ndarray                   # Level-set function
    fire_ros: jnp.ndarray              # Rate of spread (m/min)
    fire_intensity: jnp.ndarray        # Fire intensity (kW/m)
    
    # Atmosphere state
    atm_state: AtmosphereState
    
    # Coupling state
    surface_wind_u: jnp.ndarray        # Surface u wind on fire grid
    surface_wind_v: jnp.ndarray        # Surface v wind on fire grid
    indraft_u: jnp.ndarray             # Indraft contribution to u
    indraft_v: jnp.ndarray             # Indraft contribution to v
    
    # Time tracking
    time_elapsed: float                # Simulation time (minutes)
    atm_steps: int                     # Atmosphere steps taken


class CoupledSimResult(NamedTuple):
    """Results from coupled simulation."""
    
    # Final states
    phi_final: jnp.ndarray
    atm_state_final: AtmosphereState
    burned_area: float
    
    # Time series
    phi_history: Optional[jnp.ndarray] = None
    area_history: Optional[list] = None
    plume_height_history: Optional[list] = None
    max_updraft_history: Optional[list] = None
    
    # Diagnostics
    max_indraft: float = 0.0
    max_vorticity: float = 0.0
    ember_spots: int = 0


# =============================================================================
# Grid Interpolation
# =============================================================================

def interpolate_atm_to_fire(
    atm_field: jnp.ndarray,
    atm_grids: AtmosphereGrids,
    fire_shape: Tuple[int, int],
    fire_dx: float,
) -> jnp.ndarray:
    """
    Interpolate atmospheric field to fire grid.
    
    Uses bilinear interpolation from coarse atmosphere grid
    to fine fire grid.
    """
    atm_ny, atm_nx = atm_field.shape
    fire_ny, fire_nx = fire_shape
    
    # Create coordinate arrays
    atm_x = jnp.arange(atm_nx) * atm_grids.dx
    atm_y = jnp.arange(atm_ny) * atm_grids.dy
    fire_x = jnp.arange(fire_nx) * fire_dx
    fire_y = jnp.arange(fire_ny) * fire_dx
    
    # Simple nearest-neighbor for now (bilinear would be better)
    fire_xi = (fire_x / atm_grids.dx).astype(jnp.int32)
    fire_yi = (fire_y / atm_grids.dy).astype(jnp.int32)
    
    fire_xi = jnp.clip(fire_xi, 0, atm_nx - 1)
    fire_yi = jnp.clip(fire_yi, 0, atm_ny - 1)
    
    # Index into atmospheric field
    fire_field = atm_field[fire_yi[:, None], fire_xi[None, :]]
    
    return fire_field


def aggregate_fire_to_atm(
    fire_field: jnp.ndarray,
    fire_dx: float,
    atm_grids: AtmosphereGrids,
) -> jnp.ndarray:
    """
    Aggregate fire field to atmosphere grid.
    
    Sums fire intensity within each atmosphere grid cell.
    """
    fire_ny, fire_nx = fire_field.shape
    atm_nx, atm_ny = atm_grids.nx, atm_grids.ny
    
    # Ratio of grid spacings
    ratio = int(atm_grids.dx / fire_dx)
    
    if ratio <= 1:
        # Same resolution or fire is coarser
        return fire_field[:atm_ny, :atm_nx]
    
    # Aggregate by summing blocks
    atm_field = jnp.zeros((atm_ny, atm_nx))
    
    for j in range(atm_ny):
        for i in range(atm_nx):
            j_start = j * ratio
            j_end = min((j + 1) * ratio, fire_ny)
            i_start = i * ratio
            i_end = min((i + 1) * ratio, fire_nx)
            
            atm_field = atm_field.at[j, i].set(
                jnp.mean(fire_field[j_start:j_end, i_start:i_end])
            )
    
    return atm_field


# =============================================================================
# Fire Intensity Calculation
# =============================================================================

def compute_fire_intensity_from_phi(
    phi: jnp.ndarray,
    phi_prev: jnp.ndarray,
    ros: jnp.ndarray,
    dx: float,
    dt: float,
    heat_content: float = 18000.0,  # kJ/kg
    fuel_load: float = 1.0,         # kg/m²
) -> jnp.ndarray:
    """
    Compute fire intensity from level-set evolution.
    
    Fire intensity I = H * w * R where:
    - H = heat content (kJ/kg)
    - w = fuel consumed (kg/m²)
    - R = rate of spread (m/s)
    
    We estimate active fire from cells that recently burned.
    """
    # Find newly burned cells (phi crossed zero)
    newly_burned = (phi < 0) & (phi_prev >= 0)
    
    # Active fire is at the front (phi near zero and negative)
    active_fire = (phi < 0) & (phi > -2 * dx)
    
    # Intensity from Byram's equation
    # I = H * w * R (kW/m when R in m/s)
    intensity = heat_content * fuel_load * (ros / 60.0)  # ros in m/min → m/s
    
    # Only at active fire locations
    intensity = jnp.where(active_fire, intensity, 0.0)
    
    return intensity


# =============================================================================
# Wind Modification from Coupling
# =============================================================================

def compute_effective_wind(
    base_u: jnp.ndarray,
    base_v: jnp.ndarray,
    indraft_u: jnp.ndarray,
    indraft_v: jnp.ndarray,
    config: CoupledSimConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute effective wind including indraft from fire.
    
    Returns total wind components, speed, and direction.
    """
    if config.enable_indraft:
        total_u = base_u + indraft_u
        total_v = base_v + indraft_v
    else:
        total_u = base_u
        total_v = base_v
    
    # Clamp to maximum
    speed = jnp.sqrt(total_u**2 + total_v**2)
    scale = jnp.minimum(config.max_wind_speed / jnp.maximum(speed, 0.1), 1.0)
    
    total_u = total_u * scale
    total_v = total_v * scale
    
    # Recompute speed and direction
    speed = jnp.sqrt(total_u**2 + total_v**2)
    direction = jnp.rad2deg(jnp.arctan2(-total_u, -total_v)) % 360.0
    
    return total_u, total_v, speed, direction


# =============================================================================
# Ember Transport
# =============================================================================

def compute_ember_transport(
    phi: jnp.ndarray,
    fire_intensity: jnp.ndarray,
    w_column: jnp.ndarray,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    dx: float,
    dt: float,
    config: CoupledSimConfig,
) -> jnp.ndarray:
    """
    Compute ember spotting using atmospheric updraft.
    
    Embers are lofted by updrafts and transported by wind.
    Landing probability based on fall velocity and wind.
    """
    if not config.enable_ember_transport:
        return phi
    
    # Find cells with sufficient intensity for ember release
    ember_source = fire_intensity > config.ember_release_threshold
    
    if not jnp.any(ember_source):
        return phi
    
    # Maximum updraft determines lofting height
    max_w = jnp.max(w_column, axis=0)
    lofting_height = jnp.minimum(
        max_w * 30.0,  # Approximate lofting time * velocity
        config.ember_lofting_height
    )
    
    # Transport distance based on wind and lofting height
    wind_speed = jnp.sqrt(wind_u**2 + wind_v**2)
    fall_time = lofting_height / 5.0  # 5 m/s fall velocity
    transport_distance = wind_speed * fall_time
    
    # Wind direction for transport
    wind_dir_rad = jnp.arctan2(wind_v, wind_u)
    
    # Create spotting probability field
    # Higher intensity + higher updraft = more embers = higher probability
    spot_probability = (
        ember_source.astype(jnp.float32) *
        jnp.minimum(fire_intensity / 10000.0, 1.0) *
        jnp.minimum(max_w / 10.0, 1.0)
    )
    
    # Apply spotting using existing module
    # This is a simplified version - full implementation would trace trajectories
    params = SpottingParams(
        spot_probability=0.1,
        max_spot_distance=float(jnp.max(transport_distance)),
    )
    
    phi_new = apply_spotting(
        phi, fire_intensity, wind_speed, 
        jnp.rad2deg(wind_dir_rad), dx, dt, params
    )
    
    return phi_new


# =============================================================================
# Initialization
# =============================================================================

def initialize_coupled_simulation(
    terrain: jnp.ndarray,
    fuel_ros: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    config: CoupledSimConfig,
    background_wind_speed: float = 10.0,
    background_wind_dir: float = 270.0,
) -> Tuple[CoupledSimState, AtmosphereGrids, LevelSetGrids]:
    """
    Initialize coupled fire-atmosphere simulation.
    
    Parameters
    ----------
    terrain : array (ny, nx)
        Terrain elevation (m) on fire grid
    fuel_ros : array (ny, nx)
        Base rate of spread by fuel type (m/min)
    x_ign, y_ign : float
        Ignition location (m)
    config : CoupledSimConfig
        Configuration
    background_wind_speed : float
        Initial wind speed (m/s)
    background_wind_dir : float
        Initial wind direction (degrees from north)
        
    Returns
    -------
    state : CoupledSimState
        Initial state
    atm_grids : AtmosphereGrids
        Atmosphere grid structure
    fire_grids : LevelSetGrids
        Fire grid structure
    """
    fire_ny, fire_nx = terrain.shape
    fire_dx = config.fire_dx
    
    # Atmosphere grid (coarser)
    atm_nx = int(fire_nx * fire_dx / config.atm_dx)
    atm_ny = int(fire_ny * fire_dx / config.atm_dx)
    
    # Aggregate terrain to atmosphere grid
    terrain_atm = aggregate_fire_to_atm(terrain, fire_dx, 
        AtmosphereGrids(
            dx=config.atm_dx, dy=config.atm_dx,
            nx=atm_nx, ny=atm_ny, nz=config.atm_nz,
            z_w=jnp.zeros(config.atm_nz+1),
            z_u=jnp.zeros(config.atm_nz),
            dz_w=jnp.zeros(config.atm_nz),
            terrain=jnp.zeros((atm_ny, atm_nx)),
            sigma=jnp.zeros(config.atm_nz+1),
            dzdx=jnp.zeros((atm_ny, atm_nx)),
            dzdy=jnp.zeros((atm_ny, atm_nx)),
        )
    )
    
    # Create atmosphere grids
    atm_params = AtmosphereParams(
        nz=config.atm_nz,
        z_top=config.atm_z_top,
        km_h=config.km_h,
        km_v=config.km_v,
        fire_heat_flux_scale=config.heat_flux_fraction,
        plume_injection_depth=config.plume_injection_depth,
    )
    
    atm_grids = create_atmosphere_grids(
        atm_nx, atm_ny, config.atm_dx, config.atm_dx,
        terrain_atm, atm_params
    )
    
    # Initialize atmosphere
    u_init = background_wind_speed * jnp.sin(jnp.deg2rad(background_wind_dir))
    v_init = background_wind_speed * jnp.cos(jnp.deg2rad(background_wind_dir))
    
    atm_state = initialize_atmosphere(
        atm_grids, atm_params,
        u_init=float(u_init), v_init=float(v_init)
    )
    
    # Create fire grids
    x_coords = jnp.arange(fire_nx) * fire_dx
    y_coords = jnp.arange(fire_ny) * fire_dx
    
    # Initial wind on fire grid
    wind_u = u_init * jnp.ones((fire_ny, fire_nx))
    wind_v = v_init * jnp.ones((fire_ny, fire_nx))
    wind_dir = background_wind_dir * jnp.ones((fire_ny, fire_nx))
    
    # Compute L/B ratio from wind (simplified)
    wind_speed = jnp.sqrt(wind_u**2 + wind_v**2)
    lb_ratio = 1.0 + 0.1 * wind_speed  # Simple approximation
    
    fire_grids = LevelSetGrids(
        x_coords=x_coords,
        y_coords=y_coords,
        ros=fuel_ros,
        bros=fuel_ros * 0.2,  # Back fire ~20% of head fire
        fros=fuel_ros * 0.4,  # Flank ~40% of head fire
        raz=jnp.deg2rad(wind_dir),
    )
    
    # Initialize level-set
    phi = initialize_phi(x_coords, y_coords, x_ign, y_ign, initial_radius=30.0)
    
    # Initial state
    state = CoupledSimState(
        phi=phi,
        fire_ros=fuel_ros,
        fire_intensity=jnp.zeros((fire_ny, fire_nx)),
        atm_state=atm_state,
        surface_wind_u=wind_u,
        surface_wind_v=wind_v,
        indraft_u=jnp.zeros((fire_ny, fire_nx)),
        indraft_v=jnp.zeros((fire_ny, fire_nx)),
        time_elapsed=0.0,
        atm_steps=0,
    )
    
    return state, atm_grids, fire_grids


# =============================================================================
# Main Evolution Step
# =============================================================================

def evolve_coupled_step(
    state: CoupledSimState,
    fire_grids: LevelSetGrids,
    atm_grids: AtmosphereGrids,
    atm_params: AtmosphereParams,
    config: CoupledSimConfig,
    fire_dt: float,
) -> CoupledSimState:
    """
    Evolve coupled simulation by one fire time step.
    
    The atmosphere is evolved at higher frequency (smaller dt)
    and coupling occurs every coupling_interval steps.
    """
    phi_prev = state.phi
    fire_ny, fire_nx = phi_prev.shape
    
    # Compute effective wind for fire spread
    total_u, total_v, wind_speed, wind_dir = compute_effective_wind(
        state.surface_wind_u, state.surface_wind_v,
        state.indraft_u, state.indraft_v,
        config
    )
    
    # Update fire grids with current wind
    updated_fire_grids = fire_grids._replace(
        raz=jnp.deg2rad(wind_dir)
    )
    
    # Evolve fire (level-set)
    phi_new = evolve_phi(state.phi, updated_fire_grids, t_idx=0, dt=fire_dt)
    
    # Compute fire intensity
    fire_intensity = compute_fire_intensity_from_phi(
        phi_new, phi_prev, state.fire_ros,
        config.fire_dx, fire_dt
    )
    
    # Update atmosphere if it's time
    atm_state = state.atm_state
    indraft_u = state.indraft_u
    indraft_v = state.indraft_v
    surface_wind_u = state.surface_wind_u
    surface_wind_v = state.surface_wind_v
    atm_steps = state.atm_steps
    
    # Aggregate fire intensity to atmosphere grid
    intensity_atm = aggregate_fire_to_atm(
        fire_intensity, config.fire_dx, atm_grids
    )
    
    # Run atmosphere substeps
    n_atm_steps = int(fire_dt * 60 / config.atmosphere_dt)  # fire_dt in min
    n_atm_steps = max(1, n_atm_steps)
    
    for _ in range(n_atm_steps):
        atm_state = evolve_atmosphere_step(
            atm_state, atm_grids, atm_params,
            intensity_atm, config.atmosphere_dt
        )
        
        # Clamp velocities for stability
        atm_state = atm_state._replace(
            u=jnp.clip(atm_state.u, -config.max_wind_speed, config.max_wind_speed),
            v=jnp.clip(atm_state.v, -config.max_wind_speed, config.max_wind_speed),
            w=jnp.clip(atm_state.w, -config.max_updraft, config.max_updraft),
        )
    
    atm_steps += n_atm_steps
    
    # Get coupling result
    coupling = couple_atmosphere_to_fire(atm_state, atm_grids)
    
    # Interpolate atmospheric wind to fire grid
    surface_wind_u = interpolate_atm_to_fire(
        coupling.u_surface, atm_grids, (fire_ny, fire_nx), config.fire_dx
    )
    surface_wind_v = interpolate_atm_to_fire(
        coupling.v_surface, atm_grids, (fire_ny, fire_nx), config.fire_dx
    )
    
    # Compute indraft from convergence
    if config.enable_indraft:
        indraft_strength = interpolate_atm_to_fire(
            coupling.indraft_strength, atm_grids, (fire_ny, fire_nx), config.fire_dx
        )
        
        # Indraft points toward fire (gradient of phi)
        # Simplified: use phi gradient as direction
        dphi_dx = (jnp.roll(phi_new, -1, axis=1) - jnp.roll(phi_new, 1, axis=1)) / (2 * config.fire_dx)
        dphi_dy = (jnp.roll(phi_new, -1, axis=0) - jnp.roll(phi_new, 1, axis=0)) / (2 * config.fire_dx)
        grad_mag = jnp.sqrt(dphi_dx**2 + dphi_dy**2) + 1e-6
        
        # Indraft velocity toward fire front
        indraft_scale = jnp.minimum(indraft_strength * 100.0, 10.0)  # Cap at 10 m/s
        indraft_u = -indraft_scale * dphi_dx / grad_mag
        indraft_v = -indraft_scale * dphi_dy / grad_mag
        
        # Only apply near fire
        near_fire = jnp.abs(phi_new) < 500.0  # Within 500m of front
        indraft_u = jnp.where(near_fire, indraft_u, 0.0)
        indraft_v = jnp.where(near_fire, indraft_v, 0.0)
    
    # Apply ember transport
    if config.enable_ember_transport:
        # Get vertical velocity column for ember lofting
        w_column = atm_state.w
        w_max_2d = jnp.max(w_column, axis=0)
        w_max_fire = interpolate_atm_to_fire(
            w_max_2d, atm_grids, (fire_ny, fire_nx), config.fire_dx
        )
        
        # Simplified ember spotting
        phi_new = compute_ember_transport(
            phi_new, fire_intensity,
            jnp.broadcast_to(w_max_fire[None, :, :], w_column.shape),
            total_u, total_v,
            config.fire_dx, fire_dt,
            config
        )
    
    # Update time
    time_new = state.time_elapsed + fire_dt
    
    return CoupledSimState(
        phi=phi_new,
        fire_ros=state.fire_ros,
        fire_intensity=fire_intensity,
        atm_state=atm_state,
        surface_wind_u=surface_wind_u,
        surface_wind_v=surface_wind_v,
        indraft_u=indraft_u,
        indraft_v=indraft_v,
        time_elapsed=time_new,
        atm_steps=atm_steps,
    )


# =============================================================================
# Main Simulation Driver
# =============================================================================

def run_coupled_simulation(
    terrain: jnp.ndarray,
    fuel_ros: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    duration_minutes: float,
    config: CoupledSimConfig,
    background_wind_speed: float = 10.0,
    background_wind_dir: float = 270.0,
    fire_dt: float = 1.0,
    store_history: bool = True,
    store_every: int = 10,
    verbose: bool = True,
) -> CoupledSimResult:
    """
    Run WRF-SFIRE style coupled fire-atmosphere simulation.
    
    Parameters
    ----------
    terrain : array (ny, nx)
        Terrain elevation (m)
    fuel_ros : array (ny, nx)
        Base rate of spread (m/min)
    x_ign, y_ign : float
        Ignition location (m)
    duration_minutes : float
        Simulation duration
    config : CoupledSimConfig
        Configuration
    background_wind_speed : float
        Initial wind speed (m/s)
    background_wind_dir : float
        Initial wind direction (degrees)
    fire_dt : float
        Fire time step (minutes)
    store_history : bool
        Store history for visualization
    store_every : int
        Store every N steps
    verbose : bool
        Print progress
        
    Returns
    -------
    CoupledSimResult
        Simulation results
    """
    if verbose:
        print("="*70)
        print("WRF-SFIRE STYLE COUPLED FIRE-ATMOSPHERE SIMULATION")
        print("="*70)
        print(f"Fire grid: {terrain.shape[1]} x {terrain.shape[0]} @ {config.fire_dx}m")
        fire_domain = (terrain.shape[1] * config.fire_dx / 1000,
                       terrain.shape[0] * config.fire_dx / 1000)
        print(f"Fire domain: {fire_domain[0]:.1f} x {fire_domain[1]:.1f} km")
        
        atm_nx = int(terrain.shape[1] * config.fire_dx / config.atm_dx)
        atm_ny = int(terrain.shape[0] * config.fire_dx / config.atm_dx)
        print(f"Atmosphere grid: {atm_nx} x {atm_ny} x {config.atm_nz}")
        print(f"Duration: {duration_minutes:.0f} minutes")
        print(f"Background wind: {background_wind_speed:.1f} m/s from {background_wind_dir:.0f}°")
        print("="*70)
    
    # Initialize
    state, atm_grids, fire_grids = initialize_coupled_simulation(
        terrain, fuel_ros, x_ign, y_ign, config,
        background_wind_speed, background_wind_dir
    )
    
    # Atmosphere parameters
    atm_params = AtmosphereParams(
        nz=config.atm_nz,
        z_top=config.atm_z_top,
        km_h=config.km_h,
        km_v=config.km_v,
        fire_heat_flux_scale=config.heat_flux_fraction,
        plume_injection_depth=config.plume_injection_depth,
    )
    
    # History storage
    phi_history = [state.phi] if store_history else None
    area_history = []
    plume_height_history = []
    max_updraft_history = []
    
    # Diagnostics
    max_indraft_seen = 0.0
    max_vorticity_seen = 0.0
    ember_spots = 0
    
    # Run simulation
    n_steps = int(duration_minutes / fire_dt)
    
    for step in range(n_steps):
        # Evolve one step
        state = evolve_coupled_step(
            state, fire_grids, atm_grids, atm_params, config, fire_dt
        )
        
        # Compute diagnostics
        fire_mask = state.phi < 0
        burned_area = float(jnp.sum(fire_mask)) * config.fire_dx * config.fire_dx
        area_history.append(burned_area)
        
        # Atmosphere diagnostics
        coupling = couple_atmosphere_to_fire(state.atm_state, atm_grids)
        plume_height = float(jnp.max(coupling.plume_top))
        plume_height_history.append(plume_height)
        
        max_updraft = float(jnp.max(state.atm_state.w))
        max_updraft_history.append(max_updraft)
        
        max_indraft_seen = max(max_indraft_seen, float(jnp.max(coupling.indraft_strength)))
        max_vorticity_seen = max(max_vorticity_seen, float(jnp.max(jnp.abs(coupling.vorticity))))
        
        # Store history
        if store_history and (step + 1) % store_every == 0:
            phi_history.append(state.phi)
        
        # Progress
        if verbose and (step + 1) % 30 == 0:
            print(f"  t={state.time_elapsed:.0f}min: Area={burned_area/10000:.1f}ha, "
                  f"Plume={plume_height:.0f}m, MaxW={max_updraft:.1f}m/s")
    
    # Final area
    final_mask = state.phi < 0
    final_area = float(jnp.sum(final_mask)) * config.fire_dx * config.fire_dx
    
    if verbose:
        print("="*70)
        print("SIMULATION COMPLETE")
        print(f"Final burned area: {final_area/10000:.1f} ha")
        print(f"Max plume height: {max(plume_height_history):.0f} m")
        print(f"Max updraft: {max(max_updraft_history):.1f} m/s")
        print(f"Max indraft: {max_indraft_seen:.3f} 1/s")
        print(f"Max vorticity: {max_vorticity_seen:.3f} 1/s")
        print("="*70)
    
    return CoupledSimResult(
        phi_final=state.phi,
        atm_state_final=state.atm_state,
        burned_area=final_area,
        phi_history=jnp.stack(phi_history) if phi_history else None,
        area_history=area_history,
        plume_height_history=plume_height_history,
        max_updraft_history=max_updraft_history,
        max_indraft=max_indraft_seen,
        max_vorticity=max_vorticity_seen,
        ember_spots=ember_spots,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_coupled_test(
    nx: int = 100,
    ny: int = 100,
    duration_minutes: float = 60.0,
    fire_intensity_kw: float = 5000.0,
    wind_speed: float = 10.0,
) -> CoupledSimResult:
    """
    Quick test of coupled simulation.
    """
    # Flat terrain
    terrain = jnp.zeros((ny, nx))
    
    # Uniform fuel
    fuel_ros = 10.0 * jnp.ones((ny, nx))  # 10 m/min base ROS
    
    # Center ignition
    x_ign = nx * 30.0 / 2
    y_ign = ny * 30.0 / 2
    
    # Config
    config = CoupledSimConfig(
        fire_dx=30.0,
        atm_dx=100.0,
        atm_nz=15,
        atm_z_top=1500.0,
        atmosphere_dt=1.0,
        max_wind_speed=30.0,
        max_updraft=20.0,
    )
    
    return run_coupled_simulation(
        terrain, fuel_ros, x_ign, y_ign,
        duration_minutes, config,
        background_wind_speed=wind_speed,
        background_wind_dir=270.0,
        fire_dt=1.0,
        verbose=True,
    )
