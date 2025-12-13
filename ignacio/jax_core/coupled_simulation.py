"""
Coupled Fire-Atmosphere Simulation Driver.

Integrates the level-set fire spread model with the 3D atmospheric solver
for fully-coupled fire-atmosphere simulations similar to WRF-SFIRE.

The coupling works bidirectionally:
1. Fire → Atmosphere: Heat release drives convection
2. Atmosphere → Fire: Modified winds affect spread rate and direction

Coupling Strategy
-----------------
- Fire model runs at fine time step (seconds)
- Atmosphere model runs at coarser step (can be same or different)
- Surface winds from atmosphere fed to fire model
- Fire intensity from fire model fed to atmosphere heat source

This creates the feedback loop responsible for:
- Fire-induced winds (indraft doubling effective wind speed)
- Plume-dominated spread
- Fire whirl generation
- Erratic fire behavior under coupled conditions
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple, Callable
from datetime import datetime, timedelta
from functools import partial
import time

import jax
import jax.numpy as jnp

# Import fire model components
from .levelset import (
    LevelSetGrids,
    initialize_phi,
    evolve_phi,
)
from .spotting import compute_fire_intensity

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
    summarize_atmosphere_state,
)


# =============================================================================
# Data Structures
# =============================================================================

class CoupledSimConfig(NamedTuple):
    """Configuration for coupled simulation."""
    
    # Coupling control
    coupling_interval: int = 1        # Fire steps per atmosphere step
    feedback_strength: float = 1.0    # Scale factor for atmosphere→fire feedback
    
    # Fire model parameters
    base_ros: float = 10.0            # Base rate of spread (m/min)
    lb_ratio: float = 2.0             # Length-to-breadth ratio
    
    # Atmosphere parameters
    atm_params: AtmosphereParams = AtmosphereParams()
    
    # Initial conditions
    u_background: float = 5.0         # Background wind u (m/s)
    v_background: float = 0.0         # Background wind v (m/s)
    theta_surface: float = 300.0      # Surface potential temperature (K)
    
    # Simulation control
    fire_dt: float = 1.0              # Fire time step (minutes)
    atm_dt: float = 2.0               # Atmosphere time step (seconds)


class CoupledSimState(NamedTuple):
    """State for coupled simulation."""
    
    # Fire state
    phi: jnp.ndarray                  # Level-set function
    fire_grids: LevelSetGrids         # Fire spread grids
    
    # Atmosphere state
    atm_state: AtmosphereState        # 3D atmospheric state
    
    # Coupling state
    surface_wind_u: jnp.ndarray       # Current surface u wind
    surface_wind_v: jnp.ndarray       # Current surface v wind
    fire_intensity: jnp.ndarray       # Current fire intensity
    
    # Time tracking
    fire_time: float                  # Fire simulation time (minutes)
    atm_time: float                   # Atmosphere simulation time (seconds)


class CoupledSimResult(NamedTuple):
    """Results from coupled simulation."""
    
    # Final states
    phi_final: jnp.ndarray
    atm_state_final: AtmosphereState
    
    # Fire metrics
    burned_area: float
    max_ros: float
    
    # Atmosphere metrics
    max_updraft: float
    max_surface_wind: float
    max_plume_height: float
    
    # History (optional)
    phi_history: Optional[jnp.ndarray] = None
    wind_history: Optional[list] = None
    intensity_history: Optional[list] = None


# =============================================================================
# Grid Compatibility
# =============================================================================

def create_compatible_grids(
    nx: int,
    ny: int,
    dx: float,
    terrain: jnp.ndarray,
    config: CoupledSimConfig,
) -> Tuple[AtmosphereGrids, jnp.ndarray, jnp.ndarray]:
    """
    Create atmosphere grids compatible with fire grid.
    
    The atmosphere and fire models share the same horizontal grid.
    
    Parameters
    ----------
    nx, ny : int
        Horizontal dimensions
    dx : float
        Grid spacing (m)
    terrain : array (ny, nx)
        Terrain height
    config : CoupledSimConfig
        Configuration
        
    Returns
    -------
    atm_grids : AtmosphereGrids
        Atmosphere grid structure
    x_coords : array
        X coordinates for fire grid
    y_coords : array
        Y coordinates for fire grid
    """
    # Atmosphere grids
    atm_grids = create_atmosphere_grids(
        nx, ny, dx, dx, terrain, config.atm_params
    )
    
    # Fire grid coordinates
    x_coords = jnp.arange(nx) * dx
    y_coords = jnp.arange(ny) * dx
    
    return atm_grids, x_coords, y_coords


# =============================================================================
# Fire-Atmosphere Coupling Functions
# =============================================================================

@jax.jit
def compute_coupled_fire_intensity(
    phi: jnp.ndarray,
    ros: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """
    Compute fire intensity for atmosphere coupling.
    
    Fire intensity I = H * w * R where:
    - H = heat content (~18000 kJ/kg)
    - w = fuel load (kg/m²)
    - R = rate of spread (m/s)
    
    For coupling, we compute intensity only at the fire front.
    
    Parameters
    ----------
    phi : array
        Level-set function (negative = burned)
    ros : array
        Rate of spread (m/min)
    dx : float
        Grid spacing
        
    Returns
    -------
    intensity : array
        Fire line intensity (kW/m)
    """
    # Find fire front (where phi crosses zero)
    # Approximate using gradient magnitude near zero crossing
    
    # Cells that are burning (phi < 0)
    burned = phi < 0
    
    # Cells adjacent to unburned (fire front)
    burned_padded = jnp.pad(burned, 1, mode='constant', constant_values=False)
    unburned_neighbor = (
        ~burned_padded[:-2, 1:-1] |  # North
        ~burned_padded[2:, 1:-1] |   # South
        ~burned_padded[1:-1, :-2] |  # West
        ~burned_padded[1:-1, 2:]     # East
    )
    
    fire_front = burned & unburned_neighbor
    
    # Fire intensity parameters
    heat_content = 18000.0  # kJ/kg
    fuel_load = 1.5         # kg/m² (typical forest)
    
    # Convert ROS from m/min to m/s
    ros_ms = ros / 60.0
    
    # Intensity at fire front
    intensity = jnp.where(
        fire_front,
        heat_content * fuel_load * ros_ms,
        0.0
    )
    
    return intensity


@jax.jit
def update_fire_winds_from_atmosphere(
    fire_grids: LevelSetGrids,
    coupling: CouplingResult,
    background_u: float,
    background_v: float,
    feedback_strength: float,
) -> LevelSetGrids:
    """
    Update fire model winds from atmosphere coupling.
    
    Blends background wind with atmosphere-derived surface wind,
    scaled by feedback strength.
    
    Parameters
    ----------
    fire_grids : LevelSetGrids
        Current fire grids
    coupling : CouplingResult
        Atmosphere coupling result
    background_u, background_v : float
        Background wind components (m/s)
    feedback_strength : float
        Coupling strength (0 = no feedback, 1 = full)
        
    Returns
    -------
    updated_grids : LevelSetGrids
        Fire grids with updated wind
    """
    # Get coupled winds, clamping to reasonable values
    u_coupled = jnp.clip(coupling.u_surface, -50.0, 50.0)
    v_coupled = jnp.clip(coupling.v_surface, -50.0, 50.0)
    
    # Replace any NaN/inf with background
    u_coupled = jnp.where(jnp.isfinite(u_coupled), u_coupled, background_u)
    v_coupled = jnp.where(jnp.isfinite(v_coupled), v_coupled, background_v)
    
    # Blend background and coupled winds
    u_blend = background_u * (1 - feedback_strength) + u_coupled * feedback_strength
    v_blend = background_v * (1 - feedback_strength) + v_coupled * feedback_strength
    
    # Compute wind speed and direction
    wind_speed = jnp.sqrt(u_blend**2 + v_blend**2)
    wind_dir = jnp.arctan2(v_blend, u_blend)  # Direction wind is GOING TO
    
    # The fire spread direction (raz) is the direction the fire spreads
    # which is the same as the wind direction for wind-driven fires
    raz_new = wind_dir
    
    # Update ROS based on wind speed change (conservative)
    # ROS increases with wind speed but cap the enhancement
    background_speed = jnp.sqrt(background_u**2 + background_v**2) + 1e-6
    wind_factor = wind_speed / background_speed
    wind_factor = jnp.clip(wind_factor, 0.5, 2.0)  # Max 2x enhancement
    
    ros_new = fire_grids.ros * wind_factor
    
    # Also update bros and fros proportionally
    bros_new = fire_grids.bros * wind_factor
    fros_new = fire_grids.fros * wind_factor
    
    return fire_grids._replace(
        ros=ros_new,
        bros=bros_new,
        fros=fros_new,
        raz=raz_new,
    )


# =============================================================================
# Initialization
# =============================================================================

def initialize_coupled_simulation(
    nx: int,
    ny: int,
    dx: float,
    terrain: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    config: CoupledSimConfig,
) -> CoupledSimState:
    """
    Initialize coupled fire-atmosphere simulation.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    dx : float
        Grid spacing (m)
    terrain : array
        Terrain height (m)
    x_ign, y_ign : float
        Ignition coordinates
    config : CoupledSimConfig
        Configuration
        
    Returns
    -------
    CoupledSimState
        Initialized state
    """
    # Create compatible grids
    atm_grids, x_coords, y_coords = create_compatible_grids(
        nx, ny, dx, terrain, config
    )
    
    # Initialize atmosphere
    atm_state = initialize_atmosphere(
        atm_grids, config.atm_params,
        config.u_background, config.v_background,
        config.theta_surface
    )
    
    # Initialize fire level-set
    phi = initialize_phi(x_coords, y_coords, x_ign, y_ign, initial_radius=30.0)
    
    # Initial fire grids (uniform)
    ros = config.base_ros * jnp.ones((ny, nx))
    bros = ros * 0.2  # Back fire is slower
    fros = ros * 0.5  # Flank fire intermediate
    raz = jnp.arctan2(config.v_background, config.u_background) * jnp.ones((ny, nx))
    
    fire_grids = LevelSetGrids(
        x_coords=x_coords,
        y_coords=y_coords,
        ros=ros,
        bros=bros,
        fros=fros,
        raz=raz,
    )
    
    # Initial coupling (no fire yet, so background winds)
    surface_u = config.u_background * jnp.ones((ny, nx))
    surface_v = config.v_background * jnp.ones((ny, nx))
    fire_intensity = jnp.zeros((ny, nx))
    
    return CoupledSimState(
        phi=phi,
        fire_grids=fire_grids,
        atm_state=atm_state,
        surface_wind_u=surface_u,
        surface_wind_v=surface_v,
        fire_intensity=fire_intensity,
        fire_time=0.0,
        atm_time=0.0,
    )


# =============================================================================
# Time Stepping
# =============================================================================

def evolve_coupled_step(
    state: CoupledSimState,
    atm_grids: AtmosphereGrids,
    config: CoupledSimConfig,
) -> CoupledSimState:
    """
    Evolve coupled simulation by one fire time step.
    
    This runs multiple atmosphere steps per fire step.
    
    Parameters
    ----------
    state : CoupledSimState
        Current state
    atm_grids : AtmosphereGrids
        Atmosphere grid structure
    config : CoupledSimConfig
        Configuration
        
    Returns
    -------
    CoupledSimState
        Updated state
    """
    # --- Fire Model Step ---
    
    # Evolve fire level-set
    phi_new = evolve_phi(
        state.phi, state.fire_grids, 
        t_idx=0,  # Time-invariant grids for now
        dt=config.fire_dt
    )
    
    # Compute fire intensity
    fire_intensity = compute_coupled_fire_intensity(
        phi_new, state.fire_grids.ros, atm_grids.dx
    )
    
    # --- Atmosphere Model Steps ---
    
    # Number of atmosphere steps per fire step
    fire_dt_seconds = config.fire_dt * 60.0  # Convert to seconds
    n_atm_steps = max(1, int(fire_dt_seconds / config.atm_dt))
    atm_dt_actual = fire_dt_seconds / n_atm_steps
    
    atm_state = state.atm_state
    
    for _ in range(n_atm_steps):
        atm_state = evolve_atmosphere_step(
            atm_state, atm_grids, config.atm_params,
            fire_intensity, atm_dt_actual
        )
    
    # --- Coupling: Atmosphere → Fire ---
    
    coupling = couple_atmosphere_to_fire(atm_state, atm_grids)
    
    # Clamp coupling outputs to prevent runaway
    safe_u = jnp.clip(coupling.u_surface, -50.0, 50.0)
    safe_v = jnp.clip(coupling.v_surface, -50.0, 50.0)
    safe_u = jnp.where(jnp.isfinite(safe_u), safe_u, config.u_background)
    safe_v = jnp.where(jnp.isfinite(safe_v), safe_v, config.v_background)
    
    # Update fire winds from atmosphere
    fire_grids_new = update_fire_winds_from_atmosphere(
        state.fire_grids, coupling,
        config.u_background, config.v_background,
        config.feedback_strength
    )
    
    # Update times
    fire_time_new = state.fire_time + config.fire_dt
    atm_time_new = state.atm_time + fire_dt_seconds
    
    return CoupledSimState(
        phi=phi_new,
        fire_grids=fire_grids_new,
        atm_state=atm_state,
        surface_wind_u=safe_u,
        surface_wind_v=safe_v,
        fire_intensity=fire_intensity,
        fire_time=fire_time_new,
        atm_time=atm_time_new,
    )


# =============================================================================
# Main Simulation Driver
# =============================================================================

def run_coupled_simulation(
    nx: int,
    ny: int,
    dx: float,
    terrain: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    duration_minutes: float,
    config: CoupledSimConfig = None,
    store_history: bool = True,
    store_every: int = 10,
    verbose: bool = True,
) -> CoupledSimResult:
    """
    Run coupled fire-atmosphere simulation.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    dx : float
        Grid spacing (m)
    terrain : array
        Terrain height (m)
    x_ign, y_ign : float
        Ignition coordinates
    duration_minutes : float
        Simulation duration (minutes)
    config : CoupledSimConfig, optional
        Configuration (defaults provided)
    store_history : bool
        Whether to store history
    store_every : int
        Store every N steps
    verbose : bool
        Print progress
        
    Returns
    -------
    CoupledSimResult
        Simulation results
    """
    if config is None:
        config = CoupledSimConfig()
    
    if verbose:
        print("="*70)
        print("COUPLED FIRE-ATMOSPHERE SIMULATION (WRF-SFIRE-style)")
        print("="*70)
        print(f"Grid: {nx} x {ny} x {config.atm_params.nz} cells")
        print(f"Horizontal: {nx * dx/1000:.1f} x {ny * dx/1000:.1f} km")
        print(f"Vertical: {config.atm_params.z_top/1000:.1f} km")
        print(f"Duration: {duration_minutes:.0f} minutes")
        print(f"Fire dt: {config.fire_dt:.1f} min, Atm dt: {config.atm_dt:.1f} s")
        print(f"Feedback strength: {config.feedback_strength:.1%}")
        print("="*70)
    
    # Create grids
    atm_grids, _, _ = create_compatible_grids(nx, ny, dx, terrain, config)
    
    # Initialize
    state = initialize_coupled_simulation(
        nx, ny, dx, terrain, x_ign, y_ign, config
    )
    
    # Storage
    n_steps = int(duration_minutes / config.fire_dt)
    phi_history = [state.phi] if store_history else None
    wind_history = []
    intensity_history = []
    
    # Time loop
    t_start = time.time()
    
    for step in range(n_steps):
        # Evolve
        state = evolve_coupled_step(state, atm_grids, config)
        
        # Store history
        if store_history and (step + 1) % store_every == 0:
            phi_history.append(state.phi)
            wind_history.append({
                'u': state.surface_wind_u.copy(),
                'v': state.surface_wind_v.copy(),
            })
            intensity_history.append(float(jnp.max(state.fire_intensity)))
        
        # Progress
        if verbose and (step + 1) % 10 == 0:
            coupling = couple_atmosphere_to_fire(state.atm_state, atm_grids)
            burned_area = float(jnp.sum(state.phi < 0)) * dx * dx / 10000.0
            max_w = float(jnp.max(coupling.w_max))
            max_wind = float(jnp.max(coupling.wind_speed))
            
            print(f"  t={state.fire_time:.0f} min: "
                  f"Area={burned_area:.1f} ha, "
                  f"max_w={max_w:.1f} m/s, "
                  f"max_sfc_wind={max_wind:.1f} m/s")
    
    elapsed = time.time() - t_start
    
    # Final metrics
    coupling_final = couple_atmosphere_to_fire(state.atm_state, atm_grids)
    burned_area = float(jnp.sum(state.phi < 0)) * dx * dx
    
    if verbose:
        print("="*70)
        print("SIMULATION COMPLETE")
        print(f"Wall time: {elapsed:.1f} s ({elapsed/n_steps*1000:.1f} ms/step)")
        print(f"Burned area: {burned_area/10000:.1f} ha")
        print(f"Max updraft: {float(jnp.max(coupling_final.w_max)):.1f} m/s")
        print(f"Max surface wind: {float(jnp.max(coupling_final.wind_speed)):.1f} m/s")
        print(f"Max plume height: {float(jnp.max(coupling_final.plume_top)):.0f} m")
        print("="*70)
    
    return CoupledSimResult(
        phi_final=state.phi,
        atm_state_final=state.atm_state,
        burned_area=burned_area,
        max_ros=float(jnp.max(state.fire_grids.ros)),
        max_updraft=float(jnp.max(coupling_final.w_max)),
        max_surface_wind=float(jnp.max(coupling_final.wind_speed)),
        max_plume_height=float(jnp.max(coupling_final.plume_top)),
        phi_history=jnp.stack(phi_history) if phi_history else None,
        wind_history=wind_history if store_history else None,
        intensity_history=intensity_history if store_history else None,
    )


# =============================================================================
# Quick Test Function
# =============================================================================

def quick_coupled_test(
    nx: int = 100,
    ny: int = 100,
    dx: float = 50.0,
    duration_minutes: float = 30.0,
    feedback_strength: float = 0.5,
) -> CoupledSimResult:
    """
    Quick test of coupled fire-atmosphere simulation.
    
    Parameters
    ----------
    nx, ny : int
        Grid size
    dx : float
        Grid spacing (m)
    duration_minutes : float
        Duration (minutes)
    feedback_strength : float
        Atmosphere→fire feedback strength
        
    Returns
    -------
    CoupledSimResult
        Simulation results
    """
    # Flat terrain
    terrain = jnp.zeros((ny, nx))
    
    # Ignition at center
    x_ign = nx * dx / 2
    y_ign = ny * dx / 2
    
    # Configuration
    config = CoupledSimConfig(
        feedback_strength=feedback_strength,
        base_ros=10.0,
        u_background=5.0,
        v_background=0.0,
        fire_dt=1.0,
        atm_dt=2.0,
        atm_params=AtmosphereParams(
            nz=15,
            z_top=1500.0,
        ),
    )
    
    return run_coupled_simulation(
        nx, ny, dx, terrain, x_ign, y_ign,
        duration_minutes, config,
        verbose=True,
    )


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_coupled_vs_uncoupled(
    nx: int = 80,
    ny: int = 80,
    dx: float = 50.0,
    duration_minutes: float = 30.0,
) -> dict:
    """
    Compare coupled vs uncoupled fire simulation.
    
    Demonstrates the impact of fire-atmosphere feedback.
    
    Returns
    -------
    dict
        Comparison metrics
    """
    print("\n" + "="*70)
    print("COMPARISON: COUPLED vs UNCOUPLED FIRE SIMULATION")
    print("="*70)
    
    # Common setup
    terrain = jnp.zeros((ny, nx))
    x_ign = nx * dx / 2
    y_ign = ny * dx / 2
    
    # Run uncoupled (feedback_strength = 0)
    print("\n--- UNCOUPLED (no atmosphere feedback) ---")
    config_uncoupled = CoupledSimConfig(
        feedback_strength=0.0,
        base_ros=10.0,
        fire_dt=1.0,
        atm_params=AtmosphereParams(nz=10, z_top=1000.0),
    )
    result_uncoupled = run_coupled_simulation(
        nx, ny, dx, terrain, x_ign, y_ign,
        duration_minutes, config_uncoupled,
        store_history=False, verbose=True,
    )
    
    # Run coupled
    print("\n--- COUPLED (full atmosphere feedback) ---")
    config_coupled = CoupledSimConfig(
        feedback_strength=1.0,
        base_ros=10.0,
        fire_dt=1.0,
        atm_params=AtmosphereParams(nz=10, z_top=1000.0),
    )
    result_coupled = run_coupled_simulation(
        nx, ny, dx, terrain, x_ign, y_ign,
        duration_minutes, config_coupled,
        store_history=False, verbose=True,
    )
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"                    Uncoupled    Coupled     Change")
    print(f"Burned area (ha):   {result_uncoupled.burned_area/10000:8.1f}    "
          f"{result_coupled.burned_area/10000:8.1f}    "
          f"{(result_coupled.burned_area/result_uncoupled.burned_area-1)*100:+6.1f}%")
    print(f"Max ROS (m/min):    {result_uncoupled.max_ros:8.1f}    "
          f"{result_coupled.max_ros:8.1f}    "
          f"{(result_coupled.max_ros/result_uncoupled.max_ros-1)*100:+6.1f}%")
    print(f"Max updraft (m/s):  {result_uncoupled.max_updraft:8.1f}    "
          f"{result_coupled.max_updraft:8.1f}")
    print(f"Max sfc wind (m/s): {result_uncoupled.max_surface_wind:8.1f}    "
          f"{result_coupled.max_surface_wind:8.1f}")
    print("="*70)
    
    return {
        'uncoupled': result_uncoupled,
        'coupled': result_coupled,
        'area_increase_pct': (result_coupled.burned_area / result_uncoupled.burned_area - 1) * 100,
        'ros_increase_pct': (result_coupled.max_ros / result_uncoupled.max_ros - 1) * 100,
    }
