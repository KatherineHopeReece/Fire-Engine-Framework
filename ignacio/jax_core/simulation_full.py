"""
Fully-Integrated Fire Simulation with All Physics Modules.

This module provides a comprehensive fire simulation that integrates:
1. Level-set fire spread (topology-robust)
2. Solar radiation and fuel conditioning
3. Fuel moisture lag dynamics
4. Crown fire transition (Van Wagner)
5. Mass-conserving terrain wind
6. Fire-atmosphere coupling
7. Eruptive fire behavior (canyon effect)
8. Dynamic phenology (live fuel curing)
9. Smoke transport (advection-diffusion)
10. Ember spotting

This is the production-ready simulation engine for operational fire
spread prediction including all physical processes.
"""

from __future__ import annotations
from typing import NamedTuple, Optional
from datetime import datetime, timedelta
from functools import partial

import jax
import jax.numpy as jnp

# Import all physics modules
from .levelset import (
    LevelSetGrids,
    initialize_phi,
    compute_gradient_magnitude_upwind,
    evolve_phi,
)
from .solar_radiation import (
    compute_sun_position,
    compute_hillshade,
    compute_fuel_conditioning,
    FuelConditioningParams,
)
from .moisture_lag import (
    MoistureState,
    MoistureLagParams,
    initialize_moisture_state,
    update_moisture_euler,
)
from .crown_fire import (
    CrownFireParams,
    CrownFireState,
    compute_total_ros_with_crown,
)
from .wind_solver import (
    WindField,
    WindSolverParams,
    solve_wind_field,
)
from .fire_atmosphere import (
    FireAtmosphereParams,
    couple_wind_to_fire,
)
from .eruptive_fire import (
    EruptiveParams,
    EruptiveState,
    initialize_eruptive_state,
    update_eruptive_state,
    compute_eruptive_ros,
)
from .dynamic_phenology import (
    PhenologyParams,
    PhenologyState,
    initialize_phenology_state,
    update_phenology_state,
    compute_phenology_effects,
    apply_curing_to_grass_fuels,
)
from .smoke_transport import (
    SmokeParams,
    SmokeState,
    initialize_smoke_state,
    update_smoke_state,
    compute_smoke_impacts,
)
from .spotting import (
    SpottingParams,
    compute_fire_intensity,
    apply_spotting,
)


# =============================================================================
# Configuration
# =============================================================================

class FullSimConfig(NamedTuple):
    """Configuration for fully-integrated simulation."""
    
    # Module toggles
    enable_solar: bool = True
    enable_moisture_lag: bool = True
    enable_crown_fire: bool = True
    enable_terrain_wind: bool = True
    enable_fire_atmosphere: bool = False
    enable_eruptive: bool = True
    enable_phenology: bool = True
    enable_smoke: bool = True
    enable_spotting: bool = False
    
    # Module parameters
    solar_params: FuelConditioningParams = FuelConditioningParams()
    moisture_params: MoistureLagParams = MoistureLagParams()
    crown_params: CrownFireParams = CrownFireParams()
    wind_params: WindSolverParams = WindSolverParams()
    atmosphere_params: FireAtmosphereParams = FireAtmosphereParams()
    eruptive_params: EruptiveParams = EruptiveParams()
    phenology_params: PhenologyParams = PhenologyParams()
    smoke_params: SmokeParams = SmokeParams()
    spotting_params: SpottingParams = SpottingParams()
    
    # Weather update frequency (minutes)
    weather_update_interval: float = 60.0


class FullSimState(NamedTuple):
    """Complete state for integrated simulation."""
    
    # Core state
    phi: jnp.ndarray              # Level-set function
    time_elapsed: float           # Simulation time (minutes)
    
    # Physics module states
    moisture: Optional[MoistureState] = None
    wind: Optional[WindField] = None
    eruptive: Optional[EruptiveState] = None
    phenology: Optional[PhenologyState] = None
    smoke: Optional[SmokeState] = None
    
    # Tracking
    weather_last_updated: float = 0.0
    ros_history: Optional[jnp.ndarray] = None


class FullSimResult(NamedTuple):
    """Complete results from integrated simulation."""
    
    # Final state
    phi_final: jnp.ndarray
    burned_area: float
    
    # Time series
    phi_history: Optional[jnp.ndarray] = None
    area_history: Optional[list] = None
    
    # Final physics states
    moisture_final: Optional[MoistureState] = None
    wind_final: Optional[WindField] = None
    eruptive_final: Optional[EruptiveState] = None
    phenology_final: Optional[PhenologyState] = None
    smoke_final: Optional[SmokeState] = None
    
    # Diagnostics
    eruptive_warnings: Optional[jnp.ndarray] = None
    smoke_pm25: Optional[jnp.ndarray] = None
    smoke_visibility: Optional[jnp.ndarray] = None
    crown_fraction: Optional[jnp.ndarray] = None


# =============================================================================
# Initialization
# =============================================================================

def initialize_full_simulation(
    grids: LevelSetGrids,
    dem: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    config: FullSimConfig,
    simulation_datetime: Optional[datetime] = None,
    latitude: float = 51.0,
    temperature: float = 25.0,
    relative_humidity: float = 30.0,
    ffmc_initial: float = 85.0,
) -> FullSimState:
    """
    Initialize all state for integrated simulation.
    
    Parameters
    ----------
    grids : LevelSetGrids
        Base fire spread grids
    dem : array
        Digital elevation model
    x_ign, y_ign : float
        Ignition coordinates
    config : FullSimConfig
        Simulation configuration
    simulation_datetime : datetime, optional
        Start date/time
    latitude : float
        Site latitude for solar calculations
    temperature : float
        Initial temperature (°C)
    relative_humidity : float
        Initial relative humidity (%)
    ffmc_initial : float
        Initial Fine Fuel Moisture Code
        
    Returns
    -------
    FullSimState
        Initialized simulation state
    """
    shape = dem.shape
    dx = grids.dx
    
    if simulation_datetime is None:
        simulation_datetime = datetime.now()
    
    # Initialize level-set
    phi = initialize_phi(shape, x_ign, y_ign, dx, radius=30.0)
    
    # Initialize moisture state
    moisture = None
    if config.enable_moisture_lag:
        moisture = initialize_moisture_state(
            shape, ffmc_initial, temperature, relative_humidity,
            config.moisture_params
        )
    
    # Initialize wind field
    wind = None
    if config.enable_terrain_wind:
        wind = solve_wind_field(
            dem, dx,
            background_speed=10.0,
            background_direction=270.0,
            params=config.wind_params,
        )
    
    # Initialize eruptive state
    eruptive = None
    if config.enable_eruptive:
        eruptive = initialize_eruptive_state(
            shape, dem, dx, config.eruptive_params
        )
    
    # Initialize phenology state
    phenology = None
    if config.enable_phenology:
        phenology = initialize_phenology_state(
            shape, dem, simulation_datetime, config.phenology_params
        )
    
    # Initialize smoke state
    smoke = None
    if config.enable_smoke:
        smoke = initialize_smoke_state(shape)
    
    return FullSimState(
        phi=phi,
        time_elapsed=0.0,
        moisture=moisture,
        wind=wind,
        eruptive=eruptive,
        phenology=phenology,
        smoke=smoke,
        weather_last_updated=0.0,
        ros_history=jnp.zeros(shape),
    )


# =============================================================================
# Single Step Update
# =============================================================================

def evolve_full_step(
    state: FullSimState,
    grids: LevelSetGrids,
    dem: jnp.ndarray,
    slope: jnp.ndarray,
    aspect: jnp.ndarray,
    fuel_type: jnp.ndarray,
    config: FullSimConfig,
    dt: float,
    current_datetime: datetime,
    latitude: float,
    longitude: float,
    temperature: float,
    relative_humidity: float,
    wind_speed: float,
    wind_direction: float,
    precipitation: float = 0.0,
    canopy_base_height: Optional[jnp.ndarray] = None,
    canopy_bulk_density: Optional[jnp.ndarray] = None,
) -> FullSimState:
    """
    Evolve simulation by one timestep with all physics.
    
    This is the main integration routine that applies all enabled
    physics modules in the correct sequence.
    """
    phi = state.phi
    dx = grids.dx
    ny, nx = phi.shape
    
    fire_mask = phi < 0
    time_new = state.time_elapsed + dt
    
    # -------------------------------------------------------------------------
    # 1. Update phenology (daily-scale, but check each step)
    # -------------------------------------------------------------------------
    phenology = state.phenology
    if config.enable_phenology and phenology is not None:
        day_of_year = current_datetime.timetuple().tm_yday
        
        # Get solar radiation for phenology
        sun = compute_sun_position(current_datetime, latitude, longitude)
        hillshade = compute_hillshade(dem, sun, dx)
        solar_radiation = hillshade * 800.0 * max(0, sun.elevation / 90.0)
        
        phenology = update_phenology_state(
            phenology, dem, aspect, temperature, solar_radiation,
            precipitation, day_of_year, dt / 60.0,  # hours
            None,  # No NDVI
            config.phenology_params,
        )
    
    # -------------------------------------------------------------------------
    # 2. Update moisture state
    # -------------------------------------------------------------------------
    moisture = state.moisture
    if config.enable_moisture_lag and moisture is not None:
        moisture = update_moisture_euler(
            moisture, temperature, relative_humidity, dt,
            config.moisture_params
        )
    
    # -------------------------------------------------------------------------
    # 3. Update wind field (periodically or when conditions change)
    # -------------------------------------------------------------------------
    wind = state.wind
    if config.enable_terrain_wind:
        # Update wind if interval passed
        if time_new - state.weather_last_updated >= config.weather_update_interval:
            wind = solve_wind_field(
                dem, dx, wind_speed, wind_direction,
                config.wind_params
            )
    
    # -------------------------------------------------------------------------
    # 4. Compute base ROS
    # -------------------------------------------------------------------------
    ros = grids.ros.copy()
    
    # Apply solar conditioning
    if config.enable_solar:
        sun = compute_sun_position(current_datetime, latitude, longitude)
        hillshade = compute_hillshade(dem, sun, dx)
        moisture_adj = compute_fuel_conditioning(
            dem, aspect, sun, hillshade, 
            grids.ffmc * jnp.ones((ny, nx)) if hasattr(grids, 'ffmc') else 85.0 * jnp.ones((ny, nx)),
            config.solar_params
        )
        # Adjust ROS based on moisture change
        ros = ros * (1.0 - 0.02 * moisture_adj)
    
    # Apply phenology effects
    if config.enable_phenology and phenology is not None:
        phenology_effects = compute_phenology_effects(phenology, config.phenology_params)
        ros = apply_curing_to_grass_fuels(
            ros, fuel_type, phenology_effects.curing_grid
        )
        # Apply general moisture effect
        ros = ros * phenology_effects.fuel_moisture_modifier
    
    # -------------------------------------------------------------------------
    # 5. Apply wind field to ROS direction
    # -------------------------------------------------------------------------
    ros_dir = grids.ros_dir
    if config.enable_terrain_wind and wind is not None:
        # Use terrain-adjusted wind direction
        ros_dir = wind.direction
        
        # Apply fire-atmosphere coupling
        if config.enable_fire_atmosphere:
            coupled = couple_wind_to_fire(
                phi, wind, grids.ros, dx, config.atmosphere_params
            )
            ros_dir = coupled.direction
    
    # -------------------------------------------------------------------------
    # 6. Apply crown fire
    # -------------------------------------------------------------------------
    if config.enable_crown_fire:
        # Default canopy parameters if not provided
        if canopy_base_height is None:
            cbh = config.crown_params.default_canopy_base_height * jnp.ones((ny, nx))
        else:
            cbh = canopy_base_height
            
        if canopy_bulk_density is None:
            cbd = config.crown_params.default_canopy_bulk_density * jnp.ones((ny, nx))
        else:
            cbd = canopy_bulk_density
        
        ros = compute_total_ros_with_crown(
            ros, cbh, cbd, 100.0,  # FMC
            config.crown_params
        )
    
    # -------------------------------------------------------------------------
    # 7. Apply eruptive fire multiplier
    # -------------------------------------------------------------------------
    eruptive = state.eruptive
    if config.enable_eruptive and eruptive is not None:
        # Get wind speed from wind field or background
        if wind is not None:
            ws = wind.speed
        else:
            ws = wind_speed * jnp.ones((ny, nx))
        
        # Update eruptive state
        eruptive = update_eruptive_state(
            eruptive, slope, aspect, ws / 3.6, wind_direction,
            ros, fire_mask, config.eruptive_params, dt
        )
        
        # Compute ROS multiplier
        eruptive_result = compute_eruptive_ros(
            ros, eruptive, slope, aspect, ws / 3.6, wind_direction,
            config.eruptive_params
        )
        ros = ros * eruptive_result.ros_multiplier
    
    # -------------------------------------------------------------------------
    # 8. Evolve level-set
    # -------------------------------------------------------------------------
    # Create updated grids
    updated_grids = LevelSetGrids(
        ros=ros,
        ros_dir=ros_dir,
        lb_ratio=grids.lb_ratio,
        dx=dx,
    )
    
    phi_new = evolve_phi(phi, updated_grids, dt)
    
    # Apply spotting
    if config.enable_spotting:
        intensity = compute_fire_intensity(ros, fire_mask)
        phi_new = apply_spotting(
            phi_new, intensity, wind_speed / 3.6, wind_direction,
            dx, dt, config.spotting_params
        )
    
    # -------------------------------------------------------------------------
    # 9. Update smoke transport
    # -------------------------------------------------------------------------
    smoke = state.smoke
    if config.enable_smoke and smoke is not None:
        # Get wind components
        if wind is not None:
            wind_u = wind.speed * jnp.cos(jnp.deg2rad(270.0 - wind.direction))
            wind_v = wind.speed * jnp.sin(jnp.deg2rad(270.0 - wind.direction))
        else:
            wind_u = wind_speed / 3.6 * jnp.cos(jnp.deg2rad(270.0 - wind_direction))
            wind_v = wind_speed / 3.6 * jnp.sin(jnp.deg2rad(270.0 - wind_direction))
            wind_u = wind_u * jnp.ones((ny, nx))
            wind_v = wind_v * jnp.ones((ny, nx))
        
        # Fire intensity for smoke emission
        fire_intensity = compute_fire_intensity(ros, fire_mask)
        
        smoke = update_smoke_state(
            smoke, wind_u, wind_v, ros, fire_intensity,
            fire_mask, dx, dt * 60.0,  # Convert to seconds
            precipitation,
            config.smoke_params
        )
    
    # -------------------------------------------------------------------------
    # Return updated state
    # -------------------------------------------------------------------------
    weather_updated = state.weather_last_updated
    if time_new - state.weather_last_updated >= config.weather_update_interval:
        weather_updated = time_new
    
    return FullSimState(
        phi=phi_new,
        time_elapsed=time_new,
        moisture=moisture,
        wind=wind,
        eruptive=eruptive,
        phenology=phenology,
        smoke=smoke,
        weather_last_updated=weather_updated,
        ros_history=ros,
    )


# =============================================================================
# Main Simulation Driver
# =============================================================================

def run_full_simulation(
    grids: LevelSetGrids,
    dem: jnp.ndarray,
    slope: jnp.ndarray,
    aspect: jnp.ndarray,
    fuel_type: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    n_steps: int,
    dt: float,
    config: FullSimConfig,
    simulation_datetime: datetime,
    latitude: float,
    longitude: float,
    temperature: float = 25.0,
    relative_humidity: float = 30.0,
    wind_speed: float = 20.0,
    wind_direction: float = 270.0,
    ffmc_initial: float = 85.0,
    precipitation: float = 0.0,
    canopy_base_height: Optional[jnp.ndarray] = None,
    canopy_bulk_density: Optional[jnp.ndarray] = None,
    store_history: bool = True,
    store_every: int = 10,
    verbose: bool = True,
) -> FullSimResult:
    """
    Run complete fire simulation with all physics modules.
    
    Parameters
    ----------
    grids : LevelSetGrids
        Base fire spread grids (ROS, direction, L/B ratio)
    dem : array
        Digital elevation model
    slope : array
        Terrain slope (degrees)
    aspect : array
        Terrain aspect (degrees from north)
    fuel_type : array
        FBP fuel type codes
    x_ign, y_ign : float
        Ignition coordinates (grid units)
    n_steps : int
        Number of simulation steps
    dt : float
        Time step (minutes)
    config : FullSimConfig
        Configuration for all physics modules
    simulation_datetime : datetime
        Start date/time
    latitude, longitude : float
        Site location for solar calculations
    temperature : float
        Air temperature (°C)
    relative_humidity : float
        Relative humidity (%)
    wind_speed : float
        Wind speed (km/h)
    wind_direction : float
        Wind direction (degrees from north)
    ffmc_initial : float
        Initial Fine Fuel Moisture Code
    precipitation : float
        Precipitation rate (mm/hr)
    canopy_base_height : array, optional
        Canopy base height grid (m)
    canopy_bulk_density : array, optional
        Canopy bulk density grid (kg/m³)
    store_history : bool
        Whether to store phi history
    store_every : int
        Store state every N steps
    verbose : bool
        Print progress
        
    Returns
    -------
    FullSimResult
        Complete simulation results
    """
    if verbose:
        print("="*60)
        print("IGNACIO FULL FIRE SIMULATION")
        print("="*60)
        print(f"Grid: {dem.shape[0]} x {dem.shape[1]}")
        print(f"Duration: {n_steps * dt:.0f} minutes ({n_steps} steps)")
        print(f"Start: {simulation_datetime}")
        print("\nEnabled physics:")
        print(f"  Solar radiation: {config.enable_solar}")
        print(f"  Moisture lag: {config.enable_moisture_lag}")
        print(f"  Crown fire: {config.enable_crown_fire}")
        print(f"  Terrain wind: {config.enable_terrain_wind}")
        print(f"  Fire-atmosphere: {config.enable_fire_atmosphere}")
        print(f"  Eruptive fire: {config.enable_eruptive}")
        print(f"  Phenology: {config.enable_phenology}")
        print(f"  Smoke transport: {config.enable_smoke}")
        print(f"  Spotting: {config.enable_spotting}")
        print("="*60)
    
    # Initialize
    state = initialize_full_simulation(
        grids, dem, x_ign, y_ign, config,
        simulation_datetime, latitude,
        temperature, relative_humidity, ffmc_initial
    )
    
    # Storage
    phi_history = [state.phi] if store_history else None
    area_history = []
    
    # Run simulation
    current_datetime = simulation_datetime
    
    for step in range(n_steps):
        # Update time
        current_datetime = simulation_datetime + timedelta(minutes=state.time_elapsed + dt)
        
        # Evolve one step
        state = evolve_full_step(
            state, grids, dem, slope, aspect, fuel_type,
            config, dt, current_datetime, latitude, longitude,
            temperature, relative_humidity, wind_speed, wind_direction,
            precipitation, canopy_base_height, canopy_bulk_density
        )
        
        # Compute area
        fire_mask = state.phi < 0
        burned_area = float(jnp.sum(fire_mask)) * grids.dx * grids.dx
        area_history.append(burned_area)
        
        # Store history
        if store_history and (step + 1) % store_every == 0:
            phi_history.append(state.phi)
        
        # Progress
        if verbose and (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{n_steps}: {state.time_elapsed:.0f} min, "
                  f"Area: {burned_area/10000:.2f} ha")
    
    # Final area
    final_fire_mask = state.phi < 0
    final_area = float(jnp.sum(final_fire_mask)) * grids.dx * grids.dx
    
    # Compute final smoke impacts
    smoke_impacts = None
    smoke_pm25 = None
    smoke_visibility = None
    if config.enable_smoke and state.smoke is not None:
        smoke_impacts = compute_smoke_impacts(state.smoke, config.smoke_params)
        smoke_pm25 = smoke_impacts.pm25
        smoke_visibility = smoke_impacts.visibility
    
    # Get eruptive warnings
    eruptive_warnings = None
    if config.enable_eruptive and state.eruptive is not None:
        eruptive_warnings = state.eruptive.eruptive_potential
    
    if verbose:
        print("="*60)
        print("SIMULATION COMPLETE")
        print(f"Final burned area: {final_area/10000:.2f} ha")
        if smoke_impacts is not None:
            print(f"Max PM2.5: {float(jnp.max(smoke_pm25)):.1f} μg/m³")
            print(f"Min visibility: {float(jnp.min(smoke_visibility))/1000:.1f} km")
        print("="*60)
    
    return FullSimResult(
        phi_final=state.phi,
        burned_area=final_area,
        phi_history=jnp.stack(phi_history) if phi_history else None,
        area_history=area_history,
        moisture_final=state.moisture,
        wind_final=state.wind,
        eruptive_final=state.eruptive,
        phenology_final=state.phenology,
        smoke_final=state.smoke,
        eruptive_warnings=eruptive_warnings,
        smoke_pm25=smoke_pm25,
        smoke_visibility=smoke_visibility,
        crown_fraction=None,  # Would need to track
    )


# =============================================================================
# Convenience Function
# =============================================================================

def quick_full_simulation(
    dem: jnp.ndarray,
    fuel_type: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    duration_minutes: float = 120.0,
    dx: float = 30.0,
    base_ros: float = 10.0,
    wind_speed: float = 20.0,
    wind_direction: float = 270.0,
    temperature: float = 25.0,
    relative_humidity: float = 30.0,
    enable_all_physics: bool = True,
) -> FullSimResult:
    """
    Quick fire simulation with sensible defaults.
    
    Parameters
    ----------
    dem : array
        Digital elevation model
    fuel_type : array
        FBP fuel type codes
    x_ign, y_ign : float
        Ignition coordinates (grid units)
    duration_minutes : float
        Simulation duration
    dx : float
        Grid spacing (m)
    base_ros : float
        Base rate of spread (m/min)
    wind_speed : float
        Wind speed (km/h)
    wind_direction : float
        Wind direction (degrees)
    temperature : float
        Temperature (°C)
    relative_humidity : float
        Relative humidity (%)
    enable_all_physics : bool
        Enable all physics modules
        
    Returns
    -------
    FullSimResult
        Simulation results
    """
    ny, nx = dem.shape
    
    # Create base grids
    from .levelset import LevelSetGrids
    
    grids = LevelSetGrids(
        ros=base_ros * jnp.ones((ny, nx)),
        ros_dir=wind_direction * jnp.ones((ny, nx)),
        lb_ratio=2.0 * jnp.ones((ny, nx)),
        dx=dx,
    )
    
    # Compute slope and aspect
    dy, dx_arr = jnp.gradient(dem, dx)
    slope = jnp.rad2deg(jnp.arctan(jnp.sqrt(dy**2 + dx_arr**2)))
    aspect = jnp.rad2deg(jnp.arctan2(-dx_arr, -dy)) % 360.0
    
    # Configuration
    config = FullSimConfig(
        enable_solar=enable_all_physics,
        enable_moisture_lag=enable_all_physics,
        enable_crown_fire=enable_all_physics,
        enable_terrain_wind=enable_all_physics,
        enable_fire_atmosphere=False,  # Usually disable, slow
        enable_eruptive=enable_all_physics,
        enable_phenology=enable_all_physics,
        enable_smoke=enable_all_physics,
        enable_spotting=False,  # Usually disable, optional
    )
    
    # Run
    n_steps = int(duration_minutes)
    dt = 1.0
    
    return run_full_simulation(
        grids, dem, slope, aspect, fuel_type,
        x_ign, y_ign, n_steps, dt, config,
        simulation_datetime=datetime.now(),
        latitude=51.0, longitude=-115.0,
        temperature=temperature,
        relative_humidity=relative_humidity,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        verbose=False,
    )
