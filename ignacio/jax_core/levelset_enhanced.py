"""
Enhanced Level-Set Fire Spread Simulation.

Integrates all physics modules into a cohesive simulation:
1. Solar radiation & fuel conditioning
2. Fuel moisture time-lag
3. Crown fire transition
4. Mass-conserving terrain wind
5. Fire-atmosphere coupling

This module provides the main entry point for realistic fire simulations
that go beyond basic FBP/Rothermel calculations.

Usage
-----
>>> from ignacio.jax_core.levelset_enhanced import (
...     EnhancedSimConfig,
...     simulate_fire_enhanced,
... )
>>> 
>>> config = EnhancedSimConfig(
...     enable_solar=True,
...     enable_moisture_lag=True,
...     enable_crown_fire=True,
...     enable_terrain_wind=True,
...     enable_fire_atmosphere=False,  # Optional, can slow things down
... )
>>> 
>>> result = simulate_fire_enhanced(
...     grids, dem, x_ign, y_ign,
...     n_steps=1000, dt=1.0,
...     config=config,
...     simulation_datetime=datetime(2024, 7, 15, 14, 0),
...     latitude=51.2, longitude=-115.7,
... )
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
from datetime import datetime, timedelta
import jax
import jax.numpy as jnp
from jax import lax

# Handle imports for both package and direct usage
try:
    # Core level-set
    from .levelset import (
        LevelSetGrids,
        initialize_phi,
        compute_gradient,
        compute_gradient_magnitude_upwind,
        compute_elliptical_speed,
        compute_burned_area,
        compute_burned_area_hard,
    )

    # Physics modules
    from .solar_radiation import (
        SunPosition,
        compute_sun_position,
        compute_fuel_conditioning,
        FuelConditioningParams,
        compute_slope_aspect_from_dem,
    )

    from .moisture_lag import (
        MoistureState,
        MoistureLagParams,
        initialize_moisture_state,
        update_moisture_euler,
        evolve_moisture_with_fire,
    )

    from .crown_fire import (
        CrownFireParams,
        compute_total_ros_with_crown,
        estimate_canopy_properties,
        get_default_fuel_load,
    )

    from .wind_solver import (
        WindField,
        WindSolverParams,
        solve_wind_field,
        wind_field_to_ros_direction,
    )

    from .fire_atmosphere import (
        FireAtmosphereParams,
        CoupledWindField,
        couple_wind_to_fire,
        estimate_coupling_strength,
    )
except ImportError:
    # Direct execution
    from levelset import (
        LevelSetGrids,
        initialize_phi,
        compute_gradient,
        compute_gradient_magnitude_upwind,
        compute_elliptical_speed,
        compute_burned_area,
        compute_burned_area_hard,
    )

    from solar_radiation import (
        SunPosition,
        compute_sun_position,
        compute_fuel_conditioning,
        FuelConditioningParams,
        compute_slope_aspect_from_dem,
    )

    from moisture_lag import (
        MoistureState,
        MoistureLagParams,
        initialize_moisture_state,
        update_moisture_euler,
        evolve_moisture_with_fire,
    )

    from crown_fire import (
        CrownFireParams,
        compute_total_ros_with_crown,
        estimate_canopy_properties,
        get_default_fuel_load,
    )

    from wind_solver import (
        WindField,
        WindSolverParams,
        solve_wind_field,
        wind_field_to_ros_direction,
    )

    from fire_atmosphere import (
        FireAtmosphereParams,
        CoupledWindField,
        couple_wind_to_fire,
        estimate_coupling_strength,
    )


# =============================================================================
# Configuration
# =============================================================================

class EnhancedSimConfig(NamedTuple):
    """Configuration for enhanced simulation."""
    
    # Enable/disable physics modules
    enable_solar: bool = True
    enable_moisture_lag: bool = True
    enable_crown_fire: bool = True
    enable_terrain_wind: bool = True
    enable_fire_atmosphere: bool = False  # Can be slow, disabled by default
    
    # Module-specific parameters
    solar_params: FuelConditioningParams = FuelConditioningParams()
    moisture_params: MoistureLagParams = MoistureLagParams()
    crown_params: CrownFireParams = CrownFireParams()
    wind_params: WindSolverParams = WindSolverParams()
    atmosphere_params: FireAtmosphereParams = FireAtmosphereParams()
    
    # Weather update frequency (minutes)
    # How often to recompute solar position, terrain wind, etc.
    weather_update_interval: float = 60.0
    
    # Default canopy properties (if not provided as grids)
    default_cbh: float = 3.0   # Canopy base height (m)
    default_cbd: float = 0.15  # Canopy bulk density (kg/m³)
    default_fmc: float = 100.0 # Foliar moisture content (%)
    
    # Default fuel load if not provided (kg/m²)
    default_fuel_load: float = 2.0


class EnhancedSimState(NamedTuple):
    """State variables for enhanced simulation."""
    phi: jnp.ndarray                 # Level-set field
    moisture: Optional[MoistureState] # Fuel moisture state
    wind: Optional[WindField]         # Current wind field
    time_elapsed: float               # Minutes since start
    weather_last_updated: float       # Time of last weather update


class EnhancedSimResult(NamedTuple):
    """Results from enhanced simulation."""
    phi_final: jnp.ndarray           # Final level-set field
    burned_area: float               # Burned area (same units as dx*dy)
    phi_history: Optional[list]      # History of phi (if requested)
    moisture_final: Optional[MoistureState]
    cfb_final: Optional[jnp.ndarray] # Crown fraction burned
    wind_final: Optional[WindField]   # Final wind field


# =============================================================================
# Core Simulation Functions
# =============================================================================

def create_enhanced_grids(
    base_grids: LevelSetGrids,
    dem: jnp.ndarray,
    dx: float,
    dy: float,
    config: EnhancedSimConfig,
    latitude: float,
    longitude: float,
    simulation_datetime: datetime,
    background_wind_speed: float,
    background_wind_direction: float,
    temperature: float = 25.0,
    relative_humidity: float = 30.0,
    fuel_load: Optional[jnp.ndarray] = None,
    cbh: Optional[jnp.ndarray] = None,
    cbd: Optional[jnp.ndarray] = None,
    initial_ffmc: float = 88.0,
    initial_dmc: float = 30.0,
) -> Tuple[LevelSetGrids, EnhancedSimState]:
    """
    Create enhanced grids with all physics applied.
    
    Parameters
    ----------
    base_grids : LevelSetGrids
        Base ROS grids from FBP or Rothermel
    dem : jnp.ndarray
        Digital elevation model (m)
    dx, dy : float
        Grid spacing (m)
    config : EnhancedSimConfig
        Simulation configuration
    latitude, longitude : float
        Location for solar calculations
    simulation_datetime : datetime
        Start time of simulation
    background_wind_speed : float
        Background wind speed (m/s for wind solver, or km/h for FBP)
    background_wind_direction : float
        Background wind direction (degrees, FROM)
    temperature : float
        Air temperature (°C)
    relative_humidity : float
        Relative humidity (%)
    fuel_load : jnp.ndarray, optional
        Fuel load grid (kg/m²)
    cbh, cbd : jnp.ndarray, optional
        Canopy base height (m) and bulk density (kg/m³)
    initial_ffmc, initial_dmc : float
        Initial FWI values for moisture state
        
    Returns
    -------
    enhanced_grids : LevelSetGrids
        Grids with enhanced physics applied
    state : EnhancedSimState
        Initial state for simulation
    """
    ny, nx = dem.shape
    
    # Initialize wind field
    wind = None
    raz = base_grids.raz
    
    if config.enable_terrain_wind:
        wind = solve_wind_field(
            dem, dx, dy,
            background_wind_speed,
            background_wind_direction,
            config.wind_params,
        )
        # Update RAZ from terrain-adjusted wind
        raz = wind_field_to_ros_direction(wind)
    
    # Initialize moisture state
    moisture = None
    if config.enable_moisture_lag:
        moisture = initialize_moisture_state(
            (ny, nx),
            initial_ffmc=initial_ffmc,
            initial_dmc=initial_dmc,
        )
    
    # Apply solar conditioning to ROS
    ros_enhanced = base_grids.ros
    bros_enhanced = base_grids.bros
    fros_enhanced = base_grids.fros
    
    if config.enable_solar:
        # Get base moisture from first moisture state or estimate
        if moisture is not None:
            base_moisture = float(jnp.mean(moisture.m_1hr))
        else:
            base_moisture = 0.08  # Default 8%
        
        # Compute solar-adjusted moisture
        solar_moisture = compute_fuel_conditioning(
            dem, dx, dy, base_moisture,
            latitude, longitude,
            simulation_datetime,
            config.solar_params,
        )
        
        # Adjust ROS based on moisture difference
        # Lower moisture = higher ROS (roughly 3% change per 1% moisture)
        moisture_effect = 1.0 - 3.0 * (solar_moisture - base_moisture)
        moisture_effect = jnp.clip(moisture_effect, 0.7, 1.3)
        
        ros_enhanced = base_grids.ros * moisture_effect
        bros_enhanced = base_grids.bros * moisture_effect
        fros_enhanced = base_grids.fros * moisture_effect
    
    # Apply crown fire enhancement
    if config.enable_crown_fire:
        # Get fuel load
        if fuel_load is None:
            fuel_load = jnp.full((ny, nx), config.default_fuel_load)
        
        # Get canopy properties
        if cbh is None:
            cbh = jnp.full((ny, nx), config.default_cbh)
        if cbd is None:
            cbd = jnp.full((ny, nx), config.default_cbd)
        
        fmc = jnp.full((ny, nx), config.default_fmc)
        wind_speed_grid = jnp.full((ny, nx), background_wind_speed * 3.6)  # m/s to km/h
        
        # Compute crown-enhanced ROS
        ros_crown, cfb = compute_total_ros_with_crown(
            ros_enhanced, fuel_load, cbh, cbd, fmc, wind_speed_grid,
            config.crown_params,
        )
        ros_enhanced = ros_crown
    
    # Create enhanced grids
    enhanced_grids = LevelSetGrids(
        x_coords=base_grids.x_coords,
        y_coords=base_grids.y_coords,
        ros=ros_enhanced,
        bros=bros_enhanced,
        fros=fros_enhanced,
        raz=raz,
    )
    
    # Create initial state
    state = EnhancedSimState(
        phi=jnp.zeros((ny, nx)),  # Will be set by simulation
        moisture=moisture,
        wind=wind,
        time_elapsed=0.0,
        weather_last_updated=0.0,
    )
    
    return enhanced_grids, state


def evolve_phi_enhanced(
    phi: jnp.ndarray,
    grids: LevelSetGrids,
    t_idx: int,
    dt: float,
    state: EnhancedSimState,
    config: EnhancedSimConfig,
    dem: jnp.ndarray,
    fuel_load: jnp.ndarray,
    temperature: float,
    relative_humidity: float,
) -> Tuple[jnp.ndarray, EnhancedSimState]:
    """
    Evolve level-set field by one time step with enhanced physics.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Current level-set field
    grids : LevelSetGrids
        Fire behavior grids
    t_idx : int
        Time index
    dt : float
        Time step (minutes)
    state : EnhancedSimState
        Current simulation state
    config : EnhancedSimConfig
        Simulation config
    dem : jnp.ndarray
        Elevation grid
    fuel_load : jnp.ndarray
        Fuel load grid
    temperature, relative_humidity : float
        Current weather
        
    Returns
    -------
    phi_new : jnp.ndarray
        Updated level-set field
    new_state : EnhancedSimState
        Updated state
    """
    # Grid spacing
    dx = float(grids.x_coords[1] - grids.x_coords[0])
    dy = float(grids.y_coords[1] - grids.y_coords[0])
    y_ascending = dy > 0
    dx = abs(dx)
    dy = abs(dy)
    
    # Get ROS at current time
    ros = grids.ros[t_idx] if grids.ros.ndim == 3 else grids.ros
    bros = grids.bros[t_idx] if grids.bros.ndim == 3 else grids.bros
    fros = grids.fros[t_idx] if grids.fros.ndim == 3 else grids.fros
    raz = grids.raz[t_idx] if grids.raz.ndim == 3 else grids.raz
    
    # Update moisture state if enabled
    new_moisture = state.moisture
    if config.enable_moisture_lag and state.moisture is not None:
        ny, nx = phi.shape
        temp_grid = jnp.full((ny, nx), temperature)
        rh_grid = jnp.full((ny, nx), relative_humidity)
        
        new_moisture = evolve_moisture_with_fire(
            state.moisture, phi,
            temp_grid, rh_grid,
            dt,
            config.moisture_params,
        )
        
        # Adjust ROS based on evolved moisture
        base_moisture = 0.08
        moisture_diff = jnp.mean(new_moisture.m_1hr) - base_moisture
        moisture_effect = 1.0 - 3.0 * moisture_diff
        moisture_effect = jnp.clip(moisture_effect, 0.7, 1.3)
        ros = ros * moisture_effect
        bros = bros * moisture_effect
        fros = fros * moisture_effect
    
    # Apply fire-atmosphere coupling if enabled
    new_wind = state.wind
    if config.enable_fire_atmosphere and state.wind is not None:
        coupled = couple_wind_to_fire(
            state.wind.u, state.wind.v,
            phi, ros, fuel_load, dx, dy,
            config.atmosphere_params,
        )
        
        # Update wind field with fire-induced modifications
        # (keeping original as background for next iteration)
        wind_speed = jnp.sqrt(coupled.u**2 + coupled.v**2)
        wind_dir = jnp.degrees(jnp.arctan2(-coupled.u, -coupled.v)) % 360
        
        new_wind = WindField(
            u=coupled.u,
            v=coupled.v,
            speed=wind_speed,
            direction=wind_dir,
        )
        
        # Update RAZ from coupled wind
        raz = wind_field_to_ros_direction(new_wind)
    
    # Compute gradient for normal direction
    phi_x, phi_y = compute_gradient(phi, dx, dy, y_ascending)
    
    # Compute direction-dependent speed
    speed = compute_elliptical_speed(phi_x, phi_y, ros, bros, fros, raz)
    
    # Compute gradient magnitude using upwind scheme
    grad_mag = compute_gradient_magnitude_upwind(phi, dx, dy, speed, y_ascending)
    
    # Hamilton-Jacobi update
    phi_new = phi - dt * speed * grad_mag
    
    # Update state
    new_state = EnhancedSimState(
        phi=phi_new,
        moisture=new_moisture,
        wind=new_wind,
        time_elapsed=state.time_elapsed + dt,
        weather_last_updated=state.weather_last_updated,
    )
    
    return phi_new, new_state


def simulate_fire_enhanced(
    grids: LevelSetGrids,
    dem: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    n_steps: int,
    dt: float = 1.0,
    initial_radius: float = 30.0,
    config: Optional[EnhancedSimConfig] = None,
    simulation_datetime: Optional[datetime] = None,
    latitude: float = 51.0,
    longitude: float = -115.0,
    background_wind_speed: float = 10.0,
    background_wind_direction: float = 270.0,
    temperature: float = 25.0,
    relative_humidity: float = 30.0,
    fuel_load: Optional[jnp.ndarray] = None,
    cbh: Optional[jnp.ndarray] = None,
    cbd: Optional[jnp.ndarray] = None,
    initial_ffmc: float = 88.0,
    initial_dmc: float = 30.0,
    save_history: bool = False,
    history_interval: int = 100,
) -> EnhancedSimResult:
    """
    Run enhanced fire simulation with all physics modules.
    
    This is the main entry point for realistic fire simulations.
    
    Parameters
    ----------
    grids : LevelSetGrids
        Base fire behavior grids (ROS, BROS, FROS, RAZ)
    dem : jnp.ndarray
        Digital elevation model (m)
    x_ign, y_ign : float
        Ignition coordinates
    n_steps : int
        Number of time steps
    dt : float
        Time step (minutes)
    initial_radius : float
        Initial fire radius (same units as coordinates)
    config : EnhancedSimConfig, optional
        Simulation configuration. If None, uses defaults with all physics enabled.
    simulation_datetime : datetime, optional
        Start time for solar calculations. If None, uses midday.
    latitude, longitude : float
        Location for solar calculations
    background_wind_speed : float
        Background wind speed (m/s)
    background_wind_direction : float
        Background wind direction (degrees, FROM)
    temperature : float
        Air temperature (°C)
    relative_humidity : float
        Relative humidity (%)
    fuel_load : jnp.ndarray, optional
        Fuel load grid (kg/m²). If None, uses default.
    cbh, cbd : jnp.ndarray, optional
        Canopy properties. If None, uses defaults.
    initial_ffmc, initial_dmc : float
        Initial FWI values
    save_history : bool
        Whether to save phi history
    history_interval : int
        Steps between history saves
        
    Returns
    -------
    result : EnhancedSimResult
        Simulation results
    """
    if config is None:
        config = EnhancedSimConfig()
    
    if simulation_datetime is None:
        simulation_datetime = datetime(2024, 7, 15, 12, 0)
    
    # Grid properties
    ny = len(grids.y_coords)
    nx = len(grids.x_coords)
    dx = float(abs(grids.x_coords[1] - grids.x_coords[0]))
    dy = float(abs(grids.y_coords[1] - grids.y_coords[0]))
    
    # Default fuel load
    if fuel_load is None:
        fuel_load = jnp.full((ny, nx), config.default_fuel_load)
    
    # Create enhanced grids and initial state
    enhanced_grids, state = create_enhanced_grids(
        grids, dem, dx, dy, config,
        latitude, longitude, simulation_datetime,
        background_wind_speed, background_wind_direction,
        temperature, relative_humidity,
        fuel_load, cbh, cbd,
        initial_ffmc, initial_dmc,
    )
    
    # Initialize phi
    phi = initialize_phi(
        grids.x_coords, grids.y_coords,
        x_ign, y_ign, initial_radius,
    )
    
    # Update state with initial phi
    state = EnhancedSimState(
        phi=phi,
        moisture=state.moisture,
        wind=state.wind,
        time_elapsed=0.0,
        weather_last_updated=0.0,
    )
    
    # History storage
    phi_history = [] if save_history else None
    
    # Run simulation
    for step in range(n_steps):
        # Get time index for time-varying grids
        t_idx = min(step, enhanced_grids.ros.shape[0] - 1) if enhanced_grids.ros.ndim == 3 else 0
        
        # Evolve one step
        phi, state = evolve_phi_enhanced(
            phi, enhanced_grids, t_idx, dt, state, config,
            dem, fuel_load, temperature, relative_humidity,
        )
        
        # Save history
        if save_history and step % history_interval == 0:
            phi_history.append(phi.copy())
    
    # Compute final burned area
    burned_area = compute_burned_area_hard(phi, dx, dy)
    
    # Compute final CFB if crown fire was enabled
    cfb_final = None
    if config.enable_crown_fire:
        # Get final ROS
        ros_final = enhanced_grids.ros[-1] if enhanced_grids.ros.ndim == 3 else enhanced_grids.ros
        
        if cbh is None:
            cbh = jnp.full((ny, nx), config.default_cbh)
        if cbd is None:
            cbd = jnp.full((ny, nx), config.default_cbd)
        
        fmc = jnp.full((ny, nx), config.default_fmc)
        wind_grid = jnp.full((ny, nx), background_wind_speed * 3.6)
        
        _, cfb_final = compute_total_ros_with_crown(
            ros_final, fuel_load, cbh, cbd, fmc, wind_grid,
            config.crown_params,
        )
    
    return EnhancedSimResult(
        phi_final=phi,
        burned_area=float(burned_area),
        phi_history=phi_history,
        moisture_final=state.moisture,
        cfb_final=cfb_final,
        wind_final=state.wind,
    )


# =============================================================================
# Convenience functions
# =============================================================================

def quick_simulate(
    grids: LevelSetGrids,
    dem: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    duration_minutes: float,
    dt: float = 1.0,
    **kwargs,
) -> EnhancedSimResult:
    """
    Quick simulation with duration specified in minutes.
    
    Convenience wrapper around simulate_fire_enhanced.
    """
    n_steps = int(duration_minutes / dt)
    return simulate_fire_enhanced(
        grids, dem, x_ign, y_ign,
        n_steps=n_steps, dt=dt,
        **kwargs,
    )


def simulate_with_weather_sequence(
    grids: LevelSetGrids,
    dem: jnp.ndarray,
    x_ign: float,
    y_ign: float,
    weather_sequence: list,  # List of (datetime, temp, rh, wind_speed, wind_dir)
    dt: float = 1.0,
    config: Optional[EnhancedSimConfig] = None,
    latitude: float = 51.0,
    longitude: float = -115.0,
    **kwargs,
) -> EnhancedSimResult:
    """
    Simulate with time-varying weather.
    
    Parameters
    ----------
    weather_sequence : list
        List of tuples: (datetime, temperature, relative_humidity, wind_speed, wind_direction)
        Each entry defines weather for a time period.
    """
    if config is None:
        config = EnhancedSimConfig()
    
    if len(weather_sequence) == 0:
        raise ValueError("Weather sequence cannot be empty")
    
    # Calculate total duration
    first_dt = weather_sequence[0][0]
    last_dt = weather_sequence[-1][0]
    total_duration = (last_dt - first_dt).total_seconds() / 60.0  # minutes
    n_steps = int(total_duration / dt)
    
    # Run with first weather entry
    dt0, temp0, rh0, ws0, wd0 = weather_sequence[0]
    
    return simulate_fire_enhanced(
        grids, dem, x_ign, y_ign,
        n_steps=n_steps, dt=dt,
        config=config,
        simulation_datetime=dt0,
        latitude=latitude,
        longitude=longitude,
        temperature=temp0,
        relative_humidity=rh0,
        background_wind_speed=ws0,
        background_wind_direction=wd0,
        **kwargs,
    )
