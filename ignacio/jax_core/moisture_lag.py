"""
Dead Fuel Moisture Time-Lag Model.

Implements time-integration of fuel moisture instead of instantaneous
equilibrium, accounting for the fact that fuels take time to respond
to weather changes.

Physics:
- 1-hour fuels: Fine twigs, respond in ~1 hour
- 10-hour fuels: Small branches, respond in ~10 hours
- 100-hour fuels: Large branches, respond in ~100 hours
- 1000-hour fuels: Logs, respond in ~1000 hours

The moisture change follows:
    dM/dt = (M_equilibrium - M) / τ

Where τ is the timelag constant.

References:
- Nelson, R.M. (2000). Prediction of diurnal change in 10-h fuel moisture.
- Rothermel, R.C. (1983). How to Predict the Spread and Intensity of 
  Forest and Range Fires. GTR-INT-143.
- Fosberg, M.A. (1971). Moisture content calculations for the 100-hr 
  timelag fuel in fire danger rating.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


class MoistureState(NamedTuple):
    """
    State of fuel moisture at each cell.
    
    Tracks moisture for different timelag classes.
    """
    m_1hr: jnp.ndarray    # 1-hour fuel moisture (fraction)
    m_10hr: jnp.ndarray   # 10-hour fuel moisture (fraction)
    m_100hr: jnp.ndarray  # 100-hour fuel moisture (fraction)
    m_live: jnp.ndarray   # Live fuel moisture (fraction)


class MoistureLagParams(NamedTuple):
    """Parameters for fuel moisture lag model."""
    
    # Timelag constants (hours)
    tau_1hr: float = 1.0
    tau_10hr: float = 10.0
    tau_100hr: float = 100.0
    tau_live: float = 720.0  # Live fuels change very slowly (30 days)
    
    # Moisture bounds
    min_moisture: float = 0.02   # 2% - bone dry
    max_dead_moisture: float = 0.40  # 40% - saturated
    max_live_moisture: float = 3.00  # 300% for live
    
    # Adsorption/desorption asymmetry
    # Fuels dry faster than they wet (typically 2x faster)
    wetting_factor: float = 0.5  # Wetting happens at half the drying rate


def compute_equilibrium_moisture(
    temperature: float,
    relative_humidity: float,
    is_raining: bool = False,
) -> Tuple[float, float, float]:
    """
    Compute equilibrium moisture content (EMC) for dead fuels.
    
    Based on Van Wagner's EMC equations.
    
    Parameters
    ----------
    temperature : float
        Air temperature (°C)
    relative_humidity : float
        Relative humidity (%, 0-100)
    is_raining : bool
        Whether it's currently raining
        
    Returns
    -------
    emc_1hr, emc_10hr, emc_100hr : float
        Equilibrium moisture content (fraction) for each timelag class
    """
    if is_raining:
        # During rain, EMC approaches saturation
        return 0.35, 0.35, 0.35
    
    # Ensure RH is in valid range
    rh = jnp.clip(relative_humidity, 0, 100)
    temp = temperature
    
    # Van Wagner EMC equations (simplified)
    # For RH < 10%
    emc_low = 0.03229 + 0.281073 * rh - 0.000578 * rh * temp
    
    # For 10% <= RH < 50%
    emc_mid = 2.22749 + 0.160107 * rh - 0.01478 * temp
    
    # For RH >= 50%
    emc_high = 21.0606 + 0.005565 * rh**2 - 0.00035 * rh * temp - 0.483199 * rh
    
    # Select appropriate EMC based on RH
    emc = jnp.where(rh < 10, emc_low,
                    jnp.where(rh < 50, emc_mid, emc_high))
    
    # Convert from % to fraction
    emc = emc / 100.0
    
    # Fine fuels respond to current conditions
    emc_1hr = emc
    
    # Larger fuels have higher base moisture due to depth
    emc_10hr = emc + 0.01
    emc_100hr = emc + 0.02
    
    return float(emc_1hr), float(emc_10hr), float(emc_100hr)


def compute_equilibrium_moisture_grid(
    temperature: jnp.ndarray,
    relative_humidity: jnp.ndarray,
    precipitation: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute equilibrium moisture content for grids.
    
    Parameters
    ----------
    temperature : jnp.ndarray
        Air temperature grid (°C)
    relative_humidity : jnp.ndarray
        Relative humidity grid (%)
    precipitation : jnp.ndarray, optional
        Precipitation grid (mm/hr)
        
    Returns
    -------
    emc_1hr, emc_10hr, emc_100hr : jnp.ndarray
        Equilibrium moisture content grids (fraction)
    """
    rh = jnp.clip(relative_humidity, 0, 100)
    temp = temperature
    
    # Is it raining?
    is_raining = (precipitation is not None) and (precipitation > 0.1)
    
    # Van Wagner EMC equations
    emc_low = 0.03229 + 0.281073 * rh - 0.000578 * rh * temp
    emc_mid = 2.22749 + 0.160107 * rh - 0.01478 * temp
    emc_high = 21.0606 + 0.005565 * rh**2 - 0.00035 * rh * temp - 0.483199 * rh
    
    emc = jnp.where(rh < 10, emc_low,
                    jnp.where(rh < 50, emc_mid, emc_high))
    emc = emc / 100.0
    
    # Override for rain
    if precipitation is not None:
        emc = jnp.where(precipitation > 0.1, 0.35, emc)
    
    emc_1hr = emc
    emc_10hr = emc + 0.01
    emc_100hr = emc + 0.02
    
    return emc_1hr, emc_10hr, emc_100hr


def moisture_tendency(
    moisture: jnp.ndarray,
    equilibrium: jnp.ndarray,
    tau_hours: float,
    params: MoistureLagParams = MoistureLagParams(),
) -> jnp.ndarray:
    """
    Compute rate of moisture change.
    
    dM/dt = (M_eq - M) / τ
    
    With asymmetric wetting/drying rates.
    
    Parameters
    ----------
    moisture : jnp.ndarray
        Current moisture (fraction)
    equilibrium : jnp.ndarray
        Equilibrium moisture (fraction)
    tau_hours : float
        Timelag constant (hours)
    params : MoistureLagParams
        Model parameters
        
    Returns
    -------
    dM_dt : jnp.ndarray
        Rate of moisture change (fraction per hour)
    """
    diff = equilibrium - moisture
    
    # Asymmetric: wetting is slower
    effective_tau = jnp.where(
        diff > 0,  # Wetting
        tau_hours / params.wetting_factor,
        tau_hours
    )
    
    dM_dt = diff / effective_tau
    
    return dM_dt


def update_moisture_euler(
    state: MoistureState,
    temperature: jnp.ndarray,
    relative_humidity: jnp.ndarray,
    dt_hours: float,
    precipitation: Optional[jnp.ndarray] = None,
    params: MoistureLagParams = MoistureLagParams(),
) -> MoistureState:
    """
    Update moisture state using forward Euler integration.
    
    Parameters
    ----------
    state : MoistureState
        Current moisture state
    temperature : jnp.ndarray
        Air temperature (°C)
    relative_humidity : jnp.ndarray
        Relative humidity (%)
    dt_hours : float
        Time step (hours)
    precipitation : jnp.ndarray, optional
        Precipitation (mm/hr)
    params : MoistureLagParams
        Model parameters
        
    Returns
    -------
    new_state : MoistureState
        Updated moisture state
    """
    # Compute equilibrium moisture
    emc_1hr, emc_10hr, emc_100hr = compute_equilibrium_moisture_grid(
        temperature, relative_humidity, precipitation
    )
    
    # Compute tendencies
    dm_1hr = moisture_tendency(state.m_1hr, emc_1hr, params.tau_1hr, params)
    dm_10hr = moisture_tendency(state.m_10hr, emc_10hr, params.tau_10hr, params)
    dm_100hr = moisture_tendency(state.m_100hr, emc_100hr, params.tau_100hr, params)
    
    # Live fuel moisture (very slow change)
    dm_live = moisture_tendency(state.m_live, state.m_live, params.tau_live, params)
    
    # Update moisture
    new_m_1hr = state.m_1hr + dm_1hr * dt_hours
    new_m_10hr = state.m_10hr + dm_10hr * dt_hours
    new_m_100hr = state.m_100hr + dm_100hr * dt_hours
    new_m_live = state.m_live + dm_live * dt_hours
    
    # Clip to physical bounds
    new_m_1hr = jnp.clip(new_m_1hr, params.min_moisture, params.max_dead_moisture)
    new_m_10hr = jnp.clip(new_m_10hr, params.min_moisture, params.max_dead_moisture)
    new_m_100hr = jnp.clip(new_m_100hr, params.min_moisture, params.max_dead_moisture)
    new_m_live = jnp.clip(new_m_live, params.min_moisture, params.max_live_moisture)
    
    return MoistureState(
        m_1hr=new_m_1hr,
        m_10hr=new_m_10hr,
        m_100hr=new_m_100hr,
        m_live=new_m_live,
    )


def update_moisture_rk4(
    state: MoistureState,
    temperature: jnp.ndarray,
    relative_humidity: jnp.ndarray,
    dt_hours: float,
    precipitation: Optional[jnp.ndarray] = None,
    params: MoistureLagParams = MoistureLagParams(),
) -> MoistureState:
    """
    Update moisture state using RK4 integration (more accurate).
    
    Parameters
    ----------
    state : MoistureState
        Current moisture state
    temperature : jnp.ndarray
        Air temperature (°C)
    relative_humidity : jnp.ndarray
        Relative humidity (%)
    dt_hours : float
        Time step (hours)
    precipitation : jnp.ndarray, optional
        Precipitation (mm/hr)
    params : MoistureLagParams
        Model parameters
        
    Returns
    -------
    new_state : MoistureState
        Updated moisture state
    """
    # Compute equilibrium (constant over timestep)
    emc_1hr, emc_10hr, emc_100hr = compute_equilibrium_moisture_grid(
        temperature, relative_humidity, precipitation
    )
    
    def rk4_step(m, emc, tau):
        k1 = moisture_tendency(m, emc, tau, params)
        k2 = moisture_tendency(m + 0.5 * dt_hours * k1, emc, tau, params)
        k3 = moisture_tendency(m + 0.5 * dt_hours * k2, emc, tau, params)
        k4 = moisture_tendency(m + dt_hours * k3, emc, tau, params)
        return m + (dt_hours / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    new_m_1hr = rk4_step(state.m_1hr, emc_1hr, params.tau_1hr)
    new_m_10hr = rk4_step(state.m_10hr, emc_10hr, params.tau_10hr)
    new_m_100hr = rk4_step(state.m_100hr, emc_100hr, params.tau_100hr)
    
    # Clip to bounds
    new_m_1hr = jnp.clip(new_m_1hr, params.min_moisture, params.max_dead_moisture)
    new_m_10hr = jnp.clip(new_m_10hr, params.min_moisture, params.max_dead_moisture)
    new_m_100hr = jnp.clip(new_m_100hr, params.min_moisture, params.max_dead_moisture)
    
    return MoistureState(
        m_1hr=new_m_1hr,
        m_10hr=new_m_10hr,
        m_100hr=new_m_100hr,
        m_live=state.m_live,  # Live changes very slowly
    )


def initialize_moisture_state(
    shape: Tuple[int, int],
    initial_ffmc: float = 88.0,
    initial_dmc: float = 30.0,
    initial_dc: float = 150.0,
    initial_live: float = 1.0,
) -> MoistureState:
    """
    Initialize moisture state from FWI indices.
    
    Parameters
    ----------
    shape : tuple
        Grid shape (ny, nx)
    initial_ffmc : float
        Fine Fuel Moisture Code
    initial_dmc : float
        Duff Moisture Code
    initial_dc : float
        Drought Code
    initial_live : float
        Live fuel moisture (fraction)
        
    Returns
    -------
    state : MoistureState
        Initial moisture state
    """
    # Convert FFMC to moisture
    # FFMC = 59.5 * (250 - m) / (147.2 + m)
    # Solving for m: m = 147.2 * (101 - FFMC) / (59.5 + FFMC)
    m_1hr = 147.2 * (101.0 - initial_ffmc) / (59.5 + initial_ffmc) / 100.0
    
    # Convert DMC to 10-hr moisture (approximate)
    m_10hr = 20.0 * np.exp(-0.02 * initial_dmc) / 100.0 + 0.05
    
    # Convert DC to 100-hr moisture (approximate)
    m_100hr = 25.0 * np.exp(-0.01 * initial_dc) / 100.0 + 0.08
    
    return MoistureState(
        m_1hr=jnp.full(shape, m_1hr, dtype=jnp.float32),
        m_10hr=jnp.full(shape, m_10hr, dtype=jnp.float32),
        m_100hr=jnp.full(shape, m_100hr, dtype=jnp.float32),
        m_live=jnp.full(shape, initial_live, dtype=jnp.float32),
    )


def moisture_state_from_solar(
    base_state: MoistureState,
    solar_moisture_grid: jnp.ndarray,
) -> MoistureState:
    """
    Modify moisture state based on solar conditioning.
    
    Combines time-lagged moisture with spatial solar variations.
    
    Parameters
    ----------
    base_state : MoistureState
        Base moisture state from time integration
    solar_moisture_grid : jnp.ndarray
        Solar-adjusted moisture grid from solar_radiation module
        
    Returns
    -------
    adjusted_state : MoistureState
        Moisture state with solar adjustments
    """
    # Compute the adjustment from solar conditioning
    # (solar_moisture_grid is the final adjusted value, we need the delta)
    base_mean = jnp.mean(base_state.m_1hr)
    solar_adjustment = solar_moisture_grid - base_mean
    
    # Apply to all dead fuel classes (proportionally less for heavier fuels)
    return MoistureState(
        m_1hr=base_state.m_1hr + solar_adjustment,
        m_10hr=base_state.m_10hr + solar_adjustment * 0.7,
        m_100hr=base_state.m_100hr + solar_adjustment * 0.3,
        m_live=base_state.m_live,  # Live not affected by solar
    )


# =============================================================================
# Integration with Level-Set Simulation
# =============================================================================

def evolve_moisture_with_fire(
    moisture_state: MoistureState,
    phi: jnp.ndarray,
    temperature: jnp.ndarray,
    relative_humidity: jnp.ndarray,
    dt_minutes: float,
    params: MoistureLagParams = MoistureLagParams(),
) -> MoistureState:
    """
    Evolve moisture state, accounting for fire effects.
    
    Near the fire (phi close to 0), fuel preheats and dries rapidly.
    
    Parameters
    ----------
    moisture_state : MoistureState
        Current moisture state
    phi : jnp.ndarray
        Level-set field
    temperature : jnp.ndarray
        Ambient temperature
    relative_humidity : jnp.ndarray
        Ambient relative humidity
    dt_minutes : float
        Time step in minutes
    params : MoistureLagParams
        Model parameters
        
    Returns
    -------
    new_state : MoistureState
        Updated moisture state
    """
    dt_hours = dt_minutes / 60.0
    
    # Distance to fire front (proxy from phi gradient)
    # Cells near phi=0 are near the fire
    fire_proximity = jnp.exp(-jnp.abs(phi) / 0.001)  # Sharp falloff
    
    # Fire preheating effect: dramatically lowers local RH and raises temp
    effective_temp = temperature + 50.0 * fire_proximity
    effective_rh = relative_humidity * (1.0 - 0.8 * fire_proximity)
    effective_rh = jnp.maximum(effective_rh, 5.0)  # Min 5% RH near fire
    
    # Update moisture with modified conditions
    new_state = update_moisture_euler(
        moisture_state,
        effective_temp,
        effective_rh,
        dt_hours,
        params=params,
    )
    
    return new_state
