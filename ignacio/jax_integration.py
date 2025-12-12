"""
JAX Integration Module for Ignacio.

This module provides integration between the JAX-based differentiable fire
spread model and the main Ignacio framework. It enables:

1. Using calibrated parameters from JAX optimization in standard simulations
2. Running complete simulations in JAX mode for speed/GPU support
3. Converting between config-based and JAX-based parameter representations

Usage:
    from ignacio.jax_integration import (
        run_calibration,
        apply_calibration_to_config,
        run_jax_simulation,
    )
    
    # Run calibration
    result = run_calibration(config, observations)
    
    # Apply to config
    config = apply_calibration_to_config(config, result)
    
    # Or run full JAX simulation
    perimeters = run_jax_simulation(config, ignition_point)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ignacio.config import IgnacioConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Availability Check
# =============================================================================


def check_jax_available() -> bool:
    """Check if JAX is available for import."""
    try:
        import jax
        import jax.numpy as jnp
        import optax
        return True
    except ImportError:
        return False


JAX_AVAILABLE = check_jax_available()

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    # Configure JAX for better numerical precision
    jax.config.update("jax_enable_x64", True)
    
    from ignacio.jax_core.core import (
        FireParams,
        Observation,
        CalibrationResult,
        calibrate,
        calibrate_wind_and_moisture,
        create_observation,
        forward_model,
        validate,
        simulate_fire,
        create_perimeter,
        compute_area,
        FireGrids,
        fbp_pipeline,
    )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CalibrationObservation:
    """
    Fire observation for calibration.
    
    Attributes
    ----------
    fire_id : str
        Unique identifier for this fire.
    x_ignition : float
        X coordinate of ignition point.
    y_ignition : float
        Y coordinate of ignition point.
    observed_area : float
        Observed final fire area (m²).
    duration_min : float
        Fire duration in minutes.
    ffmc : float
        Fine Fuel Moisture Code (or use config default).
    bui : float
        Buildup Index (or use config default).
    wind_speed : float
        Wind speed in km/h.
    wind_direction : float
        Wind direction (degrees FROM).
    """
    fire_id: str
    x_ignition: float
    y_ignition: float
    observed_area: float
    duration_min: float
    ffmc: float = 90.0
    bui: float = 80.0
    wind_speed: float = 20.0
    wind_direction: float = 270.0


@dataclass
class JAXCalibrationResult:
    """
    Result from JAX calibration.
    
    Attributes
    ----------
    wind_adj : float
        Calibrated wind adjustment factor.
    ffmc_adj : float
        Calibrated FFMC adjustment.
    ros_scale : float
        Calibrated ROS scale factor.
    backing_frac : float
        Calibrated backing fraction.
    final_loss : float
        Final loss value.
    n_iterations : int
        Number of iterations run.
    converged : bool
        Whether optimization converged.
    loss_history : list[float]
        Loss at each iteration.
    """
    wind_adj: float = 1.0
    ffmc_adj: float = 0.0
    ros_scale: float = 1.0
    backing_frac: float = 0.2
    final_loss: float = float("inf")
    n_iterations: int = 0
    converged: bool = False
    loss_history: list = None
    
    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []


# =============================================================================
# Configuration Conversion
# =============================================================================


def config_to_fire_params(config: "IgnacioConfig") -> "FireParams":
    """
    Convert Ignacio config to JAX FireParams.
    
    Parameters
    ----------
    config : IgnacioConfig
        Ignacio configuration.
        
    Returns
    -------
    FireParams
        JAX fire parameters.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib optax")
    
    cal = config.calibration
    return FireParams(
        wind_adj=cal.wind_adj,
        ffmc_adj=cal.ffmc_adj,
        ros_scale=cal.ros_scale,
        backing_frac=cal.backing_frac,
    )


def apply_calibration_to_config(
    config: "IgnacioConfig",
    result: JAXCalibrationResult,
) -> "IgnacioConfig":
    """
    Apply calibration results to configuration.
    
    Parameters
    ----------
    config : IgnacioConfig
        Original configuration.
    result : JAXCalibrationResult
        Calibration results.
        
    Returns
    -------
    IgnacioConfig
        Updated configuration with calibrated parameters.
    """
    # Update calibration section
    config.calibration.wind_adj = result.wind_adj
    config.calibration.ffmc_adj = result.ffmc_adj
    config.calibration.ros_scale = result.ros_scale
    config.calibration.backing_frac = result.backing_frac
    
    # Also update FBP defaults to reflect calibration
    # The backing fraction is directly used
    config.fbp.backing_fraction = result.backing_frac
    
    logger.info(f"Applied calibration: wind_adj={result.wind_adj:.4f}, "
                f"ffmc_adj={result.ffmc_adj:.4f}, ros_scale={result.ros_scale:.4f}")
    
    return config


# =============================================================================
# Calibration
# =============================================================================


def run_calibration(
    config: "IgnacioConfig",
    observations: list[CalibrationObservation],
    verbose: bool = True,
) -> JAXCalibrationResult:
    """
    Run JAX-based calibration using observed fire data.
    
    Parameters
    ----------
    config : IgnacioConfig
        Ignacio configuration (provides defaults and bounds).
    observations : list[CalibrationObservation]
        Observed fires for calibration.
    verbose : bool
        Print progress.
        
    Returns
    -------
    JAXCalibrationResult
        Calibration results.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib optax")
    
    if not observations:
        raise ValueError("No observations provided for calibration")
    
    cal_config = config.calibration
    
    # Determine which parameters to calibrate
    param_names = []
    initial_values = {}
    bounds = {}
    
    if cal_config.calibrate_wind:
        param_names.append("wind_adj")
        initial_values["wind_adj"] = cal_config.wind_adj_init
        bounds["wind_adj"] = cal_config.wind_adj_bounds
    
    if cal_config.calibrate_ffmc:
        param_names.append("ffmc_adj")
        initial_values["ffmc_adj"] = cal_config.ffmc_adj_init
        bounds["ffmc_adj"] = cal_config.ffmc_adj_bounds
    
    if cal_config.calibrate_ros_scale:
        param_names.append("ros_scale")
        initial_values["ros_scale"] = cal_config.ros_scale_init
        bounds["ros_scale"] = cal_config.ros_scale_bounds
    
    if cal_config.calibrate_backing:
        param_names.append("backing_frac")
        initial_values["backing_frac"] = cal_config.backing_frac_init
        bounds["backing_frac"] = cal_config.backing_frac_bounds
    
    if not param_names:
        logger.warning("No parameters selected for calibration")
        return JAXCalibrationResult()
    
    logger.info(f"Running JAX calibration for parameters: {param_names}")
    
    # Convert observations to JAX format
    jax_observations = []
    for obs in observations:
        jax_obs = create_observation(
            fire_id=obs.fire_id,
            x_ign=obs.x_ignition,
            y_ign=obs.y_ignition,
            area=obs.observed_area,
            duration=obs.duration_min,
            ffmc=obs.ffmc,
            bui=obs.bui,
            wind=obs.wind_speed,
            wind_dir=obs.wind_direction,
        )
        jax_observations.append(jax_obs)
    
    # Run calibration
    result = calibrate(
        observations=jax_observations,
        param_names=param_names,
        initial_values=initial_values,
        bounds=bounds,
        learning_rate=cal_config.learning_rate,
        n_iterations=cal_config.n_iterations,
        convergence_tol=cal_config.convergence_tol,
        reg_strength=cal_config.regularization,
        verbose=verbose,
    )
    
    # Convert result
    return JAXCalibrationResult(
        wind_adj=result.params.wind_adj,
        ffmc_adj=result.params.ffmc_adj,
        ros_scale=result.params.ros_scale,
        backing_frac=result.params.backing_frac,
        final_loss=result.final_loss,
        n_iterations=result.n_iter,
        converged=result.converged,
        loss_history=list(result.loss_history),
    )


def quick_calibrate(
    config: "IgnacioConfig",
    observations: list[CalibrationObservation],
) -> JAXCalibrationResult:
    """
    Quick calibration of wind and moisture parameters.
    
    Convenience function for the most common calibration case.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration (for defaults).
    observations : list[CalibrationObservation]
        Observed fires.
        
    Returns
    -------
    JAXCalibrationResult
        Calibration results.
    """
    # Override config to only calibrate wind and moisture
    config.calibration.calibrate_wind = True
    config.calibration.calibrate_ffmc = True
    config.calibration.calibrate_ros_scale = False
    config.calibration.calibrate_backing = False
    
    return run_calibration(config, observations)


# =============================================================================
# JAX Simulation
# =============================================================================


def run_jax_simulation(
    config: "IgnacioConfig",
    x_ignition: float,
    y_ignition: float,
    ffmc: float = None,
    bui: float = None,
    wind_speed: float = None,
    wind_direction: float = None,
    duration_min: float = None,
    fuel_grid: np.ndarray = None,
    x_coords: np.ndarray = None,
    y_coords: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run fire simulation using JAX backend.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration (provides defaults and calibration params).
    x_ignition, y_ignition : float
        Ignition coordinates.
    ffmc : float, optional
        Fine Fuel Moisture Code (default from config).
    bui : float, optional
        Buildup Index (default from config).
    wind_speed : float, optional
        Wind speed km/h (default from config).
    wind_direction : float, optional
        Wind direction degrees (default 270).
    duration_min : float, optional
        Simulation duration (default from config).
    fuel_grid : np.ndarray, optional
        Fuel type grid (default: uniform C-2).
    x_coords, y_coords : np.ndarray, optional
        Coordinate arrays.
        
    Returns
    -------
    x_final : np.ndarray
        Final perimeter X coordinates.
    y_final : np.ndarray
        Final perimeter Y coordinates.
    area : float
        Final fire area (m²).
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available. Install with: pip install jax jaxlib optax")
    
    # Get defaults from config
    if ffmc is None:
        ffmc = config.fbp.defaults.ffmc
    if bui is None:
        bui = config.fbp.defaults.bui
    if wind_speed is None:
        wind_speed = 20.0  # Default moderate wind
    if wind_direction is None:
        wind_direction = 270.0
    if duration_min is None:
        duration_min = config.simulation.max_duration
    
    # Get calibration parameters
    params = config_to_fire_params(config)
    
    # Create observation (for the forward model interface)
    obs = create_observation(
        fire_id="jax_sim",
        x_ign=x_ignition,
        y_ign=y_ignition,
        area=0.0,  # Not used for simulation
        duration=duration_min,
        ffmc=ffmc,
        bui=bui,
        wind=wind_speed,
        wind_dir=wind_direction,
    )
    
    # Override fuel grid if provided
    if fuel_grid is not None and x_coords is not None and y_coords is not None:
        obs = Observation(
            fire_id=obs.fire_id,
            x_ign=obs.x_ign,
            y_ign=obs.y_ign,
            observed_area=obs.observed_area,
            duration=obs.duration,
            ffmc=obs.ffmc,
            bui=obs.bui,
            wind_speed=obs.wind_speed,
            wind_dir=obs.wind_dir,
            fuel_grid=jnp.array(fuel_grid),
            x_coords=jnp.array(x_coords),
            y_coords=jnp.array(y_coords),
        )
    
    # Run forward model
    area = forward_model(
        params, obs,
        dt=config.simulation.dt,
        n_vertices=config.simulation.n_vertices,
        initial_radius=config.simulation.initial_radius,
    )
    
    # For full perimeter, we need to run simulation directly
    # Compute ROS components
    ros, bros, fros, raz = fbp_pipeline(
        jnp.array(ffmc),
        jnp.array(bui),
        jnp.array(wind_speed),
        jnp.array(wind_direction),
        params,
    )
    
    # Create grids
    ny, nx = obs.fuel_grid.shape
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
    
    # Run simulation
    n_steps = max(1, int(duration_min / config.simulation.dt))
    x_final, y_final = simulate_fire(
        grids, x_ignition, y_ignition,
        n_steps=n_steps,
        dt=config.simulation.dt,
        n_vertices=config.simulation.n_vertices,
        initial_radius=config.simulation.initial_radius,
    )
    
    return np.array(x_final), np.array(y_final), float(area)


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_methods(
    config: "IgnacioConfig",
    x_ignition: float,
    y_ignition: float,
    **kwargs,
) -> dict:
    """
    Compare NumPy and JAX simulation results.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration.
    x_ignition, y_ignition : float
        Ignition coordinates.
    **kwargs
        Additional arguments passed to simulations.
        
    Returns
    -------
    dict
        Comparison metrics.
    """
    results = {"jax_available": JAX_AVAILABLE}
    
    if JAX_AVAILABLE:
        import time
        
        # JAX simulation
        start = time.time()
        x_jax, y_jax, area_jax = run_jax_simulation(
            config, x_ignition, y_ignition, **kwargs
        )
        jax_time = time.time() - start
        
        results["jax"] = {
            "area_m2": area_jax,
            "area_ha": area_jax / 10000,
            "n_vertices": len(x_jax),
            "time_sec": jax_time,
        }
        
        logger.info(f"JAX simulation: {area_jax/10000:.2f} ha in {jax_time:.2f}s")
    
    return results


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "JAX_AVAILABLE",
    "check_jax_available",
    "CalibrationObservation",
    "JAXCalibrationResult",
    "config_to_fire_params",
    "apply_calibration_to_config",
    "run_calibration",
    "quick_calibrate",
    "run_jax_simulation",
    "compare_methods",
]
