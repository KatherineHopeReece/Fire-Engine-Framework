"""
Calibration Framework for Ignacio using JAX.

This module provides gradient-based calibration of fire behavior parameters
using observed fire perimeters or areas.

Primary calibration targets:
- Wind speed adjustment factor
- Fuel moisture (FFMC) adjustment
- ROS scaling factor
- Backing fraction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax

from .fbp_jax import (
    FBPCalibrationParams,
    calculate_isi_jax,
    compute_ros_components_jax,
    compute_ros_grid_jax,
    default_params,
)
from .spread_jax import (
    FireParamsJAX,
    compute_fire_area_jax,
    create_fire_params,
    create_initial_perimeter_jax,
    evolve_perimeter_step_jax,
    sample_fire_params_jax,
    simulate_fire_jax,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Calibration Data Structures
# =============================================================================


@dataclass
class CalibrationObservation:
    """
    Single fire observation for calibration.
    
    Attributes
    ----------
    fire_id : str
        Unique identifier for this fire.
    x_ignition : float
        Ignition X coordinate.
    y_ignition : float
        Ignition Y coordinate.
    observed_area : float
        Observed final fire area (m² or ha).
    duration_min : float
        Fire duration in minutes.
    ffmc : float
        Fine Fuel Moisture Code.
    bui : float
        Buildup Index.
    wind_speed : float
        Wind speed (km/h).
    wind_direction : float
        Wind direction (degrees FROM).
    fuel_grid : jnp.ndarray
        Fuel type grid.
    x_coords : jnp.ndarray
        X coordinates of grid.
    y_coords : jnp.ndarray
        Y coordinates of grid.
    observed_perimeter : tuple[jnp.ndarray, jnp.ndarray] | None
        Optional observed perimeter (x, y) for IoU loss.
    """
    fire_id: str
    x_ignition: float
    y_ignition: float
    observed_area: float
    duration_min: float
    ffmc: float
    bui: float
    wind_speed: float
    wind_direction: float
    fuel_grid: jnp.ndarray
    x_coords: jnp.ndarray
    y_coords: jnp.ndarray
    observed_perimeter: tuple[jnp.ndarray, jnp.ndarray] | None = None


@dataclass
class CalibrationResult:
    """
    Result from calibration optimization.
    
    Attributes
    ----------
    params : FBPCalibrationParams
        Optimized calibration parameters.
    loss_history : list[float]
        Loss values during optimization.
    n_iterations : int
        Number of optimization iterations.
    converged : bool
        Whether optimization converged.
    final_loss : float
        Final loss value.
    """
    params: FBPCalibrationParams
    loss_history: list[float] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    final_loss: float = float("inf")


# =============================================================================
# Forward Model
# =============================================================================


def run_fire_forward_model(
    params: FBPCalibrationParams,
    obs: CalibrationObservation,
    dt: float = 1.0,
    n_vertices: int = 200,
    initial_radius: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run fire spread model with given calibration parameters.
    
    Parameters
    ----------
    params : FBPCalibrationParams
        Calibration parameters to evaluate.
    obs : CalibrationObservation
        Observation data (weather, fuel, ignition).
    dt : float
        Time step (minutes).
    n_vertices : int
        Number of perimeter vertices.
    initial_radius : float
        Initial fire radius (m).
        
    Returns
    -------
    x_final : jnp.ndarray
        Final perimeter X coordinates.
    y_final : jnp.ndarray
        Final perimeter Y coordinates.
    area : jnp.ndarray
        Final fire area.
    """
    # Calculate ISI with calibration adjustments
    ffmc = jnp.array(obs.ffmc)
    wind_speed = jnp.array(obs.wind_speed)
    isi = calculate_isi_jax(ffmc, wind_speed, params)
    
    # Compute ROS grid
    bui = jnp.array(obs.bui)
    ros_grid = compute_ros_grid_jax(obs.fuel_grid, isi, bui, params)
    
    # Compute ROS components
    wind_dir = jnp.array(obs.wind_direction)
    ros, bros, fros, raz = compute_ros_components_jax(
        ros_grid, wind_speed, wind_dir, params
    )
    
    # Create fire parameter grid
    # Expand to 3D if needed (single time step)
    if ros.ndim == 2:
        ros = ros[jnp.newaxis, :, :]
        bros = bros[jnp.newaxis, :, :]
        fros = fros[jnp.newaxis, :, :]
        raz = raz[jnp.newaxis, :, :]
    
    fire_params = create_fire_params(
        obs.x_coords, obs.y_coords, ros, bros, fros, raz
    )
    
    # Number of time steps
    n_steps = int(obs.duration_min / dt)
    n_steps = max(1, n_steps)
    
    # Run simulation
    x_final, y_final = simulate_fire_jax(
        fire_params,
        obs.x_ignition,
        obs.y_ignition,
        dt=dt,
        n_vertices=n_vertices,
        initial_radius=initial_radius,
        n_steps=n_steps,
    )
    
    # Compute area
    area = compute_fire_area_jax(x_final, y_final)
    
    return x_final, y_final, area


# =============================================================================
# Loss Functions
# =============================================================================


def calibration_loss_single(
    params_array: jnp.ndarray,
    obs: CalibrationObservation,
    param_names: list[str],
    base_params: FBPCalibrationParams,
    dt: float = 1.0,
    n_vertices: int = 200,
) -> jnp.ndarray:
    """
    Compute calibration loss for a single observation.
    
    Parameters
    ----------
    params_array : jnp.ndarray
        Array of parameter values being optimized.
    obs : CalibrationObservation
        Fire observation data.
    param_names : list[str]
        Names of parameters being optimized.
    base_params : FBPCalibrationParams
        Base parameters (for non-optimized values).
    dt : float
        Time step.
    n_vertices : int
        Number of perimeter vertices.
        
    Returns
    -------
    jnp.ndarray
        Loss value.
    """
    # Reconstruct full parameter set
    params_dict = base_params._asdict()
    for i, name in enumerate(param_names):
        params_dict[name] = params_array[i]
    params = FBPCalibrationParams(**params_dict)
    
    # Run forward model
    x_final, y_final, predicted_area = run_fire_forward_model(
        params, obs, dt=dt, n_vertices=n_vertices
    )
    
    # Compute loss (relative squared error on area)
    observed_area = jnp.array(obs.observed_area)
    relative_error = (predicted_area - observed_area) / (observed_area + 1e-6)
    loss = relative_error ** 2
    
    return loss


def calibration_loss_batch(
    params_array: jnp.ndarray,
    observations: list[CalibrationObservation],
    param_names: list[str],
    base_params: FBPCalibrationParams,
    dt: float = 1.0,
    n_vertices: int = 200,
) -> jnp.ndarray:
    """
    Compute mean calibration loss over batch of observations.
    
    Parameters
    ----------
    params_array : jnp.ndarray
        Array of parameter values being optimized.
    observations : list[CalibrationObservation]
        List of fire observations.
    param_names : list[str]
        Names of parameters being optimized.
    base_params : FBPCalibrationParams
        Base parameters.
    dt : float
        Time step.
    n_vertices : int
        Number of perimeter vertices.
        
    Returns
    -------
    jnp.ndarray
        Mean loss over all observations.
    """
    total_loss = jnp.array(0.0)
    
    for obs in observations:
        loss = calibration_loss_single(
            params_array, obs, param_names, base_params, dt, n_vertices
        )
        total_loss = total_loss + loss
    
    return total_loss / len(observations)


# =============================================================================
# Optimizer
# =============================================================================


@dataclass
class CalibrationConfig:
    """
    Configuration for calibration optimization.
    
    Attributes
    ----------
    param_names : list[str]
        Names of parameters to optimize.
    initial_values : dict[str, float]
        Initial parameter values.
    bounds : dict[str, tuple[float, float]]
        Parameter bounds (min, max).
    learning_rate : float
        Optimizer learning rate.
    n_iterations : int
        Maximum number of iterations.
    convergence_tol : float
        Convergence tolerance on loss change.
    dt : float
        Simulation time step.
    n_vertices : int
        Number of perimeter vertices.
    """
    param_names: list[str] = field(
        default_factory=lambda: ["wind_adj", "ffmc_adj"]
    )
    initial_values: dict[str, float] = field(
        default_factory=lambda: {"wind_adj": 1.0, "ffmc_adj": 0.0}
    )
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "wind_adj": (0.5, 2.0),
            "ffmc_adj": (-10.0, 10.0),
            "ros_scale": (0.5, 2.0),
            "backing_frac": (0.1, 0.4),
        }
    )
    learning_rate: float = 0.01
    n_iterations: int = 100
    convergence_tol: float = 1e-6
    dt: float = 1.0
    n_vertices: int = 200


def clip_to_bounds(
    params_array: jnp.ndarray,
    param_names: list[str],
    bounds: dict[str, tuple[float, float]],
) -> jnp.ndarray:
    """Clip parameters to their bounds."""
    clipped = []
    for i, name in enumerate(param_names):
        lo, hi = bounds.get(name, (-jnp.inf, jnp.inf))
        clipped.append(jnp.clip(params_array[i], lo, hi))
    return jnp.array(clipped)


def calibrate_parameters(
    observations: list[CalibrationObservation],
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """
    Calibrate fire behavior parameters using gradient descent.
    
    Parameters
    ----------
    observations : list[CalibrationObservation]
        Training data (observed fires).
    config : CalibrationConfig, optional
        Calibration configuration.
        
    Returns
    -------
    CalibrationResult
        Optimized parameters and diagnostics.
    """
    if config is None:
        config = CalibrationConfig()
    
    logger.info(f"Starting calibration with {len(observations)} observations")
    logger.info(f"Optimizing parameters: {config.param_names}")
    
    # Initialize parameters
    params_array = jnp.array([
        config.initial_values.get(name, 1.0 if "adj" not in name else 0.0)
        for name in config.param_names
    ])
    
    base_params = default_params()
    
    # Create loss function
    def loss_fn(params_array):
        return calibration_loss_batch(
            params_array,
            observations,
            config.param_names,
            base_params,
            config.dt,
            config.n_vertices,
        )
    
    # JIT compile loss and gradient
    loss_and_grad = jit(value_and_grad(loss_fn))
    
    # Setup optimizer (Adam)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params_array)
    
    # Optimization loop
    loss_history = []
    prev_loss = jnp.inf
    
    for iteration in range(config.n_iterations):
        # Compute loss and gradient
        loss, grads = loss_and_grad(params_array)
        loss_history.append(float(loss))
        
        # Log progress
        if iteration % 10 == 0:
            param_str = ", ".join(
                f"{name}={params_array[i]:.4f}"
                for i, name in enumerate(config.param_names)
            )
            logger.info(f"Iter {iteration}: loss={loss:.6f}, {param_str}")
        
        # Check convergence
        if jnp.abs(prev_loss - loss) < config.convergence_tol:
            logger.info(f"Converged at iteration {iteration}")
            break
        
        prev_loss = loss
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params_array)
        params_array = optax.apply_updates(params_array, updates)
        
        # Clip to bounds
        params_array = clip_to_bounds(params_array, config.param_names, config.bounds)
    
    # Build final parameters
    final_params_dict = base_params._asdict()
    for i, name in enumerate(config.param_names):
        final_params_dict[name] = float(params_array[i])
    final_params = FBPCalibrationParams(**final_params_dict)
    
    logger.info(f"Calibration complete. Final loss: {loss_history[-1]:.6f}")
    logger.info(f"Final parameters: {final_params}")
    
    return CalibrationResult(
        params=final_params,
        loss_history=loss_history,
        n_iterations=len(loss_history),
        converged=(len(loss_history) < config.n_iterations),
        final_loss=loss_history[-1],
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_calibrate_wind_moisture(
    observations: list[CalibrationObservation],
    learning_rate: float = 0.05,
    n_iterations: int = 50,
) -> CalibrationResult:
    """
    Quick calibration of wind and moisture adjustment factors.
    
    This is a convenience function for the most common calibration case.
    
    Parameters
    ----------
    observations : list[CalibrationObservation]
        Training observations.
    learning_rate : float
        Optimizer learning rate.
    n_iterations : int
        Maximum iterations.
        
    Returns
    -------
    CalibrationResult
        Calibration results.
    """
    config = CalibrationConfig(
        param_names=["wind_adj", "ffmc_adj"],
        initial_values={"wind_adj": 1.0, "ffmc_adj": 0.0},
        learning_rate=learning_rate,
        n_iterations=n_iterations,
    )
    
    return calibrate_parameters(observations, config)


def create_synthetic_observation(
    fire_id: str = "synthetic_001",
    x_ignition: float = 500.0,
    y_ignition: float = 500.0,
    observed_area: float = 10000.0,  # m²
    duration_min: float = 60.0,
    ffmc: float = 90.0,
    bui: float = 80.0,
    wind_speed: float = 20.0,
    wind_direction: float = 270.0,
    fuel_type: int = 2,  # C-2
    grid_size: int = 100,
    cell_size: float = 10.0,
) -> CalibrationObservation:
    """
    Create a synthetic observation for testing.
    
    Parameters
    ----------
    fire_id : str
        Fire identifier.
    x_ignition, y_ignition : float
        Ignition coordinates.
    observed_area : float
        Target fire area (m²).
    duration_min : float
        Fire duration (minutes).
    ffmc : float
        Fine Fuel Moisture Code.
    bui : float
        Buildup Index.
    wind_speed : float
        Wind speed (km/h).
    wind_direction : float
        Wind direction (degrees FROM).
    fuel_type : int
        Uniform fuel type ID.
    grid_size : int
        Grid dimensions.
    cell_size : float
        Cell size (m).
        
    Returns
    -------
    CalibrationObservation
        Synthetic observation.
    """
    # Create coordinate arrays
    x_coords = jnp.arange(grid_size) * cell_size
    y_coords = jnp.arange(grid_size) * cell_size
    
    # Create uniform fuel grid
    fuel_grid = jnp.full((grid_size, grid_size), fuel_type, dtype=jnp.int32)
    
    return CalibrationObservation(
        fire_id=fire_id,
        x_ignition=x_ignition,
        y_ignition=y_ignition,
        observed_area=observed_area,
        duration_min=duration_min,
        ffmc=ffmc,
        bui=bui,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        fuel_grid=fuel_grid,
        x_coords=x_coords,
        y_coords=y_coords,
    )


# =============================================================================
# Validation
# =============================================================================


def validate_calibration(
    params: FBPCalibrationParams,
    validation_obs: list[CalibrationObservation],
    dt: float = 1.0,
) -> dict[str, float]:
    """
    Validate calibrated parameters on held-out observations.
    
    Parameters
    ----------
    params : FBPCalibrationParams
        Calibrated parameters.
    validation_obs : list[CalibrationObservation]
        Validation observations.
    dt : float
        Time step.
        
    Returns
    -------
    dict
        Validation metrics.
    """
    errors = []
    relative_errors = []
    
    for obs in validation_obs:
        _, _, predicted_area = run_fire_forward_model(params, obs, dt=dt)
        
        error = float(predicted_area - obs.observed_area)
        rel_error = error / (obs.observed_area + 1e-6)
        
        errors.append(error)
        relative_errors.append(rel_error)
    
    errors = jnp.array(errors)
    relative_errors = jnp.array(relative_errors)
    
    return {
        "mean_error": float(jnp.mean(errors)),
        "rmse": float(jnp.sqrt(jnp.mean(errors**2))),
        "mean_relative_error": float(jnp.mean(relative_errors)),
        "rmse_relative": float(jnp.sqrt(jnp.mean(relative_errors**2))),
        "max_relative_error": float(jnp.max(jnp.abs(relative_errors))),
        "n_observations": len(validation_obs),
    }
