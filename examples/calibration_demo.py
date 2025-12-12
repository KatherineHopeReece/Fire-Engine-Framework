#!/usr/bin/env python3
"""
Example: JAX-based Fire Spread Calibration for Ignacio.

This script demonstrates how to use the differentiable fire spread model
to calibrate wind speed and fuel moisture adjustment factors.

The workflow:
1. Create synthetic "observed" fires using known parameters
2. Attempt to recover those parameters through gradient-based optimization
3. Validate the calibrated model

Usage:
    python examples/calibration_demo.py
"""

import logging
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Configure JAX
jax.config.update("jax_enable_x64", True)

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from ignacio.jax_core import (
    FBPCalibrationParams,
    CalibrationConfig,
    CalibrationObservation,
    calibrate_parameters,
    create_synthetic_observation,
    default_params,
    quick_calibrate_wind_moisture,
    run_fire_forward_model,
    validate_calibration,
    compute_fire_area_jax,
    create_initial_perimeter_jax,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def demo_basic_simulation():
    """Demonstrate basic fire simulation with JAX."""
    logger.info("=" * 60)
    logger.info("Demo 1: Basic Fire Simulation")
    logger.info("=" * 60)
    
    # Default parameters
    params = default_params()
    logger.info(f"Default parameters: {params}")
    
    # Create a synthetic observation
    obs = create_synthetic_observation(
        fire_id="demo_fire",
        observed_area=50000.0,  # 5 hectares
        duration_min=120.0,
        ffmc=92.0,
        bui=85.0,
        wind_speed=25.0,
        wind_direction=270.0,  # West wind
        fuel_type=2,  # C-2 Boreal Spruce
    )
    
    # Run forward model
    x_final, y_final, area = run_fire_forward_model(params, obs)
    
    logger.info(f"Simulated fire area: {float(area):.1f} m² ({float(area)/10000:.2f} ha)")
    logger.info(f"Perimeter vertices: {len(x_final)}")
    
    return x_final, y_final, area


def demo_parameter_sensitivity():
    """Demonstrate sensitivity to calibration parameters."""
    logger.info("=" * 60)
    logger.info("Demo 2: Parameter Sensitivity Analysis")
    logger.info("=" * 60)
    
    # Base observation
    obs = create_synthetic_observation(
        observed_area=30000.0,
        duration_min=90.0,
        ffmc=90.0,
        bui=80.0,
        wind_speed=20.0,
        wind_direction=270.0,
    )
    
    # Test wind adjustment sensitivity
    wind_adjustments = [0.5, 0.75, 1.0, 1.25, 1.5]
    logger.info("\nWind Adjustment Sensitivity:")
    logger.info("-" * 40)
    
    wind_areas = []
    for wind_adj in wind_adjustments:
        params = FBPCalibrationParams(wind_adj=wind_adj)
        _, _, area = run_fire_forward_model(params, obs)
        wind_areas.append(float(area))
        logger.info(f"  wind_adj={wind_adj:.2f}: area={float(area):.0f} m²")
    
    # Test FFMC adjustment sensitivity
    ffmc_adjustments = [-10.0, -5.0, 0.0, 5.0, 10.0]
    logger.info("\nFFMC Adjustment Sensitivity:")
    logger.info("-" * 40)
    
    ffmc_areas = []
    for ffmc_adj in ffmc_adjustments:
        params = FBPCalibrationParams(ffmc_adj=ffmc_adj)
        _, _, area = run_fire_forward_model(params, obs)
        ffmc_areas.append(float(area))
        logger.info(f"  ffmc_adj={ffmc_adj:+.1f}: area={float(area):.0f} m²")
    
    return wind_adjustments, wind_areas, ffmc_adjustments, ffmc_areas


def demo_gradient_computation():
    """Demonstrate gradient computation through the model."""
    logger.info("=" * 60)
    logger.info("Demo 3: Gradient Computation")
    logger.info("=" * 60)
    
    obs = create_synthetic_observation(
        observed_area=25000.0,
        duration_min=60.0,
        ffmc=91.0,
        bui=82.0,
        wind_speed=22.0,
    )
    
    # Define loss function
    def loss_fn(params_array):
        wind_adj, ffmc_adj = params_array
        params = FBPCalibrationParams(wind_adj=wind_adj, ffmc_adj=ffmc_adj)
        _, _, area = run_fire_forward_model(params, obs)
        target = jnp.array(obs.observed_area)
        return ((area - target) / target) ** 2
    
    # Compute loss and gradient
    from jax import value_and_grad
    loss_and_grad = jax.jit(value_and_grad(loss_fn))
    
    # Test at default parameters
    params_array = jnp.array([1.0, 0.0])
    loss, grads = loss_and_grad(params_array)
    
    logger.info(f"At default params [wind_adj=1.0, ffmc_adj=0.0]:")
    logger.info(f"  Loss: {float(loss):.6f}")
    logger.info(f"  Gradient w.r.t. wind_adj: {float(grads[0]):.6f}")
    logger.info(f"  Gradient w.r.t. ffmc_adj: {float(grads[1]):.6f}")
    
    # Test at perturbed parameters
    params_array = jnp.array([1.2, 3.0])
    loss, grads = loss_and_grad(params_array)
    
    logger.info(f"\nAt perturbed params [wind_adj=1.2, ffmc_adj=3.0]:")
    logger.info(f"  Loss: {float(loss):.6f}")
    logger.info(f"  Gradient w.r.t. wind_adj: {float(grads[0]):.6f}")
    logger.info(f"  Gradient w.r.t. ffmc_adj: {float(grads[1]):.6f}")
    
    return loss, grads


def demo_calibration():
    """Demonstrate full calibration workflow."""
    logger.info("=" * 60)
    logger.info("Demo 4: Full Calibration Workflow")
    logger.info("=" * 60)
    
    # Generate "observed" fires with known true parameters
    true_params = FBPCalibrationParams(wind_adj=1.3, ffmc_adj=2.5)
    logger.info(f"True (hidden) parameters: wind_adj={true_params.wind_adj}, ffmc_adj={true_params.ffmc_adj}")
    
    # Create synthetic observations using true parameters
    base_observations = [
        create_synthetic_observation(
            fire_id=f"train_{i}",
            observed_area=0.0,  # Will be filled
            duration_min=60.0 + i * 15,
            ffmc=88.0 + i * 2,
            bui=75.0 + i * 5,
            wind_speed=18.0 + i * 3,
            wind_direction=250.0 + i * 20,
            fuel_type=2,
        )
        for i in range(5)
    ]
    
    # Generate "observed" areas using true parameters
    observations = []
    for obs in base_observations:
        _, _, true_area = run_fire_forward_model(true_params, obs)
        # Add some noise (5% relative)
        noisy_area = float(true_area) * (1.0 + 0.05 * np.random.randn())
        obs_with_area = CalibrationObservation(
            fire_id=obs.fire_id,
            x_ignition=obs.x_ignition,
            y_ignition=obs.y_ignition,
            observed_area=noisy_area,
            duration_min=obs.duration_min,
            ffmc=obs.ffmc,
            bui=obs.bui,
            wind_speed=obs.wind_speed,
            wind_direction=obs.wind_direction,
            fuel_grid=obs.fuel_grid,
            x_coords=obs.x_coords,
            y_coords=obs.y_coords,
        )
        observations.append(obs_with_area)
        logger.info(f"  {obs.fire_id}: true_area={float(true_area):.0f}, observed={noisy_area:.0f}")
    
    # Run calibration
    logger.info("\nRunning calibration...")
    config = CalibrationConfig(
        param_names=["wind_adj", "ffmc_adj"],
        initial_values={"wind_adj": 1.0, "ffmc_adj": 0.0},
        learning_rate=0.1,
        n_iterations=100,
        convergence_tol=1e-8,
    )
    
    result = calibrate_parameters(observations, config)
    
    logger.info(f"\nCalibration Results:")
    logger.info(f"  Final loss: {result.final_loss:.6f}")
    logger.info(f"  Iterations: {result.n_iterations}")
    logger.info(f"  Converged: {result.converged}")
    logger.info(f"  Recovered wind_adj: {result.params.wind_adj:.4f} (true: {true_params.wind_adj})")
    logger.info(f"  Recovered ffmc_adj: {result.params.ffmc_adj:.4f} (true: {true_params.ffmc_adj})")
    
    # Validation
    validation_metrics = validate_calibration(result.params, observations)
    logger.info(f"\nValidation Metrics:")
    for key, value in validation_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return result, observations


def demo_quick_calibration():
    """Demonstrate quick calibration helper."""
    logger.info("=" * 60)
    logger.info("Demo 5: Quick Wind/Moisture Calibration")
    logger.info("=" * 60)
    
    # Create some observations
    observations = [
        create_synthetic_observation(
            fire_id=f"quick_{i}",
            observed_area=20000.0 + i * 5000,
            duration_min=60.0,
            ffmc=90.0,
            bui=80.0,
            wind_speed=20.0,
            wind_direction=270.0,
        )
        for i in range(3)
    ]
    
    result = quick_calibrate_wind_moisture(
        observations,
        learning_rate=0.1,
        n_iterations=30,
    )
    
    logger.info(f"Quick calibration result:")
    logger.info(f"  wind_adj: {result.params.wind_adj:.4f}")
    logger.info(f"  ffmc_adj: {result.params.ffmc_adj:.4f}")
    logger.info(f"  Final loss: {result.final_loss:.6f}")
    
    return result


def plot_calibration_results(result, observations):
    """Create visualization of calibration results."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Loss history
        ax1 = axes[0]
        ax1.semilogy(result.loss_history)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss (log scale)")
        ax1.set_title("Calibration Convergence")
        ax1.grid(True, alpha=0.3)
        
        # Predicted vs Observed areas
        ax2 = axes[1]
        pred_areas = []
        obs_areas = []
        for obs in observations:
            _, _, pred_area = run_fire_forward_model(result.params, obs)
            pred_areas.append(float(pred_area))
            obs_areas.append(obs.observed_area)
        
        ax2.scatter(obs_areas, pred_areas, s=60, alpha=0.7)
        max_area = max(max(obs_areas), max(pred_areas))
        ax2.plot([0, max_area], [0, max_area], "k--", label="1:1 line")
        ax2.set_xlabel("Observed Area (m²)")
        ax2.set_ylabel("Predicted Area (m²)")
        ax2.set_title("Predicted vs Observed")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Parameter space (if we saved iterations)
        ax3 = axes[2]
        ax3.text(0.5, 0.5, 
                 f"Final Parameters:\n\n"
                 f"wind_adj = {result.params.wind_adj:.4f}\n"
                 f"ffmc_adj = {result.params.ffmc_adj:.4f}\n\n"
                 f"Final Loss = {result.final_loss:.6f}",
                 ha='center', va='center', fontsize=12,
                 transform=ax3.transAxes)
        ax3.set_title("Calibrated Parameters")
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig("/home/claude/ignacio/calibration_results.png", dpi=150)
        logger.info("Saved calibration plot to /home/claude/ignacio/calibration_results.png")
        
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")


def main():
    """Run all demos."""
    logger.info("=" * 60)
    logger.info("Ignacio JAX Differentiable Fire Spread - Demo")
    logger.info("=" * 60)
    
    # Run demos
    demo_basic_simulation()
    print()
    
    demo_parameter_sensitivity()
    print()
    
    demo_gradient_computation()
    print()
    
    result, observations = demo_calibration()
    print()
    
    demo_quick_calibration()
    print()
    
    # Create plot
    plot_calibration_results(result, observations)
    
    logger.info("=" * 60)
    logger.info("All demos complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
