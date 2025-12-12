#!/usr/bin/env python3
"""
Simple Wind & Fuel Moisture Calibration Demo for Ignacio.

This script demonstrates the core calibration workflow using the improved
JAX numerical core. It shows how to:

1. Create fire observations from field data
2. Run gradient-based calibration to find optimal parameters
3. Validate the calibrated model
4. Visualize results

Usage:
    python examples/simple_calibration.py

Author: Ignacio Team
"""

import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

# Configure JAX for double precision (better gradients)
jax.config.update("jax_enable_x64", True)

# Import directly from jax_core to avoid dependency issues
from ignacio.jax_core.core import (
    FireParams,
    Observation,
    calibrate_wind_and_moisture,
    calibrate,
    create_observation,
    forward_model,
    validate,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def demo_basic_forward_model():
    """
    Demo 1: Basic forward model - simulate a fire with given parameters.
    """
    logger.info("=" * 60)
    logger.info("DEMO 1: Basic Forward Model")
    logger.info("=" * 60)
    
    # Default parameters (no calibration adjustments)
    params = FireParams()
    logger.info(f"Parameters: wind_adj={params.wind_adj}, ffmc_adj={params.ffmc_adj}")
    
    # Create a synthetic fire observation
    obs = create_observation(
        fire_id="demo_fire",
        area=0.0,  # We'll compute this
        duration=60.0,  # 1 hour fire
        ffmc=92.0,  # High fire danger
        bui=85.0,
        wind=25.0,  # Strong wind
        wind_dir=270.0,  # From west
    )
    
    # Run forward model
    predicted_area = forward_model(params, obs)
    
    logger.info(f"Predicted fire area: {float(predicted_area):.1f} m² ({float(predicted_area)/10000:.2f} ha)")
    
    return float(predicted_area)


def demo_parameter_sensitivity():
    """
    Demo 2: Parameter sensitivity - how do adjustments affect fire spread?
    """
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Parameter Sensitivity")
    logger.info("=" * 60)
    
    # Base observation
    obs = create_observation(
        duration=60.0,
        ffmc=90.0,
        bui=80.0,
        wind=20.0,
    )
    
    # Wind adjustment sensitivity
    logger.info("\nWind Adjustment Sensitivity:")
    logger.info("-" * 40)
    wind_adjustments = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    for wind_adj in wind_adjustments:
        params = FireParams(wind_adj=wind_adj)
        area = forward_model(params, obs)
        logger.info(f"  wind_adj={wind_adj:.2f}: area={float(area):>8.0f} m²")
    
    # FFMC adjustment sensitivity
    logger.info("\nFFMC Adjustment Sensitivity:")
    logger.info("-" * 40)
    ffmc_adjustments = [-10.0, -5.0, 0.0, 5.0, 10.0]
    
    for ffmc_adj in ffmc_adjustments:
        params = FireParams(ffmc_adj=ffmc_adj)
        area = forward_model(params, obs)
        logger.info(f"  ffmc_adj={ffmc_adj:+5.1f}: area={float(area):>8.0f} m²")


def demo_gradient_computation():
    """
    Demo 3: Gradient computation - show that we can differentiate through the model.
    """
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Gradient Computation (Differentiability)")
    logger.info("=" * 60)
    
    obs = create_observation(
        area=25000.0,  # Target area
        duration=60.0,
        ffmc=91.0,
        bui=82.0,
        wind=22.0,
    )
    
    # Define loss function: squared relative error
    def loss_fn(params_array):
        wind_adj, ffmc_adj = params_array
        params = FireParams(wind_adj=wind_adj, ffmc_adj=ffmc_adj)
        pred_area = forward_model(params, obs)
        target = jnp.array(obs.observed_area)
        return ((pred_area - target) / target) ** 2
    
    # Compute gradient
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    
    # At default parameters
    params_array = jnp.array([1.0, 0.0])
    loss, grads = loss_and_grad(params_array)
    
    logger.info(f"\nAt default params [wind_adj=1.0, ffmc_adj=0.0]:")
    logger.info(f"  Loss: {float(loss):.6f}")
    logger.info(f"  ∂L/∂(wind_adj): {float(grads[0]):.6f}")
    logger.info(f"  ∂L/∂(ffmc_adj): {float(grads[1]):.6f}")
    
    # Interpret gradients
    if grads[0] > 0:
        logger.info("  → Gradient suggests DECREASING wind_adj to reduce loss")
    else:
        logger.info("  → Gradient suggests INCREASING wind_adj to reduce loss")


def demo_calibration():
    """
    Demo 4: Full calibration - recover hidden parameters from synthetic data.
    """
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Full Calibration Workflow")
    logger.info("=" * 60)
    
    # True parameters (we pretend we don't know these)
    true_params = FireParams(wind_adj=1.3, ffmc_adj=2.5)
    logger.info(f"TRUE (hidden) parameters: wind_adj={true_params.wind_adj}, ffmc_adj={true_params.ffmc_adj}")
    
    # Generate synthetic "observed" fires using true parameters
    logger.info("\nGenerating synthetic observations...")
    observations = []
    
    weather_scenarios = [
        {"ffmc": 88.0, "bui": 75.0, "wind": 18.0, "wind_dir": 250.0, "duration": 60.0},
        {"ffmc": 90.0, "bui": 80.0, "wind": 21.0, "wind_dir": 270.0, "duration": 75.0},
        {"ffmc": 92.0, "bui": 82.0, "wind": 24.0, "wind_dir": 280.0, "duration": 90.0},
        {"ffmc": 94.0, "bui": 85.0, "wind": 27.0, "wind_dir": 260.0, "duration": 105.0},
        {"ffmc": 91.0, "bui": 78.0, "wind": 22.0, "wind_dir": 275.0, "duration": 120.0},
    ]
    
    for i, scenario in enumerate(weather_scenarios):
        # Create base observation
        base_obs = create_observation(
            fire_id=f"train_{i:02d}",
            area=0.0,
            **scenario,
        )
        
        # Generate "observed" area using true params + noise
        true_area = float(forward_model(true_params, base_obs))
        noise_factor = 1.0 + 0.05 * np.random.randn()  # 5% noise
        observed_area = true_area * noise_factor
        
        # Create observation with observed area
        obs = Observation(
            fire_id=base_obs.fire_id,
            x_ign=base_obs.x_ign,
            y_ign=base_obs.y_ign,
            observed_area=observed_area,
            duration=base_obs.duration,
            ffmc=base_obs.ffmc,
            bui=base_obs.bui,
            wind_speed=base_obs.wind_speed,
            wind_dir=base_obs.wind_dir,
            fuel_grid=base_obs.fuel_grid,
            x_coords=base_obs.x_coords,
            y_coords=base_obs.y_coords,
        )
        observations.append(obs)
        logger.info(f"  {obs.fire_id}: true={true_area:.0f} m², observed={observed_area:.0f} m²")
    
    # Run calibration
    logger.info("\nRunning calibration (this uses gradient descent)...")
    result = calibrate_wind_and_moisture(
        observations,
        learning_rate=0.1,
        n_iterations=100,
    )
    
    # Results
    logger.info("\n" + "-" * 40)
    logger.info("CALIBRATION RESULTS:")
    logger.info("-" * 40)
    logger.info(f"  Iterations: {result.n_iter}")
    logger.info(f"  Converged: {result.converged}")
    logger.info(f"  Final loss: {result.final_loss:.6f}")
    logger.info(f"  Recovered wind_adj: {result.params.wind_adj:.4f} (true: {true_params.wind_adj})")
    logger.info(f"  Recovered ffmc_adj: {result.params.ffmc_adj:.4f} (true: {true_params.ffmc_adj})")
    
    # Error analysis
    wind_error = abs(result.params.wind_adj - true_params.wind_adj) / true_params.wind_adj * 100
    ffmc_error = abs(result.params.ffmc_adj - true_params.ffmc_adj) / abs(true_params.ffmc_adj) * 100
    logger.info(f"  Wind adjustment error: {wind_error:.1f}%")
    logger.info(f"  FFMC adjustment error: {ffmc_error:.1f}%")
    
    # Validation
    logger.info("\nValidation on training set:")
    metrics = validate(result.params, observations)
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return result, observations


def demo_advanced_calibration():
    """
    Demo 5: Advanced calibration - multiple parameters with custom bounds.
    """
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 5: Advanced Multi-Parameter Calibration")
    logger.info("=" * 60)
    
    # Create observations
    observations = [
        create_observation(
            fire_id=f"adv_{i}",
            area=30000.0 + i * 10000,
            duration=60.0 + i * 15,
            ffmc=90.0 + i,
            bui=80.0 + i * 2,
            wind=20.0 + i * 2,
        )
        for i in range(4)
    ]
    
    # Calibrate more parameters
    logger.info("Calibrating: wind_adj, ffmc_adj, ros_scale")
    
    result = calibrate(
        observations,
        param_names=["wind_adj", "ffmc_adj", "ros_scale"],
        initial_values={"wind_adj": 1.0, "ffmc_adj": 0.0, "ros_scale": 1.0},
        bounds={
            "wind_adj": (0.5, 2.0),
            "ffmc_adj": (-10.0, 10.0),
            "ros_scale": (0.5, 2.0),
        },
        learning_rate=0.05,
        n_iterations=80,
        reg_strength=0.02,  # Stronger regularization
    )
    
    logger.info("\nResults:")
    logger.info(f"  wind_adj: {result.params.wind_adj:.4f}")
    logger.info(f"  ffmc_adj: {result.params.ffmc_adj:.4f}")
    logger.info(f"  ros_scale: {result.params.ros_scale:.4f}")
    logger.info(f"  Final loss: {result.final_loss:.6f}")
    
    return result


def main():
    """Run all demos."""
    logger.info("=" * 60)
    logger.info("IGNACIO - Differentiable Fire Spread Calibration Demo")
    logger.info("=" * 60)
    logger.info("")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run demos
        demo_basic_forward_model()
        demo_parameter_sensitivity()
        demo_gradient_computation()
        result, observations = demo_calibration()
        demo_advanced_calibration()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL DEMOS COMPLETE!")
        logger.info("=" * 60)
        
        # Summary
        logger.info("\nKey Takeaways:")
        logger.info("  1. The forward model maps (params, weather, fuel) → fire area")
        logger.info("  2. The model is differentiable - we can compute ∂loss/∂params")
        logger.info("  3. Gradient descent efficiently finds optimal calibration parameters")
        logger.info("  4. Wind and fuel moisture adjustments are the primary calibration targets")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())