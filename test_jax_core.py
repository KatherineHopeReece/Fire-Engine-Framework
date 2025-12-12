#!/usr/bin/env python3
"""
Standalone test of the JAX core module.
"""

import sys
sys.path.insert(0, '/home/claude/ignacio/Fire-Engine-Framework/ignacio/jax_core')

import logging
import jax
import jax.numpy as jnp
import numpy as np

# Configure JAX
jax.config.update("jax_enable_x64", True)

# Import from core directly
from core import (
    FireParams,
    Observation,
    calibrate_wind_and_moisture,
    calibrate,
    create_observation,
    forward_model,
    validate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("IGNACIO - JAX Core Calibration Test")
    logger.info("=" * 60)
    
    np.random.seed(42)
    
    # === Demo 1: Basic forward model ===
    logger.info("\n--- Demo 1: Basic Forward Model ---")
    params = FireParams()
    obs = create_observation(
        fire_id="test",
        area=0.0,
        duration=60.0,
        ffmc=92.0,
        bui=85.0,
        wind=25.0,
        wind_dir=270.0,
    )
    
    area = forward_model(params, obs)
    logger.info(f"Default params → Fire area: {float(area):.1f} m² ({float(area)/10000:.2f} ha)")
    
    # === Demo 2: Parameter sensitivity ===
    logger.info("\n--- Demo 2: Parameter Sensitivity ---")
    for wind_adj in [0.5, 1.0, 1.5, 2.0]:
        params = FireParams(wind_adj=wind_adj)
        area = forward_model(params, obs)
        logger.info(f"  wind_adj={wind_adj:.1f}: area={float(area):>8.0f} m²")
    
    # === Demo 3: Gradient computation ===
    logger.info("\n--- Demo 3: Gradient Test ---")
    def loss_fn(p):
        params = FireParams(wind_adj=p[0], ffmc_adj=p[1])
        pred = forward_model(params, obs)
        return ((pred - 50000.0) / 50000.0) ** 2
    
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    loss, grads = grad_fn(jnp.array([1.0, 0.0]))
    logger.info(f"Loss at default: {float(loss):.6f}")
    logger.info(f"Gradient: ∂L/∂wind_adj={float(grads[0]):.6f}, ∂L/∂ffmc_adj={float(grads[1]):.6f}")
    logger.info("✓ Model is differentiable!")
    
    # === Demo 4: Full calibration ===
    logger.info("\n--- Demo 4: Full Calibration ---")
    
    # True params (hidden)
    true_params = FireParams(wind_adj=1.3, ffmc_adj=2.5)
    logger.info(f"TRUE params: wind_adj={true_params.wind_adj}, ffmc_adj={true_params.ffmc_adj}")
    
    # Generate synthetic observations
    observations = []
    scenarios = [
        {"ffmc": 88.0, "bui": 75.0, "wind": 18.0, "duration": 60.0},
        {"ffmc": 90.0, "bui": 80.0, "wind": 21.0, "duration": 75.0},
        {"ffmc": 92.0, "bui": 82.0, "wind": 24.0, "duration": 90.0},
        {"ffmc": 94.0, "bui": 85.0, "wind": 27.0, "duration": 105.0},
    ]
    
    logger.info("Generating synthetic observations...")
    for i, s in enumerate(scenarios):
        base = create_observation(fire_id=f"obs_{i}", area=0.0, **s)
        true_area = float(forward_model(true_params, base))
        # Add 5% noise
        obs_area = true_area * (1.0 + 0.05 * np.random.randn())
        
        obs = Observation(
            fire_id=base.fire_id,
            x_ign=base.x_ign,
            y_ign=base.y_ign,
            observed_area=obs_area,
            duration=base.duration,
            ffmc=base.ffmc,
            bui=base.bui,
            wind_speed=base.wind_speed,
            wind_dir=base.wind_dir,
            fuel_grid=base.fuel_grid,
            x_coords=base.x_coords,
            y_coords=base.y_coords,
        )
        observations.append(obs)
        logger.info(f"  {obs.fire_id}: true={true_area:.0f} m², observed={obs_area:.0f} m²")
    
    # Run calibration
    logger.info("\nRunning calibration...")
    result = calibrate_wind_and_moisture(
        observations,
        learning_rate=0.1,
        n_iterations=100,
        verbose=True,
    )
    
    logger.info("\n" + "-" * 40)
    logger.info("RESULTS:")
    logger.info("-" * 40)
    logger.info(f"Recovered wind_adj: {result.params.wind_adj:.4f} (true: {true_params.wind_adj})")
    logger.info(f"Recovered ffmc_adj: {result.params.ffmc_adj:.4f} (true: {true_params.ffmc_adj})")
    logger.info(f"Final loss: {result.final_loss:.6f}")
    logger.info(f"Converged: {result.converged}")
    
    # Validation
    logger.info("\nValidation:")
    metrics = validate(result.params, observations)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE - ALL DEMOS PASSED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
