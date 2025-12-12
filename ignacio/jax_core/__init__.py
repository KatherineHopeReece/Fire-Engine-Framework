"""
JAX-based Differentiable Core for Ignacio.

This subpackage provides JAX-compatible implementations of fire behavior
calculations, enabling gradient-based calibration and optimization.

Modules
-------
core : Improved differentiable fire model (RECOMMENDED)
fbp_jax : Fire Behavior Prediction (ISI, ROS by fuel type)
spread_jax : Richards' elliptical fire spread equations
calibration : Gradient-based parameter calibration

Quick Start
-----------
>>> from ignacio.jax_core import (
...     calibrate_wind_and_moisture,
...     create_observation,
...     FireParams,
... )
>>> 
>>> # Create observation from fire data
>>> obs = create_observation(
...     area=50000.0,      # Observed area (m²)
...     duration=60.0,     # Duration (minutes)
...     ffmc=92.0,         # Fine Fuel Moisture Code
...     wind=25.0,         # Wind speed (km/h)
...     wind_dir=270.0,    # Wind from west
... )
>>> 
>>> # Calibrate wind and moisture adjustments
>>> result = calibrate_wind_and_moisture([obs])
>>> print(f"Wind adjustment: {result.params.wind_adj:.3f}")
>>> print(f"FFMC adjustment: {result.params.ffmc_adj:.3f}")
"""

# New improved core (RECOMMENDED)
from .core import (
    # Parameter containers
    FireParams,
    FireGrids,
    Observation,
    CalibrationResult as CalibResult,
    # FBP functions
    compute_isi,
    compute_ros_c2,
    compute_lb_ratio,
    compute_ros_components,
    fbp_pipeline,
    # Spatial operations
    bilinear_interp,
    sample_grids,
    # Richards' equations
    compute_tangents,
    ros_to_ellipse,
    richards_velocity,
    # Simulation
    create_perimeter,
    evolve_step_euler,
    evolve_step_heun,
    simulate_fire,
    compute_area,
    # Forward model
    forward_model,
    # Loss functions
    area_loss,
    calibration_loss,
    # Calibration
    calibrate,
    calibrate_wind_and_moisture,
    # Convenience
    create_observation,
    validate,
)

# Legacy exports (for backward compatibility)
from .fbp_jax import (
    FBPCalibrationParams,
    calculate_isi_jax,
    compute_lb_ratio_jax,
    compute_ros_by_fuel_jax,
    compute_ros_components_jax,
    compute_ros_grid_jax,
    default_params,
    fbp_pipeline_jax,
    ros_c1_jax,
    ros_c2_jax,
    ros_c3_jax,
    ros_c4_jax,
    ros_c5_jax,
    ros_c7_jax,
    ros_d1_jax,
    ros_o1_jax,
    ros_o1b_jax,
    ros_s1_jax,
    ros_s2_jax,
    ros_s3_jax,
)

from .spread_jax import (
    FireParamsJAX,
    bilinear_interpolate_jax,
    combined_loss_jax,
    compute_fire_area_jax,
    compute_spatial_derivatives_jax,
    create_fire_params,
    create_initial_perimeter_jax,
    evolve_perimeter_step_jax,
    richards_velocity_jax,
    ros_to_ellipse_params_jax,
    sample_fire_params_jax,
    simulate_fire_jax,
)

from .calibration import (
    CalibrationConfig,
    CalibrationObservation,
    CalibrationResult,
    calibrate_parameters,
    clip_to_bounds,
    create_synthetic_observation,
    quick_calibrate_wind_moisture,
    run_fire_forward_model,
    validate_calibration,
)

# Level-set method (robust topology)
from .levelset import (
    LevelSetGrids,
    initialize_phi,
    compute_gradient,
    compute_gradient_magnitude_upwind,
    compute_elliptical_speed,
    evolve_phi,
    simulate_fire_levelset,
    compute_burned_area,
    compute_burned_area_hard,
    simulate_fire_levelset_with_area,
)

# Level-set calibration
from .levelset_calibration import (
    CalibrationParams,
    CalibrationResult as LevelSetCalibrationResult,
    calibrate_to_observed,
    create_obs_mask_from_polygon_fast,
    apply_calibration_params,
    area_loss as ls_area_loss,
    iou_loss,
    combined_loss as ls_combined_loss,
)

# Visualization
from .visualization import (
    load_observed_fire,
    plot_comparison,
    plot_calibration_progress,
    compute_metrics,
    perimeter_to_polygon,
    levelset_to_polygon,
)


__all__ = [
    # New core (recommended)
    "FireParams",
    "FireGrids", 
    "Observation",
    "CalibResult",
    "compute_isi",
    "compute_ros_c2",
    "compute_lb_ratio",
    "compute_ros_components",
    "fbp_pipeline",
    "bilinear_interp",
    "sample_grids",
    "compute_tangents",
    "ros_to_ellipse",
    "richards_velocity",
    "create_perimeter",
    "evolve_step_euler",
    "evolve_step_heun",
    "simulate_fire",
    "compute_area",
    "forward_model",
    "area_loss",
    "calibration_loss",
    "calibrate",
    "calibrate_wind_and_moisture",
    "create_observation",
    "validate",
    # Legacy FBP JAX
    "FBPCalibrationParams",
    "calculate_isi_jax",
    "compute_lb_ratio_jax",
    "compute_ros_by_fuel_jax",
    "compute_ros_components_jax",
    "compute_ros_grid_jax",
    "default_params",
    "fbp_pipeline_jax",
    "ros_c1_jax",
    "ros_c2_jax",
    "ros_c3_jax",
    "ros_c4_jax",
    "ros_c5_jax",
    "ros_c7_jax",
    "ros_d1_jax",
    "ros_o1_jax",
    "ros_o1b_jax",
    "ros_s1_jax",
    "ros_s2_jax",
    "ros_s3_jax",
    # Legacy Spread JAX
    "FireParamsJAX",
    "bilinear_interpolate_jax",
    "combined_loss_jax",
    "compute_fire_area_jax",
    "compute_spatial_derivatives_jax",
    "create_fire_params",
    "create_initial_perimeter_jax",
    "evolve_perimeter_step_jax",
    "richards_velocity_jax",
    "ros_to_ellipse_params_jax",
    "sample_fire_params_jax",
    "simulate_fire_jax",
    # Legacy Calibration
    "CalibrationConfig",
    "CalibrationObservation",
    "CalibrationResult",
    "calibrate_parameters",
    "clip_to_bounds",
    "create_synthetic_observation",
    "quick_calibrate_wind_moisture",
    "run_fire_forward_model",
    "validate_calibration",
    # Level-set (robust topology)
    "LevelSetGrids",
    "initialize_phi",
    "simulate_fire_levelset",
    "compute_burned_area",
    "compute_burned_area_hard",
    "simulate_fire_levelset_with_area",
    # Level-set calibration
    "CalibrationParams",
    "LevelSetCalibrationResult",
    "calibrate_to_observed",
    "create_obs_mask_from_polygon_fast",
    "apply_calibration_params",
    "ls_area_loss",
    "iou_loss",
    "ls_combined_loss",
    # Visualization
    "load_observed_fire",
    "plot_comparison",
    "plot_calibration_progress",
    "compute_metrics",
    "perimeter_to_polygon",
    "levelset_to_polygon",
]
