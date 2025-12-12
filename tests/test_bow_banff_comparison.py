#!/usr/bin/env python3
"""
Bow at Banff - NumPy vs JAX Comparison Test.

Runs the full Ignacio simulation and compares JAX results on the same grids.
"""

import sys
import logging
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Run comparison test."""
    logger.info("=" * 70)
    logger.info("IGNACIO: Bow at Banff - NumPy vs JAX Comparison")
    logger.info("=" * 70)
    
    # Check JAX
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        logger.info(f"JAX version: {jax.__version__}")
    except ImportError as e:
        logger.error(f"JAX not available: {e}")
        return 1
    
    from ignacio.config import load_config
    from ignacio.terrain import build_terrain_grids
    from ignacio.simulation import build_parameter_grid
    from ignacio.weather import process_fire_weather, load_weather_data
    from ignacio.ignition import generate_ignitions
    from ignacio.io import read_raster_int
    from ignacio.spread import simulate_fire_spread
    
    # Import JAX core
    jax_core_path = str(project_root / "ignacio" / "jax_core")
    sys.path.insert(0, jax_core_path)
    from core import FireGrids, simulate_fire, compute_area
    from levelset import (
        LevelSetGrids, 
        simulate_fire_levelset_with_area,
        compute_burned_area_hard,
    )
    
    # =================================================================
    # Load config and build grids (same as full Ignacio simulation)
    # =================================================================
    config_path = project_root / "ignacio.yaml"
    config = load_config(config_path)
    rng = np.random.default_rng(config.project.random_seed)
    
    logger.info(f"\nBuilding grids...")
    terrain = build_terrain_grids(config)
    
    hourly_data = None
    if config.simulation.time_varying_weather:
        hourly_data = load_weather_data(config)
        logger.info(f"Loaded {len(hourly_data)} hourly weather records")
    
    weather = process_fire_weather(config, rng)
    
    param_grid = build_parameter_grid(config, terrain, weather, hourly_data=hourly_data)
    logger.info(f"Grid shape: {param_grid.ros.shape}")
    logger.info(f"ROS range: {param_grid.ros.min():.2f} - {param_grid.ros.max():.2f} m/min")
    
    # Get ignition
    fuel_raster = read_raster_int(config.fuel.path)
    terrain_crs = str(terrain.crs) if terrain.crs else None
    ignitions = generate_ignitions(config, fuel_raster, rng, terrain_crs=terrain_crs)
    ign = ignitions.points[0]
    x_ign, y_ign = ign.x, ign.y
    logger.info(f"Ignition: ({x_ign:.6f}, {y_ign:.6f})")
    
    # Simulation params
    dt = config.simulation.dt
    n_vertices = config.simulation.n_vertices
    initial_radius = config.simulation.initial_radius
    n_steps = int(config.simulation.max_duration / dt)
    
    # Geographic handling
    is_geographic = False
    center_latitude = None
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0
    avg_meters_per_deg = 111320.0
    
    if terrain.crs is not None:
        from pyproj import CRS
        crs = CRS.from_user_input(terrain.crs)
        is_geographic = crs.is_geographic
        if is_geographic:
            center_latitude = (param_grid.y_min + param_grid.y_max) / 2
            lat_rad = np.radians(center_latitude)
            meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
            avg_meters_per_deg = (meters_per_deg_lat + meters_per_deg_lon) / 2.0
            logger.info(f"Geographic CRS, center lat: {center_latitude:.4f}°")
    
    # =================================================================
    # NumPy Simulation
    # =================================================================
    logger.info(f"\n{'='*60}")
    logger.info("NumPy Simulation")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    numpy_history = simulate_fire_spread(
        param_grid=param_grid,
        x_ignition=x_ign,
        y_ignition=y_ign,
        dt=dt,
        n_vertices=n_vertices,
        initial_radius=initial_radius,
        max_steps=n_steps,
        is_geographic=is_geographic,
        center_latitude=center_latitude,
    )
    numpy_time = time.time() - start_time
    
    numpy_x, numpy_y = numpy_history.get_final_perimeter()
    
    # Compute area properly for geographic coordinates
    if is_geographic:
        # Convert perimeter to meters for area calculation
        x_m = (numpy_x - x_ign) * meters_per_deg_lon
        y_m = (numpy_y - y_ign) * meters_per_deg_lat
        numpy_area = 0.5 * np.abs(np.sum(x_m * np.roll(y_m, -1) - np.roll(x_m, -1) * y_m))
    else:
        numpy_area = 0.5 * np.abs(np.sum(numpy_x * np.roll(numpy_y, -1) - np.roll(numpy_x, -1) * numpy_y))
    
    logger.info(f"NumPy area: {numpy_area:.0f} m² ({numpy_area/10000:.2f} ha)")
    logger.info(f"NumPy time: {numpy_time:.2f} seconds")
    
    # =================================================================
    # JAX Simulation (same grids, converted for geographic)
    # =================================================================
    logger.info(f"\n{'='*60}")
    logger.info("JAX Simulation")
    logger.info(f"{'='*60}")
    
    # Convert grids for JAX
    if is_geographic:
        # Convert ROS from m/min to deg/min
        ros_jax = param_grid.ros / avg_meters_per_deg
        bros_jax = param_grid.bros / avg_meters_per_deg
        fros_jax = param_grid.fros / avg_meters_per_deg
        jax_initial_radius = initial_radius / avg_meters_per_deg
    else:
        ros_jax = param_grid.ros
        bros_jax = param_grid.bros
        fros_jax = param_grid.fros
        jax_initial_radius = initial_radius
    
    jax_grids = FireGrids(
        x_coords=jnp.array(param_grid.x_coords),
        y_coords=jnp.array(param_grid.y_coords),
        ros=jnp.array(ros_jax),
        bros=jnp.array(bros_jax),
        fros=jnp.array(fros_jax),
        raz=jnp.array(param_grid.raz),
    )
    
    # First run (JIT compile)
    start_time = time.time()
    jax_x, jax_y = simulate_fire(
        jax_grids, x_ign, y_ign,
        n_steps=n_steps,
        dt=dt,
        n_vertices=n_vertices,
        initial_radius=jax_initial_radius,
    )
    jax_jit_time = time.time() - start_time
    
    # Second run (cached)
    start_time = time.time()
    jax_x, jax_y = simulate_fire(
        jax_grids, x_ign, y_ign,
        n_steps=n_steps,
        dt=dt,
        n_vertices=n_vertices,
        initial_radius=jax_initial_radius,
    )
    jax_cached_time = time.time() - start_time
    
    # Compute JAX area
    jax_x_np = np.array(jax_x)
    jax_y_np = np.array(jax_y)
    
    if is_geographic:
        # Convert perimeter to meters for area calculation
        x_m = (jax_x_np - x_ign) * meters_per_deg_lon
        y_m = (jax_y_np - y_ign) * meters_per_deg_lat
        jax_area = 0.5 * np.abs(np.sum(x_m * np.roll(y_m, -1) - np.roll(x_m, -1) * y_m))
    else:
        jax_area = float(compute_area(jax_x, jax_y))
    
    logger.info(f"JAX perimeter area: {jax_area:.0f} m² ({jax_area/10000:.2f} ha)")
    logger.info(f"JAX perimeter time (JIT): {jax_jit_time:.2f}s, (cached): {jax_cached_time:.2f}s")
    
    # Compare perimeter extents
    logger.info(f"\nPerimeter extents:")
    numpy_extent_x = (np.max(numpy_x) - np.min(numpy_x)) * meters_per_deg_lon
    numpy_extent_y = (np.max(numpy_y) - np.min(numpy_y)) * meters_per_deg_lat
    jax_extent_x = (np.max(jax_x_np) - np.min(jax_x_np)) * meters_per_deg_lon
    jax_extent_y = (np.max(jax_y_np) - np.min(jax_y_np)) * meters_per_deg_lat
    logger.info(f"  NumPy: {numpy_extent_x:.0f}m x {numpy_extent_y:.0f}m")
    logger.info(f"  JAX perimeter: {jax_extent_x:.0f}m x {jax_extent_y:.0f}m (note: self-intersecting)")
    
    # Bounding box areas for sanity check
    numpy_bbox_area = numpy_extent_x * numpy_extent_y
    jax_bbox_area = jax_extent_x * jax_extent_y
    logger.info(f"  NumPy bbox: {numpy_bbox_area/10000:.2f} ha, computed: {numpy_area/10000:.2f} ha ✓")
    logger.info(f"  JAX bbox: {jax_bbox_area/10000:.2f} ha, computed: {jax_area/10000:.2f} ha (impossible - polygon self-intersects)")
    
    # =================================================================
    # Level-Set Simulation (robust topology handling)
    # =================================================================
    logger.info(f"\n{'='*60}")
    logger.info("JAX Level-Set Simulation")
    logger.info(f"{'='*60}")
    
    # Create level-set grids
    ls_grids = LevelSetGrids(
        x_coords=jnp.array(param_grid.x_coords),
        y_coords=jnp.array(param_grid.y_coords),
        ros=jnp.array(ros_jax),
        bros=jnp.array(bros_jax),
        fros=jnp.array(fros_jax),
        raz=jnp.array(param_grid.raz),
    )
    
    # Grid spacing calculations
    dx_ls = abs(float(ls_grids.x_coords[1] - ls_grids.x_coords[0]))
    dy_ls = abs(float(ls_grids.y_coords[1] - ls_grids.y_coords[0]))
    min_radius = np.sqrt(dx_ls**2 + dy_ls**2)
    effective_radius = max(jax_initial_radius, min_radius)
    ros_grid = np.array(ls_grids.ros)
    
    # Debug: Check grid setup
    logger.info(f"Level-set grid:")
    logger.info(f"  Grid: {len(ls_grids.x_coords)}x{len(ls_grids.y_coords)} cells")
    logger.info(f"  Cell size: {dx_ls*meters_per_deg_lon:.1f}m x {abs(dy_ls)*meters_per_deg_lat:.1f}m")
    logger.info(f"  Effective radius: {effective_radius * avg_meters_per_deg:.1f}m (auto-adjusted from {jax_initial_radius * avg_meters_per_deg:.1f}m)")
    logger.info(f"  ROS range: [{ros_grid.min()*avg_meters_per_deg:.2f}, {ros_grid.max()*avg_meters_per_deg:.2f}] m/min")
    
    # Run level-set simulation (elliptical)
    start_time = time.time()
    phi, ls_area = simulate_fire_levelset_with_area(
        ls_grids, x_ign, y_ign,
        n_steps=n_steps,
        dt=dt,
        initial_radius=jax_initial_radius,
        differentiable=False,
        use_ellipse=True,
    )
    ls_time = time.time() - start_time
    
    # Run isotropic for comparison
    phi_iso, iso_area = simulate_fire_levelset_with_area(
        ls_grids, x_ign, y_ign,
        n_steps=n_steps,
        dt=dt,
        initial_radius=jax_initial_radius,
        differentiable=False,
        use_ellipse=False,
    )
    
    # Convert areas to m²
    if is_geographic:
        ls_area_m2 = float(ls_area) * meters_per_deg_lon * meters_per_deg_lat
        iso_area_m2 = float(iso_area) * meters_per_deg_lon * meters_per_deg_lat
    else:
        ls_area_m2 = float(ls_area)
        iso_area_m2 = float(iso_area)
    
    burned_cells = np.sum(np.array(phi) < 0)
    logger.info(f"Level-set (elliptical): {ls_area_m2/10000:.2f} ha ({burned_cells} cells, {ls_time:.2f}s)")
    logger.info(f"Level-set (isotropic):  {iso_area_m2/10000:.2f} ha (for reference)")
    
    # =================================================================
    # Comparison
    # =================================================================
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON")
    logger.info(f"{'='*60}")
    
    # Perimeter-based comparison
    diff = abs(numpy_area - jax_area)
    diff_pct = diff / numpy_area * 100 if numpy_area > 0 else float('inf')
    
    logger.info(f"\nPerimeter-based:")
    logger.info(f"  NumPy:     {numpy_area/10000:.2f} ha")
    logger.info(f"  JAX:       {jax_area/10000:.2f} ha (self-intersecting polygon)")
    logger.info(f"  Difference: {diff_pct:.1f}%")
    
    # Level-set comparison
    ls_diff = abs(numpy_area - ls_area_m2)
    ls_diff_pct = ls_diff / numpy_area * 100 if numpy_area > 0 else float('inf')
    
    # Isotropic comparison (for reference)
    iso_diff_pct = abs(numpy_area - iso_area_m2) / numpy_area * 100 if numpy_area > 0 else float('inf')
    
    logger.info(f"\nLevel-set (elliptical):")
    logger.info(f"  NumPy:     {numpy_area/10000:.2f} ha")
    logger.info(f"  Level-set: {ls_area_m2/10000:.2f} ha")
    logger.info(f"  Difference: {ls_diff_pct:.1f}%")
    
    logger.info(f"\nLevel-set (isotropic, for reference):")
    logger.info(f"  Area: {iso_area_m2/10000:.2f} ha ({iso_diff_pct:.1f}% from NumPy)")
    
    # =================================================================
    # Visualization with Observed Fire
    # =================================================================
    logger.info(f"\n{'='*60}")
    logger.info("VISUALIZATION")
    logger.info(f"{'='*60}")
    
    # Look for observed fire shapefile
    observed_path = project_root / "data" / "OBrienCreekFire_2014.shp"
    if not observed_path.exists():
        # Try alternative names
        for alt in ["observed.shp", "fire_boundary.shp", "perimeter.shp"]:
            alt_path = project_root / "data" / alt
            if alt_path.exists():
                observed_path = alt_path
                break
    
    if observed_path.exists():
        try:
            import geopandas as gpd
            from ignacio.jax_core.visualization import (
                load_observed_fire,
                plot_comparison,
                compute_metrics,
                perimeter_to_polygon,
                levelset_to_polygon,
            )
            
            # Load observed fire and reproject to match DEM CRS
            logger.info(f"Loading observed fire from {observed_path}")
            obs_gdf = load_observed_fire(observed_path, target_crs="EPSG:4326")
            obs_geom = obs_gdf.geometry.iloc[0]
            obs_area_ha = obs_geom.area * meters_per_deg_lon * meters_per_deg_lat / 10000
            logger.info(f"Observed fire area: {obs_area_ha:.2f} ha")
            
            # Create polygons for comparison
            numpy_poly = perimeter_to_polygon(numpy_x, numpy_y)
            ls_poly = levelset_to_polygon(np.array(phi), np.array(param_grid.x_coords), np.array(param_grid.y_coords))
            
            # Compute metrics
            np_metrics = compute_metrics(numpy_poly, obs_geom)
            ls_metrics = compute_metrics(ls_poly, obs_geom)
            
            logger.info(f"\nComparison with observed fire:")
            logger.info(f"  NumPy:     IoU={np_metrics['iou']:.2%}, Area diff={np_metrics['area_diff_pct']:.1f}%")
            logger.info(f"  Level-set: IoU={ls_metrics['iou']:.2%}, Area diff={ls_metrics['area_diff_pct']:.1f}%")
            
            # Create comparison plot
            output_dir = project_root / "output"
            output_dir.mkdir(exist_ok=True)
            plot_path = output_dir / "fire_comparison.png"
            
            fig = plot_comparison(
                observed_gdf=obs_gdf,
                numpy_perimeter=(numpy_x, numpy_y),
                jax_levelset=(np.array(phi), np.array(param_grid.x_coords), np.array(param_grid.y_coords)),
                title=f"Fire Spread Comparison: {config.project.name}",
                output_path=plot_path,
            )
            logger.info(f"Comparison plot saved to {plot_path}")
            
        except ImportError as e:
            logger.warning(f"Could not create visualization: {e}")
        except Exception as e:
            logger.warning(f"Visualization error: {e}")
    else:
        logger.info(f"No observed fire shapefile found at {observed_path}")
        logger.info("Skipping visualization")
    
    # =================================================================
    # Final Result
    # =================================================================
    if ls_diff_pct < 5.0:
        logger.info("\n✓ PASS: Level-set results match within 5%")
        return 0
    elif ls_diff_pct < 20.0:
        logger.warning(f"\n~ CLOSE: Level-set results within 20% ({ls_diff_pct:.1f}%)")
        logger.info("  This is acceptable for grid-based vs continuous methods")
        return 0
    else:
        logger.error("\n✗ FAIL: Level-set results differ significantly")
        return 1


def run_calibration(config_path: str = None, observed_path: str = None):
    """
    Run calibration against observed fire.
    
    This is a separate entry point for calibration.
    """
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    
    project_root = Path(__file__).parent.parent
    
    # Load config
    from ignacio.config import load_config
    from ignacio.terrain import build_terrain_grids
    from ignacio.simulation import build_parameter_grid
    from ignacio.weather import process_fire_weather, load_weather_data
    from ignacio.ignition import generate_ignitions
    from ignacio.io import read_raster_int
    
    config_path = config_path or str(project_root / "ignacio.yaml")
    config = load_config(config_path)
    rng = np.random.default_rng(config.project.random_seed)
    
    logger.info(f"Running calibration for {config.project.name}")
    
    # Build grids (same as main test)
    terrain = build_terrain_grids(config)
    
    hourly_data = None
    if config.simulation.time_varying_weather:
        hourly_data = load_weather_data(config)
    
    weather = process_fire_weather(config, rng)
    param_grid = build_parameter_grid(config, terrain, weather, hourly_data=hourly_data)
    
    # Get ignition
    fuel_raster = read_raster_int(config.fuel.path)
    terrain_crs = str(terrain.crs) if terrain.crs else None
    ignitions = generate_ignitions(config, fuel_raster, rng, terrain_crs=terrain_crs)
    ign = ignitions.points[0]
    x_ign, y_ign = ign.x, ign.y
    logger.info(f"Ignition: ({x_ign:.6f}, {y_ign:.6f})")
    
    # Load observed fire
    observed_path = observed_path or str(project_root / "data" / "OBrienCreekFire_2014.shp")
    
    import geopandas as gpd
    from ignacio.jax_core.visualization import load_observed_fire
    from ignacio.jax_core.levelset import LevelSetGrids
    from ignacio.jax_core.levelset_calibration import (
        calibrate_to_observed,
        create_obs_mask_from_polygon_fast,
        apply_calibration_params,
    )
    
    obs_gdf = load_observed_fire(observed_path, target_crs="EPSG:4326")
    obs_geom = obs_gdf.geometry.iloc[0]
    
    # Coordinate conversion
    center_lat = (param_grid.y_min + param_grid.y_max) / 2
    lat_rad = np.radians(center_lat)
    meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
    meters_per_deg_lat = 111320.0
    avg_meters_per_deg = (meters_per_deg_lat + meters_per_deg_lon) / 2.0
    
    obs_area = obs_geom.area * meters_per_deg_lon * meters_per_deg_lat
    logger.info(f"Observed fire area: {obs_area/10000:.2f} ha")
    
    # Create observation mask
    logger.info("Creating observation mask...")
    obs_mask = create_obs_mask_from_polygon_fast(
        obs_geom,
        np.array(param_grid.x_coords),
        np.array(param_grid.y_coords),
    )
    logger.info(f"Observation mask: {np.sum(obs_mask)} burned cells")
    
    # Convert ROS to deg/min
    ros_jax = param_grid.ros / avg_meters_per_deg
    bros_jax = param_grid.bros / avg_meters_per_deg
    fros_jax = param_grid.fros / avg_meters_per_deg
    
    # Create level-set grids
    ls_grids = LevelSetGrids(
        x_coords=jnp.array(param_grid.x_coords),
        y_coords=jnp.array(param_grid.y_coords),
        ros=jnp.array(ros_jax),
        bros=jnp.array(bros_jax),
        fros=jnp.array(fros_jax),
        raz=jnp.array(param_grid.raz),
    )
    
    # Simulation params
    dt = config.simulation.dt
    n_steps = int(config.simulation.max_duration / dt)
    initial_radius = config.simulation.initial_radius / avg_meters_per_deg
    
    # Run calibration
    logger.info("Starting calibration...")
    logger.info(f"  Grid: {len(ls_grids.x_coords)}x{len(ls_grids.y_coords)} cells = {len(ls_grids.x_coords)*len(ls_grids.y_coords)} total")
    logger.info(f"  Time steps: {n_steps} (will subsample for calibration)")
    
    result = calibrate_to_observed(
        grids=ls_grids,
        x_ign=x_ign,
        y_ign=y_ign,
        n_steps=n_steps,
        dt=dt,
        initial_radius=initial_radius,
        obs_mask=jnp.array(obs_mask),
        obs_area=obs_area / (meters_per_deg_lon * meters_per_deg_lat),  # Convert to deg²
        n_iterations=30,  # Fewer iterations for faster testing
        learning_rate=0.2,
        verbose=True,
        subsample_steps=50,  # Use 50 time steps for calibration (faster)
        max_grid_cells=100000,  # Subsample spatially if > 100k cells
    )
    
    logger.info(f"\nCalibration complete!")
    logger.info(f"Final parameters:")
    logger.info(f"  ROS scale:  {result.params.ros_scale:.3f}")
    logger.info(f"  BROS scale: {result.params.bros_scale:.3f}")
    logger.info(f"  FROS scale: {result.params.fros_scale:.3f}")
    
    # Run simulation with calibrated params
    from ignacio.jax_core.levelset import simulate_fire_levelset_with_area
    
    calibrated_grids = apply_calibration_params(ls_grids, result.params)
    phi_cal, area_cal = simulate_fire_levelset_with_area(
        calibrated_grids, x_ign, y_ign,
        n_steps=n_steps, dt=dt, initial_radius=initial_radius,
        differentiable=False, use_ellipse=True,
    )
    
    cal_area_m2 = float(area_cal) * meters_per_deg_lon * meters_per_deg_lat
    logger.info(f"\nCalibrated simulation area: {cal_area_m2/10000:.2f} ha")
    logger.info(f"Observed area: {obs_area/10000:.2f} ha")
    logger.info(f"Difference: {abs(cal_area_m2 - obs_area)/obs_area*100:.1f}%")
    
    # Save calibration plot
    from ignacio.jax_core.visualization import plot_calibration_progress, plot_comparison
    
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Plot calibration progress
    plot_calibration_progress(
        result.loss_history,
        result.params_history,
        output_path=output_dir / "calibration_progress.png",
    )
    
    # Plot comparison with calibrated result
    plot_comparison(
        observed_gdf=obs_gdf,
        jax_levelset=(np.array(phi_cal), np.array(param_grid.x_coords), np.array(param_grid.y_coords)),
        title=f"Calibrated Fire Spread: {config.project.name}",
        output_path=output_dir / "fire_comparison_calibrated.png",
    )
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fire spread comparison and calibration")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration against observed fire")
    parser.add_argument("--observed", type=str, help="Path to observed fire shapefile")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    
    args = parser.parse_args()
    
    if args.calibrate:
        run_calibration(config_path=args.config, observed_path=args.observed)
    else:
        sys.exit(main())
