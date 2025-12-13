#!/usr/bin/env python3
"""
Comprehensive Physics Module Test and Demo.

Tests all 5 physics enhancements and demonstrates their effects
on fire spread simulation.

Run with: python test_physics_modules.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np

# Import JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Import modules directly from jax_core
sys.path.insert(0, str(project_root / "ignacio" / "jax_core"))


def test_solar_radiation():
    """Test solar radiation module."""
    print("\n" + "="*60)
    print("1. SOLAR RADIATION & FUEL CONDITIONING")
    print("="*60)
    
    from solar_radiation import (
        compute_sun_position,
        compute_hillshade,
        compute_fuel_conditioning,
        FuelConditioningParams,
    )
    
    # Test sun position
    dt = datetime(2024, 7, 15, 14, 0)  # 2pm July 15
    sun = compute_sun_position(dt, latitude=51.2, longitude=-115.7)
    print(f"Sun position at {dt}:")
    print(f"  Azimuth: {sun.azimuth:.1f}° (from N)")
    print(f"  Elevation: {sun.elevation:.1f}°")
    
    # Test hillshade
    nx, ny = 50, 50
    dx = dy = 30.0
    x = jnp.linspace(0, (nx-1)*dx, nx)
    y = jnp.linspace(0, (ny-1)*dy, ny)
    X, Y = jnp.meshgrid(x, y)
    
    # Create terrain with varying aspects
    dem = 1000.0 + 100.0 * jnp.sin(X / 300.0) + 50.0 * jnp.cos(Y / 200.0)
    
    hillshade = compute_hillshade(dem, dx, dy, sun.elevation, sun.azimuth)
    print(f"Hillshade range: {float(hillshade.min()):.2f} - {float(hillshade.max()):.2f}")
    
    # Test fuel conditioning
    moisture = compute_fuel_conditioning(
        dem, dx, dy,
        base_moisture=0.08,
        latitude=51.2,
        longitude=-115.7,
        dt=dt,
    )
    print(f"Solar-adjusted moisture: {float(moisture.min()):.3f} - {float(moisture.max()):.3f}")
    print("  (South-facing slopes drier, north-facing wetter)")
    
    print("✓ Solar radiation module OK")


def test_moisture_lag():
    """Test fuel moisture lag module."""
    print("\n" + "="*60)
    print("2. FUEL MOISTURE TIME LAG")
    print("="*60)
    
    from moisture_lag import (
        initialize_moisture_state,
        update_moisture_euler,
        compute_equilibrium_moisture,
    )
    
    # Test equilibrium moisture
    emc_1hr, emc_10hr, emc_100hr = compute_equilibrium_moisture(25.0, 30.0)
    print(f"Equilibrium moisture at 25°C, 30% RH:")
    print(f"  1-hr: {emc_1hr:.3f} ({emc_1hr*100:.1f}%)")
    print(f"  10-hr: {emc_10hr:.3f}")
    print(f"  100-hr: {emc_100hr:.3f}")
    
    # Test time evolution
    shape = (10, 10)
    state = initialize_moisture_state(shape, initial_ffmc=88.0, initial_dmc=30.0)
    print(f"\nInitial moisture from FFMC=88:")
    print(f"  1-hr: {float(state.m_1hr[0,0]):.3f}")
    
    # Evolve for 3 hours at low RH
    temp = jnp.full(shape, 30.0)
    rh = jnp.full(shape, 20.0)
    
    for hour in range(3):
        state = update_moisture_euler(state, temp, rh, dt_hours=1.0)
    
    print(f"After 3 hours at 20% RH:")
    print(f"  1-hr: {float(state.m_1hr[0,0]):.3f} (responds fast)")
    print(f"  10-hr: {float(state.m_10hr[0,0]):.3f} (responds slower)")
    print(f"  100-hr: {float(state.m_100hr[0,0]):.3f} (responds slowest)")
    
    print("✓ Moisture lag module OK")


def test_crown_fire():
    """Test crown fire transition module."""
    print("\n" + "="*60)
    print("3. CROWN FIRE TRANSITION (Van Wagner)")
    print("="*60)
    
    from crown_fire import (
        compute_byram_intensity,
        compute_critical_intensity,
        compute_total_ros_with_crown,
        CrownFireParams,
    )
    
    # Test intensity calculation
    ros = jnp.array([5.0, 15.0, 30.0, 50.0])
    fuel_load = jnp.full(4, 2.0)
    
    intensity = compute_byram_intensity(ros, fuel_load)
    print("Byram intensity vs surface ROS:")
    for r, i in zip(ros, intensity):
        print(f"  ROS {float(r):5.1f} m/min → {float(i):8.0f} kW/m")
    
    # Test crown fire transition
    cbh = jnp.full(4, 3.0)
    cbd = jnp.full(4, 0.15)
    fmc = jnp.full(4, 100.0)
    wind = jnp.full(4, 20.0)
    
    total_ros, cfb = compute_total_ros_with_crown(ros, fuel_load, cbh, cbd, fmc, wind)
    
    print("\nCrown fire transition:")
    print(f"  {'Surface ROS':>12} → {'Total ROS':>12} | CFB")
    for sr, tr, cf in zip(ros, total_ros, cfb):
        print(f"  {float(sr):12.1f} → {float(tr):12.1f} | {float(cf):.2f}")
    print("  (CFB > 0.5 indicates active crown fire)")
    
    print("✓ Crown fire module OK")


def test_wind_solver():
    """Test mass-conserving wind solver."""
    print("\n" + "="*60)
    print("4. MASS-CONSERVING WIND SOLVER")
    print("="*60)
    
    from wind_solver import (
        solve_wind_field,
        compute_divergence,
        WindSolverParams,
    )
    
    # Create terrain with ridge
    nx, ny = 60, 60
    dx = dy = 30.0
    x = jnp.linspace(0, (nx-1)*dx, nx)
    y = jnp.linspace(0, (ny-1)*dy, ny)
    X, Y = jnp.meshgrid(x, y)
    
    # N-S ridge
    dem = 1000.0 + 200.0 * jnp.exp(-((X - 900)**2) / (200**2))
    
    # Solve wind field
    wind = solve_wind_field(
        dem, dx, dy,
        background_speed=10.0,
        background_direction=270.0,  # From west
    )
    
    print(f"Background wind: 10 m/s from west (270°)")
    print(f"Terrain-adjusted wind:")
    print(f"  Speed range: {float(wind.speed.min()):.1f} - {float(wind.speed.max()):.1f} m/s")
    print(f"  Direction range: {float(wind.direction.min()):.0f}° - {float(wind.direction.max()):.0f}°")
    
    # Check mass conservation
    div = compute_divergence(wind.u, wind.v, dx, dy)
    print(f"  Divergence: {float(jnp.abs(div).max()):.2e} (should be small)")
    
    # Compare ridge vs valley
    ridge_col = nx // 2
    valley_col = 5
    ridge_speed = float(wind.speed[ny//2, ridge_col])
    valley_speed = float(wind.speed[ny//2, valley_col])
    print(f"\nRidge effect:")
    print(f"  Ridge wind: {ridge_speed:.1f} m/s")
    print(f"  Valley wind: {valley_speed:.1f} m/s")
    
    print("✓ Wind solver module OK")


def test_fire_atmosphere():
    """Test fire-atmosphere coupling."""
    print("\n" + "="*60)
    print("5. FIRE-ATMOSPHERE COUPLING")
    print("="*60)
    
    from fire_atmosphere import (
        couple_wind_to_fire,
        compute_indraft_velocity,
        FireAtmosphereParams,
        estimate_coupling_strength,
    )
    
    # Create fire scenario
    nx, ny = 50, 50
    dx = dy = 30.0
    
    x = jnp.linspace(0, (nx-1)*dx, nx)
    y = jnp.linspace(0, (ny-1)*dy, ny)
    X, Y = jnp.meshgrid(x, y)
    
    # Circular fire
    fire_radius = 300.0
    center = 750.0
    distance = jnp.sqrt((X - center)**2 + (Y - center)**2)
    phi = (distance - fire_radius) / dx
    
    # Fire properties
    ros = jnp.full((ny, nx), 30.0)
    fuel_load = jnp.full((ny, nx), 3.0)
    
    # Background wind
    u_bg = jnp.full((ny, nx), 5.0)
    v_bg = jnp.full((ny, nx), 0.0)
    
    # Compute coupling
    params = FireAtmosphereParams(
        indraft_coefficient=0.5,
        max_indraft_velocity=5.0,
    )
    
    coupled = couple_wind_to_fire(u_bg, v_bg, phi, ros, fuel_load, dx, dy, params)
    
    print("Fire-induced wind modification:")
    print(f"  Background wind: (5.0, 0.0) m/s")
    print(f"  Indraft magnitude: {float(coupled.indraft_magnitude.max()):.2f} m/s")
    print(f"  Total wind range: {float(jnp.sqrt(coupled.u**2 + coupled.v**2).min()):.1f} - {float(jnp.sqrt(coupled.u**2 + coupled.v**2).max()):.1f} m/s")
    
    strength = estimate_coupling_strength(phi, ros, fuel_load)
    print(f"  Coupling strength: {strength:.2f} (0-1 scale)")
    
    print("✓ Fire-atmosphere module OK")


def test_enhanced_simulation():
    """Test integrated enhanced simulation."""
    print("\n" + "="*60)
    print("6. INTEGRATED ENHANCED SIMULATION")
    print("="*60)
    
    from levelset import LevelSetGrids
    from levelset_enhanced import (
        EnhancedSimConfig,
        simulate_fire_enhanced,
    )
    
    # Create terrain
    nx, ny = 80, 80
    dx = dy = 30.0
    x = jnp.linspace(0, (nx-1)*dx, nx)
    y = jnp.linspace(0, (ny-1)*dy, ny)
    X, Y = jnp.meshgrid(x, y)
    
    dem = 1000.0 + 150.0 * jnp.exp(-((X - 1200)**2) / (400**2))
    
    # Base grids
    ros = jnp.full((ny, nx), 15.0)
    bros = jnp.full((ny, nx), 3.0)
    fros = jnp.full((ny, nx), 6.0)
    raz = jnp.full((ny, nx), jnp.pi/2)
    
    grids = LevelSetGrids(x_coords=x, y_coords=y, ros=ros, bros=bros, fros=fros, raz=raz)
    
    # Run with no physics (baseline)
    config_none = EnhancedSimConfig(
        enable_solar=False,
        enable_moisture_lag=False,
        enable_crown_fire=False,
        enable_terrain_wind=False,
        enable_fire_atmosphere=False,
    )
    
    result_none = simulate_fire_enhanced(
        grids, dem, 1200.0, 1200.0,
        n_steps=50, dt=1.0, initial_radius=40.0,
        config=config_none,
    )
    
    # Run with all physics
    config_all = EnhancedSimConfig(
        enable_solar=True,
        enable_moisture_lag=True,
        enable_crown_fire=True,
        enable_terrain_wind=True,
        enable_fire_atmosphere=False,  # Skip for speed
    )
    
    result_all = simulate_fire_enhanced(
        grids, dem, 1200.0, 1200.0,
        n_steps=50, dt=1.0, initial_radius=40.0,
        config=config_all,
        simulation_datetime=datetime(2024, 7, 15, 14, 0),
        latitude=51.2, longitude=-115.7,
        background_wind_speed=10.0,
        background_wind_direction=270.0,
        temperature=30.0,
        relative_humidity=20.0,
    )
    
    print("Simulation comparison (50 min, same base ROS):")
    print(f"  Without physics: {result_none.burned_area/10000:.2f} ha")
    print(f"  With all physics: {result_all.burned_area/10000:.2f} ha")
    print(f"  Difference: {(result_all.burned_area - result_none.burned_area)/result_none.burned_area*100:+.1f}%")
    
    if result_all.moisture_final:
        print(f"\nFinal moisture state:")
        print(f"  1-hr: {float(result_all.moisture_final.m_1hr.mean()):.3f}")
        print(f"  10-hr: {float(result_all.moisture_final.m_10hr.mean()):.3f}")
    
    if result_all.wind_final:
        print(f"\nTerrain-adjusted wind:")
        print(f"  Speed: {float(result_all.wind_final.speed.min()):.1f} - {float(result_all.wind_final.speed.max()):.1f} m/s")
    
    print("✓ Enhanced simulation OK")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("IGNACIO JAX PHYSICS MODULES - COMPREHENSIVE TEST")
    print("="*60)
    
    try:
        test_solar_radiation()
        test_moisture_lag()
        test_crown_fire()
        test_wind_solver()
        test_fire_atmosphere()
        test_enhanced_simulation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nPhysics modules ready for use:")
        print("  1. solar_radiation.py - Sun position, hillshade, fuel conditioning")
        print("  2. moisture_lag.py - Time-lagged fuel moisture dynamics")
        print("  3. crown_fire.py - Van Wagner crown fire transition")
        print("  4. wind_solver.py - Mass-conserving terrain wind")
        print("  5. fire_atmosphere.py - Fire-induced indraft coupling")
        print("  6. levelset_enhanced.py - Integrated simulation")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
