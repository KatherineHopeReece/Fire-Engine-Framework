#!/usr/bin/env python3
"""
Bow Valley at Banff Comparison Study.

This script compares multiple model configurations for the Bow Valley region,
simulating fire spread under various physics options and creating visualizations
including animated GIFs.

The 2014-03-08 fire event is used as a reference case.

Usage:
    python test_bow_banff_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys

# Add parent to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("BOW VALLEY AT BANFF - FIRE SPREAD COMPARISON STUDY")
print("=" * 70)


# =============================================================================
# Configuration Setup
# =============================================================================

def create_test_configs():
    """Create test configurations for comparison."""
    from ignacio.model_decisions import ModelDecisions, PRESETS
    from ignacio.parameters import ParameterSet
    from ignacio.config_unified import (
        IgnacioConfig, InitialConditions, IgnitionPoint,
        DomainConfig, OutputConfig, CalibrationConfig
    )
    
    # Common domain (Bow Valley - approximately)
    domain = DomainConfig(
        resolution=30.0,
        nx=150,
        ny=150,
        crs="EPSG:32611",  # UTM 11N
        nz=15,
        z_top=1500.0,
    )
    
    # Common initial conditions (2014-03-08 scenario)
    initial_conditions = InitialConditions(
        ignition_points=[
            IgnitionPoint(x=2250.0, y=2250.0, time=0.0, radius=30.0, name="main_ignition"),
        ],
        start_time=datetime(2014, 3, 8, 12, 0),
        duration_hours=2.0,  # 2 hours for comparison
        ffmc=88.0,
        dmc=30.0,
        dc=150.0,
        bui=30.0,
        moisture_1hr=6.0,
        moisture_10hr=8.0,
        moisture_100hr=10.0,
        moisture_live=90.0,
        initial_wind_speed=25.0,  # km/h
        initial_wind_direction=270.0,  # From west
        initial_temperature=15.0,
        initial_rh=25.0,
    )
    
    configs = {}
    
    # 1. Fast preset (minimal physics) - use lower base ROS
    configs['fast'] = IgnacioConfig(
        name="fast_simulation",
        description="Minimal physics for fast simulation",
        decisions=PRESETS['fast'],
        parameters=ParameterSet(values={'ros_multiplier': 0.6}),
        initial_conditions=initial_conditions,
        domain=domain,
        preset='fast',
    )
    
    # 2. Operational preset - standard ROS
    configs['operational'] = IgnacioConfig(
        name="operational_simulation",
        description="Standard operational settings",
        decisions=PRESETS['operational'],
        parameters=ParameterSet(values={'ros_multiplier': 0.8}),
        initial_conditions=initial_conditions,
        domain=domain,
        preset='operational',
    )
    
    # 3. Prometheus-compatible - slightly wider ellipse
    configs['prometheus'] = IgnacioConfig(
        name="prometheus_compatible",
        description="Prometheus-compatible settings",
        decisions=PRESETS['prometheus_compatible'],
        parameters=ParameterSet(values={'ros_multiplier': 0.7, 'length_to_breadth': 1.5}),
        initial_conditions=initial_conditions,
        domain=domain,
        preset='prometheus_compatible',
    )
    
    # 4. Research preset (full physics) - higher ROS with narrow ellipse
    configs['research'] = IgnacioConfig(
        name="research_simulation",
        description="Full physics for research",
        decisions=PRESETS['research'],
        parameters=ParameterSet(values={'ros_multiplier': 1.0, 'length_to_breadth': 3.0}),
        initial_conditions=initial_conditions,
        domain=domain,
        preset='research',
    )
    
    # 5. High wind scenario - strong wind effect
    high_wind_ic = InitialConditions(
        ignition_points=[
            IgnitionPoint(x=2250.0, y=2250.0, time=0.0, radius=30.0, name="main_ignition"),
        ],
        start_time=datetime(2014, 3, 8, 12, 0),
        duration_hours=2.0,  # Match other scenarios
        ffmc=92.0,  # Higher fire danger
        initial_wind_speed=40.0,  # Strong wind
        initial_wind_direction=270.0,
        initial_temperature=20.0,
        initial_rh=20.0,
    )
    
    configs['high_wind'] = IgnacioConfig(
        name="high_wind_scenario",
        description="High wind (40 km/h) scenario",
        decisions=PRESETS['operational'],
        parameters=ParameterSet(values={'ros_wind_factor': 1.5, 'length_to_breadth': 4.0}),
        initial_conditions=high_wind_ic,
        domain=domain,
    )
    
    # 6. Low spread scenario - reduced all factors
    configs['conservative'] = IgnacioConfig(
        name="conservative_simulation",
        description="Conservative spread estimate",
        decisions=PRESETS['operational'],
        parameters=ParameterSet(values={
            'ros_multiplier': 0.5,
            'ros_wind_factor': 0.7,
            'ros_slope_factor': 0.5,
        }),
        initial_conditions=initial_conditions,
        domain=domain,
    )
    
    return configs


# =============================================================================
# Terrain Generation (Synthetic Bow Valley)
# =============================================================================

def create_synthetic_terrain(nx, ny, dx):
    """
    Create synthetic terrain resembling Bow Valley.
    
    Features:
    - Central valley running E-W
    - Mountains on north and south
    - Ridges and gullies
    """
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)
    
    # Base valley shape (lower in center)
    valley_center = ny // 2 * dx
    valley_width = ny // 3 * dx
    valley = 200 * np.exp(-((Y - valley_center) / valley_width) ** 2)
    
    # Mountains (higher on edges)
    mountains = 800 * (np.abs(Y - valley_center) / (ny * dx / 2)) ** 2
    
    # Add ridges (sinusoidal)
    ridges = 100 * np.sin(2 * np.pi * X / (nx * dx / 4)) * np.cos(2 * np.pi * Y / (ny * dx / 3))
    
    # Combine
    elevation = 1400 + mountains - valley + ridges
    
    # Add some noise for realism
    np.random.seed(42)
    noise = 20 * np.random.randn(ny, nx)
    elevation += noise
    
    return elevation.astype(np.float32)


def compute_slope_aspect(dem, dx, dy):
    """Compute slope and aspect from DEM."""
    # Gradient in x and y
    dzdx = np.gradient(dem, dx, axis=1)
    dzdy = np.gradient(dem, dy, axis=0)
    
    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    
    # Aspect in degrees (0 = North, clockwise)
    aspect = np.degrees(np.arctan2(-dzdx, -dzdy))
    aspect = (aspect + 360) % 360
    
    return slope, aspect


# =============================================================================
# Run Simulation
# =============================================================================

def run_simulation(config, terrain, verbose=True):
    """Run fire simulation with given config and terrain."""
    import jax
    import jax.numpy as jnp
    from ignacio.jax_core import (
        initialize_phi,
        evolve_phi,
        LevelSetGrids,
    )
    
    nx = config.domain.nx
    ny = config.domain.ny
    dx = config.domain.resolution
    
    # Get ignition
    if config.initial_conditions.ignition_points:
        ign = config.initial_conditions.ignition_points[0]
        x_ign, y_ign = ign.x, ign.y
    else:
        x_ign = nx * dx / 2
        y_ign = ny * dx / 2
    
    # Create coordinates
    x_coords = jnp.arange(nx) * dx
    y_coords = jnp.arange(ny) * dx
    
    # Initialize level-set
    phi = initialize_phi(x_coords, y_coords, x_ign, y_ign, initial_radius=30.0)
    
    # Compute terrain effects
    slope, aspect = compute_slope_aspect(terrain, dx, dx)
    
    # Wind direction and speed
    wind_dir_rad = np.radians(config.initial_conditions.initial_wind_direction)
    wind_speed_kmh = config.initial_conditions.initial_wind_speed
    
    # Base ROS from FBP (simplified) - typical C-2 boreal forest
    # ISI effect on ROS: ROS = a * (1 - exp(-b * ISI))^c
    # Simplified: use wind to estimate effective spread rate
    base_ros = 5.0 * config.parameters.get('ros_multiplier')  # m/min base rate
    
    # Wind effect on ROS (simplified FBP wind function)
    # Wind increases spread rate significantly
    wind_factor = 1.0 + 0.1 * wind_speed_kmh * config.parameters.get('ros_wind_factor')
    
    # Slope effect
    slope_rad = np.radians(slope)
    slope_factor = 1.0 + 3.0 * np.sin(slope_rad) * config.parameters.get('ros_slope_factor')
    slope_factor = np.clip(slope_factor, 0.5, 5.0)
    
    # Compute ROS field
    ros = base_ros * wind_factor * slope_factor
    ros = np.clip(ros, 1.0, 50.0)  # Max 50 m/min for stability (CFL)
    ros = jnp.array(ros.astype(np.float32))
    
    if verbose:
        print(f"    ROS range: {float(jnp.min(ros)):.1f} - {float(jnp.max(ros)):.1f} m/min")
    
    # Back and flank ROS
    lb_ratio = config.parameters.get('length_to_breadth')
    bros = ros / lb_ratio
    fros = ros / np.sqrt(lb_ratio)
    
    # Spread direction (wind direction + terrain modification)
    # Simplified: just use wind direction
    raz = jnp.full((ny, nx), wind_dir_rad, dtype=jnp.float32)
    
    # Create grids
    grids = LevelSetGrids(
        x_coords=x_coords,
        y_coords=y_coords,
        ros=ros,
        bros=bros,
        fros=fros,
        raz=raz,
    )
    
    # Run simulation
    duration_min = config.initial_conditions.duration_hours * 60
    
    # CFL condition: dt * max_ros < dx
    # Use 0.5 safety factor
    max_ros = float(jnp.max(ros))
    dt = min(1.0, 0.5 * dx / max_ros)  # Adaptive time step
    n_steps = int(duration_min / dt)
    
    if verbose:
        print(f"    Time step: {dt:.2f} min (CFL-limited)")
    
    # Store history for animation
    store_interval = max(1, n_steps // 50)  # ~50 frames
    phi_history = [np.array(phi)]
    times = [0.0]
    
    if verbose:
        print(f"  Running {n_steps} steps ({duration_min:.0f} min)...")
    
    t_start = time.time()
    
    for step in range(n_steps):
        phi = evolve_phi(phi, grids, t_idx=0, dt=dt)
        
        if (step + 1) % store_interval == 0:
            phi_history.append(np.array(phi))
            times.append((step + 1) * dt)
    
    elapsed = time.time() - t_start
    
    # Final metrics
    burned_area = float(jnp.sum(phi < 0)) * dx * dx / 10000  # ha
    
    if verbose:
        print(f"  Completed in {elapsed:.2f}s ({elapsed/n_steps*1000:.1f} ms/step)")
        print(f"  Burned area: {burned_area:.2f} ha")
    
    return {
        'phi_final': np.array(phi),
        'phi_history': np.array(phi_history),
        'times': np.array(times),
        'burned_area_ha': burned_area,
        'elapsed_s': elapsed,
        'config': config,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(results, terrain, output_dir):
    """Create comparison plot of all configurations."""
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Plot terrain as background
        ax.contourf(terrain, levels=20, cmap='terrain', alpha=0.5)
        
        # Plot burned area
        burned = result['phi_final'] < 0
        ax.contourf(burned, levels=[0.5, 1], colors=['red'], alpha=0.6)
        
        # Plot fire perimeter
        ax.contour(result['phi_final'], levels=[0], colors=['darkred'], linewidths=2)
        
        # Title
        area = result['burned_area_ha']
        ax.set_title(f"{name}\n{area:.1f} ha", fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide unused
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Bow Valley Fire Spread Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_dir / 'comparison.png'}")


def create_animation(result, terrain, output_path, fps=10):
    """Create animated GIF of fire spread."""
    phi_history = result['phi_history']
    times = result['times']
    dx = 30.0  # Grid spacing
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get consistent color limits
    ny, nx = terrain.shape
    
    def update(frame):
        ax.clear()
        
        # Terrain background
        ax.contourf(terrain, levels=20, cmap='terrain', alpha=0.5)
        
        # Burned area
        burned = phi_history[frame] < 0
        if np.any(burned):
            ax.contourf(burned, levels=[0.5, 1], colors=['red'], alpha=0.6)
        
        # Fire perimeter
        ax.contour(phi_history[frame], levels=[0], colors=['darkred'], linewidths=2)
        
        # Time label
        t = times[frame]
        area = np.sum(burned) * dx**2 / 10000
        ax.set_title(f'Fire Spread - {t:.0f} min\nBurned: {area:.1f} ha')
        ax.set_aspect('equal')
        ax.axis('off')
    
    anim = FuncAnimation(fig, update, frames=len(phi_history), interval=1000/fps)
    
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close()
    
    print(f"Saved animation: {output_path}")


def plot_burned_area_timeline(results, output_dir):
    """Plot burned area over time for all configs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dx = 30.0
    
    for name, result in results.items():
        phi_history = result['phi_history']
        times = result['times']
        
        # Calculate burned area at each time
        areas = [np.sum(phi < 0) * dx**2 / 10000 for phi in phi_history]
        
        ax.plot(times, areas, label=name, linewidth=2)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Burned Area (ha)')
    ax.set_title('Fire Growth Comparison - Bow Valley')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'growth_curves.png', dpi=150)
    plt.close()
    
    print(f"Saved growth curves: {output_dir / 'growth_curves.png'}")


def create_combined_animation(results, terrain, output_path, fps=8):
    """Create side-by-side animation comparing configurations."""
    n = len(results)
    if n > 4:
        n = 4  # Limit to 4 for readability
        results = dict(list(results.items())[:4])
    
    cols = 2
    rows = (n + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    # Get max frames
    max_frames = max(len(r['phi_history']) for r in results.values())
    dx = 30.0
    
    def update(frame):
        for idx, (name, result) in enumerate(results.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            ax.clear()
            
            # Get frame (or last frame if exceeded)
            f = min(frame, len(result['phi_history']) - 1)
            phi = result['phi_history'][f]
            t = result['times'][f]
            
            # Terrain
            ax.contourf(terrain, levels=15, cmap='terrain', alpha=0.4)
            
            # Burned
            burned = phi < 0
            if np.any(burned):
                ax.contourf(burned, levels=[0.5, 1], colors=['red'], alpha=0.6)
            
            # Perimeter
            ax.contour(phi, levels=[0], colors=['darkred'], linewidths=2)
            
            area = np.sum(burned) * dx**2 / 10000
            ax.set_title(f'{name}\n{t:.0f} min | {area:.1f} ha', fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Hide unused
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
    
    anim = FuncAnimation(fig, update, frames=max_frames, interval=1000/fps)
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    plt.close()
    
    print(f"Saved combined animation: {output_path}")


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary_table(results):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Preset':<15} {'Area (ha)':<12} {'Time (s)':<10} {'ROS mult':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        config = result['config']
        preset = config.preset or 'custom'
        area = result['burned_area_ha']
        elapsed = result['elapsed_s']
        ros_mult = config.parameters.get('ros_multiplier')
        
        print(f"{name:<25} {preset:<15} {area:<12.2f} {elapsed:<10.2f} {ros_mult:<10.2f}")
    
    print("=" * 80)


# =============================================================================
# Simplified Run (without full config infrastructure)
# =============================================================================

def run_simplified_comparison():
    """Run comparison with simplified configs (no dependencies on config_unified)."""
    from dataclasses import dataclass, field
    from typing import List
    
    @dataclass
    class SimpleIgnitionPoint:
        x: float
        y: float
        time: float = 0.0
        radius: float = 30.0
        name: str = ""
    
    @dataclass 
    class SimpleParams:
        values: dict = field(default_factory=dict)
        
        def get(self, name):
            defaults = {
                'ros_multiplier': 1.0,
                'ros_wind_factor': 1.0,
                'ros_slope_factor': 1.0,
                'length_to_breadth': 2.0,
            }
            return self.values.get(name, defaults.get(name, 1.0))
    
    @dataclass
    class SimpleDomain:
        nx: int = 150
        ny: int = 150
        resolution: float = 30.0
    
    @dataclass
    class SimpleIC:
        ignition_points: List = field(default_factory=list)
        duration_hours: float = 4.0
        initial_wind_speed: float = 25.0
        initial_wind_direction: float = 270.0
        ffmc: float = 88.0
        
        def __post_init__(self):
            if not self.ignition_points:
                self.ignition_points = [SimpleIgnitionPoint(x=2250, y=2250)]
    
    @dataclass
    class SimpleConfig:
        name: str
        preset: str = 'custom'
        parameters: SimpleParams = field(default_factory=SimpleParams)
        domain: SimpleDomain = field(default_factory=SimpleDomain)
        initial_conditions: SimpleIC = field(default_factory=SimpleIC)
    
    # Create configurations
    configs = {
        'baseline': SimpleConfig(
            name='baseline',
            preset='operational',
        ),
        'high_ros': SimpleConfig(
            name='high_ros',
            preset='custom',
            parameters=SimpleParams(values={'ros_multiplier': 1.5}),
        ),
        'high_wind': SimpleConfig(
            name='high_wind',
            preset='custom',
            initial_conditions=SimpleIC(initial_wind_speed=40.0),
        ),
        'narrow_ellipse': SimpleConfig(
            name='narrow_ellipse',
            preset='custom',
            parameters=SimpleParams(values={'length_to_breadth': 4.0}),
        ),
        'wide_ellipse': SimpleConfig(
            name='wide_ellipse',
            preset='custom',
            parameters=SimpleParams(values={'length_to_breadth': 1.5}),
        ),
    }
    
    return configs


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the comparison study."""
    
    # Create output directory
    output_dir = Path('./output/bow_valley_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Try to create full configurations, fall back to simplified
    print("\nCreating test configurations...")
    try:
        configs = create_test_configs()
        print(f"  Created {len(configs)} configurations (full infrastructure)")
    except Exception as e:
        print(f"  Full config failed: {e}")
        print("  Using simplified configurations...")
        configs = run_simplified_comparison()
        print(f"  Created {len(configs)} simplified configurations")
    
    # Create synthetic terrain
    print("\nCreating synthetic Bow Valley terrain...")
    nx, ny = 150, 150
    dx = 30.0
    terrain = create_synthetic_terrain(nx, ny, dx)
    print(f"  Terrain shape: {terrain.shape}")
    print(f"  Elevation range: {terrain.min():.0f} - {terrain.max():.0f} m")
    
    # Save terrain
    plt.figure(figsize=(8, 8))
    plt.contourf(terrain, levels=20, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Synthetic Bow Valley Terrain')
    plt.axis('equal')
    plt.savefig(output_dir / 'terrain.png', dpi=150)
    plt.close()
    print(f"  Saved terrain plot: {output_dir / 'terrain.png'}")
    
    # Run simulations
    print("\nRunning simulations...")
    results = {}
    
    for name, config in configs.items():
        print(f"\n  {name}:")
        try:
            result = run_simulation(config, terrain, verbose=True)
            results[name] = result
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\nNo successful simulations. Exiting.")
        return
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Comparison plot
    plot_comparison(results, terrain, output_dir)
    
    # Growth curves
    plot_burned_area_timeline(results, output_dir)
    
    # Create individual animations
    print("\nCreating individual animations...")
    for name, result in results.items():
        gif_path = output_dir / f'animation_{name}.gif'
        try:
            create_animation(result, terrain, gif_path, fps=8)
        except Exception as e:
            print(f"  Could not create animation for {name}: {e}")
    
    # Create combined animation
    print("\nCreating combined animation...")
    try:
        create_combined_animation(results, terrain, output_dir / 'comparison.gif', fps=8)
    except Exception as e:
        print(f"  Could not create combined animation: {e}")
    
    # Summary table
    print_summary_table(results)
    
    # Final message
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.glob('*')):
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")


if __name__ == '__main__':
    main()
