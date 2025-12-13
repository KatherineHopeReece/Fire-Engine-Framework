"""
Command-line interface for Ignacio.

This module provides the CLI entry points for running simulations
and other utilities.

Commands:
- ignacio ignite (or run): Run fire growth simulation
- ignacio init: Generate configuration template
- ignacio preprocess: Preprocess input data
- ignacio calibrate: Run parameter calibration
- ignacio compare: Compare simulation configurations
- ignacio info: Display system information
- ignacio validate: Validate configuration

Authors: Katherine Hope Reece and Darri Eythorsson
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
import click

logger = logging.getLogger(__name__)


# =============================================================================
# Main CLI Group
# =============================================================================

@click.group()
@click.version_option(package_name="ignacio")
def main():
    """
    🔥 IGNACIO: Open-Source Fire Growth Simulation System
    
    An implementation of the Canadian FBP System for wildland fire
    growth modeling with JAX-accelerated differentiable physics.
    
    \b
    Quick Start:
        ignacio init                    # Create config template
        ignacio ignite config.yaml      # Run simulation
        ignacio calibrate config.yaml   # Calibrate parameters
    
    \b
    Documentation: https://github.com/your-repo/ignacio
    """
    pass


# =============================================================================
# Ignite Command (Main Simulation)
# =============================================================================

def run_simulation_command(
    config_path: Path,
    output: Optional[Path],
    preset: Optional[str],
    duration: Optional[float],
    verbose: bool,
    quiet: bool,
    use_jax: bool,
    create_gif: bool,
):
    """Core simulation logic shared by run and ignite commands."""
    
    # Set up logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    
    # Banner
    if not quiet:
        click.echo("=" * 60)
        click.echo("🔥 IGNACIO: Fire Growth Simulation System")
        click.echo("=" * 60)
    
    try:
        # Try new unified config first, fall back to old config
        try:
            from ignacio.config_unified import load_config, create_config_from_preset
            
            if preset:
                config = create_config_from_preset(preset)
                click.echo(f"Using preset: {preset}")
            else:
                config = load_config(config_path)
            
            unified_config = True
        except ImportError:
            from ignacio.config import load_config, validate_paths
            config = load_config(config_path)
            unified_config = False
        
        # Apply CLI overrides
        if output is not None:
            if unified_config:
                config.outputs.output_dir = str(output)
            else:
                config.project.output_dir = output
            click.echo(f"Output directory: {output}")
        
        if duration is not None:
            if unified_config:
                config.initial_conditions.duration_hours = duration
            click.echo(f"Duration: {duration} hours")
        
        if create_gif and unified_config:
            config.outputs.create_animation = True
            config.outputs.animation_format = 'gif'
        
        # Print config summary
        if not quiet and unified_config:
            click.echo(f"\nConfiguration: {config.name}")
            click.echo(f"Spread method: {config.decisions.get('spread_method')}")
            click.echo(f"ROS model: {config.decisions.get('ros_model')}")
        
        # Run simulation
        click.echo("\nStarting simulation...")
        
        if use_jax and unified_config:
            # Use JAX-accelerated simulation
            from ignacio.jax_core import quick_full_simulation
            import jax.numpy as jnp
            
            # Get ignition point
            if config.initial_conditions.ignition_points:
                ign = config.initial_conditions.ignition_points[0]
                x_ign, y_ign = ign.x, ign.y
            else:
                x_ign = config.domain.nx * config.domain.resolution / 2
                y_ign = config.domain.ny * config.domain.resolution / 2
            
            # Run
            results = quick_full_simulation(
                nx=config.domain.nx,
                ny=config.domain.ny,
                dx=config.domain.resolution,
                duration_minutes=config.initial_conditions.duration_hours * 60,
                x_ign=x_ign,
                y_ign=y_ign,
                wind_speed=config.initial_conditions.initial_wind_speed * 1000 / 60,  # km/h to m/min
                wind_dir=config.initial_conditions.initial_wind_direction,
            )
            
            burned_area_ha = float(jnp.sum(results.phi_final < 0)) * config.domain.resolution**2 / 10000
            
            if not quiet:
                click.echo("\n" + "=" * 60)
                click.echo("SIMULATION COMPLETE")
                click.echo("=" * 60)
                click.echo(f"Burned area: {burned_area_ha:.2f} ha")
                click.echo(f"Time steps: {results.phi_history.shape[0] if results.phi_history is not None else 'N/A'}")
                click.echo("=" * 60)
            
            # Create animation if requested
            if create_gif and results.phi_history is not None:
                _create_animation(results, config, output or Path('./output'))
        else:
            # Use original simulation
            from ignacio.simulation import run_simulation
            results = run_simulation(config)
            
            if not quiet:
                click.echo("\n" + "=" * 60)
                click.echo("SIMULATION COMPLETE")
                click.echo("=" * 60)
                click.echo(f"Fires simulated: {results.n_fires}")
                click.echo(f"Total area burned: {results.total_area_ha:.2f} ha")
                click.echo("=" * 60)
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Simulation failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--preset", "-p", type=str, help="Use preset configuration (fast, operational, research, coupled)")
@click.option("--duration", "-d", type=float, help="Simulation duration in hours")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
@click.option("--jax/--no-jax", default=True, help="Use JAX-accelerated simulation")
@click.option("--gif", is_flag=True, help="Create animation GIF")
def ignite(config_path, output, preset, duration, verbose, quiet, jax, gif):
    """
    🔥 Ignite a fire simulation.
    
    This is the main command to run fire growth simulations.
    
    \b
    Examples:
        ignacio ignite config.yaml
        ignacio ignite config.yaml --preset operational --gif
        ignacio ignite config.yaml -o ./results --duration 12 -v
    """
    run_simulation_command(config_path, output, preset, duration, verbose, quiet, jax, gif)


# Also support 'run' as an alias for backwards compatibility
@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--preset", "-p", type=str, help="Use preset configuration")
@click.option("--duration", "-d", type=float, help="Simulation duration in hours")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
@click.option("--jax/--no-jax", default=True, help="Use JAX-accelerated simulation")
@click.option("--gif", is_flag=True, help="Create animation GIF")
def run(config_path, output, preset, duration, verbose, quiet, jax, gif):
    """Run fire simulation (alias for 'ignite')."""
    run_simulation_command(config_path, output, preset, duration, verbose, quiet, jax, gif)


# =============================================================================
# Init Command
# =============================================================================

@main.command()
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("ignacio_config"), 
              help="Output path for configuration")
@click.option("--preset", "-p", type=str, default="operational",
              help="Preset to use (fast, operational, research, coupled, prometheus_compatible)")
@click.option("--single-file", is_flag=True, help="Create single YAML instead of directory")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def init(output: Path, preset: str, single_file: bool, force: bool):
    """
    Generate configuration template.
    
    Creates a new configuration with default settings and documentation.
    
    \b
    Available presets:
        fast                 - Minimal physics, fastest simulation
        operational          - Standard operational settings
        prometheus_compatible - Compatible with Prometheus fire model
        research             - Full physics for research use
        coupled              - Fire-atmosphere coupling enabled
    """
    from ignacio.config_unified import create_config_from_preset, export_config_template
    from ignacio.parameters import export_parameter_template
    from ignacio.model_decisions import export_decision_template
    
    output = Path(output)
    
    if single_file:
        output_file = output if output.suffix == '.yaml' else output.with_suffix('.yaml')
        if output_file.exists() and not force:
            click.echo(f"File exists: {output_file}. Use --force to overwrite.", err=True)
            sys.exit(1)
        
        config = create_config_from_preset(preset)
        config.to_yaml(output_file)
        click.echo(f"Created configuration: {output_file}")
    else:
        if output.exists() and not force:
            click.echo(f"Directory exists: {output}. Use --force to overwrite.", err=True)
            sys.exit(1)
        
        config = create_config_from_preset(preset)
        config.to_yaml(output, split_files=True)
        
        # Also create parameter and decision templates with documentation
        export_parameter_template(output / 'parameters_reference.yaml')
        export_decision_template(output / 'model_decisions_reference.yaml')
        
        click.echo(f"Created configuration directory: {output}")
        click.echo(f"  ├── config.yaml")
        click.echo(f"  ├── parameters.yaml")
        click.echo(f"  ├── model_decisions.yaml")
        click.echo(f"  ├── initial_conditions.yaml")
        click.echo(f"  ├── calibration.yaml")
        click.echo(f"  ├── parameters_reference.yaml (documentation)")
        click.echo(f"  └── model_decisions_reference.yaml (documentation)")


# =============================================================================
# Preprocess Command
# =============================================================================

@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--dem", type=click.Path(exists=True), help="DEM file to process")
@click.option("--fuel", type=click.Path(exists=True), help="Fuel map to process")
@click.option("--weather", type=click.Path(exists=True), help="Weather file to process")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def preprocess(config_path, dem, fuel, weather, output, verbose):
    """
    Preprocess input data for simulation.
    
    Generates slope, aspect, and validates fuel maps.
    """
    from ignacio.preprocessing import preprocess_domain
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    click.echo("Preprocessing input data...")
    
    try:
        from ignacio.config_unified import load_config
        config = load_config(config_path)
        
        output_dir = Path(output) if output else Path(config.outputs.output_dir) / 'preprocessed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Override paths if provided
        if dem:
            config.inputs.dem = dem
        if fuel:
            config.inputs.fuel_map = fuel
        if weather:
            config.inputs.weather_file = weather
        
        # Run preprocessing
        result = preprocess_domain(
            dem_path=config.inputs.dem,
            fuel_path=config.inputs.fuel_map,
            output_dir=output_dir,
        )
        
        click.echo(f"Preprocessing complete. Output in: {output_dir}")
        
    except Exception as e:
        logger.exception("Preprocessing failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# =============================================================================
# Calibrate Command
# =============================================================================

@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--observed", type=click.Path(exists=True), help="Observed fire perimeter")
@click.option("--iterations", "-n", type=int, default=100, help="Max calibration iterations")
@click.option("--algorithm", "-a", type=click.Choice(['adam', 'lbfgs', 'evolutionary']), 
              default='adam', help="Optimization algorithm")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output for calibrated parameters")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def calibrate(config_path, observed, iterations, algorithm, output, verbose):
    """
    Calibrate model parameters against observations.
    
    Uses differentiable simulation to optimize parameters.
    """
    from ignacio.config_unified import load_config
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    click.echo("=" * 60)
    click.echo("🎯 IGNACIO Parameter Calibration")
    click.echo("=" * 60)
    
    try:
        config = load_config(config_path)
        
        # Override calibration settings
        if observed:
            config.calibration.observed_perimeters = observed
        config.calibration.max_iterations = iterations
        config.calibration.algorithm = algorithm
        
        click.echo(f"Parameters to calibrate: {', '.join(config.calibration.parameters)}")
        click.echo(f"Algorithm: {algorithm}")
        click.echo(f"Max iterations: {iterations}")
        
        # Get bounds
        bounds = config.get_calibration_bounds()
        defaults = config.get_calibration_defaults()
        
        click.echo(f"\nInitial values: {defaults}")
        click.echo(f"Bounds: {bounds}")
        
        # Run calibration
        from ignacio.jax_core.calibration import gradient_calibration
        
        click.echo("\nStarting calibration...")
        
        # This would need actual implementation with observed data
        click.echo("⚠️  Calibration requires observed perimeter data.")
        click.echo("   Use --observed to specify observed fire perimeter shapefile.")
        
    except Exception as e:
        logger.exception("Calibration failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# =============================================================================
# Compare Command
# =============================================================================

@main.command()
@click.argument("configs", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("./comparison"),
              help="Output directory for comparison results")
@click.option("--duration", "-d", type=float, default=2.0, help="Simulation duration (hours)")
@click.option("--gif", is_flag=True, help="Create comparison GIF")
def compare(configs, output, duration, gif):
    """
    Compare multiple simulation configurations.
    
    Runs simulations with different configs and compares results.
    
    \b
    Examples:
        ignacio compare config1.yaml config2.yaml config3.yaml
        ignacio compare config*.yaml --gif
    """
    import numpy as np
    
    if not configs:
        click.echo("Please provide at least one configuration file.", err=True)
        sys.exit(1)
    
    click.echo("=" * 60)
    click.echo("📊 IGNACIO Configuration Comparison")
    click.echo("=" * 60)
    
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for config_path in configs:
        click.echo(f"\nRunning: {config_path}")
        
        try:
            from ignacio.config_unified import load_config
            from ignacio.jax_core import quick_full_simulation
            import jax.numpy as jnp
            
            config = load_config(config_path)
            config.initial_conditions.duration_hours = duration
            
            # Get ignition
            if config.initial_conditions.ignition_points:
                ign = config.initial_conditions.ignition_points[0]
                x_ign, y_ign = ign.x, ign.y
            else:
                x_ign = config.domain.nx * config.domain.resolution / 2
                y_ign = config.domain.ny * config.domain.resolution / 2
            
            result = quick_full_simulation(
                nx=config.domain.nx,
                ny=config.domain.ny,
                dx=config.domain.resolution,
                duration_minutes=duration * 60,
                x_ign=x_ign,
                y_ign=y_ign,
            )
            
            burned_area = float(jnp.sum(result.phi_final < 0)) * config.domain.resolution**2 / 10000
            
            results.append({
                'config': str(config_path),
                'name': config.name,
                'preset': config.preset,
                'spread_method': config.decisions.get('spread_method'),
                'burned_area_ha': burned_area,
                'phi_final': np.array(result.phi_final),
            })
            
            click.echo(f"  Burned area: {burned_area:.2f} ha")
            
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
    
    # Summary table
    click.echo("\n" + "=" * 60)
    click.echo("COMPARISON SUMMARY")
    click.echo("=" * 60)
    click.echo(f"{'Config':<30} {'Preset':<15} {'Area (ha)':<10}")
    click.echo("-" * 60)
    for r in results:
        click.echo(f"{Path(r['config']).name:<30} {r['preset'] or 'custom':<15} {r['burned_area_ha']:<10.2f}")
    
    # Create comparison plot
    if len(results) > 0:
        _create_comparison_plot(results, output)
        click.echo(f"\nComparison plot saved to: {output / 'comparison.png'}")


# =============================================================================
# Validate Command
# =============================================================================

@main.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def validate(config_path):
    """
    Validate configuration file.
    
    Checks for errors and warnings in configuration.
    """
    click.echo(f"Validating: {config_path}")
    
    try:
        from ignacio.config_unified import load_config
        
        config = load_config(config_path)
        
        click.echo("\n✓ Configuration loaded successfully")
        click.echo(f"\nName: {config.name}")
        click.echo(f"Preset: {config.preset or 'custom'}")
        
        # Check model decisions
        click.echo("\nModel Decisions:")
        for name, choice in config.decisions.choices.items():
            click.echo(f"  {name}: {choice}")
        
        # Check required inputs
        click.echo("\nInput Files:")
        issues = []
        
        if config.inputs.dem:
            if Path(config.inputs.dem).exists():
                click.echo(f"  ✓ DEM: {config.inputs.dem}")
            else:
                click.echo(f"  ✗ DEM: {config.inputs.dem} (not found)")
                issues.append("DEM file not found")
        else:
            click.echo("  ⚠ DEM: not specified")
        
        if config.inputs.fuel_map:
            if Path(config.inputs.fuel_map).exists():
                click.echo(f"  ✓ Fuel map: {config.inputs.fuel_map}")
            else:
                click.echo(f"  ✗ Fuel map: {config.inputs.fuel_map} (not found)")
                issues.append("Fuel map not found")
        else:
            click.echo("  ⚠ Fuel map: not specified")
        
        # Summary
        if issues:
            click.echo(f"\n⚠ Configuration has {len(issues)} issue(s):")
            for issue in issues:
                click.echo(f"  - {issue}")
        else:
            click.echo("\n✓ Configuration is valid")
        
    except Exception as e:
        click.echo(f"\n✗ Validation failed: {e}", err=True)
        sys.exit(1)


# =============================================================================
# Info Command
# =============================================================================

@main.command()
@click.option("--presets", is_flag=True, help="Show available presets")
@click.option("--parameters", is_flag=True, help="Show all parameters")
@click.option("--decisions", is_flag=True, help="Show all model decisions")
def info(presets, parameters, decisions):
    """Display system and configuration information."""
    import platform
    import numpy
    from ignacio import __version__
    
    click.echo("=" * 60)
    click.echo("🔥 IGNACIO Fire Growth Simulation System")
    click.echo("=" * 60)
    click.echo(f"Version: {__version__}")
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"NumPy: {numpy.__version__}")
    
    try:
        import jax
        click.echo(f"JAX: {jax.__version__}")
        click.echo(f"JAX devices: {jax.devices()}")
    except ImportError:
        click.echo("JAX: not installed")
    
    try:
        import shapely
        click.echo(f"Shapely: {shapely.__version__}")
    except ImportError:
        click.echo("Shapely: not installed (vector topology disabled)")
    
    if presets:
        from ignacio.model_decisions import PRESETS
        click.echo("\n" + "=" * 60)
        click.echo("Available Presets:")
        click.echo("=" * 60)
        for name, preset in PRESETS.items():
            click.echo(f"\n{name}:")
            for k, v in list(preset.choices.items())[:5]:
                click.echo(f"  {k}: {v}")
            if len(preset.choices) > 5:
                click.echo(f"  ... and {len(preset.choices) - 5} more")
    
    if parameters:
        from ignacio.parameters import CALIBRATABLE_PARAMETERS
        click.echo("\n" + "=" * 60)
        click.echo("Calibratable Parameters:")
        click.echo("=" * 60)
        for name, param in CALIBRATABLE_PARAMETERS.items():
            click.echo(f"  {name}: {param.default} ({param.units})")
            click.echo(f"    Range: [{param.min_val}, {param.max_val}]")
    
    if decisions:
        from ignacio.model_decisions import ALL_DECISIONS
        click.echo("\n" + "=" * 60)
        click.echo("Model Decisions:")
        click.echo("=" * 60)
        for name, decision in ALL_DECISIONS.items():
            click.echo(f"\n{name} (default: {decision.default}):")
            click.echo(f"  {decision.description}")
            for opt_name, opt in decision.options.items():
                marker = "→" if opt_name == decision.default else " "
                click.echo(f"  {marker} {opt_name}: {opt.description}")


# =============================================================================
# List Presets Command
# =============================================================================

@main.command('list-presets')
def list_presets():
    """List available configuration presets."""
    from ignacio.model_decisions import PRESETS
    
    click.echo("Available presets:\n")
    for name, preset in PRESETS.items():
        click.echo(f"  {name}")
        click.echo(f"    spread_method: {preset.choices.get('spread_method')}")
        click.echo(f"    atmosphere_coupling: {preset.choices.get('atmosphere_coupling')}")
        click.echo()


# =============================================================================
# Terrain Processing Command
# =============================================================================

@main.command()
@click.argument("dem_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True,
              help="Output directory for slope and aspect grids")
def terrain(dem_path: Path, output: Path):
    """
    Process DEM to generate slope and aspect grids.
    
    Useful for pre-processing terrain data before running simulations.
    """
    from ignacio.io import read_raster, write_raster
    from ignacio.terrain import compute_slope_aspect
    
    click.echo(f"Processing terrain from: {dem_path}")
    
    # Create output dir
    output.mkdir(parents=True, exist_ok=True)
    
    # Read DEM
    dem = read_raster(dem_path)
    dx = abs(dem.transform.a)
    dy = abs(dem.transform.e)
    
    # Compute
    slope, aspect = compute_slope_aspect(dem.data, dx, dy)
    
    # Save
    slope_path = output / "slope_deg.tif"
    aspect_path = output / "aspect_deg.tif"
    
    write_raster(slope_path, slope, dem.transform, dem.crs)
    write_raster(aspect_path, aspect, dem.transform, dem.crs)
    
    click.echo(f"Saved slope to: {slope_path}")
    click.echo(f"Saved aspect to: {aspect_path}")


# =============================================================================
# Helper Functions
# =============================================================================

def _create_animation(results, config, output_path):
    """Create animation GIF from simulation results."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import numpy as np
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        phi_history = np.array(results.phi_history)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(frame):
            ax.clear()
            ax.contourf(phi_history[frame] < 0, cmap='YlOrRd', levels=[0.5, 1])
            ax.contour(phi_history[frame], levels=[0], colors='red', linewidths=2)
            ax.set_title(f'Fire Spread - Frame {frame}')
            ax.set_aspect('equal')
        
        anim = FuncAnimation(fig, update, frames=len(phi_history), interval=100)
        
        gif_path = output_path / 'fire_spread.gif'
        anim.save(str(gif_path), writer=PillowWriter(fps=10))
        plt.close()
        
        click.echo(f"Animation saved to: {gif_path}")
        
    except Exception as e:
        click.echo(f"Could not create animation: {e}")


def _create_comparison_plot(results, output_path):
    """Create comparison plot of multiple simulations."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        n = len(results)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (result, ax) in enumerate(zip(results, axes)):
            phi = result['phi_final']
            ax.contourf(phi < 0, cmap='YlOrRd', levels=[0.5, 1])
            ax.contour(phi, levels=[0], colors='red', linewidths=1)
            ax.set_title(f"{Path(result['config']).stem}\n{result['burned_area_ha']:.1f} ha")
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Hide unused axes
        for idx in range(n, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        click.echo(f"Could not create comparison plot: {e}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
