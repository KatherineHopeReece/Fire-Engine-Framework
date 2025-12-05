#!/usr/bin/env python3
"""
Ignacio: Open-Source Fire Growth Simulation System
===================================================

Main entry point for running fire growth simulations.

This script provides a simple interface for running simulations
without using the CLI. For full CLI functionality, use:

    ignacio run ignacio.yaml

Or install the package and use:

    python -m ignacio.cli run ignacio.yaml

Usage
-----
    python run_ignacio.py [config_path]

Arguments
---------
config_path : str, optional
    Path to configuration file. Default: ignacio.yaml

Examples
--------
    python run_ignacio.py
    python run_ignacio.py my_config.yaml
    python run_ignacio.py examples/Bow-at-Banff/config.yaml

Authors: Katherine Hope Reece and Darri Eythorsson
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def main(config_path: str | Path = "ignacio.yaml") -> int:
    """
    Run Ignacio fire growth simulation.
    
    Parameters
    ----------
    config_path : str or Path
        Path to configuration YAML file.
        
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ignacio")
    
    # Banner
    print("=" * 60)
    print("IGNACIO: Open-Source Fire Growth Simulation System")
    print("Based on the Canadian FBP System")
    print("=" * 60)
    print()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        print(f"\nError: Configuration file not found: {config_path}")
        print("\nTo create a template configuration, run:")
        print("    ignacio init")
        return 1
    
    try:
        # Import here to catch import errors
        from ignacio.config import load_config, validate_paths
        from ignacio.simulation import run_simulation
        
        # Load and validate configuration
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
        
        # Validate input files exist
        warnings = validate_paths(config)
        for warning in warnings:
            logger.warning(warning)
        
        # Run simulation
        logger.info(f"Starting simulation: {config.project.name}")
        results = run_simulation(config)
        
        # Summary
        print()
        print("=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Project: {config.project.name}")
        print(f"Fires simulated: {results.n_fires}")
        print(f"Total area burned: {results.total_area_ha:.2f} ha")
        print(f"Output directory: {config.project.output_dir}")
        print("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print("\nTo install dependencies, run:")
        print("    pip install -e .")
        return 1
    except Exception as e:
        logger.exception("Simulation failed")
        return 1


if __name__ == "__main__":
    # Get config path from command line
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "ignacio.yaml"
    
    sys.exit(main(config_path))
