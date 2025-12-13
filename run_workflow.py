#!/usr/bin/env python3
"""
Ignacio Fire Spread Modeling - Complete Workflow Example

This script demonstrates the full workflow:
1. Parse station weather data (Alberta ACIS format)
2. Optionally fetch cloud data (DEM, land cover)
3. Create fuel raster from land cover
4. Set up simulation with coordinate-based ignition
5. Run fire spread simulation

Usage:
    # Basic run with local data
    python run_workflow.py --station-csv data/Banff_CS.csv
    
    # With cloud data acquisition
    python run_workflow.py --fetch-cloud --bbox -116.0 51.0 -115.5 51.5
    
    # Using config file
    python run_workflow.py --config ignacio_example.yaml
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_preprocessing(
    station_csv: Path = None,
    bbox: list = None,
    output_dir: Path = None,
    fetch_cloud: bool = False,
    ignition_coords: list = None,
):
    """Run preprocessing pipeline."""
    from ignacio.station_parser import load_station_csv, summarize_fire_weather
    from ignacio.preprocessing import (
        fetch_cloud_data,
        create_fuel_raster_from_landcover,
        create_ignition_shapefile,
        generate_config_from_data,
    )
    
    if output_dir is None:
        output_dir = Path("./data/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"files": {}}
    
    # 1. Parse station weather
    if station_csv and station_csv.exists():
        logger.info(f"Parsing station data: {station_csv}")
        df = load_station_csv(station_csv)
        summary = summarize_fire_weather(df)
        
        logger.info(f"  Records: {summary['n_records']}")
        logger.info(f"  Date range: {summary['date_range']}")
        if 'ffmc_mean' in summary:
            logger.info(f"  Mean FFMC: {summary['ffmc_mean']:.1f}")
        if 'isi_mean' in summary:
            logger.info(f"  Mean ISI: {summary['isi_mean']:.1f}")
        
        # Save converted format
        from ignacio.station_parser import convert_to_ignacio_format
        weather_path = output_dir / "weather_hourly.csv"
        convert_to_ignacio_format(df, weather_path)
        results["files"]["weather"] = weather_path
        results["weather_summary"] = summary
    
    # 2. Fetch cloud data
    if fetch_cloud and bbox:
        logger.info("Fetching cloud data...")
        try:
            cloud_files = fetch_cloud_data(
                bbox, output_dir,
                datasets=["dem", "landcover"],
            )
            results["files"].update(cloud_files)
        except Exception as e:
            logger.error(f"Cloud fetch failed: {e}")
    
    # 3. Create fuel raster from land cover
    landcover_path = results["files"].get("landcover")
    if landcover_path and Path(landcover_path).exists():
        logger.info("Creating fuel raster from land cover...")
        fuel_path = output_dir / "fuel_fbp.tif"
        try:
            create_fuel_raster_from_landcover(landcover_path, fuel_path)
            results["files"]["fuel"] = fuel_path
        except Exception as e:
            logger.error(f"Fuel raster creation failed: {e}")
    
    # 4. Create ignition shapefile
    if ignition_coords:
        logger.info(f"Creating ignition points: {len(ignition_coords)} locations")
        ignition_path = output_dir / "ignition_points.shp"
        try:
            create_ignition_shapefile(ignition_coords, ignition_path)
            results["files"]["ignition"] = ignition_path
        except Exception as e:
            logger.error(f"Ignition shapefile creation failed: {e}")
    
    # 5. Generate config
    if results["files"]:
        config_path = output_dir / "ignacio_generated.yaml"
        generate_config_from_data(
            output_dir, config_path,
            project_name="Generated Fire Simulation",
            ignition_coords=ignition_coords,
        )
        results["config"] = config_path
    
    return results


def run_simulation_from_config(config_path: Path):
    """Run fire simulation from config file."""
    try:
        from ignacio.config import load_config
        from ignacio.simulation import run_simulation
        
        logger.info(f"Loading config: {config_path}")
        config = load_config(config_path)
        
        logger.info(f"Running simulation: {config.project.name}")
        result = run_simulation(config)
        
        return result
    except ImportError as e:
        logger.error(f"Missing dependencies for simulation: {e}")
        logger.info("Install with: pip install rasterio geopandas shapely")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Ignacio Fire Spread Modeling Workflow"
    )
    
    # Data sources
    parser.add_argument(
        "--station-csv", "-s",
        type=Path,
        help="Station weather CSV (Alberta ACIS format)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Existing config file",
    )
    
    # Cloud data
    parser.add_argument(
        "--fetch-cloud",
        action="store_true",
        help="Fetch data from cloud sources",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box for cloud data (WGS84)",
    )
    
    # Ignition
    parser.add_argument(
        "--ignition",
        type=float,
        nargs=2,
        action="append",
        metavar=("LON", "LAT"),
        help="Ignition coordinate (can specify multiple)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./data/preprocessed"),
        help="Output directory",
    )
    
    # Actions
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run preprocessing, skip simulation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("IGNACIO FIRE SPREAD MODELING WORKFLOW")
    logger.info("="*60)
    
    # Run preprocessing if needed
    if args.station_csv or args.fetch_cloud or args.ignition:
        logger.info("\n--- PREPROCESSING ---")
        
        ignition_coords = None
        if args.ignition:
            ignition_coords = [(lon, lat) for lon, lat in args.ignition]
        
        results = run_preprocessing(
            station_csv=args.station_csv,
            bbox=args.bbox,
            output_dir=args.output_dir,
            fetch_cloud=args.fetch_cloud,
            ignition_coords=ignition_coords,
        )
        
        logger.info("\nPreprocessing complete!")
        logger.info(f"Output: {args.output_dir}")
        for name, path in results.get("files", {}).items():
            logger.info(f"  {name}: {path}")
        
        # Use generated config
        if not args.config and "config" in results:
            args.config = results["config"]
    
    # Run simulation
    if not args.preprocess_only and args.config:
        logger.info("\n--- SIMULATION ---")
        result = run_simulation_from_config(args.config)
        
        if result:
            logger.info("\nSimulation complete!")
    
    logger.info("\n" + "="*60)
    logger.info("WORKFLOW COMPLETE")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
