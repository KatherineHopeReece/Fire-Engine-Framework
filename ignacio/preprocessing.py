"""
Preprocessing Pipeline for Ignacio Fire Spread Modeling.

This module provides a comprehensive data acquisition and preparation pipeline
that can:
1. Fetch data from cloud sources (Copernicus DEM, ESA WorldCover, etc.)
2. Parse local station weather data (Alberta ACIS format, etc.)
3. Generate fuel maps from land cover
4. Create all necessary input files for fire simulation

Usage
-----
From command line:
    python -m ignacio.preprocessing --config ignacio.yaml --fetch-cloud

From Python:
    from ignacio.preprocessing import run_preprocessing
    run_preprocessing(config_path, fetch_cloud=True)
"""

from __future__ import annotations
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Source Configuration
# =============================================================================

class DataSourceConfig:
    """Configuration for data sources."""
    
    # Cloud data sources
    CLOUD_SOURCES = {
        "dem": {
            "name": "Copernicus DEM 30m",
            "source": "AWS S3 via STAC",
            "resolution": 30,
        },
        "landcover": {
            "name": "ESA WorldCover 10m",
            "source": "AWS S3",
            "resolution": 10,
        },
        "canopy_height": {
            "name": "Meta Canopy Height",
            "source": "AWS S3",
            "resolution": 10,
        },
        "ecoregions": {
            "name": "RESOLVE Ecoregions 2017",
            "source": "Google Storage",
        },
        "burned_area": {
            "name": "Canada NBAC",
            "source": "NRCAN",
        },
    }
    
    # Land cover to FBP fuel type mapping (ESA WorldCover classes)
    LANDCOVER_TO_FUEL = {
        10: "C2",   # Tree cover -> Boreal Spruce
        20: "C2",   # Shrubland -> Boreal Spruce (conservative)
        30: "O1a",  # Grassland -> Matted Grass
        40: "M2",   # Cropland -> Boreal Mixedwood (leafless)
        50: "NF",   # Built-up -> Non-fuel
        60: "NF",   # Bare/sparse -> Non-fuel
        70: "NF",   # Snow/ice -> Non-fuel
        80: "NF",   # Water -> Non-fuel
        90: "O1a",  # Wetland -> Matted Grass
        95: "O1a",  # Mangroves -> Matted Grass
        100: "O1a", # Moss/lichen -> Matted Grass
    }
    
    # FBP fuel type codes
    FBP_FUEL_CODES = {
        "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7,
        "D1": 8, "D2": 9,
        "M1": 10, "M2": 11, "M3": 12, "M4": 13,
        "S1": 14, "S2": 15, "S3": 16,
        "O1a": 17, "O1b": 18,
        "NF": 0,  # Non-fuel
        "WA": -1,  # Water
    }


# =============================================================================
# Preprocessing Functions
# =============================================================================

def fetch_cloud_data(
    bbox: list[float],
    output_dir: Path,
    datasets: list[str] = None,
    time_range: tuple[str, str] = None,
) -> dict[str, Path]:
    """
    Fetch data from cloud sources.
    
    Parameters
    ----------
    bbox : list[float]
        Bounding box [west, south, east, north] in WGS84
    output_dir : Path
        Output directory for downloaded data
    datasets : list[str], optional
        Which datasets to fetch. Default: all available
    time_range : tuple[str, str], optional
        Time range for time-varying data (start, end)
        
    Returns
    -------
    dict[str, Path]
        Mapping of dataset names to output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if datasets is None:
        datasets = ["dem", "landcover", "canopy_height", "ecoregions"]
    
    results = {}
    
    # Import cloud fetcher functions
    try:
        import sys
        # Temporarily modify global config in cloud_fetcher
        import cloud_fetcher
        cloud_fetcher.BBOX = bbox
        cloud_fetcher.OUTPUT_DIR = output_dir
        
        if time_range:
            cloud_fetcher.TIME_START = time_range[0]
            cloud_fetcher.TIME_END = time_range[1]
        
        if "dem" in datasets:
            logger.info("Fetching DEM...")
            cloud_fetcher.fetch_dem(bbox)
            results["dem"] = output_dir / "dem_30m.tif"
        
        if "landcover" in datasets:
            logger.info("Fetching land cover...")
            cloud_fetcher.fetch_landcover(bbox)
            results["landcover"] = output_dir / "landcover_10m.tif"
        
        if "canopy_height" in datasets:
            logger.info("Fetching canopy height...")
            cloud_fetcher.fetch_canopy_height(bbox)
            results["canopy_height"] = output_dir / "canopy_height_10m.tif"
        
        if "ecoregions" in datasets:
            logger.info("Fetching ecoregions...")
            cloud_fetcher.fetch_ecoregions(bbox)
            results["ecoregions"] = output_dir / "ecoregions.geojson"
        
        if "burned_area" in datasets:
            logger.info("Fetching burned area...")
            cloud_fetcher.fetch_burned_area(bbox)
            results["burned_area"] = output_dir / "burned_area_2020.geojson"
            
    except ImportError as e:
        logger.warning(f"Cloud fetcher not available: {e}")
        logger.info("Install required packages: pip install pystac-client boto3 rioxarray")
    
    return results


def create_fuel_raster_from_landcover(
    landcover_path: Path,
    output_path: Path,
    fuel_mapping: dict[int, str] = None,
) -> Path:
    """
    Create FBP fuel type raster from land cover classification.
    
    Parameters
    ----------
    landcover_path : Path
        Path to land cover raster (e.g., ESA WorldCover)
    output_path : Path
        Output path for fuel raster
    fuel_mapping : dict, optional
        Custom land cover to fuel type mapping
        
    Returns
    -------
    Path
        Path to created fuel raster
    """
    import rasterio
    
    if fuel_mapping is None:
        fuel_mapping = DataSourceConfig.LANDCOVER_TO_FUEL
    
    fuel_codes = DataSourceConfig.FBP_FUEL_CODES
    
    logger.info(f"Creating fuel raster from {landcover_path}")
    
    with rasterio.open(landcover_path) as src:
        landcover = src.read(1)
        profile = src.profile.copy()
    
    # Map land cover to fuel codes
    fuel_raster = np.zeros_like(landcover, dtype=np.int16)
    
    for lc_class, fuel_type in fuel_mapping.items():
        fuel_code = fuel_codes.get(fuel_type, 0)
        fuel_raster[landcover == lc_class] = fuel_code
    
    # Update profile
    profile.update(dtype=rasterio.int16, count=1, compress='lzw')
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(fuel_raster, 1)
    
    logger.info(f"Saved fuel raster to {output_path}")
    
    # Report fuel distribution
    unique, counts = np.unique(fuel_raster, return_counts=True)
    logger.info("Fuel distribution:")
    for u, c in zip(unique, counts):
        pct = c / fuel_raster.size * 100
        logger.info(f"  Code {u}: {c} cells ({pct:.1f}%)")
    
    return output_path


def parse_station_weather(
    station_csv: Path,
    output_dir: Path,
    date_range: tuple[datetime, datetime] = None,
) -> tuple[Path, dict]:
    """
    Parse station weather data and convert to Ignacio format.
    
    Parameters
    ----------
    station_csv : Path
        Path to station weather CSV
    output_dir : Path
        Output directory
    date_range : tuple, optional
        Filter to date range
        
    Returns
    -------
    tuple[Path, dict]
        Path to converted weather file and summary statistics
    """
    from .station_parser import (
        load_station_csv,
        convert_to_ignacio_format,
        summarize_fire_weather,
    )
    
    logger.info(f"Parsing station data from {station_csv}")
    
    # Load and parse
    station_data = load_station_csv(station_csv)
    
    # Filter by date range if provided
    if date_range:
        start, end = date_range
        mask = (station_data["datetime"] >= start) & (station_data["datetime"] <= end)
        station_data = station_data[mask]
        logger.info(f"Filtered to {len(station_data)} records in date range")
    
    # Convert to Ignacio format
    output_path = output_dir / "weather_hourly.csv"
    convert_to_ignacio_format(station_data, output_path)
    
    # Generate summary
    summary = summarize_fire_weather(station_data)
    
    # Save summary
    summary_path = output_dir / "weather_summary.json"
    with open(summary_path, 'w') as f:
        # Convert datetime objects to strings for JSON
        summary_json = {}
        for k, v in summary.items():
            if isinstance(v, tuple) and len(v) == 2:
                summary_json[k] = [str(v[0]), str(v[1])]
            elif isinstance(v, (np.floating, np.integer)):
                summary_json[k] = float(v)
            else:
                summary_json[k] = v
        json.dump(summary_json, f, indent=2)
    
    logger.info(f"Weather summary saved to {summary_path}")
    
    return output_path, summary


def create_ignition_shapefile(
    coordinates: list[tuple[float, float]],
    output_path: Path,
    crs: str = "EPSG:4326",
    attributes: dict = None,
) -> Path:
    """
    Create ignition point shapefile from coordinates.
    
    Parameters
    ----------
    coordinates : list[tuple[float, float]]
        List of (x, y) or (lon, lat) coordinates
    output_path : Path
        Output shapefile path
    crs : str
        Coordinate reference system
    attributes : dict, optional
        Additional attributes for each point
        
    Returns
    -------
    Path
        Path to created shapefile
    """
    import geopandas as gpd
    from shapely.geometry import Point
    
    logger.info(f"Creating ignition shapefile with {len(coordinates)} points")
    
    # Create points
    points = [Point(x, y) for x, y in coordinates]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
    
    # Add ID column
    gdf["id"] = range(1, len(coordinates) + 1)
    
    # Add attributes if provided
    if attributes:
        for key, values in attributes.items():
            if len(values) == len(coordinates):
                gdf[key] = values
    
    # Save
    gdf.to_file(output_path)
    logger.info(f"Saved ignition points to {output_path}")
    
    return output_path


def generate_config_from_data(
    data_dir: Path,
    output_path: Path,
    project_name: str = "Fire Simulation",
    ignition_coords: list[tuple[float, float]] = None,
    simulation_date: datetime = None,
    duration_hours: float = 24.0,
    physics_options: dict = None,
) -> Path:
    """
    Generate Ignacio config file from preprocessed data.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing preprocessed data
    output_path : Path
        Output config file path
    project_name : str
        Project name
    ignition_coords : list, optional
        Ignition coordinates [(x1,y1), (x2,y2), ...]
    simulation_date : datetime, optional
        Simulation start date
    duration_hours : float
        Simulation duration in hours
    physics_options : dict, optional
        Physics module options
        
    Returns
    -------
    Path
        Path to created config file
    """
    import yaml
    
    data_dir = Path(data_dir)
    
    # Find available data files
    dem_path = None
    fuel_path = None
    weather_path = None
    ignition_path = None
    
    for f in data_dir.glob("*.tif"):
        if "dem" in f.name.lower():
            dem_path = f
        elif "fuel" in f.name.lower():
            fuel_path = f
        elif "landcover" in f.name.lower() and fuel_path is None:
            fuel_path = f  # Use landcover if no fuel raster
    
    for f in data_dir.glob("*.csv"):
        if "weather" in f.name.lower():
            weather_path = f
    
    for f in data_dir.glob("*.shp"):
        if "ignition" in f.name.lower():
            ignition_path = f
    
    # Build config
    config = {
        "project": {
            "name": project_name,
            "description": f"Generated {datetime.now().isoformat()}",
            "output_dir": str(data_dir / "output"),
            "random_seed": 42,
        },
        "crs": {
            "working_crs": "EPSG:4326",
            "output_crs": "EPSG:4326",
        },
        "terrain": {
            "dem_path": str(dem_path) if dem_path else "./dem.tif",
        },
        "fuel": {
            "source_type": "raster",
            "path": str(fuel_path) if fuel_path else "./fuel.tif",
            "non_fuel_codes": [0, -1],
        },
        "ignition": {
            "source_type": "shapefile" if ignition_path else "coordinates",
        },
        "weather": {
            "station_path": str(weather_path) if weather_path else "./weather.csv",
        },
        "simulation": {
            "max_duration": duration_hours * 60,  # Convert to minutes
            "dt": 1.0,
            "n_vertices": 360,
            "initial_radius": 30.0,
        },
        "fbp": {
            "defaults": {
                "ffmc": 88.0,
                "dmc": 30.0,
                "dc": 200.0,
            },
        },
        "output": {
            "save_perimeters": True,
            "generate_plots": True,
        },
    }
    
    # Add ignition coordinates if provided
    if ignition_coords:
        config["ignition"]["source_type"] = "coordinates"
        config["ignition"]["coordinates"] = [
            {"x": x, "y": y} for x, y in ignition_coords
        ]
    elif ignition_path:
        config["ignition"]["point_path"] = str(ignition_path)
    
    # Add simulation date
    if simulation_date:
        config["simulation"]["start_datetime"] = simulation_date.isoformat()
    
    # Add physics options
    if physics_options:
        config["physics"] = physics_options
    
    # Write config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Generated config: {output_path}")
    
    return output_path


# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================

def run_preprocessing(
    config_path: Optional[Path] = None,
    bbox: Optional[list[float]] = None,
    output_dir: Optional[Path] = None,
    station_csv: Optional[Path] = None,
    fetch_cloud: bool = False,
    create_fuel: bool = True,
    ignition_coords: Optional[list[tuple[float, float]]] = None,
) -> dict:
    """
    Run complete preprocessing pipeline.
    
    Parameters
    ----------
    config_path : Path, optional
        Existing config file to use/update
    bbox : list[float], optional
        Bounding box [west, south, east, north] for cloud data
    output_dir : Path, optional
        Output directory (default: ./data/preprocessed)
    station_csv : Path, optional
        Path to station weather CSV
    fetch_cloud : bool
        Whether to fetch cloud data
    create_fuel : bool
        Whether to create fuel raster from land cover
    ignition_coords : list, optional
        Ignition coordinates
        
    Returns
    -------
    dict
        Preprocessing results and paths
    """
    logger.info("="*60)
    logger.info("IGNACIO PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path("./data/preprocessed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "output_dir": output_dir,
        "files": {},
        "summary": {},
    }
    
    # Load existing config if provided
    config = None
    if config_path and Path(config_path).exists():
        from .config import load_config
        config = load_config(config_path)
        logger.info(f"Loaded config: {config.project.name}")
        
        # Extract bbox from DEM if not provided
        if bbox is None and config.terrain.dem_path.exists():
            import rasterio
            with rasterio.open(config.terrain.dem_path) as src:
                bbox = list(src.bounds)
                logger.info(f"Extracted bbox from DEM: {bbox}")
    
    # Fetch cloud data
    if fetch_cloud and bbox:
        logger.info("\n--- Fetching Cloud Data ---")
        cloud_results = fetch_cloud_data(bbox, output_dir)
        results["files"].update(cloud_results)
    
    # Create fuel raster from land cover
    if create_fuel:
        landcover_path = results["files"].get("landcover")
        if landcover_path is None:
            # Look for existing landcover
            for f in output_dir.glob("*landcover*.tif"):
                landcover_path = f
                break
        
        if landcover_path and Path(landcover_path).exists():
            logger.info("\n--- Creating Fuel Raster ---")
            fuel_path = output_dir / "fuel_fbp.tif"
            create_fuel_raster_from_landcover(landcover_path, fuel_path)
            results["files"]["fuel"] = fuel_path
    
    # Parse station weather
    if station_csv and Path(station_csv).exists():
        logger.info("\n--- Parsing Station Weather ---")
        weather_path, weather_summary = parse_station_weather(
            Path(station_csv), output_dir
        )
        results["files"]["weather"] = weather_path
        results["summary"]["weather"] = weather_summary
    
    # Create ignition shapefile
    if ignition_coords:
        logger.info("\n--- Creating Ignition Points ---")
        ignition_path = output_dir / "ignition_points.shp"
        create_ignition_shapefile(ignition_coords, ignition_path)
        results["files"]["ignition"] = ignition_path
    
    # Generate config if we have the necessary files
    if results["files"]:
        logger.info("\n--- Generating Config ---")
        config_output = output_dir / "ignacio_generated.yaml"
        generate_config_from_data(
            output_dir, config_output,
            ignition_coords=ignition_coords,
        )
        results["config"] = config_output
    
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("Generated files:")
    for name, path in results["files"].items():
        logger.info(f"  {name}: {path}")
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ignacio Fire Model Preprocessing Pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Existing config file to use/update"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in WGS84"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/preprocessed",
        help="Output directory"
    )
    parser.add_argument(
        "--station-csv", "-s",
        type=str,
        help="Path to station weather CSV"
    )
    parser.add_argument(
        "--fetch-cloud",
        action="store_true",
        help="Fetch data from cloud sources"
    )
    parser.add_argument(
        "--ignition",
        type=float,
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="Ignition coordinate (can specify multiple)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Run preprocessing
    results = run_preprocessing(
        config_path=Path(args.config) if args.config else None,
        bbox=args.bbox,
        output_dir=Path(args.output_dir),
        station_csv=Path(args.station_csv) if args.station_csv else None,
        fetch_cloud=args.fetch_cloud,
        ignition_coords=args.ignition,
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
