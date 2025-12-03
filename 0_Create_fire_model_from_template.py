# ==============================================================
# Fire Model Project Structure Generator (Template-driven)
# Based on the Prometheus Fire Growth Simulation System
# ==============================================================

import os
import json
import sys
from datetime import datetime

# ==============================================================
# 1️⃣ PARSE PROJECT TEMPLATE FILE
# ==============================================================

def parse_template(template_path):
    """
    Parse a simple key=value text configuration file into a dictionary.

    Example text file format:
        PROJECT_NAME = Kelowna_FireModel
        BASE_DIR = /Users/Martyn/GitRepos/BurnP3+/Kelowna/
        USER = Katherine Reece
        DESCRIPTION = Kelowna wildfire growth simulation using FBP data.
        FUEL_GRID = /path/to/fuel.asc
        WEATHER_FILE = /path/to/weather.csv
        IGNITION_GRID = /path/to/ignition.tif
        START_TIME = 2024-07-15T12:00:00
        DURATION_HOURS = 24
    """
    params = {}
    with open(template_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                params[key.strip()] = value.strip()
    return params


# ==============================================================
# 2️⃣ CREATE DIRECTORY STRUCTURE
# ==============================================================

def create_fire_model_structure(base_dir, project_name):
    """
    Create the Prometheus-style directory structure for the fire model.
    """
    project_dir = os.path.join(base_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    subdirs = {
        "Fuel": ["FBP_Fuel_Grid", "Fuel_Lookup_Tables", "Fuel_Modifiers", "Fuel_Patches"],
        "Topography": ["Elevation", "Slope", "Aspect"],
        "Weather": ["Daily", "Hourly", "Spatial_Interpolation"],
        "Ignition": ["Ignition_Grids", "Ignition_Polygons"],
        "Output": ["FirePerimeters", "Statistics", "Maps", "Logs"],
        "Vectors": ["Hydrology", "Access", "Boundaries", "ValuesAtRisk", "FuelBreaks"],
        "Scenarios": []
    }

    for parent, children in subdirs.items():
        parent_path = os.path.join(project_dir, parent)
        os.makedirs(parent_path, exist_ok=True)
        for child in children:
            os.makedirs(os.path.join(parent_path, child), exist_ok=True)

    print(f"✅ Created project directory: {project_dir}")
    return project_dir


# ==============================================================
# 3️⃣ CREATE PROJECT MANIFEST (Equivalent to .FGM Project File)
# ==============================================================

def create_project_manifest(project_dir, params):
    """
    Create a JSON manifest file with project metadata and scenario placeholders.
    """
    project_name = params.get("PROJECT_NAME", "FireProject")
    user = params.get("USER", "Unknown")
    description = params.get("DESCRIPTION", "")

    manifest = {
        "project_name": project_name,
        "created_by": user,
        "description": description,
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modules": {
            "FuelCom": "Handles fuel grids, lookup tables, and modifiers.",
            "GridCom": "Manages spatial topology and projection data.",
            "WeatherCom": "Manages weather streams and hourly/daily inputs.",
            "FWICom": "Handles Fire Weather Index and subcomponents.",
            "FireEngine": "Performs fire growth and perimeter propagation.",
            "PrometheusCOM": "Combines all modules for high-level execution."
        },
        "directories": {
            "Fuel": "Fuel grids, lookup tables, and modifiers.",
            "Topography": "Elevation, slope, and aspect grids.",
            "Weather": "Weather data and interpolations.",
            "Ignition": "Ignition grids or polygons.",
            "Output": "Simulation outputs.",
            "Vectors": "Vector layers (boundaries, hydrology, etc.).",
            "Scenarios": "Simulation scenario definitions."
        },
        "scenarios": []
    }

    manifest_path = os.path.join(project_dir, f"{project_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ Created manifest file: {manifest_path}")
    return manifest_path


# ==============================================================
# 4️⃣ ADD SCENARIO TO MANIFEST
# ==============================================================

def add_scenario_from_params(project_dir, params):
    """
    Append a scenario to the project's manifest JSON file based on parsed parameters.
    """
    project_name = params.get("PROJECT_NAME", "FireProject")
    manifest_path = os.path.join(project_dir, f"{project_name}_manifest.json")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Determine fuel_grid based on FUEL_SOURCE_TYPE
    fuel_source_type = params.get("FUEL_SOURCE_TYPE", "").lower()
    if fuel_source_type == "polygon":
        fuel_grid = params.get("FUEL_POLYGON_PATH")
    elif fuel_source_type == "grid":
        fuel_grid = params.get("FUEL_GRID")
    else:
        fuel_grid = None

    # Determine ignition_file based on IGNITION_SOURCE
    ignition_source = params.get("IGNITION_SOURCE", "").lower()
    if ignition_source == "point":
        ignition_file = params.get("IGNITION_PATH")
    elif ignition_source == "grid":
        ignition_file = params.get("IGNITION_GRID")
    else:
        ignition_file = None

    scenario = {
        "scenario_name": "Scenario_1",
        "fuel_grid": fuel_grid,
        "lookup_table": params.get("LOOKUP_TABLE"),
        "topography": {
            "elevation": params.get("TOPO_ELEV"),
            "slope": params.get("TOPO_SLOPE"),
            "aspect": params.get("TOPO_ASPECT")
        },
        "weather_file": params.get("WEATHER_FILE"),
        "ignition_file": ignition_file,
        "start_time": params.get("START_TIME"),
        "duration_hours": float(params.get("DURATION_HOURS", 12)),
        "parameters": {
            "time_step_minutes": 1,
            "output_interval_minutes": 60,
            "max_burning_period_hours": 12,
            "use_dynamic_weather": True,
            "use_fuel_modifiers": True,
            "spatial_weather_interpolation": False
        },
        "status": "Pending"
    }

    manifest["scenarios"].append(scenario)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ Added scenario to manifest ({manifest_path})")


# ==============================================================
# 5️⃣ MAIN EXECUTION
# ==============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        template_file = input("Enter the path to your project template file: ").strip()
        if not template_file:
            template_file = "/Users/Martyn/Desktop/PhD/Fyah/Python_Scripts/project_template.txt"
    else:
        template_file = sys.argv[1]

    # Step 1: Parse text template
    params = parse_template(template_file)
    project_name = params.get("PROJECT_NAME", "FireProject")
    base_dir = params.get("BASE_DIR", os.getcwd())

    # Step 2: Create directory structure
    project_dir = create_fire_model_structure(base_dir, project_name)

    # Step 3: Create manifest
    create_project_manifest(project_dir, params)

    # Step 4: Add scenario
    add_scenario_from_params(project_dir, params)

    print("\n🎯 Fire model project setup complete!\n")