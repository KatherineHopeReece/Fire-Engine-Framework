# ==============================================================
# Fire Model Project Structure Generator
# Based on the Prometheus Fire Growth Simulation System
# ==============================================================

import os
import json
from datetime import datetime

# ==============================================================
# 1️⃣ DEFINE MODEL STRUCTURE
# ==============================================================

def create_fire_model_structure(base_dir, project_name):
    """
    Create a Prometheus-style fire modeling project directory structure.
    
    Parameters
    ----------
    base_dir : str
        Path to the base directory where the project will be created.
    project_name : str
        Name of the project (will be used as the folder name).
    """
    project_dir = os.path.join(base_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Define subdirectories following Prometheus-like organization
    subdirs = {
        "Fuel": [
            "FBP_Fuel_Grid",
            "Fuel_Lookup_Tables",
            "Fuel_Modifiers",
            "Fuel_Patches"
        ],
        "Topography": [
            "Elevation",
            "Slope",
            "Aspect"
        ],
        "Weather": [
            "Daily",
            "Hourly",
            "Spatial_Interpolation"
        ],
        "Ignition": [
            "Ignition_Grids",
            "Ignition_Polygons"
        ],
        "Output": [
            "FirePerimeters",
            "Statistics",
            "Maps",
            "Logs"
        ],
        "Vectors": [
            "Hydrology",
            "Access",
            "Boundaries",
            "ValuesAtRisk",
            "FuelBreaks"
        ],
        "Scenarios": []
    }

    # Create directories
    for parent, children in subdirs.items():
        parent_path = os.path.join(project_dir, parent)
        os.makedirs(parent_path, exist_ok=True)
        for child in children:
            os.makedirs(os.path.join(parent_path, child), exist_ok=True)

    print(f"✅ Created project directory: {project_dir}")
    return project_dir


# ==============================================================
# 2️⃣ DEFINE PROJECT MANIFEST (Equivalent to .FGM Project File)
# ==============================================================

def create_project_manifest(project_dir, project_name, user, description=""):
    """
    Create a JSON manifest file to store model metadata and scenario structure.
    
    Equivalent to the Prometheus `.fgm` binary project file.
    """
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
            "PrometheusCOM": "High-level API combining all subsystems."
        },
        "directories": {
            "Fuel": "Fuel data, lookup tables, and modifiers.",
            "Topography": "Elevation, slope, and aspect grids.",
            "Weather": "Daily/hourly weather data and interpolations.",
            "Ignition": "Ignition grids and polygons.",
            "Output": "Simulation outputs (perimeters, stats, maps).",
            "Vectors": "Vector layers for hydrology, access, boundaries, etc.",
            "Scenarios": "User-defined simulation scenarios."
        },
        "scenarios": []
    }

    manifest_path = os.path.join(project_dir, f"{project_name}_manifest.json")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ Created manifest file: {manifest_path}")
    return manifest_path


# ==============================================================
# 3️⃣ ADD SCENARIO TEMPLATE
# ==============================================================

def create_scenario(project_dir, scenario_name, fuel_grid, weather_file, ignition_file, start_time, duration_hours):
    """
    Add a scenario entry to the project manifest.
    """
    manifest_path = os.path.join(project_dir, f"{os.path.basename(project_dir)}_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    scenario = {
        "scenario_name": scenario_name,
        "fuel_grid": fuel_grid,
        "weather_file": weather_file,
        "ignition_file": ignition_file,
        "start_time": start_time,
        "duration_hours": duration_hours,
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

    print(f"✅ Added scenario '{scenario_name}' to manifest.")


# ==============================================================
# 4️⃣ EXAMPLE EXECUTION
# ==============================================================

if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    base_dir = "/Users/Martyn/GitRepos/BurnP3+/Kelowna/"
    project_name = "Kelowna_FireModel"
    user = "Martyn"
    description = "Kelowna wildfire growth simulation using FBP data."

    # --- CREATE PROJECT STRUCTURE ---
    project_dir = create_fire_model_structure(base_dir, project_name)

    # --- CREATE MANIFEST (.FGM-style JSON) ---
    manifest_path = create_project_manifest(project_dir, project_name, user, description)

    # --- ADD A TEST SCENARIO ---
    create_scenario(
        project_dir=project_dir,
        scenario_name="Baseline_Summer",
        fuel_grid="Fuel/FBP_Fuel_Grid/Kelowna_Fuel_2024.asc",
        weather_file="Weather/Daily/Kelowna_Weather_2024.csv",
        ignition_file="Ignition/Ignition_Grids/H_Summer_ign_randomforest.tif",
        start_time="2024-07-15T12:00:00",
        duration_hours=24
    )