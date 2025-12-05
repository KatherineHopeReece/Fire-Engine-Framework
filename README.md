# IGNACIO

**Integrated Geographic Numerical Analysis for Combustion and Ignition Operations**

IGNACIO is a Python implementation of wildland fire growth modeling based on the Canadian Forest Fire Behaviour Prediction (FBP) System. It uses Richards' elliptical wave propagation equations for realistic fire perimeter evolution, similar to the PROMETHEUS fire growth model.

## Features

- **Canadian FBP System**: Complete implementation of rate of spread equations for all 16 standard fuel types (C-1 through S-3), plus extended mixedwood codes (405-995)
- **Fire Weather Index (FWI) System**: Calculate FFMC, DMC, DC, ISI, BUI, and FWI from weather observations
- **Time-Varying Weather**: Interpolate hourly weather data throughout simulation for realistic diurnal fire behavior
- **Richards' Elliptical Spread**: Differential equation solver for realistic fire perimeter evolution
- **Marker Method**: Robust handling of perimeter topology using turning number calculations
- **Terrain Effects**: Slope and aspect corrections using official FBP slope factor equations
- **Geographic CRS Support**: Native handling of lat/lon coordinates (EPSG:4326) with automatic unit conversions
- **Automatic Grid Alignment**: Resamples fuel grids to match DEM extent and CRS using proper geospatial reprojection
- **Fire Weather Integration**: ISI threshold filtering and spread event day sampling
- **Configurable**: Single YAML configuration file for all parameters
- **GIS Output**: Export fire perimeters as shapefiles compatible with QGIS, ArcGIS, etc.

## Installation

### From Source

```bash
git clone https://github.com/fire-engine-framework/ignacio.git
cd ignacio
pip install -e .
```

### Dependencies

- Python >= 3.11
- NumPy >= 1.24
- Pandas >= 2.0
- GeoPandas >= 0.14
- Rasterio >= 1.3
- Shapely >= 2.0
- SciPy >= 1.11
- Matplotlib >= 3.7
- PyYAML >= 6.0
- Click >= 8.1
- Pydantic >= 2.0

## Quick Start

### 1. Prepare Input Data

Required inputs:
- **DEM**: Digital Elevation Model (GeoTIFF) - can be in any CRS including EPSG:4326
- **Fuel Grid**: Fuel type raster with FBP codes - automatically reprojected to match DEM
- **Ignition Points**: Point shapefile or coordinates - automatically reprojected to match DEM
- **Weather Data**: CSV with temperature, RH, wind, precipitation (or pre-calculated FWI)

### 2. Create Configuration

Create an `ignacio.yaml` configuration file:

```yaml
project:
  name: "MyFireSimulation"
  output_dir: "output"

terrain:
  dem_path: "data/dem.tif"

fuel:
  path: "data/fuels.tif"
  fuel_lookup:
    2: "C-2"
    3: "C-3"
    4: "C-4"

ignition:
  shapefile: "data/ignitions.shp"

weather:
  station_file: "data/stations.csv"
  data_file: "data/weather.csv"

simulation:
  dt: 1.0              # Time step (minutes)
  n_vertices: 300      # Perimeter resolution
  initial_radius: 10   # Initial fire size (meters)
  spread_duration: 480 # Simulation duration (minutes)
```

### 3. Run Simulation

```bash
python run_ignacio.py
# Or with a custom config:
python run_ignacio.py --config my_config.yaml
```

Or using the pip CLI

```bash
ignacio run my_config.yaml
```

### 4. View Results

Output files are written to the `output/` directory:

```
output/
├── terrain/
│   ├── slope_deg.tif      # Slope in degrees
│   └── aspect_deg.tif     # Aspect in degrees
├── fire_weather_list.csv  # Selected fire weather conditions
├── ignitions.shp          # Ignition points (reprojected)
├── fire_1/
│   └── perimeters.shp     # Time-series perimeters
├── all_perimeters.shp     # Final fire perimeters (all fires)
└── simulation_summary.csv # Summary statistics
```

Load `all_perimeters.shp` in QGIS or ArcGIS to visualize fire spread.

## Configuration

The `ignacio.yaml` file controls all aspects of the simulation:

```yaml
project:
  name: "My Fire Simulation"
  output_dir: "./output"
  random_seed: 42

terrain:
  dem_path: "./data/dem.tif"

fuel:
  path: "./data/fuels.tif"
  fuel_lookup:
    1: "C-1"
    2: "C-2"
    3: "C-3"

ignition:
  source_type: "grid"
  grid_path: "./data/ignition.tif"
  n_iterations: 100

weather:
  station_path: "./data/stations.csv"
  weather_path: "./data/weather.csv"
  isi_thresholds:
    high: 8.6
    extreme: 12.6

fbp:
  defaults:
    bui: 70.0
    isi: 10.0
  fmc: 100.0
  slope_factor: 0.5

simulation:
  dt: 1.0
  max_duration: 1440
  n_vertices: 300
```

See the [Configuration Reference](#configuration-reference) for all options.

## FBP Fuel Types

IGNACIO implements all standard Canadian FBP fuel types plus extended mixedwood codes:

### Standard Fuel Types

| Code | Type | Description |
|------|------|-------------|
| 1 | C-1 | Spruce-Lichen Woodland |
| 2 | C-2 | Boreal Spruce |
| 3 | C-3 | Mature Jack/Lodgepole Pine |
| 4 | C-4 | Immature Jack/Lodgepole Pine |
| 5 | C-5 | Red and White Pine |
| 6 | C-6 | Conifer Plantation |
| 7 | C-7 | Ponderosa Pine/Douglas-fir |
| 11 | D-1 | Leafless Aspen |
| 21 | S-1 | Jack/Lodgepole Pine Slash |
| 22 | S-2 | White Spruce/Balsam Slash |
| 23 | S-3 | Coastal Cedar/Hemlock/Douglas-fir Slash |
| 31 | O-1a | Matted Grass |
| 32 | O-1b | Standing Grass |
| 40 | M-1 | Boreal Mixedwood - Leafless |
| 50 | M-2 | Boreal Mixedwood - Green |
| 70 | M-3 | Dead Balsam Fir Mixedwood - Leafless |
| 80 | M-4 | Dead Balsam Fir Mixedwood - Green |

### Extended Mixedwood Codes (CIFFC Standard)

| Range | Type | Description |
|-------|------|-------------|
| 405-495 | M-1 | Boreal Mixedwood Leafless (5-95% conifer) |
| 505-595 | M-2 | Boreal Mixedwood Green (5-95% conifer) |
| 605-695 | M-1/M-2 | Boreal Mixedwood (5-95% conifer) |
| 705-795 | M-3 | Dead Balsam Fir Leafless (5-95% dead fir) |
| 805-895 | M-4 | Dead Balsam Fir Green (5-95% dead fir) |

### Non-Fuel Codes

| Code | Description |
|------|-------------|
| 101 | Non-fuel |
| 102 | Water |
| 105 | Vegetated Non-Fuel |
| 106 | Urban/Built-up |

## CLI Reference

```
ignacio --help

Commands:
  run       Run fire growth simulation
  validate  Validate configuration file
  terrain   Process DEM to generate slope/aspect
  init      Generate configuration template
  plot      Generate plots from results
  info      Display system information
```

### Examples

```bash
# Run simulation
ignacio run config.yaml

# Run with overrides
ignacio run config.yaml --seed 12345 --iterations 50 -v

# Validate configuration
ignacio validate config.yaml

# Process terrain
ignacio terrain dem.tif --output ./terrain

# Create template
ignacio init --output my_config.yaml
```

## API Reference

### Core Functions

```python
from ignacio import load_config, run_simulation

# Load configuration
config = load_config("ignacio.yaml")

# Run simulation
results = run_simulation(config)

# Access results
for fire in results.fires:
    print(f"Area: {fire.final_area_ha:.2f} ha")

# Get all perimeters as GeoDataFrame
perimeters = results.get_all_perimeters()
perimeters.to_file("all_fires.shp")
```

### FBP Calculations

```python
from ignacio.fbp import compute_ros, calculate_isi

# Calculate ISI from FFMC and wind
isi = calculate_isi(ffmc=89.0, wind_speed=20.0)

# Compute rate of spread
ros = compute_ros(
    fuel_type="C-2",
    isi=isi,
    bui=80.0,
    fmc=100.0,
)
print(f"ROS: {ros:.2f} m/min")
```

### Terrain Processing

```python
from ignacio.terrain import build_terrain_grids, compute_slope_aspect
from ignacio.io import read_raster

# Read DEM
dem = read_raster("dem.tif")

# Compute slope and aspect
slope, aspect = compute_slope_aspect(dem.data, dx=30.0, dy=30.0)
```

## Architecture

```
ignacio/
├── __init__.py        # Package exports
├── config.py          # Configuration loading (Pydantic)
├── io.py              # Raster/vector I/O
├── terrain.py         # DEM processing
├── ignition.py        # Ignition probability
├── weather.py         # Fire weather processing
├── fbp.py             # FBP rate of spread equations
├── spread.py          # Richards equations & marker method
├── simulation.py      # Main orchestration
├── visualization.py   # Plotting & animation
└── cli.py             # Command-line interface
```

## Scientific Background

### Fire Behaviour Prediction System

The Canadian FBP System predicts fire behaviour based on:

1. **Fire Weather Index (FWI)** components:
   - FFMC: Fine Fuel Moisture Code
   - DMC: Duff Moisture Code
   - DC: Drought Code
   - ISI: Initial Spread Index
   - BUI: Buildup Index
   - FWI: Fire Weather Index

2. **Fuel type** characteristics (a, b, c coefficients)

3. **Terrain** effects (slope, aspect)

The head fire rate of spread is computed as:

```
ROS = a * (1 - exp(-b * ISI))^c * BE
```

where BE is the buildup effect.

### Richards' Equations

Fire perimeter evolution follows Richards' differential equations for elliptical spread:

```
dx/dt = f(a, b, c, theta, ds)
dy/dt = g(a, b, c, theta, ds)
```

where:
- a = (ROS + BROS) / 2 (semi-major axis)
- b = FROS (semi-minor axis)
- c = (ROS - BROS) / 2 (center offset)
- theta = spread direction

### Marker Method

The marker method determines active vertices on the fire front using turning numbers to handle perimeter self-intersection and topology changes.

## Time-Varying Weather

IGNACIO supports time-varying weather conditions throughout the simulation, enabling realistic diurnal fire behavior patterns.

### Configuration

```yaml
simulation:
  time_varying_weather: true
  start_datetime: "2024-07-15 12:00:00"  # Simulation start time
  default_start_hour: 12  # Default hour if no datetime specified
```

### How It Works

1. **Hourly Interpolation**: Weather values (TEMP, RH, WS, WD, ISI, BUI) are linearly interpolated between hourly observations
2. **Wind Direction**: Uses circular interpolation to handle 0°/360° wraparound
3. **Per-Timestep ROS**: Rate of spread is recalculated for each simulation timestep based on interpolated weather
4. **Diurnal Patterns**: Fire intensity naturally peaks in afternoon when ISI is highest

### Example Output

```
Using time-varying weather interpolation
Time-varying weather: 2024-07-15 12:00 to 20:00
  t=0: 12:00, ISI=5.0, WD=225°, mean ROS=3.2 m/min
  t=96: 13:36, ISI=7.2, WD=230°, mean ROS=4.8 m/min
  t=192: 15:12, ISI=8.5, WD=235°, mean ROS=5.6 m/min
  t=288: 16:48, ISI=6.8, WD=230°, mean ROS=4.5 m/min
```

### Static Weather Fallback

If `time_varying_weather: false` or hourly data is unavailable, IGNACIO uses representative weather from high fire danger days (constant throughout simulation).

## Known Limitations

### Topology Handling

The current marker method uses winding numbers for perimeter topology, which works well for simple and moderately complex shapes. However:

- **Island formation**: If fire wraps around a non-fuel island (e.g., lake), the island may be incorrectly treated as burned
- **Multi-fire merging**: When multiple fires meet, the merge may produce artifacts
- **Self-intersection**: Highly irregular perimeters may develop self-intersections

For production use with complex topology, consider post-processing with polygon clipping libraries (e.g., Shapely, Clipper).

### Weather Data Requirements

Time-varying weather requires hourly observations. If your data has gaps:
- Linear interpolation spans gaps up to 1 hour
- Longer gaps use the nearest available value
- FWI indices (ISI, BUI) should be pre-calculated for best results

## References

1. Forestry Canada Fire Danger Group (1992). *Development and Structure of the Canadian Forest Fire Behavior Prediction System*. Information Report ST-X-3.

2. Tymstra, C., Bryce, R.W., Wotton, B.M., Taylor, S.W., & Armitage, O.B. (2010). *Development and Structure of Prometheus: the Canadian Wildland Fire Growth Simulation Model*. Information Report NOR-X-417.

3. Van Wagner, C.E. (1987). *Development and Structure of the Canadian Forest Fire Weather Index System*. Forestry Technical Report 35.

4. Richards, G.D. (1990). An elliptical growth model of forest fire fronts and its numerical solution. *International Journal for Numerical Methods in Engineering*, 30(6), 1163-1179.

## License

MIT License

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
