"""
Configuration loading and validation for Ignacio.

This module provides Pydantic models for validating the ignacio.yaml
configuration file and utility functions for loading configurations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================


class ProjectConfig(BaseModel):
    """Project metadata configuration."""
    
    name: str = Field(..., description="Project name")
    description: str = Field("", description="Project description")
    output_dir: Path = Field(Path("./output"), description="Output directory")
    random_seed: int | None = Field(None, description="Random seed for reproducibility")


class CRSConfig(BaseModel):
    """Coordinate reference system configuration."""
    
    working_crs: str = Field("EPSG:3005", description="Working CRS for computations")
    output_crs: str = Field("EPSG:3005", description="Output CRS for results")


class TerrainConfig(BaseModel):
    """Terrain input configuration."""
    
    dem_path: Path = Field(..., description="Path to DEM raster")
    slope_path: Path | None = Field(None, description="Pre-computed slope grid")
    aspect_path: Path | None = Field(None, description="Pre-computed aspect grid")


class FuelConfig(BaseModel):
    """Fuel layer configuration."""
    
    source_type: Literal["raster", "polygon"] = Field("raster")
    path: Path = Field(..., description="Path to fuel raster or shapefile")
    polygon_path: Path | None = Field(None, description="Polygon shapefile path")
    value_column: str | None = Field(None, description="Column for fuel codes")
    non_fuel_codes: list[int] = Field(
        default_factory=lambda: [0, 101, 102, 106, -9999],
        description="Codes representing non-burnable areas"
    )
    fuel_lookup: dict[int, str] = Field(
        default_factory=dict,
        description="Mapping of numeric IDs to FBP fuel codes"
    )


class IgnitionRule(BaseModel):
    """Rule for ignition restrictions by fuel type."""
    
    fuel_types: list[str]
    cause: str
    season: str | None = None


class IgnitionConfig(BaseModel):
    """Ignition configuration."""
    
    source_type: Literal["grid", "point", "shapefile"] = Field("grid")
    grid_path: Path | None = Field(None)
    point_path: Path | None = Field(None)
    cause: Literal["Lightning", "Human"] = Field("Lightning")
    season: Literal["Spring", "Summer", "Fall"] = Field("Summer")
    n_iterations: int = Field(10, ge=1)
    ecoregion_path: Path | None = Field(None)
    escaped_fire_rates: dict[str, dict[str, float]] = Field(default_factory=dict)
    fire_occurrence_rates: dict[str, float] = Field(default_factory=dict)
    escaped_fire_distribution: dict[int, float] = Field(
        default_factory=lambda: {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2}
    )
    historical_fire_path: Path | None = Field(None)
    ignition_rules: list[IgnitionRule] = Field(default_factory=list)
    
    @model_validator(mode="after")
    def check_source_paths(self) -> "IgnitionConfig":
        """Validate that appropriate path is provided for source type."""
        if self.source_type == "grid" and self.grid_path is None:
            raise ValueError("grid_path required when source_type is 'grid'")
        if self.source_type in ("point", "shapefile") and self.point_path is None:
            raise ValueError("point_path required when source_type is 'point' or 'shapefile'")
        return self


class WeatherColumnsConfig(BaseModel):
    """Column name mapping for weather data."""
    
    datetime: str = "DATE_TIME"
    station_code: str = "STATION_CODE"
    temperature: str = "HOURLY_TEMPERATURE"
    relative_humidity: str = "HOURLY_RELATIVE_HUMIDITY"
    wind_speed: str = "HOURLY_WIND_SPEED"
    wind_direction: str = "HOURLY_WIND_DIRECTION"
    precipitation: str = "PRECIPITATION"
    ffmc: str = "FFMC"
    dmc: str = "DMC"
    dc: str = "DC"
    isi: str = "INITIAL_SPREAD_INDEX"
    bui: str = "BUI"
    fwi: str = "FIRE_WEATHER_INDEX"


class StationColumnsConfig(BaseModel):
    """Column name mapping for station metadata."""
    
    code: str = "STATION_CODE"
    name: str = "STATION_NAME"
    latitude: str = "LATITUDE"
    longitude: str = "LONGITUDE"
    elevation: str = "ELEVATION_M"
    slope: str = "SLOPE"
    aspect: str = "ASPECT"


class ISIThresholdsConfig(BaseModel):
    """ISI threshold configuration."""
    
    moderate: float = 0.0
    high: float = 8.6
    extreme: float = 12.6


class WeatherConfig(BaseModel):
    """Weather and FWI configuration."""
    
    station_path: Path = Field(..., description="Station metadata or hourly weather CSV")
    weather_path: Path | None = Field(None, description="Hourly weather CSV (optional, uses station_path if not set)")
    station_columns: StationColumnsConfig = Field(default_factory=StationColumnsConfig)
    weather_columns: WeatherColumnsConfig = Field(default_factory=WeatherColumnsConfig)
    datetime_format: str = Field("%Y%m%d%H")
    isi_thresholds: ISIThresholdsConfig = Field(default_factory=ISIThresholdsConfig)
    filter_conditions: list[str] = Field(default_factory=lambda: ["moderate", "high", "extreme"])
    spread_event_lambda: float = Field(3.76, ge=0)
    calculate_fwi: bool = Field(True, description="Calculate FWI from hourly weather if not present")
    fwi_latitude: float | None = Field(None, description="Latitude for FWI calculation (uses DEM center if not set)")


class FBPDefaultsConfig(BaseModel):
    """Default FWI component values."""
    
    ffmc: float = Field(85.0, ge=0, le=101)
    dmc: float = Field(25.0, ge=0)
    dc: float = Field(200.0, ge=0)
    bui: float = Field(70.0, ge=0)
    isi: float = Field(10.0, ge=0)
    fwi: float = Field(20.0, ge=0)


class WindConfig(BaseModel):
    """Wind adjustment configuration."""
    
    measurement_height: float = Field(10.0, gt=0)
    canopy_adjustment: bool = Field(True)


class ElevationAdjustmentConfig(BaseModel):
    """Elevation-based adjustment configuration."""
    
    enabled: bool = Field(True)
    lapse_rate: float = Field(0.0065, ge=0)
    reference_temp: float = Field(20.0)
    reference_rh: float = Field(30.0, ge=0, le=100)


class FBPConfig(BaseModel):
    """Fire Behaviour Prediction configuration."""
    
    defaults: FBPDefaultsConfig = Field(default_factory=FBPDefaultsConfig)
    fmc: float = Field(100.0, ge=0, description="Foliar moisture content (%)")
    curing: float = Field(85.0, ge=0, le=100, description="Grass curing (%)")
    slope_factor: float = Field(0.5, ge=0, description="Slope effect strength")
    backing_fraction: float = Field(0.2, ge=0, le=1, description="BROS/ROS ratio")
    length_to_breadth: float = Field(2.0, gt=0, description="Fire shape L/B ratio")
    wind: WindConfig = Field(default_factory=WindConfig)
    elevation_adjustment: ElevationAdjustmentConfig = Field(
        default_factory=ElevationAdjustmentConfig
    )


class MarkerMethodConfig(BaseModel):
    """Marker method configuration."""
    
    enabled: bool = Field(True)
    epsilon: float = Field(1.0, gt=0)  # meters - offset for marker method


class SimulationConfig(BaseModel):
    """Simulation parameters configuration."""
    
    dt: float = Field(1.0, gt=0, description="Time step in minutes")
    max_duration: float = Field(1440, gt=0, description="Max duration in minutes")
    n_vertices: int = Field(300, ge=10, description="Fire perimeter vertices")
    initial_radius: float = Field(0.5, gt=0, description="Initial fire radius (m)")
    store_every: int = Field(10, ge=1, description="Storage frequency")
    marker_method: MarkerMethodConfig = Field(default_factory=MarkerMethodConfig)
    min_ros: float = Field(0.01, ge=0, description="Minimum ROS threshold")
    interpolation_resolution: float = Field(30.0, gt=0)
    
    # Time-varying weather
    time_varying_weather: bool = Field(True, description="Enable time-varying weather")
    start_datetime: str | None = Field(None, description="Simulation start datetime (ISO format)")
    default_start_hour: int = Field(12, ge=0, le=23, description="Default start hour if not specified")


class OutputConfig(BaseModel):
    """Output configuration."""
    
    save_perimeters: bool = Field(True)
    save_ros_grids: bool = Field(True)
    save_weather_summary: bool = Field(True)
    save_ignition_summary: bool = Field(True)
    perimeter_format: Literal["shapefile", "geojson", "geopackage"] = Field("shapefile")
    combine_perimeters: bool = Field(True)
    generate_plots: bool = Field(True)
    generate_animation: bool = Field(False)
    animation_fps: int = Field(10, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    log_file: Path | None = Field(None)

class FuelBreakConfig(BaseModel):
    """Optional Prometheus-style fuel break (fuel patch) configuration."""

    enabled: bool = Field(False, description="Enable fuel break fuel-patch override")
    path: Path | None = Field(None, description="Path to fuel break shapefile (line or polygon)")
    fuel_code: int | None = Field(
        None,
        description="Fuel code to assign inside the break (must exist in your fuel LUT)"
    )

    treatment_method: Literal["fuel_code", "zero_ros"] = Field(
        "fuel_code",
        description=(
            "How to apply the fuel break. 'fuel_code' overwrites cells with fuel_code; "
            "'zero_ros' keeps fuel codes but forces ROS=0 in the break mask."
        ),
    )

    all_touched: bool = Field(
        True,
        description="Rasterization option: burn any cell touched by geometry (good for roads)"
    )
    buffer_m: float | None = Field(
        None,
        ge=0,
        description="Optional buffer distance (meters) if the break layer is a line"
    )
    preserve_non_fuel: bool = Field(
        True,
        description="If True, do not overwrite cells already classified as non-fuel"
    )

    @model_validator(mode="after")
    def validate_break(self) -> "FuelBreakConfig":
        if self.enabled:
            if self.path is None:
                raise ValueError("fuel_break.path is required when fuel_break.enabled is True")
            if self.treatment_method == "fuel_code" and self.fuel_code is None:
                raise ValueError(
                    "fuel_break.fuel_code is required when fuel_break.enabled is True and treatment_method is 'fuel_code'"
                )
        return self


class IgnacioConfig(BaseModel):
    """Root configuration model for Ignacio."""
    
    project: ProjectConfig
    crs: CRSConfig = Field(default_factory=CRSConfig)
    terrain: TerrainConfig
    fuel: FuelConfig
    ignition: IgnitionConfig
    weather: WeatherConfig
    fbp: FBPConfig = Field(default_factory=FBPConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    # Backward-compatible single fuel break (deprecated; prefer `fuel_breaks`)
    fuel_break: FuelBreakConfig | None = Field(
        default=None,
        description="(Deprecated) Single fuel break. Prefer `fuel_breaks` list.",
    )

    # Preferred: list of fuel breaks
    fuel_breaks: list[FuelBreakConfig] = Field(
        default_factory=list,
        description="List of fuel breaks to apply (processed in order).",
    )
    
    @field_validator("project", mode="before")
    @classmethod
    def ensure_output_dir(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure output_dir is a Path."""
        if isinstance(v, dict) and "output_dir" in v:
            v["output_dir"] = Path(v["output_dir"])
        return v

    @field_validator("fuel_breaks", mode="before")
    @classmethod
    def _normalize_fuel_breaks(cls, v: Any) -> Any:
        """Allow fuel_breaks to be provided as a single mapping or a list."""
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return v



# =============================================================================
# Loading Functions
# =============================================================================


def load_config(config_path: str | Path) -> IgnacioConfig:
    """
    Load and validate configuration from a YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the ignacio.yaml configuration file.
        
    Returns
    -------
    IgnacioConfig
        Validated configuration object.
        
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration is invalid.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raise ValueError(f"Empty configuration file: {config_path}")
    
    # Resolve relative paths relative to config file location
    config_dir = config_path.parent
    raw_config = _resolve_paths(raw_config, config_dir)
    
    # Validate and create config object
    config = IgnacioConfig.model_validate(raw_config)
    
    logger.info(f"Configuration loaded: {config.project.name}")
    
    return config


def _resolve_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """
    Recursively resolve relative paths in configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    base_dir : Path
        Base directory for resolving relative paths.
        
    Returns
    -------
    dict
        Configuration with resolved paths.
    """
    path_keys = {
        "dem_path", "slope_path", "aspect_path", "path", "polygon_path",
        "grid_path", "point_path", "ecoregion_path", "historical_fire_path",
        "station_path", "weather_path", "output_dir", "log_file"
    }
    
    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve(item) for item in obj]
        elif isinstance(obj, str):
            # Check if this looks like a path
            if obj.startswith("./") or obj.startswith("../"):
                return str(base_dir / obj)
            return obj
        return obj
    
    return resolve(config)


def validate_paths(config: IgnacioConfig) -> list[str]:
    """
    Validate that required input files exist.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object to validate.
        
    Returns
    -------
    list[str]
        List of validation warnings (empty if all paths valid).
        
    Raises
    ------
    FileNotFoundError
        If required files are missing.
    """
    errors = []
    warnings = []
    
    # Required files
    required = [
        ("terrain.dem_path", config.terrain.dem_path),
        ("fuel.path", config.fuel.path),
    ]
    
    for name, path in required:
        if not Path(path).exists():
            errors.append(f"Required file not found: {name} = {path}")

    # Fuel breaks (optional) - validate paths if enabled
    if config.fuel_break is not None and config.fuel_break.enabled:
        if config.fuel_break.path and not Path(config.fuel_break.path).exists():
            errors.append(f"Fuel break file not found: fuel_break.path = {config.fuel_break.path}")

    for i, fb in enumerate(getattr(config, "fuel_breaks", []) or [], start=1):
        if getattr(fb, "enabled", False):
            if fb.path and not Path(fb.path).exists():
                errors.append(f"Fuel break file not found: fuel_breaks[{i}].path = {fb.path}")
    
    # Weather files - at least one should exist
    weather_found = False
    weather_path = config.weather.weather_path
    station_path = config.weather.station_path
    
    if weather_path and Path(weather_path).exists():
        weather_found = True
    elif Path(station_path).exists():
        weather_found = True
        if weather_path:
            warnings.append(
                f"Weather file not found ({weather_path}), "
                f"will use: {station_path}"
            )
    
    if not weather_found:
        warnings.append(
            f"No weather files found. Will use default FBP values. "
            f"Checked: {station_path}"
        )
    
    # Conditional requirements
    if config.ignition.source_type == "grid":
        if config.ignition.grid_path and not Path(config.ignition.grid_path).exists():
            errors.append(f"Ignition grid not found: {config.ignition.grid_path}")
    
    if config.ignition.source_type in ("point", "shapefile"):
        if config.ignition.point_path and not Path(config.ignition.point_path).exists():
            errors.append(f"Ignition point file not found: {config.ignition.point_path}")
    
    # Optional files
    optional = [
        ("terrain.slope_path", config.terrain.slope_path),
        ("terrain.aspect_path", config.terrain.aspect_path),
        ("ignition.ecoregion_path", config.ignition.ecoregion_path),
        ("ignition.historical_fire_path", config.ignition.historical_fire_path),
    ]
    
    for name, path in optional:
        if path is not None and not Path(path).exists():
            warnings.append(f"Optional file not found: {name} = {path}")
    
    if errors:
        raise FileNotFoundError("\n".join(errors))
    
    return warnings


def setup_logging(config: IgnacioConfig) -> None:
    """
    Configure logging based on configuration.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
    """
    level = getattr(logging, config.output.log_level)
    
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    
    if config.output.log_file:
        log_path = Path(config.output.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
