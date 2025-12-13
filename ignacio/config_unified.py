"""
Unified Configuration Module for Ignacio Fire Spread Model.

This module provides:
1. Unified config loading from YAML files
2. Separate handling of model decisions, parameters, and initial conditions
3. Calibration parameter management
4. Config validation and defaults

Configuration Structure:
------------------------
ignacio_config/
├── config.yaml           # Main config (simulation settings, paths)
├── parameters.yaml       # Tunable parameters with bounds
├── model_decisions.yaml  # SUMMA-style physics choices
└── initial_conditions.yaml  # Initial state and forcing
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import os

from .parameters import ParameterSet, ALL_PARAMETERS, get_default_parameters
from .model_decisions import ModelDecisions, ALL_DECISIONS, get_default_decisions, PRESETS


# =============================================================================
# Initial Conditions
# =============================================================================

@dataclass
class IgnitionPoint:
    """A single ignition point."""
    x: float  # meters or degrees
    y: float
    time: float = 0.0  # minutes from start
    radius: float = 30.0  # initial radius in meters
    name: str = ""


@dataclass
class InitialConditions:
    """Initial conditions for fire simulation."""
    
    # Ignition points
    ignition_points: List[IgnitionPoint] = field(default_factory=list)
    
    # Time settings
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_hours: float = 6.0
    
    # Initial moisture conditions
    ffmc: float = 85.0  # Fine Fuel Moisture Code
    dmc: float = 25.0   # Duff Moisture Code
    dc: float = 200.0   # Drought Code
    bui: float = 25.0   # Buildup Index
    
    # Initial 1hr/10hr/100hr moisture (%)
    moisture_1hr: float = 8.0
    moisture_10hr: float = 10.0
    moisture_100hr: float = 12.0
    moisture_live: float = 100.0
    
    # Ambient conditions
    initial_wind_speed: float = 10.0  # km/h
    initial_wind_direction: float = 270.0  # degrees
    initial_temperature: float = 25.0  # °C
    initial_rh: float = 30.0  # %
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InitialConditions':
        """Create from dictionary."""
        ignitions = []
        for ign in data.get('ignition_points', []):
            ignitions.append(IgnitionPoint(
                x=ign.get('x', 0),
                y=ign.get('y', 0),
                time=ign.get('time', 0),
                radius=ign.get('radius', 30),
                name=ign.get('name', ''),
            ))
        
        start = data.get('start_time')
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        
        end = data.get('end_time')
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        
        return cls(
            ignition_points=ignitions,
            start_time=start,
            end_time=end,
            duration_hours=data.get('duration_hours', 6.0),
            ffmc=data.get('ffmc', 85.0),
            dmc=data.get('dmc', 25.0),
            dc=data.get('dc', 200.0),
            bui=data.get('bui', 25.0),
            moisture_1hr=data.get('moisture_1hr', 8.0),
            moisture_10hr=data.get('moisture_10hr', 10.0),
            moisture_100hr=data.get('moisture_100hr', 12.0),
            moisture_live=data.get('moisture_live', 100.0),
            initial_wind_speed=data.get('initial_wind_speed', 10.0),
            initial_wind_direction=data.get('initial_wind_direction', 270.0),
            initial_temperature=data.get('initial_temperature', 25.0),
            initial_rh=data.get('initial_rh', 30.0),
        )
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            'ignition_points': [
                {'x': p.x, 'y': p.y, 'time': p.time, 'radius': p.radius, 'name': p.name}
                for p in self.ignition_points
            ],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_hours': self.duration_hours,
            'ffmc': self.ffmc,
            'dmc': self.dmc,
            'dc': self.dc,
            'bui': self.bui,
            'moisture_1hr': self.moisture_1hr,
            'moisture_10hr': self.moisture_10hr,
            'moisture_100hr': self.moisture_100hr,
            'moisture_live': self.moisture_live,
            'initial_wind_speed': self.initial_wind_speed,
            'initial_wind_direction': self.initial_wind_direction,
            'initial_temperature': self.initial_temperature,
            'initial_rh': self.initial_rh,
        }


# =============================================================================
# Calibration Configuration
# =============================================================================

@dataclass
class CalibrationConfig:
    """Configuration for parameter calibration."""
    
    # Parameters to calibrate
    parameters: List[str] = field(default_factory=lambda: [
        'ros_multiplier',
        'length_to_breadth',
        'ros_wind_factor',
        'ros_slope_factor',
    ])
    
    # Optimization settings
    algorithm: str = 'adam'  # adam, lbfgs, evolutionary
    max_iterations: int = 100
    learning_rate: float = 0.01
    
    # Objective function
    objective: str = 'burned_area_iou'  # burned_area_iou, perimeter_hausdorff, combined
    
    # Observation data
    observed_perimeters: Optional[str] = None  # Path to observed perimeters
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationConfig':
        return cls(
            parameters=data.get('parameters', cls.parameters),
            algorithm=data.get('algorithm', 'adam'),
            max_iterations=data.get('max_iterations', 100),
            learning_rate=data.get('learning_rate', 0.01),
            objective=data.get('objective', 'burned_area_iou'),
            observed_perimeters=data.get('observed_perimeters'),
        )
    
    def to_dict(self) -> dict:
        return {
            'parameters': self.parameters,
            'algorithm': self.algorithm,
            'max_iterations': self.max_iterations,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'observed_perimeters': self.observed_perimeters,
        }


# =============================================================================
# Output Configuration
# =============================================================================

@dataclass 
class OutputConfig:
    """Configuration for simulation output."""
    
    output_dir: str = './output'
    
    # What to save
    save_perimeters: bool = True
    save_raster: bool = True
    save_isochrones: bool = True
    save_arrival_time: bool = True
    
    # Output format
    perimeter_format: str = 'shapefile'  # shapefile, geojson, gpkg
    raster_format: str = 'geotiff'  # geotiff, netcdf
    
    # Isochrone settings
    isochrone_interval_minutes: float = 30.0
    
    # Animation
    create_animation: bool = False
    animation_fps: int = 10
    animation_format: str = 'gif'  # gif, mp4
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OutputConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# =============================================================================
# Domain Configuration
# =============================================================================

@dataclass
class DomainConfig:
    """Configuration for simulation domain."""
    
    # Grid settings
    resolution: float = 30.0  # meters
    nx: int = 100
    ny: int = 100
    
    # Coordinate reference
    crs: str = "EPSG:32611"
    
    # Vertical grid (for 3D atmosphere)
    nz: int = 15
    z_top: float = 1500.0  # meters
    
    # Domain bounds (optional, can be inferred from DEM)
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DomainConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# =============================================================================
# Input Data Paths
# =============================================================================

@dataclass
class InputPaths:
    """Paths to input data files."""
    
    dem: Optional[str] = None
    fuel_map: Optional[str] = None
    canopy_height: Optional[str] = None
    canopy_cover: Optional[str] = None
    canopy_base_height: Optional[str] = None
    canopy_bulk_density: Optional[str] = None
    weather_file: Optional[str] = None
    ignition_file: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InputPaths':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# =============================================================================
# Unified Configuration
# =============================================================================

@dataclass
class IgnacioConfig:
    """
    Unified configuration for Ignacio fire spread model.
    
    Combines:
    - Model decisions (physics choices)
    - Parameters (tunable values)
    - Initial conditions
    - Calibration settings
    - Domain and output settings
    """
    
    # Core configuration
    name: str = "ignacio_simulation"
    description: str = ""
    
    # Model configuration
    decisions: ModelDecisions = field(default_factory=get_default_decisions)
    parameters: ParameterSet = field(default_factory=get_default_parameters)
    initial_conditions: InitialConditions = field(default_factory=InitialConditions)
    
    # Calibration
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    
    # Domain and I/O
    domain: DomainConfig = field(default_factory=DomainConfig)
    inputs: InputPaths = field(default_factory=InputPaths)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    
    # Preset used (if any)
    preset: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> 'IgnacioConfig':
        """Load configuration from YAML file or directory."""
        path = Path(path)
        
        if path.is_dir():
            # Load from directory structure
            return cls._from_directory(path)
        else:
            # Load from single file
            return cls._from_file(path)
    
    @classmethod
    def _from_file(cls, path: Path) -> 'IgnacioConfig':
        """Load from single YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls._from_dict(data, base_dir=path.parent)
    
    @classmethod
    def _from_directory(cls, path: Path) -> 'IgnacioConfig':
        """Load from directory with multiple YAML files."""
        data = {}
        
        # Load main config
        main_file = path / 'config.yaml'
        if main_file.exists():
            with open(main_file, 'r') as f:
                data.update(yaml.safe_load(f) or {})
        
        # Load parameters
        param_file = path / 'parameters.yaml'
        if param_file.exists():
            with open(param_file, 'r') as f:
                param_data = yaml.safe_load(f)
                if param_data:
                    data['parameters'] = param_data.get('parameters', param_data)
        
        # Load model decisions
        decisions_file = path / 'model_decisions.yaml'
        if decisions_file.exists():
            with open(decisions_file, 'r') as f:
                dec_data = yaml.safe_load(f)
                if dec_data:
                    data['model_decisions'] = dec_data.get('model_decisions', dec_data)
        
        # Load initial conditions
        ic_file = path / 'initial_conditions.yaml'
        if ic_file.exists():
            with open(ic_file, 'r') as f:
                ic_data = yaml.safe_load(f)
                if ic_data:
                    data['initial_conditions'] = ic_data
        
        return cls._from_dict(data, base_dir=path)
    
    @classmethod
    def _from_dict(cls, data: dict, base_dir: Path = None) -> 'IgnacioConfig':
        """Create from dictionary."""
        
        # Handle preset
        preset = data.get('preset')
        if preset and preset in PRESETS:
            decisions = PRESETS[preset]
            # Override with any explicit decisions
            if 'model_decisions' in data:
                for k, v in data['model_decisions'].items():
                    decisions.set(k, v)
        else:
            decisions = ModelDecisions(choices=data.get('model_decisions', {}))
        
        # Load parameters
        parameters = ParameterSet(values=data.get('parameters', {}))
        
        # Load initial conditions
        ic_data = data.get('initial_conditions', {})
        initial_conditions = InitialConditions.from_dict(ic_data)
        
        # Load other configs
        calibration = CalibrationConfig.from_dict(data.get('calibration', {}))
        domain = DomainConfig.from_dict(data.get('domain', {}))
        outputs = OutputConfig.from_dict(data.get('outputs', data.get('output', {})))
        
        # Handle input paths
        inputs_data = data.get('inputs', data.get('input_data', {}))
        if base_dir:
            # Resolve relative paths
            for key in ['dem', 'fuel_map', 'weather_file', 'ignition_file']:
                if key in inputs_data and inputs_data[key]:
                    p = Path(inputs_data[key])
                    if not p.is_absolute():
                        inputs_data[key] = str(base_dir / p)
        inputs = InputPaths.from_dict(inputs_data)
        
        return cls(
            name=data.get('name', 'ignacio_simulation'),
            description=data.get('description', ''),
            decisions=decisions,
            parameters=parameters,
            initial_conditions=initial_conditions,
            calibration=calibration,
            domain=domain,
            inputs=inputs,
            outputs=outputs,
            preset=preset,
        )
    
    def to_yaml(self, path: str, split_files: bool = False) -> None:
        """Save configuration to YAML."""
        path = Path(path)
        
        if split_files:
            # Save to directory with separate files
            path.mkdir(parents=True, exist_ok=True)
            
            # Main config
            main_data = {
                'name': self.name,
                'description': self.description,
                'preset': self.preset,
                'domain': self.domain.to_dict(),
                'inputs': self.inputs.to_dict(),
                'outputs': self.outputs.to_dict(),
            }
            with open(path / 'config.yaml', 'w') as f:
                yaml.dump(main_data, f, default_flow_style=False)
            
            # Parameters
            with open(path / 'parameters.yaml', 'w') as f:
                yaml.dump({'parameters': self.parameters.to_dict()}, f, default_flow_style=False)
            
            # Model decisions
            with open(path / 'model_decisions.yaml', 'w') as f:
                yaml.dump({'model_decisions': self.decisions.to_dict()}, f, default_flow_style=False)
            
            # Initial conditions
            with open(path / 'initial_conditions.yaml', 'w') as f:
                yaml.dump(self.initial_conditions.to_dict(), f, default_flow_style=False)
            
            # Calibration
            with open(path / 'calibration.yaml', 'w') as f:
                yaml.dump({'calibration': self.calibration.to_dict()}, f, default_flow_style=False)
        else:
            # Save to single file
            data = {
                'name': self.name,
                'description': self.description,
                'preset': self.preset,
                'model_decisions': self.decisions.to_dict(),
                'parameters': self.parameters.to_dict(),
                'initial_conditions': self.initial_conditions.to_dict(),
                'calibration': self.calibration.to_dict(),
                'domain': self.domain.to_dict(),
                'inputs': self.inputs.to_dict(),
                'outputs': self.outputs.to_dict(),
            }
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_calibration_bounds(self) -> tuple:
        """Get parameter bounds for calibration."""
        return self.parameters.get_bounds(self.calibration.parameters)
    
    def get_calibration_defaults(self) -> list:
        """Get default values for calibration parameters."""
        return self.parameters.to_array(self.calibration.parameters)
    
    def summary(self) -> str:
        """Generate configuration summary."""
        lines = [
            f"Ignacio Configuration: {self.name}",
            "=" * 60,
            f"Description: {self.description or 'None'}",
            f"Preset: {self.preset or 'custom'}",
            "",
            "Model Decisions:",
            "-" * 40,
        ]
        
        for name, choice in self.decisions.choices.items():
            lines.append(f"  {name}: {choice}")
        
        lines.extend([
            "",
            "Key Parameters:",
            "-" * 40,
        ])
        
        key_params = ['ros_multiplier', 'length_to_breadth', 'feedback_strength']
        for name in key_params:
            lines.append(f"  {name}: {self.parameters.get(name)}")
        
        lines.extend([
            "",
            "Initial Conditions:",
            "-" * 40,
            f"  Ignition points: {len(self.initial_conditions.ignition_points)}",
            f"  Duration: {self.initial_conditions.duration_hours} hours",
            f"  FFMC: {self.initial_conditions.ffmc}",
            f"  Wind: {self.initial_conditions.initial_wind_speed} km/h @ {self.initial_conditions.initial_wind_direction}°",
            "",
            "Calibration:",
            "-" * 40,
            f"  Parameters: {', '.join(self.calibration.parameters)}",
            f"  Algorithm: {self.calibration.algorithm}",
        ])
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_config() -> IgnacioConfig:
    """Create a configuration with all defaults."""
    return IgnacioConfig()


def create_config_from_preset(preset: str) -> IgnacioConfig:
    """Create configuration from a preset."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    
    return IgnacioConfig(
        decisions=PRESETS[preset],
        preset=preset,
    )


def export_config_template(path: str) -> None:
    """Export a template configuration file."""
    config = create_default_config()
    config.name = "my_simulation"
    config.description = "Template configuration - customize as needed"
    config.to_yaml(path)


def load_config(path: str) -> IgnacioConfig:
    """Load configuration from file or directory."""
    return IgnacioConfig.from_yaml(path)
