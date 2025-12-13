"""
Parameter Definitions for Ignacio Fire Spread Model.

This module defines all tunable parameters with:
- Default values
- Physical bounds (min/max)
- Units and descriptions
- Calibration metadata

Follows SUMMA-style parameter organization for clarity and maintainability.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
import yaml


# =============================================================================
# Parameter Definition Classes
# =============================================================================

@dataclass
class ParameterDef:
    """Definition of a single parameter."""
    name: str
    default: float
    min_val: float
    max_val: float
    units: str
    description: str
    calibratable: bool = True
    category: str = "general"
    
    def validate(self, value: float) -> float:
        """Validate and clip value to bounds."""
        if value < self.min_val:
            return self.min_val
        if value > self.max_val:
            return self.max_val
        return value
    
    def to_dict(self) -> dict:
        return {
            'default': self.default,
            'min': self.min_val,
            'max': self.max_val,
            'units': self.units,
            'description': self.description,
            'calibratable': self.calibratable,
            'category': self.category,
        }


# =============================================================================
# Fire Behavior Parameters
# =============================================================================

FIRE_BEHAVIOR_PARAMS = {
    # Rate of Spread
    'ros_multiplier': ParameterDef(
        name='ros_multiplier',
        default=1.0,
        min_val=0.1,
        max_val=10.0,
        units='-',
        description='Global multiplier for rate of spread',
        category='fire_behavior',
    ),
    'ros_wind_factor': ParameterDef(
        name='ros_wind_factor',
        default=1.0,
        min_val=0.0,
        max_val=5.0,
        units='-',
        description='Wind effect multiplier on ROS',
        category='fire_behavior',
    ),
    'ros_slope_factor': ParameterDef(
        name='ros_slope_factor',
        default=1.0,
        min_val=0.0,
        max_val=5.0,
        units='-',
        description='Slope effect multiplier on ROS',
        category='fire_behavior',
    ),
    
    # Fire Shape
    'length_to_breadth': ParameterDef(
        name='length_to_breadth',
        default=2.0,
        min_val=1.0,
        max_val=8.0,
        units='-',
        description='Length-to-breadth ratio of fire ellipse',
        category='fire_behavior',
    ),
    'head_to_back_ratio': ParameterDef(
        name='head_to_back_ratio',
        default=5.0,
        min_val=1.0,
        max_val=20.0,
        units='-',
        description='Ratio of head fire to backing fire ROS',
        category='fire_behavior',
    ),
    
    # Ignition
    'initial_fire_radius': ParameterDef(
        name='initial_fire_radius',
        default=30.0,
        min_val=1.0,
        max_val=500.0,
        units='m',
        description='Initial fire radius at ignition',
        category='fire_behavior',
    ),
}

# =============================================================================
# Fuel Moisture Parameters
# =============================================================================

MOISTURE_PARAMS = {
    # Fine Fuel Moisture
    'ffmc_coefficient': ParameterDef(
        name='ffmc_coefficient',
        default=1.0,
        min_val=0.5,
        max_val=2.0,
        units='-',
        description='FFMC effect coefficient',
        category='moisture',
    ),
    
    # Moisture Time Lags
    'moisture_1hr_lag': ParameterDef(
        name='moisture_1hr_lag',
        default=1.0,
        min_val=0.5,
        max_val=3.0,
        units='hr',
        description='1-hour fuel moisture time lag',
        category='moisture',
    ),
    'moisture_10hr_lag': ParameterDef(
        name='moisture_10hr_lag',
        default=10.0,
        min_val=5.0,
        max_val=20.0,
        units='hr',
        description='10-hour fuel moisture time lag',
        category='moisture',
    ),
    'moisture_100hr_lag': ParameterDef(
        name='moisture_100hr_lag',
        default=100.0,
        min_val=50.0,
        max_val=200.0,
        units='hr',
        description='100-hour fuel moisture time lag',
        category='moisture',
    ),
    
    # Equilibrium Moisture
    'emc_high_rh_threshold': ParameterDef(
        name='emc_high_rh_threshold',
        default=0.5,
        min_val=0.3,
        max_val=0.7,
        units='-',
        description='RH threshold for high/low EMC equations',
        category='moisture',
    ),
}

# =============================================================================
# Crown Fire Parameters
# =============================================================================

CROWN_FIRE_PARAMS = {
    'cbh_default': ParameterDef(
        name='cbh_default',
        default=3.0,
        min_val=0.5,
        max_val=20.0,
        units='m',
        description='Default canopy base height',
        category='crown_fire',
    ),
    'cbd_default': ParameterDef(
        name='cbd_default',
        default=0.1,
        min_val=0.01,
        max_val=0.5,
        units='kg/m³',
        description='Default canopy bulk density',
        category='crown_fire',
    ),
    'foliar_moisture': ParameterDef(
        name='foliar_moisture',
        default=100.0,
        min_val=70.0,
        max_val=150.0,
        units='%',
        description='Foliar moisture content',
        category='crown_fire',
    ),
    'critical_surface_intensity': ParameterDef(
        name='critical_surface_intensity',
        default=300.0,
        min_val=100.0,
        max_val=1000.0,
        units='kW/m',
        description='Surface intensity threshold for crown initiation',
        category='crown_fire',
    ),
    'crown_fraction_burned_threshold': ParameterDef(
        name='crown_fraction_burned_threshold',
        default=0.9,
        min_val=0.5,
        max_val=1.0,
        units='-',
        description='CFB threshold for active crown fire',
        category='crown_fire',
    ),
}

# =============================================================================
# Spotting Parameters
# =============================================================================

SPOTTING_PARAMS = {
    'spot_probability': ParameterDef(
        name='spot_probability',
        default=0.1,
        min_val=0.0,
        max_val=1.0,
        units='-',
        description='Base probability of spot fire ignition',
        category='spotting',
    ),
    'max_spot_distance': ParameterDef(
        name='max_spot_distance',
        default=2000.0,
        min_val=100.0,
        max_val=10000.0,
        units='m',
        description='Maximum spotting distance',
        category='spotting',
    ),
    'firebrand_height_factor': ParameterDef(
        name='firebrand_height_factor',
        default=1.0,
        min_val=0.5,
        max_val=3.0,
        units='-',
        description='Firebrand lofting height multiplier',
        category='spotting',
    ),
    'ember_density': ParameterDef(
        name='ember_density',
        default=300.0,
        min_val=100.0,
        max_val=600.0,
        units='kg/m³',
        description='Ember particle density',
        category='spotting',
    ),
}

# =============================================================================
# Terrain/Wind Parameters
# =============================================================================

TERRAIN_WIND_PARAMS = {
    'terrain_wind_weight': ParameterDef(
        name='terrain_wind_weight',
        default=0.3,
        min_val=0.0,
        max_val=1.0,
        units='-',
        description='Weight of terrain-modified wind vs background',
        category='terrain',
    ),
    'valley_wind_enhancement': ParameterDef(
        name='valley_wind_enhancement',
        default=1.5,
        min_val=1.0,
        max_val=3.0,
        units='-',
        description='Valley channeling wind enhancement',
        category='terrain',
    ),
    'ridge_wind_acceleration': ParameterDef(
        name='ridge_wind_acceleration',
        default=1.3,
        min_val=1.0,
        max_val=2.0,
        units='-',
        description='Ridge wind speed-up factor',
        category='terrain',
    ),
}

# =============================================================================
# Atmosphere Coupling Parameters
# =============================================================================

ATMOSPHERE_PARAMS = {
    'feedback_strength': ParameterDef(
        name='feedback_strength',
        default=0.5,
        min_val=0.0,
        max_val=1.0,
        units='-',
        description='Fire-atmosphere coupling strength (0=off, 1=full)',
        category='atmosphere',
    ),
    'plume_injection_depth': ParameterDef(
        name='plume_injection_depth',
        default=100.0,
        min_val=50.0,
        max_val=500.0,
        units='m',
        description='Depth of fire heat injection',
        category='atmosphere',
    ),
    'eddy_viscosity_h': ParameterDef(
        name='eddy_viscosity_h',
        default=50.0,
        min_val=10.0,
        max_val=200.0,
        units='m²/s',
        description='Horizontal eddy viscosity',
        category='atmosphere',
    ),
    'eddy_viscosity_v': ParameterDef(
        name='eddy_viscosity_v',
        default=10.0,
        min_val=1.0,
        max_val=50.0,
        units='m²/s',
        description='Vertical eddy viscosity',
        category='atmosphere',
    ),
}

# =============================================================================
# Eruptive Fire Parameters
# =============================================================================

ERUPTIVE_PARAMS = {
    'canyon_slope_threshold': ParameterDef(
        name='canyon_slope_threshold',
        default=0.3,
        min_val=0.1,
        max_val=0.6,
        units='-',
        description='Minimum slope for canyon detection',
        category='eruptive',
    ),
    'eruptive_ros_multiplier': ParameterDef(
        name='eruptive_ros_multiplier',
        default=3.0,
        min_val=1.5,
        max_val=10.0,
        units='-',
        description='ROS multiplier during eruptive conditions',
        category='eruptive',
    ),
    'flame_attachment_threshold': ParameterDef(
        name='flame_attachment_threshold',
        default=0.7,
        min_val=0.3,
        max_val=1.0,
        units='-',
        description='Probability threshold for flame attachment',
        category='eruptive',
    ),
}

# =============================================================================
# Phenology Parameters
# =============================================================================

PHENOLOGY_PARAMS = {
    'base_gdd_threshold': ParameterDef(
        name='base_gdd_threshold',
        default=10.0,
        min_val=5.0,
        max_val=15.0,
        units='°C',
        description='Base temperature for GDD accumulation',
        category='phenology',
    ),
    'curing_rate': ParameterDef(
        name='curing_rate',
        default=0.02,
        min_val=0.005,
        max_val=0.1,
        units='1/day',
        description='Daily grass curing rate',
        category='phenology',
    ),
    'elevation_delay_factor': ParameterDef(
        name='elevation_delay_factor',
        default=6.5,
        min_val=4.0,
        max_val=10.0,
        units='days/100m',
        description='Phenology delay per 100m elevation',
        category='phenology',
    ),
}

# =============================================================================
# Numerical Parameters
# =============================================================================

NUMERICAL_PARAMS = {
    'dt_fire': ParameterDef(
        name='dt_fire',
        default=1.0,
        min_val=0.1,
        max_val=10.0,
        units='min',
        description='Fire model time step',
        calibratable=False,
        category='numerical',
    ),
    'dt_atmosphere': ParameterDef(
        name='dt_atmosphere',
        default=2.0,
        min_val=0.5,
        max_val=10.0,
        units='s',
        description='Atmosphere model time step',
        calibratable=False,
        category='numerical',
    ),
    'cfl_number': ParameterDef(
        name='cfl_number',
        default=0.5,
        min_val=0.1,
        max_val=0.9,
        units='-',
        description='CFL number for stability',
        calibratable=False,
        category='numerical',
    ),
    'pressure_iterations': ParameterDef(
        name='pressure_iterations',
        default=50,
        min_val=10,
        max_val=200,
        units='-',
        description='Pressure solver iterations',
        calibratable=False,
        category='numerical',
    ),
}


# =============================================================================
# Aggregate All Parameters
# =============================================================================

ALL_PARAMETERS: Dict[str, ParameterDef] = {
    **FIRE_BEHAVIOR_PARAMS,
    **MOISTURE_PARAMS,
    **CROWN_FIRE_PARAMS,
    **SPOTTING_PARAMS,
    **TERRAIN_WIND_PARAMS,
    **ATMOSPHERE_PARAMS,
    **ERUPTIVE_PARAMS,
    **PHENOLOGY_PARAMS,
    **NUMERICAL_PARAMS,
}

# Parameters suitable for calibration
CALIBRATABLE_PARAMETERS = {
    name: param for name, param in ALL_PARAMETERS.items()
    if param.calibratable
}


# =============================================================================
# Parameter Set Class
# =============================================================================

@dataclass
class ParameterSet:
    """A set of parameter values with validation."""
    
    values: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Fill in defaults for missing parameters
        for name, param_def in ALL_PARAMETERS.items():
            if name not in self.values:
                self.values[name] = param_def.default
    
    def get(self, name: str) -> float:
        """Get parameter value."""
        if name in self.values:
            return self.values[name]
        if name in ALL_PARAMETERS:
            return ALL_PARAMETERS[name].default
        raise KeyError(f"Unknown parameter: {name}")
    
    def set(self, name: str, value: float, validate: bool = True) -> None:
        """Set parameter value with optional validation."""
        if name not in ALL_PARAMETERS:
            raise KeyError(f"Unknown parameter: {name}")
        if validate:
            value = ALL_PARAMETERS[name].validate(value)
        self.values[name] = value
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return dict(self.values)
    
    def to_array(self, param_names: List[str]) -> list:
        """Export selected parameters as array (for optimization)."""
        return [self.get(name) for name in param_names]
    
    def from_array(self, param_names: List[str], values: list) -> None:
        """Import parameters from array."""
        for name, value in zip(param_names, values):
            self.set(name, value)
    
    def get_bounds(self, param_names: List[str]) -> tuple:
        """Get bounds for optimization."""
        lower = [ALL_PARAMETERS[name].min_val for name in param_names]
        upper = [ALL_PARAMETERS[name].max_val for name in param_names]
        return (lower, upper)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ParameterSet':
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(values=data.get('parameters', {}))
    
    def to_yaml(self, path: str) -> None:
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump({'parameters': self.values}, f, default_flow_style=False)


# =============================================================================
# Default Parameter Set
# =============================================================================

def get_default_parameters() -> ParameterSet:
    """Get parameter set with all defaults."""
    return ParameterSet()


def get_parameter_info() -> Dict[str, dict]:
    """Get parameter metadata for documentation."""
    return {name: param.to_dict() for name, param in ALL_PARAMETERS.items()}


def list_calibratable_parameters() -> List[str]:
    """List all parameters suitable for calibration."""
    return list(CALIBRATABLE_PARAMETERS.keys())


def export_parameter_template(path: str) -> None:
    """Export a template parameter file with all defaults and documentation."""
    output = {
        'parameters': {},
        '_documentation': {}
    }
    
    for name, param in ALL_PARAMETERS.items():
        output['parameters'][name] = param.default
        output['_documentation'][name] = {
            'description': param.description,
            'units': param.units,
            'bounds': [param.min_val, param.max_val],
            'calibratable': param.calibratable,
            'category': param.category,
        }
    
    with open(path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
