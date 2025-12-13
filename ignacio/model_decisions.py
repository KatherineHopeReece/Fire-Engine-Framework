"""
Model Decisions for Ignacio Fire Spread Model.

This module implements SUMMA-style model decisions, allowing users to
select between different physical process representations.

Each decision specifies:
- Available options
- Default choice
- Description of each option
- Physical implications

This provides a clean way to configure model complexity and physics.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml


# =============================================================================
# Decision Option Classes
# =============================================================================

@dataclass
class DecisionOption:
    """A single option for a model decision."""
    name: str
    description: str
    computational_cost: str = "low"  # low, medium, high
    recommended_for: str = ""


@dataclass
class ModelDecision:
    """A model decision with multiple options."""
    name: str
    description: str
    category: str
    options: Dict[str, DecisionOption]
    default: str
    
    def validate(self, choice: str) -> str:
        """Validate and return choice, or default if invalid."""
        if choice in self.options:
            return choice
        print(f"Warning: Invalid choice '{choice}' for {self.name}, using default '{self.default}'")
        return self.default
    
    def to_dict(self) -> dict:
        return {
            'description': self.description,
            'category': self.category,
            'options': {k: v.description for k, v in self.options.items()},
            'default': self.default,
        }


# =============================================================================
# Fire Spread Method Decisions
# =============================================================================

SPREAD_METHOD = ModelDecision(
    name='spread_method',
    description='Primary fire spread algorithm',
    category='fire_spread',
    default='levelset',
    options={
        'levelset': DecisionOption(
            name='levelset',
            description='Level-set method with signed distance function',
            computational_cost='medium',
            recommended_for='Complex fire shapes, merging fires',
        ),
        'cellular_automata': DecisionOption(
            name='cellular_automata',
            description='Cell-based spread with probabilistic ignition',
            computational_cost='low',
            recommended_for='Simple scenarios, fast simulation',
        ),
        'vector': DecisionOption(
            name='vector',
            description='Vector polygon propagation (Prometheus-style)',
            computational_cost='medium',
            recommended_for='Operational use, Prometheus compatibility',
        ),
        'hybrid': DecisionOption(
            name='hybrid',
            description='Level-set core with vector output',
            computational_cost='medium',
            recommended_for='Balanced accuracy and compatibility',
        ),
    }
)

FIRE_SHAPE = ModelDecision(
    name='fire_shape',
    description='Fire perimeter shape model',
    category='fire_spread',
    default='ellipse',
    options={
        'ellipse': DecisionOption(
            name='ellipse',
            description='Elliptical fire shape (Richards 1990)',
            computational_cost='low',
            recommended_for='Wind-driven fires',
        ),
        'double_ellipse': DecisionOption(
            name='double_ellipse',
            description='Double ellipse for asymmetric spread',
            computational_cost='low',
            recommended_for='Variable fuel conditions',
        ),
        'lemniscate': DecisionOption(
            name='lemniscate',
            description='Figure-8 shape for strong crosswind',
            computational_cost='low',
            recommended_for='Strong crosswind conditions',
        ),
    }
)

VECTOR_TOPOLOGY = ModelDecision(
    name='vector_topology',
    description='Vector polygon topology handling',
    category='fire_spread',
    default='shapely_clip',
    options={
        'none': DecisionOption(
            name='none',
            description='No topology correction (may self-intersect)',
            computational_cost='low',
            recommended_for='Fast simulation where topology is not critical',
        ),
        'shapely_clip': DecisionOption(
            name='shapely_clip',
            description='Shapely-based polygon clipping and repair',
            computational_cost='low',
            recommended_for='Prometheus compatibility, clean outputs',
        ),
        'buffer_zero': DecisionOption(
            name='buffer_zero',
            description='Buffer(0) trick for self-intersection repair',
            computational_cost='low',
            recommended_for='Quick fix for minor issues',
        ),
        'vatti': DecisionOption(
            name='vatti',
            description='Vatti clipping algorithm for complex polygons',
            computational_cost='medium',
            recommended_for='Complex overlapping perimeters',
        ),
    }
)


# =============================================================================
# Rate of Spread Decisions
# =============================================================================

ROS_MODEL = ModelDecision(
    name='ros_model',
    description='Rate of spread calculation model',
    category='fire_behavior',
    default='fbp',
    options={
        'fbp': DecisionOption(
            name='fbp',
            description='Canadian Forest Fire Behavior Prediction System',
            computational_cost='low',
            recommended_for='Canadian boreal forests',
        ),
        'rothermel': DecisionOption(
            name='rothermel',
            description='Rothermel 1972 surface fire model',
            computational_cost='low',
            recommended_for='US fuel models, grasslands',
        ),
        'behave': DecisionOption(
            name='behave',
            description='BEHAVE/BehavePlus implementation',
            computational_cost='low',
            recommended_for='US wildfire management',
        ),
        'combined': DecisionOption(
            name='combined',
            description='Ensemble of FBP and Rothermel',
            computational_cost='medium',
            recommended_for='Research, uncertainty quantification',
        ),
    }
)

SLOPE_EFFECT = ModelDecision(
    name='slope_effect',
    description='Slope effect on rate of spread',
    category='fire_behavior',
    default='exponential',
    options={
        'exponential': DecisionOption(
            name='exponential',
            description='Exponential slope factor (standard)',
            computational_cost='low',
        ),
        'linear': DecisionOption(
            name='linear',
            description='Linear slope correction',
            computational_cost='low',
        ),
        'mcarthur': DecisionOption(
            name='mcarthur',
            description='McArthur slope correction',
            computational_cost='low',
        ),
        'nelson': DecisionOption(
            name='nelson',
            description='Nelson 2000 slope model',
            computational_cost='low',
        ),
    }
)

WIND_EFFECT = ModelDecision(
    name='wind_effect',
    description='Wind effect on rate of spread',
    category='fire_behavior',
    default='beer_law',
    options={
        'beer_law': DecisionOption(
            name='beer_law',
            description="Beer's law exponential (FBP)",
            computational_cost='low',
        ),
        'power_law': DecisionOption(
            name='power_law',
            description='Power law relationship',
            computational_cost='low',
        ),
        'rothermel_wind': DecisionOption(
            name='rothermel_wind',
            description='Rothermel wind factor',
            computational_cost='low',
        ),
    }
)


# =============================================================================
# Crown Fire Decisions
# =============================================================================

CROWN_FIRE_MODEL = ModelDecision(
    name='crown_fire_model',
    description='Crown fire initiation and spread model',
    category='crown_fire',
    default='van_wagner',
    options={
        'none': DecisionOption(
            name='none',
            description='No crown fire (surface only)',
            computational_cost='low',
        ),
        'van_wagner': DecisionOption(
            name='van_wagner',
            description='Van Wagner 1977 crown fire model',
            computational_cost='low',
            recommended_for='Standard crown fire prediction',
        ),
        'cruz': DecisionOption(
            name='cruz',
            description='Cruz et al. 2005 crown fire model',
            computational_cost='low',
            recommended_for='Updated crown fire physics',
        ),
        'scott_reinhardt': DecisionOption(
            name='scott_reinhardt',
            description='Scott & Reinhardt 2001 crown fire linkage',
            computational_cost='medium',
        ),
    }
)


# =============================================================================
# Fuel Moisture Decisions
# =============================================================================

FUEL_MOISTURE_MODEL = ModelDecision(
    name='fuel_moisture_model',
    description='Dead fuel moisture calculation method',
    category='moisture',
    default='time_lag',
    options={
        'static': DecisionOption(
            name='static',
            description='Static moisture from input (no dynamics)',
            computational_cost='low',
        ),
        'time_lag': DecisionOption(
            name='time_lag',
            description='Time-lag exponential approach to EMC',
            computational_cost='low',
            recommended_for='Standard operational use',
        ),
        'nelson': DecisionOption(
            name='nelson',
            description='Nelson stick moisture model',
            computational_cost='medium',
            recommended_for='Research applications',
        ),
        'nwcg': DecisionOption(
            name='nwcg',
            description='NWCG tables for 1/10/100hr fuels',
            computational_cost='low',
        ),
    }
)

LIVE_MOISTURE_MODEL = ModelDecision(
    name='live_moisture_model',
    description='Live fuel moisture estimation',
    category='moisture',
    default='phenology',
    options={
        'static': DecisionOption(
            name='static',
            description='Static live moisture from input',
            computational_cost='low',
        ),
        'phenology': DecisionOption(
            name='phenology',
            description='Phenology-based seasonal variation',
            computational_cost='low',
            recommended_for='Seasonal fire behavior',
        ),
        'ndvi': DecisionOption(
            name='ndvi',
            description='NDVI-derived live fuel moisture',
            computational_cost='medium',
            recommended_for='Real-time estimation',
        ),
        'gdd': DecisionOption(
            name='gdd',
            description='Growing degree day model',
            computational_cost='low',
        ),
    }
)


# =============================================================================
# Terrain Wind Decisions
# =============================================================================

TERRAIN_WIND_MODEL = ModelDecision(
    name='terrain_wind_model',
    description='Terrain effect on wind field',
    category='terrain',
    default='mass_consistent',
    options={
        'none': DecisionOption(
            name='none',
            description='No terrain modification (uniform wind)',
            computational_cost='low',
        ),
        'simple': DecisionOption(
            name='simple',
            description='Simple upslope/downslope modification',
            computational_cost='low',
        ),
        'mass_consistent': DecisionOption(
            name='mass_consistent',
            description='Mass-consistent wind solver',
            computational_cost='medium',
            recommended_for='Complex terrain',
        ),
        'windninja': DecisionOption(
            name='windninja',
            description='WindNinja-style solver',
            computational_cost='high',
        ),
    }
)


# =============================================================================
# Atmosphere Coupling Decisions
# =============================================================================

ATMOSPHERE_COUPLING = ModelDecision(
    name='atmosphere_coupling',
    description='Fire-atmosphere coupling mode',
    category='atmosphere',
    default='none',
    options={
        'none': DecisionOption(
            name='none',
            description='No atmosphere coupling (prescribed winds)',
            computational_cost='low',
            recommended_for='Fast operational simulation',
        ),
        'simple': DecisionOption(
            name='simple',
            description='Simple indraft parameterization',
            computational_cost='low',
            recommended_for='Moderate coupling effects',
        ),
        'full_3d': DecisionOption(
            name='full_3d',
            description='Full 3D Navier-Stokes (WRF-SFIRE style)',
            computational_cost='high',
            recommended_for='Research, extreme fire behavior',
        ),
    }
)


# =============================================================================
# Spotting Decisions
# =============================================================================

SPOTTING_MODEL = ModelDecision(
    name='spotting_model',
    description='Ember transport and spotting model',
    category='spotting',
    default='none',
    options={
        'none': DecisionOption(
            name='none',
            description='No spotting',
            computational_cost='low',
        ),
        'albini': DecisionOption(
            name='albini',
            description='Albini 1979 spotting model',
            computational_cost='low',
            recommended_for='Basic spotting prediction',
        ),
        'sardoy': DecisionOption(
            name='sardoy',
            description='Sardoy et al. firebrand transport',
            computational_cost='medium',
        ),
        'lagrangian': DecisionOption(
            name='lagrangian',
            description='Full Lagrangian particle tracking',
            computational_cost='high',
            recommended_for='Research, detailed spotting',
        ),
    }
)


# =============================================================================
# Advanced Physics Decisions
# =============================================================================

ERUPTIVE_FIRE = ModelDecision(
    name='eruptive_fire',
    description='Eruptive fire behavior (blowup) modeling',
    category='advanced',
    default='none',
    options={
        'none': DecisionOption(
            name='none',
            description='No eruptive fire modeling',
            computational_cost='low',
        ),
        'viegas': DecisionOption(
            name='viegas',
            description='Viegas canyon effect model',
            computational_cost='low',
            recommended_for='Canyon/slope fires',
        ),
        'dold': DecisionOption(
            name='dold',
            description='Dold-Zinoviev eruptive model',
            computational_cost='medium',
        ),
    }
)

SMOKE_TRANSPORT = ModelDecision(
    name='smoke_transport',
    description='Smoke dispersion modeling',
    category='advanced',
    default='none',
    options={
        'none': DecisionOption(
            name='none',
            description='No smoke transport',
            computational_cost='low',
        ),
        'gaussian': DecisionOption(
            name='gaussian',
            description='Gaussian plume dispersion',
            computational_cost='low',
        ),
        'advection_diffusion': DecisionOption(
            name='advection_diffusion',
            description='Advection-diffusion transport',
            computational_cost='medium',
            recommended_for='Smoke impact assessment',
        ),
    }
)


# =============================================================================
# Aggregate All Decisions
# =============================================================================

ALL_DECISIONS: Dict[str, ModelDecision] = {
    # Fire Spread
    'spread_method': SPREAD_METHOD,
    'fire_shape': FIRE_SHAPE,
    'vector_topology': VECTOR_TOPOLOGY,
    
    # Rate of Spread
    'ros_model': ROS_MODEL,
    'slope_effect': SLOPE_EFFECT,
    'wind_effect': WIND_EFFECT,
    
    # Crown Fire
    'crown_fire_model': CROWN_FIRE_MODEL,
    
    # Moisture
    'fuel_moisture_model': FUEL_MOISTURE_MODEL,
    'live_moisture_model': LIVE_MOISTURE_MODEL,
    
    # Terrain
    'terrain_wind_model': TERRAIN_WIND_MODEL,
    
    # Atmosphere
    'atmosphere_coupling': ATMOSPHERE_COUPLING,
    
    # Spotting
    'spotting_model': SPOTTING_MODEL,
    
    # Advanced
    'eruptive_fire': ERUPTIVE_FIRE,
    'smoke_transport': SMOKE_TRANSPORT,
}


# =============================================================================
# Decision Set Class
# =============================================================================

@dataclass
class ModelDecisions:
    """A set of model decision choices."""
    
    choices: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        # Fill in defaults for missing decisions
        for name, decision in ALL_DECISIONS.items():
            if name not in self.choices:
                self.choices[name] = decision.default
            else:
                # Validate existing choice
                self.choices[name] = decision.validate(self.choices[name])
    
    def get(self, name: str) -> str:
        """Get decision choice."""
        if name in self.choices:
            return self.choices[name]
        if name in ALL_DECISIONS:
            return ALL_DECISIONS[name].default
        raise KeyError(f"Unknown decision: {name}")
    
    def set(self, name: str, choice: str) -> None:
        """Set decision choice."""
        if name not in ALL_DECISIONS:
            raise KeyError(f"Unknown decision: {name}")
        self.choices[name] = ALL_DECISIONS[name].validate(choice)
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return dict(self.choices)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ModelDecisions':
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(choices=data.get('model_decisions', {}))
    
    def to_yaml(self, path: str) -> None:
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump({'model_decisions': self.choices}, f, default_flow_style=False)
    
    def summary(self) -> str:
        """Generate human-readable summary of decisions."""
        lines = ["Model Decisions:", "="*50]
        for name, choice in sorted(self.choices.items()):
            decision = ALL_DECISIONS.get(name)
            if decision:
                desc = decision.options[choice].description
                lines.append(f"  {name}: {choice}")
                lines.append(f"    → {desc}")
        return "\n".join(lines)


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS = {
    'fast': ModelDecisions(choices={
        'spread_method': 'cellular_automata',
        'ros_model': 'fbp',
        'crown_fire_model': 'none',
        'fuel_moisture_model': 'static',
        'terrain_wind_model': 'none',
        'atmosphere_coupling': 'none',
        'spotting_model': 'none',
        'eruptive_fire': 'none',
        'smoke_transport': 'none',
    }),
    
    'operational': ModelDecisions(choices={
        'spread_method': 'levelset',
        'vector_topology': 'shapely_clip',
        'ros_model': 'fbp',
        'crown_fire_model': 'van_wagner',
        'fuel_moisture_model': 'time_lag',
        'terrain_wind_model': 'mass_consistent',
        'atmosphere_coupling': 'none',
        'spotting_model': 'albini',
        'eruptive_fire': 'none',
        'smoke_transport': 'none',
    }),
    
    'prometheus_compatible': ModelDecisions(choices={
        'spread_method': 'vector',
        'vector_topology': 'shapely_clip',
        'fire_shape': 'ellipse',
        'ros_model': 'fbp',
        'crown_fire_model': 'van_wagner',
        'fuel_moisture_model': 'static',
        'terrain_wind_model': 'simple',
        'atmosphere_coupling': 'none',
        'spotting_model': 'none',
    }),
    
    'research': ModelDecisions(choices={
        'spread_method': 'levelset',
        'vector_topology': 'shapely_clip',
        'ros_model': 'combined',
        'crown_fire_model': 'cruz',
        'fuel_moisture_model': 'nelson',
        'live_moisture_model': 'phenology',
        'terrain_wind_model': 'mass_consistent',
        'atmosphere_coupling': 'simple',
        'spotting_model': 'sardoy',
        'eruptive_fire': 'viegas',
        'smoke_transport': 'advection_diffusion',
    }),
    
    'coupled': ModelDecisions(choices={
        'spread_method': 'levelset',
        'ros_model': 'fbp',
        'crown_fire_model': 'van_wagner',
        'fuel_moisture_model': 'time_lag',
        'terrain_wind_model': 'mass_consistent',
        'atmosphere_coupling': 'full_3d',
        'spotting_model': 'lagrangian',
        'eruptive_fire': 'viegas',
        'smoke_transport': 'advection_diffusion',
    }),
}


def get_preset(name: str) -> ModelDecisions:
    """Get a preset configuration."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def get_default_decisions() -> ModelDecisions:
    """Get decision set with all defaults."""
    return ModelDecisions()


def get_decision_info() -> Dict[str, dict]:
    """Get decision metadata for documentation."""
    return {name: decision.to_dict() for name, decision in ALL_DECISIONS.items()}


def export_decision_template(path: str) -> None:
    """Export a template decision file with documentation."""
    output = {
        'model_decisions': {},
        '_documentation': {}
    }
    
    for name, decision in ALL_DECISIONS.items():
        output['model_decisions'][name] = decision.default
        output['_documentation'][name] = {
            'description': decision.description,
            'category': decision.category,
            'options': {k: v.description for k, v in decision.options.items()},
        }
    
    with open(path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
