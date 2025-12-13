"""
Ignacio: Open-Source Fire Growth Simulation System
===================================================

An open-source implementation mimicking the PROMETHEUS fire growth
model using the Canadian Forest Fire Behaviour Prediction (FBP)
System and Richards' elliptical wave propagation.

The system integrates:
- Fire Weather Index (FWI) calculations
- Fire Behaviour Prediction (FBP) rate of spread equations
- Richards' differential equations for elliptical fire spread
- Huygens-principle wave propagation
- Terrain corrections for slope and aspect
- JAX-based differentiable fire spread (optional)
- Enhanced physics: solar radiation, moisture lag, crown fire, terrain wind
- 3D atmospheric dynamics (WRF-SFIRE style coupling)
- SUMMA-style model decisions configuration

Modules
-------
config : Configuration loading and validation
config_unified : SUMMA-style unified configuration
model_decisions : Model physics choices
parameters : Tunable parameters with bounds
vector_topology : Shapely-based polygon repair
io : Raster and vector I/O utilities
terrain : DEM processing, slope and aspect calculation
ignition : Ignition probability and point sampling
weather : Fire weather processing and filtering
fwi : Fire Weather Index System calculations
fbp : Fire Behaviour Prediction equations
spread : Richards equations and marker method
simulation : Fire front evolution orchestration
visualization : Plotting and animation
station_parser : Alberta/Canada station CSV parsing
preprocessing : Data acquisition and preparation pipeline

References
----------
- Forestry Canada Fire Danger Group (1992). Development and Structure
  of the Canadian Forest Fire Behavior Prediction System.
- Tymstra et al. (2010). Development and Structure of Prometheus:
  the Canadian Wildland Fire Growth Simulation Model.
- Van Wagner (1987). Development and Structure of the Canadian Forest
  Fire Weather Index System.
"""

__version__ = "0.3.0"
__author__ = "Fire Engine Framework Contributors"

# Original config
from ignacio.config import (
    load_config, 
    IgnacioConfig,
    IgnitionCoordinate,
    PhysicsConfig,
    DataSourceConfig,
)

# New unified config system
try:
    from ignacio.config_unified import (
        IgnacioConfig as UnifiedConfig,
        load_config as load_unified_config,
        create_config_from_preset,
        InitialConditions,
        IgnitionPoint,
        DomainConfig,
        OutputConfig,
        CalibrationConfig,
    )
    from ignacio.model_decisions import (
        ModelDecisions,
        get_default_decisions,
        get_preset,
        PRESETS,
        ALL_DECISIONS,
    )
    from ignacio.parameters import (
        ParameterSet,
        ParameterDef,
        get_default_parameters,
        ALL_PARAMETERS,
        CALIBRATABLE_PARAMETERS,
        list_calibratable_parameters,
    )
    HAS_UNIFIED_CONFIG = True
except ImportError:
    HAS_UNIFIED_CONFIG = False

# Vector topology (optional, requires Shapely)
try:
    from ignacio.vector_topology import (
        FirePerimeter,
        clip_perimeter,
        merge_perimeters,
        levelset_to_perimeter,
        export_prometheus_format,
        process_fire_perimeters,
        generate_isochrones,
    )
    HAS_VECTOR_TOPOLOGY = True
except ImportError:
    HAS_VECTOR_TOPOLOGY = False

# Don't import simulation here to avoid circular imports with rasterio
# from ignacio.simulation import run_simulation

__all__ = [
    # Original config
    "load_config",
    "IgnacioConfig",
    "IgnitionCoordinate", 
    "PhysicsConfig",
    "DataSourceConfig",
    # Version
    "__version__",
    # Feature flags
    "HAS_UNIFIED_CONFIG",
    "HAS_VECTOR_TOPOLOGY",
]

# Add unified config exports if available
if HAS_UNIFIED_CONFIG:
    __all__.extend([
        "UnifiedConfig",
        "load_unified_config",
        "create_config_from_preset",
        "InitialConditions",
        "IgnitionPoint",
        "DomainConfig",
        "OutputConfig",
        "CalibrationConfig",
        "ModelDecisions",
        "get_default_decisions",
        "get_preset",
        "PRESETS",
        "ALL_DECISIONS",
        "ParameterSet",
        "ParameterDef",
        "get_default_parameters",
        "ALL_PARAMETERS",
        "CALIBRATABLE_PARAMETERS",
        "list_calibratable_parameters",
    ])

if HAS_VECTOR_TOPOLOGY:
    __all__.extend([
        "FirePerimeter",
        "clip_perimeter",
        "merge_perimeters",
        "levelset_to_perimeter",
        "export_prometheus_format",
        "process_fire_perimeters",
        "generate_isochrones",
    ])
