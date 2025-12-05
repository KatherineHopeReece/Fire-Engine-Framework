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

Modules
-------
config : Configuration loading and validation
io : Raster and vector I/O utilities
terrain : DEM processing, slope and aspect calculation
ignition : Ignition probability and point sampling
weather : Fire weather processing and filtering
fwi : Fire Weather Index System calculations
fbp : Fire Behaviour Prediction equations
spread : Richards equations and marker method
simulation : Fire front evolution orchestration
visualization : Plotting and animation

References
----------
- Forestry Canada Fire Danger Group (1992). Development and Structure
  of the Canadian Forest Fire Behavior Prediction System.
- Tymstra et al. (2010). Development and Structure of Prometheus:
  the Canadian Wildland Fire Growth Simulation Model.
- Van Wagner (1987). Development and Structure of the Canadian Forest
  Fire Weather Index System.
"""

__version__ = "0.1.0"
__author__ = "Fire Engine Framework Contributors"

from ignacio.config import load_config, IgnacioConfig
from ignacio.simulation import run_simulation

__all__ = [
    "load_config",
    "IgnacioConfig", 
    "run_simulation",
    "__version__",
]
