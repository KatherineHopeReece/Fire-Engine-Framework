"""
Fire Behaviour Prediction (FBP) System equations for Ignacio.

This module implements the Canadian Forest Fire Behaviour Prediction (FBP)
System equations for computing rate of spread (ROS) by fuel type.

References
----------
- Forestry Canada Fire Danger Group (1992). Development and Structure of
  the Canadian Forest Fire Behavior Prediction System. Information Report
  ST-X-3. Ottawa: Forestry Canada Science and Sustainable Development
  Directorate.
- Van Wagner, C.E. (1987). Development and Structure of the Canadian Forest
  Fire Weather Index System. Forestry Technical Report 35.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Fire Weather Index Calculations
# =============================================================================


def calculate_isi(ffmc: float, wind_speed: float) -> float:
    """
    Compute the Initial Spread Index (ISI) from FFMC and wind speed.
    
    The ISI is a numeric rating of the expected rate of fire spread.
    It combines the effects of wind and fine fuel moisture.
    
    Parameters
    ----------
    ffmc : float
        Fine Fuel Moisture Code (0-101 scale).
    wind_speed : float
        Wind speed in km/h.
        
    Returns
    -------
    float
        Initial Spread Index value.
        
    Notes
    -----
    Formula from Van Wagner (1987):
    
        m = 147.2 * (101 - FFMC) / (59.5 + FFMC)
        f_F = 91.9 * exp(-0.1386 * m) * (1 + m^5.31 / 4.93e7)
        f_W = exp(0.05039 * WS)
        ISI = 0.208 * f_F * f_W
    """
    try:
        ffmc = max(0.0, min(101.0, float(ffmc)))
        wind_speed = max(0.0, float(wind_speed))
    except (TypeError, ValueError):
        return 0.0
    
    # Moisture content from FFMC
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    
    # Fuel moisture function
    f_F = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / 4.93e7)
    
    # Wind function
    f_W = np.exp(0.05039 * wind_speed)
    
    # ISI
    isi = 0.208 * f_F * f_W
    
    return float(max(0.0, isi))


def calculate_isi_grid(ffmc_grid: np.ndarray, wind_speed_grid: np.ndarray) -> np.ndarray:
    """
    Compute ISI grid from FFMC and wind speed grids.
    
    Vectorized version of calculate_isi for array inputs.
    
    Parameters
    ----------
    ffmc_grid : np.ndarray
        FFMC values.
    wind_speed_grid : np.ndarray
        Wind speed in km/h.
        
    Returns
    -------
    np.ndarray
        ISI grid.
    """
    ffmc = np.clip(ffmc_grid, 0.0, 101.0)
    ws = np.maximum(wind_speed_grid, 0.0)
    
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    f_F = 91.9 * np.exp(-0.1386 * m) * (1.0 + np.power(m, 5.31) / 4.93e7)
    f_W = np.exp(0.05039 * ws)
    isi = 0.208 * f_F * f_W
    
    return np.clip(isi, 0.0, np.nanmax(isi)).astype(np.float32)


# =============================================================================
# Buildup Effect Functions
# =============================================================================


def _buildup_effect(bui: float, q: float, bui_threshold: float, be_max_denom: float) -> float:
    """
    Compute the Buildup Effect (BE) on rate of spread.
    
    Parameters
    ----------
    bui : float
        Buildup Index value.
    q : float
        Q coefficient for BE calculation.
    bui_threshold : float
        BUI threshold below which BE = 1.
    be_max_denom : float
        Denominator term in BE_max calculation.
        
    Returns
    -------
    float
        Buildup effect multiplier (>= 1).
    """
    if bui <= bui_threshold:
        return 1.0
    
    Q = q * (bui - bui_threshold) ** 0.9
    if Q <= 0:
        return 1.0
    
    be = np.exp(50.0 * np.log(Q) / (be_max_denom + Q))
    return float(max(1.0, be))


# =============================================================================
# Fuel Type ROS Functions
# =============================================================================


def fbp_ros_c1(isi: float, bui: float = 80.0) -> float:
    """
    C-1 Spruce-Lichen Woodland head fire rate of spread (m/min).
    
    Parameters
    ----------
    isi : float
        Initial Spread Index.
    bui : float
        Buildup Index.
        
    Returns
    -------
    float
        Head fire rate of spread in m/min.
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    # RSI parameters for C-1
    a, b, c = 90.0, 0.0649, 4.5
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    # Buildup effect
    if bui > 60.0:
        Q = 0.92 * (bui - 60.0) ** 0.91
        be = np.exp(50.0 * np.log(Q) / (450.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_c2(isi: float, bui: float = 80.0) -> float:
    """
    C-2 Boreal Spruce head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 110.0, 0.0282, 1.5
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 35.0:
        Q = 0.8 * (bui - 35.0) ** 0.92
        be = np.exp(45.0 * np.log(Q) / (300.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_c3(isi: float, bui: float = 80.0) -> float:
    """
    C-3 Mature Jack or Lodgepole Pine head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 110.0, 0.0444, 3.0
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 40.0:
        Q = 0.8 * (bui - 40.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (350.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_c4(isi: float, bui: float = 80.0) -> float:
    """
    C-4 Immature Jack or Lodgepole Pine head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 110.0, 0.0293, 1.5
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 35.0:
        Q = 0.8 * (bui - 35.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (320.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_c5(isi: float, bui: float = 80.0) -> float:
    """
    C-5 Red and White Pine head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 30.0, 0.0697, 4.0
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 40.0:
        Q = 0.8 * (bui - 40.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (350.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_c6(isi: float, bui: float = 80.0, fmc: float = 100.0) -> float:
    """
    C-6 Conifer Plantation head fire rate of spread (m/min).
    
    Includes foliar moisture content (FMC) effect.
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    fmc = max(0.0, float(fmc))
    
    # Base rate (surface fire)
    a, b, c = 30.0, 0.0800, 3.0
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    # FMC effect for crown fire potential
    fme = 1000.0 * (1.5 - 0.00275 * fmc) ** 4.0 / (460.0 + 25.9 * fmc)
    fme = max(0.0, fme)
    
    # Crown fraction burned approximation
    cfb = 1.0 - np.exp(-0.23 * max(0.0, rsi - 10.0))
    cfb = np.clip(cfb, 0.0, 1.0)
    
    # Combined ROS with crown contribution
    ros = rsi + cfb * fme
    
    if bui > 35.0:
        Q = 0.8 * (bui - 35.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (300.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, ros * be))


def fbp_ros_c7(isi: float, bui: float = 80.0) -> float:
    """
    C-7 Ponderosa Pine / Douglas-fir head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 45.0, 0.0305, 2.0
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 40.0:
        Q = 0.8 * (bui - 40.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (350.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_d1(isi: float, bui: float = 80.0) -> float:
    """
    D-1 Leafless Aspen head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 30.0, 0.0232, 1.6
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 40.0:
        Q = 0.8 * (bui - 40.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (350.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_o1(isi: float, curing: float = 85.0) -> float:
    """
    O-1 Grass fuel head fire rate of spread (m/min).
    
    Parameters
    ----------
    isi : float
        Initial Spread Index.
    curing : float
        Grass curing percentage (0-100).
        
    Returns
    -------
    float
        Head fire rate of spread in m/min.
    """
    isi = max(0.0, float(isi))
    curing = np.clip(float(curing), 0.0, 100.0)
    
    # Curing factor
    if curing < 58.8:
        cf = 0.005 * (np.exp(0.061 * curing) - 1.0)
    else:
        cf = 0.176 + 0.02 * (curing - 58.8)
    
    cf = np.clip(cf, 0.0, 1.0)
    
    # O-1a (matted grass) base ROS
    a, b, c = 190.0, 0.0310, 1.4
    ros = a * (1.0 - np.exp(-b * isi)) ** c * cf
    
    return float(max(0.0, ros))


def fbp_ros_o1b(isi: float, curing: float = 85.0) -> float:
    """
    O-1b Standing Grass head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    curing = np.clip(float(curing), 0.0, 100.0)
    
    if curing < 58.8:
        cf = 0.005 * (np.exp(0.061 * curing) - 1.0)
    else:
        cf = 0.176 + 0.02 * (curing - 58.8)
    
    cf = np.clip(cf, 0.0, 1.0)
    
    # O-1b base ROS (higher than O-1a)
    a, b, c = 250.0, 0.0350, 1.7
    ros = a * (1.0 - np.exp(-b * isi)) ** c * cf
    
    return float(max(0.0, ros))


def fbp_ros_m1(isi: float, bui: float = 80.0, pc: float = 50.0) -> float:
    """
    M-1 Boreal Mixedwood - Leafless head fire rate of spread (m/min).
    
    Parameters
    ----------
    isi : float
        Initial Spread Index.
    bui : float
        Buildup Index.
    pc : float
        Percent conifer (0-100).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    pc = np.clip(float(pc), 0.0, 100.0)
    
    # C-2 and D-1 components
    ros_c2 = fbp_ros_c2(isi, bui)
    ros_d1 = fbp_ros_d1(isi, bui)
    
    # Weighted combination
    ros = (pc / 100.0) * ros_c2 + (1.0 - pc / 100.0) * ros_d1
    
    return float(max(0.0, ros))


def fbp_ros_m2(isi: float, bui: float = 80.0, pc: float = 50.0) -> float:
    """
    M-2 Boreal Mixedwood - Green head fire rate of spread (m/min).
    
    Green deciduous reduces spread rate compared to M-1.
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    pc = np.clip(float(pc), 0.0, 100.0)
    
    ros_c2 = fbp_ros_c2(isi, bui)
    
    # Green deciduous has lower ROS
    ros_d1_green = 0.2 * fbp_ros_d1(isi, bui)
    
    ros = (pc / 100.0) * ros_c2 + (1.0 - pc / 100.0) * ros_d1_green
    
    return float(max(0.0, ros))


def fbp_ros_m3(isi: float, bui: float = 80.0, pdf: float = 50.0) -> float:
    """
    M-3 Dead Balsam Fir Mixedwood - Leafless head fire rate of spread.
    
    Parameters
    ----------
    pdf : float
        Percent dead fir (0-100).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    pdf = np.clip(float(pdf), 0.0, 100.0)
    
    # Dead fir contribution (similar to C-2)
    a, b, c = 120.0, 0.0572, 1.4
    ros_dead = a * (1.0 - np.exp(-b * isi)) ** c
    
    # D-1 live component
    ros_d1 = fbp_ros_d1(isi, bui)
    
    ros = (pdf / 100.0) * ros_dead + (1.0 - pdf / 100.0) * ros_d1
    
    return float(max(0.0, ros))


def fbp_ros_m4(isi: float, bui: float = 80.0, pdf: float = 50.0) -> float:
    """
    M-4 Dead Balsam Fir Mixedwood - Green head fire rate of spread.
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    pdf = np.clip(float(pdf), 0.0, 100.0)
    
    a, b, c = 120.0, 0.0572, 1.4
    ros_dead = a * (1.0 - np.exp(-b * isi)) ** c
    ros_d1_green = 0.2 * fbp_ros_d1(isi, bui)
    
    ros = (pdf / 100.0) * ros_dead + (1.0 - pdf / 100.0) * ros_d1_green
    
    return float(max(0.0, ros))


def fbp_ros_s1(isi: float, bui: float = 80.0) -> float:
    """
    S-1 Jack or Lodgepole Pine Slash head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 75.0, 0.0297, 1.3
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 35.0:
        Q = 0.8 * (bui - 35.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (300.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_s2(isi: float, bui: float = 80.0) -> float:
    """
    S-2 White Spruce / Balsam Slash head fire rate of spread (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 40.0, 0.0438, 1.7
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 35.0:
        Q = 0.8 * (bui - 35.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (300.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


def fbp_ros_s3(isi: float, bui: float = 80.0) -> float:
    """
    S-3 Coastal Cedar / Hemlock / Douglas-fir Slash head fire ROS (m/min).
    """
    isi = max(0.0, float(isi))
    bui = float(bui)
    
    a, b, c = 55.0, 0.0829, 3.2
    rsi = a * (1.0 - np.exp(-b * isi)) ** c
    
    if bui > 35.0:
        Q = 0.8 * (bui - 35.0) ** 0.90
        be = np.exp(45.0 * np.log(Q) / (300.0 + Q)) if Q > 0 else 1.0
    else:
        be = 1.0
    
    return float(max(0.0, rsi * be))


# =============================================================================
# Fuel Type Dispatch
# =============================================================================


# Map of fuel type codes to ROS functions
FBP_ROS_FUNCTIONS: dict[str, Callable[..., float]] = {
    "C-1": fbp_ros_c1,
    "C-2": fbp_ros_c2,
    "C-3": fbp_ros_c3,
    "C-4": fbp_ros_c4,
    "C-5": fbp_ros_c5,
    "C-6": fbp_ros_c6,
    "C-7": fbp_ros_c7,
    "D-1": fbp_ros_d1,
    "O-1": fbp_ros_o1,
    "O-1A": fbp_ros_o1,
    "O-1a": fbp_ros_o1,  # lowercase variant
    "O-1B": fbp_ros_o1b,
    "O-1b": fbp_ros_o1b,  # lowercase variant
    "M-1": fbp_ros_m1,
    "M-2": fbp_ros_m2,
    "M-3": fbp_ros_m3,
    "M-4": fbp_ros_m4,
    "S-1": fbp_ros_s1,
    "S-2": fbp_ros_s2,
    "S-3": fbp_ros_s3,
}

# Non-fuel codes
NON_FUEL_TYPES = {"NF", "WA", "NB"}


def compute_ros(
    fuel_type: str | int,
    isi: float,
    bui: float = 80.0,
    fmc: float = 100.0,
    curing: float = 85.0,
    pc: float = 50.0,
    pdf: float = 50.0,
    fuel_lookup: dict[int, str] | None = None,
) -> float:
    """
    Compute head fire rate of spread for a given fuel type.
    
    Parameters
    ----------
    fuel_type : str or int
        FBP fuel type code (e.g., "C-2") or numeric ID.
    isi : float
        Initial Spread Index.
    bui : float
        Buildup Index.
    fmc : float
        Foliar moisture content (percent, for C-6).
    curing : float
        Grass curing (percent, for O-1 fuels).
    pc : float
        Percent conifer (for M-1, M-2).
    pdf : float
        Percent dead fir (for M-3, M-4).
    fuel_lookup : dict, optional
        Mapping of numeric IDs to fuel type codes.
        
    Returns
    -------
    float
        Head fire rate of spread in m/min.
    """
    # Convert numeric ID to fuel type code
    if isinstance(fuel_type, (int, float)):
        fuel_id = int(fuel_type)
        if fuel_lookup and fuel_id in fuel_lookup:
            fuel_code = fuel_lookup[fuel_id]
        else:
            # Default mapping for common codes
            default_lookup = {
                1: "C-1", 2: "C-2", 3: "C-3", 4: "C-4", 5: "C-5",
                6: "C-6", 7: "C-7", 11: "D-1", 12: "D-1", 13: "D-1",
                21: "S-1", 22: "S-2", 23: "S-3",
                31: "O-1a", 32: "O-1b",
                40: "M-1", 50: "M-2", 60: "M-3", 70: "M-4",
                101: "NF", 102: "WA", 106: "NB",
            }
            fuel_code = default_lookup.get(fuel_id, "NF")
    else:
        fuel_code = str(fuel_type).upper().strip()
    
    # Non-fuel types
    if fuel_code in NON_FUEL_TYPES or fuel_code in (0, -9999):
        return 0.0
    
    # Get appropriate function
    ros_func = FBP_ROS_FUNCTIONS.get(fuel_code)
    
    if ros_func is None:
        logger.warning(f"Unknown fuel type: {fuel_code}, using zero ROS")
        return 0.0
    
    # Call with appropriate parameters
    if fuel_code in ("O-1", "O-1A", "O-1B", "O-1a", "O-1b"):
        return ros_func(isi, curing)
    elif fuel_code == "C-6":
        return ros_func(isi, bui, fmc)
    elif fuel_code in ("M-1", "M-2"):
        return ros_func(isi, bui, pc)
    elif fuel_code in ("M-3", "M-4"):
        return ros_func(isi, bui, pdf)
    else:
        return ros_func(isi, bui)


# =============================================================================
# Rate of Spread Grid Computation
# =============================================================================


@dataclass
class ROSComponents:
    """Components of rate of spread."""
    
    ros_head: float  # Head fire ROS (m/min)
    ros_flank: float  # Flank fire ROS (m/min)
    ros_back: float  # Back fire ROS (m/min)
    raz: float  # Rate of spread azimuth (degrees from north)
    lb_ratio: float  # Length-to-breadth ratio


def compute_ros_components(
    ros_head: float,
    wind_direction: float,
    backing_fraction: float = 0.2,
    lb_ratio: float = 2.0,
) -> ROSComponents:
    """
    Compute all rate of spread components from head fire ROS.
    
    Parameters
    ----------
    ros_head : float
        Head fire rate of spread (m/min).
    wind_direction : float
        Wind direction in degrees (direction wind comes FROM).
    backing_fraction : float
        Back fire ROS as fraction of head fire ROS.
    lb_ratio : float
        Length-to-breadth ratio of fire ellipse.
        
    Returns
    -------
    ROSComponents
        All ROS components.
        
    Notes
    -----
    The fire ellipse relationships are:
    
        ROS_back = backing_fraction * ROS_head
        ROS_flank = (ROS_head + ROS_back) / (2 * LB)
        
    The spread azimuth (RAZ) is the direction of maximum spread,
    which is the direction the wind is blowing TO (opposite of FROM).
    """
    ros_back = backing_fraction * ros_head
    ros_flank = (ros_head + ros_back) / (2.0 * lb_ratio)
    
    # RAZ is direction fire spreads TO (wind blows fire forward)
    raz = (wind_direction + 180.0) % 360.0
    
    return ROSComponents(
        ros_head=ros_head,
        ros_flank=ros_flank,
        ros_back=ros_back,
        raz=raz,
        lb_ratio=lb_ratio,
    )


def compute_ros_grid(
    fuel_grid: np.ndarray,
    isi: float,
    bui: float,
    fmc: float = 100.0,
    curing: float = 85.0,
    fuel_lookup: dict[int, str] | None = None,
    non_fuel_codes: list[int] | None = None,
) -> np.ndarray:
    """
    Compute head fire ROS for entire fuel grid.
    
    Parameters
    ----------
    fuel_grid : np.ndarray
        2D array of fuel type codes.
    isi : float
        Initial Spread Index.
    bui : float
        Buildup Index.
    fmc : float
        Foliar moisture content.
    curing : float
        Grass curing percentage.
    fuel_lookup : dict, optional
        Mapping of numeric IDs to fuel codes.
    non_fuel_codes : list, optional
        Codes to treat as non-fuel.
        
    Returns
    -------
    np.ndarray
        2D array of head fire ROS (m/min).
    """
    if non_fuel_codes is None:
        non_fuel_codes = [0, 101, 102, 106, -9999]
    
    ros_grid = np.zeros_like(fuel_grid, dtype=np.float32)
    unique_fuels = np.unique(fuel_grid)
    
    for fuel_id in unique_fuels:
        if fuel_id in non_fuel_codes or np.isnan(fuel_id):
            continue
        
        mask = fuel_grid == fuel_id
        ros = compute_ros(
            fuel_type=int(fuel_id),
            isi=isi,
            bui=bui,
            fmc=fmc,
            curing=curing,
            fuel_lookup=fuel_lookup,
        )
        ros_grid[mask] = ros
    
    return ros_grid
