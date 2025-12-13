"""
Rothermel Fire Spread Model.

Implements the Rothermel (1972) surface fire spread equations used in
FARSITE, FlamMap, and other US fire modeling systems.

This provides an alternative to the Canadian FBP system for international use.

References:
- Rothermel, R.C. (1972). A mathematical model for predicting fire spread
  in wildland fuels. USDA Forest Service Research Paper INT-115.
- Scott, J.H. & Burgan, R.E. (2005). Standard fire behavior fuel models.
  USDA Forest Service RMRS-GTR-153.
- Andrews, P.L. (2018). The Rothermel surface fire spread model and 
  associated developments: A comprehensive explanation. RMRS-GTR-371.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Dict
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Fuel Model Definitions
# =============================================================================

class FuelModel(NamedTuple):
    """
    Rothermel fuel model parameters.
    
    Based on Scott & Burgan (2005) 40 fuel models and 
    Anderson (1982) 13 original models.
    """
    name: str
    
    # Fuel loads (kg/m²)
    w_1hr: float      # 1-hour dead fuel load
    w_10hr: float     # 10-hour dead fuel load  
    w_100hr: float    # 100-hour dead fuel load
    w_live_herb: float  # Live herbaceous fuel load
    w_live_woody: float # Live woody fuel load
    
    # Surface area to volume ratios (1/m)
    sv_1hr: float     # 1-hour SAV
    sv_live_herb: float  # Live herbaceous SAV
    sv_live_woody: float # Live woody SAV
    
    # Fuel bed properties
    depth: float      # Fuel bed depth (m)
    mx_dead: float    # Dead fuel moisture of extinction (fraction)
    
    # Heat content (kJ/kg) - typically 18000-21000
    heat_content: float = 18600.0


# Standard 13 fuel models (Anderson 1982) - converted to metric
FUEL_MODELS_13 = {
    1: FuelModel("Short grass", 0.166, 0.0, 0.0, 0.0, 0.0, 11483, 4921, 4921, 0.30, 0.12),
    2: FuelModel("Timber grass/understory", 0.449, 0.224, 0.112, 0.112, 0.0, 9843, 4921, 4921, 0.30, 0.15),
    3: FuelModel("Tall grass", 0.675, 0.0, 0.0, 0.0, 0.0, 4921, 4921, 4921, 0.76, 0.25),
    4: FuelModel("Chaparral", 1.123, 0.898, 0.449, 0.0, 1.123, 6562, 4921, 4921, 1.83, 0.20),
    5: FuelModel("Brush", 0.224, 0.112, 0.0, 0.0, 0.449, 6562, 4921, 4921, 0.61, 0.20),
    6: FuelModel("Dormant brush", 0.337, 0.562, 0.449, 0.0, 0.0, 5741, 4921, 4921, 0.76, 0.25),
    7: FuelModel("Southern rough", 0.253, 0.421, 0.337, 0.0, 0.084, 5741, 4921, 4921, 0.76, 0.40),
    8: FuelModel("Compact timber litter", 0.337, 0.224, 0.562, 0.0, 0.0, 6562, 4921, 4921, 0.06, 0.30),
    9: FuelModel("Hardwood litter", 0.655, 0.091, 0.034, 0.0, 0.0, 8202, 4921, 4921, 0.06, 0.25),
    10: FuelModel("Timber understory", 0.675, 0.449, 1.124, 0.0, 0.449, 6562, 4921, 4921, 0.30, 0.25),
    11: FuelModel("Light slash", 0.337, 1.011, 1.236, 0.0, 0.0, 4921, 4921, 4921, 0.30, 0.15),
    12: FuelModel("Medium slash", 0.898, 3.146, 3.708, 0.0, 0.0, 4921, 4921, 4921, 0.70, 0.20),
    13: FuelModel("Heavy slash", 1.573, 5.168, 6.180, 0.0, 0.0, 4921, 4921, 4921, 0.91, 0.25),
}

# Scott & Burgan 40 fuel models (selected common ones)
FUEL_MODELS_40 = {
    # Grass models (GR)
    101: FuelModel("GR1 - Short sparse dry grass", 0.022, 0.0, 0.0, 0.067, 0.0, 7218, 6562, 4921, 0.12, 0.15),
    102: FuelModel("GR2 - Low load dry grass", 0.022, 0.0, 0.0, 0.224, 0.0, 6562, 5906, 4921, 0.30, 0.15),
    104: FuelModel("GR4 - Moderate load dry grass", 0.056, 0.0, 0.0, 0.427, 0.0, 6562, 5906, 4921, 0.61, 0.15),
    
    # Grass-Shrub models (GS)
    121: FuelModel("GS1 - Low load grass-shrub", 0.045, 0.0, 0.0, 0.112, 0.146, 6562, 5906, 5906, 0.27, 0.15),
    122: FuelModel("GS2 - Moderate load grass-shrub", 0.112, 0.112, 0.0, 0.135, 0.224, 6562, 5906, 5906, 0.46, 0.15),
    
    # Shrub models (SH)
    141: FuelModel("SH1 - Low load dry shrub", 0.056, 0.056, 0.0, 0.034, 0.292, 6562, 5906, 5249, 0.30, 0.15),
    145: FuelModel("SH5 - High load dry shrub", 0.798, 0.618, 0.0, 0.0, 0.674, 2461, 5906, 4921, 1.83, 0.15),
    
    # Timber-Understory models (TU)
    161: FuelModel("TU1 - Light load timber-grass-shrub", 0.045, 0.202, 0.337, 0.045, 0.202, 6562, 5906, 5906, 0.18, 0.20),
    165: FuelModel("TU5 - Very high load timber-shrub", 0.898, 0.898, 0.674, 0.0, 0.674, 5249, 5906, 4921, 0.30, 0.25),
    
    # Timber Litter models (TL)
    181: FuelModel("TL1 - Low load compact timber litter", 0.224, 0.494, 0.427, 0.0, 0.0, 6562, 5906, 4921, 0.06, 0.30),
    183: FuelModel("TL3 - Moderate load conifer litter", 0.112, 0.494, 0.618, 0.0, 0.0, 6562, 5906, 4921, 0.09, 0.20),
    186: FuelModel("TL6 - Moderate load broadleaf litter", 0.562, 0.674, 0.562, 0.0, 0.0, 6562, 5906, 4921, 0.09, 0.25),
    
    # Slash-Blowdown models (SB)
    201: FuelModel("SB1 - Low load slash", 0.337, 0.674, 1.124, 0.0, 0.0, 6562, 5906, 4921, 0.30, 0.25),
    204: FuelModel("SB4 - High load slash", 1.236, 0.843, 1.685, 0.0, 0.0, 6562, 5906, 4921, 0.82, 0.25),
}

# Combined fuel models dictionary
FUEL_MODELS = {**FUEL_MODELS_13, **FUEL_MODELS_40}


# =============================================================================
# Rothermel Model Core Equations
# =============================================================================

def compute_moisture_damping(
    moisture: float,
    mx: float,
) -> float:
    """
    Compute moisture damping coefficient.
    
    η_M = 1 - 2.59*(M/Mx) + 5.11*(M/Mx)² - 3.52*(M/Mx)³
    
    Parameters
    ----------
    moisture : float
        Fuel moisture content (fraction, e.g., 0.08 for 8%)
    mx : float
        Moisture of extinction (fraction)
        
    Returns
    -------
    eta_m : float
        Moisture damping coefficient (0-1)
    """
    rm = jnp.clip(moisture / mx, 0, 1)
    eta_m = 1.0 - 2.59 * rm + 5.11 * rm**2 - 3.52 * rm**3
    return jnp.maximum(eta_m, 0.0)


def compute_mineral_damping(
    s_e: float = 0.01,  # Effective mineral content (fraction)
) -> float:
    """
    Compute mineral damping coefficient.
    
    η_s = 0.174 * S_e^(-0.19)
    
    For most wildland fuels, S_e ≈ 0.01, giving η_s ≈ 0.42
    """
    eta_s = 0.174 * s_e**(-0.19)
    return jnp.minimum(eta_s, 1.0)


def compute_packing_ratio(
    fuel_load: float,
    depth: float,
    particle_density: float = 513.0,  # kg/m³ for wood
) -> float:
    """
    Compute packing ratio (β).
    
    β = ρ_b / ρ_p = (w_o / δ) / ρ_p
    
    where:
    - ρ_b = bulk density = fuel_load / depth
    - ρ_p = particle density
    """
    bulk_density = fuel_load / jnp.maximum(depth, 0.001)
    beta = bulk_density / particle_density
    return beta


def compute_optimum_packing_ratio(
    sigma: float,  # Characteristic SAV (1/m)
) -> float:
    """
    Compute optimum packing ratio (β_op).
    
    β_op = 3.348 * σ^(-0.8189)
    """
    # Convert from 1/m to 1/ft for empirical equation, then result is dimensionless
    sigma_ft = sigma * 0.3048
    beta_op = 3.348 * sigma_ft**(-0.8189)
    return beta_op


def compute_reaction_velocity(
    sigma: float,
    beta: float,
    beta_op: float,
) -> float:
    """
    Compute maximum reaction velocity (Γ'_max) and actual reaction velocity (Γ').
    
    Γ'_max = σ^1.5 * (495 + 0.0594*σ^1.5)^(-1)
    Γ' = Γ'_max * (β/β_op)^A * exp(A*(1 - β/β_op))
    
    where A = 133 * σ^(-0.7913)
    """
    # Convert to imperial for empirical equations
    sigma_ft = sigma * 0.3048
    
    gamma_max = sigma_ft**1.5 / (495.0 + 0.0594 * sigma_ft**1.5)
    
    A = 133.0 * sigma_ft**(-0.7913)
    
    ratio = beta / jnp.maximum(beta_op, 1e-6)
    gamma = gamma_max * ratio**A * jnp.exp(A * (1.0 - ratio))
    
    return gamma  # 1/min


def compute_reaction_intensity(
    gamma: float,
    heat_content: float,
    fuel_load: float,
    eta_m: float,
    eta_s: float,
) -> float:
    """
    Compute reaction intensity (I_R).
    
    I_R = Γ' * w_n * h * η_M * η_s
    
    where w_n ≈ w_o (net fuel load, slightly less due to minerals)
    
    Returns kW/m² (after unit conversion from BTU/ft²/min)
    """
    # Net fuel load (accounting for ~2% mineral content)
    w_n = fuel_load * 0.98
    
    # Reaction intensity (BTU/ft²/min in original, we keep metric)
    I_R = gamma * w_n * heat_content * eta_m * eta_s
    
    return I_R  # kJ/m²/min


def compute_propagating_flux_ratio(
    sigma: float,
    beta: float,
) -> float:
    """
    Compute propagating flux ratio (ξ).
    
    ξ = exp((0.792 + 0.681*σ^0.5) * (β + 0.1)) / (192 + 0.2595*σ)
    """
    sigma_ft = sigma * 0.3048
    
    numerator = jnp.exp((0.792 + 0.681 * sigma_ft**0.5) * (beta + 0.1))
    denominator = 192.0 + 0.2595 * sigma_ft
    
    xi = numerator / denominator
    return xi


def compute_heat_sink(
    bulk_density: float,
    effective_heating_number: float,
    heat_of_preignition: float,
) -> float:
    """
    Compute heat sink term (denominator of Rothermel equation).
    
    Q = ρ_b * ε * Q_ig
    
    Parameters
    ----------
    bulk_density : float
        Fuel bed bulk density (kg/m³)
    effective_heating_number : float
        ε = exp(-138/σ)
    heat_of_preignition : float
        Q_ig = 250 + 1116*M_f (kJ/kg)
    """
    return bulk_density * effective_heating_number * heat_of_preignition


def compute_wind_factor(
    wind_speed: float,  # m/min (midflame)
    sigma: float,
    beta: float,
    beta_op: float,
) -> float:
    """
    Compute wind coefficient (φ_w).
    
    φ_w = C * U^B * (β/β_op)^(-E)
    
    where:
    - C = 7.47 * exp(-0.133 * σ^0.55)
    - B = 0.02526 * σ^0.54
    - E = 0.715 * exp(-3.59e-4 * σ)
    """
    sigma_ft = sigma * 0.3048
    U_ft = wind_speed * 3.281  # m/min to ft/min
    
    C = 7.47 * jnp.exp(-0.133 * sigma_ft**0.55)
    B = 0.02526 * sigma_ft**0.54
    E = 0.715 * jnp.exp(-3.59e-4 * sigma_ft)
    
    ratio = beta / jnp.maximum(beta_op, 1e-6)
    
    phi_w = C * U_ft**B * ratio**(-E)
    
    return phi_w


def compute_slope_factor(
    slope: float,  # Slope in degrees
    beta: float,
) -> float:
    """
    Compute slope coefficient (φ_s).
    
    φ_s = 5.275 * β^(-0.3) * tan²(θ)
    """
    tan_slope = jnp.tan(jnp.radians(slope))
    phi_s = 5.275 * beta**(-0.3) * tan_slope**2
    return phi_s


def rothermel_ros(
    fuel_model: FuelModel,
    moisture_1hr: float,
    moisture_10hr: float,
    moisture_100hr: float,
    moisture_live: float,
    wind_speed: float,  # Midflame wind speed (km/h)
    slope: float,  # Slope (degrees)
) -> float:
    """
    Compute Rothermel rate of spread.
    
    R = (I_R * ξ * (1 + φ_w + φ_s)) / (ρ_b * ε * Q_ig)
    
    Parameters
    ----------
    fuel_model : FuelModel
        Fuel model parameters
    moisture_1hr : float
        1-hour timelag dead fuel moisture (fraction, e.g., 0.06)
    moisture_10hr : float
        10-hour timelag dead fuel moisture (fraction)
    moisture_100hr : float
        100-hour timelag dead fuel moisture (fraction)
    moisture_live : float
        Live fuel moisture (fraction)
    wind_speed : float
        Midflame wind speed (km/h)
    slope : float
        Terrain slope (degrees)
        
    Returns
    -------
    ros : float
        Rate of spread (m/min)
    """
    # Convert wind to m/min
    U = wind_speed * 1000 / 60  # km/h to m/min
    
    # Weighted average dead fuel moisture
    total_dead = fuel_model.w_1hr + fuel_model.w_10hr + fuel_model.w_100hr
    if total_dead > 0:
        m_dead = (fuel_model.w_1hr * moisture_1hr + 
                  fuel_model.w_10hr * moisture_10hr +
                  fuel_model.w_100hr * moisture_100hr) / total_dead
    else:
        m_dead = moisture_1hr
    
    # Total fuel load
    total_load = (fuel_model.w_1hr + fuel_model.w_10hr + fuel_model.w_100hr + 
                  fuel_model.w_live_herb + fuel_model.w_live_woody)
    
    if total_load < 0.001:  # No fuel
        return 0.0
    
    # Characteristic SAV (weighted by load)
    dead_load = fuel_model.w_1hr + fuel_model.w_10hr + fuel_model.w_100hr
    live_load = fuel_model.w_live_herb + fuel_model.w_live_woody
    
    if dead_load > 0:
        sigma_dead = fuel_model.sv_1hr  # Simplified - use 1hr SAV for dead
    else:
        sigma_dead = 0
    
    if live_load > 0:
        sigma_live = (fuel_model.w_live_herb * fuel_model.sv_live_herb + 
                      fuel_model.w_live_woody * fuel_model.sv_live_woody) / live_load
    else:
        sigma_live = 0
    
    if total_load > 0:
        sigma = (dead_load * sigma_dead + live_load * sigma_live) / total_load
    else:
        sigma = 6562  # Default
    
    sigma = jnp.maximum(sigma, 100)  # Minimum SAV
    
    # Packing ratios
    beta = compute_packing_ratio(total_load, fuel_model.depth)
    beta_op = compute_optimum_packing_ratio(sigma)
    
    # Moisture damping
    eta_m = compute_moisture_damping(m_dead, fuel_model.mx_dead)
    
    # Mineral damping
    eta_s = compute_mineral_damping()
    
    # Reaction velocity and intensity
    gamma = compute_reaction_velocity(sigma, beta, beta_op)
    I_R = compute_reaction_intensity(gamma, fuel_model.heat_content, total_load, eta_m, eta_s)
    
    # Propagating flux ratio
    xi = compute_propagating_flux_ratio(sigma, beta)
    
    # Wind and slope factors
    phi_w = compute_wind_factor(U, sigma, beta, beta_op)
    phi_s = compute_slope_factor(slope, beta)
    
    # Heat sink
    bulk_density = total_load / jnp.maximum(fuel_model.depth, 0.001)
    sigma_ft = sigma * 0.3048
    epsilon = jnp.exp(-138.0 / sigma_ft)  # Effective heating number
    Q_ig = 250.0 + 1116.0 * m_dead  # Heat of preignition (kJ/kg)
    
    heat_sink = bulk_density * epsilon * Q_ig
    
    # Rate of spread (m/min)
    R = (I_R * xi * (1.0 + phi_w + phi_s)) / jnp.maximum(heat_sink, 0.001)
    
    # Convert from ft/min to m/min if needed and ensure non-negative
    R = jnp.maximum(R, 0.0)
    
    return R


# =============================================================================
# JAX-compatible vectorized version
# =============================================================================

def compute_ros_rothermel_grid(
    fuel_codes: jnp.ndarray,
    moisture_1hr: jnp.ndarray,
    moisture_10hr: jnp.ndarray,
    moisture_100hr: jnp.ndarray,
    moisture_live: jnp.ndarray,
    wind_speed: jnp.ndarray,
    slope: jnp.ndarray,
    fuel_models: Dict[int, FuelModel] = None,
) -> jnp.ndarray:
    """
    Compute ROS grid using Rothermel model.
    
    Parameters
    ----------
    fuel_codes : jnp.ndarray
        Fuel model codes at each cell
    moisture_* : jnp.ndarray
        Fuel moisture values (can be scalar or grid)
    wind_speed : jnp.ndarray
        Midflame wind speed (km/h)
    slope : jnp.ndarray
        Terrain slope (degrees)
    fuel_models : dict, optional
        Fuel model lookup. Default uses FUEL_MODELS
        
    Returns
    -------
    ros : jnp.ndarray
        Rate of spread (m/min) at each cell
    """
    if fuel_models is None:
        fuel_models = FUEL_MODELS
    
    # Build lookup arrays for fuel properties
    max_code = max(fuel_models.keys()) + 1
    
    w_1hr = np.zeros(max_code)
    w_10hr = np.zeros(max_code)
    w_100hr = np.zeros(max_code)
    w_live_h = np.zeros(max_code)
    w_live_w = np.zeros(max_code)
    sv_1hr = np.zeros(max_code)
    sv_live_h = np.zeros(max_code)
    sv_live_w = np.zeros(max_code)
    depth = np.zeros(max_code)
    mx_dead = np.zeros(max_code)
    heat = np.zeros(max_code)
    
    for code, model in fuel_models.items():
        w_1hr[code] = model.w_1hr
        w_10hr[code] = model.w_10hr
        w_100hr[code] = model.w_100hr
        w_live_h[code] = model.w_live_herb
        w_live_w[code] = model.w_live_woody
        sv_1hr[code] = model.sv_1hr
        sv_live_h[code] = model.sv_live_herb
        sv_live_w[code] = model.sv_live_woody
        depth[code] = model.depth
        mx_dead[code] = model.mx_dead
        heat[code] = model.heat_content
    
    # Convert to JAX arrays
    w_1hr = jnp.array(w_1hr)
    w_10hr = jnp.array(w_10hr)
    w_100hr = jnp.array(w_100hr)
    w_live_h = jnp.array(w_live_h)
    w_live_w = jnp.array(w_live_w)
    sv_1hr = jnp.array(sv_1hr)
    sv_live_h = jnp.array(sv_live_h)
    sv_live_w = jnp.array(sv_live_w)
    depth_arr = jnp.array(depth)
    mx_dead_arr = jnp.array(mx_dead)
    heat_arr = jnp.array(heat)
    
    # Clip fuel codes to valid range
    codes_clipped = jnp.clip(fuel_codes, 0, max_code - 1).astype(jnp.int32)
    
    # Look up properties for each cell
    cell_w_1hr = w_1hr[codes_clipped]
    cell_w_10hr = w_10hr[codes_clipped]
    cell_w_100hr = w_100hr[codes_clipped]
    cell_w_live_h = w_live_h[codes_clipped]
    cell_w_live_w = w_live_w[codes_clipped]
    cell_sv_1hr = sv_1hr[codes_clipped]
    cell_sv_live_h = sv_live_h[codes_clipped]
    cell_sv_live_w = sv_live_w[codes_clipped]
    cell_depth = depth_arr[codes_clipped]
    cell_mx = mx_dead_arr[codes_clipped]
    cell_heat = heat_arr[codes_clipped]
    
    # Total loads
    dead_load = cell_w_1hr + cell_w_10hr + cell_w_100hr
    live_load = cell_w_live_h + cell_w_live_w
    total_load = dead_load + live_load
    
    # Weighted dead moisture
    m_dead = jnp.where(
        dead_load > 0,
        (cell_w_1hr * moisture_1hr + cell_w_10hr * moisture_10hr + 
         cell_w_100hr * moisture_100hr) / jnp.maximum(dead_load, 1e-6),
        moisture_1hr
    )
    
    # Characteristic SAV
    sigma_live = jnp.where(
        live_load > 0,
        (cell_w_live_h * cell_sv_live_h + cell_w_live_w * cell_sv_live_w) / jnp.maximum(live_load, 1e-6),
        0
    )
    sigma = jnp.where(
        total_load > 0,
        (dead_load * cell_sv_1hr + live_load * sigma_live) / jnp.maximum(total_load, 1e-6),
        6562
    )
    sigma = jnp.maximum(sigma, 100)
    
    # Packing ratios
    beta = total_load / (jnp.maximum(cell_depth, 0.001) * 513.0)
    sigma_ft = sigma * 0.3048
    beta_op = 3.348 * sigma_ft**(-0.8189)
    
    # Moisture damping
    rm = jnp.clip(m_dead / jnp.maximum(cell_mx, 0.01), 0, 1)
    eta_m = jnp.maximum(1.0 - 2.59 * rm + 5.11 * rm**2 - 3.52 * rm**3, 0)
    
    # Mineral damping
    eta_s = 0.42  # For S_e = 0.01
    
    # Reaction velocity
    gamma_max = sigma_ft**1.5 / (495.0 + 0.0594 * sigma_ft**1.5)
    A = 133.0 * sigma_ft**(-0.7913)
    ratio = beta / jnp.maximum(beta_op, 1e-6)
    gamma = gamma_max * ratio**A * jnp.exp(A * (1.0 - ratio))
    
    # Reaction intensity
    I_R = gamma * total_load * 0.98 * cell_heat * eta_m * eta_s
    
    # Propagating flux ratio
    xi_num = jnp.exp((0.792 + 0.681 * sigma_ft**0.5) * (beta + 0.1))
    xi = xi_num / (192.0 + 0.2595 * sigma_ft)
    
    # Wind factor
    U = wind_speed * 1000 / 60  # km/h to m/min
    U_ft = U * 3.281
    C = 7.47 * jnp.exp(-0.133 * sigma_ft**0.55)
    B = 0.02526 * sigma_ft**0.54
    E = 0.715 * jnp.exp(-3.59e-4 * sigma_ft)
    phi_w = C * jnp.maximum(U_ft, 0.1)**B * ratio**(-E)
    
    # Slope factor
    tan_slope = jnp.tan(jnp.radians(slope))
    phi_s = 5.275 * beta**(-0.3) * tan_slope**2
    
    # Heat sink
    bulk_density = total_load / jnp.maximum(cell_depth, 0.001)
    epsilon = jnp.exp(-138.0 / jnp.maximum(sigma_ft, 1))
    Q_ig = 250.0 + 1116.0 * m_dead
    heat_sink = bulk_density * epsilon * Q_ig
    
    # Rate of spread
    R = (I_R * xi * (1.0 + phi_w + phi_s)) / jnp.maximum(heat_sink, 0.001)
    
    # Zero ROS where no fuel
    R = jnp.where(total_load > 0.01, R, 0.0)
    R = jnp.maximum(R, 0.0)
    
    return R


def compute_ros_components_rothermel(
    fuel_codes: jnp.ndarray,
    moisture_1hr: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_direction: jnp.ndarray,
    slope: jnp.ndarray,
    aspect: jnp.ndarray,
    lb_ratio: float = 1.0,
) -> tuple:
    """
    Compute ROS, BROS, FROS and spread direction using Rothermel.
    
    Returns components compatible with level-set elliptical spread.
    
    Parameters
    ----------
    fuel_codes : jnp.ndarray
        Fuel model codes
    moisture_1hr : jnp.ndarray
        1-hour dead fuel moisture (fraction)
    wind_speed : jnp.ndarray
        Midflame wind speed (km/h)
    wind_direction : jnp.ndarray
        Wind direction (degrees, direction wind blows FROM)
    slope : jnp.ndarray
        Terrain slope (degrees)
    aspect : jnp.ndarray
        Terrain aspect (degrees)
    lb_ratio : float
        Length-to-breadth ratio for ellipse
        
    Returns
    -------
    ros : jnp.ndarray
        Head fire rate of spread (m/min)
    bros : jnp.ndarray
        Back fire rate of spread (m/min)
    fros : jnp.ndarray
        Flank fire rate of spread (m/min)
    raz : jnp.ndarray
        Spread direction (degrees, direction fire spreads TO)
    """
    # Use simplified moisture (10hr = 1hr + 0.01, 100hr = 1hr + 0.02)
    moisture_10hr = moisture_1hr + 0.01
    moisture_100hr = moisture_1hr + 0.02
    moisture_live = 1.0  # Assume 100% for live fuels (conservative)
    
    # Compute base ROS (no wind, no slope)
    ros_base = compute_ros_rothermel_grid(
        fuel_codes, moisture_1hr, moisture_10hr, moisture_100hr, moisture_live,
        jnp.zeros_like(wind_speed), jnp.zeros_like(slope)
    )
    
    # Compute ROS with wind and slope
    ros = compute_ros_rothermel_grid(
        fuel_codes, moisture_1hr, moisture_10hr, moisture_100hr, moisture_live,
        wind_speed, slope
    )
    
    # Spread direction: combine wind and slope vectors
    # Wind pushes fire in direction wind blows TO (opposite of FROM)
    wind_to = (wind_direction + 180) % 360
    
    # Slope pushes fire uphill (opposite of aspect which points downhill)
    uphill = (aspect + 180) % 360
    
    # Simple vector combination (could be more sophisticated)
    # Weight by relative contribution
    wind_weight = wind_speed / (wind_speed + slope + 0.1)
    slope_weight = slope / (wind_speed + slope + 0.1)
    
    # Circular mean of directions
    wind_rad = jnp.radians(wind_to)
    slope_rad = jnp.radians(uphill)
    
    x = wind_weight * jnp.cos(wind_rad) + slope_weight * jnp.cos(slope_rad)
    y = wind_weight * jnp.sin(wind_rad) + slope_weight * jnp.sin(slope_rad)
    
    raz = jnp.degrees(jnp.arctan2(y, x)) % 360
    
    # Back fire ROS (fire spreading against wind/slope)
    # Typically much slower - use base ROS or fraction
    bros = ros_base * 0.1  # Back fire at 10% of base
    
    # Flank fire ROS (perpendicular to spread)
    # Based on ellipse: FROS = ROS / LB
    fros = ros / jnp.maximum(lb_ratio, 1.0)
    
    return ros, bros, fros, raz
