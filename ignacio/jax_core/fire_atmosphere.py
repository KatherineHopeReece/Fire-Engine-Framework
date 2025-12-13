"""
Fire-Atmosphere Coupling Model.

Computes fire-induced wind modifications including:
1. Indraft: Air drawn toward the fire front due to buoyancy
2. Plume effects: Vertical convection modifying surface winds
3. Fire whirls: Rotational effects in extreme conditions

This explains why:
- Real fires are "fatter" than pure vector models predict
- Fire fronts slow down when they converge
- Fires create their own local wind patterns
- Extreme fires can "pulse" and surge

Physics:
- Fire releases heat → hot air rises → creates low pressure
- Surrounding air rushes in to replace rising air
- Indraft velocity scales with fire intensity
- Effect decays with distance from fire front

References:
- Clark, T.L. et al. (2004). Coupled atmosphere-fire model simulations
  in various fuel types in complex terrain.
- Coen, J.L. (2018). Some requirements for simulating wildland fire behavior
  using coupled atmosphere-fire models.
- Forthofer, J.M. & Goodrick, S.L. (2011). Review of vortices in wildland fire.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


class FireAtmosphereParams(NamedTuple):
    """Parameters for fire-atmosphere coupling."""
    
    # Indraft strength coefficient
    # Higher = stronger inward wind near fire
    indraft_coefficient: float = 0.3
    
    # Maximum indraft velocity (m/s)
    # Limits unrealistic velocities
    max_indraft_velocity: float = 5.0
    
    # Characteristic decay distance (m)
    # Indraft decays exponentially beyond this distance
    decay_distance: float = 100.0
    
    # Heat release rate per unit ROS (kW/m per m/min ROS)
    # Typical values: 5000-20000 for forest fires
    heat_release_coefficient: float = 10000.0
    
    # Minimum fire intensity for coupling (kW/m)
    # Below this, fire doesn't significantly affect winds
    min_coupling_intensity: float = 1000.0
    
    # Plume rise coefficient
    # Affects how much vertical motion reduces horizontal wind
    plume_coefficient: float = 0.1
    
    # Fire front width for heat integration (m)
    front_width: float = 30.0


class CoupledWindField(NamedTuple):
    """Wind field with fire-induced modifications."""
    
    u: jnp.ndarray           # Total u component (background + fire-induced)
    v: jnp.ndarray           # Total v component
    u_background: jnp.ndarray  # Original background u
    v_background: jnp.ndarray  # Original background v
    u_fire: jnp.ndarray      # Fire-induced u component
    v_fire: jnp.ndarray      # Fire-induced v component
    indraft_magnitude: jnp.ndarray  # Magnitude of indraft velocity


def compute_fire_front_location(
    phi: jnp.ndarray,
    threshold: float = 0.0,
) -> jnp.ndarray:
    """
    Identify fire front location from level-set field.
    
    Fire front is where phi ≈ 0.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Level-set field (negative = burned, positive = unburned)
    threshold : float
        Value defining front location
        
    Returns
    -------
    front_mask : jnp.ndarray
        Soft mask indicating proximity to fire front (0-1)
    """
    # Distance from front (in level-set units)
    distance = jnp.abs(phi - threshold)
    
    # Soft mask: 1 at front, decaying away
    # Using Gaussian-like falloff
    front_mask = jnp.exp(-distance**2 / 0.001)
    
    return front_mask


def compute_fire_front_normal(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute normal direction to fire front.
    
    Normal points from burned toward unburned (outward).
    
    Parameters
    ----------
    phi : jnp.ndarray
        Level-set field
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    nx, ny : jnp.ndarray
        Components of unit normal vector
    """
    # Gradient of phi points toward unburned
    dphidx = jnp.zeros_like(phi)
    dphidy = jnp.zeros_like(phi)
    
    dphidx = dphidx.at[:, 1:-1].set((phi[:, 2:] - phi[:, :-2]) / (2 * dx))
    dphidy = dphidy.at[1:-1, :].set((phi[2:, :] - phi[:-2, :]) / (2 * dy))
    
    # Handle boundaries
    dphidx = dphidx.at[:, 0].set((phi[:, 1] - phi[:, 0]) / dx)
    dphidx = dphidx.at[:, -1].set((phi[:, -1] - phi[:, -2]) / dx)
    dphidy = dphidy.at[0, :].set((phi[1, :] - phi[0, :]) / dy)
    dphidy = dphidy.at[-1, :].set((phi[-1, :] - phi[-2, :]) / dy)
    
    # Normalize
    magnitude = jnp.sqrt(dphidx**2 + dphidy**2) + 1e-10
    nx = dphidx / magnitude
    ny = dphidy / magnitude
    
    return nx, ny


def compute_heat_release_rate(
    phi: jnp.ndarray,
    phi_prev: jnp.ndarray,
    dt: float,
    dx: float,
    dy: float,
    fuel_load: jnp.ndarray,
    heat_content: float = 18000.0,  # kJ/kg
) -> jnp.ndarray:
    """
    Compute heat release rate from fire spread.
    
    HRR = H * w * dA/dt
    
    Where dA/dt is the rate of area being burned.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Current level-set field
    phi_prev : jnp.ndarray
        Previous level-set field
    dt : float
        Time step (seconds)
    dx, dy : float
        Grid spacing (m)
    fuel_load : jnp.ndarray
        Fuel load (kg/m²)
    heat_content : float
        Heat of combustion (kJ/kg)
        
    Returns
    -------
    hrr : jnp.ndarray
        Heat release rate (kW/m²)
    """
    # Area burned in this timestep
    # burned_prev = phi_prev < 0
    # burned_now = phi < 0
    # newly_burned = burned_now & ~burned_prev
    
    # Smooth version: rate of phi decrease
    dphi_dt = (phi_prev - phi) / dt  # Positive where fire advancing
    dphi_dt = jnp.maximum(dphi_dt, 0.0)  # Only count fire spread, not retreat
    
    # Convert phi change rate to area rate
    # This is approximate - assumes phi units relate to distance
    area_rate = dphi_dt * dx  # m²/s per cell width
    
    # Heat release
    hrr = heat_content * fuel_load * area_rate / (dx * dy)  # kW/m²
    
    return hrr


def compute_fire_intensity_at_front(
    phi: jnp.ndarray,
    ros: jnp.ndarray,
    fuel_load: jnp.ndarray,
    dx: float,
    dy: float,
    heat_content: float = 18000.0,
    params: FireAtmosphereParams = FireAtmosphereParams(),
) -> jnp.ndarray:
    """
    Compute Byram's fire line intensity near the front.
    
    I = H * w * R
    
    Parameters
    ----------
    phi : jnp.ndarray
        Level-set field
    ros : jnp.ndarray
        Rate of spread (m/min)
    fuel_load : jnp.ndarray
        Fuel load (kg/m²)
    dx, dy : float
        Grid spacing
    heat_content : float
        Heat of combustion (kJ/kg)
    params : FireAtmosphereParams
        Model parameters
        
    Returns
    -------
    intensity : jnp.ndarray
        Fire line intensity (kW/m) near front, zero elsewhere
    """
    # Find fire front
    front_mask = compute_fire_front_location(phi)
    
    # Byram's intensity
    ros_ms = ros / 60.0  # Convert m/min to m/s
    intensity = heat_content * fuel_load * ros_ms  # kW/m
    
    # Apply only near front
    intensity = intensity * front_mask
    
    return intensity


def compute_indraft_velocity(
    phi: jnp.ndarray,
    intensity: jnp.ndarray,
    dx: float,
    dy: float,
    params: FireAtmosphereParams = FireAtmosphereParams(),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute fire-induced indraft velocity field.
    
    Air is drawn toward the fire front due to buoyancy-driven
    convection. The indraft velocity scales with intensity and
    decays with distance from the front.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Level-set field
    intensity : jnp.ndarray
        Fire line intensity (kW/m)
    dx, dy : float
        Grid spacing
    params : FireAtmosphereParams
        Model parameters
        
    Returns
    -------
    u_indraft, v_indraft : jnp.ndarray
        Indraft velocity components (m/s)
    magnitude : jnp.ndarray
        Indraft magnitude (m/s)
    """
    ny, nx = phi.shape
    
    # Fire front normal (points outward from fire)
    nx_front, ny_front = compute_fire_front_normal(phi, dx, dy)
    
    # Fire front location mask
    front_mask = compute_fire_front_location(phi)
    
    # Distance from fire front (approximate using phi directly)
    # For a well-behaved level set, |phi| ≈ distance
    distance = jnp.abs(phi) * dx  # Convert to meters (approximate)
    distance = jnp.maximum(distance, dx)  # Minimum distance
    
    # Indraft magnitude from each front cell affecting each grid cell
    # We need to integrate contributions from all front cells
    
    # Simplified approach: compute indraft based on local intensity
    # and direction toward nearest high-intensity region
    
    # Find center of mass of intensity (fire centroid)
    y_idx, x_idx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
    
    total_intensity = jnp.sum(intensity) + 1e-10
    x_center = jnp.sum(x_idx * intensity) / total_intensity
    y_center = jnp.sum(y_idx * intensity) / total_intensity
    
    # Vector pointing toward fire center
    dx_to_fire = x_center - x_idx
    dy_to_fire = y_center - y_idx
    dist_to_fire = jnp.sqrt(dx_to_fire**2 + dy_to_fire**2) * dx + 1e-10
    
    # Normalize
    dir_x = dx_to_fire / (jnp.sqrt(dx_to_fire**2 + dy_to_fire**2) + 1e-10)
    dir_y = dy_to_fire / (jnp.sqrt(dx_to_fire**2 + dy_to_fire**2) + 1e-10)
    
    # Indraft magnitude: scales with total intensity, decays with distance
    # Using inverse-distance weighting
    effective_intensity = jnp.sum(intensity * front_mask) / (front_mask.sum() + 1)
    
    indraft_magnitude = (params.indraft_coefficient * 
                        jnp.sqrt(effective_intensity / 10000.0) *  # Normalize by typical intensity
                        jnp.exp(-dist_to_fire / params.decay_distance))
    
    # Cap maximum indraft
    indraft_magnitude = jnp.minimum(indraft_magnitude, params.max_indraft_velocity)
    
    # Only apply where fire is intense enough
    indraft_magnitude = jnp.where(
        effective_intensity > params.min_coupling_intensity,
        indraft_magnitude,
        0.0
    )
    
    # Don't apply indraft inside the fire (phi < 0)
    indraft_magnitude = jnp.where(phi > 0, indraft_magnitude, 0.0)
    
    # Compute velocity components
    u_indraft = indraft_magnitude * dir_x
    v_indraft = indraft_magnitude * dir_y
    
    return u_indraft, v_indraft, indraft_magnitude


def compute_indraft_field_integral(
    phi: jnp.ndarray,
    intensity: jnp.ndarray,
    dx: float,
    dy: float,
    params: FireAtmosphereParams = FireAtmosphereParams(),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute indraft field by integrating contributions from all front cells.
    
    This is more accurate but slower than the simplified version.
    Each cell feels the combined pull from all fire front cells,
    weighted by their intensity and distance.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Level-set field
    intensity : jnp.ndarray  
        Fire line intensity at each cell
    dx, dy : float
        Grid spacing
    params : FireAtmosphereParams
        Model parameters
        
    Returns
    -------
    u_indraft, v_indraft : jnp.ndarray
        Indraft velocity components
    """
    ny, nx = phi.shape
    
    # Create coordinate grids
    y_idx, x_idx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
    x_meters = x_idx * dx
    y_meters = y_idx * dy
    
    # Find front cells (high intensity)
    front_mask = intensity > params.min_coupling_intensity
    
    # Initialize indraft field
    u_indraft = jnp.zeros_like(phi)
    v_indraft = jnp.zeros_like(phi)
    
    # For efficiency, use convolution-based approach
    # Create a kernel that represents the indraft contribution pattern
    
    # Kernel size based on decay distance
    kernel_radius = int(3 * params.decay_distance / dx) + 1
    kernel_size = 2 * kernel_radius + 1
    
    # Create distance kernel
    ky, kx = jnp.meshgrid(
        jnp.arange(-kernel_radius, kernel_radius + 1),
        jnp.arange(-kernel_radius, kernel_radius + 1),
        indexing='ij'
    )
    kx_m = kx * dx
    ky_m = ky * dy
    k_dist = jnp.sqrt(kx_m**2 + ky_m**2) + 1e-10
    
    # Direction kernels (pointing toward center)
    dir_x_kernel = -kx_m / k_dist
    dir_y_kernel = -ky_m / k_dist
    
    # Magnitude decay with distance
    mag_kernel = params.indraft_coefficient * jnp.exp(-k_dist / params.decay_distance)
    mag_kernel = jnp.where(k_dist < dx, 0.0, mag_kernel)  # Zero at center
    
    # Weighted direction kernels
    kernel_u = mag_kernel * dir_x_kernel
    kernel_v = mag_kernel * dir_y_kernel
    
    # Normalize kernels
    kernel_u = kernel_u / (jnp.sum(jnp.abs(kernel_u)) + 1e-10)
    kernel_v = kernel_v / (jnp.sum(jnp.abs(kernel_v)) + 1e-10)
    
    # Convolve intensity field with direction kernels
    # Weight by sqrt(intensity) since indraft ~ sqrt(buoyancy)
    weighted_intensity = jnp.sqrt(intensity / 10000.0) * front_mask
    
    # Reshape for convolution - ensure matching dtypes
    wi_4d = weighted_intensity[None, :, :, None]
    ku_4d = kernel_u.astype(weighted_intensity.dtype)[:, :, None, None]
    kv_4d = kernel_v.astype(weighted_intensity.dtype)[:, :, None, None]
    
    u_indraft = jax.lax.conv_general_dilated(
        wi_4d, ku_4d,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )[0, :, :, 0]
    
    v_indraft = jax.lax.conv_general_dilated(
        wi_4d, kv_4d,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )[0, :, :, 0]
    
    # Scale to reasonable velocities
    u_indraft = u_indraft * params.max_indraft_velocity
    v_indraft = v_indraft * params.max_indraft_velocity
    
    # Only apply ahead of fire
    u_indraft = jnp.where(phi > 0, u_indraft, 0.0)
    v_indraft = jnp.where(phi > 0, v_indraft, 0.0)
    
    return u_indraft, v_indraft


def compute_plume_wind_reduction(
    phi: jnp.ndarray,
    intensity: jnp.ndarray,
    params: FireAtmosphereParams = FireAtmosphereParams(),
) -> jnp.ndarray:
    """
    Compute wind reduction factor due to vertical plume motion.
    
    Strong convection above the fire reduces horizontal wind speed
    at the surface (momentum carried upward).
    
    Parameters
    ----------
    phi : jnp.ndarray
        Level-set field
    intensity : jnp.ndarray
        Fire line intensity
    params : FireAtmosphereParams
        Model parameters
        
    Returns
    -------
    reduction_factor : jnp.ndarray
        Factor to multiply background wind (0-1)
    """
    # Find fire location
    fire_mask = phi < 0
    
    # Strong plume reduces horizontal wind
    # Effect is strongest directly over fire
    plume_strength = jnp.sqrt(intensity / 10000.0) * params.plume_coefficient
    plume_strength = jnp.minimum(plume_strength, 0.5)  # Max 50% reduction
    
    # Apply only over/near fire
    front_proximity = jnp.exp(-phi**2 / 0.01)
    
    reduction = 1.0 - plume_strength * front_proximity
    reduction = jnp.clip(reduction, 0.5, 1.0)
    
    return reduction


def couple_wind_to_fire(
    u_background: jnp.ndarray,
    v_background: jnp.ndarray,
    phi: jnp.ndarray,
    ros: jnp.ndarray,
    fuel_load: jnp.ndarray,
    dx: float,
    dy: float,
    params: FireAtmosphereParams = FireAtmosphereParams(),
) -> CoupledWindField:
    """
    Compute fire-coupled wind field.
    
    Main entry point for fire-atmosphere coupling.
    
    Parameters
    ----------
    u_background, v_background : jnp.ndarray
        Background wind components (m/s)
    phi : jnp.ndarray
        Level-set field
    ros : jnp.ndarray
        Rate of spread (m/min)
    fuel_load : jnp.ndarray
        Fuel load (kg/m²)
    dx, dy : float
        Grid spacing (m)
    params : FireAtmosphereParams
        Model parameters
        
    Returns
    -------
    coupled_wind : CoupledWindField
        Wind field with fire-induced modifications
    """
    # Compute fire intensity
    intensity = compute_fire_intensity_at_front(phi, ros, fuel_load, dx, dy, params=params)
    
    # Compute indraft
    u_indraft, v_indraft, indraft_mag = compute_indraft_velocity(
        phi, intensity, dx, dy, params
    )
    
    # Compute plume reduction
    plume_reduction = compute_plume_wind_reduction(phi, intensity, params)
    
    # Combine: reduced background + indraft
    u_total = u_background * plume_reduction + u_indraft
    v_total = v_background * plume_reduction + v_indraft
    
    return CoupledWindField(
        u=u_total,
        v=v_total,
        u_background=u_background,
        v_background=v_background,
        u_fire=u_indraft,
        v_fire=v_indraft,
        indraft_magnitude=indraft_mag,
    )


def update_ros_for_coupling(
    ros: jnp.ndarray,
    coupled_wind: CoupledWindField,
    original_wind_speed: float,
    wind_sensitivity: float = 0.5,
) -> jnp.ndarray:
    """
    Adjust ROS based on fire-modified wind.
    
    The fire's own indraft can enhance or reduce spread rate
    depending on direction.
    
    Parameters
    ----------
    ros : jnp.ndarray
        Original ROS from fire model
    coupled_wind : CoupledWindField
        Fire-coupled wind field
    original_wind_speed : float
        Original background wind speed
    wind_sensitivity : float
        How much ROS changes with wind (0-1)
        
    Returns
    -------
    adjusted_ros : jnp.ndarray
        ROS adjusted for fire-induced winds
    """
    # Compute effective wind speed
    effective_speed = jnp.sqrt(coupled_wind.u**2 + coupled_wind.v**2)
    
    # Wind change ratio
    speed_ratio = effective_speed / (original_wind_speed + 0.1)
    
    # Adjust ROS (simplified - real relationship is nonlinear)
    # ROS increases roughly linearly with wind for moderate winds
    ros_adjustment = 1.0 + wind_sensitivity * (speed_ratio - 1.0)
    ros_adjustment = jnp.clip(ros_adjustment, 0.5, 2.0)
    
    return ros * ros_adjustment


# =============================================================================
# Integration helpers
# =============================================================================

def should_apply_coupling(
    total_intensity: float,
    params: FireAtmosphereParams = FireAtmosphereParams(),
) -> bool:
    """
    Determine if fire-atmosphere coupling is significant.
    
    For small/low-intensity fires, coupling can be skipped
    to save computation.
    """
    return total_intensity > params.min_coupling_intensity * 10


def estimate_coupling_strength(
    phi: jnp.ndarray,
    ros: jnp.ndarray,
    fuel_load: jnp.ndarray,
) -> float:
    """
    Estimate overall fire-atmosphere coupling strength.
    
    Returns a value 0-1 indicating how much coupling matters.
    0 = negligible, 1 = very strong coupling
    """
    # Fire perimeter length (approximate)
    front_mask = jnp.abs(phi) < 0.001
    perimeter_cells = jnp.sum(front_mask)
    
    # Mean intensity at front
    front_ros = jnp.where(front_mask, ros, 0.0)
    front_fuel = jnp.where(front_mask, fuel_load, 0.0)
    mean_intensity = jnp.sum(front_ros * front_fuel * 18000 / 60) / (perimeter_cells + 1)
    
    # Coupling strength scales with intensity
    # Normalize: 10000 kW/m is moderate, 50000 kW/m is strong
    strength = float(jnp.clip(mean_intensity / 50000.0, 0.0, 1.0))
    
    return strength
