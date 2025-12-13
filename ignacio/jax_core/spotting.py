"""
Differentiable Spotting Model for Level-Set Fire Spread.

Implements ember transport and spot fire ignition as a differentiable
process that can be calibrated against observed fire behavior.

Physics:
- Embers (firebrands) are lofted by convective columns above high-intensity fires
- Wind carries embers downwind where they land and potentially ignite spot fires
- Landing distribution follows approximately log-normal with distance
- Ignition probability depends on fuel receptivity and ember characteristics

References:
- Albini, F.A. (1979). Spot fire distance from burning trees
- Sardoy et al. (2007). Modeling transport and combustion of firebrands
- Perryman et al. (2013). A cellular automata model for spot fire spread
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


class SpottingParams(NamedTuple):
    """
    Parameters controlling spotting behavior.
    
    These can be calibrated against observed fire data.
    """
    # Spotting activation threshold (fire intensity, kW/m)
    intensity_threshold: float = 2000.0
    
    # Maximum spotting distance (m)
    max_spot_distance: float = 500.0
    
    # Mean spotting distance as fraction of max (for log-normal)
    mean_distance_fraction: float = 0.3
    
    # Standard deviation of log-distance
    log_distance_std: float = 0.8
    
    # Base ignition probability for landed embers
    base_ignition_prob: float = 0.3
    
    # Wind speed multiplier for distance (distance = base * (1 + wind_mult * ws))
    wind_distance_mult: float = 0.05
    
    # Spotting rate (spots per meter of fire front per minute at threshold intensity)
    spotting_rate: float = 0.001
    
    # Intensity exponent (spotting ~ intensity^exp)
    intensity_exponent: float = 1.5
    
    # Temperature for soft thresholding (higher = sharper)
    temperature: float = 0.1


def compute_fire_intensity(
    ros: jnp.ndarray,
    fuel_load: Optional[jnp.ndarray] = None,
    heat_content: float = 18000.0,  # kJ/kg for typical forest fuels
) -> jnp.ndarray:
    """
    Compute Byram's fire intensity (kW/m).
    
    I = H * W * R
    
    where:
    - H = heat content (kJ/kg)
    - W = fuel load consumed (kg/m²)
    - R = rate of spread (m/s)
    
    Parameters
    ----------
    ros : jnp.ndarray
        Rate of spread (m/min)
    fuel_load : jnp.ndarray, optional
        Fuel load consumed (kg/m²). Default ~2 kg/m² for moderate fuel
    heat_content : float
        Heat content of fuel (kJ/kg)
        
    Returns
    -------
    intensity : jnp.ndarray
        Fire line intensity (kW/m)
    """
    if fuel_load is None:
        fuel_load = jnp.ones_like(ros) * 2.0  # Default 2 kg/m²
    
    # Convert ROS from m/min to m/s
    ros_ms = ros / 60.0
    
    # Byram's intensity (kW/m = kJ/s/m)
    intensity = heat_content * fuel_load * ros_ms
    
    return intensity


def compute_spotting_probability_field(
    phi: jnp.ndarray,
    intensity: jnp.ndarray,
    wind_speed: float,
    wind_direction: float,
    dx: float,
    dy: float,
    params: SpottingParams = SpottingParams(),
) -> jnp.ndarray:
    """
    Compute probability of spot fire ignition at each grid cell.
    
    This creates a "spotting field" that represents where embers from
    the current fire front are likely to land and ignite.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Level-set field (ny, nx). Fire front at phi=0, burned where phi<0
    intensity : jnp.ndarray
        Fire intensity at each cell (kW/m)
    wind_speed : float
        Wind speed (km/h)
    wind_direction : float
        Wind direction (degrees, direction fire spreads TO)
    dx, dy : float
        Grid spacing (m)
    params : SpottingParams
        Spotting parameters
        
    Returns
    -------
    spot_prob : jnp.ndarray
        Probability of spot fire ignition at each cell (0-1)
    """
    ny, nx = phi.shape
    
    # Find the fire front (cells near phi=0)
    epsilon = jnp.sqrt(dx**2 + dy**2)
    front_mask = jax.nn.sigmoid(-jnp.abs(phi) / (epsilon * 0.5) + 2)  # ~1 at front, ~0 away
    
    # Compute effective intensity at front (soft threshold)
    intensity_at_front = intensity * front_mask
    
    # Spotting activation (soft threshold on intensity)
    spot_activation = jax.nn.sigmoid(
        (intensity_at_front - params.intensity_threshold) / 
        (params.intensity_threshold * params.temperature)
    )
    
    # Wind direction in radians (direction embers travel)
    wind_rad = jnp.radians(wind_direction)
    
    # Maximum spotting distance (increases with wind)
    ws_ms = wind_speed / 3.6  # km/h to m/s
    max_dist = params.max_spot_distance * (1.0 + params.wind_distance_mult * ws_ms)
    
    # Create coordinate grids
    y_coords = jnp.arange(ny) * jnp.abs(dy)
    x_coords = jnp.arange(nx) * dx
    Y, X = jnp.meshgrid(y_coords, x_coords, indexing='ij')
    
    # For each potential source cell, compute landing probability at all target cells
    # This is expensive, so we use a convolution-like approach with a directional kernel
    
    # Create a directional spotting kernel
    kernel_size = int(max_dist / min(abs(dx), abs(dy))) + 1
    kernel_size = min(kernel_size, 51)  # Limit kernel size for performance
    
    # Kernel coordinates relative to center
    k_range = jnp.arange(-kernel_size, kernel_size + 1)
    KY, KX = jnp.meshgrid(k_range * abs(dy), k_range * dx, indexing='ij')
    
    # Distance from center
    K_dist = jnp.sqrt(KX**2 + KY**2) + 1e-6
    
    # Angle from center (direction of ember travel)
    K_angle = jnp.arctan2(KY, KX)
    
    # Angular alignment with wind (embers travel downwind)
    angle_diff = K_angle - wind_rad
    angle_alignment = jnp.cos(angle_diff)
    
    # Only spot downwind (positive alignment)
    downwind_mask = jax.nn.sigmoid(angle_alignment / 0.1)
    
    # Distance probability (log-normal distribution)
    mean_dist = max_dist * params.mean_distance_fraction
    log_mean = jnp.log(mean_dist + 1e-6)
    log_dist = jnp.log(K_dist + 1e-6)
    
    # Log-normal PDF (normalized)
    dist_prob = jnp.exp(-0.5 * ((log_dist - log_mean) / params.log_distance_std)**2)
    dist_prob = dist_prob / (K_dist * params.log_distance_std * jnp.sqrt(2 * jnp.pi))
    
    # Combine: directional * distance * downwind
    kernel = dist_prob * downwind_mask * (K_dist < max_dist).astype(jnp.float32)
    
    # Normalize kernel
    kernel = kernel / (jnp.sum(kernel) + 1e-6)
    
    # The source strength is intensity * activation * (1 - burned)
    # We only emit from burned side near the front
    burned_mask = jax.nn.sigmoid(-phi / epsilon)
    source_strength = spot_activation * (intensity_at_front / params.intensity_threshold) ** params.intensity_exponent
    source_strength = source_strength * burned_mask * front_mask
    
    # Convolve source with kernel to get landing probability
    # Use scipy-style 2D convolution via JAX
    # Reshape for conv: (batch, height, width, channels)
    source_4d = source_strength[None, :, :, None]  # (1, ny, nx, 1)
    kernel_4d = kernel[:, :, None, None]  # (ky, kx, 1, 1)
    
    # Pad source to handle boundaries
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    source_padded = jnp.pad(source_4d, ((0, 0), (pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant')
    
    # 2D convolution using lax.conv_general_dilated
    landing_prob = jax.lax.conv_general_dilated(
        source_padded,  # (N, H, W, C_in)
        kernel_4d,      # (H, W, C_in, C_out)
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )[0, :, :, 0]  # Extract (ny, nx)
    
    # Apply base ignition probability
    spot_prob = landing_prob * params.base_ignition_prob
    
    # Only ignite in unburned areas
    unburned_mask = jax.nn.sigmoid(phi / epsilon)
    spot_prob = spot_prob * unburned_mask
    
    # Clip to [0, 1]
    spot_prob = jnp.clip(spot_prob, 0, 1)
    
    return spot_prob


def apply_spotting(
    phi: jnp.ndarray,
    spot_prob: jnp.ndarray,
    key: Optional[jax.random.PRNGKey] = None,
    deterministic: bool = True,
) -> jnp.ndarray:
    """
    Apply spotting to level-set field.
    
    In deterministic mode (for calibration), spots are created proportionally
    to probability. In stochastic mode, random sampling determines spots.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Current level-set field
    spot_prob : jnp.ndarray
        Spotting probability at each cell
    key : PRNGKey, optional
        Random key for stochastic mode
    deterministic : bool
        If True, use soft (differentiable) spotting
        
    Returns
    -------
    phi_spotted : jnp.ndarray
        Level-set field with spot fires
    """
    if deterministic:
        # Soft spotting: reduce phi proportionally to spot probability
        # This creates "partial burns" that grow in subsequent steps
        # High probability cells get pushed toward burned (phi < 0)
        
        # Scale factor: how much to reduce phi
        epsilon = 0.001  # Small value to create initial spot
        phi_reduction = spot_prob * epsilon
        
        # Only reduce in unburned areas
        phi_spotted = phi - phi_reduction
        
    else:
        # Stochastic spotting: random sampling
        if key is None:
            key = jax.random.PRNGKey(0)
        
        random_vals = jax.random.uniform(key, shape=phi.shape)
        spots = random_vals < spot_prob
        
        # Create spots by setting phi to small negative value
        spot_phi = -0.0001
        phi_spotted = jnp.where(spots & (phi > 0), spot_phi, phi)
    
    return phi_spotted


def evolve_phi_with_spotting(
    phi: jnp.ndarray,
    grids,  # LevelSetGrids
    t: int,
    dt: float,
    wind_speed: float,
    wind_direction: float,
    dx: float,
    dy: float,
    spotting_params: SpottingParams = SpottingParams(),
    use_ellipse: bool = True,
) -> jnp.ndarray:
    """
    Evolve level-set field with both propagation and spotting.
    
    This combines the standard level-set evolution with the spotting model.
    
    Parameters
    ----------
    phi : jnp.ndarray
        Current level-set field
    grids : LevelSetGrids
        Fire spread parameters
    t : int
        Current time step index
    dt : float
        Time step (minutes)
    wind_speed : float
        Wind speed (km/h)
    wind_direction : float
        Wind direction (degrees)
    dx, dy : float
        Grid spacing (m)
    spotting_params : SpottingParams
        Spotting parameters
    use_ellipse : bool
        Whether to use elliptical spread model
        
    Returns
    -------
    phi_new : jnp.ndarray
        Updated level-set field
    """
    from .levelset import evolve_phi, get_ros_at_time
    
    # Get ROS at current time
    ros, bros, fros, raz = get_ros_at_time(grids, t)
    
    # Standard level-set evolution (propagation)
    phi_propagated = evolve_phi(phi, grids, t, dt, use_ellipse)
    
    # Compute fire intensity
    intensity = compute_fire_intensity(ros)
    
    # Compute spotting probability
    spot_prob = compute_spotting_probability_field(
        phi, intensity, wind_speed, wind_direction,
        dx, dy, spotting_params
    )
    
    # Apply spotting
    phi_spotted = apply_spotting(phi_propagated, spot_prob, deterministic=True)
    
    return phi_spotted


def simulate_fire_with_spotting(
    grids,  # LevelSetGrids
    x_ign: float,
    y_ign: float,
    wind_speed: float,
    wind_direction: float,
    n_steps: int,
    dt: float,
    initial_radius: float,
    spotting_params: SpottingParams = SpottingParams(),
    use_ellipse: bool = True,
) -> jnp.ndarray:
    """
    Simulate fire spread with spotting using level-set method.
    
    Parameters
    ----------
    grids : LevelSetGrids
        Fire spread parameter grids
    x_ign, y_ign : float
        Ignition point coordinates
    wind_speed : float
        Wind speed (km/h)
    wind_direction : float
        Wind direction (degrees, direction fire spreads TO)
    n_steps : int
        Number of time steps
    dt : float
        Time step (minutes)
    initial_radius : float
        Initial fire radius
    spotting_params : SpottingParams
        Spotting model parameters
    use_ellipse : bool
        Whether to use elliptical spread
        
    Returns
    -------
    phi : jnp.ndarray
        Final level-set field
    """
    from .levelset import initialize_phi
    
    # Initialize
    phi = initialize_phi(grids.x_coords, grids.y_coords, x_ign, y_ign, initial_radius)
    
    # Grid spacing
    dx = float(grids.x_coords[1] - grids.x_coords[0])
    dy = float(grids.y_coords[1] - grids.y_coords[0])
    
    # Evolve with spotting
    def body_fn(i, phi):
        return evolve_phi_with_spotting(
            phi, grids, i, dt,
            wind_speed, wind_direction, dx, dy,
            spotting_params, use_ellipse
        )
    
    phi_final = lax.fori_loop(0, n_steps, body_fn, phi)
    
    return phi_final


# Calibration support

def calibrate_spotting_params(
    grids,
    x_ign: float,
    y_ign: float,
    wind_speed: float,
    wind_direction: float,
    n_steps: int,
    dt: float,
    initial_radius: float,
    obs_mask: jnp.ndarray,
    obs_area: float,
    initial_params: SpottingParams = SpottingParams(),
    n_iterations: int = 50,
    learning_rate: float = 0.01,
    verbose: bool = True,
) -> Tuple[SpottingParams, list]:
    """
    Calibrate spotting parameters against observed fire.
    
    Optimizes key spotting parameters:
    - max_spot_distance
    - base_ignition_prob
    - spotting_rate
    
    Parameters
    ----------
    grids : LevelSetGrids
        Fire spread parameters
    x_ign, y_ign : float
        Ignition coordinates
    wind_speed, wind_direction : float
        Wind conditions
    n_steps : int
        Number of time steps
    dt : float
        Time step
    initial_radius : float
        Initial fire radius
    obs_mask : jnp.ndarray
        Observed burned area (1=burned)
    obs_area : float
        Total observed area
    initial_params : SpottingParams
        Starting parameters
    n_iterations : int
        Optimization iterations
    learning_rate : float
        Gradient descent step size
    verbose : bool
        Print progress
        
    Returns
    -------
    final_params : SpottingParams
        Calibrated parameters
    loss_history : list
        Loss at each iteration
    """
    from .levelset import initialize_phi, compute_burned_area
    from .levelset_calibration import iou_loss, area_loss
    
    dx = float(grids.x_coords[1] - grids.x_coords[0])
    dy = float(grids.y_coords[1] - grids.y_coords[0])
    
    def loss_fn(params_vec):
        # Unpack parameters
        spot_params = SpottingParams(
            max_spot_distance=params_vec[0],
            base_ignition_prob=params_vec[1],
            spotting_rate=params_vec[2],
            # Keep other params fixed
            intensity_threshold=initial_params.intensity_threshold,
            mean_distance_fraction=initial_params.mean_distance_fraction,
            log_distance_std=initial_params.log_distance_std,
            wind_distance_mult=initial_params.wind_distance_mult,
            intensity_exponent=initial_params.intensity_exponent,
            temperature=initial_params.temperature,
        )
        
        phi = simulate_fire_with_spotting(
            grids, x_ign, y_ign, wind_speed, wind_direction,
            n_steps, dt, initial_radius, spot_params
        )
        
        # Combined IoU + area loss
        iou = iou_loss(phi, obs_mask, dx, dy)
        area = compute_burned_area(phi, dx, dy)
        area_l = area_loss(area, obs_area)
        
        return 0.5 * iou + 0.5 * area_l
    
    # Gradient function
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    
    # Initialize
    params_vec = jnp.array([
        initial_params.max_spot_distance,
        initial_params.base_ignition_prob,
        initial_params.spotting_rate,
    ])
    
    loss_history = []
    
    if verbose:
        print(f"Calibrating spotting: {n_iterations} iterations")
        print(f"Initial: dist={params_vec[0]:.0f}m, prob={params_vec[1]:.3f}, rate={params_vec[2]:.5f}")
    
    for i in range(n_iterations):
        loss, grad = value_and_grad_fn(params_vec)
        loss_history.append(float(loss))
        
        # Update with bounds
        params_vec = params_vec - learning_rate * grad
        params_vec = jnp.array([
            jnp.clip(params_vec[0], 100, 2000),   # max_distance: 100-2000m
            jnp.clip(params_vec[1], 0.01, 0.9),  # ignition_prob: 1-90%
            jnp.clip(params_vec[2], 1e-5, 0.1),  # spotting_rate
        ])
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Iter {i+1}: loss={loss:.4f}, dist={params_vec[0]:.0f}m, prob={params_vec[1]:.3f}")
    
    final_params = SpottingParams(
        max_spot_distance=float(params_vec[0]),
        base_ignition_prob=float(params_vec[1]),
        spotting_rate=float(params_vec[2]),
        intensity_threshold=initial_params.intensity_threshold,
        mean_distance_fraction=initial_params.mean_distance_fraction,
        log_distance_std=initial_params.log_distance_std,
        wind_distance_mult=initial_params.wind_distance_mult,
        intensity_exponent=initial_params.intensity_exponent,
        temperature=initial_params.temperature,
    )
    
    if verbose:
        print(f"Final: dist={final_params.max_spot_distance:.0f}m, "
              f"prob={final_params.base_ignition_prob:.3f}, "
              f"rate={final_params.spotting_rate:.5f}")
    
    return final_params, loss_history
