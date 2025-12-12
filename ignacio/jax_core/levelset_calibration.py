"""
Calibration for Level-Set Fire Spread Model.

Provides differentiable calibration of fire spread parameters using
observed fire perimeters as targets.
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Callable
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from .levelset import (
    LevelSetGrids,
    simulate_fire_levelset,
    compute_burned_area,
    compute_burned_area_hard,
)


class CalibrationParams(NamedTuple):
    """
    Calibration parameters for fire spread.
    
    These are multiplicative adjustments to the base ROS values.
    """
    ros_scale: float = 1.0      # Scale factor for head fire ROS
    bros_scale: float = 1.0     # Scale factor for back fire ROS
    fros_scale: float = 1.0     # Scale factor for flank fire ROS
    # Could add more: wind_adj, moisture_adj, etc.


class CalibrationResult(NamedTuple):
    """Results from calibration."""
    params: CalibrationParams
    loss_history: list
    params_history: list
    final_loss: float
    n_iterations: int


def apply_calibration_params(
    grids: LevelSetGrids,
    params: CalibrationParams,
) -> LevelSetGrids:
    """
    Apply calibration parameters to ROS grids.
    
    Parameters
    ----------
    grids : LevelSetGrids
        Original parameter grids
    params : CalibrationParams
        Calibration parameters
        
    Returns
    -------
    adjusted_grids : LevelSetGrids
        Grids with adjusted ROS values
    """
    return LevelSetGrids(
        x_coords=grids.x_coords,
        y_coords=grids.y_coords,
        ros=grids.ros * params.ros_scale,
        bros=grids.bros * params.bros_scale,
        fros=grids.fros * params.fros_scale,
        raz=grids.raz,
    )


def area_loss(
    pred_area: jnp.ndarray,
    obs_area: float,
) -> jnp.ndarray:
    """
    Simple area difference loss.
    
    L = ((pred - obs) / obs)²
    """
    rel_diff = (pred_area - obs_area) / (obs_area + 1e-8)
    return rel_diff ** 2


def iou_loss(
    phi: jnp.ndarray,
    obs_mask: jnp.ndarray,
    dx: float,
    dy: float,
) -> jnp.ndarray:
    """
    Intersection over Union loss (differentiable approximation).
    
    Parameters
    ----------
    phi : jnp.ndarray
        Predicted level-set field (negative = burned)
    obs_mask : jnp.ndarray
        Observed burned area mask (1 = burned, 0 = unburned)
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    loss : jnp.ndarray
        1 - IoU (so minimizing this maximizes IoU)
    """
    # Smooth approximation of burned mask
    epsilon = jnp.sqrt(dx**2 + dy**2)
    pred_mask = jax.nn.sigmoid(-phi / epsilon)
    
    # Intersection: min(pred, obs) for each cell
    intersection = jnp.sum(pred_mask * obs_mask)
    
    # Union: max(pred, obs) = pred + obs - intersection
    union = jnp.sum(pred_mask) + jnp.sum(obs_mask) - intersection
    
    iou = intersection / (union + 1e-8)
    
    return 1.0 - iou


def combined_loss(
    phi: jnp.ndarray,
    obs_mask: jnp.ndarray,
    obs_area: float,
    dx: float,
    dy: float,
    area_weight: float = 0.5,
    iou_weight: float = 0.5,
) -> jnp.ndarray:
    """
    Combined loss: area difference + IoU.
    """
    # Compute predicted area
    epsilon = jnp.sqrt(dx**2 + dy**2)
    pred_mask = jax.nn.sigmoid(-phi / epsilon)
    pred_area = jnp.sum(pred_mask) * jnp.abs(dx * dy)
    
    # Area loss
    a_loss = area_loss(pred_area, obs_area)
    
    # IoU loss
    i_loss = iou_loss(phi, obs_mask, dx, dy)
    
    return area_weight * a_loss + iou_weight * i_loss


def forward_model(
    params: CalibrationParams,
    grids: LevelSetGrids,
    x_ign: float,
    y_ign: float,
    n_steps: int,
    dt: float,
    initial_radius: float,
) -> jnp.ndarray:
    """
    Run forward model with calibration parameters.
    
    Returns the level-set field phi.
    """
    adjusted_grids = apply_calibration_params(grids, params)
    phi = simulate_fire_levelset(
        adjusted_grids, x_ign, y_ign, n_steps, dt, initial_radius, use_ellipse=True
    )
    return phi


def calibrate_to_observed(
    grids: LevelSetGrids,
    x_ign: float,
    y_ign: float,
    n_steps: int,
    dt: float,
    initial_radius: float,
    obs_mask: jnp.ndarray,
    obs_area: float,
    initial_params: Optional[CalibrationParams] = None,
    n_iterations: int = 50,
    learning_rate: float = 0.1,
    area_weight: float = 0.5,
    iou_weight: float = 0.5,
    verbose: bool = True,
    subsample_steps: Optional[int] = None,
    max_grid_cells: int = 250000,
) -> CalibrationResult:
    """
    Calibrate fire spread parameters to match observed fire.
    
    Uses gradient descent to minimize the difference between
    simulated and observed burned area.
    
    Parameters
    ----------
    grids : LevelSetGrids
        Base parameter grids
    x_ign, y_ign : float
        Ignition coordinates
    n_steps : int
        Number of time steps
    dt : float
        Time step
    initial_radius : float
        Initial fire radius
    obs_mask : jnp.ndarray
        Observed burned area mask (same shape as grids, 1=burned)
    obs_area : float
        Observed burned area
    initial_params : CalibrationParams, optional
        Starting parameters (default: all 1.0)
    n_iterations : int
        Number of optimization iterations
    learning_rate : float
        Step size for gradient descent
    area_weight, iou_weight : float
        Weights for combined loss
    verbose : bool
        Print progress
    subsample_steps : int, optional
        Use fewer time steps for faster calibration (default: min(n_steps, 100))
    max_grid_cells : int
        Maximum number of grid cells before spatial subsampling (default: 250000)
        
    Returns
    -------
    result : CalibrationResult
    """
    if initial_params is None:
        initial_params = CalibrationParams()
    
    # Subsample time steps for faster calibration
    if subsample_steps is None:
        subsample_steps = min(n_steps, 100)  # Default to 100 steps max for calibration
    
    ny, nx = len(grids.y_coords), len(grids.x_coords)
    total_cells = ny * nx
    
    # Spatial subsampling if grid is too large
    spatial_subsample = 1
    if total_cells > max_grid_cells:
        spatial_subsample = int(np.ceil(np.sqrt(total_cells / max_grid_cells)))
        if verbose:
            print(f"Grid too large ({total_cells} cells), subsampling by {spatial_subsample}x")
    
    # Create subsampled grids
    step_indices = np.linspace(0, n_steps - 1, subsample_steps, dtype=int)
    
    if spatial_subsample > 1:
        # Create explicit index arrays for subsampling
        y_idx = np.arange(0, ny, spatial_subsample)
        x_idx = np.arange(0, nx, spatial_subsample)
        
        # Convert to numpy for indexing, then back to JAX
        x_np = np.array(grids.x_coords)
        y_np = np.array(grids.y_coords)
        ros_np = np.array(grids.ros)
        bros_np = np.array(grids.bros)
        fros_np = np.array(grids.fros)
        raz_np = np.array(grids.raz)
        obs_mask_np = np.array(obs_mask)
        
        # Subsample using numpy advanced indexing
        x_sub = jnp.array(x_np[x_idx])
        y_sub = jnp.array(y_np[y_idx])
        ros_sub = jnp.array(ros_np[np.ix_(step_indices, y_idx, x_idx)])
        bros_sub = jnp.array(bros_np[np.ix_(step_indices, y_idx, x_idx)])
        fros_sub = jnp.array(fros_np[np.ix_(step_indices, y_idx, x_idx)])
        
        # Handle raz - might be 2D (y, x) or 3D (time, y, x)
        if raz_np.ndim == 2:
            raz_sub = jnp.array(raz_np[np.ix_(y_idx, x_idx)])
        else:
            # 3D - subsample in time and space, or just take first timestep for calibration
            # Since wind direction doesn't change much during calibration, use mean or first
            raz_sub = jnp.array(raz_np[0][np.ix_(y_idx, x_idx)])  # Use first timestep
        
        cal_obs_mask = jnp.array(obs_mask_np[np.ix_(y_idx, x_idx)])
        
        cal_grids = LevelSetGrids(
            x_coords=x_sub,
            y_coords=y_sub,
            ros=ros_sub,
            bros=bros_sub,
            fros=fros_sub,
            raz=raz_sub,
        )
        
        cal_obs_area = obs_area  # Keep same target area
        
        # Adjust initial radius for coarser grid
        cal_initial_radius = initial_radius * spatial_subsample
        
        if verbose:
            print(f"Subsampled: ROS shape {ros_sub.shape}, RAZ shape {raz_sub.shape}")
    else:
        # No spatial subsampling, but still subsample temporally
        ros_np = np.array(grids.ros)
        bros_np = np.array(grids.bros)
        fros_np = np.array(grids.fros)
        raz_np = np.array(grids.raz)
        
        # Subsample temporally
        ros_sub = jnp.array(ros_np[step_indices])
        bros_sub = jnp.array(bros_np[step_indices])
        fros_sub = jnp.array(fros_np[step_indices])
        
        # Handle raz - might be 2D or 3D
        if raz_np.ndim == 3:
            raz_sub = jnp.array(raz_np[step_indices])
        else:
            raz_sub = grids.raz
        
        cal_grids = LevelSetGrids(
            x_coords=grids.x_coords,
            y_coords=grids.y_coords,
            ros=ros_sub,
            bros=bros_sub,
            fros=fros_sub,
            raz=raz_sub,
        )
        cal_obs_mask = obs_mask
        cal_obs_area = obs_area
        cal_initial_radius = initial_radius
    
    cal_n_steps = subsample_steps
    cal_dt = dt * (n_steps / subsample_steps)  # Adjust dt to cover same duration
    
    if verbose:
        cal_ny, cal_nx = len(cal_grids.y_coords), len(cal_grids.x_coords)
        print(f"Calibration grid: {cal_ny}x{cal_nx} cells, {cal_n_steps} steps")
    
    # Grid spacing
    dx = float(cal_grids.x_coords[1] - cal_grids.x_coords[0])
    dy = float(cal_grids.y_coords[1] - cal_grids.y_coords[0])
    
    # Define loss function - NOT JIT compiled at this level
    # The inner simulate_fire_levelset is already JIT compiled
    def loss_fn(params_array):
        params = CalibrationParams(
            ros_scale=params_array[0],
            bros_scale=params_array[1],
            fros_scale=params_array[2],
        )
        phi = forward_model(params, cal_grids, x_ign, y_ign, cal_n_steps, cal_dt, cal_initial_radius)
        return combined_loss(phi, cal_obs_mask, cal_obs_area, dx, dy, area_weight, iou_weight)
    
    # Gradient function - use jax.value_and_grad for efficiency
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    
    # Initialize
    params_array = jnp.array([
        initial_params.ros_scale,
        initial_params.bros_scale,
        initial_params.fros_scale,
    ])
    
    loss_history = []
    params_history = []
    
    if verbose:
        print(f"Starting calibration: {n_iterations} iterations")
        print(f"Initial params: ROS={params_array[0]:.3f}, BROS={params_array[1]:.3f}, FROS={params_array[2]:.3f}")
    
    for i in range(n_iterations):
        # Compute loss and gradient together
        loss, grad = value_and_grad_fn(params_array)
        loss = float(loss)
        
        # Store history
        loss_history.append(loss)
        params_history.append({
            'ros_scale': float(params_array[0]),
            'bros_scale': float(params_array[1]),
            'fros_scale': float(params_array[2]),
        })
        
        # Update parameters (gradient descent with projection to positive)
        params_array = params_array - learning_rate * grad
        params_array = jnp.maximum(params_array, 0.1)  # Keep positive
        params_array = jnp.minimum(params_array, 5.0)  # Reasonable bounds
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Iter {i+1}: loss={loss:.6f}, ROS={params_array[0]:.3f}, BROS={params_array[1]:.3f}, FROS={params_array[2]:.3f}")
    
    # Final params
    final_params = CalibrationParams(
        ros_scale=float(params_array[0]),
        bros_scale=float(params_array[1]),
        fros_scale=float(params_array[2]),
    )
    
    # Compute final loss with full grids
    final_phi = forward_model(final_params, grids, x_ign, y_ign, n_steps, dt, initial_radius)
    final_loss = float(combined_loss(final_phi, obs_mask, obs_area, 
                                      float(grids.x_coords[1] - grids.x_coords[0]),
                                      float(grids.y_coords[1] - grids.y_coords[0]),
                                      area_weight, iou_weight))
    
    if verbose:
        print(f"Final: loss={final_loss:.6f}")
        print(f"Final params: ROS={final_params.ros_scale:.3f}, BROS={final_params.bros_scale:.3f}, FROS={final_params.fros_scale:.3f}")
    
    return CalibrationResult(
        params=final_params,
        loss_history=loss_history,
        params_history=params_history,
        final_loss=final_loss,
        n_iterations=n_iterations,
    )


def create_obs_mask_from_polygon(
    polygon,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """
    Create observation mask from Shapely polygon.
    
    Parameters
    ----------
    polygon : Shapely Polygon or MultiPolygon
        Observed fire boundary
    x_coords, y_coords : np.ndarray
        1D coordinate arrays
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask (ny, nx) where True = burned
    """
    from shapely.geometry import Point
    
    ny, nx = len(y_coords), len(x_coords)
    mask = np.zeros((ny, nx), dtype=np.float32)
    
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            if polygon.contains(Point(x, y)):
                mask[i, j] = 1.0
    
    return mask


def create_obs_mask_from_polygon_fast(
    polygon,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """
    Fast version using rasterio/shapely prepared geometry.
    """
    try:
        from shapely.prepared import prep
        from shapely.geometry import Point
        from shapely.vectorized import contains
    except ImportError:
        # Fallback to slow version
        return create_obs_mask_from_polygon(polygon, x_coords, y_coords)
    
    ny, nx = len(y_coords), len(x_coords)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Use vectorized contains
    mask = contains(polygon, X.ravel(), Y.ravel()).reshape(ny, nx)
    
    return mask.astype(np.float32)
