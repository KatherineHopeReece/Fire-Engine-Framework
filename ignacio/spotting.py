"""
Stochastic spotting module for Ignacio.

Simulates ember transport from active fire fronts to create spot fires
ahead of the main fire. This addresses the fundamental limitation that
a smooth level set solver cannot produce disconnected burns.

Spotting is most relevant for boreal conifer fuels (C-2, C-3, C-4, C-6)
under high wind conditions, where crown fire generates significant
firebrands.

References
----------
- Albini, F.A. (1979). Spot fire distance from burning trees.
  USDA Forest Service INT-56.
- Forestry Canada (1992). Development and Structure of the Canadian
  Forest Fire Behavior Prediction System. ST-X-3.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Fuel types with significant spotting potential (crown fire fuels)
SPOTTING_FUEL_TYPES = {"C-2", "C-3", "C-4", "C-6"}

# Default parameters
DEFAULT_MIN_SPOT_DISTANCE = 500.0  # meters
DEFAULT_MAX_SPOT_DISTANCE = 2000.0  # meters
DEFAULT_ISI_THRESHOLD = 8.0  # minimum ISI for spotting
DEFAULT_SPOT_PROBABILITY = 0.15  # base probability per eligible cell


def compute_spot_ignitions(
    burned_mask: np.ndarray,
    phi: np.ndarray,
    ros: np.ndarray,
    raz: np.ndarray,
    fuel_grid: np.ndarray,
    fuel_lookup: dict[int, str],
    wind_speed: float,
    isi: float,
    dx: float,
    dy: float,
    rng: np.random.Generator,
    min_spot_distance: float = DEFAULT_MIN_SPOT_DISTANCE,
    max_spot_distance: float = DEFAULT_MAX_SPOT_DISTANCE,
    isi_threshold: float = DEFAULT_ISI_THRESHOLD,
    spot_probability: float = DEFAULT_SPOT_PROBABILITY,
    meters_per_unit: float = 1.0,
    y_ascending: bool = False,
) -> np.ndarray:
    """
    Compute stochastic spot fire ignitions from the active fire front.

    At each timestep, cells on the active fire front are sampled. For
    each, if the fuel type supports spotting and ISI exceeds the threshold,
    a spot fire may ignite at a random downwind distance.

    Parameters
    ----------
    burned_mask : np.ndarray
        Boolean (ny, nx) mask of currently burned cells.
    phi : np.ndarray
        Signed distance field (ny, nx). The active front is near phi=0.
    ros : np.ndarray
        Rate of spread grid (ny, nx) for the current timestep (m/min).
    raz : np.ndarray
        Spread direction (ny, nx) in radians (geographic azimuth).
    fuel_grid : np.ndarray
        Integer fuel type grid (ny, nx).
    fuel_lookup : dict
        Mapping of fuel grid integers to FBP fuel type strings.
    wind_speed : float
        Wind speed in km/h.
    isi : float
        Initial Spread Index.
    dx, dy : float
        Grid cell size in coordinate units.
    rng : numpy.random.Generator
        Random number generator.
    min_spot_distance : float
        Minimum spotting distance (meters).
    max_spot_distance : float
        Maximum spotting distance (meters).
    isi_threshold : float
        Minimum ISI for spotting to occur.
    spot_probability : float
        Base probability of a spot fire per eligible front cell.
    meters_per_unit : float
        Conversion factor from coordinate units to meters.

    Returns
    -------
    spot_mask : np.ndarray
        Boolean (ny, nx) mask of new spot fire ignition cells.
    """
    ny, nx = burned_mask.shape
    spot_mask = np.zeros((ny, nx), dtype=bool)

    # No spotting if ISI is below threshold
    if isi < isi_threshold:
        return spot_mask

    # Scale probability with wind speed (more wind = more spotting)
    # Linear ramp: prob doubles at 40 km/h
    ws_factor = min(2.0, max(0.5, wind_speed / 20.0))
    adj_probability = spot_probability * ws_factor

    # Scale max distance with wind speed
    ws_dist_factor = min(3.0, max(1.0, wind_speed / 15.0))
    adj_max_distance = max_spot_distance * ws_dist_factor

    # Identify active fire front cells: burned, with at least one unburned neighbor
    # and fuel that supports spotting
    front_mask = burned_mask & (ros > 0)
    if not np.any(front_mask):
        return spot_mask

    # Erode burned mask to find interior, then front = burned - interior
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(burned_mask, iterations=1)
    front_mask = burned_mask & ~interior & (ros > 0)

    if not np.any(front_mask):
        return spot_mask

    # Build spotting fuel mask (only these fuel types generate embers)
    spotting_fuel_mask = np.zeros((ny, nx), dtype=bool)
    for fid, fname in fuel_lookup.items():
        if fname.upper() in SPOTTING_FUEL_TYPES:
            spotting_fuel_mask |= (fuel_grid == fid)

    # Front cells must be in spotting fuels
    eligible_front = front_mask & spotting_fuel_mask
    front_cells = np.argwhere(eligible_front)

    if len(front_cells) == 0:
        return spot_mask

    # Sample which front cells actually generate spots
    n_front = len(front_cells)
    spot_draws = rng.random(n_front)
    active = front_cells[spot_draws < adj_probability]

    if len(active) == 0:
        return spot_mask

    # For each active cell, compute a downwind spot location
    for iy, ix in active:
        # Get spread direction at this cell (geographic azimuth in radians)
        spread_dir = float(raz[iy, ix])

        # Random distance in meters
        dist_m = rng.uniform(min_spot_distance, adj_max_distance)

        # Convert to grid offsets (geographic: azimuth 0=N, 90=E)
        dist_coord = dist_m / meters_per_unit
        d_col = dist_coord * np.sin(spread_dir) / dx
        # In raster convention (y descending), north = decreasing row
        if y_ascending:
            d_row = dist_coord * np.cos(spread_dir) / dy
        else:
            d_row = -dist_coord * np.cos(spread_dir) / dy

        target_row = int(round(iy + d_row))
        target_col = int(round(ix + d_col))

        # Bounds check
        if 0 <= target_row < ny and 0 <= target_col < nx:
            # Only ignite if: not already burned, is a fuel cell
            if not burned_mask[target_row, target_col]:
                target_fuel_id = fuel_grid[target_row, target_col]
                target_fuel_name = fuel_lookup.get(int(target_fuel_id), "NF")
                if target_fuel_name not in ("NF", "WA", "NB"):
                    spot_mask[target_row, target_col] = True

    n_spots = int(np.sum(spot_mask))
    if n_spots > 0:
        logger.debug(
            f"Spotting: {n_spots} new spots from {len(active)} active front cells "
            f"(ISI={isi:.1f}, WS={wind_speed:.0f} km/h)"
        )

    return spot_mask


def apply_spot_ignitions(
    phi: "jnp.ndarray",
    spot_mask: np.ndarray,
    spot_radius_cells: int = 2,
) -> "jnp.ndarray":
    """
    Apply spot fire ignitions to the level set field.

    Sets phi to negative values at spot locations, creating new burned
    regions in the level set.

    Parameters
    ----------
    phi : jnp.ndarray
        Current signed distance field (ny, nx).
    spot_mask : np.ndarray
        Boolean mask of spot fire locations.
    spot_radius_cells : int
        Radius in cells for each spot fire ignition.

    Returns
    -------
    phi : jnp.ndarray
        Updated signed distance field with spot fires.
    """
    import jax.numpy as jnp

    if not np.any(spot_mask):
        return phi

    # Dilate spot mask by spot_radius_cells for a small initial burned area
    if spot_radius_cells > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * spot_radius_cells + 1, 2 * spot_radius_cells + 1))
        dilated = binary_dilation(spot_mask, structure=struct)
    else:
        dilated = spot_mask

    # Set phi negative at spot locations (burned)
    spot_phi = -1.0
    phi = jnp.where(jnp.asarray(dilated), jnp.float32(spot_phi), phi)

    return phi
