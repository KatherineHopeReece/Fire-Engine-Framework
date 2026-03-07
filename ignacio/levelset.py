"""
Level set fire spread solver (JAX).

Eulerian grid-based solver for fire perimeter evolution using the level set
method. The signed distance function φ represents the fire front (φ=0),
with φ<0 inside the burned region and φ>0 outside.

This solver is fully differentiable via JAX autodiff, enabling gradient-based
calibration of fire spread parameters.

The fire front evolves according to the Hamilton-Jacobi equation:
    ∂φ/∂t + F(x,y,n)|∇φ| = 0
where F is the direction-dependent speed function.

The speed function uses angular interpolation between head (ROS), back (BROS),
and flank (FROS) fire rates of spread, following the approach of Mallet et al.
(2009) and the implementation in ``ignacio.jax_core.levelset``.

References
----------
- Osher, S. & Sethian, J.A. (1988). Fronts propagating with curvature-
  dependent speed: algorithms based on Hamilton-Jacobi formulations.
- Mallet, V. et al. (2009). Modeling wildland fire propagation with level set
  methods. Computers & Mathematics with Applications.
- Sussman, M. et al. (1994). A level set approach for computing solutions to
  incompressible two-phase flow. J. Comput. Phys.
"""

from __future__ import annotations

import logging
from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from ignacio.spread import FireParameterGrid, FirePerimeterHistory

logger = logging.getLogger(__name__)

# Numerical stability constant
EPS = 1e-8


def _require_jax() -> None:
    """Raise a clear error if JAX is not installed."""
    if not _HAS_JAX:
        raise ImportError(
            "JAX is required for the level set solver. "
            "Install it with: pip install jax jaxlib"
        )


# =============================================================================
# Level set initialization
# =============================================================================


def initialize_phi(
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
    x_center: float,
    y_center: float,
    radius: float,
) -> jnp.ndarray:
    """
    Initialize φ as a signed distance function for a circular ignition.

    Parameters
    ----------
    x_coords : jnp.ndarray
        1D array of x grid coordinates.
    y_coords : jnp.ndarray
        1D array of y grid coordinates.
    x_center, y_center : float
        Ignition center coordinates.
    radius : float
        Initial fire radius.

    Returns
    -------
    phi : jnp.ndarray
        2D signed distance field (ny, nx). Negative inside, positive outside.
    """
    X, Y = jnp.meshgrid(x_coords, y_coords)
    phi = jnp.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2) - radius
    return phi


# =============================================================================
# Gradient computations
# =============================================================================


def compute_gradient(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    y_ascending: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute ∇φ using central differences with Neumann BCs.

    Parameters
    ----------
    phi : jnp.ndarray
        2D field (ny, nx).
    dx, dy : float
        Grid spacing (positive magnitudes).
    y_ascending : bool
        If True, y increases with row index (math convention).
        If False, y decreases with row index (raster convention).

    Returns
    -------
    phi_x, phi_y : jnp.ndarray
        Gradient components in geographic coordinates.
    """
    phi_padded = jnp.pad(phi, 1, mode="edge")
    phi_x = (phi_padded[1:-1, 2:] - phi_padded[1:-1, :-2]) / (2.0 * dx)
    phi_y = (phi_padded[2:, 1:-1] - phi_padded[:-2, 1:-1]) / (2.0 * dy)

    # If y is descending (row 0 = north), flip y gradient
    phi_y = jnp.where(y_ascending, phi_y, -phi_y)

    return phi_x, phi_y


def compute_gradient_magnitude_upwind(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    y_ascending: bool = True,
) -> jnp.ndarray:
    """
    Compute |∇φ| using Godunov upwind scheme for F ≥ 0.

    For outward-propagating fronts (F > 0):
    |∇φ|² = max(D⁻x, 0)² + min(D⁺x, 0)² + max(D⁻y, 0)² + min(D⁺y, 0)²

    Parameters
    ----------
    phi : jnp.ndarray
        Signed distance field (ny, nx).
    dx, dy : float
        Grid spacing (positive magnitudes).
    y_ascending : bool
        If True, y increases with row index.

    Returns
    -------
    grad_mag : jnp.ndarray
        Upwind gradient magnitude (ny, nx).
    """
    phi_padded = jnp.pad(phi, 1, mode="edge")

    # One-sided differences in array index space
    Dx_minus = (phi_padded[1:-1, 1:-1] - phi_padded[1:-1, :-2]) / dx
    Dx_plus = (phi_padded[1:-1, 2:] - phi_padded[1:-1, 1:-1]) / dx

    Dy_minus_idx = (phi_padded[1:-1, 1:-1] - phi_padded[:-2, 1:-1]) / dy
    Dy_plus_idx = (phi_padded[2:, 1:-1] - phi_padded[1:-1, 1:-1]) / dy

    # For descending y (row 0 = north), swap the interpretation of +/-
    Dy_minus = jnp.where(y_ascending, Dy_minus_idx, -Dy_plus_idx)
    Dy_plus = jnp.where(y_ascending, Dy_plus_idx, -Dy_minus_idx)

    # Godunov scheme for F ≥ 0
    grad_mag_sq = (
        jnp.maximum(Dx_minus, 0.0) ** 2
        + jnp.minimum(Dx_plus, 0.0) ** 2
        + jnp.maximum(Dy_minus, 0.0) ** 2
        + jnp.minimum(Dy_plus, 0.0) ** 2
    )

    return jnp.sqrt(grad_mag_sq + EPS)


# =============================================================================
# Speed function
# =============================================================================


def compute_elliptical_speed(
    phi_x: jnp.ndarray,
    phi_y: jnp.ndarray,
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute direction-dependent fire spread speed using angular interpolation.

    Uses cos²/sin² blending between ROS (head), BROS (back), and FROS (flank)
    based on the angle between the outward normal and the primary spread
    direction. This follows Mallet et al. (2009) and the implementation in
    ``ignacio.jax_core.levelset.compute_elliptical_speed``.

    Parameters
    ----------
    phi_x, phi_y : jnp.ndarray
        Gradient components (proportional to outward normal).
    ros, bros, fros : jnp.ndarray
        Rate of spread components (head, back, flank).
    raz : jnp.ndarray
        Spread direction (radians, geographic azimuth clockwise from north).

    Returns
    -------
    speed : jnp.ndarray
        Speed in the normal direction at each grid point.
    """
    grad_mag = jnp.sqrt(phi_x ** 2 + phi_y ** 2 + EPS)

    # Unit outward normal
    nx = phi_x / grad_mag
    ny = phi_y / grad_mag

    # Normal direction angle (math convention: from +x, counter-clockwise)
    normal_angle = jnp.arctan2(ny, nx)

    # RAZ is geographic azimuth (0=N, 90=E, clockwise).
    # Convert to math convention (0=E, 90=N, counter-clockwise):
    spread_angle = jnp.pi / 2 - raz

    # Angle between normal and primary spread direction
    angle_diff = normal_angle - spread_angle
    # Normalize to [-π, π]
    angle_diff = jnp.arctan2(jnp.sin(angle_diff), jnp.cos(angle_diff))

    # Angular interpolation (cos²/sin² blending):
    #   θ = 0   → head fire (ROS)
    #   θ = π   → back fire (BROS)
    #   θ = ±π/2 → flank fire (FROS)
    cos_half = jnp.cos(angle_diff / 2)
    sin_half = jnp.sin(angle_diff / 2)

    # Interpolate between head and back
    head_back = ros * cos_half ** 2 + bros * sin_half ** 2

    # Blend with flank speed based on |sin(θ)|
    flank_weight = jnp.abs(jnp.sin(angle_diff))
    speed = head_back * (1 - flank_weight) + fros * flank_weight

    return jnp.maximum(speed, 0.0)


# =============================================================================
# Reinitialization
# =============================================================================


def reinitialize_phi(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    n_iter: int = 10,
    dt_reinit: float | None = None,
) -> jnp.ndarray:
    """
    Reinitialize φ to a signed distance function (Sussman et al. 1994).

    Solves: φ_τ + sign(φ₀)(|∇φ| - 1) = 0 in pseudo-time τ.

    Parameters
    ----------
    phi : jnp.ndarray
        Current φ field (ny, nx).
    dx, dy : float
        Grid spacing.
    n_iter : int
        Number of pseudo-time iterations.
    dt_reinit : float, optional
        Pseudo-time step. Defaults to 0.5 * min(dx, dy).

    Returns
    -------
    phi : jnp.ndarray
        Reinitialized φ (approximate signed distance function).
    """
    if dt_reinit is None:
        dt_reinit = 0.5 * min(dx, dy)

    phi0 = phi
    sign_phi0 = phi0 / jnp.sqrt(phi0 ** 2 + dx ** 2)

    def _reinit_step(phi_val, _):
        grad_mag = compute_gradient_magnitude_upwind(phi_val, dx, dy)
        phi_val = phi_val - dt_reinit * sign_phi0 * (grad_mag - 1.0)
        return phi_val, None

    phi, _ = lax.scan(_reinit_step, phi, None, length=n_iter)
    return phi


# =============================================================================
# Level set time stepping
# =============================================================================


def level_set_step(
    phi: jnp.ndarray,
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
    dx: float,
    dy: float,
    dt: float,
    cfl_factor: float = 0.5,
    y_ascending: bool = True,
) -> jnp.ndarray:
    """
    Advance φ by one outer timestep dt with CFL sub-stepping.

    Uses the Hamilton-Jacobi equation: φ_t + F|∇φ| = 0
    with angular interpolation for the speed function F.

    Parameters
    ----------
    phi : jnp.ndarray
        Current signed distance field (ny, nx).
    ros, bros, fros, raz : jnp.ndarray
        Fire behaviour parameter grids (ny, nx).
    dx, dy : float
        Grid spacing (positive magnitudes).
    dt : float
        Outer timestep (minutes).
    cfl_factor : float
        CFL safety factor (0 < cfl_factor ≤ 1).
    y_ascending : bool
        If True, y increases with row index.

    Returns
    -------
    phi : jnp.ndarray
        Updated signed distance field.
    """
    # Compute speed for CFL estimate
    phi_x, phi_y = compute_gradient(phi, dx, dy, y_ascending)
    F_est = compute_elliptical_speed(phi_x, phi_y, ros, bros, fros, raz)

    # Determine sub-steps from CFL condition
    F_max = jnp.max(lax.stop_gradient(F_est))
    d_min = min(dx, dy)
    n_sub_raw = dt * F_max / (cfl_factor * d_min + 1e-30)
    n_sub = jnp.maximum(jnp.ceil(n_sub_raw).astype(jnp.int32), 1)
    n_sub = jnp.minimum(n_sub, 200)
    dt_sub = dt / n_sub

    def _substep(phi_val, _):
        phi_x_i, phi_y_i = compute_gradient(phi_val, dx, dy, y_ascending)
        F = compute_elliptical_speed(phi_x_i, phi_y_i, ros, bros, fros, raz)
        grad_mag = compute_gradient_magnitude_upwind(phi_val, dx, dy, y_ascending)
        phi_val = phi_val - dt_sub * F * grad_mag
        return phi_val, None

    phi, _ = lax.scan(_substep, phi, None, length=n_sub)
    return phi


@partial(jax.jit, static_argnames=("dx", "dy", "cfl_factor", "n_substeps", "y_ascending"))
def level_set_step_fixed(
    phi: jnp.ndarray,
    ros: jnp.ndarray,
    bros: jnp.ndarray,
    fros: jnp.ndarray,
    raz: jnp.ndarray,
    dx: float,
    dy: float,
    dt: float,
    cfl_factor: float = 0.5,
    n_substeps: int = 1,
    y_ascending: bool = True,
) -> jnp.ndarray:
    """
    Like ``level_set_step`` but with a fixed (static) number of sub-steps.

    Fully JIT-compilable and differentiable since the loop length is a
    compile-time constant.
    """
    dt_sub = dt / n_substeps

    def _substep(phi_val, _):
        phi_x, phi_y = compute_gradient(phi_val, dx, dy, y_ascending)
        F = compute_elliptical_speed(phi_x, phi_y, ros, bros, fros, raz)
        grad_mag = compute_gradient_magnitude_upwind(phi_val, dx, dy, y_ascending)
        phi_val = phi_val - dt_sub * F * grad_mag
        return phi_val, None

    phi, _ = lax.scan(_substep, phi, None, length=n_substeps)
    return phi


# =============================================================================
# Contour extraction
# =============================================================================


def extract_contour(
    phi_np: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x_ignition: float | None = None,
    y_ignition: float | None = None,
    burned_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract the fire perimeter and map to world coordinates.

    If ``burned_mask`` is provided, it is used directly as the fire region
    (preferred — avoids artifacts from reinitialization).  Otherwise falls
    back to connected-component labelling of φ≤0.

    Parameters
    ----------
    phi_np : np.ndarray
        Signed distance field (ny, nx) as a numpy array.
    x_coords, y_coords : np.ndarray
        1D coordinate arrays.
    x_ignition, y_ignition : float, optional
        Ignition point. Used to identify the correct burned region.
    burned_mask : np.ndarray, optional
        Boolean mask of burned cells (ny, nx). If given, used instead of φ≤0.

    Returns
    -------
    (x_contour, y_contour) : tuple of np.ndarray, or None if no contour found.
    """
    if burned_mask is not None:
        burned = burned_mask
    else:
        burned = phi_np <= 0.0

    if not np.any(burned):
        return None

    if burned_mask is not None:
        # Cumulative burned mask: fill interior holes for contour extraction.
        # Nonfuel cells fragment the mask; filling holes produces one clean
        # outer boundary.  (Area calculations are done separately.)
        from scipy.ndimage import binary_fill_holes
        fire_mask = binary_fill_holes(burned)
    else:
        # phi-based: use connected component labeling to find the fire
        from scipy.ndimage import label

        labels, n_labels = label(burned)

        if n_labels == 0:
            return None

        if n_labels == 1:
            fire_mask = burned
        elif x_ignition is not None and y_ignition is not None:
            # Find the component containing the ignition point
            dx = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
            dy_abs = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0

            if y_coords[0] > y_coords[-1]:
                iy = int(round((y_coords[0] - y_ignition) / dy_abs))
            else:
                iy = int(round((y_ignition - y_coords[0]) / dy_abs))
            ix = int(round((x_ignition - x_coords[0]) / dx))
            iy = max(0, min(iy, burned.shape[0] - 1))
            ix = max(0, min(ix, burned.shape[1] - 1))

            ign_label = labels[iy, ix]

            # If ignition is in a nonfuel cell, search nearby
            if ign_label == 0:
                search = 5
                for r in range(max(0, iy - search), min(burned.shape[0], iy + search + 1)):
                    for c in range(max(0, ix - search), min(burned.shape[1], ix + search + 1)):
                        if labels[r, c] > 0:
                            ign_label = labels[r, c]
                            break
                    if ign_label > 0:
                        break

            if ign_label > 0:
                fire_mask = labels == ign_label
            else:
                # Fallback: use largest component
                comp_sizes = np.bincount(labels.ravel())
                comp_sizes[0] = 0
                fire_mask = labels == np.argmax(comp_sizes)
        else:
            # No ignition info: use largest component
            comp_sizes = np.bincount(labels.ravel())
            comp_sizes[0] = 0
            fire_mask = labels == np.argmax(comp_sizes)

    if not np.any(fire_mask):
        return None

    # Binary mask → contour at the mask boundary.
    # Pad with one cell of "unburned" so contours close even when fire
    # touches the grid boundary.
    phi_masked = np.where(fire_mask, -1.0, 1.0)
    phi_padded = np.pad(phi_masked, pad_width=1, mode="constant", constant_values=1.0)

    def _shoelace_area(pts):
        """Signed area via Shoelace formula (row, col coords)."""
        r, c = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.sum(r[:-1] * c[1:] - r[1:] * c[:-1])
                         + r[-1] * c[0] - r[0] * c[-1])

    try:
        from skimage.measure import find_contours

        contours = find_contours(phi_padded, level=0.0)
        if not contours:
            return None

        # Pick the contour enclosing the largest area (outer boundary)
        best = max(contours, key=_shoelace_area)
        # Shift back by 1 to undo padding offset
        row_frac = best[:, 0] - 1.0
        col_frac = best[:, 1] - 1.0
        # Clamp to valid range
        row_frac = np.clip(row_frac, 0, len(y_coords) - 1)
        col_frac = np.clip(col_frac, 0, len(x_coords) - 1)
        x_out = np.interp(col_frac, np.arange(len(x_coords)), x_coords)
        y_out = np.interp(row_frac, np.arange(len(y_coords)), y_coords)
        return x_out, y_out

    except ImportError:
        pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        cs = ax.contour(phi_padded, levels=[0.0])
        plt.close(fig)
        if not cs.allsegs or not cs.allsegs[0]:
            return None
        seg = max(cs.allsegs[0], key=lambda s: _shoelace_area(s))
        col_frac = seg[:, 0] - 1.0
        row_frac = seg[:, 1] - 1.0
        col_frac = np.clip(col_frac, 0, len(x_coords) - 1)
        row_frac = np.clip(row_frac, 0, len(y_coords) - 1)
        x_out = np.interp(col_frac, np.arange(len(x_coords)), x_coords)
        y_out = np.interp(row_frac, np.arange(len(y_coords)), y_coords)
        return x_out, y_out
    except Exception:
        return None


# =============================================================================
# Main solver entry point
# =============================================================================


def simulate_fire_spread_levelset(
    param_grid: FireParameterGrid,
    x_ignition: float,
    y_ignition: float,
    dt: float = 1.0,
    initial_radius: float = 0.5,
    store_every: int = 1,
    max_steps: int | None = None,
    min_ros: float = 0.01,
    is_geographic: bool = False,
    center_latitude: float | None = None,
    cfl_factor: float = 0.5,
    reinit_interval: int = 5,
    reinit_iters: int = 10,
    narrow_band_width: int | None = None,
    spotting_config: dict | None = None,
    fuel_grid: np.ndarray | None = None,
    fuel_lookup: dict | None = None,
) -> FirePerimeterHistory:
    """
    Simulate fire spread using the level set method (JAX).

    This function mirrors the interface of ``simulate_fire_spread()`` but uses
    an Eulerian level set approach instead of Lagrangian markers.

    Parameters
    ----------
    param_grid : FireParameterGrid
        Gridded fire behaviour parameters (numpy arrays).
    x_ignition, y_ignition : float
        Ignition point coordinates.
    dt : float
        Outer time step in minutes.
    initial_radius : float
        Initial fire radius in meters (or coordinate units if not geographic).
    store_every : int
        Store perimeter contour every N steps.
    max_steps : int, optional
        Maximum number of time steps. Defaults to param_grid.nt.
    min_ros : float
        Minimum ROS threshold (m/min) below which fire is considered extinguished.
    is_geographic : bool
        If True, coordinates are in degrees and ROS is converted accordingly.
    center_latitude : float, optional
        Center latitude for geographic conversion.
    cfl_factor : float
        CFL safety factor for internal sub-stepping.
    reinit_interval : int
        Reinitialize φ to signed distance every N steps.
    reinit_iters : int
        Number of pseudo-time iterations for reinitialization.
    narrow_band_width : int, optional
        Unused placeholder for future narrow-band optimization.

    Returns
    -------
    FirePerimeterHistory
        Evolution history with extracted contour perimeters.
    """
    _require_jax()

    # ---- Handle geographic CRS ----
    if is_geographic:
        if center_latitude is None:
            center_latitude = y_ignition
        lat_rad = np.radians(center_latitude)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
        avg_meters_per_deg = (meters_per_deg_lat + meters_per_deg_lon) / 2.0
        initial_radius_coord = initial_radius / avg_meters_per_deg
        logger.debug(
            f"Level set geographic mode: {meters_per_deg_lon:.0f} m/deg lon, "
            f"initial radius {initial_radius}m = {initial_radius_coord:.6f}°"
        )
    else:
        initial_radius_coord = initial_radius
        avg_meters_per_deg = 1.0

    # ---- Grid setup ----
    x_coords = param_grid.x_coords
    y_coords = param_grid.y_coords
    dx = abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 1.0
    dy = abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 1.0

    # ---- Detect y-axis convention ----
    y_ascending = not (len(y_coords) > 1 and y_coords[0] > y_coords[-1])
    if not y_ascending:
        logger.debug("Detected descending y-axis (raster convention)")

    # ---- Pre-crop param_grid to a generous region around ignition ----
    # For large grids, this dramatically reduces memory and per-step slicing cost.
    # Cap at 750 cells in each direction from ignition (~65 km at 90m resolution),
    # which is generous for any single fire event.
    _MAX_BUFFER = 750
    _buf_cells_x = min(_MAX_BUFFER, len(x_coords) // 2)
    _buf_cells_y = min(_MAX_BUFFER, len(y_coords) // 2)
    # Locate ignition in grid
    _ix_ign = int(round((x_ignition - x_coords[0]) / dx))
    if y_ascending:
        _iy_ign = int(round((y_ignition - y_coords[0]) / dy))
    else:
        _iy_ign = int(round((y_coords[0] - y_ignition) / dy))
    _ix_ign = max(0, min(_ix_ign, len(x_coords) - 1))
    _iy_ign = max(0, min(_iy_ign, len(y_coords) - 1))
    _cx_lo = max(0, _ix_ign - _buf_cells_x)
    _cx_hi = min(len(x_coords), _ix_ign + _buf_cells_x)
    _cy_lo = max(0, _iy_ign - _buf_cells_y)
    _cy_hi = min(len(y_coords), _iy_ign + _buf_cells_y)

    if (_cy_hi - _cy_lo < len(y_coords)) or (_cx_hi - _cx_lo < len(x_coords)):
        logger.info(
            f"Pre-cropping param_grid: {len(y_coords)}x{len(x_coords)} → "
            f"{_cy_hi-_cy_lo}x{_cx_hi-_cx_lo} "
            f"(buffer: {_buf_cells_y} rows, {_buf_cells_x} cols)"
        )
        x_coords = x_coords[_cx_lo:_cx_hi]
        y_coords = y_coords[_cy_lo:_cy_hi]
        param_grid = FireParameterGrid(
            x_coords=x_coords,
            y_coords=y_coords,
            ros=param_grid.ros[:, _cy_lo:_cy_hi, _cx_lo:_cx_hi],
            bros=param_grid.bros[:, _cy_lo:_cy_hi, _cx_lo:_cx_hi],
            fros=param_grid.fros[:, _cy_lo:_cy_hi, _cx_lo:_cx_hi],
            raz=param_grid.raz[:, _cy_lo:_cy_hi, _cx_lo:_cx_hi],
        )
        # Also crop fuel_grid if provided
        if fuel_grid is not None:
            fuel_grid = fuel_grid[_cy_lo:_cy_hi, _cx_lo:_cx_hi]

    # ---- Initialize φ ----
    # Ensure initial radius covers at least 3 grid cells
    min_radius_coord = 3.0 * max(dx, dy)
    if initial_radius_coord < min_radius_coord:
        logger.info(
            f"Initial radius {initial_radius_coord:.6f} is sub-grid "
            f"(dx={dx:.6f}, dy={dy:.6f}). Clamping to {min_radius_coord:.6f} "
            f"({min_radius_coord * avg_meters_per_deg:.0f} m)"
        )
        initial_radius_coord = min_radius_coord

    x_jax = jnp.asarray(x_coords)
    y_jax = jnp.asarray(y_coords)
    phi = initialize_phi(x_jax, y_jax, x_ignition, y_ignition, initial_radius_coord)

    # ---- Time loop ----
    if max_steps is None:
        max_steps = param_grid.nt
    n_steps = min(max_steps, param_grid.nt)

    # Extract initial contour
    phi_np_out = np.asarray(phi)
    init_burned = np.asarray(phi < 0)
    initial_contour = extract_contour(
        phi_np_out, x_coords, y_coords, x_ignition, y_ignition,
        burned_mask=init_burned,
    )
    if initial_contour is not None:
        history = FirePerimeterHistory(
            perimeters=[initial_contour],
            times=[0.0],
        )
    else:
        theta_arr = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        x_init = x_ignition + initial_radius_coord * np.cos(theta_arr)
        y_init = y_ignition + initial_radius_coord * np.sin(theta_arr)
        history = FirePerimeterHistory(
            perimeters=[(x_init, y_init)],
            times=[0.0],
        )

    # Non-fuel barrier strength
    nonfuel_phi = 10.0 * max(dx, dy)

    # Cumulative burned mask: once a cell burns (φ<0), it stays burned forever.
    # This prevents contour extraction from "losing" burned area when φ becomes
    # positive again due to nonfuel forcing or reinitialization artifacts.
    _cumulative_burned = np.array(phi < 0, copy=True)

    # Static non-fuel mask: cells that NEVER have ROS > 0 across all timesteps.
    # This prevents time-varying weather (ROS temporarily 0) from erasing burned areas.
    # Compute memory-efficiently by iterating over timesteps instead of nanmax.
    ny_pg, nx_pg = param_grid.ros.shape[1], param_grid.ros.shape[2]
    _has_fuel = np.zeros((ny_pg, nx_pg), dtype=bool)
    # Sample every 24th timestep (daily) — sufficient to identify permanent non-fuel
    for _t in range(0, param_grid.nt, max(1, param_grid.nt // 70)):
        _ros_t = param_grid.ros[_t]
        _has_fuel |= np.nan_to_num(_ros_t, nan=0.0) > 0
    _static_nonfuel = ~_has_fuel
    logger.debug(f"Static non-fuel mask: {np.sum(_static_nonfuel)} / {_static_nonfuel.size} cells "
                 f"({np.mean(_static_nonfuel)*100:.1f}%)")

    # Spotting setup
    _do_spotting = (spotting_config is not None and fuel_grid is not None
                    and fuel_lookup is not None)
    if _do_spotting:
        _spot_cfg = spotting_config
        _spot_interval = int(_spot_cfg.get("interval", 1))
        _spot_rng = np.random.default_rng(_spot_cfg.get("seed", 42))
        logger.info(f"Spotting enabled: interval={_spot_interval} steps")
    else:
        _spot_cfg = {}
        _spot_interval = 1
        _spot_rng = None

    d_min = min(dx, dy)
    ny_full, nx_full = len(y_coords), len(x_coords)

    # Use float32 for computation (sufficient for fire spread, 2× faster)
    phi = phi.astype(jnp.float32)

    # JIT-compiled evolution function for a fixed number of sub-steps.
    # Cached by (dx, dy, y_ascending, n_sub); shape changes trigger recompilation
    # but we pad crop dimensions to multiples of _PAD to limit this.
    _PAD = 64  # pad crop dims to multiples of this

    @partial(jax.jit, static_argnames=("dx_", "dy_", "y_ascending_", "n_sub_"))
    def _evolve_step(phi_, ros_, bros_, fros_, raz_, nonfuel_, nonfuel_phi_,
                     dt_sub_, dx_, dy_, y_ascending_, n_sub_):
        # Ensure all scalars match phi dtype to prevent promotion
        _dtype = phi_.dtype
        _dt = jnp.asarray(dt_sub_, dtype=_dtype)
        _nfp = jnp.asarray(nonfuel_phi_, dtype=_dtype)

        def _body(_, phi_val):
            phi_x, phi_y = compute_gradient(phi_val, dx_, dy_, y_ascending_)
            F = compute_elliptical_speed(phi_x, phi_y, ros_, bros_, fros_, raz_)
            grad_mag = compute_gradient_magnitude_upwind(phi_val, dx_, dy_, y_ascending_)
            phi_val = phi_val - _dt * F * grad_mag
            phi_val = jnp.where(nonfuel_, jnp.maximum(phi_val, _nfp), phi_val)
            return phi_val.astype(_dtype)
        return lax.fori_loop(0, n_sub_, _body, phi_)

    def _pad_dim(n):
        """Round up to nearest multiple of _PAD, clamped to grid size."""
        return min(((n + _PAD - 1) // _PAD) * _PAD, max(ny_full, nx_full))

    logger.info(f"Level set simulation: {n_steps} steps, dt={dt:.1f} min, "
                f"grid {ny_full}x{nx_full}, dx={dx:.4f}, dy={dy:.4f} (float32, bbox crop)")

    # Bounding box state — start with ignition neighborhood
    _BUFFER_BASE = 20  # minimum buffer cells around fire
    iy_ign = int(round((y_ignition - y_coords[0]) / dy)) if y_ascending else int(round((y_coords[0] - y_ignition) / dy))
    ix_ign = int(round((x_ignition - x_coords[0]) / dx))
    iy_ign = max(0, min(iy_ign, ny_full - 1))
    ix_ign = max(0, min(ix_ign, nx_full - 1))
    bbox_r0 = max(int(initial_radius_coord / d_min) + _BUFFER_BASE, 50)
    r0_lo = max(0, iy_ign - bbox_r0)
    r0_hi = min(ny_full, iy_ign + bbox_r0)
    c0_lo = max(0, ix_ign - bbox_r0)
    c0_hi = min(nx_full, ix_ign + bbox_r0)
    # Pad to multiples
    crop_ny = _pad_dim(r0_hi - r0_lo)
    crop_nx = _pad_dim(c0_hi - c0_lo)
    r0_lo = max(0, iy_ign - crop_ny // 2)
    r0_hi = min(ny_full, r0_lo + crop_ny)
    r0_lo = r0_hi - crop_ny  # adjust if clamped at top
    c0_lo = max(0, ix_ign - crop_nx // 2)
    c0_hi = min(nx_full, c0_lo + crop_nx)
    c0_lo = c0_hi - crop_nx

    _compiled_nsubs = set()
    _bbox_update_interval = 10  # recompute bbox every N steps
    _last_crop_shape = None

    for step in range(n_steps):
        t_idx = min(step, param_grid.nt - 1)

        # Check for extinguishment (only check within bbox for speed)
        ros_crop = param_grid.ros[t_idx, r0_lo:r0_hi, c0_lo:c0_hi]
        if np.all(np.nan_to_num(ros_crop, nan=0.0) < min_ros):
            logger.info(f"Fire extinguished at step {step} (all ROS below threshold in active region)")
            break

        # Extract current timestep slices (cropped to bbox)
        ros_np_c = np.nan_to_num(param_grid.ros[t_idx, r0_lo:r0_hi, c0_lo:c0_hi], nan=0.0).astype(np.float32)
        bros_np_c = np.nan_to_num(param_grid.bros[t_idx, r0_lo:r0_hi, c0_lo:c0_hi], nan=0.0).astype(np.float32)
        fros_np_c = np.nan_to_num(param_grid.fros[t_idx, r0_lo:r0_hi, c0_lo:c0_hi], nan=0.0).astype(np.float32)
        raz_np_c = np.nan_to_num(param_grid.raz[t_idx, r0_lo:r0_hi, c0_lo:c0_hi], nan=0.0).astype(np.float32)

        # Use static non-fuel mask (cropped) — cells permanently without fuel.
        # NOT derived from current-timestep ROS, to avoid erasing burned areas
        # when ROS temporarily drops to 0 (nighttime, weather data gaps, etc.).
        nonfuel_np_c = _static_nonfuel[r0_lo:r0_hi, c0_lo:c0_hi]

        # Convert ROS from m/min to coord_units/min if geographic
        if is_geographic:
            ros_np_c = ros_np_c / avg_meters_per_deg
            bros_np_c = bros_np_c / avg_meters_per_deg
            fros_np_c = fros_np_c / avg_meters_per_deg

        # LOCAL CFL: use max ROS in the crop region, not global
        F_max_local = float(np.nanmax(ros_np_c))
        if not np.isfinite(F_max_local) or F_max_local <= 0:
            F_max_local = 0.0
        n_sub = max(1, int(np.ceil(F_max_local * dt / (cfl_factor * d_min)))) if F_max_local > 0 else 1
        n_sub = min(n_sub, 500)
        # Quantize large n_sub to limit JIT recompilations
        if n_sub > 10:
            _QUANTA = [20, 50, 100, 200, 500]
            n_sub = next((q for q in _QUANTA if q >= n_sub), 500)
        dt_sub = dt / n_sub

        # Extract phi crop
        phi_crop = phi[r0_lo:r0_hi, c0_lo:c0_hi]

        # Convert to JAX float32
        ros_j = jnp.asarray(ros_np_c)
        bros_j = jnp.asarray(bros_np_c)
        fros_j = jnp.asarray(fros_np_c)
        raz_j = jnp.asarray(raz_np_c)
        nonfuel_j = jnp.asarray(nonfuel_np_c)

        crop_shape = (r0_hi - r0_lo, c0_hi - c0_lo)
        if crop_shape != _last_crop_shape:
            logger.info(f"  Crop region: {crop_shape[0]}x{crop_shape[1]} "
                        f"(rows {r0_lo}-{r0_hi}, cols {c0_lo}-{c0_hi})")
            _last_crop_shape = crop_shape

        if n_sub not in _compiled_nsubs:
            logger.info(f"  Compiling JIT kernel for n_sub={n_sub} on {crop_shape[0]}x{crop_shape[1]} crop")
            _compiled_nsubs.add(n_sub)

        # Evolve φ on cropped region
        phi_crop = _evolve_step(
            phi_crop, ros_j, bros_j, fros_j, raz_j, nonfuel_j,
            jnp.float32(nonfuel_phi), jnp.float32(dt_sub),
            dx_=dx, dy_=dy, y_ascending_=y_ascending, n_sub_=n_sub,
        )

        # Write cropped result back to full phi
        phi = phi.at[r0_lo:r0_hi, c0_lo:c0_hi].set(phi_crop)

        # Reinitialize φ periodically (on crop only)
        if (step + 1) % reinit_interval == 0:
            phi_crop = phi[r0_lo:r0_hi, c0_lo:c0_hi]
            phi_crop = reinitialize_phi(phi_crop, dx, dy, n_iter=reinit_iters)
            phi_crop = jnp.where(nonfuel_j, jnp.maximum(phi_crop, jnp.float32(nonfuel_phi)), phi_crop)
            phi = phi.at[r0_lo:r0_hi, c0_lo:c0_hi].set(phi_crop)

        # Stochastic spotting: create new ignitions ahead of the fire front
        if _do_spotting and (step + 1) % _spot_interval == 0:
            from ignacio.spotting import compute_spot_ignitions, apply_spot_ignitions
            phi_np_spot = np.asarray(phi)
            burned_now = _cumulative_burned | (phi_np_spot < 0)
            ros_np_spot = np.nan_to_num(param_grid.ros[t_idx], nan=0.0)
            raz_np_spot = np.nan_to_num(param_grid.raz[t_idx], nan=0.0)

            # Use per-timestep ISI/WS if available, else fallback to static
            _isi_arr = _spot_cfg.get("isi_array", None)
            _ws_arr = _spot_cfg.get("ws_array", None)
            _isi_t = float(_isi_arr[t_idx]) if _isi_arr is not None and t_idx < len(_isi_arr) else _spot_cfg.get("isi", 5.0)
            _ws_t = float(_ws_arr[t_idx]) if _ws_arr is not None and t_idx < len(_ws_arr) else _spot_cfg.get("wind_speed", 15.0)

            spot_mask = compute_spot_ignitions(
                burned_mask=burned_now,
                phi=phi_np_spot,
                ros=ros_np_spot,
                raz=raz_np_spot,
                fuel_grid=fuel_grid,
                fuel_lookup=fuel_lookup,
                wind_speed=_ws_t,
                isi=_isi_t,
                dx=dx, dy=dy,
                rng=_spot_rng,
                min_spot_distance=_spot_cfg.get("min_spot_distance", 500.0),
                max_spot_distance=_spot_cfg.get("max_spot_distance", 2000.0),
                isi_threshold=_spot_cfg.get("isi_threshold", 8.0),
                spot_probability=_spot_cfg.get("spot_probability", 0.15),
                meters_per_unit=avg_meters_per_deg if is_geographic else 1.0,
                y_ascending=y_ascending,
            )
            if np.any(spot_mask):
                n_spots = int(np.sum(spot_mask))
                logger.info(f"  Step {step+1}: {n_spots} spot fires ignited "
                           f"(ISI={_isi_t:.1f}, WS={_ws_t:.0f} km/h)")
                phi = apply_spot_ignitions(phi, spot_mask, spot_radius_cells=2)

        # Update cumulative burned mask (monotonically grows)
        if (step + 1) % _bbox_update_interval == 0 or (step + 1) % store_every == 0:
            phi_np_full = np.asarray(phi)
            _cumulative_burned |= (phi_np_full < 0)

        # Update bounding box periodically to follow the fire
        if (step + 1) % _bbox_update_interval == 0:
            burned = _cumulative_burned
            if np.any(burned):
                rows_burned, cols_burned = np.nonzero(burned)
                # Fire extent + buffer (enough for growth in bbox_update_interval steps)
                growth_cells = max(
                    int(F_max_local * dt * _bbox_update_interval / d_min) + 10,
                    _BUFFER_BASE,
                )
                new_r_lo = max(0, int(rows_burned.min()) - growth_cells)
                new_r_hi = min(ny_full, int(rows_burned.max()) + 1 + growth_cells)
                new_c_lo = max(0, int(cols_burned.min()) - growth_cells)
                new_c_hi = min(nx_full, int(cols_burned.max()) + 1 + growth_cells)
                # Pad to multiples for JIT stability
                new_ny = _pad_dim(new_r_hi - new_r_lo)
                new_nx = _pad_dim(new_c_hi - new_c_lo)
                # Center the padded region on the fire
                r_center = (new_r_lo + new_r_hi) // 2
                c_center = (new_c_lo + new_c_hi) // 2
                r0_lo = max(0, r_center - new_ny // 2)
                r0_hi = min(ny_full, r0_lo + new_ny)
                if r0_hi - r0_lo < new_ny:
                    r0_lo = max(0, r0_hi - new_ny)
                c0_lo = max(0, c_center - new_nx // 2)
                c0_hi = min(nx_full, c0_lo + new_nx)
                if c0_hi - c0_lo < new_nx:
                    c0_lo = max(0, c0_hi - new_nx)

        # Store contour
        if (step + 1) % store_every == 0:
            contour = extract_contour(
                np.asarray(phi), x_coords, y_coords,
                x_ignition, y_ignition,
                burned_mask=_cumulative_burned,
            )
            if contour is not None:
                history.perimeters.append(contour)
                history.times.append((step + 1) * dt)

        # Log progress
        log_interval = 10 if step < 50 else 100
        if (step + 1) % log_interval == 0:
            burned_frac = float(jnp.mean(phi < 0))
            logger.info(f"  Step {step+1}/{n_steps}: {burned_frac*100:.2f}% burned, "
                        f"crop {r0_hi-r0_lo}x{c0_hi-c0_lo}, n_sub={n_sub}")

    logger.info(f"Level set simulation complete: {len(history.perimeters)} perimeters stored")
    return history
