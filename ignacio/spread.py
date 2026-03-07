"""
Fire spread equations for Ignacio.

This module implements Richards' differential equations for elliptical fire
spread using Huygens' wave propagation principle, along with the marker
method for handling perimeter topology.

References
----------
- Richards, G.D. (1990). An elliptical growth model of forest fire fronts
  and its numerical solution. International Journal for Numerical Methods
  in Engineering, 30(6), 1163-1179.
- Anderson, D.H. et al. (1982). Modelling the spread of grass fires.
  Journal of the Australian Mathematical Society, Series B, 23(4), 451-466.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)

# =============================================================================
# Marker maintenance + higher-order advection helpers (RK4)
# =============================================================================

def _closed_diffs_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward differences on a closed ring."""
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    return dx, dy


def _segment_lengths_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx, dy = _closed_diffs_xy(x, y)
    return np.hypot(dx, dy)


def _insert_delete_markers_by_spacing(
    x: np.ndarray,
    y: np.ndarray,
    spacing: float,
    insert_factor: float = 1.5,
    delete_factor: float = 0.5,
    max_n: int = 20000,
    min_n: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Maintain marker spacing on a closed ring by inserting/deleting vertices."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 4 or spacing <= 0:
        return x, y

    # ---- Insert pass ----
    if n < max_n:
        seg = _segment_lengths_xy(x, y)
        new_x: list[float] = []
        new_y: list[float] = []
        for i in range(n):
            x0, y0 = float(x[i]), float(y[i])
            x1, y1 = float(x[(i + 1) % n]), float(y[(i + 1) % n])
            new_x.append(x0)
            new_y.append(y0)
            if seg[i] > insert_factor * spacing:
                new_x.append(0.5 * (x0 + x1))
                new_y.append(0.5 * (y0 + y1))
        x = np.asarray(new_x, dtype=float)
        y = np.asarray(new_y, dtype=float)

    # ---- Delete pass ----
    n = x.size
    if n > min_n:
        seg = _segment_lengths_xy(x, y)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if keep.sum() <= min_n:
                break
            prev_i = (i - 1) % n
            if seg[prev_i] < delete_factor * spacing and seg[i] < delete_factor * spacing:
                keep[i] = False
        x = x[keep]
        y = y[keep]

    return x, y


def _smooth_ring_local_xy(
    x: np.ndarray,
    y: np.ndarray,
    w: float = 0.10,
    n_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Very light local smoothing (Laplacian-like) for numerical noise only."""
    if n_iter <= 0 or x.size < 4:
        return x, y
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for _ in range(n_iter):
        x_prev = np.roll(x, 1)
        x_next = np.roll(x, -1)
        y_prev = np.roll(y, 1)
        y_next = np.roll(y, -1)
        x = (1 - w) * x + w * 0.5 * (x_prev + x_next)
        y = (1 - w) * y + w * 0.5 * (y_prev + y_next)
    return x, y


def _rk4_step_xy(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    vel_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RK4 step for marker advection: (x,y) -> (x_new,y_new).

    The velocity function vel_fn(x, y) must resample the parameter grid
    at arbitrary positions, not just the initial marker locations.

    Parameters
    ----------
    x : np.ndarray
        Current x positions of markers
    y : np.ndarray
        Current y positions of markers
    dt : float
        Time step
    vel_fn : callable
        Function that returns (vx, vy) = vel_fn(x, y) by sampling
        the parameter grid at arbitrary (x,y) positions

    Returns
    -------
    x_new : np.ndarray
        Updated x positions
    y_new : np.ndarray
        Updated y positions

    Notes
    -----
    Classic RK4 formula:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    # Stage 1: velocity at current position
    k1x, k1y = vel_fn(x, y)

    # Stage 2: velocity at midpoint using k1
    x_mid1 = x + 0.5 * dt * k1x
    y_mid1 = y + 0.5 * dt * k1y
    k2x, k2y = vel_fn(x_mid1, y_mid1)

    # Stage 3: velocity at midpoint using k2
    x_mid2 = x + 0.5 * dt * k2x
    y_mid2 = y + 0.5 * dt * k2y
    k3x, k3y = vel_fn(x_mid2, y_mid2)

    # Stage 4: velocity at endpoint using k3
    x_end = x + dt * k3x
    y_end = y + dt * k3y
    k4x, k4y = vel_fn(x_end, y_end)

    # Weighted average
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    return x_new, y_new


def _euler_extrapolation_step_xy(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    vel_fn,
    n_base: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Euler extrapolation (Richardson) step for marker advection.

    Computes solutions with *n_base* and *2·n_base* forward Euler sub-steps,
    then applies Richardson extrapolation to cancel the leading-order
    truncation error, yielding second-order temporal accuracy from the
    first-order Euler scheme.

    Parameters
    ----------
    x, y : np.ndarray
        Current marker positions.
    dt : float
        Time step.
    vel_fn : callable
        Velocity function ``vel_fn(x, y) -> (vx, vy)``.
    n_base : int
        Number of sub-steps for the coarse solution.  The fine solution
        uses ``2 * n_base`` sub-steps.

    Returns
    -------
    x_new, y_new : np.ndarray
        Extrapolated positions (second-order accurate).
    error_est : float
        Max pointwise error estimate ``||fine - coarse||_inf``, usable
        for adaptive step-size control.
    """
    # Coarse solution: n_base Euler steps of size dt/n_base
    x_c, y_c = x.copy(), y.copy()
    dt_c = dt / n_base
    for _ in range(n_base):
        vx, vy = vel_fn(x_c, y_c)
        x_c += dt_c * vx
        y_c += dt_c * vy

    # Fine solution: 2*n_base Euler steps of size dt/(2*n_base)
    x_f, y_f = x.copy(), y.copy()
    dt_f = dt / (2 * n_base)
    for _ in range(2 * n_base):
        vx, vy = vel_fn(x_f, y_f)
        x_f += dt_f * vx
        y_f += dt_f * vy

    # Richardson extrapolation: 2*fine - coarse cancels O(dt) error
    x_new = 2.0 * x_f - x_c
    y_new = 2.0 * y_f - y_c

    # Error estimate for adaptive step control: ||fine - coarse||_inf
    error_est = float(np.max(np.hypot(x_f - x_c, y_f - y_c)))

    return x_new, y_new, error_est


def _implicit_newton_step_xy(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    vel_fn,
    max_iters: int = 3,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implicit Euler with Newton linearization for marker advection.

    Solves the implicit system

        z^{n+1} = z^n + dt · g(z^{n+1})

    via Newton iteration.  At each iterate *m* the linearized system is:

        [I − dt · J_g] δz = −r

    where ``r = z^m − z^n − dt · g(z^m)`` is the residual and ``J_g``
    is the Jacobian ``∂g/∂z`` approximated column-wise by finite
    differences.  The initial guess is the forward-Euler predictor.

    Because the method is unconditionally A-stable, no CFL sub-stepping
    is required — a single call covers the full outer time step *dt*.

    Parameters
    ----------
    x, y : np.ndarray
        Current marker positions (N markers each).
    dt : float
        Time step.
    vel_fn : callable
        Velocity function ``vel_fn(x, y) -> (vx, vy)``.
    max_iters : int
        Maximum Newton iterations (typically 2–4 suffice).
    tol : float
        Convergence tolerance on ``max|r|``.

    Returns
    -------
    x_new, y_new : np.ndarray
        Updated marker positions.
    """
    n = len(x)
    N2 = 2 * n

    def _pack(xv, yv):
        z = np.empty(N2)
        z[0::2] = xv
        z[1::2] = yv
        return z

    def _unpack(z):
        return z[0::2].copy(), z[1::2].copy()

    def _g(z):
        xq, yq = _unpack(z)
        vx, vy = vel_fn(xq, yq)
        return _pack(vx, vy)

    z0 = _pack(x.copy(), y.copy())

    # Predictor: forward Euler
    g0 = _g(z0)
    zm = z0 + dt * g0

    # FD perturbation scale (relative to fire extent)
    scale = max(float(np.ptp(x)), float(np.ptp(y)), 1e-10)
    fd_eps = scale * 1e-7

    for k in range(max_iters):
        gm = _g(zm)

        # Residual: r = z^m - z^n - dt * g(z^m)
        r = zm - z0 - dt * gm
        res_norm = float(np.max(np.abs(r)))
        logger.debug(f"  Newton iter {k}: ||r||_inf = {res_norm:.3e}")

        if res_norm < tol:
            break

        # Build A = [I - dt * J_g] column by column via finite differences.
        # Column j: A[:,j] = e_j - dt * (g(z + eps*e_j) - g(z)) / eps
        A = np.eye(N2)
        for j in range(N2):
            zp = zm.copy()
            zp[j] += fd_eps
            gp = _g(zp)
            A[:, j] -= dt * (gp - gm) / fd_eps

        # Solve [I - dt * J_g] δz = -r
        try:
            dz = np.linalg.solve(A, -r)
        except np.linalg.LinAlgError:
            logger.warning(
                f"Newton linear solve failed at iteration {k}, "
                f"using current iterate"
            )
            break

        zm += dz

    return _unpack(zm)


# =============================================================================
# Fire Parameter Grid
# =============================================================================


@dataclass
class FireParameterGrid:
    """
    Container for gridded fire behaviour parameters over space and time.
    
    This class holds 3D arrays (time, y, x) of rate of spread components
    and provides bilinear interpolation to query values at arbitrary
    spatial positions.
    
    Attributes
    ----------
    x_coords : np.ndarray
        1D array of x coordinates (column centers).
    y_coords : np.ndarray
        1D array of y coordinates (row centers).
    ros : np.ndarray
        Head fire rate of spread (nt, ny, nx) in m/min.
    bros : np.ndarray
        Back fire rate of spread (nt, ny, nx) in m/min.
    fros : np.ndarray
        Flank fire rate of spread (nt, ny, nx) in m/min.
    raz : np.ndarray
        Rate of spread azimuth (nt, ny, nx) in radians.
    """
    
    x_coords: np.ndarray
    y_coords: np.ndarray
    ros: np.ndarray
    bros: np.ndarray
    fros: np.ndarray
    raz: np.ndarray
    
    def __post_init__(self):
        """Validate array shapes and compute grid parameters."""
        assert self.ros.ndim == 3, "ROS must be 3D (time, y, x)"
        assert self.ros.shape == self.bros.shape == self.fros.shape == self.raz.shape
        
        self.nt, self.ny, self.nx = self.ros.shape
        
        self.x_min = float(self.x_coords.min())
        self.x_max = float(self.x_coords.max())
        self.y_min = float(self.y_coords.min())
        self.y_max = float(self.y_coords.max())
        
        self.dx = float(self.x_coords[1] - self.x_coords[0]) if self.nx > 1 else 1.0
        self.dy = float(self.y_coords[1] - self.y_coords[0]) if self.ny > 1 else 1.0
        
        # Check if y coordinates are in decreasing order (typical for rasters)
        # In this case, dy will be negative and row 0 = y_max
        self.y_flipped = self.dy < 0
        if self.y_flipped:
            # Store absolute dy for interpolation
            self.dy_abs = abs(self.dy)
        else:
            self.dy_abs = self.dy
    
    def _bilinear_interpolate(
        self,
        field: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Bilinear interpolation of a 2D field at arbitrary positions.
        
        Parameters
        ----------
        field : np.ndarray
            2D array (ny, nx) to interpolate.
        x : np.ndarray
            X coordinates of query points.
        y : np.ndarray
            Y coordinates of query points.
            
        Returns
        -------
        np.ndarray
            Interpolated values at (x, y) positions.
        """
        # Convert to fractional indices
        ix = (x - self.x_min) / self.dx

        # Handle y-axis orientation
        if self.y_flipped:
            # y_coords[0] = y_max, y_coords[-1] = y_min
            # Row 0 corresponds to y_max, so we need to flip
            iy = (self.y_max - y) / self.dy_abs
        else:
            # Normal case: y_coords[0] = y_min
            iy = (y - self.y_min) / self.dy_abs

        # Guard against NaN coordinates (can arise during RK4 intermediate stages)
        nan_mask = ~(np.isfinite(ix) & np.isfinite(iy))
        if np.any(nan_mask):
            ix = np.where(nan_mask, 0.0, ix)
            iy = np.where(nan_mask, 0.0, iy)

        # Clamp to valid range
        ix = np.clip(ix, 0, self.nx - 1 - 1e-6)
        iy = np.clip(iy, 0, self.ny - 1 - 1e-6)

        # Integer indices
        ix0 = np.floor(ix).astype(int)
        iy0 = np.floor(iy).astype(int)
        ix1 = np.clip(ix0 + 1, 0, self.nx - 1)
        iy1 = np.clip(iy0 + 1, 0, self.ny - 1)

        # Fractional parts
        fx = ix - ix0
        fy = iy - iy0

        # Corner values
        f00 = field[iy0, ix0]
        f10 = field[iy0, ix1]
        f01 = field[iy1, ix0]
        f11 = field[iy1, ix1]

        # Bilinear interpolation
        f0 = f00 * (1 - fx) + f10 * fx
        f1 = f01 * (1 - fx) + f11 * fx
        result = f0 * (1 - fy) + f1 * fy

        # Zero out results for NaN input coordinates
        if np.any(nan_mask):
            result = np.where(nan_mask, 0.0, result)

        return result
    
    def sample_at(
        self,
        t_index: int,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample all fire parameters at given positions for a time step.
        
        Parameters
        ----------
        t_index : int
            Time index (0-indexed).
        x : np.ndarray
            X coordinates of query points.
        y : np.ndarray
            Y coordinates of query points.
            
        Returns
        -------
        ros : np.ndarray
            Head fire rate of spread at each point.
        bros : np.ndarray
            Back fire rate of spread at each point.
        fros : np.ndarray
            Flank fire rate of spread at each point.
        raz : np.ndarray
            Rate of spread azimuth at each point (radians).
        """
        t_index = int(np.clip(t_index, 0, self.nt - 1))
        
        ros = self._bilinear_interpolate(self.ros[t_index], x, y)
        bros = self._bilinear_interpolate(self.bros[t_index], x, y)
        fros = self._bilinear_interpolate(self.fros[t_index], x, y)
        raz = self._bilinear_interpolate(self.raz[t_index], x, y)
        
        return ros, bros, fros, raz


# =============================================================================
# Richards' Differential Equations
# =============================================================================


def compute_spatial_derivatives(
    x: np.ndarray,
    y: np.ndarray,
    ds: float = 1.0,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial derivatives along the fire front curve.
    
    Uses periodic central differences for a closed curve.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates of front vertices.
    y : np.ndarray
        Y coordinates of front vertices.
    ds : float
        Arc length parameter spacing.
    normalize : bool
        If True, return unit tangent vectors instead of raw derivatives.
        
    Returns
    -------
    x_s : np.ndarray
        Derivative of x with respect to arc parameter (or unit tangent x).
    y_s : np.ndarray
        Derivative of y with respect to arc parameter (or unit tangent y).
    """
    # Forward and backward neighbors (periodic)
    x_forward = np.roll(x, -1)
    x_backward = np.roll(x, 1)
    y_forward = np.roll(y, -1)
    y_backward = np.roll(y, 1)
    
    # Central differences
    x_s = (x_forward - x_backward) / (2.0 * ds)
    y_s = (y_forward - y_backward) / (2.0 * ds)
    
    if normalize:
        # Normalize to unit tangent vectors
        mag = np.hypot(x_s, y_s)
        mag[mag == 0] = 1.0  # Avoid division by zero
        x_s = x_s / mag
        y_s = y_s / mag
    
    return x_s, y_s


def richards_velocity(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute fire front velocities using Richards' elliptical spread equations.
    
    Implements the component form of Richards' differential equations for
    elliptical fire spread:
    
        x_t = [b^2 cos(theta) (x_s sin(theta) + y_s cos(theta)) 
              - a^2 sin(theta) (x_s cos(theta) - y_s sin(theta))]
              / sqrt(a^2 (x_s cos(theta) - y_s sin(theta))^2 
                   + b^2 (x_s sin(theta) + y_s cos(theta))^2)
              + c sin(theta)
              
        y_t = [-b^2 sin(theta) (x_s sin(theta) + y_s cos(theta))
              - a^2 cos(theta) (x_s cos(theta) - y_s sin(theta))]
              / sqrt(...)
              + c cos(theta)
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates of front vertices.
    y : np.ndarray
        Y coordinates of front vertices.
    a : np.ndarray
        Semi-major axis of ellipse = (ROS + BROS) / 2.
    b : np.ndarray
        Semi-minor axis of ellipse = FROS.
    c : np.ndarray
        Offset from center = (ROS - BROS) / 2.
    theta : np.ndarray
        Ellipse orientation angle (radians, from positive x-axis).
        
    Returns
    -------
    x_t : np.ndarray
        Time derivative of x (velocity in x direction).
    y_t : np.ndarray
        Time derivative of y (velocity in y direction).
        
    Notes
    -----
    The ellipse parameters relate to ROS components as follows:
    
        a = (ROS_head + ROS_back) / 2  (semi-major axis)
        b = ROS_flank                   (semi-minor axis)
        c = (ROS_head - ROS_back) / 2  (center offset)
        theta = RAZ (spread direction in radians)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Compute normalized spatial derivatives (unit tangent vectors)
    # This is critical for numerical stability with small coordinate values
    x_s, y_s = compute_spatial_derivatives(x, y, normalize=True)
    
    # Broadcast parameters
    a = np.broadcast_to(a, x.shape)
    b = np.broadcast_to(b, x.shape)
    c = np.broadcast_to(c, x.shape)
    theta = np.broadcast_to(theta, x.shape)
    
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    
    # Rotated derivative terms
    term1 = x_s * cos_th - y_s * sin_th
    term2 = x_s * sin_th + y_s * cos_th
    
    # Denominator - with normalized tangents, this is O(a) or O(b), not tiny
    # Use a relative epsilon based on the ellipse parameters
    eps = 1e-12 * (np.mean(a) + np.mean(b) + 1e-12)
    denom = np.sqrt(a**2 * term1**2 + b**2 * term2**2) + eps
    
    # Velocity components
    x_t = (b**2 * cos_th * term2 - a**2 * sin_th * term1) / denom + c * sin_th
    y_t = (-b**2 * sin_th * term2 - a**2 * cos_th * term1) / denom + c * cos_th
    
    return x_t, y_t


def ros_to_ellipse_params(
    ros: np.ndarray,
    bros: np.ndarray,
    fros: np.ndarray,
    raz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ROS components to ellipse parameters for Richards' equations.
    
    Parameters
    ----------
    ros : np.ndarray
        Head fire rate of spread.
    bros : np.ndarray
        Back fire rate of spread.
    fros : np.ndarray
        Flank fire rate of spread.
    raz : np.ndarray
        Rate of spread azimuth (radians).
        
    Returns
    -------
    a : np.ndarray
        Semi-major axis.
    b : np.ndarray
        Semi-minor axis.
    c : np.ndarray
        Center offset.
    theta : np.ndarray
        Orientation angle (radians).
    """
    a = 0.5 * (ros + bros)
    c = 0.5 * (ros - bros)
    b = fros
    theta = raz
    
    return a, b, c, theta


# =============================================================================
# Marker Method for Active Vertices
# =============================================================================


def compute_turning_number(
    px: float,
    py: float,
    x: np.ndarray,
    y: np.ndarray,
) -> int:
    """
    Compute the turning (winding) number of a polygon around a point.
    
    The turning number indicates how many times the curve winds around
    the point. For a simple closed curve:
    - Interior points have turning number 1 (or -1 depending on orientation)
    - Exterior points have turning number 0
    
    Parameters
    ----------
    px : float
        X coordinate of test point.
    py : float
        Y coordinate of test point.
    x : np.ndarray
        X coordinates of polygon vertices.
    y : np.ndarray
        Y coordinates of polygon vertices.
        
    Returns
    -------
    int
        Turning number (typically 0 for exterior, +/-1 for interior).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Vectors from point to each vertex
    vx = x - px
    vy = y - py
    
    # Angles to each vertex
    angles = np.arctan2(vy, vx)
    
    # Angle differences between consecutive vertices
    dtheta = np.diff(angles, append=angles[0])
    
    # Normalize to (-pi, pi]
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
    
    total_angle = np.nansum(dtheta)
    if not np.isfinite(total_angle):
        return 0
    turning_number = int(np.round(total_angle / (2.0 * np.pi)))

    return turning_number


def compute_active_vertices(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.001,
) -> np.ndarray:
    """
    Determine which vertices are active (on outer perimeter) using turning numbers.
    
    A vertex is considered active if it lies on the exterior of the fire front.
    This is determined by computing the turning number for a point slightly
    offset from the vertex along the outward normal.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates of perimeter vertices.
    y : np.ndarray
        Y coordinates of perimeter vertices.
    epsilon : float
        Small offset distance for test point placement.
        
    Returns
    -------
    active : np.ndarray
        Boolean array indicating active vertices.
        
    Notes
    -----
    For a counter-clockwise oriented front:
    - Exterior points (active) have turning number 0
    - Interior points (inactive) have non-zero turning number
    
    The offset is computed along the outward normal, estimated from the
    local tangent direction.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    
    # Compute local tangents via central difference
    x_forward = np.roll(x, -1)
    x_backward = np.roll(x, 1)
    y_forward = np.roll(y, -1)
    y_backward = np.roll(y, 1)
    
    tx = x_forward - x_backward
    ty = y_forward - y_backward
    
    # Outward normal (for CCW curve: rotate tangent 90 degrees right)
    nx = ty
    ny = -tx
    
    # Normalize
    norm = np.hypot(nx, ny)
    norm[norm == 0] = 1.0
    nx /= norm
    ny /= norm
    
    active = np.zeros(n, dtype=bool)
    
    for i in range(n):
        # Test point offset along outward normal
        px = x[i] + epsilon * nx[i]
        py = y[i] + epsilon * ny[i]
        
        tn = compute_turning_number(px, py, x, y)
        
        # Exterior points have turning number 0
        active[i] = (tn == 0)
    
    return active


# =============================================================================
# Perimeter Evolution
# =============================================================================


def evolve_perimeter_step(
    x: np.ndarray,
    y: np.ndarray,
    ros: np.ndarray,
    bros: np.ndarray,
    fros: np.ndarray,
    raz: np.ndarray,
    dt: float,
    use_markers: bool = True,
    marker_epsilon: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evolve fire perimeter by one time step.
    
    Parameters
    ----------
    x : np.ndarray
        Current X coordinates of perimeter vertices.
    y : np.ndarray
        Current Y coordinates of perimeter vertices.
    ros : np.ndarray
        Head fire rate of spread at each vertex.
    bros : np.ndarray
        Back fire rate of spread at each vertex.
    fros : np.ndarray
        Flank fire rate of spread at each vertex.
    raz : np.ndarray
        Rate of spread azimuth at each vertex (radians).
    dt : float
        Time step in same units as ROS (typically minutes).
    use_markers : bool
        If True, use marker method to freeze inactive vertices.
    marker_epsilon : float
        Offset distance for marker method.
        
    Returns
    -------
    x_new : np.ndarray
        Updated X coordinates.
    y_new : np.ndarray
        Updated Y coordinates.
    """
    # Convert ROS to ellipse parameters
    a, b, c, theta = ros_to_ellipse_params(ros, bros, fros, raz)
    
    # Compute velocities
    x_t, y_t = richards_velocity(x, y, a, b, c, theta)
    
    # Apply marker method
    if use_markers:
        active = compute_active_vertices(x, y, marker_epsilon)
        x_t[~active] = 0.0
        y_t[~active] = 0.0
    
    # Explicit Euler update
    x_new = x + dt * x_t
    y_new = y + dt * y_t
    
    return x_new, y_new


def create_initial_perimeter(
    x_center: float,
    y_center: float,
    radius: float,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an initial circular fire perimeter.
    
    Parameters
    ----------
    x_center : float
        X coordinate of ignition point.
    y_center : float
        Y coordinate of ignition point.
    radius : float
        Initial fire radius.
    n_vertices : int
        Number of vertices on perimeter.
        
    Returns
    -------
    x : np.ndarray
        X coordinates of vertices.
    y : np.ndarray
        Y coordinates of vertices.
    """
    theta = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    x = x_center + radius * np.cos(theta)
    y = y_center + radius * np.sin(theta)
    
    return x, y


# =============================================================================
# Fire Perimeter History
# =============================================================================


@dataclass
class FirePerimeterHistory:
    """Container for fire perimeter evolution history."""
    
    perimeters: list[tuple[np.ndarray, np.ndarray]]
    times: list[float]
    
    @property
    def n_steps(self) -> int:
        """Number of stored time steps."""
        return len(self.perimeters)
    
    def get_perimeter(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Get perimeter at given index."""
        return self.perimeters[index]
    
    def get_final_perimeter(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the final perimeter."""
        return self.perimeters[-1] if self.perimeters else (np.array([]), np.array([]))
    
    def compute_areas(self) -> np.ndarray:
        """Compute area enclosed by each perimeter."""
        areas = []
        for x, y in self.perimeters:
            # Shoelace formula
            area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            areas.append(area)
        return np.array(areas)


def simulate_fire_spread(
    param_grid: FireParameterGrid,
    x_ignition: float,
    y_ignition: float,
    dt: float = 1.0,
    n_vertices: int = 300,
    initial_radius: float = 0.5,
    store_every: int = 1,
    max_steps: int | None = None,
    use_markers: bool = True,
    marker_epsilon: float = 1.0,  # 1 meter offset for marker method
    min_ros: float = 0.01,
    is_geographic: bool = False,
    center_latitude: float | None = None,
    # --- New (optional) controls for higher-order advection ---
    advection_integrator: str = "euler",  # "euler" | "rk4" | "euler_extrap" | "implicit"
    resample_spacing: float | None = None,  # marker redistribution target spacing (coordinate units)
    insert_factor: float = 1.5,
    delete_factor: float = 0.5,
    smoothing_weight: float = 0.10,
    smoothing_iters: int = 1,
    redistribute_every: int = 1,
    max_substeps: int = 10,
    max_move_fraction: float = 0.5,
    # Backward-compatibility: older call sites may pass config objects here
    sim_config: object | None = None,
    **kwargs,
) -> FirePerimeterHistory:
    """
    Simulate fire spread from an ignition point.
    
    Parameters
    ----------
    param_grid : FireParameterGrid
        Gridded fire behaviour parameters.
    x_ignition : float
        X coordinate of ignition point.
    y_ignition : float
        Y coordinate of ignition point.
    dt : float
        Time step in minutes.
    n_vertices : int
        Number of vertices on fire perimeter.
    initial_radius : float
        Initial fire radius in meters.
    store_every : int
        Store perimeter every N time steps.
    max_steps : int, optional
        Maximum number of time steps. Defaults to param_grid.nt.
    use_markers : bool
        Use marker method for active vertices.
    marker_epsilon : float
        Offset for marker method.
    min_ros : float
        Minimum ROS threshold (m/min).
    is_geographic : bool
        If True, coordinates are in degrees (lat/lon) and ROS/radius
        will be converted from meters to degrees.
    center_latitude : float, optional
        Center latitude for geographic conversion. Required if is_geographic=True.
        If None, uses y_ignition.

    advection_integrator : str
        Time integration scheme for marker advection.  Options:

        - ``"euler"`` — forward Euler (first-order, legacy default).
        - ``"rk4"`` — classical fourth-order Runge–Kutta.
        - ``"euler_extrap"`` — Euler extrapolation (Richardson): runs
          coarse and fine Euler solutions and extrapolates for
          second-order accuracy with a free error estimate.
        - ``"implicit"`` — backward Euler solved by Newton iteration
          with finite-difference Jacobian.  Unconditionally A-stable
          so no CFL sub-stepping is needed, but each step is more
          expensive (O(N²) per Newton iteration for N markers).
    resample_spacing : float, optional
        Target marker spacing in coordinate units for redistribution. If None, defaults to
        `marker_epsilon` (in coordinate units) when redistribution is enabled.
    redistribute_every : int
        Apply redistribution every N steps (set 0/None to disable).
    insert_factor, delete_factor : float
        Threshold multipliers controlling when to insert/delete markers relative to target spacing.
    smoothing_weight : float
        Weight for very light local smoothing during redistribution (0 disables).
    smoothing_iters : int
        Number of light smoothing iterations (0 disables).
    max_substeps : int
        Maximum internal sub-steps for adaptive sub-stepping.
    max_move_fraction : float
        Fraction of target spacing allowed as max marker movement per sub-step.

    sim_config : object, optional
        Backward-compatibility parameter. Some call sites pass a simulation config object;
        it is not used directly by this function.
    **kwargs
        Ignored extra keyword arguments for backward compatibility.
        
    Returns
    -------
    FirePerimeterHistory
        Evolution history of the fire perimeter.
    
    Notes
    -----
    `sim_config` and `**kwargs` are accepted for backward compatibility and ignored.
    """
    # Handle geographic coordinates
    if is_geographic:
        if center_latitude is None:
            center_latitude = y_ignition
        
        # Conversion factors: meters to degrees
        lat_rad = np.radians(center_latitude)
        meters_per_deg_lat = 111320.0  # approximately constant
        meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
        
        # Convert initial radius from meters to degrees
        # Use average of x and y scale factors for circular approximation
        avg_meters_per_deg = (meters_per_deg_lat + meters_per_deg_lon) / 2.0
        initial_radius_deg = initial_radius / avg_meters_per_deg
        
        logger.debug(
            f"Geographic mode: {meters_per_deg_lon:.0f} m/deg lon, "
            f"{meters_per_deg_lat:.0f} m/deg lat, "
            f"initial radius {initial_radius}m = {initial_radius_deg:.6f}°"
        )
    else:
        initial_radius_deg = initial_radius
        meters_per_deg_lon = 1.0
        meters_per_deg_lat = 1.0
    
    # Initialize perimeter (in coordinate units - degrees if geographic)
    x, y = create_initial_perimeter(x_ignition, y_ignition, initial_radius_deg, n_vertices)
    
    # Scale marker_epsilon for geographic coordinates
    if is_geographic:
        marker_epsilon_coord = marker_epsilon / avg_meters_per_deg
    else:
        marker_epsilon_coord = marker_epsilon
    
    # Storage
    history = FirePerimeterHistory(
        perimeters=[(x.copy(), y.copy())],
        times=[0.0],
    )
    
    # Number of time steps
    if max_steps is None:
        max_steps = param_grid.nt
    n_steps = min(max_steps, param_grid.nt)
    
    logger.info(f"Starting fire simulation: {n_steps} time steps, dt={dt} min")
    logger.info(f"Advection integrator: {(advection_integrator or 'euler').strip().lower()}")
    if is_geographic:
        logger.debug(f"  Geographic mode: avg_meters_per_deg={avg_meters_per_deg:.0f}")
        logger.debug(f"  Initial radius: {initial_radius_deg:.6f} degrees ({initial_radius}m)")

    for step in range(n_steps):
        # ------------------------------------------------------------------
        # Legacy Euler path (unchanged) samples only at the current vertices.
        # RK4 path must sample the grid at intermediate stages to be correct.
        # ------------------------------------------------------------------

        # Always sample current ROS in m/min for extinguishment logic
        ros_now_m, _, _, _ = param_grid.sample_at(step, x, y)
        if np.all(ros_now_m < min_ros):
            logger.info(f"Fire extinguished at step {step} (all ROS below threshold)")
            break

        integrator = (advection_integrator or "euler").strip().lower()

        # Shared velocity closure: resample the grid at arbitrary (x,y) positions.
        # Used by both Euler and RK4 so that adaptive sub-stepping is consistent.
        def _vel_from_grid(xq: np.ndarray, yq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Compute velocity at query points by sampling parameter grid (step-local)."""
            # Sample ROS values at query positions (in m/min from FBP)
            ros_m, bros_m, fros_m, raz = param_grid.sample_at(step, xq, yq)

            # Extinction guard
            if np.all(ros_m < min_ros):
                return np.zeros_like(xq, dtype=float), np.zeros_like(yq, dtype=float)

            # Convert ROS from m/min to coordinate units/min if geographic
            if is_geographic:
                ros_u = ros_m / avg_meters_per_deg
                bros_u = bros_m / avg_meters_per_deg
                fros_u = fros_m / avg_meters_per_deg
            else:
                ros_u = ros_m
                bros_u = bros_m
                fros_u = fros_m

            # Compute Richards velocity using ellipse parameters
            a, b, c, theta = ros_to_ellipse_params(ros_u, bros_u, fros_u, raz)
            x_t, y_t = richards_velocity(xq, yq, a, b, c, theta)

            # Marker method: freeze inactive vertices
            if use_markers:
                active = compute_active_vertices(xq, yq, marker_epsilon_coord)
                x_t[~active] = 0.0
                y_t[~active] = 0.0

            return x_t, y_t

        # Determine target spacing for adaptive sub-stepping and redistribution.
        if resample_spacing is not None and resample_spacing > 0:
            target_spacing = float(resample_spacing)
        else:
            target_spacing = float(marker_epsilon_coord) if marker_epsilon_coord > 0 else 0.0

        # Save last known good coordinates for NaN recovery
        x_prev, y_prev = x.copy(), y.copy()

        if integrator == "rk4":
            # Adaptive sub-stepping to limit movement per substep (RK4)
            x_t0, y_t0 = _vel_from_grid(x, y)
            speed0 = np.hypot(x_t0, y_t0)
            vmax = float(np.max(speed0)) if speed0.size > 0 else 0.0

            if target_spacing > 0 and vmax > 0:
                max_move = max_move_fraction * target_spacing
                nsub = int(np.ceil((vmax * dt) / max_move))
                nsub = int(np.clip(nsub, 1, max_substeps))
            else:
                nsub = 1

            subdt = dt / nsub

            # Perform RK4 integration with sub-stepping
            for isub in range(nsub):
                x, y = _rk4_step_xy(x, y, subdt, _vel_from_grid)
                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                    logger.warning(f"Non-finite coordinates at step {step}, substep {isub}, reverting to previous")
                    x, y = x_prev.copy(), y_prev.copy()
                    break

        elif integrator == "euler_extrap":
            # ---------------------------------------------------------------
            # Euler Extrapolation (Richardson): run coarse (n_base steps) and
            # fine (2*n_base steps) Euler solutions, then extrapolate to
            # cancel the O(dt) error, yielding 2nd-order accuracy.
            # ---------------------------------------------------------------
            x_t0, y_t0 = _vel_from_grid(x, y)
            speed0 = np.hypot(x_t0, y_t0)
            vmax = float(np.max(speed0)) if speed0.size > 0 else 0.0

            if target_spacing > 0 and vmax > 0:
                max_move = max_move_fraction * target_spacing
                n_base = int(np.ceil((vmax * dt) / max_move))
                n_base = int(np.clip(n_base, 1, max_substeps))
            else:
                n_base = 1

            x, y, ee_err = _euler_extrapolation_step_xy(
                x, y, dt, _vel_from_grid, n_base,
            )

            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                logger.warning(
                    f"Non-finite coordinates at step {step} (euler_extrap), "
                    f"reverting to previous"
                )
                x, y = x_prev.copy(), y_prev.copy()

        elif integrator == "implicit":
            # ---------------------------------------------------------------
            # Implicit Euler with Newton linearization — unconditionally
            # stable, no CFL sub-stepping required.
            # ---------------------------------------------------------------
            x, y = _implicit_newton_step_xy(x, y, dt, _vel_from_grid)

            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                logger.warning(
                    f"Non-finite coordinates at step {step} (implicit newton), "
                    f"reverting to previous"
                )
                x, y = x_prev.copy(), y_prev.copy()

        else:
            # -------------------------------
            # Explicit Euler with optional adaptive sub-stepping
            # -------------------------------
            x_t0, y_t0 = _vel_from_grid(x, y)
            speed0 = np.hypot(x_t0, y_t0)
            vmax = float(np.max(speed0)) if speed0.size > 0 else 0.0

            if target_spacing > 0 and vmax > 0:
                max_move = max_move_fraction * target_spacing
                nsub = int(np.ceil((vmax * dt) / max_move))
                nsub = int(np.clip(nsub, 1, max_substeps))
            else:
                nsub = 1

            subdt = dt / nsub

            for isub in range(nsub):
                x_t, y_t = _vel_from_grid(x, y)
                x = x + subdt * x_t
                y = y + subdt * y_t

                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                    logger.warning(f"Non-finite coordinates at step {step}, substep {isub}, reverting to previous")
                    x, y = x_prev.copy(), y_prev.copy()
                    break

        # ------------------------------------------------------------------
        # Marker redistribution + optional light smoothing (shared)
        # Keeps a quasi-uniform discretization to improve curvature/tangent
        # estimates and reduce numerical artifacts without shrinking the front.
        # ------------------------------------------------------------------
        if target_spacing > 0 and redistribute_every and int(redistribute_every) > 0:
            if (step % int(redistribute_every)) == 0:
                x, y = _insert_delete_markers_by_spacing(
                    x,
                    y,
                    spacing=target_spacing,
                    insert_factor=insert_factor,
                    delete_factor=delete_factor,
                    max_n=int(n_vertices * 10),
                    min_n=int(n_vertices // 2),
                )

                # Very light local smoothing to suppress high-frequency numerical noise.
                # Unlike Chaikin smoothing, this does not materially shrink the perimeter.
                if smoothing_iters and int(smoothing_iters) > 0 and smoothing_weight and float(smoothing_weight) > 0:
                    x, y = _smooth_ring_local_xy(x, y, w=float(smoothing_weight), n_iter=int(smoothing_iters))

        # Log progress at intervals
        if (step + 1) % 100 == 0:
            # Compute current fire extent
            x_extent = (np.max(x) - np.min(x))
            y_extent = (np.max(y) - np.min(y))
            if is_geographic:
                x_extent_m = x_extent * meters_per_deg_lon
                y_extent_m = y_extent * meters_per_deg_lat
                logger.info(f"  Step {step+1}/{n_steps}: extent {x_extent_m:.0f}m x {y_extent_m:.0f}m")
            else:
                logger.info(f"  Step {step+1}/{n_steps}: extent {x_extent:.0f}m x {y_extent:.0f}m")

        # Store if requested
        if (step + 1) % store_every == 0:
            history.perimeters.append((x.copy(), y.copy()))
            history.times.append((step + 1) * dt)

    logger.info(f"Simulation complete: {len(history.perimeters)} perimeters stored")

    return history


# =============================================================================
# JAX-differentiable marker advection
# =============================================================================

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from functools import partial as _jax_partial

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

if _HAS_JAX:

    _EPS_JAX = 1e-12

    # -----------------------------------------------------------------
    # JAX utility functions (bilinear sampling, spatial derivatives,
    # Richards velocity) — differentiable building blocks.
    # -----------------------------------------------------------------

    def _jax_bilinear_sample(
        field: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
    ) -> jnp.ndarray:
        """
        Bilinear interpolation of a 2D field at arbitrary (x, y).

        Parameters
        ----------
        field : jnp.ndarray
            2D grid (ny, nx).
        x, y : jnp.ndarray
            Query coordinates (1D, N markers).
        x0, y0 : float
            Grid origin (x_coords[0], y_coords[0]).
        dx, dy : float
            Grid spacing.  *dy* may be negative for descending y-axis.
        """
        ny, nx = field.shape

        ix = (x - x0) / dx
        iy = (y - y0) / dy

        ix = jnp.clip(ix, 0.0, nx - 1 - 1e-6)
        iy = jnp.clip(iy, 0.0, ny - 1 - 1e-6)

        ix0 = jnp.floor(ix).astype(jnp.int32)
        iy0 = jnp.floor(iy).astype(jnp.int32)
        ix1 = jnp.minimum(ix0 + 1, nx - 1)
        iy1 = jnp.minimum(iy0 + 1, ny - 1)

        fx = ix - ix0
        fy = iy - iy0

        f00 = field[iy0, ix0]
        f10 = field[iy0, ix1]
        f01 = field[iy1, ix0]
        f11 = field[iy1, ix1]

        return (f00 * (1 - fx) + f10 * fx) * (1 - fy) + \
               (f01 * (1 - fx) + f11 * fx) * fy

    def _jax_spatial_derivatives(
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Normalized central differences on a closed marker ring."""
        x_s = (jnp.roll(x, -1) - jnp.roll(x, 1)) / 2.0
        y_s = (jnp.roll(y, -1) - jnp.roll(y, 1)) / 2.0
        mag = jnp.sqrt(x_s ** 2 + y_s ** 2 + _EPS_JAX)
        return x_s / mag, y_s / mag

    def _jax_richards_velocity(
        x: jnp.ndarray,
        y: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        theta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Richards' elliptical spread velocity (JAX)."""
        x_s, y_s = _jax_spatial_derivatives(x, y)

        cos_th = jnp.cos(theta)
        sin_th = jnp.sin(theta)

        term1 = x_s * cos_th - y_s * sin_th
        term2 = x_s * sin_th + y_s * cos_th

        denom = jnp.sqrt(a ** 2 * term1 ** 2 + b ** 2 * term2 ** 2 + _EPS_JAX)

        x_t = (b ** 2 * cos_th * term2 - a ** 2 * sin_th * term1) / denom + c * sin_th
        y_t = (-b ** 2 * sin_th * term2 - a ** 2 * cos_th * term1) / denom + c * cos_th

        return x_t, y_t

    def _jax_marker_velocity(
        x: jnp.ndarray,
        y: jnp.ndarray,
        ros_2d: jnp.ndarray,
        bros_2d: jnp.ndarray,
        fros_2d: jnp.ndarray,
        raz_2d: jnp.ndarray,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample FBP grids at markers, return Richards velocity."""
        ros = _jax_bilinear_sample(ros_2d, x, y, x0, y0, dx, dy)
        bros = _jax_bilinear_sample(bros_2d, x, y, x0, y0, dx, dy)
        fros = _jax_bilinear_sample(fros_2d, x, y, x0, y0, dx, dy)
        raz = _jax_bilinear_sample(raz_2d, x, y, x0, y0, dx, dy)

        a = 0.5 * (ros + bros)
        b = fros
        c_param = 0.5 * (ros - bros)

        return _jax_richards_velocity(x, y, a, b, c_param, raz)

    def _jax_packed_velocity(
        z: jnp.ndarray,
        ros_2d: jnp.ndarray,
        bros_2d: jnp.ndarray,
        fros_2d: jnp.ndarray,
        raz_2d: jnp.ndarray,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
    ) -> jnp.ndarray:
        """Velocity on packed state z = [x0,y0,x1,y1,...] → [vx0,vy0,...]."""
        x = z[0::2]
        y = z[1::2]
        vx, vy = _jax_marker_velocity(
            x, y, ros_2d, bros_2d, fros_2d, raz_2d, x0, y0, dx, dy,
        )
        v = jnp.zeros_like(z)
        v = v.at[0::2].set(vx)
        v = v.at[1::2].set(vy)
        return v

    # -----------------------------------------------------------------
    # Differentiable Euler Extrapolation (Richardson)
    # -----------------------------------------------------------------

    def jax_euler_extrapolation_step(
        x: jnp.ndarray,
        y: jnp.ndarray,
        dt: jnp.ndarray,
        ros_2d: jnp.ndarray,
        bros_2d: jnp.ndarray,
        fros_2d: jnp.ndarray,
        raz_2d: jnp.ndarray,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        n_base: int = 1,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Differentiable Euler extrapolation (Richardson) step in JAX.

        Runs forward Euler at two resolutions (``n_base`` and ``2·n_base``
        sub-steps) and extrapolates to cancel the leading-order error,
        yielding second-order temporal accuracy.

        Fully differentiable via standard JAX autodiff — no custom VJP
        required.  ``n_base`` must be a Python int (static for JIT).

        Parameters
        ----------
        x, y : jnp.ndarray
            Marker positions (N,).
        dt : scalar
            Time step (traced — gradients flow through it).
        ros_2d, bros_2d, fros_2d, raz_2d : jnp.ndarray
            FBP parameter grids for this timestep (ny, nx).
        x0, y0, dx, dy : float
            Grid origin and spacing (static).
        n_base : int
            Base sub-step count (static for JIT).

        Returns
        -------
        x_new, y_new : jnp.ndarray
            Extrapolated marker positions.
        """
        def _make_euler_body(dt_sub):
            def body(carry, _):
                x_, y_ = carry
                vx, vy = _jax_marker_velocity(
                    x_, y_, ros_2d, bros_2d, fros_2d, raz_2d,
                    x0, y0, dx, dy,
                )
                return (x_ + dt_sub * vx, y_ + dt_sub * vy), None
            return body

        # Coarse: n_base steps
        (x_c, y_c), _ = lax.scan(
            _make_euler_body(dt / n_base), (x, y), None, length=n_base,
        )

        # Fine: 2·n_base steps
        (x_f, y_f), _ = lax.scan(
            _make_euler_body(dt / (2 * n_base)), (x, y), None, length=2 * n_base,
        )

        # Richardson extrapolation: 2·fine − coarse
        return 2.0 * x_f - x_c, 2.0 * y_f - y_c

    # -----------------------------------------------------------------
    # Differentiable Implicit Euler with custom VJP
    # (Implicit Function Theorem — avoids differentiating through Newton)
    # -----------------------------------------------------------------

    @_jax_partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12))
    def jax_implicit_euler_step(
        x: jnp.ndarray,
        y: jnp.ndarray,
        dt,
        ros_2d: jnp.ndarray,
        bros_2d: jnp.ndarray,
        fros_2d: jnp.ndarray,
        raz_2d: jnp.ndarray,
        x0: float = 0.0,
        y0: float = 0.0,
        dx: float = 1.0,
        dy: float = 1.0,
        max_iters: int = 4,
        tol: float = 1e-10,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Implicit Euler step with custom VJP via the implicit function theorem.

        **Forward pass** — fixed-point iteration (no Jacobian needed):

            z^{m+1} = z^n + dt · g(z^m)

        **Backward pass** — one adjoint linear solve using the Jacobian at
        the converged solution z*:

            [I − dt · J_g]^T  μ = λ       (adjoint system)
            ∂L/∂z₀ = μ                    (gradient w.r.t. initial markers)
            ∂L/∂dt = μ^T g(z*)            (gradient w.r.t. time step)
            ∂L/∂θ  = dt · (∂g/∂θ)^T μ    (gradient w.r.t. FBP grids)

        Because the backward pass uses the implicit function theorem
        rather than differentiating through the iterations, it is both
        cheaper and more numerically stable than unrolled autodiff.

        Parameters
        ----------
        x, y : jnp.ndarray
            Marker positions (N,).
        dt : scalar
            Time step (traced).
        ros_2d, bros_2d, fros_2d, raz_2d : jnp.ndarray
            FBP parameter grids (ny, nx).
        x0, y0, dx, dy : float
            Grid metadata (static, non-differentiable).
        max_iters : int
            Fixed-point iterations (static).
        tol : float
            Unused (reserved for early stopping); iterations are fixed
            for JIT compatibility.

        Returns
        -------
        x_new, y_new : jnp.ndarray
            Updated marker positions.
        """
        N2 = 2 * len(x)
        z0 = jnp.zeros(N2)
        z0 = z0.at[0::2].set(x)
        z0 = z0.at[1::2].set(y)

        g_args = (ros_2d, bros_2d, fros_2d, raz_2d)

        # Forward Euler predictor
        g0 = _jax_packed_velocity(z0, *g_args, x0, y0, dx, dy)
        zm = z0 + dt * g0

        # Fixed-point iterations: z^{m+1} = z^n + dt * g(z^m)
        def _fp_body(_, zm_):
            gm = _jax_packed_velocity(zm_, *g_args, x0, y0, dx, dy)
            return z0 + dt * gm

        zm = lax.fori_loop(0, max_iters, _fp_body, zm)

        return zm[0::2], zm[1::2]

    def _implicit_fwd(x, y, dt, ros_2d, bros_2d, fros_2d, raz_2d,
                      x0, y0, dx, dy, max_iters, tol):
        """Forward rule: run primal, stash residuals for backward."""
        x_out, y_out = jax_implicit_euler_step(
            x, y, dt, ros_2d, bros_2d, fros_2d, raz_2d,
            x0, y0, dx, dy, max_iters, tol,
        )
        residuals = (x, y, dt, ros_2d, bros_2d, fros_2d, raz_2d, x_out, y_out)
        return (x_out, y_out), residuals

    def _implicit_bwd(x0, y0, dx, dy, max_iters, tol, residuals, cotangents):
        """
        Backward rule via the implicit function theorem.

        Given the implicit equation F(z*, z₀, θ) = z* − z₀ − dt·g(z*, θ) = 0,
        the adjoint μ satisfies [I − dt·J_g]^T μ = λ.  All parameter
        gradients then follow from μ without differentiating through the
        forward iterations.
        """
        x_in, y_in, dt, ros_2d, bros_2d, fros_2d, raz_2d, x_star, y_star = residuals
        lam_x, lam_y = cotangents

        N2 = 2 * len(x_star)

        # Pack converged solution and cotangent
        z_star = jnp.zeros(N2)
        z_star = z_star.at[0::2].set(x_star)
        z_star = z_star.at[1::2].set(y_star)

        lam = jnp.zeros(N2)
        lam = lam.at[0::2].set(lam_x)
        lam = lam.at[1::2].set(lam_y)

        g_args = (ros_2d, bros_2d, fros_2d, raz_2d)

        # J_g = ∂g/∂z at z* (2N × 2N Jacobian)
        J_g = jax.jacrev(_jax_packed_velocity)(
            z_star, *g_args, x0, y0, dx, dy,
        )

        # Adjoint solve: [I − dt · J_g]^T μ = λ
        A = jnp.eye(N2) - dt * J_g
        mu = jnp.linalg.solve(A.T, lam)

        # ∂L/∂z₀ = μ  (since ∂F/∂z₀ = −I)
        grad_x_in = mu[0::2]
        grad_y_in = mu[1::2]

        # ∂L/∂dt = μ^T · g(z*)
        g_star = _jax_packed_velocity(z_star, *g_args, x0, y0, dx, dy)
        grad_dt = jnp.dot(mu, g_star)

        # ∂L/∂θ = dt · (∂g/∂θ)^T · μ   via VJP of g w.r.t. parameter grids
        _, vjp_params = jax.vjp(
            lambda r, b, f, a: _jax_packed_velocity(
                z_star, r, b, f, a, x0, y0, dx, dy,
            ),
            *g_args,
        )
        grad_ros, grad_bros, grad_fros, grad_raz = vjp_params(dt * mu)

        return (grad_x_in, grad_y_in, grad_dt,
                grad_ros, grad_bros, grad_fros, grad_raz)

    jax_implicit_euler_step.defvjp(_implicit_fwd, _implicit_bwd)
