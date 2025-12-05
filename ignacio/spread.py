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
    
    total_angle = np.sum(dtheta)
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
        
    Returns
    -------
    FirePerimeterHistory
        Evolution history of the fire perimeter.
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
    if is_geographic:
        logger.debug(f"  Geographic mode: avg_meters_per_deg={avg_meters_per_deg:.0f}")
        logger.debug(f"  Initial radius: {initial_radius_deg:.6f} degrees ({initial_radius}m)")
    
    for step in range(n_steps):
        # Sample parameters at current vertex positions
        # ROS values are in m/min from FBP calculations
        ros, bros, fros, raz = param_grid.sample_at(step, x, y)
        
        # Convert ROS from m/min to coordinate units/min if geographic
        if is_geographic:
            # For each vertex, convert m/min to deg/min
            # Use different scale factors for x and y components
            # This is handled in evolve_perimeter_step by scaling the velocity
            ros_deg = ros / avg_meters_per_deg
            bros_deg = bros / avg_meters_per_deg
            fros_deg = fros / avg_meters_per_deg
            min_ros_deg = min_ros / avg_meters_per_deg
        else:
            ros_deg = ros
            bros_deg = bros
            fros_deg = fros
            min_ros_deg = min_ros
        
        # Apply minimum ROS threshold
        below_threshold = ros < min_ros  # Always compare in m/min
        if np.all(below_threshold):
            logger.info(f"Fire extinguished at step {step} (all ROS below threshold)")
            break
        
        # Evolve perimeter (using coordinate-unit ROS values)
        x_new, y_new = evolve_perimeter_step(
            x, y, ros_deg, bros_deg, fros_deg, raz, dt,
            use_markers=use_markers,
            marker_epsilon=marker_epsilon_coord,
        )
        
        x, y = x_new, y_new
        
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
