
"""
IGNACIO level-set fire propagation (JAX), Richards ellipse formulation.

Signed level set: phi < 0 burned; phi = 0 is the perimeter.

For head, back, and flank rates (m/min), define:

    a = (head + back)/2; b = flank; c = (head - back)/2

With spread unit vector u and perpendicular v, the Hamiltonian is

    H(grad(phi)) = c*(u.grad(phi)) + sqrt((a*u.grad(phi))**2
                                     + (b*v.grad(phi))**2).

It is evolved with a first-order local Lax-Friedrichs numerical Hamiltonian,
using bounds on both directional characteristic speeds for its timestep.

For constant fields and a point ignition, the exact front is a translated
ellipse. A finite-radius circular ignition instead evolves as the Minkowski
sum of the initial circle and an elliptical growth set; the resulting front
is not, in general, an exact ellipse. Numerical convergence must be tested
by refining both the spatial grid and timestep.

Contours are extracted from phi=0 (sub-cell interpolated), not a binary mask.
A sub-grid ignition may not be resolvable and is *never* silently enlarged.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Sequence

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


def initialize_phi_multiple(
    x_coords: jnp.ndarray,
    y_coords: jnp.ndarray,
    ignition_points: Sequence[tuple[float, float]],
    radius: float,
) -> jnp.ndarray:
    """
    Signed distance to the union of equal-radius ignition circles.

    ``phi < 0`` is burned; ``phi == 0`` is the initial front. Taking a
    pointwise minimum is the exact signed distance to a union of disks.

    This operation initializes topology; the existing Richards LLF solver
    subsequently evolves a *single* scalar level-set field.
    """
    points = np.asarray(ignition_points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("ignition_points must have shape (N, 2), N >= 1")

    if not np.isfinite(points).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            "Ignition positions and radius must be finite and radius > 0"
        )

    X, Y = jnp.meshgrid(x_coords, y_coords)
    phi = jnp.full(X.shape, jnp.inf, dtype=jnp.float64)

    for xc, yc in points:
        distance = jnp.hypot(
            X - float(xc), Y - float(yc)
        ) - float(radius)
        phi = jnp.minimum(phi, distance)

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

    phi_x = (
        phi_padded[1:-1, 2:] - phi_padded[1:-1, :-2]
    ) / (2.0 * dx)

    phi_y = (
        phi_padded[2:, 1:-1] - phi_padded[:-2, 1:-1]
    ) / (2.0 * dy)

    # If y is descending (row 0 = north), flip y gradient
    phi_y = jnp.where(y_ascending, phi_y, -phi_y)

    return phi_x, phi_y


def _one_sided_derivatives(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    y_ascending: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return physical-coordinate (D-x, D+x, D-y, D+y)."""

    padded = jnp.pad(phi, 1, mode="edge")
    center = padded[1:-1, 1:-1]

    xm = (center - padded[1:-1, :-2]) / dx
    xp = (padded[1:-1, 2:] - center) / dx

    ym_index = (center - padded[:-2, 1:-1]) / dy
    yp_index = (padded[2:, 1:-1] - center) / dy

    if y_ascending:
        return xm, xp, ym_index, yp_index

    return xm, xp, -yp_index, -ym_index


def _godunov_signed_gradients(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    y_ascending: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Upwind |grad(phi)| for respectively positive and negative speeds."""

    xm, xp, ym, yp = _one_sided_derivatives(
        phi, dx, dy, y_ascending
    )

    # Per-axis max is essential. Summing both one-sided contributions overcounts.
    gx_pos = jnp.maximum(
        jnp.maximum(xm, 0)**2,
        jnp.minimum(xp, 0)**2
    )
    gy_pos = jnp.maximum(
        jnp.maximum(ym, 0)**2,
        jnp.minimum(yp, 0)**2
    )

    gx_neg = jnp.maximum(
        jnp.minimum(xm, 0)**2,
        jnp.maximum(xp, 0)**2
    )
    gy_neg = jnp.maximum(
        jnp.minimum(ym, 0)**2,
        jnp.maximum(yp, 0)**2
    )

    return (
        jnp.sqrt(gx_pos + gy_pos),
        jnp.sqrt(gx_neg + gy_neg)
    )


def compute_gradient_magnitude_upwind(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    y_ascending: bool = True,
) -> jnp.ndarray:
    """Positive-speed Godunov gradient, retained for backward compatibility."""
    return _godunov_signed_gradients(
        phi, dx, dy, y_ascending
    )[0]


# =============================================================================
# Speed function
# =============================================================================

def _richards_coefficients(ros, bros, fros, raz):
    """Growth semi-axis rates and spread direction (RAZ in radians)."""

    head = jnp.maximum(jnp.asarray(ros), 0.0)
    back = jnp.maximum(jnp.asarray(bros), 0.0)
    flank = jnp.maximum(jnp.asarray(fros), 0.0)

    a = 0.5 * (head + back)
    b = flank
    c = 0.5 * (head - back)

    ux = jnp.sin(raz)  # azimuth clockwise from north
    uy = jnp.cos(raz)

    return a, b, c, ux, uy


def richards_hamiltonian(
    phi_x, phi_y, ros, bros, fros, raz
):
    """H(p) = c*u.p + sqrt(a^2*(u.p)^2 + b^2*(v.p)^2)."""

    a, b, c, ux, uy = _richards_coefficients(
        ros, bros, fros, raz
    )

    along = ux * phi_x + uy * phi_y
    across = -uy * phi_x + ux * phi_y

    return c * along + jnp.sqrt(
        (a * along)**2 + (b * across)**2
    )


def compute_elliptical_speed(
    phi_x, phi_y, ros, bros, fros, raz
):
    """Outward normal speed for Richards' ellipse (diagnostic compatibility)."""

    magnitude = jnp.sqrt(phi_x**2 + phi_y**2)

    return jnp.where(
        magnitude > EPS,
        richards_hamiltonian(
            phi_x, phi_y, ros, bros, fros, raz
        ) / jnp.maximum(magnitude, EPS),
        0.0,
    )


def _characteristic_bounds(ros, bros, fros, raz):
    """Conservative coordinate-wise |dH/dp| bounds for a monotone LLF flux."""

    a, b, c, ux, uy = _richards_coefficients(
        ros, bros, fros, raz
    )

    alpha_x = jnp.abs(c * ux) + jnp.sqrt(
        (a * ux)**2 + (b * uy)**2
    )
    alpha_y = jnp.abs(c * uy) + jnp.sqrt(
        (a * uy)**2 + (b * ux)**2
    )

    return alpha_x, alpha_y


def _numerical_hamiltonian(
    phi, ros, bros, fros, raz, dx, dy, y_ascending
):
    """
    Consistent first-order local Lax-Friedrichs Hamiltonian.

    Central gradients evaluate H. Conservative characteristic bounds supply
    LLF dissipation so that the first-order update is monotone under its CFL.
    """
    xm, xp, ym, yp = _one_sided_derivatives(
        phi, dx, dy, y_ascending
    )

    px = 0.5 * (xm + xp)
    py = 0.5 * (ym + yp)

    alpha_x, alpha_y = _characteristic_bounds(
        ros, bros, fros, raz
    )

    return (
        richards_hamiltonian(
            px, py, ros, bros, fros, raz
        )
        - 0.5 * alpha_x * (xp - xm)
        - 0.5 * alpha_y * (yp - ym)
    )


def _cfl_rate_numpy(ros, bros, fros, raz, dx, dy):
    """Maximum sum of directional characteristic speeds divided by spacing."""

    a = (
        np.maximum(ros, 0) + np.maximum(bros, 0)
    ) * 0.5
    c = (
        np.maximum(ros, 0) - np.maximum(bros, 0)
    ) * 0.5
    b = np.maximum(fros, 0)

    ux, uy = np.sin(raz), np.cos(raz)

    sx = np.abs(c * ux) + np.hypot(
        a * ux, b * uy
    )
    sy = np.abs(c * uy) + np.hypot(
        a * uy, b * ux
    )

    return float(np.max(sx / dx + sy / dy))


# =============================================================================
# Reinitialization
# =============================================================================

def reinitialize_phi(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    n_iter: int = 10,
    dt_reinit: float | None = None,
    y_ascending: bool = True,
) -> jnp.ndarray:
    """
    Signed Godunov reinitialization of |grad(phi)|=1.

    For sign(phi0)>0, use positive-speed upwinding; for sign(phi0)<0,
    use negative-speed upwinding. The original sign field stays frozen.

    This scheme can still shift a discretized zero contour: verification
    experiments should set reinit_interval=0 initially.
    """
    if n_iter <= 0:
        return phi

    if dt_reinit is None:
        dt_reinit = 0.3 * min(dx, dy)

    if dt_reinit > 0.45 * min(dx, dy):
        raise ValueError(
            "Reinitialization pseudo-time step is too large"
        )

    phi0 = phi
    h = min(dx, dy)
    sign0 = phi0 / jnp.sqrt(phi0**2 + h**2)

    def _step(current, _):
        grad_pos, grad_neg = _godunov_signed_gradients(
            current, dx, dy, y_ascending
        )
        grad = jnp.where(
            sign0 >= 0, grad_pos, grad_neg
        )
        return current - dt_reinit * sign0 * (grad - 1.0), None

    result, _ = lax.scan(
        _step, phi, None, length=n_iter
    )

    return result


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
    """Advance one outer timestep using LLF and CFL-controlled forward Euler."""

    if dx <= 0 or dy <= 0 or dt <= 0 or not 0 < cfl_factor <= 1:
        raise ValueError(
            "dx, dy, dt must be positive; cfl_factor in (0, 1]"
        )

    ax, ay = _characteristic_bounds(
        ros, bros, fros, raz
    )
    rate = jnp.max(ax / dx + ay / dy)
    n_sub = jnp.maximum(
        1,
        jnp.ceil(dt * rate / cfl_factor).astype(jnp.int32)
    )
    dt_sub = dt / n_sub

    def _update(i, state):
        return state - dt_sub * _numerical_hamiltonian(
            state, ros, bros, fros, raz,
            dx, dy, y_ascending
        )

    return lax.fori_loop(
        0, n_sub, _update, phi
    )


@partial(
    jax.jit,
    static_argnames=(
        "dx", "dy", "cfl_factor", "n_substeps",
        "y_ascending"
    )
)
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
    Fixed-substep differentiable LLF evolution.

    Caller must ensure dt/n_substeps meets the directional CFL bound.
    """
    dt_sub = dt / n_substeps

    def _step(i, state):
        return state - dt_sub * _numerical_hamiltonian(
            state, ros, bros, fros, raz,
            dx, dy, y_ascending
        )

    return lax.fori_loop(
        0, n_substeps, _step, phi
    )


# =============================================================================
# Contour extraction
# =============================================================================

def _polygon_contains_point(x, y, x0, y0):
    """Even-odd point-in-polygon check in world coordinates."""

    inside = False
    j = len(x) - 1

    for i in range(len(x)):
        crosses = (
            (y[i] > y0) != (y[j] > y0)
        )

        if crosses:
            intersection = (
                (x[j] - x[i])
                * (y0 - y[i])
                / (y[j] - y[i])
                + x[i]
            )

            if x0 < intersection:
                inside = not inside

        j = i

    return inside


def extract_contour(
    phi_np: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x_ignition: float | None = None,
    y_ignition: float | None = None,
    burned_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract the actual zero contour of phi, including sub-cell geometry.

    `burned_mask` is accepted for compatibility, but deliberately ignored:
    converting phi to a binary raster introduces grid-aligned artefacts.

    When multiple closed contours exist, prefer one enclosing the ignition.
    Other contours (e.g. spotting fires) cannot be represented in the
    historical single-perimeter return format and are not merged here.
    """
    from skimage.measure import find_contours

    field = np.asarray(phi_np, dtype=np.float64)
    xs = np.asarray(x_coords, dtype=np.float64)
    ys = np.asarray(y_coords, dtype=np.float64)

    if field.shape != (len(ys), len(xs)):
        raise ValueError(
            "phi shape does not match coordinate lengths"
        )

    if not np.isfinite(field).all():
        raise ValueError(
            "phi contains non-finite values"
        )

    if not (field.min() < 0 < field.max()):
        return None

    contours = find_contours(
        field, level=0.0
    )

    # ---------------------------------------------------------
    # H4 DIAGNOSTIC:
    # Do not silently discard open zero contours.
    # ---------------------------------------------------------

    open_contours = [
        rc for rc in contours
        if len(rc) >= 4
        and np.linalg.norm(
            rc[0] - rc[-1]
        ) > 1e-5
    ]

    if open_contours:
        raise ValueError(
            "Level-set contour extraction found "
            f"{len(open_contours)} open zero contour(s). "
            "A fire front may have reached the "
            "computational boundary. Refusing to "
            "export a potentially incomplete perimeter."
        )

    candidates = []

    for rc in contours:
        if len(rc) < 4:
            continue

        # Never join an open contour that has reached the computational edge.
        if np.linalg.norm(rc[0] - rc[-1]) > 1e-5:
            continue

        rows, cols = rc[:, 0], rc[:, 1]

        x = np.interp(
            cols, np.arange(len(xs)), xs
        )
        y = np.interp(
            rows, np.arange(len(ys)), ys
        )

        # Shift origin before the shoelace sum to avoid cancellation with
        # large projected northing/easting coordinates.
        xx, yy = x - x[0], y - y[0]

        area = 0.5 * abs(
            np.dot(xx[:-1], yy[1:])
            - np.dot(yy[:-1], xx[1:])
        )

        contains = (
            x_ignition is not None
            and y_ignition is not None
            and _polygon_contains_point(
                x, y, x_ignition, y_ignition
            )
        )

        candidates.append(
            (contains, area, x, y)
        )

    if not candidates:
        return None

    # Prefer a contour enclosing the ignition, then choose largest area.
    _, _, x, y = max(
        candidates,
        key=lambda item: (item[0], item[1])
    )

    return x, y


def _bilinear_phi_value(field, xs, ys, x, y):
    """Bilinear interpolation of a scalar field on regular (possibly flipped-y) axes."""

    col = (
        float(x) - xs[0]
    ) / (xs[1] - xs[0])

    row = (
        float(y) - ys[0]
    ) / (ys[1] - ys[0])

    if not (
        0.0 <= col < len(xs) - 1
        and 0.0 <= row < len(ys) - 1
    ):
        return float("nan")

    c0, r0 = int(np.floor(col)), int(np.floor(row))
    tx, ty = col - c0, row - r0

    return float(
        (1 - ty)
        * (
            (1 - tx) * field[r0, c0]
            + tx * field[r0, c0 + 1]
        )
        + ty
        * (
            (1 - tx) * field[r0 + 1, c0]
            + tx * field[r0 + 1, c0 + 1]
        )
    )


def extract_all_contours(
    phi_np: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    return_holes: bool = False,
):
    """
    Return all closed EXTERIOR phi=0 fronts in world coordinates.

    With ``return_holes=True`` return ``(exteriors, holes)``: exteriors CCW
    and unburned-island boundaries CW, matching the marker outer-hull
    convention, so a hole-aware history can store both.

    The isocontours are interpolated with scikit-image. A near-boundary
    interior-side sample of ``phi`` distinguishes an exterior burned-area
    boundary (inside phi < 0) from a hole boundary (inside phi > 0).

    Only exterior rings fit the multi-front history data structure. If the
    level-set field contains unburned islands, raise rather than silently
    treating them as independent fires and corrupting burned-area statistics.

    An open contour at the domain edge is an error for these experiments.
    """
    from skimage.measure import find_contours

    field = np.asarray(phi_np, dtype=np.float64)
    xs = np.asarray(x_coords, dtype=np.float64)
    ys = np.asarray(y_coords, dtype=np.float64)

    if (
        field.shape != (len(ys), len(xs))
        or len(xs) < 2
        or len(ys) < 2
    ):
        raise ValueError(
            "Level-set field shape or coordinate arrays are invalid"
        )

    if not np.isfinite(field).all():
        raise ValueError(
            "Level-set field contains non-finite values"
        )

    if not (
        np.min(field) < 0.0 < np.max(field)
    ):
        return ([], []) if return_holes else []

    d_min = min(
        abs(float(xs[1] - xs[0])),
        abs(float(ys[1] - ys[0]))
    )

    exterior = []
    hole_rings = []

    for rc in find_contours(field, 0.0):

        if len(rc) < 4:
            continue

        if np.linalg.norm(rc[0] - rc[-1]) > 1e-5:
            raise ValueError(
                "Open zero contour reached the computational boundary; "
                "enlarge the level-set domain before comparing fire perimeters."
            )

        x = np.interp(
            rc[:, 1], np.arange(len(xs)), xs
        )
        y = np.interp(
            rc[:, 0], np.arange(len(ys)), ys
        )

        # The marching-squares output contains a duplicated closure vertex.
        if np.hypot(
            x[0] - x[-1],
            y[0] - y[-1]
        ) < 1e-7:
            x, y = x[:-1], y[:-1]

        if len(x) < 3:
            continue

        xx, yy = x - x[0], y - y[0]

        twice_area = float(
            np.dot(xx, np.roll(yy, -1))
            - np.dot(yy, np.roll(xx, -1))
        )

        if abs(twice_area) < 1e-9:
            continue

        # Orient all output rings counter-clockwise (world x/y coordinates).
        if twice_area < 0:
            x, y = x[::-1], y[::-1]

        # A positive-area CCW ring has its geometric interior on the left.
        # Sample a short distance to the left of several contour edges.
        # For an exterior fire boundary the level-set interior is negative.
        inside_values = []

        indices = np.linspace(
            0, len(x) - 1,
            min(30, len(x)),
            dtype=int,
        )

        for j in indices:
            k = (int(j) + 1) % len(x)

            ex, ey = (
                x[k] - x[j],
                y[k] - y[j],
            )

            seg_len = float(
                np.hypot(ex, ey)
            )

            if seg_len < 1e-10:
                continue

            px = (
                0.5 * (x[j] + x[k])
                - (ey / seg_len) * 0.35 * d_min
            )
            py = (
                0.5 * (y[j] + y[k])
                + (ex / seg_len) * 0.35 * d_min
            )

            value = _bilinear_phi_value(
                field, xs, ys, px, py
            )

            if np.isfinite(value):
                inside_values.append(value)

        if not inside_values:
            raise ValueError(
                "Could not classify an extracted level-set contour"
            )

        if (
            np.count_nonzero(
                np.asarray(inside_values) < 0.0
            ) > len(inside_values) / 2
        ):
            exterior.append((x, y))
        else:
            # Unburned island: store clockwise (burned land on the left).
            hole_rings.append((x[::-1].copy(), y[::-1].copy()))

    if hole_rings and not return_holes:
        raise NotImplementedError(
            f"Level-set field has {len(hole_rings)} unburned interior hole contour(s). "
            "Exterior-only FirePerimeterHistory cannot represent interior "
            "boundaries without area errors; a hole-aware geometry export is required."
        )

    # Deterministic ordering, largest burned component first.
    def _area(ring):
        xx = ring[0] - ring[0][0]
        yy = ring[1] - ring[1][0]

        return 0.5 * abs(
            float(
                np.dot(xx, np.roll(yy, -1))
                - np.dot(yy, np.roll(xx, -1))
            )
        )

    exterior.sort(
        key=_area, reverse=True
    )

    if return_holes:
        hole_rings.sort(key=_area, reverse=True)
        return exterior, hole_rings
    return exterior


def _levelset_burned_geometry(exteriors, holes):
    """Exact burned Polygon/MultiPolygon: exteriors minus unburned islands."""
    import shapely
    from shapely.geometry import Polygon

    grid = 1e-6  # snap-rounded overlay, as in topology.py, for valid output
    burned = shapely.union_all(
        [Polygon(np.column_stack(r)).buffer(0) for r in exteriors],
        grid_size=grid)
    if holes:
        islands = shapely.union_all(
            [Polygon(np.column_stack(r)).buffer(0) for r in holes],
            grid_size=grid)
        burned = shapely.difference(burned, islands, grid_size=grid)
    return burned


# =============================================================================
# Main solver entry point
# =============================================================================

def simulate_fire_spread_levelset(
    param_grid: FireParameterGrid,
    x_ignition: float | None = None,
    y_ignition: float | None = None,
    dt: float = 1.0,
    initial_radius: float = 0.5,
    store_every: int = 1,
    max_steps: int | None = None,
    min_ros: float = 0.01,
    is_geographic: bool = False,
    center_latitude: float | None = None,
    cfl_factor: float = 0.5,
    reinit_interval: int = 0,
    reinit_iters: int = 10,
    narrow_band_width: int | None = None,
    spotting_config: dict | None = None,
    fuel_grid: np.ndarray | None = None,
    fuel_lookup: dict | None = None,
    ignition_points: Sequence[tuple[float, float]] | None = None,
) -> FirePerimeterHistory:
    """
    Simulate fire spread using the level set method (JAX).

    This function mirrors the interface of ``simulate_fire_spread()`` but uses
    an Eulerian level set approach instead of Lagrangian markers.

    Parameters
    ----------
    param_grid : FireParameterGrid
        Gridded fire behaviour parameters (numpy arrays).
    x_ignition, y_ignition : float, optional
        Legacy single-ignition coordinates.
    ignition_points : sequence of (float, float), optional
        Simultaneous ignition centres in the metric working CRS. Providing
        two or more enables multi-front contour extraction and history.
    dt : float
        Outer time step in minutes.
    initial_radius : float
        Exact initial radius in metres. Not automatically enlarged.
    store_every : int
        Store perimeter contour every N steps.
    max_steps : int, optional
        Maximum number of time steps. Defaults to param_grid.nt.
    min_ros : float
        Minimum ROS threshold (m/min) below which fire is considered extinguished.
    is_geographic : bool
        Must be False. Work in a projected metric CRS (e.g. EPSG:32611).
    center_latitude : float, optional
        Unused legacy argument; geographic operation is disallowed.
    cfl_factor : float
        CFL safety factor for internal sub-stepping.
    reinit_interval : int
        Reinitialize phi every N steps; 0 disables it (verification default).
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

    if dt <= 0 or store_every < 1 or not 0 < cfl_factor <= 1:
        raise ValueError(
            "dt > 0, store_every >= 1 and 0 < cfl_factor <= 1 required"
        )

    if initial_radius <= 0:
        raise ValueError("initial_radius must be positive")

    if reinit_interval < 0:
        raise ValueError(
            "reinit_interval must be >= 0 (0 disables reinitialization)"
        )

    if is_geographic:
        raise ValueError(
            "Level-set Richards solver requires a projected metric working CRS. "
            "Set working_crs (e.g. EPSG:32611) and reproject DEM/fuels/ignitions. "
            "The old mean-metres-per-degree conversion distorts ellipses."
        )

    # Positions and ROS are in metres and metres/minute, respectively.
    initial_radius_coord = float(initial_radius)

    if ignition_points is None:
        if x_ignition is None or y_ignition is None:
            raise ValueError(
                "Supply x_ignition/y_ignition or ignition_points"
            )

        points = np.asarray(
            [(x_ignition, y_ignition)], dtype=float
        )
        multi_mode = False  # preserve the historical one-front output path

    else:
        points = np.asarray(
            ignition_points, dtype=float
        )

        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError(
                "ignition_points must have shape (N, 2), N >= 1"
            )

        multi_mode = len(points) > 1

    if not np.isfinite(points).all():
        raise ValueError(
            "Ignition coordinates must be finite"
        )

    x_ignition, y_ignition = map(
        float, points[0]
    )

    # ---- Grid setup ----
    x_coords = param_grid.x_coords
    y_coords = param_grid.y_coords

    dx = (
        abs(float(x_coords[1] - x_coords[0]))
        if len(x_coords) > 1
        else 1.0
    )

    dy = (
        abs(float(y_coords[1] - y_coords[0]))
        if len(y_coords) > 1
        else 1.0
    )

    # ---- Detect y-axis convention ----
    y_ascending = not (
        len(y_coords) > 1
        and y_coords[0] > y_coords[-1]
    )

    if not y_ascending:
        logger.debug(
            "Detected descending y-axis (raster convention)"
        )

    # =========================================================
    # Pre-cropping of parameter grid
    # =========================================================

    # For large grids, this dramatically reduces memory and per-step slicing cost.
    # Cap at 750 cells in each direction from ignition (~65 km at 90m resolution),
    # which is generous for any single fire event.
    _MAX_BUFFER = 750

    _buf_cells_x = min(
        _MAX_BUFFER, len(x_coords) // 2
    )
    _buf_cells_y = min(
        _MAX_BUFFER, len(y_coords) // 2
    )

    # Locate ALL ignition centres, avoiding truncation by the pre-crop.
    # Retain the original one-ignition crop when multi_mode is false.
    _ix = np.rint(
        (points[:, 0] - float(x_coords[0]))
        / float(x_coords[1] - x_coords[0])
    ).astype(int)

    _iy = np.rint(
        (points[:, 1] - float(y_coords[0]))
        / float(y_coords[1] - y_coords[0])
    ).astype(int)

    _r_cells_x = int(
        np.ceil(initial_radius_coord / dx)
    ) + 2

    _r_cells_y = int(
        np.ceil(initial_radius_coord / dy)
    ) + 2

    if multi_mode:
        _cx_lo = max(
            0, int(_ix.min()) - _buf_cells_x - _r_cells_x
        )
        _cx_hi = min(
            len(x_coords),
            int(_ix.max()) + _buf_cells_x + _r_cells_x + 1
        )
        _cy_lo = max(
            0, int(_iy.min()) - _buf_cells_y - _r_cells_y
        )
        _cy_hi = min(
            len(y_coords),
            int(_iy.max()) + _buf_cells_y + _r_cells_y + 1
        )

    else:
        _ix_ign = max(
            0, min(int(_ix[0]), len(x_coords) - 1)
        )
        _iy_ign = max(
            0, min(int(_iy[0]), len(y_coords) - 1)
        )

        _cx_lo = max(
            0, _ix_ign - _buf_cells_x
        )
        _cx_hi = min(
            len(x_coords),
            _ix_ign + _buf_cells_x
        )
        _cy_lo = max(
            0, _iy_ign - _buf_cells_y
        )
        _cy_hi = min(
            len(y_coords),
            _iy_ign + _buf_cells_y
        )

    # =========================================================
    # H4 DIAGNOSTIC: Full initial computational domain
    # =========================================================

    # Retain the entire original computational domain.
    verification_full_domain = True

    if verification_full_domain:
        _cx_lo = 0
        _cx_hi = len(x_coords)

        _cy_lo = 0
        _cy_hi = len(y_coords)

        logger.info(
            "Verification mode: initial pre-cropping disabled. "
            "Using full %d x %d level-set domain.",
            len(y_coords),
            len(x_coords),
        )

    # =========================================================
    # Apply pre-cropping only if required
    # =========================================================

    if (
        (_cy_hi - _cy_lo < len(y_coords))
        or (_cx_hi - _cx_lo < len(x_coords))
    ):
        logger.info(
            f"Pre-cropping param_grid: {len(y_coords)}x{len(x_coords)} -> "
            f"{_cy_hi-_cy_lo}x{_cx_hi-_cx_lo} "
            f"(buffer: {_buf_cells_y} rows, {_buf_cells_x} cols)"
        )

        x_coords = x_coords[_cx_lo:_cx_hi]
        y_coords = y_coords[_cy_lo:_cy_hi]

        param_grid = FireParameterGrid(
            x_coords=x_coords,
            y_coords=y_coords,
            ros=param_grid.ros[
                :, _cy_lo:_cy_hi, _cx_lo:_cx_hi
            ],
            bros=param_grid.bros[
                :, _cy_lo:_cy_hi, _cx_lo:_cx_hi
            ],
            fros=param_grid.fros[
                :, _cy_lo:_cy_hi, _cx_lo:_cx_hi
            ],
            raz=param_grid.raz[
                :, _cy_lo:_cy_hi, _cx_lo:_cx_hi
            ],
        )

        # Also crop fuel_grid if provided
        if fuel_grid is not None:
            fuel_grid = fuel_grid[
                _cy_lo:_cy_hi,
                _cx_lo:_cx_hi,
            ]

    # ---- Initialize phi at the exact configured physical radius ----

    # Do not artificially enlarge a 10-m ignition to multiple coarse cells.
    if initial_radius_coord < 2.5 * max(dx, dy):
        logger.warning(
            "Initial radius %.2f m is under-resolved (dx=%.2f, dy=%.2f m). "
            "Refine the input grid for a valid numerical verification test.",
            initial_radius_coord, dx, dy,
        )

    x_jax = jnp.asarray(
        x_coords, dtype=jnp.float64
    )
    y_jax = jnp.asarray(
        y_coords, dtype=jnp.float64
    )

    if multi_mode:
        # Check each circle's radius is resolved by at least one grid node;
        # a different ignition must never mask an under-resolved input.
        xx = np.asarray(x_coords, dtype=float)
        yy = np.asarray(y_coords, dtype=float)

        for i, (xc, yc) in enumerate(points, 1):
            if not (
                xx.min() + initial_radius_coord
                < xc
                < xx.max() - initial_radius_coord
                and yy.min() + initial_radius_coord
                < yc
                < yy.max() - initial_radius_coord
            ):
                raise ValueError(
                    f"Ignition {i} is too close to the cropped grid edge"
                )

            px = xx[
                np.argmin(np.abs(xx - xc))
            ]
            py = yy[
                np.argmin(np.abs(yy - yc))
            ]

            if np.hypot(px - xc, py - yc) >= initial_radius_coord:
                raise ValueError(
                    f"Ignition {i} has no burned grid node; refine grid"
                )

        phi = initialize_phi_multiple(
            x_jax, y_jax,
            points, initial_radius_coord
        )

    else:
        phi = initialize_phi(
            x_jax,
            y_jax,
            x_ignition,
            y_ignition,
            initial_radius_coord,
        )

    if not bool(jnp.any(phi < 0)):
        raise ValueError(
            f"Ignition radius {initial_radius_coord:g} m is smaller than the "
            "grid can resolve at this ignition location. The phi field has "
            "no burned grid node. Refine DEM/fuels and rerun; do not silently "
            "increase radius for a verification experiment."
        )

    # ---- Time loop ----
    if max_steps is None:
        max_steps = param_grid.nt

    n_steps = min(
        max_steps, param_grid.nt
    )

    # Extract initial contour
    phi_np_out = np.asarray(phi)

    if multi_mode:
        initial_fronts, initial_holes = extract_all_contours(
            phi_np_out, x_coords, y_coords, return_holes=True
        )

        if not initial_fronts:
            raise ValueError(
                "No initial phi=0 contours found for multiple ignitions"
            )

        # For disjoint initial circles, contour count must equal ignition count.
        initially_disjoint = all(
            np.hypot(
                *(points[i] - points[j])
            ) > (
                2.0 * initial_radius_coord
                + 2.0 * max(dx, dy)
            )
            for i in range(len(points))
            for j in range(i + 1, len(points))
        )

        if initially_disjoint and len(initial_fronts) != len(points):
            raise ValueError(
                f"Expected {len(points)} initial fire fronts; "
                f"found {len(initial_fronts)}. "
                "Check grid resolution, crop boundaries, and ignition coordinates."
            )

        history = FirePerimeterHistory(
            perimeters=[
                initial_fronts[0]
                if len(initial_fronts) == 1
                else None
            ],
            times=[0.0],
            fronts_by_time=[
                [
                    (x.copy(), y.copy())
                    for x, y in initial_fronts
                ]
            ],
            holes_by_time=[
                [(x.copy(), y.copy()) for x, y in initial_holes]
            ],
            burned_geometries=[
                _levelset_burned_geometry(initial_fronts, initial_holes)
            ],
        )

        previous_front_count = len(initial_fronts)

        logger.info(
            "Level-set initialized %d simultaneous ignitions "
            "(%d zero contours)",
            len(points),
            previous_front_count,
        )

    else:
        initial_contour = extract_contour(
            phi_np_out,
            x_coords,
            y_coords,
            x_ignition,
            y_ignition,
        )

        if initial_contour is None:
            raise ValueError(
                "Initial phi=0 contour cannot be extracted on this grid. "
                "Check that the ignition is inside the domain and sufficiently "
                "resolved; do not substitute an analytical curve for a missing "
                "numerical contour."
            )

        history = FirePerimeterHistory(
            perimeters=[initial_contour],
            times=[0.0],
        )

    # Non-fuel barrier strength
    nonfuel_phi = 10.0 * max(dx, dy)

    # Cumulative burned mask is used for crop selection and spotting only.
    # The output perimeter comes directly from phi=0, never this binary mask.
    _cumulative_burned = np.array(
        phi < 0, copy=True
    )

    # Static non-fuel mask: cells that NEVER have ROS > 0 across all timesteps.
    # This prevents time-varying weather (ROS temporarily 0) from erasing burned areas.
    # Compute memory-efficiently by iterating over timesteps instead of nanmax.
    ny_pg, nx_pg = (
        param_grid.ros.shape[1],
        param_grid.ros.shape[2],
    )

    _has_fuel = np.zeros(
        (ny_pg, nx_pg), dtype=bool
    )

    # Sample every 24th timestep (daily) — sufficient to identify permanent non-fuel
    for _t in range(
        0,
        param_grid.nt,
        max(1, param_grid.nt // 70),
    ):
        _ros_t = param_grid.ros[_t]
        _has_fuel |= (
            np.nan_to_num(_ros_t, nan=0.0) > 0
        )

    _static_nonfuel = ~_has_fuel

    logger.debug(
        f"Static non-fuel mask: "
        f"{np.sum(_static_nonfuel)} / {_static_nonfuel.size} cells "
        f"({np.mean(_static_nonfuel)*100:.1f}%)"
    )

    # Spotting setup
    _do_spotting = (
        spotting_config is not None
        and fuel_grid is not None
        and fuel_lookup is not None
    )

    if _do_spotting:
        _spot_cfg = spotting_config
        _spot_interval = int(
            _spot_cfg.get("interval", 1)
        )
        _spot_rng = np.random.default_rng(
            _spot_cfg.get("seed", 42)
        )

        logger.info(
            f"Spotting enabled: interval={_spot_interval} steps"
        )

    else:
        _spot_cfg = {}
        _spot_interval = 1
        _spot_rng = None

    d_min = min(dx, dy)
    ny_full, nx_full = (
        len(y_coords), len(x_coords)
    )

    # Use float32 for computation (sufficient for fire spread, 2× faster)
    phi = phi.astype(jnp.float32)

    # JIT-compiled evolution function for a fixed number of sub-steps.
    # Cached by (dx, dy, y_ascending, n_sub); shape changes trigger recompilation
    # but we pad crop dimensions to multiples of _PAD to limit this.
    _PAD = 64

    @partial(
        jax.jit,
        static_argnames=(
            "dx_", "dy_", "y_ascending_", "n_sub_"
        )
    )
    def _evolve_step(
        phi_,
        ros_,
        bros_,
        fros_,
        raz_,
        nonfuel_,
        nonfuel_phi_,
        dt_sub_,
        dx_,
        dy_,
        y_ascending_,
        n_sub_,
    ):
        """Monotone local Lax-Friedrichs step for the Richards Hamiltonian."""

        _dt = jnp.asarray(
            dt_sub_, dtype=phi_.dtype
        )
        _nfp = jnp.asarray(
            nonfuel_phi_, dtype=phi_.dtype
        )

        def _body(_, current):
            h = _numerical_hamiltonian(
                current,
                ros_,
                bros_,
                fros_,
                raz_,
                dx_,
                dy_,
                y_ascending_,
            )

            updated = current - _dt * h

            return jnp.where(
                nonfuel_,
                jnp.maximum(updated, _nfp),
                updated,
            )

        return lax.fori_loop(
            0, n_sub_, _body, phi_
        )

    def _pad_dim(n, maximum):
        """Pad a dimension without ever exceeding that dimension's grid size."""
        return min(
            ((n + _PAD - 1) // _PAD) * _PAD,
            maximum,
        )

    logger.info(
        f"Level set simulation: {n_steps} steps, dt={dt:.1f} min, "
        f"grid {ny_full}x{nx_full}, dx={dx:.4f}, dy={dy:.4f} "
        "(float32, bbox crop)"
    )

    # Bounding box state — include EVERY ignition from the first outer step.
    _BUFFER_BASE = 20

    bbox_r0 = max(
        int(initial_radius_coord / d_min) + _BUFFER_BASE,
        50,
    )

    ix_centres = np.rint(
        (points[:, 0] - float(x_coords[0]))
        / float(x_coords[1] - x_coords[0])
    ).astype(int)

    iy_centres = np.rint(
        (points[:, 1] - float(y_coords[0]))
        / float(y_coords[1] - y_coords[0])
    ).astype(int)

    ix_min, ix_max = (
        int(ix_centres.min()),
        int(ix_centres.max()),
    )

    iy_min, iy_max = (
        int(iy_centres.min()),
        int(iy_centres.max()),
    )

    r0_lo = max(0, iy_min - bbox_r0)
    r0_hi = min(ny_full, iy_max + bbox_r0 + 1)

    c0_lo = max(0, ix_min - bbox_r0)
    c0_hi = min(nx_full, ix_max + bbox_r0 + 1)

    # Pad to multiples while keeping the entire multi-ignition bbox inside.
    crop_ny = _pad_dim(
        r0_hi - r0_lo, ny_full
    )
    crop_nx = _pad_dim(
        c0_hi - c0_lo, nx_full
    )

    center_y = (iy_min + iy_max) // 2
    center_x = (ix_min + ix_max) // 2

    r0_lo = int(np.clip(
        center_y - crop_ny // 2,
        0,
        ny_full - crop_ny,
    ))
    r0_hi = r0_lo + crop_ny

    c0_lo = int(np.clip(
        center_x - crop_nx // 2,
        0,
        nx_full - crop_nx,
    ))
    c0_hi = c0_lo + crop_nx

    _compiled_nsubs = set()
    _bbox_update_interval = 10
    _last_crop_shape = None

    for step in range(n_steps):

        t_idx = min(
            step, param_grid.nt - 1
        )

        # Check for extinguishment (only check within bbox for speed)
        ros_crop = param_grid.ros[
            t_idx,
            r0_lo:r0_hi,
            c0_lo:c0_hi,
        ]

        if np.all(
            np.nan_to_num(
                ros_crop, nan=0.0
            ) < min_ros
        ):
            logger.info(
                f"Fire extinguished at step {step} "
                "(all ROS below threshold in active region)"
            )
            break

        # Extract current timestep slices (cropped to bbox)
        ros_np_c = np.nan_to_num(
            param_grid.ros[
                t_idx, r0_lo:r0_hi, c0_lo:c0_hi
            ],
            nan=0.0,
        ).astype(np.float32)

        bros_np_c = np.nan_to_num(
            param_grid.bros[
                t_idx, r0_lo:r0_hi, c0_lo:c0_hi
            ],
            nan=0.0,
        ).astype(np.float32)

        fros_np_c = np.nan_to_num(
            param_grid.fros[
                t_idx, r0_lo:r0_hi, c0_lo:c0_hi
            ],
            nan=0.0,
        ).astype(np.float32)

        raz_np_c = np.nan_to_num(
            param_grid.raz[
                t_idx, r0_lo:r0_hi, c0_lo:c0_hi
            ],
            nan=0.0,
        ).astype(np.float32)

        if np.max(np.abs(raz_np_c)) > 2.0 * np.pi + 1e-3:
            raise ValueError(
                "param_grid.raz must be in radians "
                "(geographic azimuth from north), "
                "not degrees. Check the FBP-to-level-set "
                "parameter conversion."
            )

        # Use static non-fuel mask (cropped) — cells permanently without fuel.
        # NOT derived from current-timestep ROS, to avoid erasing burned areas
        # when ROS temporarily drops to 0 (nighttime, weather data gaps, etc.).
        nonfuel_np_c = _static_nonfuel[
            r0_lo:r0_hi,
            c0_lo:c0_hi,
        ]

        # LOCAL CFL: sum of x/y characteristic speeds is required for LLF.
        cfl_rate = _cfl_rate_numpy(
            ros_np_c,
            bros_np_c,
            fros_np_c,
            raz_np_c,
            dx,
            dy,
        )

        n_sub = max(
            1,
            int(np.ceil(cfl_rate * dt / cfl_factor)),
        )

        if n_sub > 500:
            raise ValueError(
                f"Level-set step requires {n_sub} CFL substeps (>500). "
                "Reduce the outer dt or increase spatial resolution."
            )

        if n_sub > 10:
            _QUANTA = [20, 50, 100, 200, 500]
            n_sub = next(
                q for q in _QUANTA
                if q >= n_sub
            )

        dt_sub = dt / n_sub
        F_max_local = float(
            np.max(ros_np_c)
        )

        # Extract phi crop
        phi_crop = phi[
            r0_lo:r0_hi,
            c0_lo:c0_hi,
        ]

        # Convert to JAX float32
        ros_j = jnp.asarray(ros_np_c)
        bros_j = jnp.asarray(bros_np_c)
        fros_j = jnp.asarray(fros_np_c)
        raz_j = jnp.asarray(raz_np_c)
        nonfuel_j = jnp.asarray(nonfuel_np_c)

        crop_shape = (
            r0_hi - r0_lo,
            c0_hi - c0_lo,
        )

        if crop_shape != _last_crop_shape:
            logger.info(
                f"  Crop region: {crop_shape[0]}x{crop_shape[1]} "
                f"(rows {r0_lo}-{r0_hi}, cols {c0_lo}-{c0_hi})"
            )
            _last_crop_shape = crop_shape

        if n_sub not in _compiled_nsubs:
            logger.info(
                f"  Compiling JIT kernel for n_sub={n_sub} "
                f"on {crop_shape[0]}x{crop_shape[1]} crop"
            )
            _compiled_nsubs.add(n_sub)

        # Evolve φ on cropped region
        phi_crop = _evolve_step(
            phi_crop,
            ros_j,
            bros_j,
            fros_j,
            raz_j,
            nonfuel_j,
            jnp.float32(nonfuel_phi),
            jnp.float32(dt_sub),
            dx_=dx,
            dy_=dy,
            y_ascending_=y_ascending,
            n_sub_=n_sub,
        )

        # Write cropped result back to full phi
        phi = phi.at[
            r0_lo:r0_hi,
            c0_lo:c0_hi,
        ].set(phi_crop)

        # Reinitialize φ periodically (on crop only)
        if (
            reinit_interval > 0
            and (step + 1) % reinit_interval == 0
        ):
            phi_crop = phi[
                r0_lo:r0_hi,
                c0_lo:c0_hi,
            ]

            phi_crop = reinitialize_phi(
                phi_crop,
                dx,
                dy,
                n_iter=reinit_iters,
                y_ascending=y_ascending,
            )

            phi_crop = jnp.where(
                nonfuel_j,
                jnp.maximum(
                    phi_crop,
                    jnp.float32(nonfuel_phi),
                ),
                phi_crop,
            )

            phi = phi.at[
                r0_lo:r0_hi,
                c0_lo:c0_hi,
            ].set(phi_crop)

        # Stochastic spotting: create new ignitions ahead of the fire front
        if (
            _do_spotting
            and (step + 1) % _spot_interval == 0
        ):
            from ignacio.spotting import (
                compute_spot_ignitions,
                apply_spot_ignitions,
            )

            phi_np_spot = np.asarray(phi)
            burned_now = (
                _cumulative_burned
                | (phi_np_spot < 0)
            )

            ros_np_spot = np.nan_to_num(
                param_grid.ros[t_idx], nan=0.0
            )
            raz_np_spot = np.nan_to_num(
                param_grid.raz[t_idx], nan=0.0
            )

            # Use per-timestep ISI/WS if available, else fallback to static
            _isi_arr = _spot_cfg.get(
                "isi_array", None
            )
            _ws_arr = _spot_cfg.get(
                "ws_array", None
            )

            _isi_t = (
                float(_isi_arr[t_idx])
                if _isi_arr is not None
                and t_idx < len(_isi_arr)
                else _spot_cfg.get("isi", 5.0)
            )

            _ws_t = (
                float(_ws_arr[t_idx])
                if _ws_arr is not None
                and t_idx < len(_ws_arr)
                else _spot_cfg.get("wind_speed", 15.0)
            )

            spot_mask = compute_spot_ignitions(
                burned_mask=burned_now,
                phi=phi_np_spot,
                ros=ros_np_spot,
                raz=raz_np_spot,
                fuel_grid=fuel_grid,
                fuel_lookup=fuel_lookup,
                wind_speed=_ws_t,
                isi=_isi_t,
                dx=dx,
                dy=dy,
                rng=_spot_rng,
                min_spot_distance=_spot_cfg.get(
                    "min_spot_distance", 500.0
                ),
                max_spot_distance=_spot_cfg.get(
                    "max_spot_distance", 2000.0
                ),
                isi_threshold=_spot_cfg.get(
                    "isi_threshold", 8.0
                ),
                spot_probability=_spot_cfg.get(
                    "spot_probability", 0.15
                ),
                meters_per_unit=1.0,
                y_ascending=y_ascending,
            )

            if np.any(spot_mask):
                n_spots = int(
                    np.sum(spot_mask)
                )

                logger.info(
                    f"  Step {step+1}: {n_spots} spot fires ignited "
                    f"(ISI={_isi_t:.1f}, WS={_ws_t:.0f} km/h)"
                )

                phi = apply_spot_ignitions(
                    phi,
                    spot_mask,
                    spot_radius_cells=2,
                )

        # Update cumulative burned mask (monotonically grows)
        if (
            (step + 1) % _bbox_update_interval == 0
            or (step + 1) % store_every == 0
        ):
            phi_np_full = np.asarray(phi)

            _cumulative_burned |= (
                phi_np_full < 0
            )

        # Update bounding box periodically to follow the fire
        if (step + 1) % _bbox_update_interval == 0:
            burned = _cumulative_burned

            if np.any(burned):
                rows_burned, cols_burned = np.nonzero(
                    burned
                )

                # Fire extent + buffer (enough for growth in bbox_update_interval steps)
                growth_cells = max(
                    int(
                        F_max_local
                        * dt
                        * _bbox_update_interval
                        / d_min
                    ) + 10,
                    _BUFFER_BASE,
                )

                new_r_lo = max(
                    0,
                    int(rows_burned.min()) - growth_cells,
                )

                new_r_hi = min(
                    ny_full,
                    int(rows_burned.max()) + 1 + growth_cells,
                )

                new_c_lo = max(
                    0,
                    int(cols_burned.min()) - growth_cells,
                )

                new_c_hi = min(
                    nx_full,
                    int(cols_burned.max()) + 1 + growth_cells,
                )

                # Pad to multiples for JIT stability
                new_ny = _pad_dim(
                    new_r_hi - new_r_lo,
                    ny_full,
                )
                new_nx = _pad_dim(
                    new_c_hi - new_c_lo,
                    nx_full,
                )

                # Center the padded region on the fire
                r_center = (
                    new_r_lo + new_r_hi
                ) // 2

                c_center = (
                    new_c_lo + new_c_hi
                ) // 2

                r0_lo = max(
                    0,
                    r_center - new_ny // 2,
                )
                r0_hi = min(
                    ny_full,
                    r0_lo + new_ny,
                )

                if r0_hi - r0_lo < new_ny:
                    r0_lo = max(
                        0, r0_hi - new_ny
                    )

                c0_lo = max(
                    0,
                    c_center - new_nx // 2,
                )
                c0_hi = min(
                    nx_full,
                    c0_lo + new_nx,
                )

                if c0_hi - c0_lo < new_nx:
                    c0_lo = max(
                        0, c0_hi - new_nx
                    )

        # Store all disconnected exterior fronts for multi-ignition runs.
        if (step + 1) % store_every == 0:

            if multi_mode:
                fronts, holes = extract_all_contours(
                    np.asarray(phi),
                    x_coords,
                    y_coords,
                    return_holes=True,
                )

                if not fronts:
                    raise ValueError(
                        "No closed exterior zero contours at "
                        f"t={(step+1)*dt:.4f} min"
                    )

                history.fronts_by_time.append(
                    [
                        (x.copy(), y.copy())
                        for x, y in fronts
                    ]
                )

                history.holes_by_time.append(
                    [(x.copy(), y.copy()) for x, y in holes]
                )
                history.burned_geometries.append(
                    _levelset_burned_geometry(fronts, holes)
                )

                history.perimeters.append(
                    fronts[0]
                    if len(fronts) == 1
                    else None
                )

                history.times.append(
                    float((step + 1) * dt)
                )

                if len(fronts) < previous_front_count:
                    # This is first observed at a saved snapshot, NOT an
                    # exact substep contact time like the marker diagnostic.
                    history.merger_events.append({
                        "time_min": float(
                            (step + 1) * dt
                        ),
                        "fronts_before": previous_front_count,
                        "fronts_after": len(fronts),
                        "event_type": "first_saved_snapshot",
                    })

                    logger.info(
                        "Level-set fronts reduced from %d to %d "
                        "by t=%.4f min",
                        previous_front_count,
                        len(fronts),
                        (step + 1) * dt,
                    )

                previous_front_count = len(fronts)

            else:
                contour = extract_contour(
                    np.asarray(phi),
                    x_coords,
                    y_coords,
                    x_ignition,
                    y_ignition,
                )

                if contour is not None:
                    history.perimeters.append(
                        contour
                    )
                    history.times.append(
                        (step + 1) * dt
                    )

        # Log progress
        log_interval = (
            10 if step < 50 else 100
        )

        if (step + 1) % log_interval == 0:
            burned_frac = float(
                jnp.mean(phi < 0)
            )

            logger.info(
                f"  Step {step+1}/{n_steps}: "
                f"{burned_frac*100:.2f}% burned, "
                f"crop {r0_hi-r0_lo}x{c0_hi-c0_lo}, "
                f"n_sub={n_sub}"
            )

    logger.info(
        "Level-set simulation complete: %d stored snapshots, %d ignition(s)",
        len(history.times),
        len(points),
    )

    return history
