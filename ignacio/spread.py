"""
Fire spread equations for Ignacio.

This module implements Richards' differential equations for elliptical fire
spread using Huygens' wave propagation principle, along with the marker
method for handling perimeter topology. Multi-ignition runs may use either the
legacy polygon-union routine or an experimental, oriented-linework outer-boundary
reconstruction which preserves genuine unburned islands. The reconstruction
is delegated to ignacio.topology, which implements an oriented planar
arrangement inspired by Barber et al. (2009), NOT their exact edge-reversal
algorithm.
Legacy single-ignition numerical routines remain available.

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
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Use the shared, tested topology implementation. Both package imports and
# direct module imports remain supported for older IGNACIO workflows.
# Do not import spread.py from topology.py (this module calls topology only).
if __package__:
    from . import topology as _marker_topology
else:
    try:
        from ignacio import topology as _marker_topology
    except ModuleNotFoundError as exc:
        if exc.name not in ("ignacio", "ignacio.topology"):
            raise
        import topology as _marker_topology


logger = logging.getLogger(__name__)

# =============================================================================
# Marker maintenance + higher-order advection helpers (Heun/RK4)
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
    barrier_geometry=None,
    barrier_interior=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Maintain marker spacing on a closed ring by inserting/deleting vertices.

    With ``barrier_geometry`` and ``barrier_interior``, a new midpoint that
    would fall strictly inside non-fuel moves to the nearest barrier
    boundary point. If that point is more than half the edge length away,
    the midpoint is skipped instead.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 4 or spacing <= 0:
        return x, y

    # ---- Insert pass ----
    if n < max_n:
        seg = _segment_lengths_xy(x, y)
        split = seg > insert_factor * spacing
        mid_x = 0.5 * (x + np.roll(x, -1))
        mid_y = 0.5 * (y + np.roll(y, -1))
        use_mid = split.copy()
        if (barrier_geometry is not None and barrier_interior is not None
                and not barrier_interior.is_empty and split.any()):
            import shapely
            from shapely.geometry import Point
            from shapely.ops import nearest_points

            shapely.prepare(barrier_interior)
            bad = split & shapely.contains_xy(barrier_interior, mid_x, mid_y)
            if bad.any():
                boundary = barrier_geometry.boundary
                for i in np.flatnonzero(bad):
                    nearest = nearest_points(boundary, Point(mid_x[i], mid_y[i]))[0]
                    if np.hypot(nearest.x - mid_x[i], nearest.y - mid_y[i]) <= 0.5 * seg[i]:
                        mid_x[i], mid_y[i] = nearest.x, nearest.y
                    else:
                        use_mid[i] = False
        new_x: list[float] = []
        new_y: list[float] = []
        for i in range(n):
            new_x.append(float(x[i]))
            new_y.append(float(y[i]))
            if use_mid[i]:
                new_x.append(float(mid_x[i]))
                new_y.append(float(mid_y[i]))
        x = np.asarray(new_x, dtype=float)
        y = np.asarray(new_y, dtype=float)

    # ---- Delete pass ----
    # Never delete two neighbouring markers in one pass. `seg` is computed
    # once, so deleting a run of neighbours used to collapse a densely
    # spaced ring onto one short stretch of `min_n` markers (a 6,362 m2
    # circle at 0.19 m spacing became 12 m2). Skipping neighbours keeps each
    # merged edge below 2 * delete_factor * spacing; dense rings thin out
    # over successive passes instead.
    n = x.size
    if n > min_n:
        seg = _segment_lengths_xy(x, y)
        keep = np.ones(n, dtype=bool)
        n_keep = n
        for i in range(n):
            if n_keep <= min_n:
                break
            prev_i, next_i = (i - 1) % n, (i + 1) % n
            if not (keep[prev_i] and keep[next_i]):
                continue
            if seg[prev_i] < delete_factor * spacing and seg[i] < delete_factor * spacing:
                keep[i] = False
                n_keep -= 1
        x = x[keep]
        y = y[keep]

    return x, y


def _limit_step_to_burnable(
    param_grid: "FireParameterGrid",
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    sample_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Shorten each proposed marker move so it stops before non-fuel.

    The straight path (x0, y0) -> (x1, y1) of every marker is sampled at
    ``sample_frac`` of a grid cell, so a step cannot jump over a one-cell
    barrier. Each marker is moved to the last burnable sample on its path.
    Markers that start in non-fuel, or whose proposal is non-finite, stay
    where they are. Requires one-to-one marker correspondence.
    """
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    if param_grid.nonfuel_mask is None or x1.size == 0:
        return x1, y1
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    dxs, dys = x1 - x0, y1 - y0
    step = np.hypot(dxs, dys)
    finite = np.isfinite(step)
    if not finite.any():
        return x0.copy(), y0.copy()
    cell = min(abs(float(param_grid.dx)), abs(float(param_grid.dy)))
    n = max(1, int(np.ceil(float(np.max(step[finite])) / (sample_frac * cell))))
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    xs = x0[:, None] + s[None, :] * dxs[:, None]
    ys = y0[:, None] + s[None, :] * dys[:, None]
    blocked = ~param_grid.is_burnable_at(xs, ys)
    hit = blocked.any(axis=1)
    first = blocked.argmax(axis=1)
    frac = np.where(first > 0, s[np.maximum(first - 1, 0)], 0.0)
    frac = np.where(hit, frac, 1.0)
    frac = np.where(finite, frac, 0.0)
    dxs = np.where(finite, dxs, 0.0)
    dys = np.where(finite, dys, 0.0)
    return x0 + frac * dxs, y0 + frac * dys


def _limit_step_to_burnable_geometry(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    barrier_interior,
    *,
    clearance_m: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Stop individual marker paths at their first obstacle-interior encounter.

    Unlike a raster sample, this intersects the complete straight movement
    segment with the polygonized non-fuel mask. It is enabled only with
    ``barrier_mode='raise'`` and does not repair edges joining adjacent markers.
    Those edges are checked separately, and unresolved crossings fail fast.

    The first interior encounter is backed off slightly so that boundary
    contact does not create a floating-point penetration. This CPU/Shapely
    operation is deliberately separate from JAX advection.
    """
    from shapely.geometry import LineString, Point

    if barrier_interior is None or barrier_interior.is_empty:
        return np.asarray(x1, dtype=float), np.asarray(y1, dtype=float)

    old_x, old_y, new_x, new_y = np.broadcast_arrays(
        np.asarray(x0, dtype=float), np.asarray(y0, dtype=float),
        np.asarray(x1, dtype=float), np.asarray(y1, dtype=float),
    )
    result_x, result_y = new_x.copy(), new_y.copy()

    for index in np.ndindex(old_x.shape):
        start = (float(old_x[index]), float(old_y[index]))
        end = (float(new_x[index]), float(new_y[index]))
        if not np.isfinite((*start, *end)).all():
            result_x[index], result_y[index] = start
            continue
        if start == end:
            continue

        path = LineString((start, end))
        if not path.intersects(barrier_interior):
            continue
        intrusion = path.intersection(barrier_interior)
        # A single point of tangency is not movement through an obstacle.
        if intrusion.is_empty or intrusion.length <= 1e-10:
            continue

        # Intersection of a line and polygons is linear linework (possibly
        # mixed with isolated points). Use all line endpoints so ordering of
        # MultiLineStrings and ring direction cannot alter the first entry.
        entries = []
        stack = [intrusion]
        while stack:
            part = stack.pop()
            if part.is_empty:
                continue
            if part.geom_type in ('LineString', 'LinearRing'):
                for xy in (part.coords[0], part.coords[-1]):
                    entries.append(path.project(Point(xy)))
            elif hasattr(part, 'geoms'):
                stack.extend(part.geoms)

        if not entries:
            raise MarkerBarrierError(
                'A marker path intersects obstacle interior but entry '
                'could not be reconstructed geometrically'
            )
        dist = max(0.0, min(entries) - float(clearance_m))
        clipped = path.interpolate(dist)
        result_x[index], result_y[index] = clipped.x, clipped.y

    return result_x, result_y


def _apply_marker_barrier_limit(
    param_grid: "FireParameterGrid",
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    *,
    barrier_mode: str,
    barrier_interior,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve legacy raster limiting unless strict barriers are enabled."""
    if barrier_mode == 'raise' and barrier_interior is not None:
        return _limit_step_to_burnable_geometry(
            x0, y0, x1, y1, barrier_interior,
        )
    return _limit_step_to_burnable(param_grid, x0, y0, x1, y1)


# -----------------------------------------------------------------------------
# Perimeter-edge barrier repair
#
# The limiters above act on each marker's own path. Two neighbouring markers
# can both sit on fuel while the straight edge joining them (the perimeter)
# cuts across a non-fuel cell. The helpers below replace every such chord
# with a detour along the barrier boundary, so neither a vertex nor an edge
# of the marker ring lies inside non-fuel. Marker count changes, so call
# them only where one-to-one marker identity is no longer needed.
# -----------------------------------------------------------------------------

def _linear_parts(geometry):
    """Flatten line-like parts of an intersection result."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [geometry]
    if hasattr(geometry, "geoms"):
        parts = []
        for item in geometry.geoms:
            parts.extend(_linear_parts(item))
        return parts
    return []


def _boundary_path(line, d0: float, d1: float) -> list:
    """Vertices of a closed boundary from distance d0 to d1 in ring direction.

    ``line`` is the boundary as a closed LineString. The path includes both
    end points and every boundary corner between them, wrapping past the
    ring's start point when ``d1 < d0``.
    """
    from shapely.ops import substring

    length = float(line.length)

    def _coords(a: float, b: float) -> list:
        if b <= a:
            return []
        piece = substring(line, a, b)
        return list(piece.coords)

    if d1 >= d0:
        return _coords(d0, d1)
    first = _coords(d0, length)
    second = _coords(0.0, d1)
    if first and second:
        second = second[1:]  # the ring start point is shared
    return first + second


def _left_boundary_detour(line, a, b) -> list | None:
    """Boundary path from a to b lying to the LEFT of the chord a -> b.

    Burned land lies to the left of every directed marker edge (CCW
    exteriors, CW holes). The barrier piece on the left of a crossing chord
    is therefore wrongly inside the burned polygon; walking around that
    piece's boundary instead puts it outside. A path to the left of the chord
    closes (with b -> a) into a clockwise, negative-area polygon.

    Only the SHORTER boundary path is accepted. Marker edges are at most a
    few metres long, so a genuine crossing clips a barrier corner and the
    piece to cut away is the small one. When the left path is the long way
    round (for example a long edge of a thin burned strip grazing a corner),
    the "left" piece is not locally inside the burned area; walking around
    it would turn the ring inside out. Return None and leave that edge to
    the topology barrier clip.
    """
    da, db = float(line.project(a)), float(line.project(b))
    forward = _boundary_path(line, da, db)
    backward = _boundary_path(line, db, da)[::-1]
    candidates = []
    for path in (forward, backward):
        if len(path) < 2:
            continue
        xy = np.asarray(path, dtype=float)
        length = float(np.hypot(*np.diff(xy, axis=0).T).sum())
        candidates.append((length, _ring_signed_area(xy[:, 0], xy[:, 1]), path))
    if not candidates:
        return None
    shortest = min(length for length, _, _ in candidates)
    left = [c for c in candidates if c[1] < 0.0]
    if not left:
        return None
    length, _, path = min(left, key=lambda c: c[1])
    if length > shortest * (1.0 + 1e-9) + 1e-9:
        return None
    return path


def _edge_barrier_detour(p0, p1, barrier_geometry, barrier_interior,
                         barrier_polygons, tol: float) -> list | None:
    """Points to insert between p0 and p1 so the edge avoids non-fuel.

    Returns None when a crossing cannot be resolved along one boundary ring
    (for example an end point that is itself inside non-fuel); the caller
    then keeps the chord and the topology barrier clip handles it.
    """
    from shapely.geometry import LineString, Point

    segment = LineString([p0, p1])
    pieces = []
    for piece in _linear_parts(segment.intersection(barrier_geometry)):
        # Pieces lying along a barrier face only touch it; skip them.
        if piece.length <= tol or not piece.intersects(barrier_interior):
            continue
        ends = sorted((Point(xy) for xy in (piece.coords[0], piece.coords[-1])),
                      key=segment.project)
        pieces.append((segment.project(ends[0]), ends[0], ends[1]))
    if not pieces:
        return None

    detour = []
    for _, a, b in sorted(pieces, key=lambda item: item[0]):
        boundary = None
        for polygon in barrier_polygons:
            for ring in (polygon.exterior, *polygon.interiors):
                if ring.distance(a) < tol and ring.distance(b) < tol:
                    boundary = LineString(ring.coords)
                    break
            if boundary is not None:
                break
        if boundary is None:
            return None
        path = _left_boundary_detour(boundary, a, b)
        if path is None:
            return None
        detour.extend(path)
    return detour


def _wrap_ring_around_barriers(x, y, barrier_geometry, barrier_interior,
                               *, tol: float = 1e-6):
    """Return a marker ring with no vertex or edge inside non-fuel.

    1. A vertex strictly inside a barrier moves to the nearest barrier
       boundary point.
    2. Every edge crossing a barrier interior gains markers along the
       barrier boundary (entry point, the barrier corners on the burned
       side, exit point), so the ring wraps the barrier instead of cutting
       through it.

    Returns ``(x, y, stats)``. Marker count and identity change.
    """
    import shapely
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    stats = {"moved_vertices": 0, "wrapped_edges": 0, "unresolved_edges": 0}
    if (barrier_geometry is None or barrier_interior is None
            or barrier_interior.is_empty or x.size < 3
            or not (np.isfinite(x).all() and np.isfinite(y).all())):
        return x, y, stats

    shapely.prepare(barrier_interior)
    inside = shapely.contains_xy(barrier_interior, x, y)
    if inside.any():
        boundary = barrier_geometry.boundary
        for i in np.flatnonzero(inside):
            nearest = nearest_points(boundary, Point(x[i], y[i]))[0]
            x[i], y[i] = nearest.x, nearest.y
        stats["moved_vertices"] = int(inside.sum())

    p = np.column_stack((x, y))
    q = np.roll(p, -1, axis=0)
    edges = shapely.linestrings(np.stack((p, q), axis=1))
    hits = shapely.intersects(edges, barrier_interior)
    if not hits.any():
        return x, y, stats

    polygons = _geometry_polygons(barrier_geometry)
    out = []
    for i in range(len(p)):
        out.append(p[i])
        if not hits[i]:
            continue
        detour = _edge_barrier_detour(p[i], q[i], barrier_geometry,
                                      barrier_interior, polygons, tol)
        if detour is None:
            stats["unresolved_edges"] += 1
            continue
        stats["wrapped_edges"] += 1
        out.extend(np.asarray(detour, dtype=float))

    xy = np.asarray(out, dtype=float)
    # Drop consecutive duplicates (an end point already on the boundary).
    step = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    xy = xy[step > tol]

    # Safety net: a corner detour removes only a small sliver. If the ring
    # flips orientation or its area changes by more than half, the local
    # rule was wrong for this geometry; keep the original ring and let the
    # topology barrier clip handle it.
    area_before = _ring_signed_area(x, y)
    area_after = (_ring_signed_area(xy[:, 0], xy[:, 1])
                  if len(xy) >= 3 else 0.0)
    if (len(xy) < 3 or np.sign(area_after) != np.sign(area_before)
            or abs(area_after - area_before) > 0.5 * abs(area_before)):
        logger.debug("Barrier edge repair reverted: ring area %.4f -> %.4f m2",
                     area_before, area_after)
        stats["unresolved_edges"] += stats["wrapped_edges"]
        stats["wrapped_edges"] = 0
        return x, y, stats

    if stats["unresolved_edges"]:
        logger.debug("Barrier edge repair left %d crossing edge(s) for "
                     "topology clipping", stats["unresolved_edges"])
    return xy[:, 0], xy[:, 1], stats


def _wrap_rings_around_barriers(rings, barrier_geometry, barrier_interior):
    """Apply :func:`_wrap_ring_around_barriers` to every ring."""
    wrapped = []
    totals = {"moved_vertices": 0, "wrapped_edges": 0, "unresolved_edges": 0}
    for x, y in rings:
        xw, yw, stats = _wrap_ring_around_barriers(
            x, y, barrier_geometry, barrier_interior)
        wrapped.append((xw, yw))
        for key, value in stats.items():
            totals[key] += value
    return wrapped, totals


# =============================================================================
# Categorical barrier geometry and fail-fast diagnostic checks
# =============================================================================

class MarkerBarrierError(ValueError):
    """A proposed marker trajectory or perimeter entered non-burnable fuel."""


def _barrier_geometry_from_grid(param_grid: "FireParameterGrid"):
    """Polygonize categorical non-fuel cells using grid-centre affine geometry.

    This is intentionally a CPU/Shapely diagnostic and is not used by the
    differentiable JAX functions below. The fuel mask MUST align with ROS.
    """
    if param_grid.nonfuel_mask is None:
        raise ValueError("Barrier checks require FireParameterGrid.nonfuel_mask")
    try:
        from affine import Affine
        from rasterio.features import shapes
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("Barrier diagnostics require rasterio and shapely") from exc

    mask = np.asarray(param_grid.nonfuel_mask, dtype=np.uint8)
    dx = float(param_grid.x_coords[1] - param_grid.x_coords[0])
    dy = float(param_grid.y_coords[1] - param_grid.y_coords[0])
    transform = Affine.translation(
        float(param_grid.x_coords[0]) - 0.5 * dx,
        float(param_grid.y_coords[0]) - 0.5 * dy,
    ) * Affine.scale(dx, dy)
    parts = [
        shape(geom) for geom, value in shapes(
            mask, mask=mask.astype(bool), transform=transform
        ) if int(value) == 1
    ]
    return unary_union(parts) if parts else None


def _save_barrier_diagnostic(
    old_ring, new_ring, violation_geometry, *,
    reason: str, phase: str, time_min: float | None = None,
    diagnostic_dir: str = "marker_barrier_debug",
):
    """Persist first failure geometries without altering the solver state."""
    from pathlib import Path
    from uuid import uuid4
    import json
    from shapely.geometry import LineString

    folder = Path(diagnostic_dir)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"barrier_{uuid4().hex[:10]}"
    xy_new = np.column_stack(new_ring)
    (folder / f"{name}_candidate_ring.wkt").write_text(
        LineString(np.vstack([xy_new, xy_new[0]])).wkt, encoding="utf-8"
    )
    if old_ring is not None:
        xy_old = np.column_stack(old_ring)
        (folder / f"{name}_previous_ring.wkt").write_text(
            LineString(np.vstack([xy_old, xy_old[0]])).wkt, encoding="utf-8"
        )
    if violation_geometry is not None:
        (folder / f"{name}_intersection.wkt").write_text(
            violation_geometry.wkt, encoding="utf-8"
        )
    (folder / f"{name}_metadata.json").write_text(
        json.dumps({"time_min": time_min, "phase": phase, "reason": reason},
                   indent=2), encoding="utf-8"
    )
    logger.error("Marker barrier violation diagnostics: %s", folder.resolve())


def validate_marker_boundary(
    x: np.ndarray, y: np.ndarray, barrier_geometry,
    *, time_min: float | None = None, phase: str = "perimeter",
    tolerance_m: float = 1e-6,
    diagnostic_dir: str = "marker_barrier_debug",
) -> None:
    """Reject a ring segment crossing an obstacle INTERIOR.

    Boundary contact is permitted. This does not test whether a ring encloses
    an unburned island; full geometry export can represent such holes, but
    interior-boundary evolution remains a separate unsolved capability.
    """
    import shapely
    from shapely.geometry import Point
    if barrier_geometry is None or barrier_geometry.is_empty:
        return
    pts = np.column_stack((np.asarray(x, dtype=float),
                           np.asarray(y, dtype=float)))
    if len(pts) < 3 or not np.isfinite(pts).all():
        raise MarkerBarrierError("Invalid marker coordinates during barrier check")
    # Intersect each edge on its own. A self-crossing ring intersected as one
    # LineString makes GEOS fall back to snapping, whose tolerance at UTM
    # coordinates exceeds the 1e-7 m gap between a barrier face and the
    # shrunken obstacle interior, so edges lying on a face were reported
    # as inside it.
    edges = shapely.linestrings(np.stack((pts, np.roll(pts, -1, axis=0)), axis=1))
    pieces = shapely.intersection(edges, barrier_geometry)
    inside_length = float(shapely.length(pieces).sum())
    if inside_length > tolerance_m:
        intersection = shapely.union_all(pieces[shapely.length(pieces) > 0])
        reason = (f"{phase}: perimeter lies {inside_length:.6f} m "
                  "inside non-burnable fuel")
        _save_barrier_diagnostic(None, (x, y), intersection,
                                 reason=reason, phase=phase,
                                 time_min=time_min,
                                 diagnostic_dir=diagnostic_dir)
        raise MarkerBarrierError(reason)
    # Tangential intersections have zero length; points strictly inside are
    # still prohibited. Covers rare zero-length malformed segments.
    for xy in pts:
        if barrier_geometry.contains(Point(*xy)):
            reason = f"{phase}: marker lies inside non-burnable fuel at {tuple(xy)}"
            _save_barrier_diagnostic(None, (x, y), Point(*xy),
                                     reason=reason, phase=phase,
                                     time_min=time_min,
                                     diagnostic_dir=diagnostic_dir)
            raise MarkerBarrierError(reason)


def validate_marker_barrier_crossings(
    x_old: np.ndarray, y_old: np.ndarray,
    x_new: np.ndarray, y_new: np.ndarray,
    barrier_geometry,
    *, time_min: float | None = None, phase: str = "advection",
    tolerance_m: float = 1e-6,
    diagnostic_dir: str = "marker_barrier_debug",
    check_boundary: bool = True,
) -> None:
    """Validate swept marker paths AND the proposed connected perimeter.

    This method requires one-to-one marker identity. Call only before/after
    advection, NOT after insertion or deletion of marker vertices.
    Pass ``check_boundary=False`` when the perimeter edges are repaired
    (wrapped around barriers) afterwards and validated separately.
    """
    from shapely.geometry import LineString, Point
    if barrier_geometry is None or barrier_geometry.is_empty:
        return
    a = np.column_stack((np.asarray(x_old), np.asarray(y_old)))
    b = np.column_stack((np.asarray(x_new), np.asarray(y_new)))
    if a.shape != b.shape:
        raise ValueError("Marker correspondence changed; use boundary-only validation")
    for i, (p0, p1) in enumerate(zip(a, b)):
        segment = LineString([p0, p1]) if not np.array_equal(p0, p1) else Point(*p1)
        clipped = segment.intersection(barrier_geometry)
        if clipped.length > tolerance_m or barrier_geometry.contains(Point(*p1)):
            reason = (f"{phase}: marker {i} crossed non-burnable fuel "
                      f"at time {time_min}; overlap={clipped.length:.6f} m")
            _save_barrier_diagnostic((x_old, y_old), (x_new, y_new),
                                     clipped, reason=reason, phase=phase,
                                     time_min=time_min,
                                     diagnostic_dir=diagnostic_dir)
            raise MarkerBarrierError(reason)
    if check_boundary:
        validate_marker_boundary(x_new, y_new, barrier_geometry,
                                 time_min=time_min, phase=phase,
                                 tolerance_m=tolerance_m,
                                 diagnostic_dir=diagnostic_dir)


def _snapshot_fire_geometry(rings, barrier_geometry=None):
    """Build complete Polygon/MultiPolygon for optional *export-only* history.

    Subtracting obstacles is for area accounting and hole preservation in
    exports. It does NOT correct trajectories or implement inner-ring evolution.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polygons = []
    for x, y in rings:
        polygon = Polygon(np.column_stack((x, y)))
        if not polygon.is_valid:
            raise ValueError("Cannot export invalid marker polygon as burned geometry")
        polygons.append(polygon)
    geom = unary_union(polygons)
    if barrier_geometry is not None and not barrier_geometry.is_empty:
        geom = geom.difference(barrier_geometry)
    return geom


def _clip_burned_geometry_to_fuel(
    geometry,
    barrier_geometry,
    *,
    phase: str = "burned_geometry",
    time_min: float | None = None,
):
    """Exclude categorical non-fuel from internal polygonal burned geometry.

    This is applied to swept areas and the internal reconstruction, NOT just
    the exported GeoPackage. Clipping alone does not make a marker path or
    a disconnected burned component physically reachable. Strict mode also
    checks ring edges and newly formed components.
    """
    if geometry is None or geometry.is_empty:
        return geometry
    if barrier_geometry is None or barrier_geometry.is_empty:
        return geometry

    import shapely
    from shapely.validation import explain_validity

    if not geometry.is_valid:
        raise MarkerTopologyError(
            f"{phase} at t={time_min}: input geometry is invalid: "
            f"{explain_validity(geometry)}"
        )

    # Most early H3 timesteps are nowhere near an obstacle. Avoid expensive
    # overlay operations until the candidate geometry actually touches one.
    if not geometry.intersects(barrier_geometry):
        return geometry

    clipped = geometry.difference(barrier_geometry)
    removed_area = max(0.0, float(geometry.area - clipped.area))
    if removed_area > 1e-6:
        logger.debug(
            "Excluded %.8f m2 of non-burnable area from %s at t=%s min",
            removed_area, phase, time_min,
        )
    if not clipped.is_valid:
        raise MarkerTopologyError(
            f"{phase} at t={time_min}: barrier clipping invalid: "
            f"{explain_validity(clipped)}"
        )
    if clipped.geom_type not in (
        'Polygon', 'MultiPolygon', 'GeometryCollection'
    ) and not clipped.is_empty:
        raise MarkerTopologyError(
            f"{phase} at t={time_min}: unexpected geometry {clipped.geom_type}"
        )
    if clipped.geom_type == 'GeometryCollection':
        parts = _swept_polygon_parts(clipped)
        clipped = shapely.union_all(list(parts), grid_size=0)
    remaining_overlap = float(clipped.intersection(barrier_geometry).area)
    if remaining_overlap > 1e-6:
        raise MarkerBarrierError(
            f"{phase} at t={time_min}: {remaining_overlap:.8f} m2 "
            "of non-burnable cells remain after clipping"
        )
    return clipped


def _validate_burned_geometry_connectivity(
    previous_geometry,
    current_geometry,
    *,
    time_min: float,
    tolerance_m: float = 1e-5,
):
    """Fail if an existing fire generates a detached burned polygon.

    Each polygon in the reconstructed burn must touch some previously burned
    ground. This does not independently prove a viable path through fuel; it
    complements per-marker and perimeter-edge crossing tests. Existing
    separate ignitions are permitted because each has prior burned ground.
    """
    if (previous_geometry is None or previous_geometry.is_empty
            or current_geometry is None or current_geometry.is_empty):
        return
    anchor = previous_geometry.buffer(float(tolerance_m))
    for idx, part in enumerate(_geometry_polygons(current_geometry)):
        if not part.intersects(anchor):
            raise MarkerBarrierError(
                f"Detached burned component {idx} at t={time_min:.6f} min; "
                "barrier-aware front reconnection is required in topology.py"
            )


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


def _heun_step_xy(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    vel_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Take one second-order Heun step for marker advection."""
    vx_start, vy_start = vel_fn(x, y)
    x_predict = x + dt * vx_start
    y_predict = y + dt * vy_start
    vx_predict, vy_predict = vel_fn(x_predict, y_predict)

    x_new = x + 0.5 * dt * (vx_start + vx_predict)
    y_new = y + 0.5 * dt * (vy_start + vy_predict)
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
    nonfuel_mask: np.ndarray | None = None
    
    def __post_init__(self):
        """Validate array shapes and compute grid parameters."""
        assert self.ros.ndim == 3, "ROS must be 3D (time, y, x)"
        assert self.ros.shape == self.bros.shape == self.fros.shape == self.raz.shape
        
        self.nt, self.ny, self.nx = self.ros.shape
        
        #Validate the optional categorical non-fuel mask
        if self.nonfuel_mask is not None:

            self.nonfuel_mask = np.asarray(
                self.nonfuel_mask,
                dtype=bool,
            )

            if self.nonfuel_mask.shape != (
                self.ny, self.nx
            ):
                raise ValueError(
                    "nonfuel_mask shape must match "
                    f"ROS spatial shape {(self.ny, self.nx)}; "
                    f"received {self.nonfuel_mask.shape}"
                )
        
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
    
    
    def is_burnable_at(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Test whether coordinates lie inside burnable cells.

        Uses nearest-cell lookup for categorical fuel classes.
        Coordinates must be in the same projected CRS as
        the parameter grid.

        This method does not test movement trajectories.
        """

        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )

        if self.nonfuel_mask is None:
            return np.ones(x.shape, dtype=bool)

        # Grid coordinates describe cell centres.
        # Signed spacing handles both ascending and
        # descending y coordinates.
        col_f = (
            (x - float(self.x_coords[0]))
            / float(self.x_coords[1] - self.x_coords[0])
        )

        row_f = (
            (y - float(self.y_coords[0]))
            / float(self.y_coords[1] - self.y_coords[0])
        )

        finite = np.isfinite(col_f) & np.isfinite(row_f)

        inside = (
            finite
            & (col_f >= -0.5)
            & (col_f < self.nx - 0.5)
            & (row_f >= -0.5)
            & (row_f < self.ny - 0.5)
        )

        safe_col = np.where(finite, col_f, 0.0)
        safe_row = np.where(finite, row_f, 0.0)

        col = np.clip(
            np.floor(safe_col + 0.5).astype(int),
            0,
            self.nx - 1,
        )

        row = np.clip(
            np.floor(safe_row + 0.5).astype(int),
            0,
            self.ny - 1,
        )

        return inside & ~self.nonfuel_mask[row, col]

    
    def get_barrier_geometry(self):
        """Return cached polygonized nonfuel_mask for CPU diagnostics."""
        if self.nonfuel_mask is None:
            return None
        if not hasattr(self, "_barrier_geometry_cache"):
            self._barrier_geometry_cache = _barrier_geometry_from_grid(self)
        return self._barrier_geometry_cache

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
        # Azimuth is circular. A scalar average of 359 and 1 degrees would
        # yield 180 degrees, reversing the spread direction. Interpolate unit
        # direction components, then reconstruct the azimuth. A near-zero
        # resultant (opposing directions) uses the nearest cell's direction.
        sin_raz = self._bilinear_interpolate(np.sin(self.raz[t_index]), x, y)
        cos_raz = self._bilinear_interpolate(np.cos(self.raz[t_index]), x, y)
        raz = np.mod(np.arctan2(sin_raz, cos_raz), 2.0 * np.pi)
        ambiguous = np.hypot(sin_raz, cos_raz) < 1e-8
        if np.any(ambiguous):
            xq, yq = np.broadcast_arrays(np.asarray(x), np.asarray(y))
            ix = np.rint((xq - self.x_coords[0]) / self.dx).astype(int)
            iy = np.rint((yq - self.y_coords[0]) / self.dy).astype(int)
            ix = np.clip(ix, 0, self.nx - 1)
            iy = np.clip(iy, 0, self.ny - 1)
            nearest_raz = np.mod(self.raz[t_index, iy, ix], 2.0 * np.pi)
            raz = np.where(ambiguous, nearest_raz, raz)
            logger.debug('Ambiguous RAZ interpolation at %d point(s); using nearest-cell azimuth',
                         int(np.count_nonzero(ambiguous)))

        if self.nonfuel_mask is not None:

            burnable = self.is_burnable_at(x, y)

            ros = np.where(burnable, ros, 0.0)
            bros = np.where(burnable, bros, 0.0)
            fros = np.where(burnable, fros, 0.0)

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
    """Perimeter history for one or several simultaneous ignition fronts.

    Legacy ``perimeters`` entries remain (x, y) tuples for a single front.
    In multiple-ignition runs, entries are None while separate fronts exist.
    Use ``get_fronts(i)`` or ``fronts_by_time`` in those cases: a disconnected
    fire must never be falsely serialized as one closed ring.
    """

    perimeters: list[tuple[np.ndarray, np.ndarray] | None]
    times: list[float]
    fronts_by_time: list[list[tuple[np.ndarray, np.ndarray]]] | None = None
    merger_events: list[dict] = field(default_factory=list)
    topology_repairs: list[dict] = field(default_factory=list)
    burned_geometries: list | None = None  # Optional full Polygon/MultiPolygon snapshots
    holes_by_time: list[list[tuple[np.ndarray, np.ndarray]]] | None = None

    @property
    def n_steps(self) -> int:
        return len(self.times)

    def get_fronts(self, index: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """All exterior closed rings at a stored time (copies)."""
        if self.fronts_by_time is None:
            perimeter = self.perimeters[index]
            rings = [] if perimeter is None else [perimeter]
        else:
            rings = self.fronts_by_time[index]
        return [(x.copy(), y.copy()) for x, y in rings]

    def get_holes(self, index: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return clockwise inner fire-front rings at a saved time (copies).

        These are *active* interior boundaries of the burned region; they
        are NOT additional burned fire components. Fixed non-fuel barrier
        holes are represented only in :meth:`get_burned_geometry`.
        """
        if self.holes_by_time is None:
            return []
        return [(x.copy(), y.copy()) for x, y in self.holes_by_time[index]]

    def get_perimeter(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Legacy single-ring accessor; reject ambiguous multi-ring time steps."""
        fronts = self.get_fronts(index)
        if len(fronts) != 1:
            raise ValueError(
                f"Snapshot {index} has {len(fronts)} distinct fire fronts. "
                "Use get_fronts(index), not get_perimeter(index)."
            )
        return fronts[0]

    def get_final_perimeter(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.times:
            return np.array([]), np.array([])
        return self.get_perimeter(-1)

    def get_burned_geometry(self, index: int):
        """Return exact saved geometry when available, or legacy ring union."""
        if self.burned_geometries is not None:
            return self.burned_geometries[index]
        return _snapshot_fire_geometry(self.get_fronts(index))

    def compute_areas(self) -> np.ndarray:
        """Return burned area, honoring optional hole-bearing geometries."""
        if self.burned_geometries is not None:
            return np.asarray([float(g.area) for g in self.burned_geometries])
        # Legacy fallback: original exterior-ring shoelace calculation.
        result = []
        for i in range(self.n_steps):
            total = 0.0
            for x, y in self.get_fronts(i):
                xx = np.asarray(x) - x[0]
                yy = np.asarray(y) - y[0]
                total += 0.5 * abs(np.dot(xx, np.roll(yy, -1))
                                   - np.dot(yy, np.roll(xx, -1)))
            result.append(total)
        return np.asarray(result, dtype=float)

    def to_geodataframe(self, crs=None):
        """One row per active exterior ring at each stored time.

        Requires GeoPandas and Shapely. Save with ``gdf.to_file(path,
        driver='GPKG')``. If burned_geometries is provided, holes and
        separate polygon components are preserved during export. Legacy
        exterior-only history still lacks hole-aware propagation.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon

        records = []
        if self.burned_geometries is not None:
            if len(self.burned_geometries) != len(self.times):
                raise ValueError("Burned geometry history and time history have different lengths")
            for i, (t, geom) in enumerate(zip(self.times, self.burned_geometries)):
                for j, part in enumerate(_geometry_polygons(geom)):
                    records.append({
                        "step": i, "t_min": float(t), "front_id": j,
                        "area_m2": float(part.area), "geometry": part,
                    })
        else:
            for i, t in enumerate(self.times):
                for j, (x, y) in enumerate(self.get_fronts(i)):
                    poly = Polygon(np.column_stack((x, y)))
                    records.append({
                        "step": i, "t_min": float(t), "front_id": j,
                        "area_m2": float(poly.area), "geometry": poly,
                    })
        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)



def simulate_fire_spread(
    param_grid: FireParameterGrid,
    x_ignition: float | None = None,
    y_ignition: float | None = None,
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
    advection_integrator: str = "euler",  # "euler" | "heun" | "rk4" | "euler_extrap" | "implicit"
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
    ignition_points: Sequence[tuple[float, float]] | None = None,
    barrier_mode: str = "off",  # "off" (legacy) or "raise" (strict barrier enforcement)
    barrier_debug_dir: str = "marker_barrier_debug",
    export_burned_geometry: bool = False,
    topology_mode: str = "legacy",  # legacy | outer_hull; multi-ignition only
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
        - ``"heun"`` — second-order Heun (explicit trapezoidal) method.
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

    barrier_mode : {"off", "raise"}
        "off" retains legacy raster-sampled marker motion and does not
        guarantee obstacle-safe fronts. "raise" geometrically stops marker
        trajectories before entering non-fuel, validates connected perimeter
        edges and reconstructed burns, and fails if topology.py cannot route
        the active front around an obstacle. It is NOT full obstacle-aware
        edge reconnection; that remains to be implemented in topology.py.
    barrier_debug_dir : str
        Directory receiving first-failure WKT and JSON records.
    export_burned_geometry : bool
        Store Polygon/MultiPolygon snapshots, subtracting the non-fuel mask
        for area accounting only; this does not repair front propagation.
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
    `ignition_points=[(x1,y1),(x2,y2),...]` enables simultaneous ignitions
    on a projected metric grid. Multi-front history uses `get_fronts(i)`.
    `topology_mode='outer_hull'` opt-in for multi-ignition runs retains
    interior rings via oriented-linework polygonization, winding classification,
    and union with previously burned ground; it is an experimental approximate
    outer-hull implementation rather than the exact Barber et al. algorithm.
    For historical compatibility the default remains `topology_mode='legacy'`.
    The outer-hull mode always stores exact burned geometries so areas include
    holes, even when `export_burned_geometry=False`.
    """
    # Opt-in barrier diagnostics keep legacy physics untouched by default.
    if barrier_mode not in {"off", "raise"}:
        raise ValueError("barrier_mode must be 'off' or 'raise'")
    if barrier_mode != "off" and param_grid.nonfuel_mask is None:
        raise ValueError("barrier_mode='raise' requires nonfuel_mask on FireParameterGrid")
    if barrier_mode == "raise" and is_geographic:
        raise ValueError("Barrier diagnostics require projected metric coordinates")
    # Needed whenever non-fuel exists: perimeter edges are wrapped around
    # barriers in every marker path, not only in diagnostic or export modes.
    barrier_geometry = (param_grid.get_barrier_geometry()
                        if param_grid.nonfuel_mask is not None
                        else None)
    barrier_interior = (barrier_geometry.buffer(-1e-7)
                        if barrier_geometry is not None and not barrier_geometry.is_empty
                        else None)
    if export_burned_geometry and barrier_mode == "off":
        logger.warning("Export mask removes barriers in GeoPackage only; it does NOT fix front propagation")

    # The outer_hull topology (loop removal, unburned islands, barrier holes)
    # lives in the multi-front solver, which also works for one ignition.
    if (ignition_points is None and topology_mode == "outer_hull"
            and x_ignition is not None and y_ignition is not None):
        ignition_points = [(float(x_ignition), float(y_ignition))]

    # Multiple ignitions are an explicit opt-in. Preserve the legacy
    # single-ignition numerical path exactly when not requested.
    if ignition_points is not None:
        points = np.asarray(ignition_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1:
            raise ValueError("ignition_points must have shape (N, 2), N >= 1")
        if not np.isfinite(points).all():
            raise ValueError("Ignition coordinates must be finite")
        if len(points) == 1 and topology_mode != "outer_hull":
            x_ignition, y_ignition = map(float, points[0])
        else:
            return _simulate_multiple_ignitions(
                param_grid=param_grid,
                ignition_points=[tuple(map(float, pt)) for pt in points],
                dt=dt, n_vertices=n_vertices, initial_radius=initial_radius,
                store_every=store_every, max_steps=max_steps,
                use_markers=use_markers, marker_epsilon=marker_epsilon,
                min_ros=min_ros, is_geographic=is_geographic,
                advection_integrator=advection_integrator,
                resample_spacing=resample_spacing,
                insert_factor=insert_factor, delete_factor=delete_factor,
                smoothing_weight=smoothing_weight,
                smoothing_iters=smoothing_iters,
                redistribute_every=redistribute_every,
                max_substeps=max_substeps, max_move_fraction=max_move_fraction,
                barrier_mode=barrier_mode, barrier_interior=barrier_interior,
                barrier_geometry=barrier_geometry,
                barrier_debug_dir=barrier_debug_dir,
                export_burned_geometry=export_burned_geometry,
                topology_mode=topology_mode,
            )
    if x_ignition is None or y_ignition is None:
        raise ValueError("Pass x_ignition and y_ignition, or ignition_points")

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
    
    # Validate ignition footprint before starting a constrained experiment.
    if barrier_mode == "raise":
        validate_marker_boundary(x, y, barrier_interior,
                                 time_min=0.0, phase="initial_front",
                                 diagnostic_dir=barrier_debug_dir)
    history = FirePerimeterHistory(
        perimeters=[(x.copy(), y.copy())],
        times=[0.0],
        burned_geometries=([_snapshot_fire_geometry([(x, y)], barrier_geometry)]
                           if export_burned_geometry else None),
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
        # Higher-order paths must sample the grid at intermediate stages.
        # ------------------------------------------------------------------

        # Always sample current ROS in m/min for extinguishment logic
        ros_now_m, _, _, _ = param_grid.sample_at(step, x, y)
        if np.all(ros_now_m < min_ros):
            logger.info(f"Fire extinguished at step {step} (all ROS below threshold)")
            break

        integrator = (advection_integrator or "euler").strip().lower()

        # Shared velocity closure: resample the grid at arbitrary (x,y) positions.
        # Used by all integrators so that adaptive sub-stepping is consistent.
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

            # Preserve per-vertex extinction and enforce the categorical
            # non-fuel mask after continuous ROS interpolation.
            allowed = np.asarray(ros_m >= min_ros, dtype=bool)
            if param_grid.nonfuel_mask is not None:
                allowed &= param_grid.is_burnable_at(xq, yq)
            if use_markers:
                allowed &= compute_active_vertices(xq, yq, marker_epsilon_coord)
            return np.where(allowed, x_t, 0.0), np.where(allowed, y_t, 0.0)

        # Determine target spacing for adaptive sub-stepping and redistribution.
        if resample_spacing is not None and resample_spacing > 0:
            target_spacing = float(resample_spacing)
        else:
            target_spacing = float(marker_epsilon_coord) if marker_epsilon_coord > 0 else 0.0

        # Save last known good coordinates for NaN recovery
        x_prev, y_prev = x.copy(), y.copy()

        def _check_advection(old_x, old_y, new_x, new_y, sub_index, phase):
            if barrier_mode == "raise":
                validate_marker_barrier_crossings(
                    old_x, old_y, new_x, new_y, barrier_interior,
                    time_min=(step + (sub_index + 1) / max(nsub, 1)) * dt,
                    phase=phase, diagnostic_dir=barrier_debug_dir,
                    check_boundary=False,
                )

        if integrator == "heun":
            # Adaptive sub-stepping to limit movement per substep (Heun)
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
                xo, yo = x.copy(), y.copy()
                x, y = _heun_step_xy(x, y, subdt, _vel_from_grid)
                if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                    x, y = _apply_marker_barrier_limit(param_grid, xo, yo, x, y,
                                                     barrier_mode=barrier_mode,
                                                     barrier_interior=barrier_interior)
                    _check_advection(xo, yo, x, y, isub, "heun")
                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                    logger.warning(
                        f"Non-finite coordinates at step {step}, substep {isub}, "
                        "reverting to previous"
                    )
                    x, y = x_prev.copy(), y_prev.copy()
                    break

        elif integrator == "rk4":
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
                xo, yo = x.copy(), y.copy()
                x, y = _rk4_step_xy(x, y, subdt, _vel_from_grid)
                if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                    x, y = _apply_marker_barrier_limit(param_grid, xo, yo, x, y,
                                                     barrier_mode=barrier_mode,
                                                     barrier_interior=barrier_interior)
                    _check_advection(xo, yo, x, y, isub, "rk4")
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
            if np.isfinite(x).all() and np.isfinite(y).all():
                x, y = _apply_marker_barrier_limit(param_grid, x_prev, y_prev, x, y,
                                                     barrier_mode=barrier_mode,
                                                     barrier_interior=barrier_interior)

            if barrier_mode == "raise" and np.isfinite(x).all() and np.isfinite(y).all():
                validate_marker_barrier_crossings(
                    x_prev, y_prev, x, y, barrier_interior,
                    time_min=(step+1)*dt, phase="euler_extrap",
                    diagnostic_dir=barrier_debug_dir, check_boundary=False)
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
            if np.isfinite(x).all() and np.isfinite(y).all():
                x, y = _apply_marker_barrier_limit(param_grid, x_prev, y_prev, x, y,
                                                     barrier_mode=barrier_mode,
                                                     barrier_interior=barrier_interior)

            if barrier_mode == "raise" and np.isfinite(x).all() and np.isfinite(y).all():
                validate_marker_barrier_crossings(
                    x_prev, y_prev, x, y, barrier_interior,
                    time_min=(step+1)*dt, phase="implicit",
                    diagnostic_dir=barrier_debug_dir, check_boundary=False)
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
                xo, yo = x.copy(), y.copy()
                x_t, y_t = _vel_from_grid(x, y)
                x = x + subdt * x_t
                y = y + subdt * y_t
                if np.isfinite(x).all() and np.isfinite(y).all():
                    x, y = _apply_marker_barrier_limit(param_grid, xo, yo, x, y,
                                                     barrier_mode=barrier_mode,
                                                     barrier_interior=barrier_interior)
                    _check_advection(xo, yo, x, y, isub, "euler")

                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                    logger.warning(f"Non-finite coordinates at step {step}, substep {isub}, reverting to previous")
                    x, y = x_prev.copy(), y_prev.copy()
                    break

        # Neighbouring markers may both be on fuel while the edge joining
        # them cuts a non-fuel cell. Wrap such edges along the barrier.
        x, y, _ = _wrap_ring_around_barriers(x, y, barrier_geometry, barrier_interior)
        if barrier_mode == "raise":
            validate_marker_boundary(
                x, y, barrier_interior,
                time_min=(step+1)*dt, phase="advected_front",
                diagnostic_dir=barrier_debug_dir)

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
                    barrier_geometry=barrier_geometry,
                    barrier_interior=barrier_interior,
                )

                # Very light local smoothing to suppress high-frequency numerical noise.
                # Unlike Chaikin smoothing, this does not materially shrink the perimeter.
                if smoothing_iters and int(smoothing_iters) > 0 and smoothing_weight and float(smoothing_weight) > 0:
                    xs0, ys0 = x.copy(), y.copy()
                    x, y = _smooth_ring_local_xy(x, y, w=float(smoothing_weight), n_iter=int(smoothing_iters))
                    # Smoothing keeps marker identity; never let it pull a marker into non-fuel.
                    x, y = _apply_marker_barrier_limit(param_grid, xs0, ys0, x, y,
                                                     barrier_mode=barrier_mode,
                                                     barrier_interior=barrier_interior)
                # Smoothing or a relocated midpoint can leave an edge across
                # a barrier corner; wrap it again.
                x, y, _ = _wrap_ring_around_barriers(
                    x, y, barrier_geometry, barrier_interior)
                # Vertex count/order may change: check ONLY the new boundary.
                if barrier_mode == "raise":
                    validate_marker_boundary(
                        x, y, barrier_interior,
                        time_min=(step+1)*dt, phase="redistribution",
                        diagnostic_dir=barrier_debug_dir)

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
            if history.burned_geometries is not None:
                history.burned_geometries.append(
                    _snapshot_fire_geometry([(x, y)], barrier_geometry))

    logger.info(f"Simulation complete: {len(history.perimeters)} perimeters stored")

    return history


# =============================================================================
# Multiple simultaneous ignitions and fire-front mergers (CPU marker solver)
# =============================================================================

# Notes on numerical scope:
# * Each currently disjoint fire is a separate, counter-clockwise marker ring.
# * When burned polygons overlap, Shapely's geometric union removes interior
#   front segments. The union exterior is resampled for continued advection.
# * Existing per-ring integration routines, including forward Euler, are reused.
# * Geometric union is a topology operation, NOT an additional burn rate.
# * Legacy merging uses buffer(0) repairs, and rejects interior holes.
# * Experimental topology_mode='outer_hull' nodes intersecting linework,
#   polygonizes, classifies faces by directed winding, and preserves holes.
#   This is NOT the exact sequential crossing algorithm of Barber et al. 2009.
# * Both Shapely topology paths are CPU operations and nondifferentiable;
#   JAX's separate advection routines below remain unchanged.


def _require_shapely():
    try:
        import shapely  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Multiple-front merging requires Shapely. Install with pip install shapely."
        ) from exc


def _geometry_polygons(geom):
    """Flatten Polygon, MultiPolygon, or GeometryCollection to polygons."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = []
        for item in geom.geoms:
            parts.extend(_geometry_polygons(item))
        return parts
    return []


def _validated_polygon(ring):
    """Build a valid marker polygon, reporting topological repairs explicitly."""
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity
    from shapely.ops import unary_union

    x, y = ring
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if (len(x) < 4 or len(y) != len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("A marker ring is too short or contains non-finite coordinates")
    poly = Polygon(np.column_stack((x, y)))
    if poly.area <= 0:
        raise ValueError("Degenerate zero-area marker perimeter")
    if not poly.is_valid:
        # Richards propagation near a freshly merged cusp may self-cross.
        # Use geometry repair and preserve the PREVIOUS burned region in the
        # caller; otherwise repairing a bowtie could artificially unburn land.
        why = explain_validity(poly)
        logger.warning("Repairing self-intersecting ring: %s", why)
        repaired = poly.buffer(0)
        parts = _geometry_polygons(repaired)
        if not parts:
            raise ValueError("Self-intersection repair produced no polygons: " + why)
        poly = unary_union(parts)
        if not poly.is_valid:
            raise ValueError("Unable to repair invalid marker front: " + why)
    for part in _geometry_polygons(poly):
        if part.interiors:
            
            from pathlib import Path
            from uuid import uuid4
            
            debug_dir = Path("marker_topology_debug")
            debug_dir.mkdir(exist_ok=True)
            
            run_id = uuid4().hex[:8]
            
            original = Polygon(np.column_stack((x, y)))
            
            original_path = debug_dir / (
                f"invalid_ring_{run_id}.wkt"   
            )
            repaired_path = debug_dir / (
                f"repaired_polygon_{run_id}.wkt"
            )
            original_path.write_text(
                original.wkt,
                encoding="utf-8",
            )
            repaired_path.write_text(
                poly.wkt,
                encoding="utf-8",
            )
            hole_areas = [
                Polygon(ring).area
                for ring in part.interiors
            ]
            
            logger.error(
                "Topology failure: %d holes; "
                "hole areas (m2): %s",
                len(part.interiors),
                hole_areas,
            )
            
            logger.error(
                "Invalid ring saved to: %s",
                original_path.resolve()    
            )
            
            logger.error(
                "Repaired polygon saved to: %s",
                repaired_path.resolve()
            )  
            
            raise ValueError(
                "Topology repair produced an unburned hole. "
                "Diagnostic geometries saved."
            )    
            
    return poly


def _ring_from_polygon(poly, spacing, min_vertices=32, max_vertices=20000):
    """Resample the exterior into uniformly spaced CCW markers."""
    from shapely.geometry.polygon import orient

    if not poly.is_valid or poly.is_empty or poly.area <= 0:
        raise ValueError("Cannot resample invalid/empty fire polygon")
    if poly.interiors:
        raise ValueError(
            "Union produced an unburned island/hole; hole-aware ring evolution "
            "is not implemented. Refusing to discard the interior boundary."
        )
    poly = orient(poly, sign=1.0)
    boundary = poly.exterior
    perimeter = float(boundary.length)
    n = int(np.clip(np.ceil(perimeter / spacing), min_vertices, max_vertices))
    dists = np.arange(n, dtype=float) * (perimeter / n)
    coords = np.array([[boundary.interpolate(float(d)).x,
                        boundary.interpolate(float(d)).y] for d in dists])
    return coords[:, 0], coords[:, 1]


def _merge_intersecting_fronts(rings, spacing, n_vertices, previous_geometry=None):
    """Remove internal boundaries when separate burned polygons overlap.

    Returns (rings, did_merge, union_geometry). The original rings remain
    untouched if there is no merger (to avoid changing marker discretization).
    """
    from shapely.ops import unary_union

    from shapely.geometry import Polygon
    had_repair = any(not Polygon(np.column_stack(ring)).is_valid for ring in rings)
    polygons = [_validated_polygon(ring) for ring in rings]
    union = unary_union(polygons)
    if had_repair and previous_geometry is not None:
        # Repair can otherwise delete a sliver that was already burned.
        union = unary_union([previous_geometry, union])
    parts = sorted(_geometry_polygons(union), key=lambda g: (-g.area, g.centroid.x))
    if len(parts) > len(rings):
        raise ValueError("Topology unexpectedly fragmented an existing fire front")
    if not parts:
        raise ValueError("No valid fire polygons remain after geometric union")
    for part in parts:
        if part.interiors:
            raise ValueError(
                "Front union formed an unburned island. Inner-ring propagation "
                "must be added before simulating this topology."
            )
    if len(parts) == len(rings) and not had_repair:
        return rings, False, union

    # Formerly overlapping interiors have become inactive. Only true exterior
    # boundaries are retained and resampled for further Richards advection.
    result = [_ring_from_polygon(part, spacing,
                                min_vertices=max(32, min(n_vertices // 2, 100)),
                                max_vertices=max(20000, 10 * n_vertices))
              for part in parts]
    return result, len(parts) < len(rings), union


def _copy_rings(rings):
    return [(x.copy(), y.copy()) for x, y in rings]


def _append_front_snapshot(
    history, rings, time, barrier_geometry=None, *,
    burned_geometry=None, outer_hull=False,
):
    """Store exterior and interior rings without converting holes into fire."""
    copy = _copy_rings(rings)
    history.times.append(float(time))
    if outer_hull:
        exteriors = [(x, y) for x, y in copy if _ring_signed_area(x, y) > 0]
        holes = [(x, y) for x, y in copy if _ring_signed_area(x, y) < 0]
        history.fronts_by_time.append(exteriors)
        if history.holes_by_time is not None:
            history.holes_by_time.append(holes)
        history.perimeters.append(exteriors[0] if len(exteriors) == 1 else None)
        if history.burned_geometries is not None:
            if burned_geometry is None:
                raise ValueError('Outer-hull snapshots require exact burned_geometry')
            history.burned_geometries.append(burned_geometry)
    else:
        history.fronts_by_time.append(copy)
        history.perimeters.append(copy[0] if len(copy) == 1 else None)
        if history.burned_geometries is not None:
            history.burned_geometries.append(
                _snapshot_fire_geometry(copy, barrier_geometry))


# Backwards-compatible exception name: both modules now raise the same type.
MarkerTopologyError = _marker_topology.MarkerTopologyError


def _ring_signed_area(x, y) -> float:
    """Signed area: positive CCW exterior, negative CW interior."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return 0.0
    # Translate to avoid catastrophic cancellation on UTM coordinates.
    xx, yy = x - x[0], y - y[0]
    return float(0.5 * np.sum(xx * np.roll(yy, -1) -
                               yy * np.roll(xx, -1)))


# These older geometry utilities remain available for backwards compatibility.
# Experimental front reconstruction itself is exclusively in topology.py.


def _ring_pair_crossings(rings) -> bool:
    """Determine whether two different boundaries touch or intersect."""
    from shapely.geometry import LineString
    lines = []
    for x, y in rings:
        xy = np.column_stack((x, y))
        lines.append(LineString(np.vstack((xy, xy[0]))))
    return any(lines[i].intersects(lines[j])
               for i in range(len(lines))
               for j in range(i + 1, len(lines)))


def _rings_from_burned_geometry(geom, spacing, n_vertices):
    """Return CCW exteriors and CW holes without changing their geometry.

    Crucially, *retain every original boundary corner* and only INSERT new
    points along existing straight edges. Uniform linear interpolation of a
    new set of arc-length stations cuts polygon corners and artificially
    adds/removes burned area even with no propagation.
    """
    from shapely.geometry.polygon import orient

    result = []
    min_count = max(16, min(n_vertices // 2, 100))
    max_count = max(20000, 10 * n_vertices)
    for part in _geometry_polygons(geom):
        part = orient(part, sign=1.0)
        for boundary in (part.exterior, *part.interiors):
            original = np.asarray(boundary.coords[:-1], dtype=float)
            edges = np.roll(original, -1, axis=0) - original
            lengths = np.linalg.norm(edges, axis=1)
            length = float(lengths.sum())
            if length <= 1e-10:
                continue
            effective_spacing = min(spacing, length / min_count)
            samples = []
            for p0, edge, edge_length in zip(original, edges, lengths):
                nseg = max(1, int(np.ceil(edge_length / effective_spacing)))
                for j in range(nseg):
                    samples.append(p0 + (j / nseg) * edge)
            if len(samples) > max_count:
                raise MarkerTopologyError(
                    f'Topology needs {len(samples)} markers, limit={max_count}. '
                    'Increase limit or use a less fragmented landscape.'
                )
            coords = np.asarray(samples, dtype=float)
            result.append((coords[:, 0], coords[:, 1]))
    if not result:
        raise MarkerTopologyError('Topology reconstruction produced no marker rings')
    return result


def _swept_polygon_parts(geometry):
    """Extract polygonal components, preserving valid holes."""
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        yield geometry

    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for component in geometry.geoms:
            yield from _swept_polygon_parts(component)


def _safe_swept_union(polys):
    """Union swept-area polygons with geometry validation.

    Try the configured fixed-precision overlay first.
    If GEOS fails, retry at double precision.

    Never silently replace a failed union with an approximate convex hull or
    bounding polygon.
    """
    import shapely
    from shapely.errors import GEOSException
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity

    cleaned = []
    repaired_count = 0

    for i, geometry in enumerate(polys):
        if geometry is None or geometry.is_empty:
            continue

        if not geometry.is_valid:
            repaired_count += 1
            logger.warning(
                "Invalid swept polygon %d: %s",
                i,
                explain_validity(geometry),
            )
            geometry = shapely.make_valid(geometry)

        parts = list(_swept_polygon_parts(geometry))

        if not parts:
            logger.warning(
                "Swept geometry %d has no polygonal area",
                i,
            )
            continue

        for part in parts:
            if not part.is_empty and part.area > 0:
                if not part.is_valid:
                    raise ValueError(
                        f"Swept polygon {i} remains invalid: "
                        f"{explain_validity(part)}"
                    )
                cleaned.append(part)

    if not cleaned:
        return Polygon()

    if repaired_count:
        logger.warning(
            "Repaired %d invalid swept geometries",
            repaired_count,
        )

    grid_size = float(_marker_topology.OVERLAY_GRID_M)

    try:
        result = shapely.union_all(
            cleaned,
            grid_size=grid_size,
        )
    except GEOSException:
        logger.warning(
            "Fixed-precision swept union failed "
            "(grid_size=%g). Retrying at double precision.",
            grid_size,
            exc_info=True,
        )
        # Do not silently suppress a second GEOS failure.
        result = shapely.union_all(
            cleaned,
            grid_size=0,
        )

    if not result.is_valid:
        raise ValueError(
            "Swept union produced invalid geometry: "
            + explain_validity(result)
        )

    if result.geom_type not in ("Polygon", "MultiPolygon"):
        if not result.is_empty:
            raise ValueError(
                "Swept union produced unexpected geometry: "
                + result.geom_type
            )

    return result


def _swept_region(old_rings, new_rings, barrier_interior=None):
    """Union of the area swept by every marker edge during one substep.

    Barber et al. (2009) distinguish an inverted self-crossing loop (burned on
    both sides, to be clipped) from a genuine unburned island (never crossed
    by the front). Winding numbers alone cannot: both have winding 0. Any land
    a marker edge actually passed over is burned, so unioning the swept area
    into the candidate burn fills artifact loops but leaves true islands.

    Each edge (o_i, o_{i+1}) -> (n_i, n_{i+1}) sweeps a quadrilateral, split
    into two triangles so a twisted (bow-tie) quad is still valid geometry.
    ``old_rings`` and ``new_rings`` must correspond marker-for-marker.

    With ``barrier_interior``, every triangle reaching into non-fuel is
    dropped whole rather than clipped. Such a triangle comes from an edge
    that cut a barrier corner, so it cannot show that the ground behind
    the barrier burned. Real burning near the barrier is still recorded by
    the triangles of the edges that wrap the barrier.
    """
    import shapely

    tris = []
    for (ox, oy), (nx_, ny_) in zip(old_rings, new_rings):
        o = np.column_stack((ox, oy))
        n = np.column_stack((nx_, ny_))
        if o.shape != n.shape:
            raise ValueError("_swept_region needs one-to-one old/new markers")
        o1, n1 = np.roll(o, -1, axis=0), np.roll(n, -1, axis=0)
        tris.append(np.stack([o, o1, n1, o], axis=1))
        tris.append(np.stack([o, n1, n, o], axis=1))
    if not tris:
        return None
    polys = shapely.polygons(np.concatenate(tris))
    # Drop numerically degenerate triangles (stationary markers) and snap the
    # union so the swept region is always valid overlay input.
    polys = polys[shapely.area(polys) > 1e-10]
    if barrier_interior is not None and not barrier_interior.is_empty and len(polys):
        shapely.prepare(barrier_interior)
        crossing = shapely.intersects(polys, barrier_interior)
        if crossing.any():
            logger.debug("Dropped %d swept triangle(s) reaching into non-fuel",
                         int(crossing.sum()))
        polys = polys[~crossing]
    if len(polys) == 0:
        return None
    return _safe_swept_union(polys)


def _outer_hull_from_oriented_linework(
    rings, spacing, n_vertices, *, previous_geometry=None,
    swept_geometry=None,
    barrier_geometry=None, time_min=None, debug_dir="marker_topology_debug",
):
    """Compatibility wrapper around :mod:`ignacio.topology`.

    The only source of truth for experimental oriented-linework topology is
    ``ignacio.topology._outer_hull_from_oriented_linework``. Retain this wrapper
    so existing callers and all three call sites in the multiple-ignition
    simulation continue working without changing the Richards integrators.

    Returns (active_moving_rings, Polygon/MultiPolygon, diagnostic_dict).
    Exteriors are CCW; active interior rings are CW. Fixed barrier rings are
    kept in burned_geometry but are excluded from active_moving_rings.
    """
    return _marker_topology._outer_hull_from_oriented_linework(
        rings,
        spacing,
        n_vertices,
        previous_geometry=previous_geometry,
        swept_geometry=swept_geometry,
        barrier_geometry=barrier_geometry,
        time_min=time_min,
        debug_dir=debug_dir,
    )


def _simulate_multiple_ignitions(
    *, param_grid, ignition_points, dt, n_vertices, initial_radius,
    store_every, max_steps, use_markers, marker_epsilon, min_ros, is_geographic,
    advection_integrator, resample_spacing, insert_factor, delete_factor,
    smoothing_weight, smoothing_iters, redistribute_every,
    max_substeps, max_move_fraction,
    barrier_mode="off", barrier_interior=None, barrier_geometry=None,
    barrier_debug_dir="marker_barrier_debug", export_burned_geometry=False,
    topology_mode="legacy",
):
    """Evolve multiple circular ignitions, resolving fire-front topology.

    The user must explicitly pass ignition_points through simulate_fire_spread().
    This is intended for controlled numerical tests on projected metre grids.
    """
    _require_shapely()
    if not use_markers:
        raise ValueError("Multiple ignition front merging requires use_markers=True")
    if is_geographic:
        raise ValueError(
            "Multiple-ignition experiments require a projected metre-based CRS. "
            "Reproject DEM, fuel and ignition points consistently before running."
        )
    if dt <= 0 or store_every < 1 or initial_radius <= 0 or n_vertices < 8:
        raise ValueError("Require dt>0, store_every>=1, radius>0, n_vertices>=8")
    if max_substeps < 1 or not (0 < max_move_fraction <= 1):
        raise ValueError("max_substeps>=1 and max_move_fraction in (0,1] required")
    if not (0 < insert_factor and 0 < delete_factor < insert_factor):
        raise ValueError("Invalid marker insertion/deletion factors")
    if smoothing_iters < 0 or not (0 <= smoothing_weight <= 0.5):
        raise ValueError("Invalid smoothing settings")
    integrator = (advection_integrator or "euler").strip().lower()
    if integrator not in {"euler", "heun", "rk4", "euler_extrap", "implicit"}:
        raise ValueError(f"Unsupported multi-front integrator: {integrator}")
    if topology_mode not in {"legacy", "outer_hull"}:
        raise ValueError("topology_mode must be 'legacy' or 'outer_hull'")
    outer_hull = topology_mode == 'outer_hull'
    if outer_hull and param_grid.nonfuel_mask is not None and barrier_mode == 'off':
        logger.warning('barrier_mode=off: barrier crossings are repaired '
                       '(edges wrapped, swept triangles dropped) but not reported')
    if (resample_spacing is None or resample_spacing <= 0):
        raise ValueError(
            "For multi-front runs, set resample_spacing explicitly in metres "
            "(e.g., 1.0 or 2.0 on a 2 m grid)."
        )
    spacing = float(resample_spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("resample_spacing must be finite and > 0")
    points = np.asarray(ignition_points, dtype=float)
    for idx, (px, py) in enumerate(points):
        if not (param_grid.x_min + initial_radius < px < param_grid.x_max - initial_radius
                and param_grid.y_min + initial_radius < py < param_grid.y_max - initial_radius):
            raise ValueError(
                f"Ignition {idx+1} at ({px},{py}) is outside the safe grid extent."
            )

    rings = [create_initial_perimeter(float(px), float(py),
                                      float(initial_radius), int(n_vertices))
             for px, py in points]
    history = FirePerimeterHistory(
        perimeters=[], times=[], fronts_by_time=[],
        holes_by_time=[] if outer_hull else None,
        # Exact geometry is required for correct hole-aware area accounting,
        # even if the caller did not explicitly request geometry export.
        burned_geometries=[] if (outer_hull or export_burned_geometry) else None)
    for ring in rings:
        if barrier_mode == "raise":
            validate_marker_boundary(*ring, barrier_interior,
                                     time_min=0.0, phase="initial_front",
                                     diagnostic_dir=barrier_debug_dir)
    if outer_hull:
        rings, burned_geometry, topo_info = _outer_hull_from_oriented_linework(
            rings, spacing, n_vertices, barrier_geometry=barrier_geometry,
            time_min=0.0, debug_dir=barrier_debug_dir)
        if topo_info['fronts_after'] < len(points):
            history.merger_events.append({
                'time_min': 0.0, 'fronts_before': len(points),
                'fronts_after': topo_info['fronts_after'],
            })
        burned_geometry = _clip_burned_geometry_to_fuel(
            burned_geometry, barrier_geometry, phase='initial_burn', time_min=0.0,
        )
        if barrier_mode == "raise":
            for xx, yy in rings:
                validate_marker_boundary(
                    xx, yy, barrier_interior, time_min=0.0,
                    phase="initial_reconstructed_front",
                    diagnostic_dir=barrier_debug_dir,
                )
        _append_front_snapshot(history, rings, 0.0,
                               burned_geometry=burned_geometry, outer_hull=True)
    else:
        rings, merged, _ = _merge_intersecting_fronts(rings, spacing, n_vertices)
        if merged:
            history.merger_events.append({
                "time_min": 0.0, "fronts_before": len(points), "fronts_after": len(rings),
            })
        _append_front_snapshot(history, rings, 0.0, barrier_geometry)

    n_steps = param_grid.nt if max_steps is None else min(int(max_steps), param_grid.nt)
    logger.info("Multi-ignition marker run: %d ignitions, %d steps, dt=%.4f min, %s",
                len(points), n_steps, dt, integrator)

    for step in range(n_steps):
        # Find max velocity across all active rings for topology-safe substeps.
        vmax = 0.0
        for x, y in rings:
            ros, bros, fros, raz = param_grid.sample_at(step, x, y)
            if np.any(ros >= min_ros):
                a, b, c, theta = ros_to_ellipse_params(ros, bros, fros, raz)
                vx, vy = richards_velocity(x, y, a, b, c, theta)
                vmax = max(vmax, float(np.max(np.hypot(vx, vy))))
        if vmax < min_ros:
            logger.info("Fire extinguished at step %d", step)
            break

        # This threshold limits distance travelled by a marker before checking
        # whether fronts have crossed. NEVER silently cap needed substeps.
        required = max(1, int(np.ceil(vmax * dt / (max_move_fraction * spacing))))
        if required > max_substeps:
            raise ValueError(
                f"Need {required} topology substeps but max_substeps={max_substeps} "
                f"at step {step}. Reduce dt or raise max_substeps."
            )
        sub_dt = dt / required

        for sub in range(required):
            from shapely.ops import unary_union
            previous_geometry = (burned_geometry if outer_hull else
                                 unary_union([_validated_polygon(r) for r in rings]))
            new_rings = []
            for x, y in rings:
                # Legacy rings are CCW exterior-only; outer_hull also evolves
                # clockwise interior boundaries of unburned islands. The
                # common velocity formulation uses right-hand normals, so
                # winding direction determines propagation toward unburned fuel.
                def vel_fn(xq, yq):
                    ros, bros, fros, raz = param_grid.sample_at(step, xq, yq)
                    a, b, c, theta = ros_to_ellipse_params(ros, bros, fros, raz)
                    vx, vy = richards_velocity(xq, yq, a, b, c, theta)
                    mask = np.asarray(ros) >= min_ros
                    if param_grid.nonfuel_mask is not None:
                        mask &= param_grid.is_burnable_at(xq, yq)
                    return np.where(mask, vx, 0.0), np.where(mask, vy, 0.0)

                if integrator == "euler":
                    vx, vy = vel_fn(x, y)
                    xn, yn = x + sub_dt * vx, y + sub_dt * vy
                elif integrator == "heun":
                    xn, yn = _heun_step_xy(x, y, sub_dt, vel_fn)
                elif integrator == "rk4":
                    xn, yn = _rk4_step_xy(x, y, sub_dt, vel_fn)
                elif integrator == "euler_extrap":
                    xn, yn, _ = _euler_extrapolation_step_xy(
                        x, y, sub_dt, vel_fn, n_base=1)
                else:
                    xn, yn = _implicit_newton_step_xy(x, y, sub_dt, vel_fn)
                if np.isfinite(xn).all() and np.isfinite(yn).all():
                    xn, yn = _apply_marker_barrier_limit(param_grid, x, y, xn, yn,
                                                         barrier_mode=barrier_mode,
                                                         barrier_interior=barrier_interior)
                if barrier_mode == "raise" and np.isfinite(xn).all() and np.isfinite(yn).all():
                    when = step * dt + (sub + 1) * sub_dt
                    # Edges are wrapped around barriers below and their
                    # boundaries validated after that.
                    validate_marker_barrier_crossings(
                        x, y, xn, yn, barrier_interior,
                        time_min=when, phase="multi_" + integrator,
                        diagnostic_dir=barrier_debug_dir, check_boundary=False)
                new_rings.append((xn, yn))

            # Do not let the legacy grid sampler silently clamp fire markers
            # at the edges of the computational domain.
            for xx, yy in new_rings:
                if (np.min(xx) < param_grid.x_min or np.max(xx) > param_grid.x_max
                        or np.min(yy) < param_grid.y_min or np.max(yy) > param_grid.y_max):
                    raise ValueError(
                        "A fire front reached the parameter-grid boundary. "
                        "Use a larger domain or end the simulation earlier."
                    )
            when = step * dt + (sub + 1) * sub_dt
            # The swept region needs one-to-one markers, so build it before
            # edge wrapping changes the marker count. `rings` is still the
            # pre-step set. Triangles reaching into non-fuel are dropped.
            swept = (_swept_region(rings, new_rings, barrier_interior)
                     if outer_hull else None)
            new_rings, wrap_stats = _wrap_rings_around_barriers(
                new_rings, barrier_geometry, barrier_interior)
            if wrap_stats["wrapped_edges"] or wrap_stats["moved_vertices"]:
                history.topology_repairs.append({
                    "time_min": float(when), "reason": "barrier_edge_wrap",
                    **wrap_stats,
                })
            # Track numerical topology repairs separately from true mergers.
            from shapely.geometry import Polygon
            if any(not Polygon(np.column_stack(r)).is_valid for r in new_rings):
                history.topology_repairs.append({"time_min": float(when),
                                                 "reason": "self_intersection"})
            before = len([r for r in new_rings if _ring_signed_area(*r) > 0])
            if barrier_mode == "raise":
                # Marker paths may be valid while edges linking successive
                # vertices cut across an obstacle. Do not attempt to repair
                # such crossings by silently masking the output geometry.
                for xx, yy in new_rings:
                    validate_marker_boundary(
                        xx, yy, barrier_interior,
                        time_min=when, phase="pre_topology_edges",
                        diagnostic_dir=barrier_debug_dir,
                    )
            if outer_hull:
                swept = _clip_burned_geometry_to_fuel(
                    swept, barrier_geometry, phase="swept_region", time_min=when,
                )
                rings, burned_geometry, topo_info = _outer_hull_from_oriented_linework(
                    new_rings, spacing, n_vertices,
                    previous_geometry=previous_geometry,
                    swept_geometry=swept,
                    barrier_geometry=barrier_geometry,
                    time_min=when, debug_dir=barrier_debug_dir)
                burned_geometry = _clip_burned_geometry_to_fuel(
                    burned_geometry, barrier_geometry,
                    phase="reconstructed_burn", time_min=when,
                )
                if barrier_mode == "raise":
                    _validate_burned_geometry_connectivity(
                        previous_geometry, burned_geometry, time_min=when,
                    )
                did_merge = topo_info['fronts_after'] < before
                if topo_info['topology_changed']:
                    history.topology_repairs.append({
                        'time_min': float(when), 'reason': 'oriented_linework',
                        **topo_info,
                    })
            else:
                rings, did_merge, _ = _merge_intersecting_fronts(
                    new_rings, spacing, n_vertices, previous_geometry=previous_geometry)
            if barrier_mode == "raise":
                for xx, yy in rings:
                    validate_marker_boundary(
                        xx, yy, barrier_interior,
                        time_min=step * dt + (sub + 1) * sub_dt,
                        phase="multi_union", diagnostic_dir=barrier_debug_dir)
            if did_merge:
                when = step * dt + (sub + 1) * sub_dt
                event = {"time_min": float(when),
                         "fronts_before": before,
                         "fronts_after": (topo_info['fronts_after'] if outer_hull else len(rings))}
                history.merger_events.append(event)
                logger.info("Fire fronts merged at %.4f min: %d -> %d",
                             when, before, event['fronts_after'])

        if (redistribute_every and int(redistribute_every) > 0
                and (step + 1) % int(redistribute_every) == 0):
            updated_rings = []
            for x, y in rings:
                x, y = _insert_delete_markers_by_spacing(
                    x, y, spacing=spacing,
                    insert_factor=insert_factor,
                    delete_factor=delete_factor,
                    max_n=max(20000, int(n_vertices * 10)),
                    min_n=max(16, min(n_vertices // 2, 100)),
                    barrier_geometry=barrier_geometry,
                    barrier_interior=barrier_interior,
                )
                if smoothing_weight > 0 and smoothing_iters > 0:
                    xs0, ys0 = x.copy(), y.copy()
                    x, y = _smooth_ring_local_xy(
                        x, y, w=smoothing_weight, n_iter=int(smoothing_iters))
                    x, y = _apply_marker_barrier_limit(param_grid, xs0, ys0, x, y,
                                                     barrier_mode=barrier_mode,
                                                     barrier_interior=barrier_interior)
                # Smoothing or a relocated midpoint can leave an edge across
                # a barrier corner; wrap it again.
                x, y, _ = _wrap_ring_around_barriers(
                    x, y, barrier_geometry, barrier_interior)
                if barrier_mode == "raise":
                    validate_marker_boundary(
                        x, y, barrier_interior,
                        time_min=(step+1)*dt, phase="multi_redistribution",
                        diagnostic_dir=barrier_debug_dir)
                updated_rings.append((x, y))
            # Smoothing can create an intersection; resolve again. Retain
            # the pre-smoothing burned area if topology requires repair.
            from shapely.ops import unary_union
            previous_geometry = (burned_geometry if outer_hull else
                                 unary_union([_validated_polygon(r) for r in rings]))
            from shapely.geometry import Polygon
            if any(not Polygon(np.column_stack(r)).is_valid for r in updated_rings):
                history.topology_repairs.append({"time_min": float((step+1)*dt),
                                                 "reason": "after_redistribution"})
            before = len([r for r in updated_rings if _ring_signed_area(*r) > 0])
            if outer_hull:
                rings, burned_geometry, topo_info = _outer_hull_from_oriented_linework(
                    updated_rings, spacing, n_vertices,
                    previous_geometry=previous_geometry,
                    barrier_geometry=barrier_geometry,
                    time_min=(step+1)*dt, debug_dir=barrier_debug_dir)
                burned_geometry = _clip_burned_geometry_to_fuel(
                    burned_geometry, barrier_geometry,
                    phase="redistributed_burn", time_min=(step+1)*dt,
                )
                if barrier_mode == "raise":
                    _validate_burned_geometry_connectivity(
                        previous_geometry, burned_geometry,
                        time_min=(step+1)*dt,
                    )
                    for xx, yy in rings:
                        validate_marker_boundary(
                            xx, yy, barrier_interior,
                            time_min=(step+1)*dt,
                            phase="redistributed_reconstructed_front",
                            diagnostic_dir=barrier_debug_dir,
                        )
                did_merge = topo_info['fronts_after'] < before
                if topo_info['topology_changed']:
                    history.topology_repairs.append({
                        'time_min': float((step + 1) * dt),
                        'reason': 'redistribution_linework', **topo_info,
                    })
            else:
                rings, did_merge, _ = _merge_intersecting_fronts(
                    updated_rings, spacing, n_vertices,
                    previous_geometry=previous_geometry)
            if did_merge:
                when = (step + 1) * dt
                history.merger_events.append({
                    "time_min": float(when),
                    "fronts_before": before,
                    "fronts_after": (topo_info['fronts_after'] if outer_hull else len(rings))})
                logger.info("Fire fronts merged after redistribution at %.4f min", when)

        if (step + 1) % store_every == 0:
            _append_front_snapshot(
                history, rings, (step + 1) * dt, barrier_geometry,
                burned_geometry=burned_geometry if outer_hull else None,
                outer_hull=outer_hull)

    logger.info("Multiple-front simulation complete: %d snapshots, %d merges, %d repairs",
                history.n_steps, len(history.merger_events), len(history.topology_repairs))
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
