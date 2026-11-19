"""IGNACIO: topology reconstruction for moving wildfire marker fronts.

This is a Barber et al. (2009)-inspired *oriented planar arrangement*, NOT a
literal implementation of their sequential crossing-resolution algorithm.

A proposed marker step supplies oriented closed rings. Outer boundaries must
be counterclockwise (CCW), and inner boundaries of still-burnable islands must
be clockwise (CW): burned land lies to the left of each directed edge.

Algorithm
---------
1. Node all input linework at crossings (Shapely unary_union).
2. Polygonize the resulting planar graph into non-overlapping faces.
3. Classify each face using the sum of signed winding numbers of the
   *original oriented rings*. A positive winding count represents burned land.
4. Union selected faces with the previously accepted burned region, so
   topology repair cannot silently unburn land.
5. Extract true outer boundaries and holes; preserve geometric corners while
   inserting optional additional markers along edges.

This module does NOT perform temporal integration, physical barrier collision
handling, or the full edge-reversal/clipping algorithm described by Barber et
al. In ambiguous high-winding configurations, compare against independent
geometric references and consider a more sophisticated front-tracing method.

Recommended coordinate reference system: projected metric CRS (metres).

Requires: numpy, shapely 2.x (grid-snapped OverlayNG unions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Literal, Sequence
from uuid import uuid4

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.errors import GEOSException
from shapely.geometry.polygon import orient
from shapely.ops import polygonize_full, unary_union

logger = logging.getLogger(__name__)

Ring = tuple[np.ndarray, np.ndarray]


# Snap-rounding grid (m) for burned-geometry overlays. Plain unary_union on
# UTM-scale coordinates can return invalid output ("nested shells") when a
# swept sliver is ~1e-19 m^2; snap-rounded OverlayNG always returns valid
# geometry. 1 um changes burned area by well under 1e-4 m^2.
OVERLAY_GRID_M = 1e-6


def _robust_union(geoms):
    """Precision-controlled polygonal union with a double-precision fallback.

    This is also used for the large set of noded fire-front faces. For invalid
    *candidate* faces, make_valid extracts all polygonal components rather
    than silently taking a convex hull or discarding interior boundaries.
    Previously accepted state is validated separately and never auto-repaired.
    """
    cleaned = []
    for geometry in geoms:
        if geometry is None or geometry.is_empty:
            continue
        if not geometry.is_valid:
            logger.warning('Repairing invalid union input: %s',
                           shapely.is_valid_reason(geometry))
            geometry = shapely.make_valid(geometry)
        cleaned.extend(part for part in _polygon_parts(geometry)
                       if part.area > 0 and not part.is_empty)

    if not cleaned:
        return Polygon()
    try:
        result = shapely.union_all(cleaned, grid_size=OVERLAY_GRID_M)
    except GEOSException:
        logger.warning('Fixed-precision topology union failed; retrying at '
                       'double precision', exc_info=True)
        result = shapely.union_all(cleaned, grid_size=0)
    if not result.is_valid:
        raise MarkerTopologyError('Topological union returned invalid geometry')
    return result


def _robust_difference(a, b):
    """Exclude obstacles without silently accepting invalid overlay output."""
    if a is None or a.is_empty:
        return Polygon()
    if b is None or b.is_empty:
        return a
    try:
        result = shapely.difference(a, b, grid_size=OVERLAY_GRID_M)
    except GEOSException:
        logger.warning('Fixed-precision topology difference failed; retrying '
                       'at double precision', exc_info=True)
        result = shapely.difference(a, b, grid_size=0)
    if not result.is_valid:
        raise MarkerTopologyError('Topological difference returned invalid geometry')
    return result


def _robust_intersection(a, b):
    """Valid polygon intersection with the same precision fallback policy."""
    if a is None or a.is_empty or b is None or b.is_empty:
        return Polygon()
    try:
        result = shapely.intersection(a, b, grid_size=OVERLAY_GRID_M)
    except GEOSException:
        logger.warning('Fixed-precision topology intersection failed; '
                       'retrying at double precision', exc_info=True)
        result = shapely.intersection(a, b, grid_size=0)
    if not result.is_valid:
        raise MarkerTopologyError('Topological intersection returned invalid geometry')
    return result


def _barrier_overlap_area(geometry, barrier_geometry) -> float:
    """Return area, not line contact, in prescribed non-burnable polygons."""
    if (geometry is None or geometry.is_empty or
            barrier_geometry is None or barrier_geometry.is_empty):
        return 0.0
    return float(_robust_intersection(geometry, barrier_geometry).area)


def _check_barrier_exclusion(geometry, barrier_geometry, *, phase: str):
    """Refuse to return a reconstructed burned region containing non-fuel.

    The small area tolerance addresses micrometre-grid roundoff at adjoining
    edges; boundary contact itself is permitted. The check uses the complete
    internal polygon, not an exported/masked GeoPackage layer.
    """
    if barrier_geometry is None or barrier_geometry.is_empty:
        return
    overlap = _barrier_overlap_area(geometry, barrier_geometry)
    # At 1e-6 m snap precision, boundary round-off may enclose very thin
    # ribbons. Report overlap and refuse physically meaningful penetration.
    tolerance = max(1e-8, 1e-9 * float(geometry.area),
                    OVERLAY_GRID_M * float(geometry.length))
    if overlap > tolerance:
        raise MarkerTopologyError(
            f'{phase}: reconstructed burn overlaps non-fuel by '
            f'{overlap:.8g} m2 (tolerance {tolerance:.8g} m2)'
        )


def _check_reachable_components(previous_geometry, candidate, *,
                                tolerance_m2: float = 1e-8) -> int:
    """Reject newly disconnected burn islands without an existing fire seed.

    When earlier burned area exists, every new burned component must share
    area with at least one old component. This is a necessary, not sufficient,
    propagation test: a fast jump around a barrier within a single *connected*
    candidate still requires swept-path checks in spread.py.
    Initial multi-ignitions are exempt because there is no earlier geometry.
    """
    if previous_geometry is None or previous_geometry.is_empty:
        return 0
    disconnected = 0
    for part in _polygon_parts(candidate):
        shared = _robust_intersection(part, previous_geometry)
        # A purely point-like or edge-only contact does not demonstrate
        # a physically connected area-growing front.
        if shared.area <= tolerance_m2:
            disconnected += 1
    if disconnected:
        raise MarkerTopologyError(
            f'{disconnected} newly burned component(s) have no area overlap '
            'with previously burned ground; possible barrier jump'
        )
    return 0


class MarkerTopologyError(ValueError):
    """Topology cannot be reconstructed reliably from the supplied linework."""


@dataclass
class TopologyBoundary:
    x: np.ndarray
    y: np.ndarray
    role: Literal["exterior", "hole", "fixed_barrier"]
    component_id: int
    active: bool = True

    def as_ring(self) -> Ring:
        return self.x.copy(), self.y.copy()


@dataclass
class TopologyResult:
    burned_geometry: object
    boundaries: list[TopologyBoundary]
    info: dict = field(default_factory=dict)

    @property
    def active_rings(self) -> list[Ring]:
        """Moving marker rings; fixed barrier boundaries are excluded."""
        return [boundary.as_ring() for boundary in self.boundaries
                if boundary.active]

    @property
    def exterior_rings(self) -> list[Ring]:
        return [boundary.as_ring() for boundary in self.boundaries
                if boundary.role == "exterior"]

    @property
    def interior_rings(self) -> list[Ring]:
        return [boundary.as_ring() for boundary in self.boundaries
                if boundary.role == "hole"]


def signed_area(x: np.ndarray, y: np.ndarray) -> float:
    """Signed ring area; positive for CCW in a metric XY coordinate system."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise MarkerTopologyError("A ring needs >=3 corresponding XY vertices")
    # Subtract one origin to reduce cancellation with large UTM coordinates.
    xx, yy = x - x[0], y - y[0]
    return float(0.5 * np.sum(xx * np.roll(yy, -1) -
                              yy * np.roll(xx, -1)))


def _clean_ring(ring: Ring) -> Ring:
    if len(ring) != 2:
        raise MarkerTopologyError("Each ring must contain (x, y) arrays")
    x, y = (np.asarray(a, dtype=float) for a in ring)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise MarkerTopologyError("Ring x and y must be matching 1-D arrays")
    if len(x) > 1 and x[0] == x[-1] and y[0] == y[-1]:
        x, y = x[:-1], y[:-1]
    if len(x) < 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise MarkerTopologyError("Ring has <3 vertices or non-finite coordinates")
    if np.unique(np.column_stack((x, y)), axis=0).shape[0] < 3:
        raise MarkerTopologyError("Ring contains fewer than three distinct points")
    return x.copy(), y.copy()


def _remove_spikes(ring: Ring, spacing: float, *,
                   max_turn_deg: float = 170.0,
                   max_leg_fraction: float = 0.1) -> tuple[Ring, int, float]:
    """Remove zero-area fold-backs ("spikes") from a marker ring.

    A spike is a marker where the ring turns back on itself by more than
    ``max_turn_deg`` and both adjoining edges are shorter than
    ``max_leg_fraction * spacing``. It arises when markers on
    micrometre-long edges get garbage tangents and step backwards. The
    out-and-back linework encloses no area, but polygonize_full reports it
    as dangling or cut linework and reconstruction refuses the step.

    Removing a spike vertex deletes a triangle of area at most
    0.5 * l1 * l2 * sin(180 deg - max_turn_deg), i.e. under 4e-3 m^2 for
    2 m spacing, and previously burned ground is restored by the union with
    previous_geometry anyway. Returns (ring, removed_count, abs_area_change).
    """
    x, y = (np.asarray(a, dtype=float).copy() for a in ring)
    max_leg = max_leg_fraction * float(spacing)
    cos_limit = np.cos(np.radians(max_turn_deg))
    area0 = signed_area(x, y)
    removed = 0
    while len(x) > 4:
        ax, ay = x - np.roll(x, 1), y - np.roll(y, 1)
        bx, by = np.roll(x, -1) - x, np.roll(y, -1) - y
        la, lb = np.hypot(ax, ay), np.hypot(bx, by)
        cos_turn = (ax * bx + ay * by) / np.maximum(la * lb, 1e-300)
        spike = (cos_turn < cos_limit) & (la < max_leg) & (lb < max_leg)
        if not spike.any():
            break
        # Remove at most every other vertex of a run of spikes per pass,
        # so neighbours are re-evaluated against the new geometry.
        idx = np.flatnonzero(spike)
        drop = [idx[0]]
        for i in idx[1:]:
            if i - drop[-1] > 1:
                drop.append(i)
        if len(drop) > 1 and drop[0] == 0 and drop[-1] == len(x) - 1:
            drop.pop()  # ring wraps: first and last are neighbours
        if len(x) - len(drop) < 4:
            break
        keep = np.ones(len(x), dtype=bool)
        keep[drop] = False
        x, y = x[keep], y[keep]
        removed += len(drop)
    return (x, y), removed, abs(signed_area(x, y) - area0)


def winding_number(px: float, py: float, ring: Ring) -> int:
    """Signed winding count around a point; CCW = +1, CW = -1.

    Only call with points strictly away from the input linework (e.g. the
    interior representative point of a planar face).
    """
    x, y = ring
    angles = np.arctan2(y - py, x - px)
    delta = np.diff(angles, append=angles[0])
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return int(np.rint(float(np.sum(delta)) / (2.0 * np.pi)))


def _polygon_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        parts: list[Polygon] = []
        for item in geom.geoms:
            parts.extend(_polygon_parts(item))
        return parts
    return []


def _validate_polygonal_geometry(geom, name: str) -> None:
    if geom is None:
        return
    if not geom.is_valid:
        raise MarkerTopologyError(f"{name} is invalid; refusing implicit repair")
    if not geom.is_empty and not _polygon_parts(geom):
        raise MarkerTopologyError(f"{name} has no polygonal components")


def _densify_boundary(boundary, spacing: float, min_vertices: int,
                      max_vertices: int) -> Ring:
    """Insert points along existing straight edges without moving corners."""
    xy = np.asarray(boundary.coords[:-1], dtype=float)
    if xy.shape[0] < 3:
        raise MarkerTopologyError("Extracted boundary has fewer than 3 vertices")

    edges = np.roll(xy, -1, axis=0) - xy
    lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 0:
        raise MarkerTopologyError("Extracted boundary has no length")

    effective_spacing = min(spacing, perimeter / max(min_vertices, 3))
    counts = np.maximum(1, np.ceil(lengths / effective_spacing).astype(int))
    if int(counts.sum()) > max_vertices:
        raise MarkerTopologyError(
            f"Boundary requires {int(counts.sum())} markers, "
            f"limit={max_vertices}. Increase the limit or refine more locally."
        )

    points = []
    for start, edge, count in zip(xy, edges, counts):
        for j in range(int(count)):
            points.append(start + (j / count) * edge)
    result = np.asarray(points, dtype=float)
    return result[:, 0], result[:, 1]


def _save_diagnostic(rings: Sequence[Ring], debug_dir: str,
                     message: str, *, time_min: float | None,
                     linework=None, previous_geometry=None,
                     extra_geometries=None) -> Path:
    folder = Path(debug_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"topology_{uuid4().hex[:10]}"
    if linework is not None:
        (folder / f"{stem}_noded.wkt").write_text(linework.wkt, encoding="utf-8")
    if previous_geometry is not None:
        (folder / f"{stem}_previous.wkt").write_text(
            previous_geometry.wkt, encoding="utf-8")
    for name, geom in (extra_geometries or {}).items():
        if geom is not None:
            (folder / f"{stem}_{name}.wkt").write_text(geom.wkt, encoding="utf-8")
    for i, (x, y) in enumerate(rings):
        xy = np.column_stack((x, y))
        (folder / f"{stem}_input_{i}.wkt").write_text(
            LineString(np.vstack((xy, xy[0]))).wkt, encoding="utf-8")
    (folder / f"{stem}_details.json").write_text(
        json.dumps({"time_min": time_min, "error": message}, indent=2),
        encoding="utf-8")
    return folder / f"{stem}_details.json"


def reconstruct_front_topology(
    rings: Sequence[Ring],
    spacing: float,
    n_vertices: int = 128,
    *,
    previous_geometry=None,
    swept_geometry=None,
    barrier_geometry=None,
    time_min: float | None = None,
    debug_dir: str = "marker_topology_debug",
    max_vertices: int = 20000,
    min_hole_area: float | None = None,
) -> TopologyResult:
    """Classify and reconstruct fire-front topology from directed linework.

    Parameters
    ----------
    rings : sequence of (x,y) arrays
        Oriented moving marker boundaries. Exteriors CCW; interior boundaries
        surrounding burnable islands CW. Do not pass stationary barrier rings
        as moving fronts.
    spacing : float
        Approximate maximum edge length between reconstructed markers (m).
    n_vertices : int
        Baseline vertex count for minimum boundary sampling.
    previous_geometry : Polygon | MultiPolygon | None
        The last ACCEPTED burned geometry. It is unioned with candidate burn to
        prevent unburning caused by numerical topology repair.
    swept_geometry : Polygon | MultiPolygon | None
        Area swept by the marker edges during this step. Faces it covers are
        burned regardless of winding number, which clips inverted merger-cusp
        loops (Barber et al. 2009, Fig. 9) while keeping true unburned
        islands (Fig. 10). Omit when marker identity is not preserved.
    barrier_geometry : Polygon | MultiPolygon | None
        Prescribed non-burnable mask used to clip the INTERNAL candidate fire
        geometry, not only exported boundaries. Clipped exterior boundaries
        are rebuilt along obstacle edges without crossing obstacle interiors.
        Holes entirely covered by barriers are fixed and excluded from moving
        marker rings. Mixed barrier/burnable holes are APPROXIMATED as
        ordinary active holes: spread.py's barrier limiters hold the markers
        on barrier faces, so only the fuel edge burns inwards. A
        segment-level solver with frozen face segments would be exact.
        Marker path and pre-topology segment constraints remain in spread.py.
    min_hole_area : float | None
        Unburned islands smaller than this (m^2) are treated as burned out:
        tiny CW input rings are dropped and tiny output holes are filled.
        Default ``spacing**2``. A pocket left at a merger neck shrinks every
        step while keeping its marker count, until its markers fold over each
        other and the noded graph can no longer be polygonized.

    Returns
    -------
    TopologyResult
        .burned_geometry: accurate hole-bearing Polygon/MultiPolygon
        .boundaries: marked exterior, active hole, fixed barrier boundaries
        .active_rings: (x,y) tuples suitable for the Richards integrator
        .info: diagnostic event counts and areas

    Note
    ----
    Winding-based face classification is not equivalent to Barber et al.'s
    complete directed edge-reversal/clipping algorithm. A self-crossing ring
    may contain an ambiguous negative-winding lobe; validate against known
    burned history or independent truth before relying on the classification.
    """
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and > 0 metres")
    if n_vertices < 8 or max_vertices < 8:
        raise ValueError("n_vertices and max_vertices must both be >= 8")
    if not rings:
        raise MarkerTopologyError("At least one fire-front ring is required")
    _validate_polygonal_geometry(previous_geometry, "previous_geometry")
    _validate_polygonal_geometry(barrier_geometry, "barrier_geometry")
    _validate_polygonal_geometry(swept_geometry, "swept_geometry")

    if min_hole_area is None:
        min_hole_area = float(spacing) ** 2
    clean = []
    pruned_input_holes = 0
    pruned_hole_area = 0.0
    removed_spikes = 0
    spike_area_change = 0.0
    for ring in rings:
        ring, n_spikes, spike_area = _remove_spikes(_clean_ring(ring), spacing)
        ring = _clean_ring(ring)
        removed_spikes += n_spikes
        spike_area_change += spike_area
        area = signed_area(*ring)
        if area < 0 and -area < min_hole_area:
            # A collapsing unburned pocket: let the enclosing exterior burn it.
            pruned_input_holes += 1
            pruned_hole_area += -area
            continue
        clean.append(ring)
    if not clean:
        raise MarkerTopologyError("All input rings were pruned as tiny holes")
    lines = [LineString(np.vstack((np.column_stack((x, y)), [x[0], y[0]])))
             for x, y in clean]
    if any(line.length <= 0 for line in lines):
        raise MarkerTopologyError("Zero-length front linework")

    n_self_cross = sum(not line.is_simple for line in lines)
    n_invalid = sum(not Polygon(np.column_stack(r)).is_valid for r in clean)
    n_pair_cross = sum(lines[i].intersects(lines[j])
                       for i in range(len(lines))
                       for j in range(i + 1, len(lines)))
    noded = None
    try:
        # unary_union performs intersection noding; polygonize_full produces
        # faces and reports unused linework. No buffer(0) geometry repair.
        noded = unary_union(lines)
        faces_gc, dangles, cuts, invalids = polygonize_full(noded)
        faces = list(faces_gc.geoms)
        if not faces:
            raise MarkerTopologyError("No closed faces after intersection noding")

        # Even a small dangling segment means that a candidate ring was not
        # faithfully represented in the planar arrangement.
        leftovers = float(dangles.length + cuts.length + invalids.length)
        if leftovers > 1e-8:
            raise MarkerTopologyError(
                f"Noded graph has {leftovers:.6g} m of dangling/cut/invalid "
                "linework; refusing to guess a burned region"
            )

        selected = []
        rejected_negative = 0
        for face in faces:
            sample = face.representative_point()
            w = sum(winding_number(sample.x, sample.y, ring)
                    for ring in clean)
            if w > 0:
                selected.append(face)
            elif w < 0:
                rejected_negative += 1

        if not selected:
            raise MarkerTopologyError(
                "No positively oriented burned faces. Check input ring "
                "orientation and inspect self-crossing loops."
            )

        candidate = _robust_union(selected)
        swept_added_area = 0.0
        dropped_swept_debris = 0
        if swept_geometry is not None and not swept_geometry.is_empty:
            # Barber's "burned on both sides" rule: a face the markers swept
            # across is burned even if its winding number is 0 (an inverted
            # merger-cusp loop). Genuine islands are never swept and survive.
            # Apply the obstacle mask before this union so that a swept edge
            # cannot make a barrier cell internally burned.
            admissible_swept = _robust_difference(swept_geometry, barrier_geometry)
            # The swept rule fills folded-over loops, which always touch the
            # front. A swept part touching neither the candidate nor the
            # previous burn, or with no real area, is snap-rounding debris
            # from near-degenerate triangles; keeping it creates a detached
            # "burned" speck that fails the reachability check below.
            kept_parts = []
            for part in _polygon_parts(admissible_swept):
                if part.area <= 1e-8:
                    dropped_swept_debris += 1
                elif part.intersects(candidate) or (
                        previous_geometry is not None
                        and not previous_geometry.is_empty
                        and part.intersects(previous_geometry)):
                    kept_parts.append(part)
                else:
                    dropped_swept_debris += 1
            admissible_swept = _robust_union(kept_parts)
            swept_added_area = float(_robust_difference(admissible_swept,
                                                       candidate).area)
            candidate = _robust_union([candidate, admissible_swept])
        lost_area = 0.0
        if previous_geometry is not None and not previous_geometry.is_empty:
            lost_area = float(_robust_difference(previous_geometry, candidate).area)
            candidate = _robust_union([candidate, previous_geometry])

        # Fill output holes below min_hole_area before barrier clipping, so a
        # genuine non-fuel hole is cut back out by the barrier difference.
        pruned_output_holes = 0
        filled = []
        for poly in _polygon_parts(candidate):
            keep = []
            for interior in poly.interiors:
                hole_area = Polygon(interior).area
                if hole_area < min_hole_area:
                    pruned_output_holes += 1
                    pruned_hole_area += hole_area
                else:
                    keep.append(interior)
            filled.append(Polygon(poly.exterior, keep))
        if pruned_output_holes:
            candidate = _robust_union(filled)

        masked_area = 0.0
        if barrier_geometry is not None and not barrier_geometry.is_empty:
            # A mismatch here would break the burned-area monotonicity claim.
            if previous_geometry is not None and not previous_geometry.is_empty:
                prior_inside = _barrier_overlap_area(previous_geometry,
                                                     barrier_geometry)
                # Snap-rounding to OVERLAY_GRID_M can leave slivers up to one
                # grid unit wide along barrier faces; that is not a crossing.
                snap_tol = OVERLAY_GRID_M * float(previous_geometry.length)
                if prior_inside > max(1e-8, previous_geometry.area * 1e-10, snap_tol):
                    raise MarkerTopologyError(
                        "Previously burned geometry intersects a prescribed "
                        "non-burnable barrier. Correct the accepted state first."
                    )
            masked_area = _barrier_overlap_area(candidate, barrier_geometry)
            candidate = _robust_difference(candidate, barrier_geometry)
            _check_barrier_exclusion(candidate, barrier_geometry,
                                     phase='final topology clipping')

        if candidate.is_empty or not candidate.is_valid:
            raise MarkerTopologyError("Reconstructed burned geometry is empty or invalid")
        parts = sorted(_polygon_parts(candidate),
                       key=lambda p: (-p.area, p.centroid.x, p.centroid.y))
        if not parts:
            raise MarkerTopologyError("Burned result has no polygonal area")
        # Detached specks far below marker resolution (a few-cm loop of the
        # ring at a barrier face, overlay slivers) are numerical debris, not
        # a barrier jump: a real jump leaves a component on the scale of the
        # marker spacing. Drop them; anything larger still fails below.
        dropped_fragments = 0
        if previous_geometry is not None and not previous_geometry.is_empty:
            min_fragment_area = 1e-3 * float(spacing) ** 2
            kept = []
            for part in parts:
                if (part.area < min_fragment_area and
                        _robust_intersection(part, previous_geometry).area <= 1e-8):
                    dropped_fragments += 1
                else:
                    kept.append(part)
            if dropped_fragments:
                if not kept:
                    raise MarkerTopologyError("Only sub-resolution fragments remain")
                parts = kept
                candidate = _robust_union(parts)
        # Refuse disconnected fire islands created by clipping a candidate
        # across an obstacle. Already-seeded separate ignitions remain valid.
        _check_reachable_components(previous_geometry, candidate)
        if previous_geometry is None or previous_geometry.is_empty:
            # One newly supplied exterior cannot create two fires on opposite
            # sides of a non-burnable wall. Separately supplied ignition
            # exteriors are allowed to remain disconnected.
            original_exteriors = sum(signed_area(*r) > 0 for r in clean)
            if len(parts) > original_exteriors:
                raise MarkerTopologyError(
                    'Barrier clipping split a candidate into more disconnected '
                    'fires than the supplied exterior seeds; refusing an '
                    'unseeded ignition across a barrier'
                )

        boundaries: list[TopologyBoundary] = []
        obstacle_contact_length = 0.0
        fixed_barrier_holes = 0
        mixed_barrier_holes = 0
        for part_id, part in enumerate(parts):
            # Overlay noise leaves micrometre segments that fold into cut
            # edges after the next advection; drop them, keep real corners.
            # IMPORTANT: do not simplify obstacle-adjacent geometry: even a
            # topologically valid simplification can cut across a raster cell
            # corner, so preserve the exact clipped barrier detour instead.
            touching_barrier = (barrier_geometry is not None and
                                not barrier_geometry.is_empty and
                                part.boundary.intersects(barrier_geometry))
            if not touching_barrier:
                part = shapely.simplify(part, 1e-3 * float(spacing),
                                        preserve_topology=True)
            else:
                obstacle_contact_length += float(
                    part.boundary.intersection(barrier_geometry.boundary).length
                )
            part = orient(part, sign=1.0)
            x, y = _densify_boundary(part.exterior, spacing, n_vertices,
                                     max_vertices)
            if signed_area(x, y) <= 0:
                raise MarkerTopologyError("Exterior ring orientation error")
            boundaries.append(TopologyBoundary(x, y, "exterior", part_id))
            for interior in part.interiors:
                hole_poly = Polygon(interior)
                role: Literal["hole", "fixed_barrier"] = "hole"
                active = True
                if barrier_geometry is not None and hole_poly.area > 0:
                    ratio = _barrier_overlap_area(hole_poly, barrier_geometry) / hole_poly.area
                    if ratio > 1.0 - 1e-8:
                        role, active = "fixed_barrier", False
                        fixed_barrier_holes += 1
                    elif ratio > 1e-8:
                        # Barrier plus unburned fuel (e.g. the sheltered
                        # pocket behind a square the fire has closed around).
                        # APPROXIMATION: evolve it as an ordinary active hole.
                        # Its markers on barrier faces cannot advance because
                        # spread.py's barrier limiters stop them, so only the
                        # fuel edge burns inwards. Once the pocket burns out
                        # the hole is pure barrier and becomes fixed_barrier.
                        # Those face segments are not formally frozen.
                        mixed_barrier_holes += 1
                hx, hy = _densify_boundary(interior, spacing, n_vertices,
                                           max_vertices)
                if signed_area(hx, hy) >= 0:
                    raise MarkerTopologyError("Interior ring orientation error")
                boundaries.append(TopologyBoundary(hx, hy, role, part_id,
                                                   active=active))

        actual_holes = sum(len(p.interiors) for p in parts)
        expected_active_count = len(parts) + actual_holes - fixed_barrier_holes
        geometry_changed = (
            n_self_cross > 0 or n_pair_cross > 0 or
            expected_active_count != len(clean) or
            lost_area > max(1e-8, candidate.area * 1e-10,
                            OVERLAY_GRID_M * candidate.length) or
            masked_area > 1e-8 or
            pruned_input_holes > 0 or pruned_output_holes > 0 or
            dropped_fragments > 0 or
            # Spiked input rings must not be handed back as moving markers.
            removed_spikes > 0
        )

        # If there was no topology event and no clipping, leave original
        # markers untouched. This prevents unnecessary numerical remeshing.
        # Only the public structured boundaries are resampled; the compatibility
        # wrapper below can pass the original markers back to spread.py.
        return TopologyResult(
            burned_geometry=candidate,
            boundaries=boundaries,
            info={
                "time_min": time_min,
                "input_rings": len(clean),
                "self_intersecting_rings": int(n_self_cross),
                "invalid_rings": int(n_invalid),  # previous spread.py API
                "pair_crossings": bool(n_pair_cross),
                "pair_crossing_count": int(n_pair_cross),
                "planar_faces": len(faces),
                "burned_faces": len(selected),
                "negative_winding_faces": rejected_negative,
                "fronts_after": len(parts),
                "holes_after": actual_holes,
                "fixed_barrier_holes": fixed_barrier_holes,
                "mixed_barrier_holes": mixed_barrier_holes,
                "recovered_burned_area_m2": float(lost_area),
                "swept_added_area_m2": float(swept_added_area),
                "dropped_swept_debris": int(dropped_swept_debris),
                "dropped_detached_fragments": int(dropped_fragments),
                "pruned_holes": int(pruned_input_holes + pruned_output_holes),
                "pruned_hole_area_m2": float(pruned_hole_area),
                "removed_spikes": int(removed_spikes),
                "spike_area_change_m2": float(spike_area_change),
                "masked_nonfuel_area_m2": float(masked_area),
                "barrier_contact_length_m": float(obstacle_contact_length),
                "barrier_overlap_after_m2": float(
                    _barrier_overlap_area(candidate, barrier_geometry)),
                "reachability_checked": bool(previous_geometry is not None and
                                             not previous_geometry.is_empty),
                "topology_changed": bool(geometry_changed),
            },
        )
    except (MarkerTopologyError, GEOSException) as exc:
        path = _save_diagnostic(clean, debug_dir, str(exc),
                                time_min=time_min, linework=noded,
                                previous_geometry=previous_geometry,
                                extra_geometries={"swept": swept_geometry,
                                                  "barrier": barrier_geometry})
        raise MarkerTopologyError(f"{exc} [diagnostics: {path}]") from exc


def _outer_hull_from_oriented_linework(
    rings: Sequence[Ring],
    spacing: float,
    n_vertices: int,
    *,
    previous_geometry=None,
    swept_geometry=None,
    barrier_geometry=None,
    time_min: float | None = None,
    debug_dir: str = "marker_topology_debug",
):
    """Compatibility entry point matching the revised ignacio/spread.py.

    Returns (moving_rings, full_burned_geometry, diagnostic_information).
    """
    result = reconstruct_front_topology(
        rings, spacing, n_vertices,
        previous_geometry=previous_geometry,
        swept_geometry=swept_geometry,
        barrier_geometry=barrier_geometry,
        time_min=time_min,
        debug_dir=debug_dir,
    )
    if result.info["topology_changed"]:
        moving_rings = result.active_rings
    else:
        moving_rings = [_clean_ring(r) for r in rings]
    return moving_rings, result.burned_geometry, result.info


__all__ = [
    "MarkerTopologyError", "TopologyBoundary", "TopologyResult",
    "signed_area", "winding_number", "reconstruct_front_topology",
    "_outer_hull_from_oriented_linework",
]
