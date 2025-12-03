

#!/usr/bin/env python3
"""
ROS_Spread.py

Evolves a fire perimeter (front) through time using:

  * Richards' differential equations for elliptical spread (Huygens-based),
  * Spatially varying ROS / BROS / FROS / RAZ built from:
        - FBP time series (Bow_FBP_out.csv),
        - DEM-derived slope/aspect/topography (Bow_FBP_Grid.npz),
  * A simple marker method (turning-number based) to decide which
    perimeter vertices are ACTIVE (allowed to move) or INACTIVE (frozen)
    at each time step.

This script is meant to sit *after*:
  1. Build_FBP_Grid.py   -> produces Bow_FBP_Grid.npz
  2. ROS_Math.py         -> defines FireParamGrid and ROS field construction

In short:
  DEM + FBP  --> FireParamGrid (ROS_Math)
  FireParamGrid + Richards + markers  --> evolving fire front (this script)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# We import from ROS_Math to re-use:
#   - build_param_grid_from_fbp_and_dem
#   - richards_velocity
#   - load_inputs (for FWI/FBP)
from ROS_Math import (
    load_inputs,
    build_param_grid_from_fbp_and_dem,
    richards_velocity,
    FBP_GRID_NPZ,
    FWI_CSV,
    FBP_CSV,
)

# ============================================================
# 1. Marker method: determine ACTIVE vertices via turning number
# ============================================================

def compute_turning_number_for_point(px, py, x, y):
    """
    Approximate the turning (winding) number of the polygonal front
    around a point (px, py).

    Turning number is defined as:
        (total angle swept by the curve around the point) / (2*pi)

    Here we discretize the curve as segments between (x[i], y[i]) and
    (x[i+1], y[i+1]) (with wrap-around), and sum the change in angle
    of the vector from the point to consecutive vertices.

    Notes:
      * This is a standard winding number calculation. For simple,
        non-self-intersecting CCW polygons, interior points yield 1,
        exterior points yield 0.
      * When the point lies exactly on the curve (as our vertices do),
        the winding number is not strictly defined. In practice, we
        slightly offset the evaluation point outward along a local
        normal direction to avoid degeneracy (see compute_active_vertices).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Vectors from point to each vertex
    vx = x - px
    vy = y - py

    # Angles to each vertex
    angles = np.arctan2(vy, vx)

    # Angle differences between consecutive vertices
    dtheta = np.diff(angles, append=angles[0])

    # Normalize each angle difference into (-pi, pi] for robustness
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi

    total_angle = np.sum(dtheta)
    turning_number = int(np.round(total_angle / (2.0 * np.pi)))

    return turning_number


def compute_active_vertices(x, y, eps=1e-3):
    """
    Determine which vertices of the fire perimeter are 'active' using
    a simplified turning-number approach.

    Concept:
      * A vertex should be active if it lies on the *outer* fire front,
        i.e. if the curve does NOT wrap around it (turning number ~ 0).
      * Vertices that are inside loops (e.g., from cross-overs or figure-8
        shapes) have non-zero turning numbers and are declared inactive.

    Implementation details:
      * For each vertex i, we:
          - Estimate the local tangent and a rough outward normal.
          - Offset the evaluation point slightly (eps) along that normal.
          - Compute the turning number around that offset point.
          - If |turning_number| == 0 -> active; else -> inactive.
      * This is not a perfect clone of Bryce–Richards or Scan Line, but
        it captures the main idea: distinguish exterior from interior
        regions via turning number.

    Parameters
    ----------
    x, y : 1D arrays
        Coordinates of perimeter vertices in counter-clockwise order.
    eps : float
        Small offset distance used to move the test point slightly off
        the curve along a normal direction to avoid degeneracy.

    Returns
    -------
    active : 1D boolean array
        True for active vertices, False for inactive (frozen) ones.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    # Compute local tangents via central difference
    x_f = np.roll(x, -1)
    x_b = np.roll(x, 1)
    y_f = np.roll(y, -1)
    y_b = np.roll(y, 1)

    tx = x_f - x_b
    ty = y_f - y_b

    # For a CCW-oriented front, the outward normal can be approximated
    # as pointing to the "right" of the tangent, i.e. (nx, ny) = (ty, -tx)
    nx = ty
    ny = -tx

    # Normalize normals; avoid division by zero
    norm = np.hypot(nx, ny)
    norm[norm == 0.0] = 1.0
    nx /= norm
    ny /= norm

    active = np.zeros(n, dtype=bool)

    for i in range(n):
        # Offset point slightly outward along normal
        px = x[i] + eps * nx[i]
        py = y[i] + eps * ny[i]

        tn = compute_turning_number_for_point(px, py, x, y)

        # Exterior points have turning number 0
        active[i] = (tn == 0)

    return active


# ============================================================
# 2. Perimeter evolution with Richards + active/inactive vertices
# ============================================================

def simulate_fire_front_with_markers(
    param_grid,
    dt=1.0,
    n_points=200,
    store_every=1,
    initial_radius=0.5,
):
    """
    Evolve the fire front using a FireParamGrid that gives:
        ROS(x,y,t), BROS(x,y,t), FROS(x,y,t), RAZ(x,y,t),

    but with an additional marker step to determine which vertices
    are ACTIVE (allowed to move) or INACTIVE (frozen) each step,
    using a turning-number test.

    Parameters
    ----------
    param_grid : FireParamGrid
        Precomputed gridded ROS/BROS/FROS/RAZ over the landscape.
        (Typically built by ROS_Math.build_param_grid_from_fbp_and_dem)
    dt : float
        Time step (same units as ROS).
    n_points : int
        Number of vertices in the polygonal fire front.
    store_every : int
        Store the perimeter every 'store_every' time steps.
    initial_radius : float
        Radius of the initial circular ignition.

    Returns
    -------
    history : list of (x, y)
        Stored perimeters through time.
    """
    # Initial circular front
    s = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = initial_radius * np.cos(s)
    y = initial_radius * np.sin(s)

    history = [(x.copy(), y.copy())]

    nt = param_grid.nt

    for j in range(nt):
        # 1) Interpolate ROS/BROS/FROS/RAZ at each vertex
        ros_i, bros_i, fros_i, raz_i = param_grid.sample_at(j, x, y)

        # 2) Convert to a,b,c,theta per vertex (arrays)
        a = 0.5 * (ros_i + bros_i)
        c = 0.5 * (ros_i - bros_i)
        b = fros_i
        theta = raz_i  # radians

        # 3) Compute velocities via Richards PDE
        xt, yt = richards_velocity(x, y, a, b, c, theta)

        # 4) Determine which vertices are active
        active = compute_active_vertices(x, y)

        # 5) Freeze inactive vertices
        xt[~active] = 0.0
        yt[~active] = 0.0

        # 6) Update front
        x = x + dt * xt
        y = y + dt * yt

        if (j + 1) % store_every == 0:
            history.append((x.copy(), y.copy()))

    return history


# ============================================================
# 3. Plotting helper
# ============================================================

def plot_history(history, title="Richards Fire Spread (with markers)"):
    plt.figure(figsize=(6, 6))
    n_frames = len(history)
    for i, (x, y) in enumerate(history):
        alpha = 0.2 + 0.8 * i / max(1, (n_frames - 1))
        plt.plot(x, y, linewidth=1.0, alpha=alpha)

    plt.scatter([0], [0], marker="*", s=80, label="Ignition")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. CLI / main entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evolve a fire perimeter using Richards' spread equations, "
            "spatially varying ROS from ROS_Math, and a turning-number "
            "based marker method for active/inactive vertices."
        )
    )
    parser.add_argument(
        "--fwi_csv",
        type=str,
        default=None,
        help="Path to FWI CSV (Bow_FWI_out.csv). If omitted, uses ROS_Math.FWI_CSV default.",
    )
    parser.add_argument(
        "--fbp_csv",
        type=str,
        default=None,
        help="Path to FBP CSV (Bow_FBP_out.csv). If omitted, uses ROS_Math.FBP_CSV default.",
    )
    parser.add_argument(
        "--grid_npz",
        type=str,
        default=FBP_GRID_NPZ,
        help="Path to FBP terrain grid NPZ (Build_FBP_Grid output).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step in same units as ROS (default: 1.0).",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=300,
        help="Number of vertices on the fire perimeter (default: 300).",
    )
    parser.add_argument(
        "--store_every",
        type=int,
        default=1,
        help="Store perimeter every N time steps (default: 1).",
    )
    parser.add_argument(
        "--initial_radius",
        type=float,
        default=0.5,
        help="Initial ignition radius (default: 0.5 in map units).",
    )
    parser.add_argument(
        "--LB",
        type=float,
        default=2.0,
        help="Length-to-breadth ratio used when constructing FROS from ROS (default: 2.0).",
    )
    parser.add_argument(
        "--backing_fraction",
        type=float,
        default=0.2,
        help="BROS / ROS ratio used when constructing BROS from ROS (default: 0.2).",
    )
    parser.add_argument(
        "--slope_factor",
        type=float,
        default=0.5,
        help=(
            "Strength of slope effect on ROS in ROS_Math.build_param_grid_from_fbp_and_dem "
            "(default: 0.5)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use CLI paths if provided; otherwise fall back to ROS_Math defaults
    df_fwi, df_fbp = load_inputs(
        fwi_path=args.fwi_csv or FWI_CSV,
        fbp_path=args.fbp_csv or FBP_CSV,
    )

    # Build the spatially varying ROS grid that includes slope and aspect effects
    param_grid = build_param_grid_from_fbp_and_dem(
        df_fbp,
        grid_npz_path=args.grid_npz,
        LB=args.LB,
        backing_fraction=args.backing_fraction,
        ros_col="ROS",
        raz_col="RAZ",
        slope_factor=args.slope_factor,
    )

    # Evolve the fire front with markers
    history = simulate_fire_front_with_markers(
        param_grid,
        dt=args.dt,
        n_points=args.n_points,
        store_every=args.store_every,
        initial_radius=args.initial_radius,
    )

    # Plot result
    plot_history(history)


if __name__ == "__main__":
    main()