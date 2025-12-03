

#!/usr/bin/env python3
"""
Build_FBP_Grid.py

Constructs a spatial grid of terrain parameters (elevation, slope, aspect)
from a DEM for use in fire behaviour and rate-of-spread calculations.

This script is intended to be run *before* ROS_Math so that ROS_Math can
use the resulting gridded topographic fields when computing ROS, BROS,
FROS and RAZ over the landscape.

Outputs:
    - A NumPy .npz file containing:
        * dem          : 2D array of elevations
        * slope_deg    : 2D array of slope (degrees from horizontal)
        * aspect_deg   : 2D array of aspect (degrees, clockwise from north)
        * x_coords     : 1D array of x coordinates for column centers
        * y_coords     : 1D array of y coordinates for row centers
        * transform    : 6-element affine transform (rasterio-style)

Example usage:
    python Build_FBP_Grid.py \
        --dem "/Users/Martyn/Desktop/PhD/Fyah/Prometheus_Old/BowatBanff - Copy/domain_Bow_at_Banff_lumped_elv.tif" \
        --out "/Users/Martyn/Desktop/PhD/Fyah/Test/Bow_FBP_Grid.npz"
"""

import argparse
import os
from typing import Tuple

import numpy as np

try:
    import rasterio
except ImportError as e:
    raise ImportError(
        "This script requires the 'rasterio' package. "
        "Install it with: pip install rasterio"
    ) from e


def compute_slope_aspect(
    dem: np.ndarray,
    transform
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and aspect from a DEM using finite differences.

    Parameters
    ----------
    dem : np.ndarray
        2D array of elevations (masked or NaN where invalid).
        Units are assumed to be meters.
    transform : affine.Affine
        Affine transform for the DEM (rasterio transform).

    Returns
    -------
    slope_deg : np.ndarray
        2D array of slope in degrees from horizontal (0 = flat).
    aspect_deg : np.ndarray
        2D array of aspect in degrees, clockwise from north (0 = north,
        90 = east, 180 = south, 270 = west). NaN where slope is undefined
        or where DEM is invalid.
    """
    # Copy and convert to float64 for numerical stability
    z = np.array(dem, dtype="float64")

    # Identify invalid cells (masked or NaN)
    if np.ma.isMaskedArray(dem):
        invalid = dem.mask | ~np.isfinite(z)
    else:
        invalid = ~np.isfinite(z)

    # Extract cell size from the affine transform.
    # For a standard north-up raster:
    #   transform = [x_origin, x_res, 0, y_origin, 0, -y_res]
    dx = transform.a
    dy = -transform.e  # transform.e is typically negative

    if dx == 0 or dy == 0:
        raise ValueError("Non-positive cell resolution detected from transform.")

    # Compute partial derivatives using central differences in the interior
    # and first differences at the borders.
    dz_dy, dz_dx = np.gradient(z, dy, dx)  # numpy uses order (rows, cols) => (y, x)

    # Slope magnitude: tan(slope) = sqrt((dz/dx)^2 + (dz/dy)^2)
    grad_mag = np.hypot(dz_dx, dz_dy)
    slope_rad = np.arctan(grad_mag)
    slope_deg = np.degrees(slope_rad)

    # Aspect: direction of steepest descent.
    # Common GIS convention:
    #   aspect = atan2(dz/dx, -dz/dy)  (radians)
    # then convert to degrees and wrap to [0, 360).
    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.mod(90.0 - aspect_deg, 360.0)  # convert to 0=N, 90=E, ...

    # Where slope is ~0 (flat), aspect is undefined -> set to NaN.
    flat = grad_mag < 1e-6
    aspect_deg[flat] = np.nan

    # Propagate invalid DEM cells
    slope_deg[invalid] = np.nan
    aspect_deg[invalid] = np.nan

    return slope_deg, aspect_deg


def compute_xy_coords(transform, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D x and y coordinate arrays for cell centers given a rasterio
    affine transform and raster dimensions.

    Parameters
    ----------
    transform : affine.Affine
        Affine transform for the raster.
    width : int
        Number of columns.
    height : int
        Number of rows.

    Returns
    -------
    x_coords : np.ndarray
        1D array (length = width) of x coordinates of cell centers.
    y_coords : np.ndarray
        1D array (length = height) of y coordinates of cell centers.
    """
    # Column indices 0..width-1, row indices 0..height-1
    cols = np.arange(width)
    rows = np.arange(height)

    # For a north-up raster:
    #   x_center = x_origin + (col + 0.5) * x_res
    #   y_center = y_origin + (row + 0.5) * y_res
    x_coords = transform.c + (cols + 0.5) * transform.a
    y_coords = transform.f + (rows + 0.5) * transform.e

    return x_coords, y_coords


def build_fbp_grid(
    dem_path: str,
    out_path: str,
    overwrite: bool = False
) -> None:
    """
    High-level wrapper to read a DEM, compute slope and aspect, and save
    the FBP grid products to an .npz file.

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF (or other rasterio-readable format).
    out_path : str
        Path to the output .npz file to write.
    overwrite : bool
        If False and the output file exists, raise an error. If True,
        overwrite existing output.
    """
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"Output file '{out_path}' already exists. "
            "Use --overwrite to replace it."
        )

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Reading DEM from: {dem_path}")
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True)
        transform = src.transform

    print("Computing slope and aspect...")
    slope_deg, aspect_deg = compute_slope_aspect(dem, transform)

    height, width = dem.shape
    x_coords, y_coords = compute_xy_coords(transform, width, height)

    # Prepare transform as a simple array for easy storage
    transform_array = np.array(
        [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        dtype="float64",
    )

    print(f"Saving FBP grid to: {out_path}")
    np.savez(
        out_path,
        dem=np.array(dem, dtype="float64"),
        slope_deg=slope_deg,
        aspect_deg=aspect_deg,
        x_coords=x_coords,
        y_coords=y_coords,
        transform=transform_array,
    )

    print("Done. The FBP grid can now be used by ROS_Math to build")
    print("spatially varying ROS/BROS/FROS/RAZ fields that account for slope and aspect.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a gridded FBP terrain dataset (elevation, slope, aspect) "
                    "from a DEM for use in rate-of-spread calculations."
    )
    parser.add_argument(
        "--dem",
        type=str,
        default="/Users/Martyn/Desktop/PhD/Fyah/Prometheus_Old/BowatBanff - Copy/domain_Bow_at_Banff_lumped_elv.tif",
        help="Path to the DEM GeoTIFF (default: Bow_at_Banff example DEM).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/Users/Martyn/Desktop/PhD/Fyah/Test/Bow_FBP_Grid.npz",
        help="Path to output .npz file containing FBP terrain grid (default sends output to Fyah/Test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_fbp_grid(
        dem_path=args.dem,
        out_path=args.out,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()