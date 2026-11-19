
"""
Generate a synthetic fractal DEM using the 1/f Fourier filtering method
described by Bova, Mell & Hoffman (2016).

Outputs
-------
1. DEM_fractal_1overf.tif       - Synthetic elevation raster
2. DEM_fractal_3D.png           - 3D terrain visualization
3. DEM_fractal_elevation_map.png - Plan-view elevation map

The input DEM provides the grid geometry, CRS, and valid-data mask.
Its flat elevations are replaced with synthetic elevations of 0-200 m.
"""

from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import Geod
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import colors


# ============================================================
# 1. USER SETTINGS
# ============================================================

INPUT_DEM = Path(
    "/Users/Martyn/GitRepos/Fire-Engine-Framework-dev/"
    "ignacio/LaughTests/LaughTest_Data/DEM_flat.tif"
)

OUTPUT_DIR = INPUT_DEM.parent

OUTPUT_DEM = OUTPUT_DIR / "DEM_fractal_1overf.tif"
OUTPUT_3D = OUTPUT_DIR / "DEM_fractal_3D.png"
OUTPUT_MAP = OUTPUT_DIR / "DEM_fractal_elevation_map.png"

# Reproducibility
RANDOM_SEED = 42

# Elevation range (metres)
MIN_ELEVATION = 0.0
MAX_ELEVATION = 200.0

# Maximum allowed terrain slope (degrees)
MAX_SLOPE_DEG = 85.0

# Spectral filtering:
# Fourier amplitude is multiplied by 1 / frequency^exponent.
# Exponent 1.0 reproduces the described 1/f filtering method.
SPECTRAL_EXPONENT = 1.0

# Maximum sampled grid size along the longest plot dimension
MAX_PLOT_CELLS = 275

# Set False to suppress interactive plot windows
SHOW_PLOTS = True


# ============================================================
# 2. READ INPUT DEM
# ============================================================

def read_reference_dem(path):
    if not path.exists():
        raise FileNotFoundError(f"Input DEM not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        valid = ~np.ma.getmaskarray(data)

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

        height, width = data.shape

    if not valid.any():
        raise ValueError("The input DEM has no valid pixels.")

    if crs is None:
        raise ValueError(
            "Input DEM has no CRS. Define its CRS before generating terrain."
        )

    if not np.isclose(transform.b, 0) or not np.isclose(
        transform.d, 0
    ):
        raise ValueError(
            "Rotated rasters are not supported by this script."
        )

    print("\nREFERENCE DEM")
    print("-" * 55)
    print(f"Input:      {path}")
    print(f"Grid:       {height} rows x {width} columns")
    print(f"CRS:        {crs}")
    print(f"Resolution: {abs(transform.a):.8f} x "
          f"{abs(transform.e):.8f} CRS units")

    return valid, profile, transform, crs


# ============================================================
# 3. DETERMINE PIXEL SPACING IN METRES
# ============================================================

def pixel_spacing_metres(transform, crs, shape):
    """
    Return physical pixel width and height in metres.

    Projected CRS: convert the CRS linear units to metres.
    Geographic CRS: calculate approximate metre spacing at
    the centre of the raster using a geodesic calculation.

    The geographic approximation is appropriate for a small
    experimental domain but not for a very large region.
    """

    if crs.is_projected:
        unit_factor = crs.linear_units_factor[1]

        dx = abs(transform.a) * unit_factor
        dy = abs(transform.e) * unit_factor

    elif crs.is_geographic:
        rows, cols = shape

        lon, lat = xy(
            transform,
            rows // 2,
            cols // 2,
            offset="center"
        )

        geod = Geod(ellps="WGS84")

        _, _, dx = geod.inv(
            lon, lat,
            lon + abs(transform.a), lat
        )

        _, _, dy = geod.inv(
            lon, lat,
            lon, lat + abs(transform.e)
        )

        warnings.warn(
            "The input DEM uses geographic coordinates. "
            "Pixel spacing in metres is estimated at the "
            "raster centre. A projected metric CRS is "
            "preferable for numerical experiments."
        )

    else:
        raise ValueError(
            "Unsupported CRS: cannot calculate metre spacing."
        )

    if dx <= 0 or dy <= 0:
        raise ValueError("Invalid raster pixel spacing.")

    return dx, dy


# ============================================================
# 4. GENERATE FRACTAL TERRAIN USING FFT
# ============================================================

def generate_fractal_terrain(
    shape,
    dx,
    dy,
    seed=42,
    spectral_exponent=1.0
):
    """
    Bova-style synthetic terrain generation:

    1. Generate random white noise.
    2. Compute its 2D Fourier transform.
    3. Multiply Fourier amplitudes by 1/f.
    4. Compute the inverse Fourier transform.

    The zero-frequency (DC) component is removed.
    """

    rows, cols = shape

    rng = np.random.default_rng(seed)

    # Step 1: random noise field
    noise = rng.standard_normal((rows, cols))

    # Step 2: Fourier transform
    noise_fft = np.fft.rfft2(noise)

    # Step 3: physical spatial frequencies (cycles/metre)
    fy = np.fft.fftfreq(rows, d=dy)
    fx = np.fft.rfftfreq(cols, d=dx)

    frequency = np.hypot(
        fy[:, np.newaxis],
        fx[np.newaxis, :]
    )

    # 1/f amplitude filter; avoid division by zero
    frequency_filter = np.zeros_like(frequency)

    np.divide(
        1.0,
        frequency ** spectral_exponent,
        out=frequency_filter,
        where=frequency > 0
    )

    frequency_filter[0, 0] = 0.0

    filtered_fft = noise_fft * frequency_filter

    # Step 4: inverse Fourier transform
    terrain = np.fft.irfft2(
        filtered_fft,
        s=(rows, cols)
    )

    return terrain


# ============================================================
# 5. NORMALIZE ELEVATION TO 0-200 METRES
# ============================================================

def normalize_elevation(terrain, valid):
    """
    Linearly map valid terrain elevations onto the specified
    minimum and maximum elevation range.
    """

    z_min = np.min(terrain[valid])
    z_max = np.max(terrain[valid])

    if np.isclose(z_min, z_max):
        raise ValueError(
            "The generated terrain has insufficient variation."
        )

    scaled = (terrain - z_min) / (z_max - z_min)

    scaled = (
        MIN_ELEVATION
        + scaled * (MAX_ELEVATION - MIN_ELEVATION)
    )

    return scaled


# ============================================================
# 6. CALCULATE TERRAIN SLOPE
# ============================================================

def calculate_slope(terrain, dx, dy):
    """
    Calculate slope from elevation gradients.

    tan(theta) = sqrt((dz/dx)^2 + (dz/dy)^2)

    theta = arctan(gradient magnitude)
    """

    dz_dy, dz_dx = np.gradient(
        terrain,
        dy,
        dx
    )

    gradient_magnitude = np.hypot(
        dz_dx,
        dz_dy
    )

    slope_degrees = np.degrees(
        np.arctan(gradient_magnitude)
    )

    return slope_degrees


# ============================================================
# 7. APPLY OPTIONAL SLOPE CONSTRAINT
# ============================================================

def constrain_maximum_slope(
    original_terrain,
    valid,
    dx,
    dy
):
    """
    Preserve the 0-200 m elevation range while limiting
    maximum computed slope to approximately 85 degrees.

    Gaussian smoothing is applied only if necessary.

    Note: smoothing modifies the original 1/f spectrum.
    """

    terrain = normalize_elevation(
        original_terrain, valid
    )

    slope = calculate_slope(
        terrain, dx, dy
    )

    maximum_slope = np.max(slope[valid])

    print("\nINITIAL TERRAIN")
    print("-" * 55)
    print(f"Maximum slope: {maximum_slope:.3f} degrees")

    if maximum_slope <= MAX_SLOPE_DEG:
        return terrain, slope, 0.0

    # Increase smoothing progressively when required.
    smoothing_scales = [
        0.6, 1.0, 1.5, 2.3, 3.5,
        5.0, 8.0, 12.0, 18.0,
        26.0, 40.0
    ]

    for sigma in smoothing_scales:

        smoothed = gaussian_filter(
            original_terrain,
            sigma=sigma,
            mode="wrap"
        )

        terrain = normalize_elevation(
            smoothed, valid
        )

        slope = calculate_slope(
            terrain, dx, dy
        )

        maximum_slope = np.max(slope[valid])

        print(
            f"Gaussian sigma={sigma:.1f} pixels: "
            f"maximum slope={maximum_slope:.3f} degrees"
        )

        if maximum_slope <= MAX_SLOPE_DEG:
            return terrain, slope, sigma

    raise RuntimeError(
        "Could not satisfy the maximum slope constraint "
        "while retaining the requested elevation range. "
        "Check the physical size and resolution of the DEM."
    )


# ============================================================
# 8. WRITE NEW GEOTIFF
# ============================================================

def save_dem(terrain, valid, profile, output_path):
    nodata = -9999.0

    output = np.where(
        valid,
        terrain,
        nodata
    ).astype(np.float32)

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3
    )

    with rasterio.open(
        output_path, "w", **profile
    ) as dst:
        dst.write(output, 1)

    print(f"\nSaved DEM: {output_path}")


# ============================================================
# 9. CREATE 3D TERRAIN PLOT
# ============================================================


def plot_terrain_3d(terrain, valid, dx, dy, output_path):
    """
    Publication-quality 3D visualization of synthetic fractal terrain.

    Improvements:
    - Stronger vertical relief (visualization only)
    - High-contrast terrain colours
    - Directional hillshading
    - Light smoothing for plotting only
    - Cleaner axes and background
    - Orthographic camera
    - High-resolution manuscript export

    The underlying DEM is NOT modified.
    """

    from matplotlib.colors import (
        LinearSegmentedColormap,
        Normalize,
        LightSource,
    )
    from matplotlib.cm import ScalarMappable
    from scipy.ndimage import gaussian_filter

    # =====================================================
    # VISUALIZATION SETTINGS
    # =====================================================

    VERTICAL_EXAGGERATION = 100.0

    MAX_PLOT_CELLS = 250

    # Smooth the DISPLAYED surface only.
    DISPLAY_SMOOTHING = 1.2

    # Hillshade illumination
    LIGHT_AZIMUTH = 315
    LIGHT_ALTITUDE = 45
    HILLSHADE_EXAGGERATION = 45.0

    # Camera
    VIEW_ELEVATION = 37
    VIEW_AZIMUTH = -65

    # =====================================================
    # PREPARE GRID
    # =====================================================

    rows, cols = terrain.shape

    nrows = min(rows, MAX_PLOT_CELLS)
    ncols = min(cols, MAX_PLOT_CELLS)

    # Reverse rows so Y increases northward
    row_idx = np.unique(
        np.linspace(
            0, rows - 1, nrows, dtype=int
        )
    )[::-1]

    col_idx = np.unique(
        np.linspace(
            0, cols - 1, ncols, dtype=int
        )
    )

    # Coordinates in kilometres
    x = (col_idx + 0.5) * dx / 1000.0

    y = (
        rows - row_idx - 0.5
    ) * dy / 1000.0

    X, Y = np.meshgrid(x, y)

    Z_raw = terrain[
        np.ix_(row_idx, col_idx)
    ].astype(float)

    mask = valid[
        np.ix_(row_idx, col_idx)
    ]

    # =====================================================
    # DISPLAY-ONLY SMOOTHING
    # =====================================================

    # Weighted smoothing prevents nodata cells from
    # contaminating valid elevations.

    values = np.where(mask, Z_raw, 0.0)

    weights = gaussian_filter(
        mask.astype(float),
        sigma=DISPLAY_SMOOTHING
    )

    smoothed = gaussian_filter(
        values,
        sigma=DISPLAY_SMOOTHING
    )

    Z_display = smoothed / np.maximum(
        weights, 1e-10
    )

    Z_display[~mask] = np.nan

    # =====================================================
    # STRONGER TERRAIN COLOUR MAP
    # =====================================================

    # Dark green valleys -> green slopes ->
    # earthy brown ridges -> pale mountain summits.

    terrain_cmap = LinearSegmentedColormap.from_list(
        "manuscript_terrain",
        [
            (0.00, "#184B46"),
            (0.20, "#347A59"),
            (0.40, "#79A276"),
            (0.60, "#B5AA78"),
            (0.80, "#B68C63"),
            (1.00, "#F0DFBB"),
        ],
        N=256
    )

    norm = Normalize(
        vmin=MIN_ELEVATION,
        vmax=MAX_ELEVATION
    )

    # =====================================================
    # DIRECTIONAL HILLSHADING
    # =====================================================

    light = LightSource(
        azdeg=LIGHT_AZIMUTH,
        altdeg=LIGHT_ALTITUDE
    )

    # Spacing of the subsampled display grid in metres
    plot_dx = (
        abs(x[1] - x[0]) * 1000.0
        if len(x) > 1 else dx
    )

    plot_dy = (
        abs(y[1] - y[0]) * 1000.0
        if len(y) > 1 else dy
    )

    # Fill missing values solely for shading calculation
    Z_shade = np.where(
        mask,
        Z_display,
        np.nanmedian(Z_display)
    )

    facecolors = light.shade(
        Z_shade,
        cmap=terrain_cmap,
        norm=norm,
        vert_exag=HILLSHADE_EXAGGERATION,
        dx=plot_dx,
        dy=plot_dy,
        blend_mode="overlay"
    )

    facecolors[~mask, 3] = 0

    # =====================================================
    # VERTICAL EXAGGERATION
    # =====================================================

    # Horizontal coordinates are in km.
    # Convert elevation to km and apply exaggeration.
    #
    # Important: ONLY the plotted elevations are scaled.
    # The saved GeoTIFF remains unchanged.

    Z_plot = (
        Z_display
        * VERTICAL_EXAGGERATION
        / 1000.0
    )

    # =====================================================
    # CREATE FIGURE
    # =====================================================

    fig = plt.figure(
        figsize=(15, 10),
        facecolor="white"
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
        computed_zorder=False
    )

    ax.set_proj_type("ortho")

    surface = ax.plot_surface(
        X,
        Y,
        Z_plot,
        facecolors=facecolors,
        rcount=len(row_idx),
        ccount=len(col_idx),
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
        alpha=1.0
    )

    # =====================================================
    # CORRECT PHYSICAL ASPECT RATIO
    # =====================================================

    x_span = np.ptp(x)
    y_span = np.ptp(y)

    z_span = (
        (MAX_ELEVATION - MIN_ELEVATION)
        * VERTICAL_EXAGGERATION
        / 1000.0
    )

    ax.set_box_aspect(
        (
            x_span,
            y_span,
            z_span
        )
    )

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    ax.set_zlim(
        MIN_ELEVATION
        * VERTICAL_EXAGGERATION / 1000.0,
        MAX_ELEVATION
        * VERTICAL_EXAGGERATION / 1000.0
    )

    # =====================================================
    # CAMERA POSITION
    # =====================================================

    ax.view_init(
        elev=VIEW_ELEVATION,
        azim=VIEW_AZIMUTH
    )

    # =====================================================
    # LABELS
    # =====================================================

    ax.set_xlabel(
        "Easting distance (km)",
        fontsize=11,
        labelpad=14
    )

    ax.set_ylabel(
        "Northing distance (km)",
        fontsize=11,
        labelpad=14
    )

    ax.set_zlabel(
        "Elevation (m)",
        fontsize=11,
        labelpad=10
    )

    # Display true elevations on the vertical axis
    elevation_ticks = np.linspace(
        MIN_ELEVATION,
        MAX_ELEVATION,
        5
    )

    ax.set_zticks(
        elevation_ticks
        * VERTICAL_EXAGGERATION / 1000.0
    )

    ax.set_zticklabels(
        [f"{v:.0f}" for v in elevation_ticks]
    )

    ax.tick_params(
        labelsize=9,
        pad=3
    )

    # =====================================================
    # REMOVE DISTRACTING GREY 3D BACKGROUND
    # =====================================================

    for axis in [
        ax.xaxis,
        ax.yaxis,
        ax.zaxis
    ]:
        axis.pane.fill = False
        axis.pane.set_edgecolor(
            (0.75, 0.75, 0.75, 0.5)
        )

    ax.grid(False)

    # =====================================================
    # TITLE
    # =====================================================

    fig.suptitle(
        "Synthetic Fractal Terrain",
        fontsize=19,
        fontweight="bold",
        y=0.97
    )

    fig.text(
        0.50,
        0.92,
        "1/f Fourier-filtered topography | "
        f"Vertical exaggeration: {VERTICAL_EXAGGERATION:.0f}×",
        ha="center",
        fontsize=11,
        color="#555555"
    )

    # Give the terrain more room in the figure
    ax.set_position(
        [0.04, 0.04, 0.80, 0.84]
    )

    # =====================================================
    # COLOUR BAR
    # =====================================================

    sm = ScalarMappable(
        norm=norm,
        cmap=terrain_cmap
    )

    sm.set_array([])

    cax = fig.add_axes(
        [0.88, 0.24, 0.022, 0.52]
    )

    cbar = fig.colorbar(
        sm,
        cax=cax
    )

    cbar.set_label(
        "Elevation (m)",
        fontsize=12,
        labelpad=12
    )

    cbar.ax.tick_params(
        labelsize=10
    )

    # =====================================================
    # SAVE
    # =====================================================

    fig.savefig(
        output_path,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="white"
    )

    # Optional vector export for manuscript
    pdf_path = (
        output_path.parent
        / f"{output_path.stem}.pdf"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="white"
    )

    print("\nImproved 3D terrain plots saved:")
    print(f"PNG: {output_path}")
    print(f"PDF: {pdf_path}")

    return fig


# ============================================================
# 10. CREATE PLAN-VIEW ELEVATION MAP
# ============================================================

def plot_elevation_map(
    terrain,
    valid,
    dx,
    dy,
    output_path
):
    rows, cols = terrain.shape

    extent = [
        0,
        cols * dx / 1000.0,
        0,
        rows * dy / 1000.0
    ]

    plot_data = np.ma.masked_where(
        ~valid,
        terrain
    )

    fig, ax = plt.subplots(
        figsize=(10, 8),
        facecolor="white"
    )

    im = ax.imshow(
        plot_data,
        origin="upper",
        extent=extent,
        cmap="terrain",
        vmin=MIN_ELEVATION,
        vmax=MAX_ELEVATION,
        interpolation="nearest"
    )

    ax.set_title(
        "Synthetic Fractal Elevation",
        fontsize=15,
        weight="bold",
        pad=15
    )

    ax.set_xlabel("Easting distance (km)")
    ax.set_ylabel("Northing distance (km)")

    ax.set_aspect("equal")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label("Elevation (m)")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved elevation map: {output_path}")

    return fig


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    valid, profile, transform, crs = read_reference_dem(
        INPUT_DEM
    )

    shape = (
        profile["height"],
        profile["width"]
    )

    dx, dy = pixel_spacing_metres(
        transform,
        crs,
        shape
    )

    print("\nPHYSICAL GRID")
    print("-" * 55)
    print(f"Pixel width:   {dx:.4f} m")
    print(f"Pixel height:  {dy:.4f} m")
    print(f"Domain width:  {shape[1] * dx:.2f} m")
    print(f"Domain height: {shape[0] * dy:.2f} m")

    print("\nGenerating synthetic fractal terrain...")

    raw_terrain = generate_fractal_terrain(
        shape=shape,
        dx=dx,
        dy=dy,
        seed=RANDOM_SEED,
        spectral_exponent=SPECTRAL_EXPONENT
    )

    terrain, slope, smoothing_sigma = constrain_maximum_slope(
        raw_terrain,
        valid,
        dx,
        dy
    )

    print("\nFINAL TERRAIN STATISTICS")
    print("-" * 55)

    print(
        f"Minimum elevation: "
        f"{np.min(terrain[valid]):.3f} m"
    )

    print(
        f"Maximum elevation: "
        f"{np.max(terrain[valid]):.3f} m"
    )

    print(
        f"Mean elevation: "
        f"{np.mean(terrain[valid]):.3f} m"
    )

    print(
        f"Maximum slope: "
        f"{np.max(slope[valid]):.3f} degrees"
    )

    print(
        f"Mean slope: "
        f"{np.mean(slope[valid]):.3f} degrees"
    )

    print(
        f"95th percentile slope: "
        f"{np.percentile(slope[valid], 95):.3f} degrees"
    )

    print(
        f"Additional smoothing: "
        f"{smoothing_sigma:.2f} pixels"
    )

    # Save DEM
    save_dem(
        terrain,
        valid,
        profile,
        OUTPUT_DEM
    )

    # Create figures
    plot_terrain_3d(
        terrain,
        valid,
        dx,
        dy,
        OUTPUT_3D
    )

    plot_elevation_map(
        terrain,
        valid,
        dx,
        dy,
        OUTPUT_MAP
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close("all")

    print("\nAll outputs generated successfully.")


if __name__ == "__main__":
    main()
