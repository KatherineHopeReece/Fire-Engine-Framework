import os
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG: input file paths (your real ones)
# ============================================================

FWI_CSV = "/Users/Martyn/Desktop/PhD/Fyah/Prometheus_Old/Sensitivity_Analysis/Bow_FWI_out.csv"
FBP_CSV = "/Users/Martyn/Desktop/PhD/Fyah/Prometheus_Old/Sensitivity_Analysis/Bow_FBP_out.csv"

# Output from Build_FBP_Grid.py
FBP_GRID_NPZ = "/Users/Martyn/Desktop/PhD/Fyah/Test/Bow_FBP_Grid.npz"

# ============================================================
# CONFIG: output directory + output files
# ============================================================
OUTPUT_DIR = "/Users/Martyn/Desktop/PhD/Fyah/Test"
HISTORY_NPZ = f"{OUTPUT_DIR}/Bow_Richards_History.npz"
PLOT_PNG = f"{OUTPUT_DIR}/Bow_Richards_Perimeters.png"
ERROR_LOG = f"{OUTPUT_DIR}/Bow_Richards_ERROR.txt"
FAIL_FLAG = f"{OUTPUT_DIR}/Bow_Richards_FAIL.flag"

# ============================================================
# 1. Load FWI and FBP CSVs
# ============================================================

def load_inputs(fwi_path=FWI_CSV, fbp_path=FBP_CSV):
    """
    Load the FWI and FBP CSVs. For Richards' spread, we mainly
    need ROS and RAZ from the FBP file, but we also read FWI for
    later use if you want it.
    """
    df_fwi = pd.read_csv(fwi_path)
    df_fbp = pd.read_csv(fbp_path)

    # For spread simulation, we really just need df_fbp.
    # We'll return both so you can inspect them if needed.
    return df_fwi, df_fbp

# ============================================================
# 2. Spatial derivatives along the fire front
# ============================================================

def compute_spatial_derivatives(x, y, ds=1.0):
    """
    Periodic central differences for x_s, y_s along the front.
    Index i plays the role of the parameter s.
    """
    x_f = np.roll(x, -1)
    x_b = np.roll(x, 1)
    y_f = np.roll(y, -1)
    y_b = np.roll(y, 1)

    xs = (x_f - x_b) / (2.0 * ds)
    ys = (y_f - y_b) / (2.0 * ds)
    return xs, ys

# ============================================================
# 3. Richards’ differential equations as a velocity field
# ============================================================

def richards_velocity(x, y, a, b, c, theta):
    """
    Compute the temporal derivatives x_t, y_t for the fire front,
    given Richards ellipse parameters (a, b, c, theta).

    Implementing the component form:

      xt = [b^2 cosθ (xs sinθ + ys cosθ) - a^2 sinθ (xs cosθ - ys sinθ)]
           ----------------------------------------------------------- + c sinθ
               sqrt( a^2 (xs cosθ - ys sinθ)^2
                    + b^2 (xs sinθ + ys cosθ)^2 )

      yt = [-b^2 sinθ (xs sinθ + ys cosθ) - a^2 cosθ (xs cosθ - ys sinθ)]
           ------------------------------------------------------------ + c cosθ
               sqrt( a^2 (xs cosθ - ys sinθ)^2
                    + b^2 (xs sinθ + ys cosθ)^2 )

    where:
      xs, ys are derivatives along the curve (with respect to s).

    NOTE: a, b, c, theta can be scalars or arrays broadcastable to x,y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    xs, ys = compute_spatial_derivatives(x, y)

    # Broadcast parameters to shape of x,y
    a = np.broadcast_to(a, x.shape)
    b = np.broadcast_to(b, x.shape)
    c = np.broadcast_to(c, x.shape)
    theta = np.broadcast_to(theta, x.shape)

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    term1 = xs * cos_th - ys * sin_th
    term2 = xs * sin_th + ys * cos_th

    denom = np.sqrt(a**2 * term1**2 + b**2 * term2**2) + 1e-9

    xt = (b**2 * cos_th * term2 - a**2 * sin_th * term1) / denom + c * sin_th
    yt = (-b**2 * sin_th * term2 - a**2 * cos_th * term1) / denom + c * cos_th

    return xt, yt

# ============================================================
# 4. Gridded parameter field + interpolation
# ============================================================

class FireParamGrid:
    """
    Holds gridded ROS, BROS, FROS, RAZ over a 2D landscape and provides
    bilinear interpolation at arbitrary vertex positions.

    Coordinates:
      x_coords: 1D array of x grid coordinates (length nx)
      y_coords: 1D array of y grid coordinates (length ny)

    Fields:
      ros[t, j, i], bros[t, j, i], fros[t, j, i], raz[t, j, i]
      where t indexes time, j = y index, i = x index.
    """

    def __init__(self, x_coords, y_coords, ros, bros, fros, raz):
        self.x_coords = np.asarray(x_coords)
        self.y_coords = np.asarray(y_coords)
        self.ros = np.asarray(ros)
        self.bros = np.asarray(bros)
        self.fros = np.asarray(fros)
        self.raz = np.asarray(raz)

        assert self.ros.shape == self.bros.shape == self.fros.shape == self.raz.shape
        assert self.ros.ndim == 3  # (nt, ny, nx)

        self.nt, self.ny, self.nx = self.ros.shape

        # assume regular grid:
        self.x_min = self.x_coords.min()
        self.x_max = self.x_coords.max()
        self.y_min = self.y_coords.min()
        self.y_max = self.y_coords.max()
        self.dx = (self.x_coords[1] - self.x_coords[0]) if self.nx > 1 else 1.0
        self.dy = (self.y_coords[1] - self.y_coords[0]) if self.ny > 1 else 1.0

    def _bilinear_sample(self, field_t, x, y):
        """
        Bilinear interpolation of a single 2D field (ny, nx) at positions (x,y).
        x,y are 1D arrays (vertices).
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # compute fractional indices
        ix = (x - self.x_min) / self.dx
        iy = (y - self.y_min) / self.dy

        # clamp to [0, nx-1], [0, ny-1]
        ix = np.clip(ix, 0, self.nx - 1 - 1e-6)
        iy = np.clip(iy, 0, self.ny - 1 - 1e-6)

        ix0 = np.floor(ix).astype(int)
        iy0 = np.floor(iy).astype(int)
        ix1 = np.clip(ix0 + 1, 0, self.nx - 1)
        iy1 = np.clip(iy0 + 1, 0, self.ny - 1)

        fx = ix - ix0
        fy = iy - iy0

        # corners
        f00 = field_t[iy0, ix0]
        f10 = field_t[iy0, ix1]
        f01 = field_t[iy1, ix0]
        f11 = field_t[iy1, ix1]

        # bilinear interpolation
        f0 = f00 * (1 - fx) + f10 * fx
        f1 = f01 * (1 - fx) + f11 * fx
        f = f0 * (1 - fy) + f1 * fy

        return f

    def sample_at(self, t_index, x, y):
        """
        Sample ROS, BROS, FROS, RAZ at vertex positions (x,y)
        for time index t_index (integer).

        t_index is clipped into [0, nt-1].
        Returns arrays ros_i, bros_i, fros_i, raz_i of same shape as x.
        """
        t_index = int(np.clip(t_index, 0, self.nt - 1))

        field_ros = self.ros[t_index]
        field_bros = self.bros[t_index]
        field_fros = self.fros[t_index]
        field_raz = self.raz[t_index]

        ros_i = self._bilinear_sample(field_ros, x, y)
        bros_i = self._bilinear_sample(field_bros, x, y)
        fros_i = self._bilinear_sample(field_fros, x, y)
        raz_i = self._bilinear_sample(field_raz, x, y)

        return ros_i, bros_i, fros_i, raz_i

# ============================================================
# 5. Build FireParamGrid from FBP time series + DEM-based slope/aspect
# ============================================================

def build_param_grid_from_fbp_and_dem(
    df_fbp,
    grid_npz_path=FBP_GRID_NPZ,
    LB=2.0,
    backing_fraction=0.2,
    ros_col="ROS",
    raz_col="RAZ",
    slope_factor=0.5,
):
    """
    Build a FireParamGrid using:
      - Temporal ROS/RAZ from Bow_FBP_out.csv (df_fbp),
      - Spatial slope/aspect/topography from the DEM grid (Build_FBP_Grid output).

    Approach:
      * For each time step j:
          - Take "flat" head ROS and RAZ from FBP row j.
          - Use slope and aspect to scale ROS per cell:
                ROS_cell = ROS_flat * (1 + slope_factor * tan(slope) * cos(delta))
            where delta is the angle between heading direction and upslope.
          - Compute BROS and FROS per cell using backing_fraction and LB.
          - Store ROS/BROS/FROS/RAZ in (t,y,x) arrays.

    This is a simple parameterization that "considers" slope/aspect/topography.
    You can later replace the slope_factor formula with a more faithful
    CFSFBP slope correction if desired.
    """
    data = np.load(grid_npz_path)
    slope_deg = data["slope_deg"]
    aspect_deg = data["aspect_deg"]  # downslope direction in degrees (0=N, 90=E, ...)
    x_coords = data["x_coords"]
    y_coords = data["y_coords"]

    ny, nx = slope_deg.shape
    nt = len(df_fbp)

    ros = np.zeros((nt, ny, nx), dtype="float64")
    bros = np.zeros_like(ros)
    fros = np.zeros_like(ros)
    raz = np.zeros_like(ros)

    # Precompute upslope direction from aspect (assumed downslope)
    upslope_deg = (aspect_deg + 180.0) % 360.0
    upslope_rad = np.deg2rad(upslope_deg)

    slope_rad = np.deg2rad(slope_deg)
    tan_slope = np.tan(slope_rad)

    k = backing_fraction

    for t_index, (_, row) in enumerate(df_fbp.iterrows()):
        ROS_flat = float(row[ros_col])   # base head ROS (e.g., m/min) from FBP
        RAZ_flat_deg = float(row[raz_col])  # base effective wind/heading (deg from +y)
        RAZ_flat_rad = np.deg2rad(RAZ_flat_deg)

        # Angle difference between heading and upslope at each cell
        # (wrap via sin/cos for numerical stability)
        delta = RAZ_flat_rad - upslope_rad
        cos_delta = np.cos(delta)

        # Simple slope correction:
        #   more spread if heading is aligned with upslope, less if not.
        #   ROS_cell >= 0 enforced via np.maximum.
        slope_mult = 1.0 + slope_factor * tan_slope * cos_delta
        ROS_cell = ROS_flat * slope_mult
        ROS_cell = np.maximum(ROS_cell, 0.0)

        # Compute BROS and FROS per cell using the same relationships:
        BROS_cell = k * ROS_cell
        FROS_cell = (ROS_cell + BROS_cell) / (2.0 * LB)

        ros[t_index, :, :] = ROS_cell
        bros[t_index, :, :] = BROS_cell
        fros[t_index, :, :] = FROS_cell
        # For now, keep RAZ the same everywhere in space (could also be
        # modified by slope if desired)
        raz[t_index, :, :] = np.deg2rad(RAZ_flat_deg)

    return FireParamGrid(x_coords, y_coords, ros, bros, fros, raz)

# ============================================================
# 6. Time integration with gridded a(x,y), b(x,y), c(x,y), theta(x,y)
# ============================================================

def simulate_fire_front_with_grid(
    param_grid,
    dt=1.0,
    n_points=200,
    store_every=1,
):
    """
    Evolve the fire front using a FireParamGrid that gives you
    ROS(x,y,t), BROS(x,y,t), FROS(x,y,t), RAZ(x,y,t).

    This yields spatially varying a(x,y), b(x,y), c(x,y), theta(x,y),
    as per the continuous description of Richards' model.

    Parameters
    ----------
    param_grid : FireParamGrid
        Precomputed gridded ROS/BROS/FROS/RAZ over the landscape.
    dt : float
        Time step (same units as ROS).
    n_points : int
        Number of vertices in the polygonal fire front.
    store_every : int
        Store the perimeter every 'store_every' time steps.
    """

    # Initial front: small circle
    s = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r0 = 0.5
    x = r0 * np.cos(s)
    y = r0 * np.sin(s)

    history = [(x.copy(), y.copy())]

    nt = param_grid.nt

    for j in range(nt):
        # 1) Interpolate ROS/BROS/FROS/RAZ at each vertex
        ros_i, bros_i, fros_i, raz_i = param_grid.sample_at(j, x, y)

        # 2) Convert to a,b,c,theta per vertex (arrays)
        a = 0.5 * (ros_i + bros_i)
        c = 0.5 * (ros_i - bros_i)
        b = fros_i
        theta = raz_i  # already in radians

        # 3) Compute velocity via Richards PDE with spatially varying params
        xt, yt = richards_velocity(x, y, a, b, c, theta)

        # 4) Explicit Euler update
        x = x + dt * xt
        y = y + dt * yt

        if (j + 1) % store_every == 0:
            history.append((x.copy(), y.copy()))

    return history

# ============================================================
# 7. Plotting helper
# ============================================================

def plot_history(history, title="Richards Fire Spread with DEM-based ROS", save_path=None, show=True):
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

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

# ============================================================
# 8. Main test run using YOUR file paths + DEM grid
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load data
        df_fwi, df_fbp = load_inputs(FWI_CSV, FBP_CSV)

        # --- sanity checks / column names ---
        print("FBP columns:", list(df_fbp.columns))
        print("FWI columns:", list(df_fwi.columns))

        ros_col = "ROS"
        raz_col = "RAZ"

        if ros_col not in df_fbp.columns or raz_col not in df_fbp.columns:
            raise ValueError(
                f"Expected columns '{ros_col}' and '{raz_col}' in FBP CSV, "
                f"but got: {list(df_fbp.columns)}"
            )

        # Optionally subset time steps for testing, e.g.:
        # df_fbp = df_fbp.head(50).copy()

        # Build the spatially varying ROS grid that includes slope and aspect effects
        LB = 2.0               # assumed length-to-breadth ratio
        backing_fraction = 0.2 # BROS / ROS
        slope_factor = 0.5     # strength of slope effect on ROS

        param_grid = build_param_grid_from_fbp_and_dem(
            df_fbp,
            grid_npz_path=FBP_GRID_NPZ,
            LB=LB,
            backing_fraction=backing_fraction,
            ros_col=ros_col,
            raz_col=raz_col,
            slope_factor=slope_factor,
        )

        # Simulate perimeter evolution using the gridded parameters
        history = simulate_fire_front_with_grid(
            param_grid,
            dt=1.0,  # 1 time unit per row (e.g., 1 minute)
            n_points=300,
            store_every=max(1, param_grid.nt // 20),  # ~20 frames
        )

        # Save history to NPZ in OUTPUT_DIR
        np.savez(
            HISTORY_NPZ,
            x_frames=[h[0] for h in history],
            y_frames=[h[1] for h in history],
        )

        # Plot result (save to OUTPUT_DIR, still show interactively)
        plot_history(
            history,
            title="Richards Fire Spread (slope & aspect from DEM)",
            save_path=PLOT_PNG,
            show=True,
        )

        # Clear any previous failure flag on success
        if os.path.exists(FAIL_FLAG):
            os.remove(FAIL_FLAG)

        print(f"Saved history to: {HISTORY_NPZ}")
        print(f"Saved plot to: {PLOT_PNG}")

    except Exception as e:
        # Write a detailed error log + a simple fail flag file
        tb = traceback.format_exc()
        with open(ERROR_LOG, "w") as f:
            f.write(tb)

        with open(FAIL_FLAG, "w") as f:
            f.write("FAILED\n")
            f.write(str(e) + "\n")

        print("Run failed. See error log:", ERROR_LOG)
        raise