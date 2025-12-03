from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol, xy
from shapely.geometry import Polygon


# --------------------------------------------------------------
# Data structures
# --------------------------------------------------------------

@dataclass
class EngineSettings:
    distance_resolution_m: float = 30.0
    max_dt_equilibrium_s: float = 60.0
    max_dt_accel_s: float = 2.0
    smoothing_omega: float = 0.0
    stop_at_boundary: bool = False
    export_outputs: bool = True
    output_dir: str = "Outputs"


@dataclass
class Grids:
    fuel: np.ndarray
    ROS_H: np.ndarray
    ROS_F: np.ndarray
    ROS_B: np.ndarray
    RAZ: np.ndarray
    transform: rasterio.Affine
    crs: any
    nonfuel_values: Tuple[int, ...] = (0, -9999)


# --------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------

def _inside_raster(shape: Tuple[int, int], r: int, c: int) -> bool:
    nr, nc = shape
    return (0 <= r < nr) and (0 <= c < nc)


def _sample_cell_indices(transform: rasterio.Affine, x: float, y: float) -> Tuple[int, int]:
    r, c = rowcol(transform, x, y)
    return int(r), int(c)


def _bresenham_cells_between(transform, x0, y0, x1, y1, shape):
    """Grid cells intersected by segment P0→P1."""
    r0, c0 = _sample_cell_indices(transform, x0, y0)
    r1, c1 = _sample_cell_indices(transform, x1, y1)
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err, r, c = (dr if dr > dc else -dc) // 2, r0, c0

    while True:
        if _inside_raster(shape, r, c):
            cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = err
        if e2 > -dr:
            err -= dc
            r += sr
        if e2 < dc:
            err += dr
            c += sc
    return cells


def _clip_to_first_nonfuel(x0, y0, x1, y1, grids: Grids) -> Tuple[float, float, bool]:
    """Clip propagation segment to first nonfuel or boundary cell."""
    nr, nc = grids.fuel.shape
    cells = _bresenham_cells_between(grids.transform, x0, y0, x1, y1, (nr, nc))
    
    for idx, (r, c) in enumerate(cells[1:], start=1):
        # Check if outside raster bounds
        if not _inside_raster((nr, nc), r, c):
            if idx > 0:
                r_prev, c_prev = cells[idx - 1]
                x_prev, y_prev = xy(grids.transform, r_prev, c_prev, offset="center")
                return x_prev, y_prev, True
            return x0, y0, True
        
        # Check if non-fuel cell (water, urban, etc.)
        fuel_val = grids.fuel[r, c]
        if fuel_val in grids.nonfuel_values or np.isnan(fuel_val):
            if idx > 0:
                r_prev, c_prev = cells[idx - 1]
                x_prev, y_prev = xy(grids.transform, r_prev, c_prev, offset="center")
                return x_prev, y_prev, True
            return x0, y0, True
    
    return x1, y1, False


def validate_ignition_location(x, y, grids):
    """
    Check if ignition location has valid fuel and ROS data.
    Returns: (is_valid, reason)
    """
    try:
        r, c = rowcol(grids.transform, x, y)
        
        # Check if within raster bounds
        if not _inside_raster(grids.fuel.shape, r, c):
            return False, "outside raster bounds"
        
        # Check if fuel type is valid
        fuel_val = grids.fuel[r, c]
        if fuel_val in grids.nonfuel_values or np.isnan(fuel_val):
            return False, f"non-fuel cell (fuel={fuel_val})"
        
        # Check if ROS values exist
        ros_h = grids.ROS_H[r, c]
        ros_f = grids.ROS_F[r, c]
        ros_b = grids.ROS_B[r, c]
        
        if np.isnan(ros_h) or np.isnan(ros_f) or np.isnan(ros_b):
            return False, "no ROS data"
        
        if ros_h <= 0 and ros_f <= 0 and ros_b <= 0:
            return False, "ROS values are zero"
        
        return True, "valid"
    
    except Exception as e:
        return False, f"error: {str(e)}"


# --------------------------------------------------------------
# Ignition polygon initialization
# --------------------------------------------------------------

def ignition_point(x: float, y: float, n_vertices: int = 16, diameter_m: float = 0.5) -> np.ndarray:
    """Initial pseudo-point ignition (regular polygon)."""
    r = diameter_m / 2.0
    ang = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    return np.stack([x + r * np.cos(ang), y + r * np.sin(ang)], axis=1)


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """Ensure counter-clockwise orientation."""
    x, y = vertices[:, 0], vertices[:, 1]
    area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return vertices if area2 > 0 else vertices[::-1]


# --------------------------------------------------------------
# Richards (1993/1999) 2-D fire spread equations
# --------------------------------------------------------------

def _central_tangent(vtx, i):
    n = len(vtx)
    p_prev, p_next = vtx[(i - 1) % n], vtx[(i + 1) % n]
    t = p_next - p_prev
    norm = np.hypot(*t) + 1e-12
    return t[0] / norm, t[1] / norm


def _ae_be_ce(ROSh, FROSh, BROSh):
    """Ellipse parameters (a_e, b_e, c_e)."""
    return 0.5 * (ROSh + BROSh), FROSh, 0.5 * (ROSh - BROSh)


def _deltaP_2d(xs, ys, theta_deg, a_e, b_e, c_e):
    """Richards 2-D ellipse spread equations."""
    th = math.radians(theta_deg)
    s, c = math.sin(th), math.cos(th)
    A, B = -ys * s + xs * c, ys * c + xs * s
    denom = math.sqrt((a_e ** 2) * (A ** 2) + (b_e ** 2) * (B ** 2)) + 1e-12
    vx = (-(a_e ** 2) * s * A + (b_e ** 2) * c * B) / denom + c_e * s
    vy = (-(a_e ** 2) * c * A - (b_e ** 2) * s * B) / denom + c_e * c
    return vx, vy


# --------------------------------------------------------------
# Fire spread propagation with robust NaN handling
# --------------------------------------------------------------

def propagate_vertices(vertices_xy, grids: Grids, settings: EngineSettings):
    vtx = ensure_ccw(vertices_xy.copy())
    n = len(vtx)
    ROSh, ROSf, ROSb, RAZd = [np.zeros(n) for _ in range(4)]

    # Sample ROS/RAZ at each vertex with validation
    valid_vertex_count = 0
    for i, (x, y) in enumerate(vtx):
        try:
            r, c = rowcol(grids.transform, x, y)
            if not _inside_raster(grids.fuel.shape, r, c):
                continue
            
            ros_h = grids.ROS_H[r, c]
            ros_f = grids.ROS_F[r, c]
            ros_b = grids.ROS_B[r, c]
            raz = grids.RAZ[r, c]
            
            # Only use if valid (not NaN and positive)
            if not (np.isnan(ros_h) or np.isnan(ros_f) or np.isnan(ros_b) or np.isnan(raz)):
                if ros_h > 0 or ros_f > 0 or ros_b > 0:
                    ROSh[i] = ros_h
                    ROSf[i] = ros_f
                    ROSb[i] = ros_b
                    RAZd[i] = raz
                    valid_vertex_count += 1
        except Exception:
            continue

    # Check if we have any valid ROS values
    if valid_vertex_count == 0:
        return vtx, {"dt_s": 0.0, "moved": False, "error": "no valid vertices"}

    # Get max velocity, filtering out invalid values
    all_ros = np.concatenate([ROSh, ROSf, ROSb])
    valid_ros = all_ros[(all_ros > 0) & ~np.isnan(all_ros)]
    
    if len(valid_ros) == 0:
        return vtx, {"dt_s": 0.0, "moved": False, "error": "all ROS invalid"}
    
    vmax = np.max(valid_ros)
    dt = min(settings.max_dt_equilibrium_s, settings.distance_resolution_m / vmax)
    dP = np.zeros_like(vtx)

    # Calculate displacement for each vertex
    for i in range(n):
        # Skip vertices with no valid ROS
        if ROSh[i] == 0 and ROSf[i] == 0 and ROSb[i] == 0:
            continue
        
        xs, ys = _central_tangent(vtx, i)
        a_e, b_e, c_e = _ae_be_ce(ROSh[i], ROSf[i], ROSb[i])
        vx, vy = _deltaP_2d(xs, ys, RAZd[i], a_e, b_e, c_e)
        
        # Check for NaN in velocity
        if np.isnan(vx) or np.isnan(vy):
            continue
        
        dP[i] = [vx * dt, vy * dt]

    # Optional smoothing
    if settings.smoothing_omega > 0:
        w = settings.smoothing_omega
        dP = (1 - w) * dP + 0.5 * w * (dP + np.roll(dP, -1, axis=0))

    # Apply displacements with boundary checking
    new_vtx, hit_break = vtx.copy(), False
    for i in range(n):
        x0, y0 = vtx[i]
        x1, y1 = x0 + dP[i, 0], y0 + dP[i, 1]
        
        # Check for NaN in new position
        if np.isnan(x1) or np.isnan(y1):
            new_vtx[i] = [x0, y0]
            continue
        
        try:
            x1c, y1c, hit = _clip_to_first_nonfuel(x0, y0, x1, y1, grids)
            new_vtx[i] = [x1c, y1c]
            hit_break |= hit
        except Exception:
            new_vtx[i] = [x0, y0]
            continue

    if settings.stop_at_boundary and hit_break:
        return vtx, {"dt_s": 0.0, "moved": False, "hit_break": True}
    
    # Check if fire actually moved
    movement = np.linalg.norm(new_vtx - vtx)
    if movement < 1e-6:  # Less than 1mm
        return vtx, {"dt_s": 0.0, "moved": False, "error": "no movement"}
    
    return new_vtx, {"dt_s": dt, "moved": True, "hit_break": hit_break}


# --------------------------------------------------------------
# Vertex rediscretization
# --------------------------------------------------------------

def insert_new_vertices(vertices_xy, Lx, max_new=5):
    """Prometheus Eqs. [31]-[36]."""
    vtx, n = vertices_xy.copy(), len(vertices_xy)
    new_vertices = []
    for i in range(n):
        p_prev, p_curr, p_next = vtx[(i - 1) % n], vtx[i], vtx[(i + 1) % n]
        L_prev, L_next = np.linalg.norm(p_curr - p_prev), np.linalg.norm(p_next - p_curr)
        v1, v2 = p_prev - p_curr, p_next - p_curr
        cosb = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)
        beta_p = math.acos(cosb)
        need = lambda L: (L / Lx) > math.sin(beta_p / 2.0)
        if need(L_prev):
            new_vertices.append((p_prev + p_curr) / 2)
        new_vertices.append(p_curr)
        if need(L_next):
            new_vertices.append((p_curr + p_next) / 2)
        if len(new_vertices) > n + max_new:
            break

    result = [new_vertices[0]]
    for p in new_vertices[1:]:
        if np.linalg.norm(np.array(p) - np.array(result[-1])) > Lx / 1000.0:
            result.append(p)
    return np.array(result) if len(result) >= 3 else vertices_xy


def remove_close_vertices(vertices_xy, Lx):
    if len(vertices_xy) < 3:
        return vertices_xy
    out = [vertices_xy[0]]
    for p in vertices_xy[1:]:
        if np.linalg.norm(p - out[-1]) >= Lx / 1000.0:
            out.append(p)
    return np.array(out) if len(out) >= 3 else vertices_xy


# --------------------------------------------------------------
# Fire statistics and export
# --------------------------------------------------------------

def fire_perimeter_stats(vertices_xy, fuel_breaks=None):
    poly = Polygon(vertices_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if fuel_breaks:
        for br in fuel_breaks:
            poly = poly.difference(br)
    return dict(
        perimeter_m=poly.length,
        area_m2=poly.area,
        centroid=(poly.centroid.x, poly.centroid.y),
    )


def export_vertex_stats(vertices_xy, step_idx, stats, out_dir):
    df = pd.DataFrame(vertices_xy, columns=["x", "y"])
    df["vertex_id"] = np.arange(len(vertices_xy))
    df["step"] = step_idx
    df["Perimeter_m"] = stats["perimeter_m"]
    df["Area_m2"] = stats["area_m2"]
    df.to_csv(f"{out_dir}/fire_vertices_step{step_idx:04d}.csv", index=False)


def export_perimeters_shapefile(perims, crs, out_path):
    geoms, times = [], []
    for i, v in enumerate(perims):
        poly = Polygon(v)
        if not poly.is_valid:
            poly = poly.buffer(0)
        geoms.append(poly)
        times.append(i)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"time_step": times},
        geometry=geoms,
        crs=crs,
    )
    
    # Calculate area and perimeter in proper units
    # If CRS is geographic, reproject to a projected CRS for accurate measurements
    if gdf.crs and gdf.crs.is_geographic:
        # Project to appropriate UTM zone for accurate measurements
        gdf_projected = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["area_ha"] = gdf_projected.geometry.area / 10000.0
        gdf["perim_m"] = gdf_projected.geometry.length
    else:
        # Already in projected coordinates
        gdf["area_ha"] = gdf.geometry.area / 10000.0
        gdf["perim_m"] = gdf.geometry.length
    
    gdf.to_file(out_path)


# --------------------------------------------------------------
# FBP → ROS/RAZ raster generation
# --------------------------------------------------------------
def generate_ros_raz_from_fbp(fuel_path: str, fbp_csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(fuel_path) as src:
        fuel = src.read(1)
        meta = src.meta.copy()
        meta.update(count=1, dtype="float32")

    fbp = pd.read_csv(fbp_csv_path)
    if not {"ID", "ROS", "RAZ"}.issubset(set(fbp.columns)):
        raise ValueError("FBP CSV must contain: ID, ROS, RAZ")
    fbp = fbp[["ID", "ROS", "RAZ"]].copy()
    fbp_grp = fbp.groupby("ID", as_index=True).mean(numeric_only=True)

    ROS_H = np.full_like(fuel, np.nan, dtype="float32")
    ROS_F = np.full_like(fuel, np.nan, dtype="float32")
    ROS_B = np.full_like(fuel, np.nan, dtype="float32")
    RAZ = np.full_like(fuel, np.nan, dtype="float32")

    for fuel_id, row in fbp_grp.iterrows():
        mask = (fuel == fuel_id)
        if not np.any(mask):
            continue
        rosh = float(row["ROS"])
        razd = float(row["RAZ"])
        ROS_H[mask] = rosh
        ROS_F[mask] = rosh * 0.30
        ROS_B[mask] = rosh * 0.10
        RAZ[mask] = razd

    for name, data in [("ROS_H", ROS_H), ("ROS_F", ROS_F), ("ROS_B", ROS_B), ("RAZ", RAZ)]:
        out_path = os.path.join(out_dir, f"{name}.tif")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)
        print(f"Saved {out_path}")


# --------------------------------------------------------------
# Stochastic helpers
# --------------------------------------------------------------

def sample_weather_conditions(weather_df, ignition_time=None, duration_hours=8):
    """Sample weather conditions for fire simulation."""
    if weather_df.empty:
        return None
    
    if ignition_time is None and "datetime" in weather_df.columns:
        min_time = weather_df["datetime"].min()
        max_time = weather_df["datetime"].max() - timedelta(hours=duration_hours)
        if max_time > min_time:
            random_seconds = random.uniform(0, (max_time - min_time).total_seconds())
            ignition_time = min_time + timedelta(seconds=random_seconds)
        else:
            ignition_time = min_time
    
    if ignition_time and "datetime" in weather_df.columns:
        end_time = ignition_time + timedelta(hours=duration_hours)
        weather_window = weather_df[
            (weather_df["datetime"] >= ignition_time) & 
            (weather_df["datetime"] <= end_time)
        ]
        if len(weather_window) > 0:
            return weather_window.to_dict('records')
    
    # Fallback to random sampling
    n_samples = min(len(weather_df), duration_hours)
    return weather_df.sample(n=n_samples).to_dict('records') if n_samples > 0 else None


def adjust_ros_for_weather(base_ros, base_raz, wind_speed, wind_dir, temp=None, rh=None):
    """Adjust ROS/RAZ based on weather."""
    wind_factor = np.exp(0.05 * wind_speed)
    
    if temp is not None and rh is not None:
        moisture_effect = 1.0 + 0.01 * (temp - 20) - 0.005 * (rh - 30)
        moisture_effect = np.clip(moisture_effect, 0.7, 1.3)
    else:
        moisture_effect = 1.0
    
    adjusted_ros = base_ros * wind_factor * moisture_effect
    adjusted_raz = 0.8 * wind_dir + 0.2 * base_raz
    
    return adjusted_ros, adjusted_raz


def create_weather_adjusted_grids(base_grids, weather_record):
    """Create grids adjusted for specific weather conditions."""
    wind_speed = weather_record.get('wind_speed', 10.0)
    wind_dir = weather_record.get('wind_dir', 0.0)
    temp = weather_record.get('temperature', 25.0)
    rh = weather_record.get('humidity', 40.0)
    
    adj_ros_h, _ = adjust_ros_for_weather(base_grids.ROS_H, base_grids.RAZ, wind_speed, wind_dir, temp, rh)
    adj_ros_f, _ = adjust_ros_for_weather(base_grids.ROS_F, base_grids.RAZ, wind_speed, wind_dir, temp, rh)
    adj_ros_b, adj_raz = adjust_ros_for_weather(base_grids.ROS_B, base_grids.RAZ, wind_speed, wind_dir, temp, rh)
    
    return Grids(
        fuel=base_grids.fuel,
        ROS_H=adj_ros_h,
        ROS_F=adj_ros_f,
        ROS_B=adj_ros_b,
        RAZ=adj_raz,
        transform=base_grids.transform,
        crs=base_grids.crs,
        nonfuel_values=base_grids.nonfuel_values
    )


def perturb_ros_raz_grids(grids, variability_factor=0.2, seed=None):
    """Add spatial randomness to ROS/RAZ grids."""
    if seed is not None:
        np.random.seed(seed)
    
    shape = grids.ROS_H.shape
    ros_mult = np.random.lognormal(mean=0, sigma=variability_factor, size=shape)
    raz_offset = np.random.normal(0, 15, size=shape)

    return Grids(
        fuel=grids.fuel,
        ROS_H=grids.ROS_H * ros_mult,
        ROS_F=grids.ROS_F * ros_mult,
        ROS_B=grids.ROS_B * ros_mult,
        RAZ=(grids.RAZ + raz_offset) % 360,
        transform=grids.transform,
        crs=grids.crs,
        nonfuel_values=grids.nonfuel_values
    )


def sample_ignition_size(distribution="lognormal") -> float:
    if distribution == "lognormal":
        d = np.random.lognormal(mean=1.6, sigma=0.8)
        return float(np.clip(d, 0.5, 100.0))
    return 10.0


def sample_burning_duration(season="summer", cause="lightning") -> float:
    if cause == "lightning":
        base = random.uniform(240, 720)
    else:
        base = random.uniform(120, 480)
    
    if season == "spring":
        base *= random.uniform(0.5, 0.8)
    elif season == "summer":
        base *= random.uniform(0.8, 1.5)
    elif season == "fall":
        base *= random.uniform(0.6, 1.0)
    
    return float(base)


# --------------------------------------------------------------
# Stochastic simulation
# --------------------------------------------------------------

def simulate_fire_stochastic(vertices_xy, grids_base, settings,
                             weather_conditions=None, max_minutes=None,
                             max_steps=10_000, variability_factor=0.2, seed=None):
    """Simulate fire with stochastic variability."""
    
    # Apply weather if provided
    if weather_conditions:
        weather_rec = weather_conditions[0] if isinstance(weather_conditions, list) else weather_conditions
        grids = create_weather_adjusted_grids(grids_base, weather_rec)
    else:
        grids = grids_base
    
    # Add fine-scale spatial variability
    if variability_factor > 0:
        grids = perturb_ros_raz_grids(grids, variability_factor, seed)
    
    if max_minutes is None:
        max_minutes = 480
    
    perims = [ensure_ccw(vertices_xy)]
    t_elapsed = 0.0
    weather_idx = 0
    consecutive_failures = 0

    for step in range(max_steps):
        # Update weather if time-varying
        if weather_conditions and isinstance(weather_conditions, list):
            if t_elapsed > (weather_idx + 1) * 60 and weather_idx + 1 < len(weather_conditions):
                weather_idx += 1
                weather_rec = weather_conditions[weather_idx]
                grids = create_weather_adjusted_grids(grids_base, weather_rec)
                if variability_factor > 0:
                    grids = perturb_ros_raz_grids(grids, variability_factor, seed)
        
        nxt, info = propagate_vertices(perims[-1], grids, settings)
        
        if not info.get("moved"):
            consecutive_failures += 1
            if consecutive_failures >= 5:
                break
            continue
        else:
            consecutive_failures = 0

        nxt = insert_new_vertices(nxt, settings.distance_resolution_m)
        nxt = remove_close_vertices(nxt, settings.distance_resolution_m)

        stats = fire_perimeter_stats(nxt)
        
        # Print every 10 steps
        if step % 10 == 0 or step < 3:
            print(f"  Step {step+1:03d} | Area={stats['area_m2']/1e4:.2f}ha | Time={t_elapsed:.0f}min")

        if settings.export_outputs:
            export_vertex_stats(nxt, step + 1, stats, settings.output_dir)

        perims.append(nxt)
        t_elapsed += info["dt_s"] / 60.0
        
        if t_elapsed >= max_minutes:
            break

    # Final stats
    if len(perims) > 1:
        final_stats = fire_perimeter_stats(perims[-1])
    else:
        final_stats = {"area_m2": 0, "perimeter_m": 0}

    if settings.export_outputs and len(perims) > 1:
        try:
            export_perimeters_shapefile(perims, grids.crs, f"{settings.output_dir}/FirePerimeters.shp")
        except Exception as e:
            print(f"  ⚠️ Export failed: {e}")

    return perims, {
        "final_area_ha": final_stats["area_m2"] / 1e4,
        "final_perimeter_m": final_stats["perimeter_m"],
        "duration_min": t_elapsed,
        "n_steps": len(perims) - 1,
        "success": len(perims) > 1
    }


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

if __name__ == "__main__":
    IGNITION_PATH = "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Ignition/Ignition_Polygons/ignition_summary.csv"
    WEATHER_PATH = "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Weather/fire_weather_list.csv"
    FUEL_PATH = "/Users/Martyn/GitRepos/BurnP3+/Kelowna/PreProcessing/DerivedTerrain/fuels.tif"
    FBP_CSV_PATH = "/Users/Martyn/GitRepos/BritishColumbia/FBP/Output/FBP_Output_Summer2010.csv"
    DERIVED_DIR = "/Users/Martyn/GitRepos/BurnP3+/Kelowna/PreProcessing/DerivedTerrain"
    OUTPUT_DIR = "/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Output/FirePerimeters"

    # Generate ROS/RAZ rasters if needed
    ros_paths = [os.path.join(DERIVED_DIR, f"{n}.tif") for n in ["ROS_H", "ROS_F", "ROS_B", "RAZ"]]
    if not all(os.path.exists(p) for p in ros_paths):
        print("Generating ROS/RAZ rasters from FBP CSV...")
        generate_ros_raz_from_fbp(FUEL_PATH, FBP_CSV_PATH, DERIVED_DIR)

    # Load grids
    with rasterio.open(FUEL_PATH) as f:
        fuel, transform, crs = f.read(1), f.transform, f.crs
    with rasterio.open(ros_paths[0]) as f:
        ROS_H = f.read(1) / 60.0  # Convert m/min → m/s
    with rasterio.open(ros_paths[1]) as f:
        ROS_F = f.read(1) / 60.0
    with rasterio.open(ros_paths[2]) as f:
        ROS_B = f.read(1) / 60.0
    with rasterio.open(ros_paths[3]) as f:
        RAZ = f.read(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grids = Grids(fuel, ROS_H, ROS_F, ROS_B, RAZ, transform, crs)

    # Load ignition summary
    ign_summary = pd.read_csv(IGNITION_PATH)
    print(f"Loaded ignition summary with {len(ign_summary)} entries")

    # Load weather (optional)
    weather = pd.DataFrame()
    if os.path.exists(WEATHER_PATH):
        weather = pd.read_csv(WEATHER_PATH)
        if "DATE_TIME" in weather.columns:
            weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")
            weather.rename(columns={"DATE_TIME": "datetime"}, inplace=True)
        elif "datetime" in weather.columns:
            weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
        
        rename_map = {
            "HOURLY_WIND_SPEED": "wind_speed",
            "HOURLY_WIND_DIRECTION": "wind_dir",
            "HOURLY_TEMPERATURE": "temperature",
            "HOURLY_RELATIVE_HUMIDITY": "humidity",
        }
        weather.rename(columns={k: v for k, v in rename_map.items() if k in weather.columns}, inplace=True)
        
        if "datetime" in weather.columns and not weather.empty:
            print(f"Loaded weather: {len(weather)} records from {weather['datetime'].min()} to {weather['datetime'].max()}")
        else:
            print("Weather loaded but no datetime column found")
    else:
        print("⚠️ Weather file not found")

    # Process ignitions
    processed_ignitions = 0
    skipped_ignitions = 0

    # A) Direct coordinates in CSV
    if "x" in ign_summary.columns and "y" in ign_summary.columns:
        coord_rows = ign_summary.copy()
        coord_rows["x_num"] = pd.to_numeric(coord_rows["x"], errors="coerce")
        coord_rows["y_num"] = pd.to_numeric(coord_rows["y"], errors="coerce")
        coord_rows = coord_rows[coord_rows["x_num"].notna() & coord_rows["y_num"].notna()]

        if len(coord_rows) > 0:
            print(f"\nProcessing {len(coord_rows)} ignition point(s) from CSV...")
            for _, row in coord_rows.iterrows():
                x, y = float(row["x_num"]), float(row["y_num"])
                season = row.get("season", "summer")
                cause = row.get("cause", "unknown")
                iteration = int(row.get("iteration", processed_ignitions))

                # Validate location
                is_valid, reason = validate_ignition_location(x, y, grids)
                if not is_valid:
                    print(f"⚠️ Skipping fire #{processed_ignitions}: {reason}")
                    skipped_ignitions += 1
                    processed_ignitions += 1
                    continue

                # Sample stochastic parameters
                ignition_diam = sample_ignition_size("lognormal")
                burn_minutes = sample_burning_duration(season, cause)
                sim_seed = hash(f"{iteration}_{x}_{y}") % (2**32)

                # Sample weather
                weather_sample = None
                if not weather.empty:
                    weather_sample = sample_weather_conditions(weather, duration_hours=int(burn_minutes/60))

                # Setup
                fire_output_dir = os.path.join(OUTPUT_DIR, f"fire_{processed_ignitions:04d}_iter{iteration}")
                os.makedirs(fire_output_dir, exist_ok=True)

                settings = EngineSettings(
                    distance_resolution_m=30.0,
                    smoothing_omega=0.4,
                    export_outputs=True,
                    output_dir=fire_output_dir
                )

                start = ignition_point(x, y, n_vertices=16, diameter_m=ignition_diam)

                # Simulate
                try:
                    print(f"\nFire #{processed_ignitions} | ({x:.2f}, {y:.2f}) | {season}/{cause}")
                    perims, fire_stats = simulate_fire_stochastic(
                        start, grids, settings,
                        weather_conditions=weather_sample,
                        max_minutes=burn_minutes,
                        variability_factor=0.2,
                        seed=sim_seed
                    )
                    
                    if fire_stats["success"]:
                        print(f"✓ Final: {fire_stats['final_area_ha']:.2f} ha in {fire_stats['duration_min']:.0f} min")
                    else:
                        print(f"⚠️ Fire failed to spread")
                
                except Exception as e:
                    print(f"❌ Fire crashed: {e}")
                
                processed_ignitions += 1

    # B) Shapefiles listed in summary
    shp_series = ign_summary.get("shapefile_path", pd.Series([], dtype=object))
    valid_paths = [p for p in shp_series if isinstance(p, str) and p.strip() and os.path.exists(p)]
    unique_paths = sorted(set(valid_paths))
    
    if unique_paths:
        print(f"\nProcessing {len(unique_paths)} ignition shapefile(s)...")
        
        for shp_path in unique_paths:
            gdf = gpd.read_file(shp_path)
            if gdf.empty:
                print(f"⚠️ Empty shapefile: {shp_path}")
                continue

            # Get metadata
            shp_rows = ign_summary[ign_summary["shapefile_path"] == shp_path]
            season = shp_rows["season"].iloc[0] if "season" in shp_rows.columns and not shp_rows.empty else "summer"
            cause = shp_rows["cause"].iloc[0] if "cause" in shp_rows.columns and not shp_rows.empty else "unknown"
            iter_in = int(shp_rows["iteration"].iloc[0]) if "iteration" in shp_rows.columns and not shp_rows.empty else None

            # Reproject if needed
            if gdf.crs and crs and gdf.crs != crs:
                gdf = gdf.to_crs(crs)
            
            gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
            print(f"\nLoaded {len(gdf)} points from {os.path.basename(shp_path)}")

            for _, prow in gdf.iterrows():
                x, y = float(prow["x"]), float(prow["y"])
                iteration = iter_in if iter_in is not None else processed_ignitions

                # Validate location
                is_valid, reason = validate_ignition_location(x, y, grids)
                if not is_valid:
                    if skipped_ignitions < 10:  # Only print first 10 skips
                        print(f"⚠️ Skip #{processed_ignitions}: {reason}")
                    skipped_ignitions += 1
                    processed_ignitions += 1
                    continue

                # Sample stochastic parameters
                ignition_diam = sample_ignition_size("lognormal")
                burn_minutes = sample_burning_duration(season, cause)
                sim_seed = hash(f"{iteration}_{x}_{y}") % (2**32)

                # Sample weather
                weather_sample = None
                if not weather.empty:
                    weather_sample = sample_weather_conditions(weather, duration_hours=int(burn_minutes/60))

                # Setup
                fire_output_dir = os.path.join(OUTPUT_DIR, f"fire_{processed_ignitions:04d}_iter{iteration}")
                os.makedirs(fire_output_dir, exist_ok=True)

                settings = EngineSettings(
                    distance_resolution_m=30.0,
                    smoothing_omega=0.4,
                    export_outputs=True,
                    output_dir=fire_output_dir
                )

                start = ignition_point(x, y, n_vertices=16, diameter_m=ignition_diam)

                # Simulate
                try:
                    if processed_ignitions % 100 == 0 or processed_ignitions < 5:
                        print(f"\nFire #{processed_ignitions} | ({x:.2f}, {y:.2f}) | {season}/{cause}")
                    
                    perims, fire_stats = simulate_fire_stochastic(
                        start, grids, settings,
                        weather_conditions=weather_sample,
                        max_minutes=burn_minutes,
                        variability_factor=0.2,
                        seed=sim_seed
                    )
                    
                    if fire_stats["success"] and (processed_ignitions % 100 == 0 or processed_ignitions < 5):
                        print(f"✓ Final: {fire_stats['final_area_ha']:.2f} ha in {fire_stats['duration_min']:.0f} min")
                
                except Exception as e:
                    print(f"❌ Fire #{processed_ignitions} crashed: {e}")
                
                processed_ignitions += 1

    # Combine all perimeters
    print(f"\n🧩 Combining all fire perimeters...")
    all_records = []
    for root, _, files in os.walk(OUTPUT_DIR):
        if "FirePerimeters.shp" in files:
            try:
                gdf = gpd.read_file(os.path.join(root, "FirePerimeters.shp"))
                gdf["fire_id"] = os.path.basename(root)
                if "area_ha" not in gdf.columns:
                    gdf["area_ha"] = gdf.geometry.area / 10000.0
                if "perim_m" not in gdf.columns:
                    gdf["perim_m"] = gdf.geometry.length
                all_records.append(gdf)
            except Exception as e:
                print(f"⚠️ Could not read {root}: {e}")

    if all_records:
        all_perims = gpd.GeoDataFrame(pd.concat(all_records, ignore_index=True), crs=crs)
        out_all = os.path.join(OUTPUT_DIR, "ALL_Perimeters.shp")
        all_perims.to_file(out_all)
        print(f"✅ Combined {len(all_perims)} perimeter features from {len(all_records)} fires")
    else:
        print("⚠️ No perimeters found to combine")

    print(f"\n{'='*60}")
    print(f"✅ SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total ignitions processed: {processed_ignitions}")
    print(f"Successful fires: {processed_ignitions - skipped_ignitions}")
    print(f"Skipped (invalid location): {skipped_ignitions}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")