"""
Fire simulation orchestration for Ignacio.

This module provides the main entry points for running fire growth simulations,
coordinating the terrain, weather, ignition, FBP, and spread components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from ignacio.config import IgnacioConfig
from ignacio.fbp import compute_ros, compute_ros_components
from ignacio.ignition import IgnitionPoint, IgnitionSet, generate_ignitions, save_ignitions
from ignacio.io import RasterData, read_raster, read_raster_int, write_raster, write_vector, read_vector, rasterize_geometries
from ignacio.spread import (
    FireParameterGrid,
    FirePerimeterHistory,
    create_initial_perimeter,
    simulate_fire_spread,
)
from ignacio.terrain import TerrainGrids, build_terrain_grids, compute_slope_factor
from ignacio.weather import (
    FireWeatherList,
    get_representative_weather,
    load_weather_data,
    process_fire_weather,
    save_fire_weather_list,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FireResult:
    """Result from a single fire simulation."""
    
    ignition: IgnitionPoint
    history: FirePerimeterHistory
    final_area_ha: float = 0.0
    final_perimeter_m: float = 0.0
    dur_min: float = 0.0
    
    def get_final_polygon(self, output_crs: str, source_crs: str | None = None) -> gpd.GeoDataFrame:
        """Get final fire perimeter as GeoDataFrame.
        
        Parameters
        ----------
        output_crs : str
            Desired output CRS.
        source_crs : str, optional
            CRS of the source coordinates. If different from output_crs,
            the polygon will be reprojected.
        """
        x, y = self.history.get_final_perimeter()
        if len(x) == 0:
            return gpd.GeoDataFrame({"geometry": []}, crs=output_crs)
        
        # Close polygon
        coords = list(zip(x, y))
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        polygon = Polygon(coords)
        
        # Create GeoDataFrame with source CRS
        gdf = gpd.GeoDataFrame(
            {
                "area_ha": [self.final_area_ha],
                "perim_m": [self.final_perimeter_m],
                "dur_min": [self.dur_min],
                "cause": [self.ignition.cause],
                "season": [self.ignition.season],
                "iteration": [self.ignition.iteration],
                "geometry": [polygon],
            },
            crs=source_crs if source_crs else output_crs,
        )
        
        # Reproject if needed
        if source_crs and source_crs != output_crs:
            gdf = gdf.to_crs(output_crs)
        
        return gdf


@dataclass
class SimulationResults:
    """Results from complete simulation run."""
    
    fires: list[FireResult] = field(default_factory=list)
    config: IgnacioConfig | None = None
    terrain: TerrainGrids | None = None
    weather: FireWeatherList | None = None
    ignitions: IgnitionSet | None = None
    
    @property
    def n_fires(self) -> int:
        """Number of simulated fires."""
        return len(self.fires)
    
    @property
    def total_area_ha(self) -> float:
        """Total burned area in hectares."""
        return sum(f.final_area_ha for f in self.fires)
    
    def get_all_perimeters(self, source_crs: str | None = None) -> gpd.GeoDataFrame:
        """Combine all fire perimeters into single GeoDataFrame.
        
        Parameters
        ----------
        source_crs : str, optional
            CRS of the source coordinates (terrain CRS).
        """
        if not self.fires:
            return gpd.GeoDataFrame()
        
        output_crs = self.config.crs.output_crs if self.config else "EPSG:4326"
        gdfs = [f.get_final_polygon(output_crs, source_crs=source_crs) for f in self.fires]
        
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=output_crs)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics for all fires."""
        data = []
        for i, fire in enumerate(self.fires):
            data.append({
                "fire_id": i,
                "x": fire.ignition.x,
                "y": fire.ignition.y,
                "cause": fire.ignition.cause,
                "season": fire.ignition.season,
                "iteration": fire.ignition.iteration,
                "area_ha": fire.final_area_ha,
                "perimeter_m": fire.final_perimeter_m,
                "dur_min": fire.dur_min,
            })
        return pd.DataFrame(data)


# =============================================================================
# Parameter Grid Building
# =============================================================================


def build_parameter_grid(
    config: IgnacioConfig,
    terrain: TerrainGrids,
    weather: FireWeatherList,
    n_timesteps: int | None = None,
    hourly_data: pd.DataFrame | None = None,
    start_datetime: pd.Timestamp | None = None,
) -> FireParameterGrid:
    """
    Build spatially-varying fire parameter grid.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
    terrain : TerrainGrids
        Terrain data.
    weather : FireWeatherList
        Fire weather data (used for representative values if no hourly data).
    n_timesteps : int, optional
        Number of time steps. Defaults to max_duration / dt.
    hourly_data : DataFrame, optional
        Hourly weather data for time-varying simulation.
    start_datetime : Timestamp, optional
        Simulation start time for time-varying weather.
        
    Returns
    -------
    FireParameterGrid
        Grid of ROS parameters over space and time.
    """
    from ignacio.weather import HourlyWeatherInterpolator
    
    fbp_config = config.fbp
    sim_config = config.simulation
    
    if n_timesteps is None:
        n_timesteps = int(sim_config.max_duration / sim_config.dt)
    
    # Determine if using time-varying weather
    use_time_varying = (
        sim_config.time_varying_weather 
        and hourly_data is not None 
        and len(hourly_data) > 0
    )
    
    if use_time_varying:
        logger.info("Using time-varying weather interpolation")
        interpolator = HourlyWeatherInterpolator(hourly_data=hourly_data)
        
        # Determine start time
        if start_datetime is None:
            if sim_config.start_datetime:
                start_datetime = pd.Timestamp(sim_config.start_datetime)
            else:
                # Use a representative date from the weather data with default hour
                if "DATE" in hourly_data.columns:
                    dates = pd.to_datetime(hourly_data["DATE"]).dropna()
                    if len(dates) > 0:
                        mid_date = dates.iloc[len(dates) // 2]
                        start_datetime = pd.Timestamp(mid_date) + pd.Timedelta(hours=sim_config.default_start_hour)
                        logger.info(f"Using mid-season start: {start_datetime}")
                
                if start_datetime is None:
                    start_datetime = pd.Timestamp("2024-07-15 12:00:00")
                    logger.warning(f"No date info found, using default: {start_datetime}")
    else:
        interpolator = None
        start_datetime = None
    
    # Get representative/initial weather
    weather_vals = get_representative_weather(weather, config)
    # Combined fuel break mask for zero_ros treatment (can come from multiple breaks)
    fuel_break_zero_ros_mask = None
    
    # Load fuel grid
    fuel_raster = read_raster_int(config.fuel.path)
    fuel_grid = fuel_raster.data

    # Get coordinate arrays from terrain (this is our reference grid)
    x_coords, y_coords = terrain.get_coordinate_arrays()
    ny, nx = terrain.shape

    logger.info(f"Terrain grid shape: {ny} x {nx}")
    logger.info(f"Fuel grid shape: {fuel_grid.shape[0]} x {fuel_grid.shape[1]}")
    logger.debug(f"Terrain CRS: {terrain.dem.crs}, Fuel CRS: {fuel_raster.crs}")

    # Check if fuel grid matches terrain grid
    if fuel_grid.shape != terrain.shape:
        logger.warning(
            f"Fuel grid shape {fuel_grid.shape} differs from terrain {terrain.shape}. "
            f"Resampling fuel grid to match terrain."
        )
        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.warp import reproject

            fuel_resampled = np.zeros((ny, nx), dtype=fuel_grid.dtype)
            reproject(
                source=fuel_grid,
                destination=fuel_resampled,
                src_transform=fuel_raster.transform,
                src_crs=fuel_raster.crs,
                dst_transform=terrain.dem.transform,
                dst_crs=terrain.dem.crs,
                resampling=Resampling.nearest,
            )
            fuel_grid = fuel_resampled
            logger.info(f"Reprojected fuel grid to shape: {fuel_grid.shape}")
        except Exception as e:
            logger.warning(f"Rasterio reprojection failed: {e}, falling back to simple zoom")
            from scipy.ndimage import zoom
            zoom_factors = (ny / fuel_raster.data.shape[0], nx / fuel_raster.data.shape[1])
            fuel_grid = zoom(fuel_raster.data, zoom_factors, order=0)
            logger.info(f"Zoomed fuel grid to shape: {fuel_grid.shape}")

    # Load and process fuel breaks if enabled
    # Supports either:
    #   - config.fuel_break (single FuelBreakConfig)
    #   - config.fuel_breaks (list[FuelBreakConfig])
    #   - config.fuel_break being a list/tuple (for forward-compat)
    def _iter_fuel_breaks() -> list[Any]:
        """Return a list of fuel break config objects (backward compatible).

        NOTE: We intentionally return BOTH enabled and disabled entries so the
        logs can confirm what was parsed from YAML.
        """
        out: list[Any] = []

        # Backward-compatible single fuel_break (allow it to be a list/tuple too)
        fb_single = getattr(config, "fuel_break", None)
        if fb_single is not None:
            if isinstance(fb_single, (list, tuple)):
                out.extend(list(fb_single))
            else:
                out.append(fb_single)

        # Preferred new API: additional fuel_breaks list
        fb_list = getattr(config, "fuel_breaks", None)
        if fb_list:
            out.extend(list(fb_list))

        return out

    fuel_breaks = _iter_fuel_breaks()

    if fuel_breaks:
        enabled_ct = sum(1 for fb in fuel_breaks if getattr(fb, "enabled", False))
        logger.info(
            f"Found {len(fuel_breaks)} fuel break config(s) in config (enabled={enabled_ct}, disabled={len(fuel_breaks)-enabled_ct})"
        )
        for i, fb in enumerate(fuel_breaks, start=1):
            logger.info(
                f"  Fuel break {i}: enabled={getattr(fb, 'enabled', False)}, path={getattr(fb, 'path', None)}, treatment={getattr(fb, 'treatment_method', None)}, fuel_code={getattr(fb, 'fuel_code', None)}"
            )

    if fuel_breaks:
        logger.info(f"Processing {len([fb for fb in fuel_breaks if getattr(fb, 'enabled', False)])} enabled fuel break layer(s)")

        # Build a combined mask for zero_ros treatment across all breaks
        fuel_break_zero_ros_mask = np.zeros((ny, nx), dtype=bool)

        for fb_idx, fb_cfg in enumerate(fuel_breaks, start=1):
            if not getattr(fb_cfg, "enabled", False):
                logger.info(f"Skipping fuel break {fb_idx}: enabled=False")
                continue
            logger.info(f"Loading fuel break {fb_idx}/{len(fuel_breaks)} from: {fb_cfg.path}")

            try:
                fuel_break_path = Path(fb_cfg.path)

                # -----------------------------------------------------------------
                # Load break layer (raster or vector) and build mask on terrain grid
                # -----------------------------------------------------------------
                if fuel_break_path.suffix.lower() in [".tif", ".tiff"]:
                    # Raster fuel break
                    fuel_break_raster = read_raster_int(fuel_break_path)
                    fuel_break_grid = fuel_break_raster.data

                    # Resample if needed
                    if fuel_break_grid.shape != (ny, nx):
                        logger.info(
                            f"Resampling fuel break {fb_idx} from {fuel_break_grid.shape} to {(ny, nx)}"
                        )
                        try:
                            import rasterio
                            from rasterio.enums import Resampling
                            from rasterio.warp import reproject

                            fuel_break_resampled = np.zeros((ny, nx), dtype=fuel_break_grid.dtype)
                            reproject(
                                source=fuel_break_grid,
                                destination=fuel_break_resampled,
                                src_transform=fuel_break_raster.transform,
                                src_crs=fuel_break_raster.crs,
                                dst_transform=terrain.dem.transform,
                                dst_crs=terrain.dem.crs,
                                resampling=Resampling.nearest,
                            )
                            fuel_break_grid = fuel_break_resampled
                            logger.info(f"Reprojected fuel break {fb_idx} to shape: {fuel_break_grid.shape}")
                        except Exception as e:
                            logger.warning(f"Rasterio reprojection failed for fuel break {fb_idx}: {e}, using zoom")
                            from scipy.ndimage import zoom

                            zoom_factors = (
                                ny / fuel_break_raster.data.shape[0],
                                nx / fuel_break_raster.data.shape[1],
                            )
                            fuel_break_grid = zoom(fuel_break_raster.data, zoom_factors, order=0)
                            logger.info(f"Zoomed fuel break {fb_idx} to shape: {fuel_break_grid.shape}")

                    # Create mask for fuel break cells (any non-zero value)
                    # NOTE: keep simple and robust; users can provide a binary break raster
                    fb_mask = (fuel_break_grid > 0)

                else:
                    # Vector fuel break (shapefile, GeoJSON, etc.)
                    fuel_break_gdf = read_vector(fuel_break_path, target_crs=terrain.dem.crs)
                    logger.info(f"Loaded {len(fuel_break_gdf)} fuel break features (break {fb_idx})")

                    # Buffer geometries if specified (buffer in meters; use projected CRS if needed)
                    if fb_cfg.buffer_m is not None and fb_cfg.buffer_m > 0:
                        logger.info(f"Buffering fuel break {fb_idx} by {fb_cfg.buffer_m} m")

                        if terrain.dem.crs is not None:
                            try:
                                import pyproj

                                crs_obj = pyproj.CRS.from_user_input(terrain.dem.crs)
                                if crs_obj.is_geographic:
                                    centroid = fuel_break_gdf.to_crs("EPSG:4326").geometry.unary_union.centroid
                                    lon, lat = float(centroid.x), float(centroid.y)
                                    utm_crs = pyproj.CRS.from_user_input(
                                        f"+proj=utm +zone={(int((lon + 180) / 6) + 1)} +datum=WGS84 +units=m +no_defs"
                                        + (" +south" if lat < 0 else "")
                                    )

                                    gdf_utm = fuel_break_gdf.to_crs(utm_crs)
                                    gdf_utm.geometry = gdf_utm.geometry.buffer(fb_cfg.buffer_m)
                                    fuel_break_gdf = gdf_utm.to_crs(terrain.dem.crs)
                                else:
                                    fuel_break_gdf.geometry = fuel_break_gdf.geometry.buffer(fb_cfg.buffer_m)
                            except Exception as e:
                                logger.warning(
                                    f"Fuel break {fb_idx} buffering CRS handling failed ({e}); buffering in layer CRS"
                                )
                                fuel_break_gdf.geometry = fuel_break_gdf.geometry.buffer(fb_cfg.buffer_m)
                        else:
                            fuel_break_gdf.geometry = fuel_break_gdf.geometry.buffer(fb_cfg.buffer_m)

                    # Rasterize to match terrain grid
                    logger.info(f"Rasterizing {len(fuel_break_gdf)} fuel break features (break {fb_idx})")
                    fuel_break_grid = rasterize_geometries(
                        fuel_break_gdf,
                        reference=terrain.dem,
                        value=1,
                        fill=0,
                        dtype="uint8",
                        all_touched=fb_cfg.all_touched,
                    )
                    fb_mask = (fuel_break_grid == 1)

                # -------------------------------------------------------------
                # Apply preserve_non_fuel filter (per break)
                # -------------------------------------------------------------
                if getattr(fb_cfg, "preserve_non_fuel", False):
                    non_fuel_mask = np.isin(fuel_grid, config.fuel.non_fuel_codes)
                    fb_apply_mask = fb_mask & ~non_fuel_mask
                else:
                    fb_apply_mask = fb_mask

                n_cells = int(np.sum(fb_apply_mask))
                if n_cells <= 0:
                    logger.warning(
                        f"Fuel break {fb_idx}: no cells affected (check geometry/buffer and preserve_non_fuel setting)"
                    )
                    continue

                # -------------------------------------------------------------
                # Apply treatment
                # -------------------------------------------------------------
                treatment = getattr(fb_cfg, "treatment_method", "fuel_code")

                if treatment == "fuel_code":
                    # Overwrite fuel codes inside mask
                    fuel_grid[fb_apply_mask] = fb_cfg.fuel_code
                    logger.info(
                        f"Applied fuel break {fb_idx} (fuel_code): {n_cells} cells set to fuel code {fb_cfg.fuel_code}"
                    )

                elif treatment == "zero_ros":
                    # Accumulate into combined mask for later ROS forcing
                    fuel_break_zero_ros_mask |= fb_apply_mask
                    logger.info(
                        f"Applied fuel break {fb_idx} (zero_ros): {n_cells} cells will have ROS forced to 0"
                    )

                else:
                    logger.warning(
                        f"Fuel break {fb_idx}: unknown treatment_method='{treatment}'. No treatment applied."
                    )

            except Exception as e:
                logger.error(f"Failed to load/apply fuel break {fb_idx}: {e}")
                logger.warning("Continuing simulation without this fuel break")
                import traceback

                logger.debug(traceback.format_exc())

        # If no cells ended up in the zero_ros mask, disable it
        if fuel_break_zero_ros_mask is not None and not np.any(fuel_break_zero_ros_mask):
            fuel_break_zero_ros_mask = None
    
    # Initialize arrays
    ros_arr = np.zeros((n_timesteps, ny, nx), dtype=np.float32)
    bros_arr = np.zeros_like(ros_arr)
    fros_arr = np.zeros_like(ros_arr)
    raz_arr = np.zeros_like(ros_arr)
    
    logger.info(f"Building parameter grid: {n_timesteps} timesteps, {ny}x{nx} cells")
    
    # Get unique fuel types
    unique_fuels = np.unique(fuel_grid)
    non_fuel = config.fuel.non_fuel_codes
    
    logger.info(f"Unique fuel codes in grid: {sorted([int(f) for f in unique_fuels if np.isfinite(f)])}")
    
    # Build ROS grid for each timestep
    if use_time_varying:
        # Time-varying weather: compute ROS per timestep
        dt_minutes = sim_config.dt
        log_interval = max(1, n_timesteps // 5)  # Log ~5 times

        for t in range(n_timesteps):
            sim_time = start_datetime + pd.Timedelta(minutes=t * dt_minutes)
            weather_t = interpolator.interpolate_at(sim_time)

            # Get weather values for this timestep
            isi_t = weather_t.get("ISI", weather_vals["isi"])
            bui_t = weather_t.get("BUI", weather_vals["bui"])
            wind_dir_t = weather_t.get("WD", weather_t.get("WIND_DIRECTION", weather_vals["wind_direction"]))
            temp_t = weather_t.get("TEMP", weather_t.get("TEMPERATURE", weather_vals["temperature"]))
            rh_t = weather_t.get("RH", weather_t.get("RELATIVE_HUMIDITY", weather_vals["relative_humidity"]))

            # Compute base ROS for each fuel type
            base_ros = np.zeros((ny, nx), dtype=np.float32)
            for fuel_id in unique_fuels:
                if fuel_id in non_fuel or np.isnan(fuel_id):
                    continue

                ros = compute_ros(
                    fuel_type=int(fuel_id),
                    isi=isi_t,
                    bui=bui_t,
                    fmc=fbp_config.fmc,
                    curing=fbp_config.curing,
                    fuel_lookup=config.fuel.fuel_lookup,
                )
                mask = fuel_grid == fuel_id
                base_ros[mask] = ros

            # Compute slope factor for this wind direction
            slope_factor = compute_slope_factor(
                terrain.slope_deg,
                terrain.aspect_deg,
                wind_dir_t,
                fbp_config.slope_factor,
            )

            # Apply slope correction
            ros_corrected = base_ros * slope_factor

            # Apply fuel breaks (zero_ros method) - combined mask across all breaks
            if fuel_break_zero_ros_mask is not None:
                ros_corrected[fuel_break_zero_ros_mask] = 0.0

            # Elevation adjustment if enabled
            if fbp_config.elevation_adjustment.enabled:
                from ignacio.terrain import compute_elevation_adjustment

                elev_factor = compute_elevation_adjustment(
                    terrain.dem.data,
                    temp_t,
                    rh_t,
                    fbp_config.elevation_adjustment.lapse_rate,
                    fbp_config.elevation_adjustment.reference_temp,
                    fbp_config.elevation_adjustment.reference_rh,
                )
                ros_corrected *= elev_factor

            # Compute other ROS components
            backing = fbp_config.backing_fraction
            lb_ratio = fbp_config.length_to_breadth

            ros_arr[t] = ros_corrected
            bros_arr[t] = backing * ros_corrected
            fros_arr[t] = (ros_corrected + bros_arr[t]) / (2.0 * lb_ratio)
            raz_arr[t] = np.radians((wind_dir_t + 180.0) % 360.0)

            if t % log_interval == 0:
                hour = sim_time.hour
                valid_ros = ros_corrected[ros_corrected > 0]
                if len(valid_ros) > 0:
                    logger.debug(
                        f"  t={t}: {sim_time.strftime('%H:%M')}, ISI={isi_t:.1f}, "
                        f"WD={wind_dir_t:.0f}°, mean ROS={np.mean(valid_ros):.2f} m/min"
                    )

        # Log summary
        logger.info(
            f"Time-varying weather: {start_datetime.strftime('%Y-%m-%d %H:%M')} to "
            f"{(start_datetime + pd.Timedelta(minutes=n_timesteps * dt_minutes)).strftime('%H:%M')}"
        )

    else:
        # Static weather: compute once and broadcast
        # Compute base ROS for each fuel type
        ros_by_fuel = {}
        for fuel_id in unique_fuels:
            if fuel_id in non_fuel or np.isnan(fuel_id):
                ros_by_fuel[fuel_id] = 0.0
                continue

            ros = compute_ros(
                fuel_type=int(fuel_id),
                isi=weather_vals["isi"],
                bui=weather_vals["bui"],
                fmc=fbp_config.fmc,
                curing=fbp_config.curing,
                fuel_lookup=config.fuel.fuel_lookup,
            )
            ros_by_fuel[fuel_id] = ros

            if ros > 0:
                fuel_name = config.fuel.fuel_lookup.get(int(fuel_id), f"ID-{int(fuel_id)}")
                logger.debug(f"Fuel {fuel_name} (code {int(fuel_id)}): ROS = {ros:.2f} m/min")

        # Compute slope factor
        wind_dir = weather_vals["wind_direction"]
        slope_factor = compute_slope_factor(
            terrain.slope_deg,
            terrain.aspect_deg,
            wind_dir,
            fbp_config.slope_factor,
        )

        # Build ROS grid
        base_ros = np.zeros((ny, nx), dtype=np.float32)
        for fuel_id, ros in ros_by_fuel.items():
            if np.isnan(fuel_id):
                continue
            mask = fuel_grid == fuel_id
            base_ros[mask] = ros

        # Apply slope correction
        ros_corrected = base_ros * slope_factor

        # Apply fuel breaks (zero_ros method) - combined mask across all breaks
        if fuel_break_zero_ros_mask is not None:
            ros_corrected[fuel_break_zero_ros_mask] = 0.0

        # Elevation adjustment if enabled
        if fbp_config.elevation_adjustment.enabled:
            from ignacio.terrain import compute_elevation_adjustment

            elev_factor = compute_elevation_adjustment(
                terrain.dem.data,
                weather_vals["temperature"],
                weather_vals["relative_humidity"],
                fbp_config.elevation_adjustment.lapse_rate,
                fbp_config.elevation_adjustment.reference_temp,
                fbp_config.elevation_adjustment.reference_rh,
            )
            ros_corrected *= elev_factor

        # Compute other ROS components
        backing = fbp_config.backing_fraction
        lb_ratio = fbp_config.length_to_breadth

        bros_base = backing * ros_corrected
        fros_base = (ros_corrected + bros_base) / (2.0 * lb_ratio)

        # Rate of spread azimuth (direction fire spreads TO)
        raz_deg = (wind_dir + 180.0) % 360.0
        raz_rad = np.radians(raz_deg)

        # Fill time dimension (constant weather)
        for t in range(n_timesteps):
            ros_arr[t] = ros_corrected
            bros_arr[t] = bros_base
            fros_arr[t] = fros_base
            raz_arr[t] = raz_rad
    
    # Log ROS statistics (from first timestep)
    ros_t0 = ros_arr[0]
    valid_ros = ros_t0[ros_t0 > 0]
    if len(valid_ros) > 0:
        logger.info(
            f"ROS statistics: min={np.min(valid_ros):.2f}, max={np.max(valid_ros):.2f}, "
            f"mean={np.mean(valid_ros):.2f} m/min"
        )
        logger.info(f"Cells with ROS > 0: {len(valid_ros)} ({100*len(valid_ros)/ros_t0.size:.1f}%)")
    else:
        logger.warning("No cells have ROS > 0! Check fuel type mapping.")
    
    return FireParameterGrid(
        x_coords=x_coords,
        y_coords=y_coords,
        ros=ros_arr,
        bros=bros_arr,
        fros=fros_arr,
        raz=raz_arr,
    )


# =============================================================================
# Single Fire Simulation
# =============================================================================


def simulate_single_fire(
    ignition: IgnitionPoint,
    param_grid: FireParameterGrid,
    config: IgnacioConfig,
    is_geographic: bool = False,
    center_latitude: float | None = None,
) -> FireResult:
    """
    Simulate a single fire from ignition point.
    
    Parameters
    ----------
    ignition : IgnitionPoint
        Ignition location and metadata.
    param_grid : FireParameterGrid
        Fire behaviour parameter grid.
    config : IgnacioConfig
        Configuration object.
    is_geographic : bool
        If True, coordinates are in degrees (lat/lon).
    center_latitude : float, optional
        Center latitude for geographic conversion.
        
    Returns
    -------
    FireResult
        Simulation results for this fire.
    """
    sim_config = config.simulation
    
    # Check if ignition is within grid bounds
    if not (param_grid.x_min <= ignition.x <= param_grid.x_max and
            param_grid.y_min <= ignition.y <= param_grid.y_max):
        logger.warning(
            f"Ignition ({ignition.x}, {ignition.y}) outside grid bounds, skipping"
        )
        return FireResult(
            ignition=ignition,
            history=FirePerimeterHistory(perimeters=[], times=[]),
        )
    
    # Run spread simulation
    history = simulate_fire_spread(
        param_grid=param_grid,
        x_ignition=ignition.x,
        y_ignition=ignition.y,
        dt=sim_config.dt,
        n_vertices=sim_config.n_vertices,
        initial_radius=sim_config.initial_radius,
        store_every=sim_config.store_every,
        use_markers=sim_config.marker_method.enabled,
        marker_epsilon=sim_config.marker_method.epsilon,
        min_ros=sim_config.min_ros,
        is_geographic=is_geographic,
        center_latitude=center_latitude,
    )
    
    # Compute final metrics
    final_area_ha = 0.0
    final_perimeter_m = 0.0
    
    if history.perimeters:
        x, y = history.get_final_perimeter()
        if len(x) > 2:
            if is_geographic:
                # Convert from degrees to meters for area/perimeter calculation
                if center_latitude is None:
                    center_latitude = np.mean(y)
                lat_rad = np.radians(center_latitude)
                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
                
                # Convert coordinates to local meters
                x_m = (x - np.mean(x)) * meters_per_deg_lon
                y_m = (y - np.mean(y)) * meters_per_deg_lat
                
                # Area via shoelace formula (in m²)
                area_m2 = 0.5 * np.abs(np.sum(x_m * np.roll(y_m, -1) - np.roll(x_m, -1) * y_m))
                final_area_ha = area_m2 / 10000.0
                
                # Perimeter length (in m)
                dx = np.diff(np.append(x_m, x_m[0]))
                dy = np.diff(np.append(y_m, y_m[0]))
                final_perimeter_m = float(np.sum(np.hypot(dx, dy)))
            else:
                # Projected coordinates - direct calculation
                # Area via shoelace formula
                area_m2 = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
                final_area_ha = area_m2 / 10000.0
                
                # Perimeter length
                dx = np.diff(np.append(x, x[0]))
                dy = np.diff(np.append(y, y[0]))
                final_perimeter_m = float(np.sum(np.hypot(dx, dy)))
    
    spread_duration = history.times[-1] if history.times else 0.0
    
    return FireResult(
        ignition=ignition,
        history=history,
        final_area_ha=final_area_ha,
        final_perimeter_m=final_perimeter_m,
        dur_min=spread_duration,
    )


# =============================================================================
# Main Simulation Entry Point
# =============================================================================


def run_simulation(config: IgnacioConfig) -> SimulationResults:
    """
    Run complete fire growth simulation.
    
    This is the main entry point for running simulations. It orchestrates:
    1. Terrain processing
    2. Weather processing
    3. Ignition generation
    4. Parameter grid building
    5. Fire spread simulation for each ignition
    6. Output generation
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
        
    Returns
    -------
    SimulationResults
        Complete simulation results.
    """
    # Set up random number generator
    rng = np.random.default_rng(config.project.random_seed)
    
    # Create output directory
    output_dir = Path(config.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process terrain
    logger.info("=" * 60)
    logger.info("Step 1: Processing terrain")
    logger.info("=" * 60)
    terrain = build_terrain_grids(config)
    
    # Step 2: Process weather
    logger.info("=" * 60)
    logger.info("Step 2: Processing fire weather")
    logger.info("=" * 60)
    
    # Load hourly data first (before process_fire_weather converts to daily)
    hourly_data = None
    if config.simulation.time_varying_weather:
        try:
            hourly_data = load_weather_data(config)
            if len(hourly_data) > 0 and "HOUR" in hourly_data.columns:
                logger.info(f"Loaded {len(hourly_data)} hourly records for time-varying simulation")
            else:
                logger.info("Hourly data not suitable for time-varying simulation, using static weather")
                hourly_data = None
        except Exception as e:
            logger.warning(f"Could not load hourly data: {e}, using static weather")
            hourly_data = None
    
    weather = process_fire_weather(config, rng)
    
    if config.output.save_weather_summary:
        weather_path = output_dir / "fire_weather_list.csv"
        save_fire_weather_list(weather, weather_path)
    
    # Step 3: Generate ignitions
    logger.info("=" * 60)
    logger.info("Step 3: Generating ignitions")
    logger.info("=" * 60)
    fuel_raster = read_raster_int(config.fuel.path)
    
    # Get terrain CRS to ensure ignitions are in the same coordinate system
    terrain_crs = terrain.crs
    if terrain_crs is not None:
        terrain_crs = str(terrain_crs)
    ignitions = generate_ignitions(config, fuel_raster, rng, terrain_crs=terrain_crs)
    
    if config.output.save_ignition_summary:
        ignition_path = output_dir / "ignitions.shp"
        save_ignitions(ignitions, ignition_path)
    
    # Step 4: Build parameter grid
    logger.info("=" * 60)
    logger.info("Step 4: Building parameter grid")
    logger.info("=" * 60)
    param_grid = build_parameter_grid(
        config, 
        terrain, 
        weather,
        hourly_data=hourly_data,
    )
    
    # Step 5: Run simulations
    logger.info("=" * 60)
    logger.info("Step 5: Running fire simulations")
    logger.info("=" * 60)
    
    results = SimulationResults(
        config=config,
        terrain=terrain,
        weather=weather,
        ignitions=ignitions,
    )
    
    n_ignitions = ignitions.n_points
    logger.info(f"Simulating {n_ignitions} fire(s)")
    logger.debug(
        f"Grid bounds: X=[{param_grid.x_min:.6f}, {param_grid.x_max:.6f}], "
        f"Y=[{param_grid.y_min:.6f}, {param_grid.y_max:.6f}]"
    )
    
    # Detect if terrain is in geographic CRS
    is_geographic = False
    center_latitude = None
    if terrain.crs is not None:
        try:
            import pyproj
            crs = pyproj.CRS.from_user_input(terrain.crs)
            is_geographic = crs.is_geographic
        except Exception:
            # Heuristic: if x bounds are in typical longitude range
            if -180 <= param_grid.x_min <= 180 and -180 <= param_grid.x_max <= 180:
                is_geographic = True
    
    if is_geographic:
        center_latitude = (param_grid.y_min + param_grid.y_max) / 2.0
        logger.info(f"Geographic CRS detected, center latitude: {center_latitude:.4f}°")
    
    # Get fuel grid for diagnostics (use the already-resampled param_grid data)
    # Re-load and resample fuel for diagnostics
    fuel_raster = read_raster_int(config.fuel.path)
    fuel_grid = fuel_raster.data
    if fuel_grid.shape != terrain.shape:
        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.warp import reproject
            
            fuel_resampled = np.zeros(terrain.shape, dtype=fuel_grid.dtype)
            reproject(
                source=fuel_grid,
                destination=fuel_resampled,
                src_transform=fuel_raster.transform,
                src_crs=fuel_raster.crs,
                dst_transform=terrain.dem.transform,
                dst_crs=terrain.dem.crs,
                resampling=Resampling.nearest,
            )
            fuel_grid = fuel_resampled
        except Exception:
            from scipy.ndimage import zoom
            zoom_factors = (terrain.shape[0] / fuel_grid.shape[0], terrain.shape[1] / fuel_grid.shape[1])
            fuel_grid = zoom(fuel_grid, zoom_factors, order=0)
    
    for i, ignition in enumerate(ignitions.points):
        logger.info(f"Fire {i+1}/{n_ignitions}: ({ignition.x:.6f}, {ignition.y:.6f})")
        
        # Diagnostic: what's the fuel and ROS at ignition location?
        try:
            ros_arr, _, _, _ = param_grid.sample_at(0, np.array([ignition.x]), np.array([ignition.y]))
            ros_at_ign = ros_arr[0]
            
            # Get grid indices for fuel lookup
            x_frac = (ignition.x - param_grid.x_min) / (param_grid.x_max - param_grid.x_min)
            x_idx = int(x_frac * (len(param_grid.x_coords) - 1))
            x_idx = max(0, min(x_idx, fuel_grid.shape[1] - 1))
            
            if param_grid.y_flipped:
                y_frac = (param_grid.y_max - ignition.y) / (param_grid.y_max - param_grid.y_min)
            else:
                y_frac = (ignition.y - param_grid.y_min) / (param_grid.y_max - param_grid.y_min)
            
            y_idx = int(y_frac * (len(param_grid.y_coords) - 1))
            y_idx = max(0, min(y_idx, fuel_grid.shape[0] - 1))
            
            fuel_at_ign = fuel_grid[y_idx, x_idx]
            fuel_name = config.fuel.fuel_lookup.get(int(fuel_at_ign), f"Unknown ({fuel_at_ign})")
            logger.info(f"  Fuel: {fuel_name}, ROS: {ros_at_ign:.2f} m/min")
            
            if ros_at_ign < 0.01:
                logger.warning(f"  ROS at ignition is very low! Fire may not spread.")
        except Exception as e:
            logger.warning(f"  Could not get ignition diagnostics: {e}")
        
        fire_result = simulate_single_fire(
            ignition, param_grid, config,
            is_geographic=is_geographic,
            center_latitude=center_latitude,
        )
        results.fires.append(fire_result)
        
        if fire_result.final_area_ha > 0:
            logger.info(
                f"  - Area: {fire_result.final_area_ha:.2f} ha, "
                f"Perimeter: {fire_result.final_perimeter_m:.0f} m"
            )
    
    # Step 6: Save outputs
    logger.info("=" * 60)
    logger.info("Step 6: Saving outputs")
    logger.info("=" * 60)
    
    if config.output.save_perimeters:
        _save_results(results, output_dir, config)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Simulation Complete")
    logger.info("=" * 60)
    logger.info(f"Total fires simulated: {results.n_fires}")
    logger.info(f"Total area burned: {results.total_area_ha:.2f} ha")
    logger.info(f"Output directory: {output_dir}")
    
    return results


def _save_results(
    results: SimulationResults,
    output_dir: Path,
    config: IgnacioConfig,
) -> None:
    """Save simulation results to files."""
    # Get source CRS from terrain
    source_crs = None
    if results.terrain and results.terrain.crs:
        source_crs = str(results.terrain.crs)
    
    # Save individual perimeters
    perimeters_dir = output_dir / "perimeters"
    perimeters_dir.mkdir(exist_ok=True)
    
    for i, fire in enumerate(results.fires):
        if fire.final_area_ha > 0:
            gdf = fire.get_final_polygon(config.crs.output_crs, source_crs=source_crs)
            
            # Determine format
            fmt = config.output.perimeter_format
            if fmt == "shapefile":
                path = perimeters_dir / f"fire_{i:04d}.shp"
                driver = "ESRI Shapefile"
            elif fmt == "geojson":
                path = perimeters_dir / f"fire_{i:04d}.geojson"
                driver = "GeoJSON"
            else:
                path = perimeters_dir / f"fire_{i:04d}.gpkg"
                driver = "GPKG"
            
            write_vector(gdf, path, driver=driver)
    
    # Save combined perimeters
    if config.output.combine_perimeters:
        all_perimeters = results.get_all_perimeters(source_crs=source_crs)
        if len(all_perimeters) > 0:
            fmt = config.output.perimeter_format
            if fmt == "shapefile":
                combined_path = output_dir / "all_perimeters.shp"
            elif fmt == "geojson":
                combined_path = output_dir / "all_perimeters.geojson"
            else:
                combined_path = output_dir / "all_perimeters.gpkg"
            
            write_vector(all_perimeters, combined_path)
            logger.info(f"Saved combined perimeters to {combined_path}")
    
    # Save summary CSV
    summary = results.get_summary()
    summary_path = output_dir / "simulation_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved simulation summary to {summary_path}")
