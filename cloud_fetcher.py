import os
import json
import shutil
import re
import zipfile
import warnings
import datetime
from pathlib import Path
import requests

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rioxarray.merge import merge_arrays
from shapely.geometry import box, shape
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BBOX = [-116.55, 50.95, -115.5, 51.76]  # [West, South, East, North]
TIME_START = "2020-05-01"
TIME_END = "2020-10-01"
OUTPUT_DIR = Path("data/domain_Bow_at_Banff_inputs")

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_REQUEST_PAYER"] = "requester"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


# ==============================================================================
# 1. COPERNICUS DEM (AWS S3 via STAC)
# ==============================================================================
def fetch_dem(bbox):
    print("\n1. Fetching Copernicus DEM (30m)...")
    out_path = OUTPUT_DIR / "dem_30m.tif"

    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return

    try:
        from pystac_client import Client
        catalog = Client.open("https://earth-search.aws.element84.com/v1")
        roi = box(*bbox)

        search = catalog.search(collections=["cop-dem-glo-30"], intersects=roi)
        items = search.item_collection()

        print(f"   Found {len(items)} tiles via STAC.")

        dem_datasets = []
        for item in items:
            url = item.assets["data"].href
            ds = rioxarray.open_rasterio(url)
            dem_datasets.append(ds)

        if dem_datasets:
            print("   Merging and clipping...")
            dem_merged = merge_arrays(dem_datasets)
            dem_merged.rio.write_crs("EPSG:4326", inplace=True)
            dem_clip = dem_merged.rio.clip_box(*bbox)

            dem_clip.rio.to_raster(out_path, compress='LZW')
            print(f"   [SUCCESS] Saved: {out_path}")
            dem_clip.close()
        else:
            print("   [ERROR] No DEM tiles found.")

    except Exception as e:
        print(f"   [ERROR] Failed to fetch DEM: {e}")


# ==============================================================================
# 2. ESA WORLDCOVER (Direct S3 - FIXED: Multi-tile support)
# ==============================================================================
def fetch_landcover(bbox):
    print("\n2. Fetching ESA WorldCover 10m (v100)...")
    final_path = OUTPUT_DIR / "landcover_10m.tif"

    if final_path.exists():
        print(f"   [SKIP] File exists: {final_path}")
        return

    # FIX: Calculate ALL tiles that intersect the bbox (3-degree tiles)
    west, south, east, north = bbox
    
    # Get range of tile origins needed
    lat_min = int(np.floor(south / 3) * 3)
    lat_max = int(np.floor(north / 3) * 3)
    lon_min = int(np.floor(west / 3) * 3)
    lon_max = int(np.floor(east / 3) * 3)
    
    tiles_to_fetch = []
    for lat in range(lat_min, lat_max + 1, 3):
        for lon in range(lon_min, lon_max + 1, 3):
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            filename = f"ESA_WorldCover_10m_2020_v100_{lat_str}{lon_str}_Map.tif"
            tiles_to_fetch.append(filename)
    
    print(f"   Tiles needed: {tiles_to_fetch}")
    
    bucket = "esa-worldcover"
    downloaded_arrays = []
    temp_files = []
    
    for filename in tiles_to_fetch:
        key = f"v100/2020/map/{filename}"
        local_path = OUTPUT_DIR / f"temp_{filename}"
        
        try:
            print(f"   Downloading {filename}...")
            s3.download_file(bucket, key, str(local_path))
            temp_files.append(local_path)
            
            ds = rioxarray.open_rasterio(local_path)
            ds.rio.write_crs("EPSG:4326", inplace=True)
            downloaded_arrays.append(ds)
            
        except Exception as e:
            print(f"   [WARN] Could not download {filename}: {e}")
    
    if not downloaded_arrays:
        print("   [ERROR] No tiles downloaded.")
        return
    
    try:
        # Merge if multiple tiles, otherwise use single
        if len(downloaded_arrays) > 1:
            print("   Merging tiles...")
            merged = merge_arrays(downloaded_arrays)
        else:
            merged = downloaded_arrays[0]
        
        merged.rio.write_crs("EPSG:4326", inplace=True)
        clipped = merged.rio.clip_box(*bbox)
        clipped.rio.to_raster(final_path, compress='LZW')
        print(f"   [SUCCESS] Saved: {final_path}")
        
        # Cleanup
        for ds in downloaded_arrays:
            ds.close()
        for f in temp_files:
            if f.exists():
                os.remove(f)
                
    except Exception as e:
        print(f"   [ERROR] Merge/clip failed: {e}")


# ==============================================================================
# 3. META CANOPY HEIGHT (FIXED: Proper array handling)
# ==============================================================================
def fetch_canopy_height(bbox):
    print("\n3. Fetching Meta Canopy Height...")
    final_path = OUTPUT_DIR / "canopy_height_10m.tif"
    if final_path.exists() and final_path.stat().st_size > 1000:
        print(f"   [SKIP] File exists.")
        return

    bucket = "dataforgood-fb-data"
    prefix = "forests/v1/alsgedi_global_v6_float/"
    index_key = prefix + "tiles.geojson"
    local_index = OUTPUT_DIR / "chm_index.geojson"

    # 1. Download Index
    if not local_index.exists():
        try:
            s3.download_file(bucket, index_key, str(local_index))
        except Exception as e:
            print(f"   [ERROR] Could not download CHM index: {e}")
            return

    # 2. Identify Tiles
    target_files = []
    roi = box(*bbox)
    with open(local_index) as f:
        data = json.load(f)
        for feature in data['features']:
            if shape(feature['geometry']).intersects(roi):
                props = feature['properties']
                fname = props.get('filename') or props.get('tile') or str(props.get('quadkey'))
                if not fname.endswith('.tif'):
                    fname += ".tif"
                target_files.append(fname)

    if not target_files:
        print("   [WARNING] No tiles found in index for bbox.")
        return

    print(f"   Found {len(target_files)} tiles to process...")

    # 3. Download all tiles first, then process
    temp_files = []
    for fname in target_files:
        key = prefix + "chm/" + fname
        local_tif = OUTPUT_DIR / f"temp_chm_{fname}"

        try:
            if not local_tif.exists():
                print(f"   Downloading {fname}...")
                s3.download_file(bucket, key, str(local_tif))
            temp_files.append(local_tif)
        except Exception as e:
            print(f"   [WARN] Could not download {fname}: {e}")

    if not temp_files:
        print("   [ERROR] No tiles downloaded.")
        return

    # 4. FIX: Load, clip, and collect arrays properly
    clipped_arrays = []
    for local_tif in temp_files:
        try:
            # Load data into memory before closing file handle
            ds = rioxarray.open_rasterio(local_tif)
            ds.rio.write_crs("EPSG:4326", inplace=True)

            # Check if tile actually overlaps bbox in raster space
            tile_bounds = ds.rio.bounds()
            if (tile_bounds[2] < bbox[0] or tile_bounds[0] > bbox[2] or
                tile_bounds[3] < bbox[1] or tile_bounds[1] > bbox[3]):
                ds.close()
                continue

            # Clip and load into memory
            ds_clipped = ds.rio.clip_box(*bbox)
            ds_loaded = ds_clipped.load()  # FIX: Load data into memory
            clipped_arrays.append(ds_loaded)
            ds.close()

        except Exception as e:
            print(f"   [WARN] Could not process {local_tif.name}: {e}")

    # 5. Merge
    if clipped_arrays:
        try:
            print(f"   Merging {len(clipped_arrays)} clipped tiles...")
            if len(clipped_arrays) > 1:
                chm_merged = merge_arrays(clipped_arrays)
            else:
                chm_merged = clipped_arrays[0]

            chm_merged.rio.to_raster(final_path, compress='LZW')
            print(f"   [SUCCESS] Saved: {final_path}")

        except Exception as e:
            print(f"   [ERROR] Merge failed: {e}")
    else:
        print("   [ERROR] No valid data extracted from tiles.")

    # Cleanup temp files
    for f in temp_files:
        if f.exists():
            os.remove(f)


# ==============================================================================
# 4. WEATHER (ERA5-Land - FIXED: Proper lazy loading)
# ==============================================================================
def fetch_weather(bbox):
    print("\n4. Fetching Weather (ERA5-Land)...")
    out_path = OUTPUT_DIR / "weather_era5.nc"
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return

    try:
        import gcsfs

        # FIX: Use the standard ERA5-Land hourly dataset instead
        # The v3 store has different structure; use Google's public ERA5
        zarr_store = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        print(f"   Accessing Zarr store...")

        fs = gcsfs.GCSFileSystem(token='anon')
        mapper = fs.get_mapper(zarr_store)

        # FIX: Keep chunks for lazy loading - don't use chunks=None!
        ds = xr.open_zarr(mapper)

        print(f"   Available variables: {list(ds.data_vars)[:10]}...")
        print(f"   Dimensions: {dict(ds.dims)}")

        var_map = {
            '2m_temperature': 't2m',
            'total_precipitation': 'tp',
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10',
        }

        available = set(ds.data_vars)
        to_keep = [k for k in var_map.keys() if k in available]

        if not to_keep:
            print(f"   [WARN] Expected variables not found. Available: {list(available)[:20]}")
            # Try alternative names
            alt_map = {'t2m': 't2m', 'tp': 'tp', 'u10': 'u10', 'v10': 'v10'}
            to_keep = [k for k in alt_map.keys() if k in available]
            var_map = {k: k for k in to_keep}

        print(f"   Using variables: {to_keep}")

        # FIX: Slice BEFORE selecting variables to minimize data access
        # Handle longitude: ARCO ERA5 uses 0-360
        west = (bbox[0] + 360) % 360
        east = (bbox[2] + 360) % 360
        north, south = bbox[3], bbox[1]

        print(f"   Slicing: time={TIME_START} to {TIME_END}, lat={south}-{north}, lon={west}-{east}")

        # Check coordinate names
        lat_coord = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_coord = 'longitude' if 'longitude' in ds.coords else 'lon'
        
        # FIX: Slice spatially first with isel to get indices, then select
        # This is more efficient for zarr
        lat_vals = ds[lat_coord].values
        lon_vals = ds[lon_coord].values
        
        # Find indices (ERA5 lat is often descending)
        if lat_vals[0] > lat_vals[-1]:  # Descending
            lat_idx = np.where((lat_vals >= south) & (lat_vals <= north))[0]
        else:  # Ascending
            lat_idx = np.where((lat_vals >= south) & (lat_vals <= north))[0]
            
        lon_idx = np.where((lon_vals >= west) & (lon_vals <= east))[0]
        
        print(f"   Lat indices: {len(lat_idx)}, Lon indices: {len(lon_idx)}")

        # Select subset
        ds_spatial = ds[to_keep].isel({lat_coord: lat_idx, lon_coord: lon_idx})
        ds_slice = ds_spatial.sel(time=slice(TIME_START, TIME_END))

        # Rename variables
        rename_dict = {k: var_map[k] for k in to_keep if k in var_map and k != var_map[k]}
        if rename_dict:
            ds_slice = ds_slice.rename(rename_dict)

        # FIX: Compute/download the subset
        print(f"   Downloading subset (this may take a few minutes)...")
        ds_loaded = ds_slice.compute()

        ds_loaded.to_netcdf(out_path)
        print(f"   [SUCCESS] Saved: {out_path}")

    except Exception as e:
        print(f"   [ERROR] ERA5 download failed: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# 4b. WEATHER ALTERNATIVE (Open-Meteo API - faster fallback)
# ==============================================================================
def fetch_weather_openmeteo(bbox):
    """Alternative weather fetch using Open-Meteo API - much faster for small domains"""
    print("\n4b. Fetching Weather (Open-Meteo API fallback)...")
    out_path = OUTPUT_DIR / "weather_openmeteo.nc"
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return

    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        # Setup cached session
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=retry_session)

        # Center point of bbox
        lat = (bbox[1] + bbox[3]) / 2
        lon = (bbox[0] + bbox[2]) / 2

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": TIME_START,
            "end_date": TIME_END,
            "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"]
        }

        response = om.weather_api(url, params=params)[0]
        hourly = response.Hourly()

        time_range = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        ds = xr.Dataset({
            't2m': (['time'], hourly.Variables(0).ValuesAsNumpy()),
            'tp': (['time'], hourly.Variables(1).ValuesAsNumpy()),
            'wind': (['time'], hourly.Variables(2).ValuesAsNumpy()),
        }, coords={'time': time_range})

        ds.to_netcdf(out_path)
        print(f"   [SUCCESS] Saved: {out_path}")

    except ImportError:
        print("   [INFO] openmeteo_requests not installed. Skipping fallback.")
    except Exception as e:
        print(f"   [ERROR] Open-Meteo failed: {e}")


# ==============================================================================
# 5. MODIS LANDCOVER (FIXED: Better error handling + progress)
# ==============================================================================
def fetch_modis_landcover(bbox):
    print("\n5. Fetching MODIS Land Cover (MCD12Q1 v061)...")
    out_path = OUTPUT_DIR / "modis_landclass_mode.tif"
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return

    # FIX: Use fewer years and add timeout/progress
    years = range(2015, 2020)  # Reduced from 2001-2020 to speed up
    base_url = "https://zenodo.org/records/8367523/files"

    arrays = []
    out_meta = None

    print(f"   Processing {len(list(years))} years (2015-2019)...")

    for year in years:
        fname = f"lc_mcd12q1v061.t1_c_500m_s_{year}0101_{year}1231_go_epsg.4326_v20230818.tif"
        url = f"{base_url}/{fname}"
        local_tmp = OUTPUT_DIR / f"temp_modis_{year}.tif"

        try:
            print(f"   [{year}] ", end="", flush=True)

            # Download with timeout
            if not local_tmp.exists():
                with requests.get(url, stream=True, timeout=60) as r:
                    if r.status_code != 200:
                        print(f"HTTP {r.status_code} - skipped")
                        continue
                    
                    # FIX: Show download progress
                    total = int(r.headers.get('content-length', 0))
                    with open(local_tmp, 'wb') as f:
                        downloaded = 0
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                    print(f"downloaded ({downloaded/1e6:.1f}MB), ", end="")

            # Open & Crop
            with rasterio.open(local_tmp) as src:
                window = from_bounds(*bbox, src.transform)
                data = src.read(1, window=window)

                if out_meta is None:
                    out_transform = src.window_transform(window)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": data.shape[0],
                        "width": data.shape[1],
                        "transform": out_transform,
                        "compress": "lzw"
                    })

                if data.shape == (out_meta['height'], out_meta['width']):
                    arrays.append(data)
                    print("cropped ✓")
                else:
                    print(f"shape mismatch ({data.shape})")

            # Cleanup immediately to save space
            if local_tmp.exists():
                os.remove(local_tmp)

        except requests.Timeout:
            print("timeout - skipped")
        except Exception as e:
            print(f"error: {e}")
            if local_tmp.exists():
                os.remove(local_tmp)

    if not arrays:
        print("   [ERROR] No MODIS data processed.")
        return

    print(f"   Computing mode from {len(arrays)} years...")
    stack = np.stack(arrays, axis=0)

    def calc_mode(arr):
        valid = arr[arr != 255]
        if valid.size == 0:
            return 255
        vals, counts = np.unique(valid, return_counts=True)
        return vals[np.argmax(counts)]

    mode_data = np.apply_along_axis(calc_mode, 0, stack).astype('uint8')

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mode_data, 1)

    print(f"   [SUCCESS] Saved: {out_path}")


# ==============================================================================
# 6. ECOREGIONS (RESOLVE 2017)
# ==============================================================================
def fetch_ecoregions(bbox):
    print("\n6. Fetching Ecoregions (RESOLVE 2017)...")
    out_path = OUTPUT_DIR / "ecoregions.geojson"
    if out_path.exists():
        print(f"   [SKIP] File exists.")
        return

    url = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
    zip_path = OUTPUT_DIR / "Ecoregions2017.zip"
    extract_dir = OUTPUT_DIR / "Ecoregions2017_tmp"

    try:
        if not zip_path.exists():
            print(f"   Downloading (~400MB)...")
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

        shp_files = list(extract_dir.glob("**/*.shp"))
        if not shp_files:
            raise FileNotFoundError("No SHP found in zip")

        gdf = gpd.read_file(shp_files[0])
        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")

        if gdf.crs != bbox_gdf.crs:
            gdf = gdf.to_crs(bbox_gdf.crs)

        gdf_clipped = gpd.clip(gdf, bbox_gdf)
        gdf_clipped.to_file(out_path, driver="GeoJSON")
        print(f"   [SUCCESS] Saved: {out_path}")

    except Exception as e:
        print(f"   [ERROR] {e}")
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)


# ==============================================================================
# 7. BURNED AREA (Canada NBAC - Dynamic Finder)
# ==============================================================================
def fetch_burned_area(bbox):
    print("\n7. Fetching Burned Area (Canada NBAC 2020)...")
    out_path = OUTPUT_DIR / "burned_area_2020.geojson"
    if out_path.exists():
        print(f"   [SKIP] File exists.")
        return

    base_url = "https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/"
    print(f"   Checking {base_url} for 2020 file...")

    try:
        resp = requests.get(base_url, timeout=30)
        resp.raise_for_status()

        match = re.search(r'href="(nbac_2020_[^"]+\.zip)"', resp.text)

        if not match:
            print("   [ERROR] Could not find 2020 NBAC zip file in listing.")
            return

        filename = match.group(1)
        file_url = base_url + filename
        print(f"   Found: {filename}")

        zip_path = OUTPUT_DIR / filename
        extract_dir = OUTPUT_DIR / "nbac_tmp"

        if not zip_path.exists():
            with requests.get(file_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

        shp_files = list(extract_dir.glob("*.shp"))
        if not shp_files:
            raise Exception("No shapefile in zip")

        gdf = gpd.read_file(shp_files[0])
        if gdf.crs:
            gdf = gdf.to_crs("EPSG:4326")
        else:
            gdf.set_crs(epsg=4326, inplace=True)

        bbox_geom = box(*bbox)
        gdf_clipped = gdf[gdf.intersects(bbox_geom)]

        if gdf_clipped.empty:
            print("   [INFO] No fires in bbox.")
            gpd.GeoDataFrame(columns=['geometry'], geometry=[], crs="EPSG:4326").to_file(out_path, driver="GeoJSON")
        else:
            gdf_clipped.to_file(out_path, driver="GeoJSON")
            print(f"   [SUCCESS] Found {len(gdf_clipped)} fires. Saved.")

    except Exception as e:
        print(f"   [ERROR] NBAC fetch failed: {e}")
    finally:
        if 'extract_dir' in locals() and extract_dir.exists():
            shutil.rmtree(extract_dir)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"--- Cloud Data Fetcher (Fixed) ---")
    print(f"Target BBOX: {BBOX}")
    print(f"Time Range: {TIME_START} to {TIME_END}")

    fetch_dem(BBOX)
    fetch_landcover(BBOX)
    fetch_canopy_height(BBOX)
    fetch_weather(BBOX)
    fetch_modis_landcover(BBOX)
    fetch_ecoregions(BBOX)
    fetch_burned_area(BBOX)

    print("\n" + "="*60)
    print("Ingestion Complete.")
    print(f"Output directory: {OUTPUT_DIR}")