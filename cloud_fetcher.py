import os
import json
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rioxarray.merge import merge_arrays
from shapely.geometry import box, shape
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import warnings

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BBOX = [-116.55, 50.95, -115.5, 51.76] 
TIME_START = "2020-05-01"
TIME_END = "2020-10-01"
OUTPUT_DIR = "data/domain_Bow_at_Banff_inputs"

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_REQUEST_PAYER"] = "requester" 
os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# ==============================================================================
# 1. COPERNICUS DEM (Skipping implementation as it works)
# ==============================================================================
def fetch_dem(bbox):
    out_path = f"{OUTPUT_DIR}/dem_30m.tif"
    if os.path.exists(out_path):
        print(f"\n1. Copernicus DEM: [SKIP] Exists.")
    else:
        # Re-paste previous working code if needed, but for brevity assuming done
        print(f"\n1. Copernicus DEM: Please run previous version to fetch if missing.")

# ==============================================================================
# 2. ESA WORLDCOVER (Skipping implementation as it works)
# ==============================================================================
def fetch_landcover(bbox):
    out_path = f"{OUTPUT_DIR}/landcover_10m.tif"
    if os.path.exists(out_path):
        print(f"\n2. ESA WorldCover: [SKIP] Exists.")
    else:
        print(f"\n2. ESA WorldCover: Please run previous version to fetch if missing.")

# ==============================================================================
# 3. META CANOPY HEIGHT (Robust Merge & Clip)
# ==============================================================================
def fetch_canopy_height(bbox):
    print("\n3. Fetching Meta Canopy Height...")
    final_path = f"{OUTPUT_DIR}/canopy_height_10m.tif"
    if os.path.exists(final_path):
        print(f"   [SKIP] File exists: {final_path}")
        return

    bucket = "dataforgood-fb-data"
    prefix = "forests/v1/alsgedi_global_v6_float/"
    index_key = prefix + "tiles.geojson"
    local_index = f"{OUTPUT_DIR}/chm_index.geojson"
    
    # 1. Index
    if not os.path.exists(local_index):
        try:
            s3.download_file(bucket, index_key, local_index)
        except Exception:
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
                if not fname.endswith('.tif'): fname += ".tif"
                target_files.append(fname)

    if not target_files:
        print("   [WARNING] No tiles found.")
        return

    # 3. Download & Load
    chm_datasets = []
    temp_files = []
    
    print(f"   Processing {len(target_files)} tiles...")
    
    for fname in target_files:
        key = prefix + "chm/" + fname
        local_tif = f"{OUTPUT_DIR}/{fname}"
        
        try:
            if not os.path.exists(local_tif):
                s3.download_file(bucket, key, local_tif)
                temp_files.append(local_tif) # Mark for potential deletion
            
            # Open
            ds = rioxarray.open_rasterio(local_tif)
            # CRITICAL: Meta CHM is EPSG:4326 but sometimes missing headers
            if ds.rio.crs is None:
                ds.rio.write_crs("EPSG:4326", inplace=True)
            
            chm_datasets.append(ds)
            
        except Exception as e:
            print(f"   [ERROR] Failed to load {fname}: {e}")

    # 4. Merge & Clip (Robust)
    if chm_datasets:
        try:
            print("   Merging...")
            chm_merged = merge_arrays(chm_datasets)
            
            print("   Clipping...")
            # Use from_disk=True to save memory if needed, but here we do in-memory
            # Pad the bbox slightly to avoid edge errors
            pad = 0.001
            padded_bbox = [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad]
            
            try:
                chm_clip = chm_merged.rio.clip_box(*bbox)
            except Exception:
                print("   [RETRY] Standard clip failed, trying padded clip...")
                chm_clip = chm_merged.rio.clip_box(*padded_bbox)
                # Then crop to exact
                chm_clip = chm_clip.rio.clip_box(*bbox)

            chm_clip.rio.to_raster(final_path, compress='LZW')
            print(f"   [SUCCESS] Saved: {final_path}")
            
            # Only delete temps on success
            for f in temp_files: os.remove(f)
            
        except Exception as e:
            print(f"   [ERROR] Merge/Clip failed: {e}")
            print(f"   [INFO] Kept raw tiles in {OUTPUT_DIR} for inspection.")
        finally:
            for ds in chm_datasets: ds.close()

# ==============================================================================
# 4. WEATHER (ERA5-Land: Long Variable Names)
# ==============================================================================
def fetch_weather(bbox):
    print("\n4. Fetching Weather (ERA5-Land)...")
    out_path = f"{OUTPUT_DIR}/weather_era5.nc"
    if os.path.exists(out_path):
        print(f"   [SKIP] File exists: {out_path}")
        return

    try:
        import gcsfs
        # ARCO Era5 (Single Level)
        zarr_store = "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2"
        print(f"   Accessing Zarr store: {zarr_store}")
        
        fs = gcsfs.GCSFileSystem(token='anon')
        mapper = fs.get_mapper(zarr_store)
        ds = xr.open_zarr(mapper, chunks=None, decode_coords=False)
        
        # --- VARIABLE MAPPING (Long -> Short) ---
        # Based on cloud_downloader.py and ARCO documentation
        # 2m_temperature -> t2m
        # total_precipitation -> tp
        # 10m_u_component_of_wind -> u10
        # 10m_v_component_of_wind -> v10
        # volumetric_soil_water_layer_1 -> swvl1
        
        long_names = [
            '2m_temperature', 
            'total_precipitation', 
            '10m_u_component_of_wind', 
            '10m_v_component_of_wind', 
            'volumetric_soil_water_layer_1'
        ]
        
        # Verify existence
        available = list(ds.data_vars)
        to_load = [v for v in long_names if v in available]
        
        if not to_load:
            print(f"   [ERROR] None of the expected variables found. Available: {available[:5]}...")
            return

        ds_mini = ds[to_load]
        
        # Rename immediately
        rename_map = {
            '2m_temperature': 't2m',
            'total_precipitation': 'tp',
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10',
            'volumetric_soil_water_layer_1': 'swvl1'
        }
        # Only rename what we found
        final_map = {k: v for k, v in rename_map.items() if k in to_load}
        ds_mini = ds_mini.rename(final_map)

        # --- SLICING ---
        print("   Slicing Coordinates...")
        
        # Time Index
        base_time = pd.Timestamp("1900-01-01")
        t_start = (pd.Timestamp(TIME_START) - base_time).total_seconds() / 3600
        t_end = (pd.Timestamp(TIME_END) - base_time).total_seconds() / 3600
        t_slice = slice(int(t_start), int(t_end) + 24)
        
        # Space Index (0.25 deg)
        north, south = bbox[3], bbox[1]
        west = (bbox[0] + 360) % 360
        east = (bbox[2] + 360) % 360
        
        lat_slice = slice(int((90-north)/0.25), int((90-south)/0.25)+1)
        lon_slice = slice(int(west/0.25), int(east/0.25)+1)
        
        print(f"   Loading subset ({to_load})...")
        ds_final = ds_mini.isel(time=t_slice, latitude=lat_slice, longitude=lon_slice)
        
        # Decode CF (apply scale/offset)
        ds_final = xr.decode_cf(ds_final)
        
        ds_final.to_netcdf(out_path)
        print(f"   [SUCCESS] Downloaded ERA5 data: {out_path}")

    except Exception as e:
        print(f"   [ERROR] Zarr download failed: {e}")

# ==============================================================================
if __name__ == "__main__":
    fetch_dem(BBOX)
    fetch_landcover(BBOX)
    fetch_canopy_height(BBOX)
    fetch_weather(BBOX)
    print("\nData Ingestion Complete.")