"""
Extended Fire Model Data Fetchers - V2 (FIXED)
===============================================
Corrected API endpoints and parameters based on actual service testing.

Datasets:
1. NASA FIRMS Active Fire (uses MODIS as fallback)
2. ESA FireCCI51 Burned Area (GEE only - CDS discontinued)
3. Copernicus GEFF-ERA5 Fire Weather Index (EWDS, no area subset)
4. LANDFIRE FBFM40 + Canopy Fuels (USA only - skips for Canada)
5. Canadian FBP Fuel Types 2024 (WCS 1.0.0 with correct params)
6. MTBS Burn Severity (USA only - skips for Canada)

Author: Fire-Engine-Framework
"""

import os
import json
import time
import shutil
import zipfile
import tempfile
import re
from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO, BytesIO
import requests

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from pyproj import Transformer
import rioxarray
from rioxarray.merge import merge_arrays

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BBOX = [-116.55, 50.95, -115.5, 51.76]  # [West, South, East, North]
TIME_START = "2020-05-01"
TIME_END = "2020-10-01"
OUTPUT_DIR = Path("data/domain_Bow_at_Banff_inputs")

# API Keys
FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_in_usa(bbox):
    """Check if bbox overlaps continental USA, Alaska, or Hawaii."""
    west, south, east, north = bbox
    
    # CONUS bounds
    conus = (-125, 24, -66, 50)
    # Alaska bounds
    alaska = (-180, 51, -129, 72)
    # Hawaii bounds  
    hawaii = (-161, 18, -154, 23)
    
    def overlaps(b1, b2):
        return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])
    
    return overlaps(bbox, conus) or overlaps(bbox, alaska) or overlaps(bbox, hawaii)


def is_in_canada(bbox):
    """Check if bbox overlaps Canada."""
    west, south, east, north = bbox
    # Canada approximate bounds
    canada = (-141, 41.7, -52, 83)
    return not (east < canada[0] or west > canada[2] or north < canada[1] or south > canada[3])


def transform_bbox_to_crs(bbox, from_crs="EPSG:4326", to_crs="EPSG:3978"):
    """Transform bbox from one CRS to another."""
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    
    # Transform all corners
    west, south, east, north = bbox
    corners = [
        (west, south),
        (west, north),
        (east, south),
        (east, north)
    ]
    
    transformed = [transformer.transform(x, y) for x, y in corners]
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    
    return [min(xs), min(ys), max(xs), max(ys)]


# ==============================================================================
# 1. NASA FIRMS ACTIVE FIRE (FIXED - Use NRT with MODIS fallback)
# ==============================================================================
def fetch_firms_active_fire(bbox, start_date=None, end_date=None, map_key=None):
    """
    Fetch NASA FIRMS active fire detections.
    
    NRT data (last 10 days): No auth required
    Archive data: Requires FIRMS MAP_KEY
    """
    print("\n=== Fetching NASA FIRMS Active Fire Data ===")
    out_path_gpkg = OUTPUT_DIR / "active_fires_viirs.gpkg"
    
    if out_path_gpkg.exists():
        print(f"   [SKIP] File exists: {out_path_gpkg}")
        return
    
    map_key = map_key or FIRMS_MAP_KEY
    west, south, east, north = bbox
    
    # For historical data, we need the archive API
    if start_date and end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_ago = (datetime.now() - end_dt).days
        
        if days_ago > 10:
            # Historical archive data needed
            if not map_key:
                print("   [WARN] Archive data requires FIRMS MAP_KEY")
                print("   Get one at: https://firms.modaps.eosdis.nasa.gov/api/area/")
                print("   Trying to download last 10 days NRT instead...")
                days = 10
            else:
                # For archive: try direct CSV download from archive
                print(f"   Downloading archive data for {start_date} to {end_date}...")
                
                # Archive uses different URL structure
                # https://firms.modaps.eosdis.nasa.gov/api/country/csv/MAP_KEY/VIIRS_SNPP_SP/CAN/10/2020-05-01
                # But for area queries, use:
                all_fires = []
                
                # Download in chunks (max 10 days per request for archive)
                current = start_dt
                while current <= end_dt:
                    chunk_end = min(current + timedelta(days=9), end_dt)
                    days_in_chunk = (chunk_end - current).days + 1
                    date_str = current.strftime("%Y-%m-%d")
                    
                    # VIIRS_SNPP_SP = standard processed archive
                    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_SP/{west},{south},{east},{north}/{days_in_chunk}/{date_str}"
                    
                    print(f"   Fetching {date_str} ({days_in_chunk} days)...")
                    try:
                        response = requests.get(url, timeout=120)
                        if response.status_code == 200:
                            text = response.text.strip()
                            if text and not text.startswith("<!"):
                                df = pd.read_csv(StringIO(text))
                                if not df.empty and 'latitude' in df.columns:
                                    all_fires.append(df)
                        else:
                            print(f"   [WARN] HTTP {response.status_code} for {date_str}")
                    except Exception as e:
                        print(f"   [WARN] Failed chunk {date_str}: {e}")
                    
                    current = chunk_end + timedelta(days=1)
                
                if all_fires:
                    df = pd.concat(all_fires, ignore_index=True)
                    gdf = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df.longitude, df.latitude),
                        crs="EPSG:4326"
                    )
                    print(f"   Found {len(gdf)} fire detections")
                    gdf.to_file(out_path_gpkg, driver="GPKG")
                    print(f"   [SUCCESS] Saved: {out_path_gpkg}")
                    return
                else:
                    print("   [INFO] No fires found in archive, trying NRT...")
                    days = 10
        else:
            days = min(10, days_ago + 1)
    else:
        days = 7  # Default
    
    # Try NRT endpoints (VIIRS first, then MODIS)
    sources = ["VIIRS_SNPP_NRT", "MODIS_NRT"]
    
    for source in sources:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/DEMO_KEY/{source}/{west},{south},{east},{north}/{days}"
        print(f"   Trying {source}...")
        print(f"   URL: {url[:80]}...")
        
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                text = response.text.strip()
                
                if not text or text.startswith("<!DOCTYPE") or text.startswith("<"):
                    print(f"   [INFO] No fire data from {source}")
                    continue
                
                df = pd.read_csv(StringIO(text))
                
                if df.empty or 'latitude' not in df.columns:
                    print(f"   [INFO] No fires detected by {source}")
                    continue
                
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs="EPSG:4326"
                )
                print(f"   Found {len(gdf)} fire detections from {source}")
                gdf.to_file(out_path_gpkg, driver="GPKG")
                print(f"   [SUCCESS] Saved: {out_path_gpkg}")
                return
                
            elif response.status_code == 500:
                print(f"   [WARN] Server error for {source}, trying next...")
                continue
            else:
                print(f"   [WARN] HTTP {response.status_code} for {source}")
                continue
                
        except Exception as e:
            print(f"   [ERROR] {source} failed: {e}")
            continue
    
    # If we get here, create empty file
    print("   [INFO] No fire detections available, creating empty file")
    gdf = gpd.GeoDataFrame(columns=['geometry'], geometry=[], crs="EPSG:4326")
    gdf.to_file(out_path_gpkg, driver="GPKG")
    print(f"   [SUCCESS] Created empty file: {out_path_gpkg}")


# ==============================================================================
# 2. ESA FireCCI51 BURNED AREA (GEE only - CDS satellite-fire-burned-area discontinued)
# ==============================================================================
def fetch_firecci51_burned_area(bbox, year=2020):
    """
    Fetch ESA FireCCI51 burned area data.
    
    NOTE: The CDS satellite-fire-burned-area dataset has been retired.
    This function requires Google Earth Engine.
    """
    print(f"\n=== Fetching FireCCI51 Burned Area ({year}) ===")
    out_path = OUTPUT_DIR / f"firecci51_burned_area_{year}.tif"
    
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return
    
    try:
        import ee
        
        # Try to initialize
        try:
            ee.Initialize(opt_url='https://earthengine.googleapis.com')
        except Exception as init_err:
            # Check if already initialized
            try:
                ee.Number(1).getInfo()
            except:
                print(f"   [ERROR] Earth Engine not initialized: {init_err}")
                print("   ")
                print("   To enable Earth Engine:")
                print("   1. pip install earthengine-api")
                print("   2. earthengine authenticate")
                print("   3. Enable Earth Engine API at:")
                print("      https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
                print("   ")
                print("   Alternative: Download from Zenodo manually:")
                print("   https://zenodo.org/records/4288419")
                return
        
        print("   Using Google Earth Engine...")
        west, south, east, north = bbox
        region = ee.Geometry.Rectangle([west, south, east, north])
        
        # FireCCI 5.1 collection
        firecci = ee.ImageCollection('ESA/CCI/FireCCI/5_1') \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .select('BurnDate')
        
        # Get max burn date (most recent burn)
        fire_composite = firecci.max().clip(region)
        
        # Get download URL
        url = fire_composite.getDownloadURL({
            'scale': 250,
            'crs': 'EPSG:4326',
            'region': region,
            'format': 'GEO_TIFF'
        })
        
        print(f"   Downloading from Earth Engine...")
        response = requests.get(url, timeout=300)
        
        if response.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(response.content)
            print(f"   [SUCCESS] Saved: {out_path}")
        else:
            print(f"   [ERROR] Download failed: HTTP {response.status_code}")
            
    except ImportError:
        print("   [ERROR] earthengine-api not installed")
        print("   Run: pip install earthengine-api")
    except Exception as e:
        print(f"   [ERROR] {e}")


# ==============================================================================
# 3. COPERNICUS FIRE WEATHER INDEX (FIXED - no area subset in request)
# ==============================================================================
def fetch_fire_weather_index(bbox, start_date, end_date):
    """
    Fetch Fire Weather Index from Copernicus CEMS EWDS.
    
    IMPORTANT: The EWDS API doesn't support 'area' subsetting for this dataset.
    We download the full global grid and clip locally.
    """
    print("\n=== Fetching Fire Weather Index (GEFF-ERA5) ===")
    out_path = OUTPUT_DIR / "fire_weather_index.nc"
    
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return
    
    try:
        import cdsapi
    except ImportError:
        print("   [ERROR] cdsapi not installed. Run: pip install cdsapi")
        return
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    year = str(start.year)
    
    # Get months - must be sorted strings
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(f"{current.month:02d}")
        current = (current + timedelta(days=32)).replace(day=1)
    months = sorted(list(set(months)))
    
    # Days
    days = [f"{d:02d}" for d in range(1, 32)]
    
    print(f"   Year: {year}")
    print(f"   Months: {months}")
    print(f"   Bbox: {bbox}")
    
    # Check for EWDS credentials
    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        print("   [ERROR] ~/.cdsapirc not found")
        print("   ")
        print("   Create it with:")
        print("   url: https://ewds.climate.copernicus.eu/api")
        print("   key: YOUR_PERSONAL_ACCESS_TOKEN")
        print("   ")
        print("   Get your token from: https://ewds.climate.copernicus.eu/profile")
        return
    
    # Read config to check URL
    with open(cdsapirc) as f:
        config = f.read()
    
    if "ewds.climate.copernicus.eu" not in config:
        print("   [WARN] ~/.cdsapirc doesn't point to EWDS")
        print("   Update URL to: https://ewds.climate.copernicus.eu/api")
    
    try:
        # Create client with EWDS URL explicitly
        c = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api")
        
        # Request parameters - NO area subset (not supported)
        # Download single variable first, then can add more
        request = {
            'product_type': 'reanalysis',
            'variable': 'fire_weather_index',
            'system_version': '4_1',
            'dataset_type': 'consolidated_dataset',
            'year': year,
            'month': months,
            'day': days,
            'data_format': 'netcdf_legacy',
        }
        
        print(f"   Submitting request to EWDS (this may take a while)...")
        
        # Download to temp file first
        temp_path = OUTPUT_DIR / "fwi_global_temp.nc"
        c.retrieve('cems-fire-historical-v1', request, str(temp_path))
        
        print(f"   Downloaded global dataset, now subsetting...")
        
        # Open and subset
        ds = xr.open_dataset(temp_path)
        
        # Get coordinate names
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        
        # Check if longitude is 0-360
        if ds[lon_name].max() > 180:
            # Convert bbox to 0-360
            west = (bbox[0] + 360) % 360
            east = (bbox[2] + 360) % 360
            if west > east:  # Crosses 0
                ds_subset = ds.sel({
                    lat_name: slice(bbox[3], bbox[1]),
                })
                ds_subset = xr.concat([
                    ds_subset.sel({lon_name: slice(west, 360)}),
                    ds_subset.sel({lon_name: slice(0, east)})
                ], dim=lon_name)
            else:
                ds_subset = ds.sel({
                    lat_name: slice(bbox[3], bbox[1]),
                    lon_name: slice(west, east)
                })
        else:
            ds_subset = ds.sel({
                lat_name: slice(bbox[3], bbox[1]),
                lon_name: slice(bbox[0], bbox[2])
            })
        
        ds_subset.to_netcdf(out_path)
        
        # Cleanup
        ds.close()
        os.remove(temp_path)
        
        print(f"   [SUCCESS] Saved: {out_path}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"   [ERROR] FWI download failed: {error_msg[:200]}")
        
        if "400" in error_msg or "Bad Request" in error_msg:
            print("   ")
            print("   Possible causes:")
            print("   1. You haven't accepted the dataset terms yet")
            print("      Visit: https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1")
            print("      Scroll to bottom and accept the license")
            print("   2. Your API key may be expired/invalid")
            print("      Get new key from: https://ewds.climate.copernicus.eu/profile")
            print("   ")
            print("   Trying alternative sources...")
            _fetch_fwi_alternative(bbox, start_date, end_date, out_path)


def _fetch_fwi_alternative(bbox, start_date, end_date, out_path):
    """Try alternative FWI sources when EWDS fails."""
    
    # For Canada, try CWFIS daily grids
    if is_in_canada(bbox):
        print("   Trying CWFIS daily FWI grids (Canada)...")
        _fetch_cwfis_fwi_grids(bbox, start_date, end_date)
    
    # Also try ERA5-Land for raw meteorological variables
    print("   ")
    print("   Alternative: Calculate FWI from ERA5-Land variables")
    print("   The ERA5 weather file (weather_era5.nc) contains:")
    print("   - Temperature, humidity, wind, precipitation")
    print("   Use the 'cffdrs' R package or Python equivalent to compute FWI")
    print("   ")
    print("   pip install cffdrs  # Python port (if available)")
    print("   # Or use R: install.packages('cffdrs')")


def _fetch_cwfis_fwi_grids(bbox, start_date, end_date):
    """
    Download CWFIS daily FWI grids from the datamart.
    These are ASCII grids with FWI values for Canada.
    """
    out_dir = OUTPUT_DIR / "cwfis_fwi"
    out_dir.mkdir(exist_ok=True)
    
    # CWFIS provides daily grids
    base_url = "https://cwfis.cfs.nrcan.gc.ca/downloads/cffdrs"
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    year = start.year
    
    # Try to download a sample day
    sample_date = start + timedelta(days=30)  # Pick a mid-fire-season day
    if sample_date > end:
        sample_date = start
    
    date_str = sample_date.strftime("%Y%m%d")
    
    # Try different URL patterns
    patterns = [
        f"{base_url}/{year}/fwi{date_str}.asc",
        f"{base_url}/{year}/fwi{date_str}.nc",
        f"{base_url}/fwi{date_str}.asc",
    ]
    
    for url in patterns:
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                print(f"   [FOUND] CWFIS FWI grids available at: {base_url}/{year}/")
                print(f"   Download manually or use CWFIS data request form:")
                print(f"   https://cwfis.cfs.nrcan.gc.ca/datamart")
                return
        except:
            continue
    
    print(f"   [INFO] CWFIS daily grids not found at expected URLs")
    print(f"   Request data from: https://cwfis.cfs.nrcan.gc.ca/datamart")


# ==============================================================================
# NBAC BURNED AREA EXTRACTION
# ==============================================================================
def fetch_nbac_burned_area(bbox, year=2020):
    """
    Extract burned areas from downloaded NBAC zip file.
    
    NBAC = National Burned Area Composite
    This is the authoritative Canadian burned area dataset.
    """
    print(f"\n=== Extracting NBAC Burned Area ({year}) ===")
    out_path = OUTPUT_DIR / f"nbac_{year}.gpkg"
    
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return
    
    # Check for downloaded zip
    zip_files = list(OUTPUT_DIR.glob(f"NBAC_{year}*.zip"))
    if not zip_files:
        zip_files = list(OUTPUT_DIR.glob(f"nbac_{year}*.zip"))
    
    if not zip_files:
        print(f"   [INFO] NBAC zip file not found in {OUTPUT_DIR}")
        print(f"   Download from: https://cwfis.cfs.nrcan.gc.ca/datamart/download/nbac")
        return
    
    print(f"   Found: {zip_files[0].name}")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"   Extracting...")
            with zipfile.ZipFile(zip_files[0], 'r') as z:
                z.extractall(tmpdir)
            
            # Find shapefile
            shp_files = list(Path(tmpdir).glob("**/*.shp"))
            if not shp_files:
                print(f"   [ERROR] No shapefile found in zip")
                return
            
            print(f"   Reading shapefile: {shp_files[0].name}")
            gdf = gpd.read_file(shp_files[0])
            
            # Ensure CRS is set
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.to_crs("EPSG:4326")
            
            print(f"   Total fires in dataset: {len(gdf)}")
            
            # Filter to bbox
            west, south, east, north = bbox
            bbox_geom = box(west, south, east, north)
            gdf_filtered = gdf[gdf.intersects(bbox_geom)]
            
            # Also filter by year if column exists
            year_cols = [c for c in gdf_filtered.columns if 'year' in c.lower()]
            if year_cols and not gdf_filtered.empty:
                try:
                    gdf_filtered = gdf_filtered[gdf_filtered[year_cols[0]] == year]
                except:
                    pass
            
            if not gdf_filtered.empty:
                gdf_filtered.to_file(out_path, driver="GPKG")
                print(f"   [SUCCESS] Found {len(gdf_filtered)} burned areas in bbox: {out_path}")
                
                # Print some stats
                if 'SIZE_HA' in gdf_filtered.columns:
                    total_ha = gdf_filtered['SIZE_HA'].sum()
                    print(f"   Total area burned: {total_ha:,.0f} ha")
            else:
                print(f"   [INFO] No burned areas in bbox for {year}")
                # Create empty file
                gdf_empty = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
                gdf_empty.to_file(out_path, driver="GPKG")
                
    except Exception as e:
        print(f"   [ERROR] NBAC extraction failed: {e}")


# ==============================================================================
# 4. LANDFIRE FBFM40 + CANOPY FUELS (USA ONLY - Skip for Canada)
# ==============================================================================
def fetch_landfire_fuels(bbox, email="user@example.com"):
    """
    Fetch LANDFIRE fuel data via LFPS API.
    USA only (CONUS, Alaska, Hawaii).
    """
    print("\n=== Fetching LANDFIRE Fuel Data ===")
    out_path = OUTPUT_DIR / "landfire_fuels.tif"
    
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return
    
    # Check bbox is in USA
    if not is_in_usa(bbox):
        print(f"   [SKIP] LANDFIRE only covers USA (CONUS, AK, HI)")
        print(f"   Your bbox {bbox} is outside USA coverage.")
        return
    
    print(f"   [INFO] Bbox is in USA, proceeding with LANDFIRE...")
    
    # LFPS API endpoint
    lfps_url = "https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/GPServer/LandfireProductService/submitJob"
    
    # Products
    products = ["240FBFM40", "240CC", "240CH", "240CBH", "240CBD"]
    
    # Format AOI
    aoi = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
    
    params = {
        "Layer_List": ";".join(products),
        "Area_of_Interest": aoi,
        "Output_Projection": "4326",
        "Resample_Resolution": "30",
        "f": "json"
    }
    
    print(f"   Products: {products}")
    print(f"   AOI: {aoi}")
    
    try:
        print("   Submitting job to LFPS...")
        response = requests.get(lfps_url, params=params, timeout=60)
        
        # Check content type
        if 'application/json' not in response.headers.get('content-type', ''):
            print(f"   [ERROR] LFPS returned non-JSON response")
            print(f"   Response: {response.text[:200]}")
            return
        
        result = response.json()
        
        if "jobId" not in result:
            print(f"   [ERROR] Job submission failed: {result}")
            return
        
        job_id = result["jobId"]
        print(f"   Job ID: {job_id}")
        
        # Poll for completion
        status_url = f"https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/GPServer/LandfireProductService/jobs/{job_id}"
        
        max_wait = 600
        wait_time = 0
        
        while wait_time < max_wait:
            status_resp = requests.get(status_url, params={"f": "json"}, timeout=30)
            status = status_resp.json()
            job_status = status.get("jobStatus", "")
            
            print(f"   Status: {job_status} ({wait_time}s)")
            
            if job_status == "esriJobSucceeded":
                break
            elif job_status in ["esriJobFailed", "esriJobCancelled"]:
                print(f"   [ERROR] Job failed: {status}")
                return
            
            time.sleep(30)
            wait_time += 30
        
        if wait_time >= max_wait:
            print(f"   [ERROR] Job timed out")
            return
        
        # Get download URL
        result_url = f"{status_url}/results/Output_File"
        result_resp = requests.get(result_url, params={"f": "json"}, timeout=30)
        result_data = result_resp.json()
        
        download_url = result_data.get("value", {}).get("url")
        if not download_url:
            print(f"   [ERROR] No download URL in response")
            return
        
        # Download
        print(f"   Downloading...")
        zip_path = OUTPUT_DIR / "landfire_temp.zip"
        
        with requests.get(download_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        
        # Extract
        extract_dir = OUTPUT_DIR / "landfire_temp"
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        
        tif_files = list(extract_dir.glob("**/*.tif"))
        if tif_files:
            shutil.copy(tif_files[0], out_path)
            print(f"   [SUCCESS] Saved: {out_path}")
        
        # Cleanup
        if zip_path.exists():
            os.remove(zip_path)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
            
    except Exception as e:
        print(f"   [ERROR] LANDFIRE fetch failed: {e}")


# ==============================================================================
# 5. CANADIAN FBP FUEL TYPES (FIXED - WCS 1.0.0 with correct parameters)
# ==============================================================================
def fetch_canadian_fbp_fuels(bbox):
    """
    Fetch Canadian FBP Fuel Types using WCS 1.0.0.
    
    The CWFIS WCS requires:
    - Version 1.0.0 (not 2.0.1!)
    - BBOX in EPSG:3978 (Canada Atlas Lambert)
    - WIDTH and HEIGHT parameters
    - coverage parameter (not CoverageID)
    """
    print("\n=== Fetching Canadian FBP Fuel Types ===")
    out_path = OUTPUT_DIR / "canadian_fbp_fuels.tif"
    
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return
    
    if not is_in_canada(bbox):
        print(f"   [SKIP] Canadian FBP fuels only cover Canada")
        return
    
    # Transform bbox to EPSG:3978 (Canada Atlas Lambert)
    bbox_3978 = transform_bbox_to_crs(bbox, "EPSG:4326", "EPSG:3978")
    
    print(f"   Input bbox (EPSG:4326): {bbox}")
    print(f"   Transformed bbox (EPSG:3978): {bbox_3978}")
    
    # Calculate dimensions (~30m resolution)
    width = int((bbox_3978[2] - bbox_3978[0]) / 30)
    height = int((bbox_3978[3] - bbox_3978[1]) / 30)
    
    # Cap at reasonable size
    max_dim = 5000
    if width > max_dim or height > max_dim:
        scale = max(width, height) / max_dim
        width = int(width / scale)
        height = int(height / scale)
    
    print(f"   Requesting {width}x{height} pixels")
    
    # WCS 1.0.0 request (as documented by CWFIS)
    wcs_url = "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs"
    
    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": "public:fwi_fsr_current",  # Try FWI layer first as test
        "BBOX": f"{bbox_3978[0]},{bbox_3978[1]},{bbox_3978[2]},{bbox_3978[3]}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "CRS": "EPSG:3978",
        "FORMAT": "geotiff"
    }
    
    # First, let's try to get the FBP fuel types layer
    # The layer name might be different - let's check capabilities first
    print("   Checking available WCS coverages...")
    
    cap_url = f"{wcs_url}?service=WCS&version=1.0.0&request=GetCapabilities"
    try:
        cap_resp = requests.get(cap_url, timeout=30)
        if cap_resp.status_code == 200:
            # Look for fuel type coverage names
            cap_text = cap_resp.text
            if "cffdrs_fbp_fuel_types" in cap_text:
                params["coverage"] = "public:cffdrs_fbp_fuel_types"
            elif "fbp_fuel" in cap_text.lower():
                # Try to find the coverage name
                import re
                matches = re.findall(r'<CoverageOfferingBrief>.*?<name>(.*?)</name>', 
                                    cap_text, re.DOTALL)
                for m in matches:
                    if 'fbp' in m.lower() or 'fuel' in m.lower():
                        params["coverage"] = m
                        break
    except:
        pass
    
    # Try direct GeoTIFF download from NFIS (more reliable)
    print("   Trying NFIS direct download...")
    nfis_url = "https://ca.nfis.org/fss/fss"
    nfis_params = {
        "command": "retrieveByName",
        "fileName": "FBP_Canada_30m_3978_22052024_forRelease.tif",
        "fileNameSpace": "fire_behaviour_prediction",
        "promptToSave": "true"
    }
    
    try:
        # Check if file exists and is accessible
        head_resp = requests.head(
            f"{nfis_url}?command={nfis_params['command']}&fileName={nfis_params['fileName']}&fileNameSpace={nfis_params['fileNameSpace']}",
            timeout=10
        )
        
        if head_resp.status_code == 200:
            content_length = int(head_resp.headers.get('content-length', 0))
            
            if content_length > 0:
                print(f"   Full dataset available ({content_length / 1e9:.2f} GB)")
                print(f"   This is the complete Canada fuel map.")
                print(f"   For a local subset, WCS is preferred but may not be available.")
                
                # Ask if user wants full download (it's ~2GB)
                # For now, skip and try WCS
                
    except Exception as e:
        print(f"   [INFO] NFIS check failed: {e}")
    
    # Try WCS request
    print("   Attempting WCS GetCoverage...")
    
    # Try different coverage names
    coverage_names = [
        "public:cffdrs_fbp_fuel_types",
        "cffdrs_fbp_fuel_types", 
        "public:fbpft",
        "fbpft"
    ]
    
    for cov_name in coverage_names:
        params["coverage"] = cov_name
        
        try:
            url_with_params = f"{wcs_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
            print(f"   Trying coverage: {cov_name}")
            
            response = requests.get(url_with_params, timeout=120)
            
            if response.status_code == 200:
                # Check if it's actually a GeoTIFF (not an error XML)
                content = response.content
                if content[:4] in [b'II*\x00', b'MM\x00*']:  # TIFF magic numbers
                    # Save temp file in 3978
                    temp_path = OUTPUT_DIR / "fbp_temp_3978.tif"
                    with open(temp_path, 'wb') as f:
                        f.write(content)
                    
                    print(f"   Downloaded in EPSG:3978, reprojecting...")
                    
                    # Reproject to 4326
                    da = rioxarray.open_rasterio(temp_path)
                    da_reproj = da.rio.reproject("EPSG:4326")
                    da_reproj = da_reproj.rio.clip_box(*bbox)
                    da_reproj.rio.to_raster(out_path, compress='LZW')
                    
                    os.remove(temp_path)
                    print(f"   [SUCCESS] Saved: {out_path}")
                    return
                else:
                    # Probably an XML error
                    print(f"   [INFO] {cov_name} returned XML (not data)")
            else:
                print(f"   [INFO] {cov_name} returned HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   [WARN] {cov_name} failed: {e}")
    
    # If WCS fails, provide instructions
    print("   [WARN] WCS not available for FBP fuel types")
    print("   ")
    print("   The FBP fuel types can be downloaded manually:")
    print("   1. Full national dataset (2GB):")
    print("      https://open.canada.ca/data/en/dataset/4e66dd2f-5cd0-42fd-b82c-a430044b31de")
    print("   ")
    print("   2. Or use the CWFIS interactive map to view coverage:")
    print("      https://cwfis.cfs.nrcan.gc.ca/interactive-map")


# ==============================================================================
# 6. MTBS BURN SEVERITY (USA ONLY)
# ==============================================================================
def fetch_mtbs_burn_severity(bbox, year=2020):
    """
    Fetch MTBS burn severity data.
    USA only.
    """
    print(f"\n=== Fetching MTBS Burn Severity ({year}) ===")
    out_path = OUTPUT_DIR / f"mtbs_burn_severity_{year}.tif"
    perimeter_path = OUTPUT_DIR / f"mtbs_perimeters_{year}.gpkg"
    
    if out_path.exists() or perimeter_path.exists():
        print(f"   [SKIP] Files exist")
        return
    
    if not is_in_usa(bbox):
        print(f"   [SKIP] MTBS only covers USA territories")
        print(f"   Your bbox {bbox} is outside USA coverage.")
        return
    
    # Try Earth Engine
    try:
        import ee
        
        try:
            ee.Initialize(opt_url='https://earthengine.googleapis.com')
        except:
            try:
                ee.Number(1).getInfo()
            except:
                raise ImportError("GEE not initialized")
        
        print("   Using Google Earth Engine...")
        west, south, east, north = bbox
        region = ee.Geometry.Rectangle([west, south, east, north])
        
        mtbs = ee.ImageCollection('USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1') \
            .filterDate(f'{year}-01-01', f'{year}-12-31')
        
        if mtbs.size().getInfo() == 0:
            print(f"   [INFO] No MTBS data for {year}")
            return
        
        mtbs_image = mtbs.first().clip(region)
        
        url = mtbs_image.getDownloadURL({
            'scale': 30,
            'crs': 'EPSG:4326',
            'region': region,
            'format': 'GEO_TIFF'
        })
        
        response = requests.get(url, timeout=300)
        
        if response.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(response.content)
            print(f"   [SUCCESS] Saved: {out_path}")
            return
            
    except ImportError:
        print("   [INFO] Earth Engine not available")
    except Exception as e:
        print(f"   [WARN] Earth Engine failed: {e}")
    
    print("   [INFO] MTBS requires Google Earth Engine for raster data")
    print("   Perimeters can be downloaded from:")
    print("   https://www.mtbs.gov/direct-download")


# ==============================================================================
# CWFIS HOTSPOTS (FIXED - WFS 1.1.0 with local filtering)
# ==============================================================================
def fetch_cwfis_hotspots(bbox, days=7):
    """
    Fetch CWFIS satellite hotspots for Canada.
    This is more reliable than FIRMS for Canadian coverage.
    
    Uses WFS 1.1.0 (more compatible) and filters locally.
    """
    print(f"\n=== Fetching CWFIS Hotspots (last {days} days) ===")
    out_path = OUTPUT_DIR / "cwfis_hotspots.gpkg"
    
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return
    
    if not is_in_canada(bbox):
        print(f"   [SKIP] CWFIS only covers Canada")
        return
    
    west, south, east, north = bbox
    
    # Use WFS 1.1.0 with OWS endpoint (more compatible)
    # Don't include bbox in request - filter locally instead
    wfs_url = "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows"
    
    params = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": "public:hotspots_last7days",
        "outputFormat": "json",
        "srsName": "EPSG:4326",
    }
    
    try:
        print(f"   Requesting all Canadian hotspots (will filter to bbox)...")
        response = requests.get(wfs_url, params=params, timeout=60)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                if data.get("features"):
                    gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
                    
                    # Filter to bbox locally
                    bbox_geom = box(west, south, east, north)
                    gdf_filtered = gdf[gdf.intersects(bbox_geom)]
                    
                    if not gdf_filtered.empty:
                        gdf_filtered.to_file(out_path, driver="GPKG")
                        print(f"   [SUCCESS] Found {len(gdf_filtered)} hotspots in bbox: {out_path}")
                        print(f"   (Total in Canada: {len(gdf)})")
                    else:
                        print(f"   [INFO] No hotspots in bbox (but {len(gdf)} active in Canada)")
                        gdf_empty = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
                        gdf_empty.to_file(out_path, driver="GPKG")
                        print(f"   [SUCCESS] Created empty file: {out_path}")
                else:
                    print("   [INFO] No hotspots currently active in Canada")
                    gdf = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
                    gdf.to_file(out_path, driver="GPKG")
                    print(f"   [SUCCESS] Created empty file: {out_path}")
            except json.JSONDecodeError:
                print(f"   [ERROR] Failed to parse WFS response as JSON")
                print(f"   Response preview: {response.text[:200]}")
        else:
            print(f"   [ERROR] WFS returned HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"   [ERROR] CWFIS hotspots failed: {e}")


# ==============================================================================
# CWFIS FWI GRIDS (BONUS - Canadian daily FWI)
# ==============================================================================
def fetch_cwfis_fwi_archive(bbox, date_str="2020-07-01"):
    """
    Fetch archived FWI grid from CWFIS.
    Daily grids in PNG format (good for visualization).
    
    For analysis, use the CFFDRS data instead.
    """
    print(f"\n=== Fetching CWFIS FWI for {date_str} ===")
    out_path = OUTPUT_DIR / f"cwfis_fwi_{date_str.replace('-', '')}.png"
    
    if out_path.exists():
        print(f"   [SKIP] File exists: {out_path}")
        return
    
    # Parse date
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    
    # FWI map URL
    url = f"https://cwfis.cfs.nrcan.gc.ca/data/maps/fwi_fbp/{year}/fwi{date_str.replace('-', '')}.png"
    
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(response.content)
            print(f"   [SUCCESS] Saved: {out_path}")
            print(f"   Note: This is a PNG image, not georeferenced data.")
            print(f"   For analysis, use CFFDRS daily grids from:")
            print(f"   https://cwfis.cfs.nrcan.gc.ca/downloads/cffdrs/")
        else:
            print(f"   [ERROR] HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   [ERROR] {e}")


# ==============================================================================
# MAIN
# ==============================================================================
def fetch_all_fire_data(bbox, start_date, end_date):
    """Fetch all available fire-related datasets."""
    
    year = int(start_date[:4])
    
    print("=" * 70)
    print("FIRE MODEL DATA FETCHER V2")
    print("=" * 70)
    print(f"BBOX: {bbox}")
    print(f"Time Range: {start_date} to {end_date}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Location: {'USA' if is_in_usa(bbox) else ''} {'Canada' if is_in_canada(bbox) else ''}")
    print("=" * 70)
    
    # 1. FIRMS Active Fire (Global)
    fetch_firms_active_fire(bbox, start_date, end_date)
    
    # 2. CWFIS Hotspots (Canada - more reliable)
    fetch_cwfis_hotspots(bbox)
    
    # 3. FireCCI51 Burned Area (Global - requires GEE)
    fetch_firecci51_burned_area(bbox, year)
    
    # 4. Fire Weather Index (Global - requires EWDS account)
    fetch_fire_weather_index(bbox, start_date, end_date)
    
    # 5. LANDFIRE Fuels (USA only)
    fetch_landfire_fuels(bbox)
    
    # 6. Canadian FBP Fuels (Canada only)
    fetch_canadian_fbp_fuels(bbox)
    
    # 7. MTBS Burn Severity (USA only)
    fetch_mtbs_burn_severity(bbox, year)
    
    # 8. Extract NBAC if downloaded (Canada)
    if is_in_canada(bbox):
        fetch_nbac_burned_area(bbox, year)
    
    print("\n" + "=" * 70)
    print("FIRE DATA FETCH COMPLETE")
    print("=" * 70)
    
    # Summary
    print("\nDownloaded files:")
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file():
            size = f.stat().st_size
            if size > 1e6:
                print(f"  {f.name}: {size/1e6:.1f} MB")
            else:
                print(f"  {f.name}: {size/1e3:.1f} KB")


if __name__ == "__main__":
    fetch_all_fire_data(BBOX, TIME_START, TIME_END)