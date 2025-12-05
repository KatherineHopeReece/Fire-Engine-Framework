"""
Raster and vector I/O utilities for Ignacio.

This module provides functions for reading and writing geospatial data,
including rasters (GeoTIFF) and vectors (shapefiles, GeoJSON, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine, rowcol, xy
from shapely.geometry import Point, Polygon, MultiPolygon

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RasterData:
    """Container for raster data with metadata."""
    
    data: np.ndarray
    transform: Affine
    crs: CRS
    nodata: float | None = None
    
    @property
    def shape(self) -> tuple[int, int]:
        """Return (height, width) of the raster."""
        return self.data.shape
    
    @property
    def height(self) -> int:
        """Return raster height (number of rows)."""
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        """Return raster width (number of columns)."""
        return self.data.shape[1]
    
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (left, bottom, right, top) bounds."""
        left = self.transform.c
        top = self.transform.f
        right = left + self.width * self.transform.a
        bottom = top + self.height * self.transform.e
        return (left, bottom, right, top)
    
    @property
    def resolution(self) -> tuple[float, float]:
        """Return (x_res, y_res) cell size."""
        return (abs(self.transform.a), abs(self.transform.e))
    
    def xy(self, row: int, col: int) -> tuple[float, float]:
        """Convert row/col indices to x/y coordinates."""
        return xy(self.transform, row, col, offset="center")
    
    def rowcol(self, x: float, y: float) -> tuple[int, int]:
        """Convert x/y coordinates to row/col indices."""
        return rowcol(self.transform, x, y)
    
    def get_coordinate_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return 1D arrays of x and y coordinates for cell centers."""
        cols = np.arange(self.width)
        rows = np.arange(self.height)
        x_coords = self.transform.c + (cols + 0.5) * self.transform.a
        y_coords = self.transform.f + (rows + 0.5) * self.transform.e
        return x_coords, y_coords


# =============================================================================
# Raster I/O
# =============================================================================


def read_raster(path: str | Path, band: int = 1, masked: bool = True) -> RasterData:
    """
    Read a raster file into a RasterData container.
    
    Parameters
    ----------
    path : str or Path
        Path to the raster file.
    band : int
        Band number to read (1-indexed).
    masked : bool
        If True, mask nodata values with NaN.
        
    Returns
    -------
    RasterData
        Container with raster data and metadata.
    """
    path = Path(path)
    logger.debug(f"Reading raster: {path}")
    
    with rasterio.open(path) as src:
        data = src.read(band, masked=masked)
        nodata = src.nodata
        
        # Convert to float64 for consistent handling
        if masked and hasattr(data, "filled"):
            # For masked arrays, convert to float and fill with NaN
            data = data.astype(np.float64)
            data = data.filled(np.nan)
        else:
            data = data.astype(np.float64)
            # Mark nodata values as NaN if nodata is defined
            if nodata is not None:
                data[data == nodata] = np.nan
        
        return RasterData(
            data=data,
            transform=src.transform,
            crs=src.crs,
            nodata=nodata,
        )


def read_raster_int(path: str | Path, band: int = 1, nodata_value: int = -9999) -> RasterData:
    """
    Read an integer raster file (like fuel grids) preserving integer values.
    
    Parameters
    ----------
    path : str or Path
        Path to the raster file.
    band : int
        Band number to read (1-indexed).
    nodata_value : int
        Value to use for nodata cells in output.
        
    Returns
    -------
    RasterData
        Container with raster data as float (nodata as nodata_value).
    """
    path = Path(path)
    logger.debug(f"Reading integer raster: {path}")
    
    with rasterio.open(path) as src:
        data = src.read(band, masked=True)
        nodata = src.nodata
        
        # Convert to float but preserve integer values
        if hasattr(data, "filled"):
            # Fill masked values with the specified nodata value
            data = data.filled(nodata_value).astype(np.float64)
        else:
            data = data.astype(np.float64)
            if nodata is not None:
                data[data == nodata] = nodata_value
        
        return RasterData(
            data=data,
            transform=src.transform,
            crs=src.crs,
            nodata=nodata_value,
        )


def write_raster(
    path: str | Path,
    data: np.ndarray,
    transform: Affine,
    crs: CRS | str,
    nodata: float | None = None,
    dtype: str = "float32",
) -> None:
    """
    Write a numpy array to a GeoTIFF file.
    
    Parameters
    ----------
    path : str or Path
        Output file path.
    data : np.ndarray
        2D array to write.
    transform : Affine
        Affine transform for georeferencing.
    crs : CRS or str
        Coordinate reference system.
    nodata : float, optional
        NoData value.
    dtype : str
        Output data type.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Writing raster: {path}")
    
    if isinstance(crs, str):
        crs = CRS.from_string(crs)
    
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def rasterize_polygons(
    gdf: gpd.GeoDataFrame,
    reference: RasterData,
    value_column: str | None = None,
    fill: float = 0,
    dtype: str = "int32",
) -> np.ndarray:
    """
    Rasterize polygon features to match a reference raster.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with polygon geometries.
    reference : RasterData
        Reference raster for extent and resolution.
    value_column : str, optional
        Column containing raster values. If None, uses 1 for all features.
    fill : float
        Fill value for areas outside polygons.
    dtype : str
        Output data type.
        
    Returns
    -------
    np.ndarray
        Rasterized array matching reference dimensions.
    """
    if value_column is not None and value_column in gdf.columns:
        values = gdf[value_column].values
    else:
        values = np.ones(len(gdf), dtype=int)
    
    shapes = list(zip(gdf.geometry, values))
    
    result = rasterize(
        shapes=shapes,
        out_shape=reference.shape,
        transform=reference.transform,
        fill=fill,
        dtype=dtype,
    )
    
    return result


# =============================================================================
# Vector I/O
# =============================================================================


def read_vector(
    path: str | Path,
    target_crs: str | CRS | None = None,
) -> gpd.GeoDataFrame:
    """
    Read a vector file into a GeoDataFrame.
    
    Parameters
    ----------
    path : str or Path
        Path to vector file (shapefile, GeoJSON, etc.).
    target_crs : str or CRS, optional
        Target CRS to reproject to.
        
    Returns
    -------
    GeoDataFrame
        Vector data with geometries.
    """
    path = Path(path)
    logger.debug(f"Reading vector: {path}")
    
    gdf = gpd.read_file(path)
    
    if target_crs is not None:
        if gdf.crs is None:
            logger.warning(f"Vector has no CRS, assuming {target_crs}")
            gdf = gdf.set_crs(target_crs)
        else:
            gdf = gdf.to_crs(target_crs)
    
    return gdf


def write_vector(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
    driver: str | None = None,
) -> None:
    """
    Write a GeoDataFrame to a vector file.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Data to write.
    path : str or Path
        Output file path.
    driver : str, optional
        Output driver. Inferred from extension if not provided.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Writing vector: {path}")
    
    if driver is None:
        ext = path.suffix.lower()
        driver_map = {
            ".shp": "ESRI Shapefile",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".gpkg": "GPKG",
        }
        driver = driver_map.get(ext, "ESRI Shapefile")
    
    gdf.to_file(path, driver=driver)


def points_to_geodataframe(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    crs: str | CRS,
    **attributes: Any,
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from point coordinates.
    
    Parameters
    ----------
    x : array-like
        X coordinates.
    y : array-like
        Y coordinates.
    crs : str or CRS
        Coordinate reference system.
    **attributes
        Additional columns to include in the GeoDataFrame.
        
    Returns
    -------
    GeoDataFrame
        Point geometries with attributes.
    """
    geometry = [Point(xi, yi) for xi, yi in zip(x, y)]
    
    data = {"geometry": geometry}
    data.update(attributes)
    
    return gpd.GeoDataFrame(data, crs=crs)


def polygon_from_vertices(
    x: np.ndarray,
    y: np.ndarray,
    crs: str | CRS,
) -> gpd.GeoDataFrame:
    """
    Create a polygon GeoDataFrame from vertex coordinates.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates of vertices.
    y : np.ndarray
        Y coordinates of vertices.
    crs : str or CRS
        Coordinate reference system.
        
    Returns
    -------
    GeoDataFrame
        Single polygon geometry.
    """
    # Close the polygon if not already closed
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    
    coords = list(zip(x, y))
    polygon = Polygon(coords)
    
    return gpd.GeoDataFrame({"geometry": [polygon]}, crs=crs)


# =============================================================================
# CSV I/O
# =============================================================================


def read_csv(
    path: str | Path,
    columns: dict[str, str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a CSV file with optional column renaming.
    
    Parameters
    ----------
    path : str or Path
        Path to CSV file.
    columns : dict, optional
        Mapping of original column names to new names.
    **kwargs
        Additional arguments passed to pd.read_csv.
        
    Returns
    -------
    DataFrame
        Loaded data.
    """
    path = Path(path)
    logger.debug(f"Reading CSV: {path}")
    
    df = pd.read_csv(path, **kwargs)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.upper()
    
    if columns:
        # Rename columns to standard names
        rename_map = {v.upper(): k for k, v in columns.items() if v.upper() in df.columns}
        df = df.rename(columns=rename_map)
    
    return df


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """
    Write a DataFrame to CSV.
    
    Parameters
    ----------
    df : DataFrame
        Data to write.
    path : str or Path
        Output file path.
    **kwargs
        Additional arguments passed to df.to_csv.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Writing CSV: {path}")
    
    kwargs.setdefault("index", False)
    df.to_csv(path, **kwargs)


# =============================================================================
# NumPy Archive I/O
# =============================================================================


def save_npz(
    path: str | Path,
    **arrays: np.ndarray,
) -> None:
    """
    Save arrays to a compressed NumPy archive.
    
    Parameters
    ----------
    path : str or Path
        Output file path.
    **arrays
        Named arrays to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Saving NPZ: {path}")
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load arrays from a NumPy archive.
    
    Parameters
    ----------
    path : str or Path
        Path to NPZ file.
        
    Returns
    -------
    dict
        Dictionary of array names to arrays.
    """
    path = Path(path)
    logger.debug(f"Loading NPZ: {path}")
    
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
