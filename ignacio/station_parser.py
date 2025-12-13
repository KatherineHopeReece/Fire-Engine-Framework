"""
Station Data Parser for Canadian Weather Station CSV Files.

Parses weather station data in the format used by Alberta/Canada climate
stations (e.g., Banff CS data from ACIS/DataStream).

Example input format (Alberta ACIS):
    "Station Name","Date (Local Standard Time)","Air Temp. Avg. (?C)",...

This module converts station data to the format expected by Ignacio's
weather processing pipeline.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Column name mappings for different station data formats
# Uses partial matching (case-insensitive)
ALBERTA_STATION_COLUMNS = {
    # Input column name patterns -> standardized names
    "date": ["date (local", "date", "datetime", "time"],
    "station_name": ["station name", "station_name", "station"],
    "temperature": ["air temp", "temperature", "temp"],
    "relative_humidity": ["relative humidity", "humidity", "rh"],
    "precipitation": ["precip", "precipitation", "rain"],
    "wind_speed": ["wind speed", "windspeed", "ws"],
    "wind_direction": ["wind dir", "winddir", "wd"],
    "ffmc": ["fine fuel moisture", "ffmc"],
    "isi": ["initial spread", "isi"],
    "dmc": ["duff moisture", "dmc"],
    "dc": ["drought code", "dc"],
    "bui": ["buildup index", "bui"],
    "fwi": ["fire weather index", "fwi"],
}


def find_column(df: pd.DataFrame, column_type: str) -> Optional[str]:
    """Find a column in DataFrame by matching against known patterns."""
    patterns = ALBERTA_STATION_COLUMNS.get(column_type, [])
    for col in df.columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern.lower() in col_lower:
                return col
    return None


def parse_alberta_date(date_str: str) -> datetime:
    """Parse date strings in Alberta station format."""
    if pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # Try various formats
    formats = [
        "%d-%B-%Y",      # "01-February-2014"
        "%d-%b-%Y",      # "01-Feb-2014"
        "%Y-%m-%d",      # "2014-02-01"
        "%d/%m/%Y",      # "01/02/2014"
        "%m/%d/%Y",      # "02/01/2014"
        "%Y%m%d",        # "20140201"
        "%Y-%m-%d %H:%M:%S",  # "2014-02-01 12:00:00"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try pandas parser as fallback
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except:
        raise ValueError(f"Cannot parse date: {date_str}")


def load_station_csv(
    csv_path: Path | str,
    date_column: Optional[str] = None,
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and parse a station weather CSV file.
    
    Parameters
    ----------
    csv_path : Path or str
        Path to CSV file
    date_column : str, optional
        Name of date column (auto-detected if not provided)
    date_format : str, optional
        Date format string (auto-detected if not provided)
        
    Returns
    -------
    pd.DataFrame
        Parsed weather data with standardized column names
    """
    csv_path = Path(csv_path)
    logger.info(f"Loading station data from {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    
    # Find date column
    if date_column is None:
        date_column = find_column(df, "date")
    if date_column is None:
        raise ValueError("Could not find date column in CSV")
    
    # Parse dates
    if date_format:
        df["datetime"] = pd.to_datetime(df[date_column], format=date_format)
    else:
        # Try auto-detection
        try:
            df["datetime"] = pd.to_datetime(df[date_column])
        except:
            df["datetime"] = df[date_column].apply(parse_alberta_date)
    
    # Standardize column names
    standardized = {"datetime": df["datetime"]}
    
    for std_name, _ in ALBERTA_STATION_COLUMNS.items():
        col = find_column(df, std_name)
        if col is not None:
            standardized[std_name] = pd.to_numeric(df[col], errors="coerce")
    
    result = pd.DataFrame(standardized)
    
    # Sort by date
    result = result.sort_values("datetime").reset_index(drop=True)
    
    logger.info(f"Parsed columns: {list(result.columns)}")
    logger.info(f"Date range: {result['datetime'].min()} to {result['datetime'].max()}")
    
    return result


def extract_weather_for_date(
    station_data: pd.DataFrame,
    target_date: datetime,
    hour: int = 12,
) -> dict:
    """
    Extract weather conditions for a specific date/time.
    
    Parameters
    ----------
    station_data : pd.DataFrame
        Parsed station data
    target_date : datetime
        Target date
    hour : int
        Hour of day (default 12 = noon, typical fire weather observation time)
        
    Returns
    -------
    dict
        Weather conditions for the specified time
    """
    # Find closest date
    target = pd.Timestamp(target_date.replace(hour=hour))
    station_data["time_diff"] = abs(station_data["datetime"] - target)
    closest = station_data.loc[station_data["time_diff"].idxmin()]
    
    result = {
        "datetime": closest["datetime"],
        "temperature": closest.get("temperature", 20.0),
        "relative_humidity": closest.get("relative_humidity", 30.0),
        "wind_speed": closest.get("wind_speed", 10.0),
        "wind_direction": closest.get("wind_direction", 270.0),
        "precipitation": closest.get("precipitation", 0.0),
        "ffmc": closest.get("ffmc", 85.0),
        "dmc": closest.get("dmc", 25.0),
        "dc": closest.get("dc", 200.0),
        "isi": closest.get("isi", 10.0),
        "bui": closest.get("bui", 70.0),
        "fwi": closest.get("fwi", 20.0),
    }
    
    # Handle NaN values with defaults
    defaults = {
        "temperature": 20.0,
        "relative_humidity": 30.0,
        "wind_speed": 10.0,
        "wind_direction": 270.0,
        "precipitation": 0.0,
        "ffmc": 85.0,
        "dmc": 25.0,
        "dc": 200.0,
        "isi": 10.0,
        "bui": 70.0,
        "fwi": 20.0,
    }
    
    for key, default in defaults.items():
        if pd.isna(result[key]):
            result[key] = default
    
    return result


def get_weather_sequence(
    station_data: pd.DataFrame,
    start_date: datetime,
    duration_hours: int,
    interval_hours: int = 1,
) -> list[dict]:
    """
    Extract a sequence of weather conditions for multi-day simulation.
    
    Parameters
    ----------
    station_data : pd.DataFrame
        Parsed station data
    start_date : datetime
        Start date/time
    duration_hours : int
        Simulation duration in hours
    interval_hours : int
        Interval between weather updates
        
    Returns
    -------
    list[dict]
        List of weather conditions at each interval
    """
    from datetime import timedelta
    
    weather_sequence = []
    current_time = start_date
    end_time = start_date + timedelta(hours=duration_hours)
    
    while current_time <= end_time:
        weather = extract_weather_for_date(
            station_data, current_time, hour=current_time.hour
        )
        weather["datetime"] = current_time
        weather_sequence.append(weather)
        current_time += timedelta(hours=interval_hours)
    
    return weather_sequence


def convert_to_ignacio_format(
    station_data: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Convert station data to Ignacio's expected hourly weather format.
    
    Parameters
    ----------
    station_data : pd.DataFrame
        Parsed station data
    output_path : Path, optional
        If provided, save converted data to CSV
        
    Returns
    -------
    pd.DataFrame
        Data in Ignacio format
    """
    # Ignacio expects these columns
    ignacio_columns = {
        "DATE_TIME": station_data["datetime"].dt.strftime("%Y%m%d%H"),
        "STATION_CODE": "STATION",
        "HOURLY_TEMPERATURE": station_data.get("temperature", 20.0),
        "HOURLY_RELATIVE_HUMIDITY": station_data.get("relative_humidity", 30.0),
        "HOURLY_WIND_SPEED": station_data.get("wind_speed", 10.0),
        "HOURLY_WIND_DIRECTION": station_data.get("wind_direction", 270.0),
        "PRECIPITATION": station_data.get("precipitation", 0.0),
        "FFMC": station_data.get("ffmc", 85.0),
        "DMC": station_data.get("dmc", 25.0),
        "DC": station_data.get("dc", 200.0),
        "INITIAL_SPREAD_INDEX": station_data.get("isi", 10.0),
        "BUI": station_data.get("bui", 70.0),
        "FIRE_WEATHER_INDEX": station_data.get("fwi", 20.0),
    }
    
    result = pd.DataFrame(ignacio_columns)
    
    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved Ignacio-format weather to {output_path}")
    
    return result


def summarize_fire_weather(
    station_data: pd.DataFrame,
    date_range: Optional[tuple] = None,
) -> dict:
    """
    Summarize fire weather conditions for a date range.
    
    Parameters
    ----------
    station_data : pd.DataFrame
        Parsed station data
    date_range : tuple, optional
        (start_date, end_date) to filter
        
    Returns
    -------
    dict
        Summary statistics
    """
    df = station_data.copy()
    
    if date_range:
        start, end = date_range
        df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
    
    summary = {
        "n_records": len(df),
        "date_range": (df["datetime"].min(), df["datetime"].max()),
    }
    
    for col in ["temperature", "relative_humidity", "wind_speed", "ffmc", "isi", "fwi"]:
        if col in df.columns:
            summary[f"{col}_mean"] = df[col].mean()
            summary[f"{col}_max"] = df[col].max()
            summary[f"{col}_min"] = df[col].min()
    
    # Fire danger days
    if "ffmc" in df.columns:
        summary["high_ffmc_days"] = (df["ffmc"] > 89).sum()
    if "isi" in df.columns:
        summary["high_isi_days"] = (df["isi"] > 10).sum()
    if "fwi" in df.columns:
        summary["extreme_fwi_days"] = (df["fwi"] > 25).sum()
    
    return summary
