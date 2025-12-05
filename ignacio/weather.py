"""
Fire weather processing module for Ignacio.

This module handles loading and filtering fire weather data based on
ISI thresholds to identify burning condition days.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ignacio.config import IgnacioConfig
from ignacio.io import read_csv, write_csv

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class WeatherRecord:
    """A single weather observation."""
    
    datetime: pd.Timestamp
    station_code: str
    temperature: float
    relative_humidity: float
    wind_speed: float
    wind_direction: float
    ffmc: float | None = None
    dmc: float | None = None
    dc: float | None = None
    isi: float | None = None
    bui: float | None = None
    fwi: float | None = None
    precipitation: float = 0.0
    category: str = "moderate"


@dataclass
class FireWeatherList:
    """Collection of fire weather records meeting burning conditions."""
    
    records: pd.DataFrame
    station_metadata: pd.DataFrame | None = None
    
    @property
    def n_records(self) -> int:
        """Number of weather records."""
        return len(self.records)
    
    def get_high_conditions(self) -> pd.DataFrame:
        """Get records with high fire weather conditions."""
        if "category" in self.records.columns:
            return self.records[self.records["category"] == "high"]
        return pd.DataFrame()
    
    def get_extreme_conditions(self) -> pd.DataFrame:
        """Get records with extreme fire weather conditions."""
        if "category" in self.records.columns:
            return self.records[self.records["category"] == "extreme"]
        return pd.DataFrame()
    
    def sample_weather_stream(
        self,
        n_days: int,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """
        Sample a weather stream for fire simulation.
        
        Parameters
        ----------
        n_days : int
            Number of days to sample.
        rng : Generator, optional
            Random number generator.
            
        Returns
        -------
        DataFrame
            Sampled weather records.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if len(self.records) == 0:
            return pd.DataFrame()
        
        # Sample with replacement if needed
        n_samples = min(n_days, len(self.records))
        indices = rng.choice(len(self.records), size=n_samples, replace=n_samples > len(self.records))
        
        return self.records.iloc[indices].reset_index(drop=True)


# =============================================================================
# Weather Processing Functions
# =============================================================================


def load_station_metadata(config: IgnacioConfig) -> pd.DataFrame:
    """
    Load weather station metadata.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
        
    Returns
    -------
    DataFrame
        Station metadata.
    """
    station_path = config.weather.station_path
    col_config = config.weather.station_columns
    
    logger.info(f"Loading station metadata from {station_path}")
    
    df = read_csv(station_path)
    
    # Standardize column names
    rename_map = {}
    for attr in ["code", "name", "latitude", "longitude", "elevation", "slope", "aspect"]:
        col_name = getattr(col_config, attr, None)
        if col_name and col_name.upper() in df.columns:
            rename_map[col_name.upper()] = attr.upper()
    
    df = df.rename(columns=rename_map)
    
    # Ensure station code is string
    if "CODE" in df.columns:
        df["CODE"] = df["CODE"].astype(str)
    
    logger.info(f"Loaded {len(df)} station records")
    
    return df


def detect_weather_file_format(df: pd.DataFrame) -> str:
    """
    Detect the format of a weather data file.
    
    Returns
    -------
    str
        One of: 'hourly_weather', 'daily_fwi', 'station_metadata', 'unknown'
    """
    cols_upper = [c.upper() for c in df.columns]
    
    # Check for hourly weather data (has TEMP, RH, WS, WD)
    hourly_indicators = ['TEMP', 'RH', 'WS', 'WD', 'TEMPERATURE', 'WIND_SPEED']
    has_hourly = sum(1 for c in hourly_indicators if c in cols_upper) >= 3
    
    # Check for FWI data (has FFMC, DMC, DC, ISI, BUI)
    fwi_indicators = ['FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
    has_fwi = sum(1 for c in fwi_indicators if c in cols_upper) >= 3
    
    # Check for station metadata (has LAT, LON or LATITUDE, LONGITUDE)
    station_indicators = ['LATITUDE', 'LONGITUDE', 'LAT', 'LON', 'STATION_NAME', 'ELEVATION']
    has_station = sum(1 for c in station_indicators if c in cols_upper) >= 2
    
    if has_hourly and has_fwi:
        return 'hourly_with_fwi'
    elif has_hourly:
        return 'hourly_weather'
    elif has_fwi:
        return 'daily_fwi'
    elif has_station:
        return 'station_metadata'
    else:
        return 'unknown'


def load_weather_data(config: IgnacioConfig) -> pd.DataFrame:
    """
    Load hourly weather observations.
    
    Automatically detects file format and handles various column naming conventions.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
        
    Returns
    -------
    DataFrame
        Weather observations.
    """
    weather_path = config.weather.weather_path
    station_path = config.weather.station_path
    col_config = config.weather.weather_columns
    datetime_format = config.weather.datetime_format
    
    # Determine which file to load
    df = None
    source_file = None
    
    if weather_path and Path(weather_path).exists():
        source_file = weather_path
        logger.info(f"Loading weather data from {weather_path}")
        df = read_csv(weather_path)
    elif Path(station_path).exists():
        source_file = station_path
        logger.info(f"Loading weather data from {station_path}")
        df = read_csv(station_path)
    else:
        logger.warning("No weather data file found, using defaults")
        return pd.DataFrame()
    
    # Detect and log file format
    file_format = detect_weather_file_format(df)
    logger.info(f"Detected file format: {file_format} ({len(df)} records)")
    logger.debug(f"Available columns: {list(df.columns)}")
    
    # Standardize column names - handle common variations
    col_mapping = {
        # Temperature
        'TEMP': 'TEMPERATURE',
        'HOURLY_TEMPERATURE': 'TEMPERATURE',
        'AIR_TEMP': 'TEMPERATURE',
        # Relative humidity
        'RH': 'RELATIVE_HUMIDITY',
        'HOURLY_RELATIVE_HUMIDITY': 'RELATIVE_HUMIDITY',
        # Wind speed
        'WS': 'WIND_SPEED',
        'HOURLY_WIND_SPEED': 'WIND_SPEED',
        'WINDSPEED': 'WIND_SPEED',
        # Wind direction
        'WD': 'WIND_DIRECTION',
        'HOURLY_WIND_DIRECTION': 'WIND_DIRECTION',
        'WINDDIR': 'WIND_DIRECTION',
        # Precipitation
        'PRECIP': 'PRECIPITATION',
        'RAIN': 'PRECIPITATION',
        'PPT': 'PRECIPITATION',
        # FWI components
        'INITIAL_SPREAD_INDEX': 'ISI',
        'BUILDUP_INDEX': 'BUI',
        'FIRE_WEATHER_INDEX': 'FWI',
    }
    
    # Apply column mapping (case-insensitive)
    rename_map = {}
    for col in df.columns:
        col_upper = col.upper()
        if col_upper in col_mapping:
            rename_map[col] = col_mapping[col_upper]
        elif col_upper != col:
            rename_map[col] = col_upper
    
    df = df.rename(columns=rename_map)
    
    # Parse datetime - try configured column first, then common alternatives
    datetime_col = None
    for col in ["DATETIME", "HOURLY", "DATE", "DATE_TIME", "TIMESTAMP"]:
        if col in df.columns:
            datetime_col = col
            break
    
    if datetime_col:
        try:
            df["DATETIME"] = pd.to_datetime(df[datetime_col], format=datetime_format)
        except ValueError:
            try:
                # Try flexible parsing (handles various date formats)
                df["DATETIME"] = pd.to_datetime(df[datetime_col], dayfirst=True, errors="coerce")
            except Exception:
                logger.warning(f"Could not parse datetime column {datetime_col}")
        
        if "DATETIME" in df.columns and df["DATETIME"].notna().any():
            # Only extract hour from datetime if there's no existing HOUR column
            # (otherwise we'd overwrite actual hour data with 0s from date-only strings)
            if "HOUR" not in df.columns:
                df["HOUR"] = df["DATETIME"].dt.hour
            df["DATE"] = df["DATETIME"].dt.date
    
    # Try to find hour column if not already set
    if "HOUR" not in df.columns:
        for col in ["HOUR", "HR", "HH"]:
            if col in df.columns:
                df["HOUR"] = pd.to_numeric(df[col], errors="coerce")
                break
    
    # Ensure HOUR is numeric if it exists
    if "HOUR" in df.columns:
        df["HOUR"] = pd.to_numeric(df["HOUR"], errors="coerce")
        logger.debug(f"HOUR column range: {df['HOUR'].min()}-{df['HOUR'].max()}")
    
    # Ensure station code is string if present
    if "STATION_CODE" in df.columns:
        df["STATION_CODE"] = df["STATION_CODE"].astype(str)
    
    # Log what we found
    has_weather = all(c in df.columns for c in ['TEMPERATURE', 'RELATIVE_HUMIDITY', 'WIND_SPEED'])
    has_fwi = all(c in df.columns for c in ['FFMC', 'ISI', 'BUI'])
    
    if has_weather:
        temp_range = f"{df['TEMPERATURE'].min():.1f}-{df['TEMPERATURE'].max():.1f}°C"
        logger.info(f"  Weather data: TEMP={temp_range}, {len(df)} records")
    if has_fwi:
        isi_range = f"{df['ISI'].min():.1f}-{df['ISI'].max():.1f}"
        bui_range = f"{df['BUI'].min():.1f}-{df['BUI'].max():.1f}"
        logger.info(f"  FWI indices: ISI={isi_range}, BUI={bui_range}")
    
    return df


def classify_fire_weather(
    df: pd.DataFrame,
    isi_thresholds: dict[str, float],
) -> pd.DataFrame:
    """
    Classify weather records by fire weather intensity.
    
    Parameters
    ----------
    df : DataFrame
        Weather records with ISI column.
    isi_thresholds : dict
        ISI thresholds for categories.
        
    Returns
    -------
    DataFrame
        Records with added 'category' column.
    """
    df = df.copy()
    
    # Default category
    df["category"] = "moderate"
    
    # Get thresholds
    high_threshold = isi_thresholds.get("high", 8.6)
    extreme_threshold = isi_thresholds.get("extreme", 12.6)
    
    # ISI column name variations
    isi_col = None
    for col in ["ISI", "INITIAL_SPREAD_INDEX", "isi"]:
        if col in df.columns:
            isi_col = col
            break
    
    if isi_col is None:
        logger.warning("No ISI column found in weather data")
        return df
    
    # Classify
    isi_values = pd.to_numeric(df[isi_col], errors="coerce")
    df.loc[(isi_values >= high_threshold) & (isi_values < extreme_threshold), "category"] = "high"
    df.loc[isi_values >= extreme_threshold, "category"] = "extreme"
    
    # Log the distribution
    logger.info(f"ISI range: {isi_values.min():.2f} - {isi_values.max():.2f}")
    logger.info(f"High threshold: {high_threshold}, Extreme threshold: {extreme_threshold}")
    
    return df


def extract_daily_weather(
    df: pd.DataFrame,
    method: str = "noon",
) -> pd.DataFrame:
    """
    Extract daily weather values from hourly data.
    
    Parameters
    ----------
    df : DataFrame
        Hourly weather data.
    method : str
        Extraction method: "noon" for 12:00 values, "max" for daily maximum ISI.
        
    Returns
    -------
    DataFrame
        Daily weather records.
    """
    if "HOUR" not in df.columns:
        logger.warning("No HOUR column, returning original data")
        return df
    
    if method == "noon":
        # Use noon (12:00) observations
        daily = df[df["HOUR"] == 12].copy()
        
        if len(daily) == 0:
            logger.warning("No noon observations found, falling back to max method")
            method = "max"
    
    if method == "max":
        # Use daily maximum ISI
        isi_col = None
        for col in ["ISI", "INITIAL_SPREAD_INDEX"]:
            if col in df.columns:
                isi_col = col
                break
        
        if isi_col is None:
            logger.warning("No ISI column for max aggregation")
            return df
        
        agg_dict = {isi_col: "max"}
        
        # Add other columns as mean
        for col in ["TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED", "WIND_DIRECTION"]:
            if col in df.columns:
                agg_dict[col] = "mean"
        
        group_cols = ["STATION_CODE", "DATE"] if "STATION_CODE" in df.columns else ["DATE"]
        daily = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    return daily


def assign_spread_event_days(
    df: pd.DataFrame,
    lambda_param: float = 3.76,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Assign number of spread event days using Poisson distribution.
    
    Parameters
    ----------
    df : DataFrame
        Fire weather records.
    lambda_param : float
        Poisson parameter (mean spread event days).
    rng : Generator, optional
        Random number generator.
        
    Returns
    -------
    DataFrame
        Records with SPREAD_EVENT_DAYS column.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    df = df.copy()
    
    # Draw from Poisson distribution
    spread_days = rng.poisson(lambda_param, size=len(df))
    
    # Minimum of 1 day
    spread_days = np.clip(spread_days, 1, None)
    
    df["SPREAD_EVENT_DAYS"] = spread_days
    
    return df


# =============================================================================
# Main Weather Processing
# =============================================================================


def process_fire_weather(
    config: IgnacioConfig,
    rng: np.random.Generator | None = None,
) -> FireWeatherList:
    """
    Process fire weather data for simulation.
    
    Parameters
    ----------
    config : IgnacioConfig
        Configuration object.
    rng : Generator, optional
        Random number generator.
        
    Returns
    -------
    FireWeatherList
        Processed fire weather data.
    """
    if rng is None:
        seed = config.project.random_seed
        rng = np.random.default_rng(seed)
    
    weather_config = config.weather
    
    # Load data
    try:
        station_df = load_station_metadata(config)
    except Exception as e:
        logger.warning(f"Could not load station metadata: {e}")
        station_df = pd.DataFrame()
    
    weather_df = load_weather_data(config)
    
    if len(weather_df) == 0:
        logger.warning("No weather data loaded, using empty weather list with defaults")
        return FireWeatherList(records=pd.DataFrame(), station_metadata=station_df)
    
    # Check if we need to calculate FWI from hourly weather
    has_fwi = all(col in weather_df.columns for col in ['FFMC', 'ISI', 'BUI'])
    has_hourly_weather = all(col in weather_df.columns for col in ['TEMPERATURE', 'RELATIVE_HUMIDITY', 'WIND_SPEED'])
    
    if not has_fwi and has_hourly_weather and config.weather.calculate_fwi:
        logger.info("Calculating FWI indices from hourly weather data")
        try:
            from ignacio.fwi import calculate_fwi_from_weather
            
            # Determine latitude for FWI calculation
            latitude = config.weather.fwi_latitude
            if latitude is None:
                # Try to get from terrain config or use default
                latitude = 51.0  # Default to mid-latitude
                logger.debug(f"Using default latitude {latitude}° for FWI calculation")
            
            # Log what we're passing to FWI calculation
            logger.debug(f"Weather data columns for FWI: {list(weather_df.columns)}")
            logger.debug(f"Weather data shape: {weather_df.shape}")
            
            # Calculate FWI from hourly data
            fwi_df = calculate_fwi_from_weather(
                weather_df,
                latitude=latitude,
                temp_col='TEMPERATURE',
                rh_col='RELATIVE_HUMIDITY',
                ws_col='WIND_SPEED',
                precip_col='PRECIPITATION' if 'PRECIPITATION' in weather_df.columns else 'PRECIP',
            )
            
            if 'ISI' in fwi_df.columns and len(fwi_df) > 0:
                logger.info(
                    f"FWI calculation complete: {len(fwi_df)} days, "
                    f"ISI range {fwi_df['ISI'].min():.1f}-{fwi_df['ISI'].max():.1f}, "
                    f"BUI range {fwi_df['BUI'].min():.1f}-{fwi_df['BUI'].max():.1f}"
                )
                # Use the calculated FWI data instead of raw hourly
                weather_df = fwi_df
        except ImportError:
            logger.warning("FWI module not available, using default FWI values")
        except Exception as e:
            import traceback
            logger.warning(f"FWI calculation failed: {e}, using default FWI values")
            logger.debug(f"FWI error traceback:\n{traceback.format_exc()}")
    
    # Extract daily values
    logger.info("Extracting daily weather values")
    daily_df = extract_daily_weather(weather_df, method="noon")
    
    if len(daily_df) == 0:
        logger.warning("No daily values extracted, using all records")
        daily_df = weather_df.copy()
    
    # Check if ISI is available for classification
    isi_col = None
    for col in ["ISI", "INITIAL_SPREAD_INDEX"]:
        if col in daily_df.columns:
            isi_col = col
            break
    
    if isi_col is not None:
        # Classify by ISI thresholds
        logger.info("Classifying fire weather conditions by ISI")
        thresholds = {
            "moderate": weather_config.isi_thresholds.moderate,
            "high": weather_config.isi_thresholds.high,
            "extreme": weather_config.isi_thresholds.extreme,
        }
        daily_df = classify_fire_weather(daily_df, thresholds)
        
        # Filter to desired conditions
        filter_conditions = weather_config.filter_conditions
        if filter_conditions:
            logger.info(f"Filtering to conditions: {filter_conditions}")
            filtered_df = daily_df[daily_df["category"].isin(filter_conditions)]
            
            if len(filtered_df) == 0:
                logger.warning(
                    f"No records match filter {filter_conditions}, using all records. "
                    f"ISI range in data: {daily_df[isi_col].min():.1f} - {daily_df[isi_col].max():.1f}"
                )
            else:
                daily_df = filtered_df
    else:
        logger.warning(
            "No ISI column found in weather data. "
            "Using all records without filtering. Default ISI will be used for FBP calculations."
        )
        daily_df["category"] = "unknown"
    
    logger.info(f"Selected {len(daily_df)} fire weather records")
    
    # Assign spread event days
    logger.info("Assigning spread event days")
    daily_df = assign_spread_event_days(
        daily_df,
        lambda_param=weather_config.spread_event_lambda,
        rng=rng,
    )
    
    # Merge with station metadata if possible
    if "STATION_CODE" in daily_df.columns and "CODE" in station_df.columns:
        station_df = station_df.rename(columns={"CODE": "STATION_CODE"})
        merge_cols = ["STATION_CODE"]
        for col in ["NAME", "LATITUDE", "LONGITUDE", "ELEVATION"]:
            if col in station_df.columns:
                merge_cols.append(col)
        
        daily_df = daily_df.merge(
            station_df[merge_cols],
            on="STATION_CODE",
            how="left",
        )
    
    # Log summary
    if "category" in daily_df.columns:
        category_counts = daily_df["category"].value_counts()
        logger.info(f"Category distribution: {category_counts.to_dict()}")
    
    if "SPREAD_EVENT_DAYS" in daily_df.columns:
        logger.info(
            f"Spread event days: mean={daily_df['SPREAD_EVENT_DAYS'].mean():.2f}, "
            f"min={daily_df['SPREAD_EVENT_DAYS'].min()}, "
            f"max={daily_df['SPREAD_EVENT_DAYS'].max()}"
        )
    
    return FireWeatherList(
        records=daily_df,
        station_metadata=station_df,
    )


def get_representative_weather(
    fire_weather: FireWeatherList,
    config: IgnacioConfig,
    use_high_danger: bool = True,
    percentile: float = 75.0,
) -> dict[str, float]:
    """
    Get representative weather values for fire simulation.
    
    By default, uses weather from high fire danger days (ISI >= high threshold).
    Falls back to upper percentile of all data if no high danger days exist.
    
    Parameters
    ----------
    fire_weather : FireWeatherList
        Fire weather data.
    config : IgnacioConfig
        Configuration object.
    use_high_danger : bool
        If True, prefer weather from high/extreme danger days.
    percentile : float
        Percentile to use (default 75th = upper quartile for fire-conducive conditions).
        
    Returns
    -------
    dict
        Dictionary of weather values.
    """
    df = fire_weather.records
    defaults = config.fbp.defaults
    
    weather = {
        "ffmc": defaults.ffmc,
        "dmc": defaults.dmc,
        "dc": defaults.dc,
        "isi": defaults.isi,
        "bui": defaults.bui,
        "fwi": defaults.fwi,
        "temperature": 25.0,
        "relative_humidity": 40.0,
        "wind_speed": 15.0,
        "wind_direction": 180.0,
    }
    
    if len(df) == 0:
        logger.warning("No weather records available, using default FBP values")
        return weather
    
    # Filter to high-danger days if requested and category column exists
    high_danger_df = df
    if use_high_danger and "category" in df.columns:
        high_danger_df = df[df["category"].isin(["high", "extreme"])]
        if len(high_danger_df) > 0:
            logger.info(f"Using {len(high_danger_df)} high/extreme fire danger days for representative weather")
        else:
            logger.info(f"No high/extreme days found, using {percentile:.0f}th percentile of all {len(df)} days")
            high_danger_df = df
    
    # Update with actual values from the selected data
    col_mapping = {
        "ffmc": ["FFMC"],
        "dmc": ["DMC"],
        "dc": ["DC"],
        "isi": ["ISI", "INITIAL_SPREAD_INDEX"],
        "bui": ["BUI"],
        "fwi": ["FWI", "FIRE_WEATHER_INDEX"],
        "temperature": ["TEMPERATURE", "HOURLY_TEMPERATURE", "TEMP"],
        "relative_humidity": ["RELATIVE_HUMIDITY", "HOURLY_RELATIVE_HUMIDITY", "RH"],
        "wind_speed": ["WIND_SPEED", "HOURLY_WIND_SPEED", "WS"],
        "wind_direction": ["WIND_DIRECTION", "HOURLY_WIND_DIRECTION", "WD"],
    }
    
    # Use median for high-danger days (they're already selected for danger)
    # Use percentile for all-data fallback (to get fire-conducive conditions)
    use_percentile = len(high_danger_df) == len(df) and use_high_danger
    
    for key, col_names in col_mapping.items():
        for col in col_names:
            if col in high_danger_df.columns:
                values = pd.to_numeric(high_danger_df[col], errors="coerce").dropna()
                if len(values) > 0:
                    if use_percentile and key in ["isi", "bui", "fwi", "wind_speed"]:
                        # Use upper percentile for fire-critical variables
                        weather[key] = float(np.percentile(values, percentile))
                    else:
                        weather[key] = float(np.median(values))
                break
    
    logger.info(
        f"Representative weather: ISI={weather['isi']:.1f}, BUI={weather['bui']:.1f}, "
        f"WS={weather['wind_speed']:.1f} km/h, WD={weather['wind_direction']:.0f} deg"
    )
    
    return weather


def save_fire_weather_list(
    fire_weather: FireWeatherList,
    output_path: Path,
) -> None:
    """
    Save fire weather list to CSV.
    
    Parameters
    ----------
    fire_weather : FireWeatherList
        Fire weather data.
    output_path : Path
        Output file path.
    """
    write_csv(fire_weather.records, output_path)
    logger.info(f"Saved fire weather list to {output_path}")


# =============================================================================
# Time-Varying Weather Interpolation
# =============================================================================


@dataclass
class HourlyWeatherInterpolator:
    """
    Interpolates weather values for any datetime from hourly observations.
    
    Supports linear interpolation between hours and handles edge cases
    like missing data and out-of-range times.
    """
    
    hourly_data: pd.DataFrame
    _datetime_index: pd.DatetimeIndex | None = None
    
    def __post_init__(self):
        """Build datetime index for efficient lookup."""
        df = self.hourly_data.copy()
        
        # Build datetime from DATE and HOUR columns
        if "DATE" in df.columns and "HOUR" in df.columns:
            # Convert DATE to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
                df["DATE"] = pd.to_datetime(df["DATE"])
            
            # Create full datetime
            df["DATETIME"] = df["DATE"] + pd.to_timedelta(df["HOUR"], unit="h")
        elif "DATETIME" in df.columns:
            df["DATETIME"] = pd.to_datetime(df["DATETIME"])
        else:
            logger.warning("Cannot build datetime index for interpolation")
            return
        
        # Sort by datetime and set index
        df = df.sort_values("DATETIME").set_index("DATETIME")
        self.hourly_data = df
        self._datetime_index = df.index
    
    def interpolate_at(self, dt: pd.Timestamp | datetime) -> dict[str, float]:
        """
        Interpolate weather values at a specific datetime.
        
        Parameters
        ----------
        dt : Timestamp or datetime
            Target datetime for interpolation.
            
        Returns
        -------
        dict
            Dictionary of interpolated weather values.
        """
        if self._datetime_index is None or len(self._datetime_index) == 0:
            logger.warning("No datetime index available, returning defaults")
            return self._get_defaults()
        
        dt = pd.Timestamp(dt)
        df = self.hourly_data
        
        # Handle out-of-range times
        if dt <= self._datetime_index[0]:
            return self._row_to_dict(df.iloc[0])
        if dt >= self._datetime_index[-1]:
            return self._row_to_dict(df.iloc[-1])
        
        # Find bracketing times
        idx = self._datetime_index.searchsorted(dt)
        t0 = self._datetime_index[idx - 1]
        t1 = self._datetime_index[idx]
        
        # Interpolation weight
        total_seconds = (t1 - t0).total_seconds()
        if total_seconds == 0:
            weight = 0.0
        else:
            weight = (dt - t0).total_seconds() / total_seconds
        
        # Interpolate numeric columns
        row0 = df.iloc[idx - 1]
        row1 = df.iloc[idx]
        
        result = {}
        for col in df.columns:
            if col in ["DATE", "HOUR", "DATETIME", "category"]:
                continue
            try:
                v0 = float(row0[col])
                v1 = float(row1[col])
                
                # Special handling for wind direction (circular interpolation)
                if col in ["WD", "WIND_DIRECTION"]:
                    result[col] = self._interpolate_angle(v0, v1, weight)
                else:
                    result[col] = v0 + weight * (v1 - v0)
            except (ValueError, TypeError):
                pass
        
        return result
    
    def _interpolate_angle(self, a0: float, a1: float, weight: float) -> float:
        """Interpolate angles (for wind direction) handling wraparound."""
        # Convert to radians
        r0 = np.radians(a0)
        r1 = np.radians(a1)
        
        # Use vector averaging
        x = (1 - weight) * np.cos(r0) + weight * np.cos(r1)
        y = (1 - weight) * np.sin(r0) + weight * np.sin(r1)
        
        # Convert back to degrees
        angle = np.degrees(np.arctan2(y, x))
        if angle < 0:
            angle += 360
        return angle
    
    def _row_to_dict(self, row: pd.Series) -> dict[str, float]:
        """Convert a DataFrame row to a weather dictionary."""
        result = {}
        for col in row.index:
            if col in ["DATE", "HOUR", "DATETIME", "category"]:
                continue
            try:
                result[col] = float(row[col])
            except (ValueError, TypeError):
                pass
        return result
    
    def _get_defaults(self) -> dict[str, float]:
        """Return default weather values."""
        return {
            "TEMP": 25.0,
            "TEMPERATURE": 25.0,
            "RH": 40.0,
            "RELATIVE_HUMIDITY": 40.0,
            "WS": 15.0,
            "WIND_SPEED": 15.0,
            "WD": 180.0,
            "WIND_DIRECTION": 180.0,
            "FFMC": 85.0,
            "DMC": 25.0,
            "DC": 200.0,
            "ISI": 5.0,
            "BUI": 50.0,
            "FWI": 15.0,
        }
    
    def get_weather_sequence(
        self,
        start_dt: pd.Timestamp | datetime,
        n_steps: int,
        dt_minutes: float = 1.0,
    ) -> list[dict[str, float]]:
        """
        Get a sequence of weather values for simulation timesteps.
        
        Parameters
        ----------
        start_dt : Timestamp or datetime
            Simulation start time.
        n_steps : int
            Number of timesteps.
        dt_minutes : float
            Minutes per timestep.
            
        Returns
        -------
        list[dict]
            Weather values for each timestep.
        """
        weather_seq = []
        for t in range(n_steps):
            sim_dt = pd.Timestamp(start_dt) + pd.Timedelta(minutes=t * dt_minutes)
            weather_seq.append(self.interpolate_at(sim_dt))
        
        return weather_seq


def create_weather_interpolator(
    hourly_data: pd.DataFrame,
) -> HourlyWeatherInterpolator:
    """
    Create a weather interpolator from hourly observations.
    
    Parameters
    ----------
    hourly_data : DataFrame
        Hourly weather data with DATE, HOUR, and weather columns.
        
    Returns
    -------
    HourlyWeatherInterpolator
        Interpolator object.
    """
    return HourlyWeatherInterpolator(hourly_data=hourly_data)
