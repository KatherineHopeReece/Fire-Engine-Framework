"""
Canadian Fire Weather Index (FWI) System calculations.

This module implements the Canadian Forest Fire Weather Index System
for computing fire danger indices from weather observations.

References
----------
- Van Wagner, C.E. (1987). Development and Structure of the Canadian
  Forest Fire Weather Index System. Forestry Technical Report 35.
- Van Wagner, C.E. & Pickett, T.L. (1985). Equations and FORTRAN program
  for the Canadian Forest Fire Weather Index System. Forestry Technical
  Report 33.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# FWI Component Data Class
# =============================================================================


@dataclass
class FWIComponents:
    """Container for FWI System components."""
    
    ffmc: float  # Fine Fuel Moisture Code (0-101)
    dmc: float   # Duff Moisture Code (0+)
    dc: float    # Drought Code (0+)
    isi: float   # Initial Spread Index (0+)
    bui: float   # Buildup Index (0+)
    fwi: float   # Fire Weather Index (0+)
    dsr: float   # Daily Severity Rating (0+)


# =============================================================================
# Fine Fuel Moisture Code (FFMC)
# =============================================================================


def calculate_ffmc(
    temperature: float,
    relative_humidity: float,
    wind_speed: float,
    precipitation: float,
    ffmc_prev: float = 85.0,
) -> float:
    """
    Calculate Fine Fuel Moisture Code (FFMC).
    
    FFMC represents the moisture content of litter and other cured
    fine fuels on the forest floor. It indicates the ease of ignition
    and flammability of fine fuels.
    
    Parameters
    ----------
    temperature : float
        Noon temperature (degrees Celsius).
    relative_humidity : float
        Noon relative humidity (percent).
    wind_speed : float
        Noon wind speed (km/h).
    precipitation : float
        24-hour precipitation (mm).
    ffmc_prev : float
        Previous day's FFMC (default 85.0 for startup).
        
    Returns
    -------
    float
        Fine Fuel Moisture Code (0-101 scale).
    """
    # Ensure valid ranges
    temperature = float(temperature)
    relative_humidity = np.clip(float(relative_humidity), 0, 100)
    wind_speed = max(0.0, float(wind_speed))
    precipitation = max(0.0, float(precipitation))
    ffmc_prev = np.clip(float(ffmc_prev), 0, 101)
    
    # Convert FFMC to moisture content (percent)
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    
    # Apply rainfall effect
    if precipitation > 0.5:
        rf = precipitation - 0.5
        
        if mo <= 150.0:
            mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        else:
            mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf)) + \
                 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf)
        
        if mr > 250.0:
            mr = 250.0
        mo = mr
    
    # Equilibrium Moisture Content (EMC)
    # Drying phase
    ed = 0.942 * relative_humidity ** 0.679 + \
         11.0 * np.exp((relative_humidity - 100.0) / 10.0) + \
         0.18 * (21.1 - temperature) * (1.0 - np.exp(-0.115 * relative_humidity))
    
    # Wetting phase  
    ew = 0.618 * relative_humidity ** 0.753 + \
         10.0 * np.exp((relative_humidity - 100.0) / 10.0) + \
         0.18 * (21.1 - temperature) * (1.0 - np.exp(-0.115 * relative_humidity))
    
    # Log drying/wetting rate
    if mo > ed:
        # Drying
        ko = 0.424 * (1.0 - (relative_humidity / 100.0) ** 1.7) + \
             0.0694 * np.sqrt(wind_speed) * (1.0 - (relative_humidity / 100.0) ** 8)
        kd = ko * 0.581 * np.exp(0.0365 * temperature)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    elif mo < ew:
        # Wetting
        k1 = 0.424 * (1.0 - ((100.0 - relative_humidity) / 100.0) ** 1.7) + \
             0.0694 * np.sqrt(wind_speed) * (1.0 - ((100.0 - relative_humidity) / 100.0) ** 8)
        kw = k1 * 0.581 * np.exp(0.0365 * temperature)
        m = ew - (ew - mo) * 10.0 ** (-kw)
    else:
        m = mo
    
    # Convert moisture content back to FFMC
    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    ffmc = np.clip(ffmc, 0.0, 101.0)
    
    return float(ffmc)


# =============================================================================
# Duff Moisture Code (DMC)
# =============================================================================


def calculate_dmc(
    temperature: float,
    relative_humidity: float,
    precipitation: float,
    dmc_prev: float = 6.0,
    month: int = 7,
    latitude: float = 51.0,
) -> float:
    """
    Calculate Duff Moisture Code (DMC).
    
    DMC represents the moisture content of loosely compacted organic
    layers of moderate depth. It gives an indication of fuel consumption
    in moderate duff layers and medium-sized woody material.
    
    Parameters
    ----------
    temperature : float
        Noon temperature (degrees Celsius).
    relative_humidity : float
        Noon relative humidity (percent).
    precipitation : float
        24-hour precipitation (mm).
    dmc_prev : float
        Previous day's DMC (default 6.0 for startup).
    month : int
        Month of year (1-12).
    latitude : float
        Latitude in degrees (for day length adjustment).
        
    Returns
    -------
    float
        Duff Moisture Code.
    """
    temperature = float(temperature)
    relative_humidity = np.clip(float(relative_humidity), 0, 100)
    precipitation = max(0.0, float(precipitation))
    dmc_prev = max(0.0, float(dmc_prev))
    
    # Day length factors by month (for latitude ~46°N)
    day_length_factors = {
        1: 6.5, 2: 7.5, 3: 9.0, 4: 12.8, 5: 13.9, 6: 13.9,
        7: 12.4, 8: 10.9, 9: 9.4, 10: 8.0, 11: 7.0, 12: 6.0
    }
    le = day_length_factors.get(month, 9.0)
    
    # Adjust for latitude if significantly different from 46°N
    if abs(latitude - 46.0) > 10:
        # Simple linear adjustment
        le = le + (latitude - 46.0) * 0.1
        le = np.clip(le, 1.0, 20.0)
    
    # Rain effect
    if precipitation > 1.5:
        re = 0.92 * precipitation - 1.27
        
        mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
        
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * np.log(dmc_prev)
        else:
            b = 6.2 * np.log(dmc_prev) - 17.2
        
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * np.log(mr - 20.0)
        
        if pr < 0:
            pr = 0.0
        dmc_prev = pr
    
    # Temperature effect (only if T > -1.1)
    if temperature > -1.1:
        k = 1.894 * (temperature + 1.1) * (100.0 - relative_humidity) * le * 1e-6
    else:
        k = 0.0
    
    dmc = dmc_prev + 100.0 * k
    
    return float(max(0.0, dmc))


# =============================================================================
# Drought Code (DC)
# =============================================================================


def calculate_dc(
    temperature: float,
    precipitation: float,
    dc_prev: float = 15.0,
    month: int = 7,
    latitude: float = 51.0,
) -> float:
    """
    Calculate Drought Code (DC).
    
    DC represents the moisture content of deep, compact organic layers.
    It is a useful indicator of seasonal drought effects on forest fuels
    and the amount of smoldering in deep duff layers and large logs.
    
    Parameters
    ----------
    temperature : float
        Noon temperature (degrees Celsius).
    precipitation : float
        24-hour precipitation (mm).
    dc_prev : float
        Previous day's DC (default 15.0 for startup).
    month : int
        Month of year (1-12).
    latitude : float
        Latitude in degrees.
        
    Returns
    -------
    float
        Drought Code.
    """
    temperature = float(temperature)
    precipitation = max(0.0, float(precipitation))
    dc_prev = max(0.0, float(dc_prev))
    
    # Day length factors by month
    fl_factors = {
        1: -1.6, 2: -1.6, 3: -1.6, 4: 0.9, 5: 3.8, 6: 5.8,
        7: 6.4, 8: 5.0, 9: 2.4, 10: 0.4, 11: -1.6, 12: -1.6
    }
    fl = fl_factors.get(month, 1.4)
    
    # Rain effect
    if precipitation > 2.8:
        rd = 0.83 * precipitation - 1.27
        qo = 800.0 * np.exp(-dc_prev / 400.0)
        qr = qo + 3.937 * rd
        dr = 400.0 * np.log(800.0 / qr)
        
        if dr < 0:
            dr = 0.0
        dc_prev = dr
    
    # Temperature effect
    if temperature > -2.8:
        v = 0.36 * (temperature + 2.8) + fl
        if v < 0:
            v = 0.0
    else:
        v = 0.0
    
    dc = dc_prev + 0.5 * v
    
    return float(max(0.0, dc))


# =============================================================================
# Initial Spread Index (ISI)
# =============================================================================


def calculate_isi(ffmc: float, wind_speed: float) -> float:
    """
    Calculate Initial Spread Index (ISI).
    
    ISI combines the effects of wind and fine fuel moisture on
    rate of spread without the influence of variable quantities
    of fuel.
    
    Parameters
    ----------
    ffmc : float
        Fine Fuel Moisture Code.
    wind_speed : float
        Noon wind speed (km/h).
        
    Returns
    -------
    float
        Initial Spread Index.
    """
    ffmc = np.clip(float(ffmc), 0, 101)
    wind_speed = max(0.0, float(wind_speed))
    
    # Moisture content from FFMC
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    
    # Fine fuel moisture function
    ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    
    # Wind function
    fw = np.exp(0.05039 * wind_speed)
    
    # ISI
    isi = 0.208 * ff * fw
    
    return float(max(0.0, isi))


# =============================================================================
# Buildup Index (BUI)
# =============================================================================


def calculate_bui(dmc: float, dc: float) -> float:
    """
    Calculate Buildup Index (BUI).
    
    BUI combines DMC and DC to represent the total amount of fuel
    available for combustion.
    
    Parameters
    ----------
    dmc : float
        Duff Moisture Code.
    dc : float
        Drought Code.
        
    Returns
    -------
    float
        Buildup Index.
    """
    dmc = max(0.0, float(dmc))
    dc = max(0.0, float(dc))
    
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * \
              (0.92 + (0.0114 * dmc) ** 1.7)
    
    return float(max(0.0, bui))


# =============================================================================
# Fire Weather Index (FWI)
# =============================================================================


def calculate_fwi(isi: float, bui: float) -> float:
    """
    Calculate Fire Weather Index (FWI).
    
    FWI combines ISI and BUI to represent fire intensity.
    
    Parameters
    ----------
    isi : float
        Initial Spread Index.
    bui : float
        Buildup Index.
        
    Returns
    -------
    float
        Fire Weather Index.
    """
    isi = max(0.0, float(isi))
    bui = max(0.0, float(bui))
    
    if bui <= 80.0:
        fd = 0.626 * bui ** 0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))
    
    b = 0.1 * isi * fd
    
    if b > 1.0:
        fwi = np.exp(2.72 * (0.434 * np.log(b)) ** 0.647)
    else:
        fwi = b
    
    return float(max(0.0, fwi))


def calculate_dsr(fwi: float) -> float:
    """
    Calculate Daily Severity Rating (DSR).
    
    DSR is a transformation of FWI that is more suitable for
    averaging over time.
    
    Parameters
    ----------
    fwi : float
        Fire Weather Index.
        
    Returns
    -------
    float
        Daily Severity Rating.
    """
    fwi = max(0.0, float(fwi))
    dsr = 0.0272 * fwi ** 1.77
    return float(dsr)


# =============================================================================
# Complete FWI Calculation
# =============================================================================


def calculate_fwi_components(
    temperature: float,
    relative_humidity: float,
    wind_speed: float,
    precipitation: float,
    ffmc_prev: float = 85.0,
    dmc_prev: float = 6.0,
    dc_prev: float = 15.0,
    month: int = 7,
    latitude: float = 51.0,
) -> FWIComponents:
    """
    Calculate all FWI System components for a single day.
    
    Parameters
    ----------
    temperature : float
        Noon temperature (degrees Celsius).
    relative_humidity : float
        Noon relative humidity (percent).
    wind_speed : float
        Noon wind speed (km/h).
    precipitation : float
        24-hour precipitation (mm).
    ffmc_prev : float
        Previous day's FFMC.
    dmc_prev : float
        Previous day's DMC.
    dc_prev : float
        Previous day's DC.
    month : int
        Month of year (1-12).
    latitude : float
        Latitude in degrees.
        
    Returns
    -------
    FWIComponents
        All FWI System components.
    """
    ffmc = calculate_ffmc(temperature, relative_humidity, wind_speed, precipitation, ffmc_prev)
    dmc = calculate_dmc(temperature, relative_humidity, precipitation, dmc_prev, month, latitude)
    dc = calculate_dc(temperature, precipitation, dc_prev, month, latitude)
    isi = calculate_isi(ffmc, wind_speed)
    bui = calculate_bui(dmc, dc)
    fwi = calculate_fwi(isi, bui)
    dsr = calculate_dsr(fwi)
    
    return FWIComponents(
        ffmc=ffmc,
        dmc=dmc,
        dc=dc,
        isi=isi,
        bui=bui,
        fwi=fwi,
        dsr=dsr,
    )


# =============================================================================
# Process Weather DataFrame to FWI
# =============================================================================


def calculate_fwi_from_weather(
    weather_df: pd.DataFrame,
    temp_col: str = "TEMP",
    rh_col: str = "RH",
    ws_col: str = "WS",
    precip_col: str = "PRECIP",
    date_col: str = "DATE",
    hour_col: str = "HOUR",
    latitude: float = 51.0,
    noon_hour: int = 12,
) -> pd.DataFrame:
    """
    Calculate FWI components from hourly weather data.
    
    Extracts noon values and computes daily FWI components with
    proper carry-over of moisture codes.
    
    Parameters
    ----------
    weather_df : pd.DataFrame
        Hourly weather observations.
    temp_col : str
        Temperature column name.
    rh_col : str
        Relative humidity column name.
    ws_col : str
        Wind speed column name.
    precip_col : str
        Precipitation column name.
    date_col : str
        Date column name.
    hour_col : str
        Hour column name.
    latitude : float
        Latitude for day length adjustment.
    noon_hour : int
        Hour to use as "noon" (default 12).
        
    Returns
    -------
    pd.DataFrame
        Daily FWI components with columns:
        DATE, TEMP, RH, WS, PRECIP_24H, FFMC, DMC, DC, ISI, BUI, FWI, DSR
    """
    df = weather_df.copy()
    
    # Standardize column names - handle both short and long naming conventions
    # Note: Prefer DATE column if it exists (already parsed), fall back to HOURLY/DATETIME
    col_map = {}
    has_date_col = any(c.upper() == "DATE" for c in df.columns)
    
    for col in df.columns:
        col_upper = col.upper()
        # Temperature
        if col_upper in [temp_col.upper(), "TEMPERATURE", "HOURLY_TEMPERATURE", "TEMP"]:
            col_map[col] = "TEMP"
        # Relative humidity
        elif col_upper in [rh_col.upper(), "RELATIVE_HUMIDITY", "HOURLY_RELATIVE_HUMIDITY", "RH"]:
            col_map[col] = "RH"
        # Wind speed
        elif col_upper in [ws_col.upper(), "WIND_SPEED", "HOURLY_WIND_SPEED", "WS"]:
            col_map[col] = "WS"
        # Precipitation
        elif col_upper in [precip_col.upper() if precip_col else "", "PRECIPITATION", "PRECIP", "RAIN"]:
            col_map[col] = "PRECIP"
        # Wind direction (not required for FWI but useful for fire spread)
        elif col_upper in ["WIND_DIRECTION", "WD"]:
            col_map[col] = "WD"
        # Date - prefer already-parsed DATE column, skip HOURLY/DATETIME if DATE exists
        elif col_upper == "DATE":
            col_map[col] = "DATE"
        elif col_upper in ["HOURLY", "DATETIME", "DATE_TIME"] and not has_date_col:
            col_map[col] = "DATE"
        # Hour
        elif col_upper in [hour_col.upper(), "HR", "HOUR"]:
            col_map[col] = "HOUR"
    
    df = df.rename(columns=col_map)
    
    logger.debug(f"FWI calculation: mapped columns {list(col_map.keys())} -> {list(col_map.values())}")
    logger.debug(f"After mapping, columns are: {list(df.columns)}")
    
    # Parse date if needed - handle various input formats
    if "DATE" in df.columns:
        # Check if DATE is already datetime-like
        sample = df["DATE"].dropna().iloc[0] if df["DATE"].notna().any() else None
        
        if sample is not None:
            # If it's already a date object, convert to string first for consistent parsing
            if hasattr(sample, 'year') and not hasattr(sample, 'hour'):
                # It's a date object, convert to datetime
                df["DATE"] = pd.to_datetime(df["DATE"])
            elif isinstance(sample, str):
                # Parse string dates
                try:
                    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
                except Exception:
                    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            # else it's already datetime64, no parsing needed
            
            # Extract hour if not present and datetime has hour info
            if "HOUR" not in df.columns:
                try:
                    if hasattr(df["DATE"].iloc[0], 'hour'):
                        df["HOUR"] = df["DATE"].dt.hour
                except Exception:
                    pass
            
            # Convert to just date (no time) - use normalize to avoid issues
            try:
                df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize().dt.date
            except Exception:
                df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    
    # Ensure HOUR is numeric
    if "HOUR" in df.columns:
        df["HOUR"] = pd.to_numeric(df["HOUR"], errors="coerce")
    
    # Make sure DATE is valid before grouping
    df = df.dropna(subset=["DATE"])
    if len(df) == 0:
        logger.warning("No valid dates found in weather data")
        return pd.DataFrame(columns=["DATE", "TEMP", "RH", "WS", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "DSR"])
    
    n_days = len(set(df["DATE"]))
    logger.debug(f"Weather data has {len(df)} records over {n_days} unique days")
    
    # Get noon observations
    if "HOUR" in df.columns:
        unique_hours = sorted(df["HOUR"].dropna().unique())
        logger.debug(f"Hours present in data: {unique_hours}")
        noon_df = df[df["HOUR"] == noon_hour].copy()
    else:
        logger.warning("No HOUR column found, using all records")
        noon_df = df.copy()
    
    if len(noon_df) == 0:
        logger.warning(f"No records at hour {noon_hour}, trying hour 13")
        noon_df = df[df["HOUR"] == 13].copy()
    
    if len(noon_df) == 0:
        logger.warning("No noon records found, using daily means")
        # Aggregate to get one record per day
        agg_dict = {}
        for col in ["TEMP", "RH", "WS"]:
            if col in df.columns:
                agg_dict[col] = "mean"
        if "PRECIP" in df.columns:
            agg_dict["PRECIP"] = "sum"
        if "WD" in df.columns:
            agg_dict["WD"] = "mean"
        
        if agg_dict:
            noon_df = df.groupby("DATE", as_index=False).agg(agg_dict)
        else:
            logger.warning("No aggregatable columns found")
            noon_df = df.drop_duplicates(subset=["DATE"]).copy()
    else:
        # If we have noon values but multiple per day (e.g., multiple stations),
        # take the mean to get one record per day
        agg_dict = {}
        for col in ["TEMP", "RH", "WS"]:
            if col in noon_df.columns:
                agg_dict[col] = "mean"
        if "PRECIP" in noon_df.columns:
            agg_dict["PRECIP"] = "mean"
        if "WD" in noon_df.columns:
            agg_dict["WD"] = "mean"
        
        if agg_dict:
            noon_df = noon_df.groupby("DATE", as_index=False).agg(agg_dict)
        else:
            noon_df = noon_df.drop_duplicates(subset=["DATE"]).copy()
    
    # Calculate 24-hour precipitation (sum for each day)
    if "PRECIP" in df.columns:
        try:
            # Use a dict for mapping to avoid duplicate key issues
            daily_precip = df.groupby("DATE")["PRECIP"].sum().to_dict()
            noon_df["PRECIP_24H"] = noon_df["DATE"].apply(lambda x: daily_precip.get(x, 0.0))
        except Exception as e:
            logger.debug(f"Precip mapping failed: {e}, using zeros")
            noon_df["PRECIP_24H"] = 0.0
    else:
        noon_df["PRECIP_24H"] = 0.0
    
    # Sort by date
    noon_df = noon_df.sort_values("DATE").reset_index(drop=True)
    
    # Initialize previous values
    ffmc_prev = 85.0
    dmc_prev = 6.0
    dc_prev = 15.0
    
    results = []
    
    for idx, row in noon_df.iterrows():
        temp = row.get("TEMP", 20.0) if pd.notna(row.get("TEMP")) else 20.0
        rh = row.get("RH", 50.0) if pd.notna(row.get("RH")) else 50.0
        ws = row.get("WS", 10.0) if pd.notna(row.get("WS")) else 10.0
        precip = row.get("PRECIP_24H", 0.0) if pd.notna(row.get("PRECIP_24H")) else 0.0
        wd = row.get("WD", 180.0) if "WD" in row and pd.notna(row.get("WD")) else 180.0
        
        # Get month for day length adjustment
        date = row.get("DATE")
        if hasattr(date, "month"):
            month = date.month
        else:
            month = 7  # Default to July
        
        # Calculate FWI components
        components = calculate_fwi_components(
            temperature=temp,
            relative_humidity=rh,
            wind_speed=ws,
            precipitation=precip,
            ffmc_prev=ffmc_prev,
            dmc_prev=dmc_prev,
            dc_prev=dc_prev,
            month=month,
            latitude=latitude,
        )
        
        results.append({
            "DATE": date,
            "TEMP": temp,
            "RH": rh,
            "WS": ws,
            "WD": wd,
            "PRECIP_24H": precip,
            "FFMC": components.ffmc,
            "DMC": components.dmc,
            "DC": components.dc,
            "ISI": components.isi,
            "BUI": components.bui,
            "FWI": components.fwi,
            "DSR": components.dsr,
        })
        
        # Update previous values for next iteration
        ffmc_prev = components.ffmc
        dmc_prev = components.dmc
        dc_prev = components.dc
    
    result_df = pd.DataFrame(results)
    
    logger.info(
        f"Calculated FWI for {len(result_df)} days. "
        f"ISI range: {result_df['ISI'].min():.2f} - {result_df['ISI'].max():.2f}"
    )
    
    return result_df
