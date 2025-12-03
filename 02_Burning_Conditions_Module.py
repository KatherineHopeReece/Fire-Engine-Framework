# ==============================================================
# BurningConditions_FireWeatherList.py
# ==============================================================

"""
Generates Fire Weather List for Burning Conditions Module
Filters hourly weather data for high and extreme ISI conditions
and merges with station metadata.
"""

import pandas as pd
from pathlib import Path

# --------------------------------------------------------------
# USER CONFIGURATION
# --------------------------------------------------------------
template_path = Path("/Users/Martyn/Desktop/PhD/Fyah/Python_Scripts/project_template.txt")

STATION_PATH = None
WEATHER_PATH = None

with open(template_path, "r") as file:
    for line in file:
        stripped = line.strip()
        if stripped.startswith("STATION_PATH"):
            STATION_PATH = Path(stripped.split("=", 1)[1].strip())
        elif stripped.startswith("WEATHER_PATH"):
            WEATHER_PATH = Path(stripped.split("=", 1)[1].strip())

OUTPUT_PATH = Path("/Users/Martyn/Desktop/PhD/Fyah/Kelowna_FireModel/Weather/fire_weather_list.csv")

print(f"Station path: {STATION_PATH}")
print(f"Weather path: {WEATHER_PATH}")

ISI_THRESHOLDS = {"high": 8.6, "extreme": 12.6}

# --------------------------------------------------------------
# STEP 1: LOAD DATA
# --------------------------------------------------------------
print("Loading station and weather data...")

stations = pd.read_csv(STATION_PATH)
weather = pd.read_csv(WEATHER_PATH)

# Clean up column names
weather.columns = weather.columns.str.strip().str.upper()
stations.columns = stations.columns.str.strip().str.upper()

# Merge on station code (ensuring both are strings)
stations["STATION_CODE"] = stations["STATION_CODE"].astype(str)
weather["STATION_CODE"] = weather["STATION_CODE"].astype(str)

# --------------------------------------------------------------
# STEP 2: EXTRACT DAILY / NOON RECORDS
# --------------------------------------------------------------
# Parse datetime and extract hour
try:
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], format="%Y%m%d%H")
except Exception:
    # Fallback: try flexible ISO 8601 parsing
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], infer_datetime_format=True, errors="coerce")
weather["HOUR"] = weather["DATE_TIME"].dt.hour
weather["DATE"] = weather["DATE_TIME"].dt.date

# Option 1: Use 12:00 observations if available
noon_weather = weather[weather["HOUR"] == 12].copy()

# Option 2 (fallback): Use daily max ISI if noon not present
if noon_weather.empty:
    noon_weather = (
        weather.groupby(["STATION_CODE", "DATE"])
        .agg({"INITIAL_SPREAD_INDEX": "max",
              "FIRE_WEATHER_INDEX": "max",
              "HOURLY_TEMPERATURE": "mean",
              "HOURLY_RELATIVE_HUMIDITY": "mean",
              "HOURLY_WIND_SPEED": "mean",
              "HOURLY_WIND_DIRECTION": "mean"})
        .reset_index()
    )

# --------------------------------------------------------------
# STEP 3: FILTER BY ISI THRESHOLDS
# --------------------------------------------------------------
noon_weather["CATEGORY"] = "moderate"
noon_weather.loc[
    (noon_weather["INITIAL_SPREAD_INDEX"] >= ISI_THRESHOLDS["high"]) &
    (noon_weather["INITIAL_SPREAD_INDEX"] < ISI_THRESHOLDS["extreme"]),
    "CATEGORY"
] = "high"
noon_weather.loc[
    noon_weather["INITIAL_SPREAD_INDEX"] >= ISI_THRESHOLDS["extreme"],
    "CATEGORY"
] = "extreme"

filtered = noon_weather[noon_weather["CATEGORY"].isin(["high", "extreme"])].copy()

print(f"Selected {len(filtered)} high/extreme fire weather records out of {len(noon_weather)} total.")

# --------------------------------------------------------------
# STEP 4: MERGE WITH STATION METADATA
# --------------------------------------------------------------
fire_weather_list = filtered.merge(
    stations[["STATION_CODE", "STATION_NAME", "LATITUDE", "LONGITUDE", "ELEVATION_M"]],
    on="STATION_CODE",
    how="left"
)

# --------------------------------------------------------------
# STEP 5: SAVE OUTPUT
# --------------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fire_weather_list.to_csv(OUTPUT_PATH, index=False)

print("✅ Fire Weather List saved to:")
print(OUTPUT_PATH)

# Optional summary
summary = fire_weather_list["CATEGORY"].value_counts(normalize=True) * 100
print("\nProportion of categories (%):")
print(summary.round(1))

# --------------------------------------------------------------
# DIAGNOSTIC SECTION
# --------------------------------------------------------------
import matplotlib.pyplot as plt

print("\nDataset Info:")
print(fire_weather_list.info())

print("\nDataset Head:")
print(fire_weather_list.head())

print("\nCategory Counts:")
print(fire_weather_list["CATEGORY"].value_counts())

print(f"\nNumber of Unique Stations: {fire_weather_list['STATION_CODE'].nunique()}")

# Bar plot of category counts
counts = fire_weather_list["CATEGORY"].value_counts()
plt.figure(figsize=(8,5))
counts.plot(kind='bar')
plt.title("High vs Extreme Fire Weather Records")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show(block=False)

# --------------------------------------------------------------
# STEP 6: DISTRIBUTION OF SPREAD EVENT DAYS
# --------------------------------------------------------------
import numpy as np

fw_list = pd.read_csv(OUTPUT_PATH)

LAMBDA = 3.76

spread_days = np.random.poisson(LAMBDA, size=len(fw_list))
spread_days_clipped = np.clip(spread_days, a_min=1, a_max=None)

fw_list['SPREAD_EVENT_DAYS'] = spread_days_clipped

spread_output_path = OUTPUT_PATH.parent / "spread_event_days.csv"
fw_list.to_csv(spread_output_path, index=False)

print("\nSpread Event Days Summary Statistics:")
print(f"Min: {fw_list['SPREAD_EVENT_DAYS'].min()}")
print(f"Max: {fw_list['SPREAD_EVENT_DAYS'].max()}")
print(f"Mean: {fw_list['SPREAD_EVENT_DAYS'].mean():.2f}")

plt.figure(figsize=(8,5))
plt.hist(fw_list['SPREAD_EVENT_DAYS'], bins=range(1, fw_list['SPREAD_EVENT_DAYS'].max()+2), align='left', edgecolor='black')
plt.title("Distribution of Spread Event Days")
plt.xlabel("Number of Spread Event Days")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show(block=False)