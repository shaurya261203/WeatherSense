import pandas as pd
from pathlib import Path


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "india_weather_hourly.csv"
OUTPUT_FILE = BASE_DIR / "data" / "weather_ml_ready.csv"


# ============================================
# LOAD DATA
# ============================================

print("Loading weather data...")

df = pd.read_csv(INPUT_FILE)

df["time"] = pd.to_datetime(df["time"])

# Make sure each city's data is in chronological order
df = df.sort_values(
    ["city", "time"]
).reset_index(drop=True)

print(f"Original rows: {len(df):,}")


# ============================================
# TIME FEATURES
# ============================================

print("Creating time features...")

df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.dayofweek
df["month"] = df["time"].dt.month
df["day_of_year"] = df["time"].dt.dayofyear

# Cyclic time features
df["hour_sin"] = __import__("numpy").sin(
    2 * __import__("numpy").pi * df["hour"] / 24
)

df["hour_cos"] = __import__("numpy").cos(
    2 * __import__("numpy").pi * df["hour"] / 24
)

df["month_sin"] = __import__("numpy").sin(
    2 * __import__("numpy").pi * df["month"] / 12
)

df["month_cos"] = __import__("numpy").cos(
    2 * __import__("numpy").pi * df["month"] / 12
)


# ============================================
# LAG FEATURES
# ============================================

print("Creating lag features...")

group = df.groupby("city", group_keys=False)

# Temperature history
for lag in [1, 3, 6, 24, 168]:
    df[f"temperature_lag_{lag}h"] = group[
        "temperature_2m"
    ].shift(lag)

# Humidity history
for lag in [1, 6, 24]:
    df[f"humidity_lag_{lag}h"] = group[
        "relative_humidity_2m"
    ].shift(lag)

# Pressure history
for lag in [1, 3, 6, 24]:
    df[f"pressure_lag_{lag}h"] = group[
        "pressure_msl"
    ].shift(lag)


# ============================================
# ROLLING WEATHER FEATURES
# ============================================

print("Creating rolling features...")

# IMPORTANT:
# shift(1) means we only use PAST data.
# This prevents future-data leakage.

df["temperature_3h_mean"] = group[
    "temperature_2m"
].transform(
    lambda x: x.shift(1).rolling(3).mean()
)

df["temperature_6h_mean"] = group[
    "temperature_2m"
].transform(
    lambda x: x.shift(1).rolling(6).mean()
)

df["temperature_24h_mean"] = group[
    "temperature_2m"
].transform(
    lambda x: x.shift(1).rolling(24).mean()
)

df["humidity_6h_mean"] = group[
    "relative_humidity_2m"
].transform(
    lambda x: x.shift(1).rolling(6).mean()
)

df["pressure_6h_mean"] = group[
    "pressure_msl"
].transform(
    lambda x: x.shift(1).rolling(6).mean()
)

df["rain_24h_sum"] = group[
    "rain"
].transform(
    lambda x: x.shift(1).rolling(24).sum()
)


# ============================================
# PRESSURE CHANGE
# ============================================

df["pressure_change_3h"] = (
    df["pressure_msl"]
    - df["pressure_lag_3h"]
)

df["pressure_change_24h"] = (
    df["pressure_msl"]
    - df["pressure_lag_24h"]
)


# ============================================
# TARGET VARIABLES
# ============================================

print("Creating prediction targets...")

# NEXT-HOUR TEMPERATURE
df["target_temperature_1h"] = group[
    "temperature_2m"
].shift(-1)

# NEXT-HOUR RAIN
df["target_rain_1h"] = (
    group["rain"].shift(-1) > 0
).astype(int)


# ============================================
# REMOVE INVALID ROWS
# ============================================

print("Cleaning dataset...")

df = df.dropna().reset_index(drop=True)


# ============================================
# SAVE
# ============================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("ML DATASET CREATED")
print("=" * 60)

print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")
print(f"Cities     : {df['city'].nunique()}")

print("\nCities:")
print(df["city"].value_counts())

print("\nTarget temperature:")
print(df["target_temperature_1h"].describe())

print("\nRain target:")
print(df["target_rain_1h"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)