import requests
import pandas as pd
from pathlib import Path
import time


# ============================================
# WEATHER SENSE - INDIA WEATHER DATA
# ============================================

# Project folders
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# ============================================
# 10 REPRESENTATIVE INDIAN CITIES
# ============================================

cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Patna": (25.5941, 85.1376),
    "Guwahati": (26.1445, 91.7362)
}


# ============================================
# HISTORICAL PERIOD
# ============================================

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"


# ============================================
# WEATHER VARIABLES
# ============================================

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "weather_code"
]


API_URL = "https://archive-api.open-meteo.com/v1/archive"


# Store data from every city
all_data = []


# ============================================
# DOWNLOAD DATA
# ============================================

for city, (latitude, longitude) in cities.items():

    print(f"\nDownloading weather data for {city}...")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=120
        )

        response.raise_for_status()

        weather_data = response.json()

        df = pd.DataFrame(weather_data["hourly"])

        # Add location information
        df["city"] = city
        df["latitude"] = latitude
        df["longitude"] = longitude

        all_data.append(df)

        print(f"✓ {city}: {len(df):,} rows downloaded")

        # Small pause between API requests
        time.sleep(1)

    except Exception as e:

        print(f"✗ Failed for {city}")
        print(f"  Error: {e}")


# ============================================
# COMBINE ALL CITIES
# ============================================

if all_data:

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Convert time column
    final_df["time"] = pd.to_datetime(
        final_df["time"]
    )

    # Sort by city and time
    final_df = final_df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    # Output path
    output_file = DATA_DIR / "india_weather_hourly.csv"

    # Save dataset
    final_df.to_csv(
        output_file,
        index=False
    )

    # ========================================
    # SUMMARY
    # ========================================

    print("\n" + "=" * 60)
    print("WEATHER DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Rows       : {len(final_df):,}")
    print(f"Columns    : {len(final_df.columns)}")
    print(f"Cities     : {final_df['city'].nunique()}")

    print(
        f"Date range : "
        f"{final_df['time'].min()} → "
        f"{final_df['time'].max()}"
    )

    print(f"\nSaved to:")
    print(output_file)

else:

    print("\nNo weather data was downloaded.")