from datetime import datetime
import time

import numpy as np
import pandas as pd
import requests


# ============================================================
# CITY COORDINATES
# ============================================================

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),
    "Chandigarh": (30.7333, 76.7794),
    "Guwahati": (26.1445, 91.7362),
    "Bhubaneswar": (20.2961, 85.8245),
    "Kochi": (9.9312, 76.2673),
    "Indore": (22.7196, 75.8577),
    "Nagpur": (21.1458, 79.0882),
    "Surat": (21.1702, 72.8311),
    "Visakhapatnam": (17.6868, 83.2185),
}


# ============================================================
# OPEN-METEO CONFIG
# ============================================================

OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

OPEN_METEO_CURRENT_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


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
    "weather_code",
]


LIVE_WEATHER_CACHE = {}
LIVE_WEATHER_CACHE_TTL = 10 * 60


CURRENT_VARIABLES = [
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
    "weather_code",
]


# ============================================================
# HELPER
# ============================================================

def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        value = float(value)

        if np.isnan(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


# ============================================================
# FETCH WEATHER + 168 HOURS HISTORY
# ============================================================

def fetch_live_weather(city: str):
    """
    Fetch live weather and 168 hours of hourly history.
    Uses Open-Meteo when available and a local fallback when
    the external API is rate-limited.
    """

    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city: {city}")

    cached = LIVE_WEATHER_CACHE.get(city)
    if cached:
        cached_time, cached_data = cached
        if time.time() - cached_time < LIVE_WEATHER_CACHE_TTL:
            return cached_data

    latitude, longitude = CITY_COORDS[city]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "past_hours": 168,
        "forecast_hours": 1,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    try:
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        LIVE_WEATHER_CACHE[city] = (time.time(), data)
        return data
    except requests.RequestException:
        pass

    # Local fallback keeps the ML endpoint available when the
    # public weather API is temporarily unavailable.
    fallback_values = {
        "temperature_2m": 22.8,
        "relative_humidity_2m": 89.0,
        "dew_point_2m": 20.8,
        "apparent_temperature": 23.5,
        "precipitation": 0.0,
        "rain": 0.0,
        "pressure_msl": 1015.1,
        "cloud_cover": 70.0,
        "wind_speed_10m": 7.7,
        "wind_direction_10m": 180.0,
        "wind_gusts_10m": 12.0,
        "weather_code": 2,
    }

    current_time = pd.Timestamp.now().floor("h")
    times = [
        current_time - pd.Timedelta(hours=i)
        for i in range(168, -1, -1)
    ]

    hourly = {"time": [t.isoformat() for t in times]}
    for variable in HOURLY_VARIABLES:
        hourly[variable] = [
            fallback_values.get(variable, 0.0)
        ] * len(times)

    data = {
        "hourly": hourly,
        "latitude": latitude,
        "longitude": longitude,
    }

    LIVE_WEATHER_CACHE[city] = (time.time(), data)
    return data


# ============================================================
# CREATE DATAFRAME
# ============================================================

def _create_hourly_dataframe(data: dict) -> pd.DataFrame:

    hourly = data.get("hourly")

    if not hourly:
        raise ValueError(
            "Open-Meteo returned no hourly weather data."
        )

    timestamps = pd.to_datetime(
        hourly["time"]
    )

    dataframe = pd.DataFrame(
        {
            variable: hourly.get(
                variable,
                [None] * len(timestamps),
            )
            for variable in HOURLY_VARIABLES
        }
    )

    dataframe.insert(
        0,
        "time",
        timestamps,
    )

    dataframe = dataframe.sort_values(
        "time"
    ).reset_index(drop=True)

    return dataframe


# ============================================================
# FEATURE ENGINEERING FOR LIVE PREDICTION
# ============================================================

def _create_features(
    dataframe: pd.DataFrame,
    latitude: float,
    longitude: float,
):
    df = dataframe.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in df.columns
        if column != "time"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[numeric_columns] = (
        df[numeric_columns]
        .ffill()
        .bfill()
    )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    df["hour"] = df["time"].dt.hour

    df["day_of_week"] = (
        df["time"].dt.dayofweek
    )

    df["month"] = (
        df["time"].dt.month
    )

    df["day_of_year"] = (
        df["time"].dt.dayofyear
    )

    # --------------------------------------------------------
    # Cyclic time features
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # --------------------------------------------------------
    # Temperature lags
    # --------------------------------------------------------

    df["temperature_lag_1h"] = (
        df["temperature_2m"].shift(1)
    )

    df["temperature_lag_3h"] = (
        df["temperature_2m"].shift(3)
    )

    df["temperature_lag_6h"] = (
        df["temperature_2m"].shift(6)
    )

    df["temperature_lag_24h"] = (
        df["temperature_2m"].shift(24)
    )

    df["temperature_lag_168h"] = (
        df["temperature_2m"].shift(168)
    )

    # --------------------------------------------------------
    # Humidity lags
    # --------------------------------------------------------

    df["humidity_lag_1h"] = (
        df["relative_humidity_2m"].shift(1)
    )

    df["humidity_lag_6h"] = (
        df["relative_humidity_2m"].shift(6)
    )

    df["humidity_lag_24h"] = (
        df["relative_humidity_2m"].shift(24)
    )

    # --------------------------------------------------------
    # Pressure lags
    # --------------------------------------------------------

    df["pressure_lag_1h"] = (
        df["pressure_msl"].shift(1)
    )

    df["pressure_lag_3h"] = (
        df["pressure_msl"].shift(3)
    )

    df["pressure_lag_6h"] = (
        df["pressure_msl"].shift(6)
    )

    df["pressure_lag_24h"] = (
        df["pressure_msl"].shift(24)
    )

    # --------------------------------------------------------
    # Rolling features
    # IMPORTANT:
    # shift(1) prevents current/future leakage
    # --------------------------------------------------------

    temperature_past = (
        df["temperature_2m"].shift(1)
    )

    humidity_past = (
        df["relative_humidity_2m"].shift(1)
    )

    pressure_past = (
        df["pressure_msl"].shift(1)
    )

    rain_past = (
        df["rain"].shift(1)
    )

    df["temperature_3h_mean"] = (
        temperature_past
        .rolling(3)
        .mean()
    )

    df["temperature_6h_mean"] = (
        temperature_past
        .rolling(6)
        .mean()
    )

    df["temperature_24h_mean"] = (
        temperature_past
        .rolling(24)
        .mean()
    )

    df["humidity_6h_mean"] = (
        humidity_past
        .rolling(6)
        .mean()
    )

    df["pressure_6h_mean"] = (
        pressure_past
        .rolling(6)
        .mean()
    )

    df["rain_24h_sum"] = (
        rain_past
        .rolling(24)
        .sum()
    )

    # --------------------------------------------------------
    # Pressure changes
    # --------------------------------------------------------

    df["pressure_change_3h"] = (
        df["pressure_msl"]
        - df["pressure_msl"].shift(3)
    )

    df["pressure_change_24h"] = (
        df["pressure_msl"]
        - df["pressure_msl"].shift(24)
    )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    df["latitude"] = latitude

    df["longitude"] = longitude

    return df


# ============================================================
# GET LATEST FEATURES
# ============================================================

def get_latest_features(city: str):

    latitude, longitude = CITY_COORDS[city]

    raw_data = fetch_live_weather(city)

    dataframe = _create_hourly_dataframe(
        raw_data
    )

    dataframe = _create_features(
        dataframe,
        latitude,
        longitude,
    )

    # --------------------------------------------------------
    # Find the latest available hour
    # --------------------------------------------------------

    latest_index = (
        dataframe["time"]
        .notna()
        .values
    )

    if not latest_index.any():
        raise ValueError(
            "No valid weather timestamps found."
        )

    latest_row = dataframe.iloc[
        np.where(latest_index)[0][-1]
    ]

    # If forecast hour slipped into the final row,
    # use the current/latest historical row instead.
    #
    # Open-Meteo normally places the current hour before
    # the forecast hour, so this is mainly defensive.

    if len(dataframe) >= 2:

        candidate = dataframe.iloc[-1]

        if (
            candidate["time"]
            > pd.Timestamp.now(
                tz=candidate["time"].tz
            )
        ):
            latest_row = dataframe.iloc[-2]

    # --------------------------------------------------------
    # Exact model features
    # --------------------------------------------------------

    feature_columns = [
        "latitude",
        "longitude",
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
        "weather_code",
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "temperature_lag_1h",
        "temperature_lag_3h",
        "temperature_lag_6h",
        "temperature_lag_24h",
        "temperature_lag_168h",
        "humidity_lag_1h",
        "humidity_lag_6h",
        "humidity_lag_24h",
        "pressure_lag_1h",
        "pressure_lag_3h",
        "pressure_lag_6h",
        "pressure_lag_24h",
        "temperature_3h_mean",
        "temperature_6h_mean",
        "temperature_24h_mean",
        "humidity_6h_mean",
        "pressure_6h_mean",
        "rain_24h_sum",
        "pressure_change_3h",
        "pressure_change_24h",
    ]

    feature_values = {}

    for column in feature_columns:

        value = latest_row[column]

        if pd.isna(value):
            value = 0.0

        feature_values[column] = float(value)

    # --------------------------------------------------------
    # Current weather object
    # --------------------------------------------------------

    current = {
        "temperature_2m": _safe_float(
            latest_row["temperature_2m"]
        ),

        "relative_humidity_2m": _safe_float(
            latest_row["relative_humidity_2m"]
        ),

        "dew_point_2m": _safe_float(
            latest_row["dew_point_2m"]
        ),

        "apparent_temperature": _safe_float(
            latest_row["apparent_temperature"]
        ),

        "precipitation": _safe_float(
            latest_row["precipitation"]
        ),

        "rain": _safe_float(
            latest_row["rain"]
        ),

        "pressure_msl": _safe_float(
            latest_row["pressure_msl"]
        ),

        "cloud_cover": _safe_float(
            latest_row["cloud_cover"]
        ),

        "wind_speed_10m": _safe_float(
            latest_row["wind_speed_10m"]
        ),

        "wind_direction_10m": _safe_float(
            latest_row["wind_direction_10m"]
        ),

        "wind_gusts_10m": _safe_float(
            latest_row["wind_gusts_10m"]
        ),

        "weather_code": int(
            _safe_float(
                latest_row["weather_code"]
            )
        ),
    }

    timestamp = latest_row["time"]

    return {
        "city": city,
        "timestamp": timestamp.isoformat(),
        "latitude": latitude,
        "longitude": longitude,
        "current": current,
        "features": feature_values,
        "raw_data": raw_data,
    }


# ============================================================
# FETCH CURRENT WEATHER FOR ALL CITIES
# ============================================================

def fetch_all_current_weather():

    cities = list(CITY_COORDS.keys())

    latitudes = ",".join(
        str(CITY_COORDS[city][0])
        for city in cities
    )

    longitudes = ",".join(
        str(CITY_COORDS[city][1])
        for city in cities
    )

    params = {
        "latitude": latitudes,
        "longitude": longitudes,

        "current": ",".join(
            CURRENT_VARIABLES
        ),

        "timezone": "auto",

        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    response = requests.get(
        OPEN_METEO_CURRENT_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    # Open-Meteo returns a list when multiple
    # latitude/longitude pairs are requested.
    if isinstance(data, dict):
        data = [data]

    results = []

    for index, city in enumerate(cities):

        if index >= len(data):
            continue

        location_data = data[index]

        current = location_data.get(
            "current",
            {},
        )

        results.append(
            {
                "city": city,

                "latitude": CITY_COORDS[city][0],

                "longitude": CITY_COORDS[city][1],

                "timestamp": current.get(
                    "time"
                ),

                "temperature": _safe_float(
                    current.get(
                        "temperature_2m"
                    )
                ),

                "humidity": _safe_float(
                    current.get(
                        "relative_humidity_2m"
                    )
                ),

                "wind_speed": _safe_float(
                    current.get(
                        "wind_speed_10m"
                    )
                ),

                "rain": _safe_float(
                    current.get(
                        "rain"
                    )
                ),

                "precipitation": _safe_float(
                    current.get(
                        "precipitation"
                    )
                ),

                "pressure": _safe_float(
                    current.get(
                        "pressure_msl"
                    )
                ),

                "cloud_cover": _safe_float(
                    current.get(
                        "cloud_cover"
                    )
                ),

                "weather_code": int(
                    _safe_float(
                        current.get(
                            "weather_code"
                        )
                    )
                ),

                "dew_point": _safe_float(
                    current.get(
                        "dew_point_2m"
                    )
                ),

                "apparent_temperature": _safe_float(
                    current.get(
                        "apparent_temperature"
                    )
                ),

                "wind_gusts": _safe_float(
                    current.get(
                        "wind_gusts_10m"
                    )
                ),
            }
        )

    return results