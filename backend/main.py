from pathlib import Path

import joblib
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.live_weather import (
    CITY_COORDS,
    fetch_all_current_weather,
    get_latest_features,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="WeatherSense API",
    description="WeatherSense ML Weather Intelligence Platform",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

TEMPERATURE_MODEL_PATH = (
    MODEL_DIR / "temperature_model.joblib"
)

RAIN_MODEL_PATH = (
    MODEL_DIR / "rain_model.joblib"
)


# ============================================================
# LOAD MODELS
# ============================================================

temperature_model = joblib.load(
    TEMPERATURE_MODEL_PATH
)

rain_model = joblib.load(
    RAIN_MODEL_PATH
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# RISK
# ============================================================

def calculate_risk(
    rain_probability,
    temperature_change,
    wind_speed,
    precipitation,
):
    score = 0

    if rain_probability >= 80:
        score += 45
    elif rain_probability >= 60:
        score += 30
    elif rain_probability >= 35:
        score += 15

    if precipitation >= 10:
        score += 30
    elif precipitation >= 5:
        score += 20
    elif precipitation >= 1:
        score += 10

    if wind_speed >= 50:
        score += 25
    elif wind_speed >= 35:
        score += 15
    elif wind_speed >= 25:
        score += 8

    if abs(temperature_change) >= 5:
        score += 15
    elif abs(temperature_change) >= 3:
        score += 8

    if score >= 80:
        return "SEVERE"

    if score >= 60:
        return "WARNING"

    if score >= 35:
        return "WATCH"

    return "NORMAL"


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "project": "WeatherSense",
        "version": "2.0",
        "status": "online",
        "capabilities": [
            "live weather",
            "ML next-hour prediction",
            "16-day forecast",
            "weather risk intelligence",
            "global city weather",
        ],
    }


# ============================================================
# CITIES
# ============================================================

@app.get("/cities")
def get_cities():

    return {
        "count": len(CITY_COORDS),
        "cities": [
            {
                "city": city,
                "latitude": coordinates[0],
                "longitude": coordinates[1],
            }
            for city, coordinates
            in CITY_COORDS.items()
        ],
    }


# ============================================================
# ALL CITY WEATHER
# ============================================================

@app.get("/cities/weather")
def get_all_city_weather():

    try:

        weather = (
            fetch_all_current_weather()
        )

        return {
            "count": len(weather),
            "source": "Open-Meteo",
            "cities": weather,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# 16-DAY FORECAST
# ============================================================

@app.get("/forecast/{city}")
def get_forecast(city: str):

    matched_city = None

    for configured_city in CITY_COORDS:

        if (
            configured_city.lower()
            == city.lower()
        ):
            matched_city = configured_city
            break

    if matched_city is None:

        raise HTTPException(
            status_code=404,
            detail={
                "error": "City not found",
                "available_cities":
                    list(CITY_COORDS.keys()),
            },
        )

    latitude, longitude = (
        CITY_COORDS[matched_city]
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "daily": ",".join([
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "precipitation_sum",
            "rain_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
        ]),

        "forecast_days": 16,

        "timezone": "auto",

        "temperature_unit": "celsius",

        "wind_speed_unit": "kmh",

        "precipitation_unit": "mm",
    }

    try:

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        daily = data["daily"]

        forecast = []

        for index, date in enumerate(
            daily["time"]
        ):

            forecast.append({

                "date": date,

                "weather_code":
                    daily[
                        "weather_code"
                    ][index],

                "temperature_max":
                    daily[
                        "temperature_2m_max"
                    ][index],

                "temperature_min":
                    daily[
                        "temperature_2m_min"
                    ][index],

                "feels_like_max":
                    daily[
                        "apparent_temperature_max"
                    ][index],

                "feels_like_min":
                    daily[
                        "apparent_temperature_min"
                    ][index],

                "precipitation":
                    daily[
                        "precipitation_sum"
                    ][index],

                "rain":
                    daily[
                        "rain_sum"
                    ][index],

                "rain_probability":
                    daily[
                        "precipitation_probability_max"
                    ][index],

                "wind":
                    daily[
                        "wind_speed_10m_max"
                    ][index],

                "wind_gusts":
                    daily[
                        "wind_gusts_10m_max"
                    ][index],

                "humidity_max":
                    daily[
                        "relative_humidity_2m_max"
                    ][index],

                "humidity_min":
                    daily[
                        "relative_humidity_2m_min"
                    ][index],
            })

        return {

            "city": matched_city,

            "latitude": latitude,

            "longitude": longitude,

            "source": "Open-Meteo",

            "forecast_days": len(
                forecast
            ),

            "forecast": forecast,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed: {error}",
        )


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "project": "WeatherSense",

        "temperature_model": {

            "type":
                "Machine Learning Regression",

            "mae_celsius":
                0.361,

            "r2":
                0.990,
        },

        "rain_model": {

            "type":
                "Machine Learning Classification",

            "accuracy":
                0.913,

            "f1":
                0.768,

            "roc_auc":
                0.953,
        },

        "prediction_horizon":
            "1 hour",

        "historical_context":
            "168 hours",

        "features":
            len(FEATURE_COLUMNS),
    }


# ============================================================
# ML PREDICTION
# ============================================================

@app.get("/predict/{city}")
def predict_weather(city: str):

    matched_city = None

    for configured_city in CITY_COORDS:

        if (
            configured_city.lower()
            == city.lower()
        ):

            matched_city = (
                configured_city
            )

            break

    if matched_city is None:

        raise HTTPException(
            status_code=404,
            detail={
                "error":
                    "City not found",

                "available_cities":
                    list(CITY_COORDS.keys()),
            },
        )

    try:

        live_data = (
            get_latest_features(
                matched_city
            )
        )

        current = live_data[
            "current"
        ]

        X = pd.DataFrame(
            [live_data["features"]],
            columns=FEATURE_COLUMNS,
        )

        predicted_temperature = float(
            temperature_model.predict(X)[0]
        )

        rain_prediction = int(
            rain_model.predict(X)[0]
        )

        if hasattr(
            rain_model,
            "predict_proba",
        ):

            rain_probability = float(
                rain_model
                .predict_proba(X)[0][1]
            ) * 100

        else:

            rain_probability = (
                100.0
                if rain_prediction
                else 0.0
            )

        current_temperature = float(
            current["temperature_2m"]
        )

        temperature_change = (
            predicted_temperature
            - current_temperature
        )

        rain_expected = (
            rain_probability >= 50
        )

        risk_level = calculate_risk(
            rain_probability,
            temperature_change,
            float(
                current[
                    "wind_speed_10m"
                ]
            ),
            float(
                current[
                    "precipitation"
                ]
            ),
        )

        return {

            "city":
                matched_city,

            "timestamp":
                live_data["timestamp"],

            "current": {

                "temperature":
                    current[
                        "temperature_2m"
                    ],

                "humidity":
                    current[
                        "relative_humidity_2m"
                    ],

                "wind_speed":
                    current[
                        "wind_speed_10m"
                    ],

                "rain":
                    current["rain"],

                "precipitation":
                    current[
                        "precipitation"
                    ],

                "pressure":
                    current[
                        "pressure_msl"
                    ],

                "weather_code":
                    current[
                        "weather_code"
                    ],

                "cloud_cover":
                    current[
                        "cloud_cover"
                    ],

                "dew_point":
                    current[
                        "dew_point_2m"
                    ],

                "apparent_temperature":
                    current[
                        "apparent_temperature"
                    ],
            },

            "prediction": {

                "temperature_next_hour":
                    round(
                        predicted_temperature,
                        2,
                    ),

                "temperature_change":
                    round(
                        temperature_change,
                        2,
                    ),

                "rain_probability":
                    round(
                        rain_probability,
                        1,
                    ),

                "rain_expected":
                    rain_expected,

                "risk_level":
                    risk_level,

                "model":
                    "WeatherSense",

                "temperature_mae_celsius":
                    0.361,

                "temperature_r2":
                    0.990,

                "rain_roc_auc":
                    0.953,

                "prediction_horizon":
                    "1 hour",

                "historical_context_hours":
                    168,
            },

            "source": {

                "current_weather":
                    "Open-Meteo",

                "historical_context":
                    "Open-Meteo",

                "ml_model":
                    "WeatherSense",
            },
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )