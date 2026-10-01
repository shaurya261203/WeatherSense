import joblib
import pandas as pd

from live_weather import get_latest_features


# ============================================================
# LOAD MODELS
# ============================================================

temperature_model = joblib.load(
    "models/temperature_model.joblib"
)

rain_model = joblib.load(
    "models/rain_model.joblib"
)


# ============================================================
# CITY
# ============================================================

CITY = "Bengaluru"


# ============================================================
# GET LIVE WEATHER + FEATURES
# ============================================================

latest, feature_df = get_latest_features(CITY)


# ============================================================
# MODEL FEATURES
# ============================================================

temperature_features = temperature_model.feature_names_in_
rain_features = rain_model.feature_names_in_


X_temperature = pd.DataFrame(
    [latest[temperature_features]],
    columns=temperature_features
)

X_rain = pd.DataFrame(
    [latest[rain_features]],
    columns=rain_features
)


# ============================================================
# PREDICTIONS
# ============================================================

temperature_prediction = temperature_model.predict(
    X_temperature
)[0]


rain_prediction = rain_model.predict(
    X_rain
)[0]


# Probability of rain
rain_probability = (
    rain_model.predict_proba(X_rain)[0][1]
    if hasattr(rain_model, "predict_proba")
    else None
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 60)
print("WEATHERSENSE LIVE ML PREDICTION")
print("=" * 60)

print(f"\nCity: {CITY}")

print("\nLIVE WEATHER")
print("-" * 40)

print(
    f"Temperature : {latest['temperature_2m']:.1f} °C"
)

print(
    f"Humidity    : {latest['relative_humidity_2m']:.0f} %"
)

print(
    f"Wind        : {latest['wind_speed_10m']:.1f} km/h"
)

print(
    f"Rain        : {latest['rain']:.1f} mm"
)

print(
    f"Pressure    : {latest['pressure_msl']:.1f} hPa"
)

print(
    f"Weather code: {int(latest['weather_code'])}"
)

print(
    f"Updated     : {latest['time']}"
)


print("\nAI NEXT-HOUR PREDICTION")
print("-" * 40)

print(
    f"Temperature : {temperature_prediction:.1f} °C"
)

print(
    f"Rain        : {'YES' if rain_prediction == 1 else 'NO'}"
)

if rain_probability is not None:
    print(
        f"Rain probability: {rain_probability * 100:.1f}%"
    )


print("\n" + "=" * 60)