import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================
# PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "weather_ml_ready.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================
# LOAD DATA
# ============================================

print("Loading ML dataset...")

df = pd.read_csv(DATA_FILE)

df["time"] = pd.to_datetime(df["time"])

print(f"Dataset size: {len(df):,} rows")


# ============================================
# FEATURES
# ============================================

features = [
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
    "pressure_change_24h"
]


# ============================================
# TIME-BASED SPLIT
# ============================================

# 2021-2024 → training
# 2025       → testing

train = df[
    df["time"] < "2025-01-01"
].copy()

test = df[
    df["time"] >= "2025-01-01"
].copy()

print("\nTRAIN / TEST SPLIT")
print("-" * 40)
print(f"Training rows : {len(train):,}")
print(f"Testing rows  : {len(test):,}")


X_train = train[features]
X_test = test[features]


# ============================================
# 1. TEMPERATURE MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAINING TEMPERATURE MODEL")
print("=" * 60)

y_train_temp = train["target_temperature_1h"]
y_test_temp = test["target_temperature_1h"]


temperature_model = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

temperature_model.fit(
    X_train,
    y_train_temp
)

temp_predictions = temperature_model.predict(X_test)


# Metrics
temp_mae = mean_absolute_error(
    y_test_temp,
    temp_predictions
)

temp_rmse = np.sqrt(
    mean_squared_error(
        y_test_temp,
        temp_predictions
    )
)

temp_r2 = r2_score(
    y_test_temp,
    temp_predictions
)


print("\nTemperature Results")
print(f"MAE  : {temp_mae:.3f} °C")
print(f"RMSE : {temp_rmse:.3f} °C")
print(f"R²   : {temp_r2:.3f}")


# ============================================
# SAVE TEMPERATURE MODEL
# ============================================

temperature_model_path = (
    MODEL_DIR / "temperature_model.joblib"
)

joblib.dump(
    temperature_model,
    temperature_model_path
)

print(
    f"\nTemperature model saved:\n"
    f"{temperature_model_path}"
)


# ============================================
# 2. RAIN MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAINING RAIN MODEL")
print("=" * 60)

y_train_rain = train["target_rain_1h"]
y_test_rain = test["target_rain_1h"]


rain_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

rain_model.fit(
    X_train,
    y_train_rain
)


rain_predictions = rain_model.predict(
    X_test
)

rain_probabilities = rain_model.predict_proba(
    X_test
)[:, 1]


# ============================================
# RAIN METRICS
# ============================================

accuracy = accuracy_score(
    y_test_rain,
    rain_predictions
)

precision = precision_score(
    y_test_rain,
    rain_predictions,
    zero_division=0
)

recall = recall_score(
    y_test_rain,
    rain_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_rain,
    rain_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test_rain,
    rain_probabilities
)


print("\nRain Prediction Results")

print(f"Accuracy  : {accuracy:.3f}")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1 Score  : {f1:.3f}")
print(f"ROC-AUC   : {roc_auc:.3f}")


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test_rain,
        rain_predictions
    )
)


# ============================================
# SAVE RAIN MODEL
# ============================================

rain_model_path = (
    MODEL_DIR / "rain_model.joblib"
)

joblib.dump(
    rain_model,
    rain_model_path
)

print(
    f"\nRain model saved:\n"
    f"{rain_model_path}"
)


# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nModels created:")

print("✓ Temperature model")
print("✓ Rain prediction model")

print("\nSaved in:")

print(MODEL_DIR)