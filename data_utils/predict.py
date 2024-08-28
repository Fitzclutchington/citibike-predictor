import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_path = "/Users/fitz/Documents/citibike-predictor/model.pkl"


def preprocess_input(date: str, lat: float, lon: float, capacity: int):
    # Convert date to datetime and generate features (hour, day_of_week, etc.)
    date = pd.to_datetime(date)
    hour = date.hour
    day_of_week = date.dayofweek
    month = date.month

    # Generate cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(np.array([lat, lon, capacity]).reshape(1, -1))

    # Create the feature array
    features = np.array(
        [
            hour_sin,
            hour_cos,
            day_of_week_sin,
            day_of_week_cos,
            month_sin,
            month_cos,
            lat,
            lon,
            capacity,
        ]
    )

    # Reshape features for prediction
    return features.reshape(1, -1)


def predict_ebikes(date: str, lat: float, lon: float, capacity: int):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    features = preprocess_input(date, lat, lon, capacity)
    print(f"features: {features}")
    prediction = model.predict(features)
    return prediction[0] 
