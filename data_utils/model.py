import glob
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "lat_norm",
    "lon_norm",
    "capacity_norm",
]

PRED_COLS = ["num_ebikes_available"]

D_TYPES = {
    "station_id": str,
    # Use Pandas Int16 dtype to allow for nullable integers
    "num_bikes_available": "float64",
    "num_ebikes_available": "float64",
    "num_bikes_disabled": "float64",
    "num_docks_available": "float64",
    "num_docks_disabled": "float64",
    "is_installed": "Int16",
    "is_renting": "Int16",
    "is_returning": "Int16",
    "station_status_last_reported": "Int64",
    "station_name": str,
    "lat": float,
    "lon": float,
    "region_id": str,
    "capacity": "Int16",
}


# TODO: this is slightly different from the read_csv function in preprocess
# This should be standardized
def read_csv(filename: str) -> pd.DataFrame:
    """
    Read DataFrame from a CSV file ``filename`` and convert to a
    preferred schema.
    """
    df = pd.read_csv(
        filename,
        sep=",",
        na_values="\\N",
        dtype=D_TYPES,
    )
    # Read in timestamps as UNIX/POSIX epochs but then convert to the local
    # bike share timezone.
    df["station_status_last_reported"] = pd.to_datetime(
        df["station_status_last_reported"], unit="s", origin="unix", utc=True
    ).dt.tz_convert("US/Eastern")

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:

    df["hour"] = df["station_status_last_reported"].dt.hour
    df["day_of_week"] = df["station_status_last_reported"].dt.dayofweek
    df["month"] = df["station_status_last_reported"].dt.month

    # Transforming hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Transforming day_of_week
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Transforming month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Normalize lat, lon and capacity
    scaler = StandardScaler()
    df[["lat_norm", "lon_norm", "capacity_norm"]] = scaler.fit_transform(
        df[["lat", "lon", "capacity"]]
    )
    return df[FEATURE_COLS + PRED_COLS]


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = None
) -> dict[str, pd.DataFrame | pd.Series]:
    X = df[FEATURE_COLS]
    y = df[PRED_COLS[0]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# TODO: make model type an input
def train(X_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(
    X_test: pd.DataFrame, y_test: pd.DataFrame, model: RandomForestRegressor
) -> dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def save_model(model: RandomForestRegressor, output_file: Path | str):
    with open(output_file, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    # TODO: make this a user input
    data_path = "/Users/fitz/Documents/citibike-predictor/data/1h_resample"
    data_files = glob.glob(f"{data_path}/citi_bike_data_*.csv")

    print("loading data")
    t0 = time.perf_counter()
    dfs = []
    for data_file in sorted(data_files):
        tmp_df = read_csv(data_file)
        dfs.append(tmp_df)
        print(f"loaded file {data_file}")
    df = pd.concat(dfs)

    t1 = time.perf_counter()
    print(f"data loaded, time: {t1 - t0}")

    print("generating features")
    t0 = time.perf_counter()
    df = generate_features(df)
    t1 = time.perf_counter()
    print(f"generated features, time: {t1 - t0}")

    print("splitting data")
    t0 = time.perf_counter()
    model_data = split_data(df, seed=42)
    t1 = time.perf_counter()
    print(f"split_data, time: {t1 - t0}")

    print("training model")
    t0 = time.perf_counter()
    model = train(model_data["X_train"], model_data["y_train"])
    t1 = time.perf_counter()
    print(f"model created, time: {t1 - t0}")

    print("saving model")
    t0 = time.perf_counter()
    output_file = "model.pkl"
    save_model(model, output_file)
    t1 = time.perf_counter()
    print(f"Saving model, time: {t1 - t0}")

    print("evaluating model")
    t0 = time.perf_counter()
    metrics = evaluate(model_data["X_test"], model_data["y_test"], model)
    t1 = time.perf_counter()
    print(f"Final metrics {metrics}, time: {t1 - t0}")
