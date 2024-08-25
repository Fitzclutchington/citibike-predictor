from pathlib import Path

import pandas as pd


def read_csv(filename: str) -> pd.DataFrame:
    """
    Read DataFrame from a CSV file ``filename`` and convert to a
    preferred schema.
    """
    df = pd.read_csv(
        filename,
        sep=",",
        na_values="\\N",
        dtype={
            "station_id": str,
            # Use Pandas Int16 dtype to allow for nullable integers
            "num_bikes_available": "Int16",
            "num_ebikes_available": "Int16",
            "num_bikes_disabled": "Int16",
            "num_docks_available": "Int16",
            "num_docks_disabled": "Int16",
            "is_installed": "Int16",
            "is_renting": "Int16",
            "is_returning": "Int16",
            "station_status_last_reported": "Int64",
            "station_name": str,
            "lat": float,
            "lon": float,
            "region_id": str,
            "capacity": "Int16",
            # Use pandas boolean dtype to allow for nullable booleans
            "has_kiosk": "boolean",
            "station_information_last_updated": "Int64",
            "missing_station_information": "boolean",
        },
    )
    # Read in timestamps as UNIX/POSIX epochs but then convert to the local
    # bike share timezone.
    df["station_status_last_reported"] = pd.to_datetime(
        df["station_status_last_reported"], unit="s", origin="unix", utc=True
    ).dt.tz_convert("US/Eastern")

    df["station_information_last_updated"] = pd.to_datetime(
        df["station_information_last_updated"], unit="s", origin="unix", utc=True
    ).dt.tz_convert("US/Eastern")
    return df


def resample(df: pd.DataFrame, interval: str = "1h") -> pd.DataFrame:
    """
    - sort df by station_id, station_status_last_reported
    - resample df by time interval provided
    """

    # TODO: make this configurable
    # TODO: investigate different agg strats
    # NOTE: does using mean for discrete values like num_bikes_available have major consequences?
    agg_strat = {
        "num_bikes_available": "mean",
        "num_ebikes_available": "mean",
        "num_bikes_disabled": "mean",
        "num_docks_available": "mean",
        "num_docks_disabled": "mean",
        "is_installed": "first",
        "is_renting": "first",
        "is_returning": "first",
        "station_name": "first",
        "lat": "first",
        "lon": "first",
        "region_id": "first",
        "capacity": "first",
    }
    # remove rows with missing info
    df = df[df["missing_station_information"] == False]
    df.set_index(["station_id", "station_status_last_reported"], inplace=True)
    df.sort_index(inplace=True)
    df = df.groupby("station_id").resample(interval, level=1).agg(agg_strat)
    
    return df

if __name__ == "__main__":
    data_path = "/Users/fitz/Documents/citibike-predictor/data/citi_bike_data_00000.csv"
    df = read_csv(data_path)
    df = resample(df)
    # TODO: Save preprocessed data (parquet? csv?)
    # How do we partition?
