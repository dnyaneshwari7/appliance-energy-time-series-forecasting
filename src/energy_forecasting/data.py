from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TARGET


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the appliance energy CSV and parse the timestamp index."""
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the appliance energy data.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    return df


def prepare_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 10-minute observations to hourly means and interpolate small gaps."""
    hourly = df.resample("h").mean()
    hourly = hourly.interpolate(method="time").dropna()
    hourly = hourly.asfreq("h")
    return hourly


def train_test_split_time(df: pd.DataFrame, test_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_steps:
        raise ValueError("Data is shorter than the requested test period.")
    return df.iloc[:-test_steps].copy(), df.iloc[-test_steps:].copy()


def forecast_window(test: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return test.iloc[:horizon].copy()

