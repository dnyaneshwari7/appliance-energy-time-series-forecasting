from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET


SENSOR_FEATURES = [
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
]

WEATHER_FEATURES = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    return out


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    lags = lags or [1, 2, 3, 6, 12, 24, 48, 168]
    out = df.copy()
    for lag in lags:
        out[f"{TARGET}_lag_{lag}"] = out[TARGET].shift(lag)
    for window in [3, 6, 12, 24, 168]:
        shifted = out[TARGET].shift(1)
        out[f"{TARGET}_roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"{TARGET}_roll_std_{window}"] = shifted.rolling(window).std()
    return out


def make_supervised_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    out = add_lag_features(add_time_features(df))
    candidates = (
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
        + [c for c in SENSOR_FEATURES + WEATHER_FEATURES if c in out.columns]
        + [c for c in out.columns if c.startswith(f"{TARGET}_lag_") or c.startswith(f"{TARGET}_roll_")]
    )
    model_df = out[candidates + [TARGET]].dropna()
    return model_df[candidates], model_df[TARGET]


def available_exog_columns(df: pd.DataFrame) -> list[str]:
    time_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    out = add_time_features(df)
    return [c for c in WEATHER_FEATURES + time_cols if c in out.columns]

