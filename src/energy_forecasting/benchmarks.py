from __future__ import annotations

import pandas as pd


def mean_forecast(y_train: pd.Series, index: pd.Index) -> pd.Series:
    return pd.Series(float(y_train.mean()), index=index, name="mean")


def naive_forecast(y_train: pd.Series, index: pd.Index) -> pd.Series:
    return pd.Series(float(y_train.iloc[-1]), index=index, name="naive")


def seasonal_naive_forecast(y_train: pd.Series, index: pd.Index, seasonality: int) -> pd.Series:
    history = list(y_train.astype(float).values)
    predictions = []
    for _ in range(len(index)):
        predictions.append(history[-seasonality])
        history.append(predictions[-1])
    return pd.Series(predictions, index=index, name=f"seasonal_naive_{seasonality}")


def drift_forecast(y_train: pd.Series, index: pd.Index) -> pd.Series:
    slope = (float(y_train.iloc[-1]) - float(y_train.iloc[0])) / max(1, len(y_train) - 1)
    values = [float(y_train.iloc[-1]) + slope * step for step in range(1, len(index) + 1)]
    return pd.Series(values, index=index, name="drift")


def benchmark_forecasts(y_train: pd.Series, index: pd.Index) -> dict[str, pd.Series]:
    return {
        "mean": mean_forecast(y_train, index),
        "naive": naive_forecast(y_train, index),
        "daily_seasonal_naive": seasonal_naive_forecast(y_train, index, 24),
        "weekly_seasonal_naive": seasonal_naive_forecast(y_train, index, 168),
        "drift": drift_forecast(y_train, index),
    }

