from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mase(y_true, y_pred, y_train, seasonality: int = 24) -> float:
    y_train = pd.Series(y_train).astype(float)
    if len(y_train) <= seasonality:
        return float("nan")
    scale = np.abs(y_train.iloc[seasonality:].values - y_train.iloc[:-seasonality].values).mean()
    if scale == 0:
        return float("nan")
    return float(mae(y_true, y_pred) / scale)


def evaluate_forecast(name: str, y_true, y_pred, y_train, seasonality: int = 24) -> dict:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(float)
    return {
        "model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=seasonality),
        "Bias": float((y_pred - y_true).mean()),
    }

