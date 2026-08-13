from __future__ import annotations

import pandas as pd


def foundation_model_surrogate(y_train: pd.Series, index: pd.Index) -> pd.Series:
    """Transparent local stand-in for a foundation model forecast.

    Chronos/TimesFM/TimeGPT may require large downloads or API access. This
    surrogate mimics a robust zero-shot target-only forecast by blending recent
    daily and weekly seasonal profiles. It is reported as a local foundation
    baseline placeholder unless a real foundation model is configured.
    """
    daily = []
    weekly = []
    hist = list(y_train.astype(float).values)
    for _ in range(len(index)):
        daily_val = hist[-24]
        weekly_val = hist[-168]
        pred = 0.65 * daily_val + 0.35 * weekly_val
        daily.append(pred)
        weekly.append(weekly_val)
        hist.append(pred)
    return pd.Series(daily, index=index, name="foundation_surrogate")

