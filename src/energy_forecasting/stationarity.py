from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_tests(series: pd.Series) -> pd.DataFrame:
    """Run ADF and KPSS tests for non-stationarity evidence."""
    y = pd.Series(series).dropna().astype(float)
    adf_stat, adf_p, adf_lags, adf_nobs, *_ = adfuller(y, autolag="AIC")
    try:
        kpss_stat, kpss_p, kpss_lags, _ = kpss(y, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p, kpss_lags = float("nan"), float("nan"), float("nan")
    return pd.DataFrame(
        [
            {
                "test": "ADF",
                "statistic": adf_stat,
                "p_value": adf_p,
                "lags": adf_lags,
                "nobs": adf_nobs,
                "interpretation": "Reject unit root if p < 0.05",
            },
            {
                "test": "KPSS",
                "statistic": kpss_stat,
                "p_value": kpss_p,
                "lags": kpss_lags,
                "nobs": len(y),
                "interpretation": "Reject stationarity if p < 0.05",
            },
        ]
    )


def differenced_series(series: pd.Series, seasonal_period: int = 24) -> pd.DataFrame:
    y = pd.Series(series).dropna().astype(float)
    return pd.DataFrame(
        {
            "original": y,
            "first_difference": y.diff(),
            f"seasonal_difference_{seasonal_period}": y.diff(seasonal_period),
        }
    )

