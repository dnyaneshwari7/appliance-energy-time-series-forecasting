from __future__ import annotations

import itertools
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def sarimax_grid_search(
    y_train: pd.Series,
    exog_train: pd.DataFrame | None = None,
    p_values=range(0, 7),
    d_values=range(0, 3),
    q_values=range(0, 7),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
    max_combinations: int | None = 80,
) -> pd.DataFrame:
    """Search SARIMAX p,d,q values by AIC.

    The assignment asks for p=[0,6], d=[0,2], q=[0,6]. For runtime practicality,
    max_combinations can cap the demo run; set it to None for the full 147-model grid.
    """
    rows = []
    combos = list(itertools.product(p_values, d_values, q_values))
    if max_combinations is not None:
        combos = combos[:max_combinations]
    for order in combos:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=80)
                rows.append({"order": str(order), "aic": fit.aic, "bic": fit.bic, "converged": fit.mle_retvals.get("converged", None)})
            except Exception as exc:
                rows.append({"order": str(order), "aic": float("inf"), "bic": float("inf"), "converged": False, "error": str(exc)[:140]})
    return pd.DataFrame(rows).sort_values("aic")


def fit_sarimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=120)


def forecast_sarimax(fit, index: pd.Index, exog_test: pd.DataFrame | None = None) -> tuple[pd.Series, pd.DataFrame]:
    forecast = fit.get_forecast(steps=len(index), exog=exog_test)
    mean = forecast.predicted_mean
    mean.index = index
    mean.name = "sarimax"
    conf = forecast.conf_int()
    conf.index = index
    return mean, conf

