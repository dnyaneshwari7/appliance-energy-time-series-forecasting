from __future__ import annotations

import ast
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_forecasting.benchmarks import benchmark_forecasts
from energy_forecasting.config import (
    DAILY_PERIOD,
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES,
    FORECASTS,
    FORECAST_HORIZON,
    METRICS,
    TARGET,
    TEST_STEPS,
)
from energy_forecasting.data import forecast_window, load_raw_data, prepare_hourly_data, train_test_split_time
from energy_forecasting.features import add_time_features, available_exog_columns, make_supervised_table
from energy_forecasting.foundation import foundation_model_surrogate
from energy_forecasting.metrics import evaluate_forecast
from energy_forecasting.ml_model import feature_importance, fit_feature_model, forecast_feature_model
from energy_forecasting.plots import (
    save_acf_pacf,
    save_forecast_plot,
    save_initial_plots,
    save_metrics_plot,
    save_residual_diagnostics,
)
from energy_forecasting.sarimax_model import fit_sarimax, forecast_sarimax, sarimax_grid_search
from energy_forecasting.stationarity import differenced_series, stationarity_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run appliance energy forecasting pipeline.")
    parser.add_argument(
        "--full-sarimax-grid",
        action="store_true",
        help="Run the full p=0..6, d=0..2, q=0..6 SARIMAX grid. This is slow but matches the full assignment grid.",
    )
    parser.add_argument(
        "--sarimax-grid-limit",
        type=int,
        default=36,
        help="Maximum SARIMAX p,d,q combinations in normal mode. Ignored when --full-sarimax-grid is used.",
    )
    parser.add_argument(
        "--sarimax-train-days",
        type=int,
        default=30,
        help="Most recent training days used for SARIMAX tuning to keep runtime practical.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading and preparing data...")
    raw = load_raw_data(DATA_RAW)
    hourly = prepare_hourly_data(raw)
    hourly.to_csv(DATA_PROCESSED / "appliance_hourly.csv")

    train, test = train_test_split_time(hourly, TEST_STEPS)
    horizon_test = forecast_window(test, FORECAST_HORIZON)
    y_train = train[TARGET]
    y_test = horizon_test[TARGET]

    print("Creating EDA and stationarity outputs...")
    save_initial_plots(hourly, TARGET, FIGURES)
    save_acf_pacf(y_train, FIGURES, "Original series")
    save_acf_pacf(y_train.diff().dropna(), FIGURES, "First differenced series")
    stationarity_tests(y_train).to_csv(METRICS / "stationarity_tests.csv", index=False)
    differenced_series(y_train, DAILY_PERIOD).to_csv(DATA_PROCESSED / "stationarity_differences.csv")

    print("Fitting benchmark models...")
    forecasts = benchmark_forecasts(y_train, y_test.index)

    print("Preparing exogenous variables for SARIMAX...")
    train_exog_base = add_time_features(train)
    test_exog_base = add_time_features(horizon_test)
    exog_cols = available_exog_columns(train)
    X_train_exog = train_exog_base[exog_cols]
    X_test_exog = test_exog_base[exog_cols]

    # Use the most recent training period for a practical grid search runtime.
    sarimax_train_window = min(len(y_train), 24 * args.sarimax_train_days)
    y_sarimax_train = y_train.iloc[-sarimax_train_window:]
    X_sarimax_train = X_train_exog.iloc[-sarimax_train_window:]
    print("Running SARIMAX AIC grid search...")
    grid_limit = None if args.full_sarimax_grid else args.sarimax_grid_limit
    print(
        "SARIMAX grid mode:",
        "full 147-combination grid" if grid_limit is None else f"fast grid capped at {grid_limit} combinations",
    )
    grid = sarimax_grid_search(
        y_sarimax_train,
        X_sarimax_train,
        p_values=range(0, 7),
        d_values=range(0, 3),
        q_values=range(0, 7),
        max_combinations=grid_limit,
    )
    grid.to_csv(METRICS / "sarimax_grid_search.csv", index=False)
    best_order = ast.literal_eval(grid.iloc[0]["order"])
    print(f"Best SARIMAX order by AIC: {best_order}")

    sarimax_fit = fit_sarimax(y_sarimax_train, X_sarimax_train, order=best_order)
    sarimax_pred, sarimax_conf = forecast_sarimax(sarimax_fit, y_test.index, X_test_exog)
    forecasts["sarimax_exog"] = sarimax_pred
    sarimax_conf.to_csv(FORECASTS / "sarimax_confidence_intervals.csv")
    save_residual_diagnostics(pd.Series(sarimax_fit.resid), FIGURES)

    print("Fitting feature-based machine-learning model...")
    X_all, y_all = make_supervised_table(hourly)
    X_train_ml = X_all.loc[X_all.index <= train.index[-1]]
    y_train_ml = y_all.loc[X_train_ml.index]
    X_test_ml = X_all.loc[y_test.index]
    feature_model = fit_feature_model(X_train_ml, y_train_ml)
    forecasts["feature_model_gradient_boosting"] = forecast_feature_model(feature_model, X_test_ml, y_test.index)
    feature_importance(feature_model, X_train_ml.tail(1000), y_train_ml.tail(1000)).to_csv(
        METRICS / "feature_importance.csv",
        index=False,
    )

    print("Creating foundation-model comparison baseline...")
    forecasts["foundation_surrogate"] = foundation_model_surrogate(y_train, y_test.index)

    print("Evaluating models...")
    metrics = pd.DataFrame(
        [
            evaluate_forecast(name, y_test, pred, y_train, seasonality=DAILY_PERIOD)
            for name, pred in forecasts.items()
        ]
    ).sort_values("RMSE")
    metrics.to_csv(METRICS / "model_comparison.csv", index=False)

    forecast_df = pd.DataFrame({"actual": y_test, **forecasts})
    forecast_df.to_csv(FORECASTS / "all_forecasts_24h.csv")
    save_forecast_plot(y_test, forecasts, FIGURES, "forecast_comparison_24h.png")
    save_metrics_plot(metrics, FIGURES)

    summary = {
        "target": TARGET,
        "frequency": "hourly",
        "forecast_horizon_hours": FORECAST_HORIZON,
        "test_period_days": int(TEST_STEPS / 24),
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "best_model_by_rmse": metrics.iloc[0]["model"],
        "best_sarimax_order": str(best_order),
        "sarimax_grid_mode": "full" if grid_limit is None else "fast",
        "sarimax_grid_combinations_evaluated": int(len(grid)),
        "sarimax_train_days": int(args.sarimax_train_days),
        "exogenous_variables": exog_cols,
        "note": "foundation_surrogate is a transparent local stand-in unless Chronos/TimesFM/TimeGPT is configured.",
    }
    (METRICS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
