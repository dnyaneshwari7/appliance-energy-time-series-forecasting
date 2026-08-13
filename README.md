# Appliance Energy Time-Series Forecasting

This repository contains a reproducible time-series case study for forecasting household appliance energy demand using the Appliances Energy Prediction dataset.

The project compares:

- Mean forecast
- Naive forecast
- Daily seasonal naive forecast
- Weekly seasonal naive forecast
- Drift forecast
- SARIMAX with daily seasonality and exogenous variables
- Feature-based gradient boosting machine-learning model using lag, rolling, time, sensor, and weather features
- Local foundation-model surrogate baseline for comparison with Chronos, TimesFM, or TimeGPT style workflows

## Forecasting Problem

The target variable is:

```text
Appliances
```

The original data are sampled every 10 minutes. The pipeline resamples them to hourly averages to make SARIMAX fitting and comparison manageable.

The forecasting task is:

```text
Forecast appliance energy use for the next 24 hours.
```

The final 14 days are reserved as the test period. The first 24 hours of that test period are used for the headline forecast plots and metrics.

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── energydata.csv
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   └── metrics/
├── scripts/
│   ├── run_pipeline.py
│   └── run_demo_pipeline_reference.py
├── src/
│   └── energy_forecasting/
│       ├── benchmarks.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── foundation.py
│       ├── metrics.py
│       ├── ml_model.py
│       ├── plots.py
│       ├── sarimax_model.py
│       └── stationarity.py
└── tests/
```

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Full Pipeline

From the repository root:

```bash
python scripts/run_pipeline.py
```

For a faster development run:

```bash
python scripts/run_pipeline.py --sarimax-grid-limit 36 --sarimax-train-days 30
```

For the final full assignment grid over all `p=0..6`, `d=0..2`, `q=0..6` combinations:

```bash
python scripts/run_pipeline.py --full-sarimax-grid --sarimax-train-days 60
```

The pipeline produces:

- `data/processed/appliance_hourly.csv`
- `data/processed/stationarity_differences.csv`
- `outputs/metrics/stationarity_tests.csv`
- `outputs/metrics/sarimax_grid_search.csv`
- `outputs/metrics/model_comparison.csv`
- `outputs/metrics/feature_importance.csv`
- `outputs/metrics/run_summary.json`
- `outputs/forecasts/all_forecasts_24h.csv`
- `outputs/forecasts/sarimax_confidence_intervals.csv`
- `outputs/figures/*.png`

## Important Modelling Notes

The assignment asks for SARIMAX hyperparameter tuning over:

```text
p = 0 to 6
d = 0 to 2
q = 0 to 6
```

The pipeline supports the full 147-combination grid with `--full-sarimax-grid`. The default run uses a capped grid so the code can be tested quickly while developing. For the final report results, run the full-grid command above.

The foundation-model component is implemented as a transparent local surrogate. It blends daily and weekly seasonal profiles to provide a target-only zero-shot baseline. This is included because Chronos, TimesFM, or TimeGPT may require large downloads or API access. If a real foundation model is available, replace `foundation_model_surrogate` in `src/energy_forecasting/foundation.py` with the real model call and rerun the pipeline.

## Report-Ready Outputs

Use these outputs in the final 6-8 page report:

- Initial time-series plot: `outputs/figures/01_hourly_series.png`
- Daily seasonality plot: `outputs/figures/02_daily_seasonal_profile.png`
- Weekly pattern plot: `outputs/figures/03_weekly_profile.png`
- ACF/PACF plots: `outputs/figures/original_series_acf_pacf.png`
- Differenced ACF/PACF: `outputs/figures/first_differenced_series_acf_pacf.png`
- Forecast comparison: `outputs/figures/forecast_comparison_24h.png`
- Model metric comparison: `outputs/figures/model_rmse_comparison.png`
- SARIMAX residual diagnostics: `outputs/figures/sarimax_residual_diagnostics.png`
- Metrics table: `outputs/metrics/model_comparison.csv`

## Key Questions for the Report

The report should answer:

1. Which benchmark is strongest and what does that say about appliance-use structure?
2. Does SARIMAX improve on the strongest seasonal benchmark?
3. Does the feature-based model improve when lag, rolling, time, sensor, and weather variables are added?
4. Does the foundation-model approach outperform simpler approaches?
5. Which variables are genuinely known at the forecast origin?
6. Which model is recommended for practical smart-home forecasting?

## Reproducibility

The dataset file should be placed at:

```text
data/raw/energydata.csv
```

The repository includes modular code so that preprocessing, benchmark forecasts, SARIMAX, feature engineering, evaluation, and plotting can be inspected and reused independently.
