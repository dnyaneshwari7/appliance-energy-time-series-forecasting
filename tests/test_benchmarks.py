import pandas as pd

from energy_forecasting.benchmarks import drift_forecast, seasonal_naive_forecast


def test_seasonal_naive_forecast() -> None:
    y = pd.Series(range(10))
    index = pd.RangeIndex(3)
    forecast = seasonal_naive_forecast(y, index, seasonality=2)
    assert forecast.tolist() == [8, 9, 8]


def test_drift_forecast_length() -> None:
    y = pd.Series([10, 20, 30])
    index = pd.RangeIndex(4)
    forecast = drift_forecast(y, index)
    assert len(forecast) == 4

