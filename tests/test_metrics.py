import pandas as pd

from energy_forecasting.metrics import mae, mase, rmse


def test_mae_and_rmse() -> None:
    assert mae([1, 2, 3], [1, 2, 5]) == 2 / 3
    assert round(rmse([1, 2, 3], [1, 2, 5]), 4) == 1.1547


def test_mase_returns_float() -> None:
    y_train = pd.Series([1, 2, 3, 4, 5, 6])
    value = mase([7, 8], [6, 7], y_train, seasonality=1)
    assert value == 1.0

