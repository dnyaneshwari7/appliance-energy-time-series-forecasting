import pandas as pd

from energy_forecasting.features import add_time_features


def test_add_time_features() -> None:
    df = pd.DataFrame({"Appliances": [1, 2]}, index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]))
    out = add_time_features(df)
    assert {"hour", "dayofweek", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"}.issubset(out.columns)

