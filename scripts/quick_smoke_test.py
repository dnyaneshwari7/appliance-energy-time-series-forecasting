from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from energy_forecasting.benchmarks import benchmark_forecasts
from energy_forecasting.metrics import evaluate_forecast


def main() -> None:
    index = pd.date_range("2024-01-01", periods=220, freq="h")
    y = pd.Series(range(220), index=index, name="Appliances")
    train = y.iloc[:-24]
    test = y.iloc[-24:]
    forecasts = benchmark_forecasts(train, test.index)
    rows = [evaluate_forecast(name, test, pred, train) for name, pred in forecasts.items()]
    print(pd.DataFrame(rows).sort_values("RMSE").head())


if __name__ == "__main__":
    main()

