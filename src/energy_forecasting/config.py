from pathlib import Path

TARGET = "Appliances"
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168
FORECAST_HORIZON = 24
TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * 24
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "energydata.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"
FORECASTS = OUTPUTS / "forecasts"

for path in [DATA_PROCESSED, FIGURES, METRICS, FORECASTS]:
    path.mkdir(parents=True, exist_ok=True)

