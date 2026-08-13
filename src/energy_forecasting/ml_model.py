from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def fit_feature_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "gradient_boosting"):
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return model.fit(X_train, y_train)
    # GradientBoostingRegressor is used by default because it is stable in
    # restricted Windows environments where thread-pool creation can fail.
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=350,
                    learning_rate=0.04,
                    max_depth=3,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model.fit(X_train, y_train)


def forecast_feature_model(model, X_future: pd.DataFrame, index: pd.Index) -> pd.Series:
    return pd.Series(model.predict(X_future), index=index, name="feature_model")


def feature_importance(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=8, random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    return (
        pd.DataFrame({"feature": X.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
