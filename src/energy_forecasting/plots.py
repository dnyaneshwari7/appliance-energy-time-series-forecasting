from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def save_initial_plots(hourly: pd.DataFrame, target: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    hourly[target].plot(ax=ax, color="#1f77b4", linewidth=0.8)
    ax.set_title("Hourly appliance energy use")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance energy use")
    fig.tight_layout()
    fig.savefig(output_dir / "01_hourly_series.png", dpi=180)
    plt.close(fig)

    daily_profile = hourly[target].groupby(hourly.index.hour).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    daily_profile.plot(kind="bar", ax=ax, color="#2ca02c")
    ax.set_title("Average appliance energy use by hour of day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean appliance energy use")
    fig.tight_layout()
    fig.savefig(output_dir / "02_daily_seasonal_profile.png", dpi=180)
    plt.close(fig)

    weekly_profile = hourly[target].groupby(hourly.index.dayofweek).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    weekly_profile.plot(kind="bar", ax=ax, color="#9467bd")
    ax.set_title("Average appliance energy use by day of week")
    ax.set_xlabel("Day of week, Monday=0")
    ax.set_ylabel("Mean appliance energy use")
    fig.tight_layout()
    fig.savefig(output_dir / "03_weekly_profile.png", dpi=180)
    plt.close(fig)


def save_acf_pacf(series: pd.Series, output_dir: Path, prefix: str) -> None:
    y = pd.Series(series).dropna().astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y, ax=axes[0], lags=72)
    plot_pacf(y, ax=axes[1], lags=72, method="ywm")
    axes[0].set_title(f"{prefix}: ACF")
    axes[1].set_title(f"{prefix}: PACF")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix.lower().replace(' ', '_')}_acf_pacf.png", dpi=180)
    plt.close(fig)


def save_forecast_plot(actual: pd.Series, forecasts: dict[str, pd.Series], output_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    actual.plot(ax=ax, color="black", linewidth=2, label="Actual")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#8c564b"]
    for color, (name, forecast) in zip(colors, forecasts.items()):
        forecast.plot(ax=ax, linewidth=1.8, label=name, color=color)
    ax.set_title("24-hour appliance energy forecasts")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Appliance energy use")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def save_metrics_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    plot_df = metrics.sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(plot_df["model"], plot_df["RMSE"], color="#1f77b4")
    ax.set_title("Model comparison by RMSE")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "model_rmse_comparison.png", dpi=180)
    plt.close(fig)


def save_residual_diagnostics(residuals: pd.Series, output_dir: Path) -> None:
    residuals = pd.Series(residuals).dropna().astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    residuals.plot(kind="hist", bins=40, ax=axes[0], color="#1f77b4")
    axes[0].set_title("SARIMAX residual distribution")
    plot_acf(residuals, ax=axes[1], lags=72)
    axes[1].set_title("SARIMAX residual ACF")
    fig.tight_layout()
    fig.savefig(output_dir / "sarimax_residual_diagnostics.png", dpi=180)
    plt.close(fig)

