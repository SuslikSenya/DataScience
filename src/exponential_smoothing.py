import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import TS_DECOMP_PERIOD_SYN, REPORTS_DIR, FIGURES_DIR, FORECAST_HORIZONS, SYN_TREND_TYPE, \
    LINEAR_TRAIN_RATIO_DS10, TS_DECOMP_PERIOD_DS10
from src.synthetic_core import generate_trend
from src.ts_analysis import generate_extrapolation_x, metrics_regression


def run_exp_smoothing_synthetic(
        x_train: np.ndarray,
        trend_train: np.ndarray,
        y_train: np.ndarray,
) -> list:
    configs = [
        {
            "name": "SES",
            "trend": None,
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False,
        },
        {
            "name": "Holt",
            "trend": "add",
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False,
        },
        {
            "name": "HW_add",
            "trend": "add",
            "seasonal": "add",
            "seasonal_periods": TS_DECOMP_PERIOD_SYN,
            "damped_trend": False,
        },
    ]

    metrics_rows = []

    es_dir = os.path.join(REPORTS_DIR, "synthetic_exp_smoothing")
    es_fig_dir = os.path.join(FIGURES_DIR, "synthetic_exp_smoothing")
    os.makedirs(es_dir, exist_ok=True)
    os.makedirs(es_fig_dir, exist_ok=True)

    x_horizons = generate_extrapolation_x(x_train, FORECAST_HORIZONS)
    if x_horizons:
        max_h = max(FORECAST_HORIZONS)
        x_future_full = x_horizons[max_h]
    else:
        x_future_full = np.array([], dtype=float)

    for cfg in configs:
        try:
            model = ExponentialSmoothing(
                y_train,
                trend=cfg["trend"],
                seasonal=cfg["seasonal"],
                seasonal_periods=cfg["seasonal_periods"],
                damped_trend=cfg["damped_trend"],
            ).fit(optimized=True)
        except Exception as e:
            print(f"[WARN] ExpSmoothing synthetic {cfg['name']} failed: {e}")
            continue

        y_fit = np.asarray(model.fittedvalues, dtype=float)
        y_fit = y_fit[: len(trend_train)]

        m_train = metrics_regression(trend_train, y_fit)
        for k, v in m_train.items():
            metrics_rows.append(
                {
                    "dataset": "synthetic",
                    "series": "trend",
                    "family": "exp_smoothing",
                    "model": cfg["name"],
                    "subset": "train",
                    "metric": k,
                    "value": v,
                }
            )

        if x_future_full.size > 0:
            n_steps_full = len(x_future_full)
            y_future_full = np.asarray(model.forecast(steps=n_steps_full), dtype=float)
            trend_future_full = generate_trend(x_future_full, SYN_TREND_TYPE)
            m_future_full = metrics_regression(trend_future_full, y_future_full)
            for k, v in m_future_full.items():
                metrics_rows.append(
                    {
                        "dataset": "synthetic",
                        "series": "trend",
                        "family": "exp_smoothing",
                        "model": cfg["name"],
                        "subset": f"h_{max_h}",
                        "metric": k,
                        "value": v,
                    }
                )

            df_extr = pd.DataFrame(
                {
                    "x": x_future_full,
                    "y_pred": y_future_full,
                    "y_true_trend": trend_future_full,
                }
            )
            df_extr.to_csv(
                os.path.join(es_dir, f"{cfg['name']}_extrapolation_h{max_h}.csv"),
                index=False,
            )

            plt.figure(figsize=(12, 6))
            plt.plot(x_train, y_train, label="train noisy+clean", alpha=0.4)
            plt.plot(x_train, trend_train, "--", label="train trend")
            plt.plot(x_train, y_fit, label=f"{cfg['name']} fitted", linewidth=2)

            plt.axvline(x_train[-1], color="black", linewidth=1.0)

            plt.plot(
                x_future_full,
                trend_future_full,
                "r--",
                label=f"true trend (h={max_h})",
            )
            plt.plot(
                x_future_full,
                y_future_full,
                label=f"{cfg['name']} forecast (h={max_h})",
            )
            plt.title(f"Synthetic: ExpSmoothing {cfg['name']}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    es_fig_dir,
                    f"synthetic_exp_smoothing_{cfg['name']}.png",
                )
            )
            plt.close()

    if metrics_rows:
        df_metrics_es = pd.DataFrame(metrics_rows)
        df_metrics_es.to_csv(
            os.path.join(REPORTS_DIR, "synthetic_exp_smoothing_metrics.csv"),
            index=False,
        )

    return metrics_rows


def run_exp_smoothing_nbu_for_currency(
        train_dates: np.ndarray,
        test_dates: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        currency: str,
) -> list:
    configs = [
        {
            "name": "SES",
            "trend": None,
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False,
        },
        {
            "name": "Holt",
            "trend": "add",
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False,
        },
    ]

    metrics_rows = []

    es_dir = os.path.join(REPORTS_DIR, "real_nbu_exp_smoothing")
    es_fig_dir = os.path.join(FIGURES_DIR, "real_nbu_exp_smoothing")
    os.makedirs(es_dir, exist_ok=True)
    os.makedirs(es_fig_dir, exist_ok=True)

    for cfg in configs:
        try:
            model = ExponentialSmoothing(
                y_train,
                trend=cfg["trend"],
                seasonal=cfg["seasonal"],
                seasonal_periods=cfg["seasonal_periods"],
                damped_trend=cfg["damped_trend"],
            ).fit(optimized=True)
        except Exception as e:
            print(f"[WARN] ExpSmoothing NBU {currency} {cfg['name']} failed: {e}")
            continue

        y_fit = np.asarray(model.fittedvalues, dtype=float)
        y_fit = y_fit[: len(y_train)]

        m_train = metrics_regression(y_train, y_fit)
        for k, v in m_train.items():
            metrics_rows.append(
                {
                    "currency": currency,
                    "family": "exp_smoothing",
                    "model": cfg["name"],
                    "dataset": "train",
                    "metric": k,
                    "value": v,
                }
            )

        if len(y_test) > 0:
            y_forecast = np.asarray(model.forecast(steps=len(y_test)), dtype=float)
            m_test = metrics_regression(y_test, y_forecast)
            for k, v in m_test.items():
                metrics_rows.append(
                    {
                        "currency": currency,
                        "family": "exp_smoothing",
                        "model": cfg["name"],
                        "dataset": "test",
                        "metric": k,
                        "value": v,
                    }
                )

            dates_full = np.concatenate([train_dates, test_dates])
            y_full = np.concatenate([y_train, y_test])
            y_pred_full = np.concatenate([y_fit, y_forecast])

            plt.figure(figsize=(12, 6))
            plt.plot(dates_full, y_full, label=f"{currency} real", alpha=0.6)
            plt.plot(
                dates_full,
                y_pred_full,
                label=f"{currency} {cfg['name']} exp_smoothing",
                linestyle="--",
            )
            plt.axvline(train_dates[-1], color="black", linewidth=1.0)
            plt.title(f"NBU {currency}: ExpSmoothing {cfg['name']}")
            plt.xlabel("date")
            plt.ylabel("rate")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    es_fig_dir,
                    f"{currency}_exp_smoothing_{cfg['name']}.png",
                )
            )
            plt.close()

    return metrics_rows


def run_exp_smoothing_dataset10_region(
        months_all: pd.DatetimeIndex,
        y: np.ndarray,
        region: str,
) -> list:
    configs = [
        {
            "name": "SES",
            "trend": None,
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False,
        },
        {
            "name": "Holt",
            "trend": "add",
            "seasonal": None,
            "seasonal_periods": None,
            "damped_trend": False,
        },
        {
            "name": "HW_add",
            "trend": "add",
            "seasonal": "add",
            "seasonal_periods": TS_DECOMP_PERIOD_DS10,
            "damped_trend": False,
        },
    ]

    metrics_rows = []

    es_dir = os.path.join(REPORTS_DIR, "dataset10_exp_smoothing")
    es_fig_dir = os.path.join(FIGURES_DIR, "dataset10_exp_smoothing")
    os.makedirs(es_dir, exist_ok=True)
    os.makedirs(es_fig_dir, exist_ok=True)

    n = len(y)
    cut = int(LINEAR_TRAIN_RATIO_DS10 * n)
    cut = max(1, min(n - 1, cut))
    y_train = y[:cut]
    y_test = y[cut:]
    train_months = months_all[:cut]
    test_months = months_all[cut:]

    for cfg in configs:
        try:
            model = ExponentialSmoothing(
                y_train,
                trend=cfg["trend"],
                seasonal=cfg["seasonal"],
                seasonal_periods=cfg["seasonal_periods"],
                damped_trend=cfg["damped_trend"],
            ).fit(optimized=True)
        except Exception as e:
            print(f"[WARN] ExpSmoothing Dataset6 {region} {cfg['name']} failed: {e}")
            continue

        y_fit = np.asarray(model.fittedvalues, dtype=float)
        y_fit = y_fit[: len(y_train)]

        m_train = metrics_regression(y_train, y_fit)
        for k, v in m_train.items():
            metrics_rows.append(
                {
                    "region": region,
                    "family": "exp_smoothing",
                    "model": cfg["name"],
                    "dataset": "train",
                    "metric": k,
                    "value": v,
                }
            )

        if len(y_test) > 0:
            y_forecast = np.asarray(model.forecast(steps=len(y_test)), dtype=float)
            m_test = metrics_regression(y_test, y_forecast)
            for k, v in m_test.items():
                metrics_rows.append(
                    {
                        "region": region,
                        "family": "exp_smoothing",
                        "model": cfg["name"],
                        "dataset": "test",
                        "metric": k,
                        "value": v,
                    }
                )

            months_full = np.concatenate([train_months, test_months])
            y_full = np.concatenate([y_train, y_test])
            y_pred_full = np.concatenate([y_fit, y_forecast])

            plt.figure(figsize=(9, 5))
            plt.plot(months_full, y_full, marker="o", label="Sales real", alpha=0.6)
            plt.plot(
                months_full,
                y_pred_full,
                linestyle="--",
                marker="o",
                label=f"{cfg['name']} exp_smoothing",
            )
            plt.axvline(train_months[-1], color="black", linewidth=1.0)
            plt.title(f"Dataset6 {region}: ExpSmoothing {cfg['name']}")
            plt.xlabel("Month")
            plt.ylabel("Sales")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    es_fig_dir,
                    f"{region}_exp_smoothing_{cfg['name']}.png",
                )
            )
            plt.close()

    return metrics_rows
