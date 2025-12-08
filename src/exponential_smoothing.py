import os

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import REPORTS_DIR, FIGURES_DIR
from src.ts_analysis import metrics_regression


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

