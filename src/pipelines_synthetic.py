import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    SYN_N_TRAIN,
    SYN_N_TEST,
    SYN_X_RANGE_TRAIN,
    SYN_X_RANGE_TEST,
    SYN_TREND_TYPE,
    SYN_NOISE_TYPE,
    SYN_ADD_ANOMALIES,
    SYN_ANOMALY_PERCENTAGE,
    TS_DECOMP_MODEL,
    TS_DECOMP_PERIOD_SYN,
    TS_SYNTHETIC_YEARS,
    TS_NOISE_SCALE,
    ANOMALY_WINDOW,
    ANOMALY_BASE_K,
    ANOMALY_ALPHA,
    ANOMALY_BINS,
    AB_ALPHA,
    AB_BETA,
    ABG_ALPHA,
    ABG_BETA,
    ABG_GAMMA,
    DT, MA_WINDOW_CANDIDATES, ARIMA_D_RANGE, ARIMA_Q_RANGE, ARIMA_P_RANGE, FORECAST_HORIZONS,
)

from src.filters import (
    EntropyAnomalyDetector,
    AlphaBetaFilter,
    AlphaBetaGammaFilter,
    AdaptiveAlphaBetaGammaFilter,
    run_filter_series,
)
from src.models import NNModels
from src.ts_analysis import (
    metrics_regression,
    analyze_matrix,
    decompose_and_plot, generate_extrapolation_x
)
from src.synthetic_core import generate_trend, generate_noise, inject_anomalies


def pipeline_synthetic():
    x_train = np.linspace(*SYN_X_RANGE_TRAIN, SYN_N_TRAIN)
    x_test = np.linspace(*SYN_X_RANGE_TEST, SYN_N_TEST)

    trend_train = generate_trend(x_train, SYN_TREND_TYPE)
    trend_test = generate_trend(x_test, SYN_TREND_TYPE)

    noise_raw = generate_noise(SYN_NOISE_TYPE, SYN_N_TRAIN)
    if SYN_ADD_ANOMALIES:
        noise_raw = inject_anomalies(noise_raw, SYN_ANOMALY_PERCENTAGE)

    detector = EntropyAnomalyDetector(
        window=ANOMALY_WINDOW,
        base_k=ANOMALY_BASE_K,
        alpha=ANOMALY_ALPHA,
        bins=ANOMALY_BINS,
    )
    noise_clean, anomalies_count = detector.clean(noise_raw)
    y_train = trend_train + noise_clean

    anomaly_report = pd.DataFrame(
        [
            {"metric": "total_anomalies", "value": detector.total_anomalies},
            {"metric": "avg_entropy", "value": detector.avg_entropy},
            {"metric": "avg_k_local", "value": detector.avg_k_local},
        ]
    )
    anomaly_report.to_csv(
        os.path.join(REPORTS_DIR, "synthetic_anomaly_report.csv"), index=False
    )

    noise_stats = pd.DataFrame(
        [
            {"metric": "noise_raw_mean", "value": float(noise_raw.mean())},
            {"metric": "noise_raw_std", "value": float(noise_raw.std())},
            {"metric": "noise_clean_std", "value": float(noise_clean.std())},
            {"metric": "anomalies_removed", "value": anomalies_count},
        ]
    )
    noise_stats.to_csv(
        os.path.join(REPORTS_DIR, "synthetic_noise_report.csv"), index=False
    )

    filters = {
        "AB": AlphaBetaFilter(AB_ALPHA, AB_BETA, DT),
        "ABG": AlphaBetaGammaFilter(ABG_ALPHA, ABG_BETA, ABG_GAMMA, DT),
        "ABG_adaptive": AdaptiveAlphaBetaGammaFilter(
            ABG_ALPHA, ABG_BETA, ABG_GAMMA, DT
        ),
    }

    filter_rows = []
    filter_outputs = {}
    for name, flt in filters.items():
        y_filt = run_filter_series(flt, y_train)
        filter_outputs[name] = y_filt
        m = metrics_regression(trend_train, y_filt)
        m["bias"] = float(np.mean(trend_train - y_filt))
        for k, v in m.items():
            filter_rows.append(
                {"filter": name, "dataset": "synthetic_train", "metric": k, "value": v}
            )

    df_filter = pd.DataFrame(filter_rows)
    df_filter.to_csv(
        os.path.join(REPORTS_DIR, "synthetic_filter_report.csv"), index=False
    )

    best_filter_name = (
        df_filter[df_filter["metric"] == "mse"].sort_values("value").iloc[0]["filter"]
    )
    best_output = filter_outputs[best_filter_name]

    plt.figure(figsize=(12, 6))
    plt.plot(x_train, y_train, label="train noisy+clean", alpha=0.6)
    plt.plot(x_train, trend_train, label="true trend", linestyle="--")
    plt.plot(x_train, best_output, label=f"best filter: {best_filter_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.join(FIGURES_DIR, "synthetic_filters"), exist_ok=True)
    plt.savefig(
        os.path.join(
            FIGURES_DIR, "synthetic_filters", f"{best_filter_name}_synthetic.png"
        )
    )
    plt.close()

    metrics_rows = []

    models = NNModels()
    models.fit_all(x_train, y_train)

    for name, m in models.models.items():
        y_pred_train = m.predict(x_train)
        y_pred_test = m.predict(x_test)

        plt.figure(figsize=(12, 6))
        plt.plot(x_train, y_train, label="train noisy", alpha=0.6)
        plt.plot(x_train, trend_train, label="train trend", linestyle="--")
        plt.plot(x_train, y_pred_train, label=f"train pred ({name})", linewidth=2)
        plt.axvline(x_train[-1], color="black", linewidth=1.0)
        plt.plot(x_test, trend_test, label="test trend", linestyle="--")
        plt.plot(x_test, y_pred_test, label=f"test pred ({name})", linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "synthetic_models"), exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "synthetic_models", f"{name}.png"))
        plt.close()

        train_metrics = metrics_regression(trend_train, y_pred_train)
        test_metrics = metrics_regression(trend_test, y_pred_test)
        for dataset, mm in [("train", train_metrics), ("test", test_metrics)]:
            for k, v in mm.items():
                metrics_rows.append(
                    {
                        "dataset": "synthetic",
                        "series": "trend",
                        "family": "regression",
                        "model": name,
                        "subset": dataset,
                        "metric": k,
                        "value": v,
                    }
                )

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(
        os.path.join(REPORTS_DIR, "synthetic_model_metrics.csv"), index=False
    )

    x_horizons = generate_extrapolation_x(x_train, FORECAST_HORIZONS)
    extr_dir = os.path.join(REPORTS_DIR, "synthetic_extrapolation")
    os.makedirs(extr_dir, exist_ok=True)

    for model_name, m in models.models.items():
        for h, x_future in x_horizons.items():
            y_future = m.predict(x_future)
            df_extr = pd.DataFrame({"x": x_future, "y_pred": y_future})
            df_extr.to_csv(
                os.path.join(extr_dir, f"{model_name}_h{h}.csv"),
                index=False,
            )

    data = pd.DataFrame(
        {"trend": trend_train, "noisy_clean": y_train},
        index=pd.date_range("2025-01-01", periods=len(y_train), freq="D"),
    ).T

    analyze_matrix(
        data=data,
        name="synthetic_pipeline",
        model=TS_DECOMP_MODEL,
        period=TS_DECOMP_PERIOD_SYN,
        synthetic_years=TS_SYNTHETIC_YEARS,
        noise_scale=TS_NOISE_SCALE,
        reports_dir=os.path.join(REPORTS_DIR, "ts_synthetic"),
        figures_dir=os.path.join(FIGURES_DIR, "ts_synthetic"),
    )

    dates_train = pd.date_range("2025-01-01", periods=len(y_train), freq="D")
    decompose_and_plot(
        y=y_train,
        dates=dates_train,
        title="Synthetic decompose (noisy_clean)",
        fig_path=os.path.join(
            FIGURES_DIR, "synthetic_pipeline_decomposition_noisy_clean.png"
        ),
        model=TS_DECOMP_MODEL,
        period=TS_DECOMP_PERIOD_SYN,
    )
