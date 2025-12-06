import os

import numpy as np
from typing import Dict, Any

from .data import Config, TrendGenerator, NoiseGenerator, EntropyAnomalyDetector
from .filters import run_filter_series, AlphaBetaFilter, AlphaBetaGammaFilter, AdaptiveAlphaBetaGammaFilter
from .metrics_plot import ModelMetrics, DataMetrics, Plot
from .reports import ReportGenerator


# !=============================================================================
# ! Synthetic Pipeline
# !=============================================================================


def pipeline_synthetic(cfg: Config):
    print("Starting synthetic pipeline.")

    trend_gen = TrendGenerator(cfg)
    noise_gen = NoiseGenerator(cfg)

    # Generate synthetic data
    x_train = np.linspace(*cfg.x_range_train, cfg.n_train)
    x_test = np.linspace(*cfg.x_range_test, cfg.n_test)

    trend_train = trend_gen.generate(x_train, cfg.trend_type)
    trend_test = trend_gen.generate(x_test, cfg.trend_type)

    noise_raw = noise_gen.generate(cfg.noise_type, cfg.n_train, cfg.add_anomalies)

    detector = EntropyAnomalyDetector(
        window=cfg.anomaly_window,
        base_k=cfg.anomaly_threshold,
        alpha=1.2,
        bins=30,
    )

    noise_clean, anomalies_count = detector.clean(noise_raw)
    y_train = trend_train + noise_clean

    # Save anomaly report
    ReportGenerator.save_anomaly_report(
        detector, os.path.join(cfg.save_report_path, "synthetic_anomaly_report.csv")
    )

    data_metrics = {
        "noise_mean": float(noise_raw.mean()),
        "noise_std": float(noise_raw.std()),
        "noise_clean_std": float(noise_clean.std()),
        "anomalies_removed": anomalies_count,
    }

    ReportGenerator.save_data_report(
        data_metrics, os.path.join(cfg.save_report_path, "synthetic_data_report.csv")
    )

    Plot.plot_noise(
        noise_raw,
        noise_clean,
        os.path.join(cfg.save_plot_path, "noise", "synthetic_noise.png"),
    )

    filters = {
        "AB": AlphaBetaFilter(cfg.ab_alpha, cfg.ab_beta, cfg.dt),
        "ABG": AlphaBetaGammaFilter(cfg.abg_alpha, cfg.abg_beta, cfg.abg_gamma, cfg.dt),
        "ABG_adaptive": AdaptiveAlphaBetaGammaFilter(
            cfg.abg_alpha, cfg.abg_beta, cfg.abg_gamma, cfg.dt
        ),
    }

    filter_outputs = {}
    filter_reports = []

    for name, flt in filters.items():
        y_filt = run_filter_series(flt, y_train)
        filter_outputs[name] = y_filt

        metrics = ModelMetrics.compute(trend_train, y_filt)
        metrics["bias"] = float(np.mean(trend_train - y_filt))

        filter_reports.append(
            {
                "filter": name,
                "dataset": "synthetic_train",
                "metrics": metrics,
            }
        )

    ReportGenerator.save_filter_report(
        filter_reports,
        os.path.join(cfg.save_report_path, "synthetic_filter_report.csv"),
    )

    best_report = min(filter_reports, key=lambda r: r["metrics"]["mse"])
    best_filter_name = best_report["filter"]
    best_filter_output = filter_outputs[best_filter_name]

    Plot.plot_best_filter_synthetic(
        x_train,
        y_train,
        trend_train,
        best_filter_name,
        best_filter_output,
        os.path.join(
            cfg.save_plot_path, "filters", f"{best_filter_name}_synthetic.png"
        ),
    )

    reports = []
    for name, model in cfg.models.items():
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        rep = {
            "model": name,
            "train_metrics": ModelMetrics.compute(trend_train, pred_train),
            "test_metrics": ModelMetrics.compute(trend_test, pred_test),
        }
        reports.append(rep)

        Plot.plot_data(
            x_train,
            y_train,
            trend_train,
            pred_train,
            x_test,
            pred_test,
            trend_test,
            fname=f"{cfg.save_plot_path}/{name}_synthetic.png",
        )

    ReportGenerator.save_model_report(
        reports, f"{cfg.save_report_path}/model_report.csv"
    )
