import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    ANOMALY_WINDOW,
    ANOMALY_BASE_K,
    ANOMALY_ALPHA,
    ANOMALY_BINS,
    ABG_ALPHA,
    ABG_BETA,
    ABG_GAMMA,
    DT,
    NBU_CURRENCIES,
    NBU_START_DATE,
    NBU_END_DATE,
    NBU_TRAIN_RATIO,
    NBU_CSV_PATH,
    TS_DECOMP_MODEL,
    TS_DECOMP_PERIOD_NBU,
    TS_SYNTHETIC_YEARS,
    TS_NOISE_SCALE,
)

from src.filters import (
    EntropyAnomalyDetector,
    AdaptiveAlphaBetaGammaFilter,
    run_filter_series,
)
from src.models import Models
from src.ts_analysis import metrics_regression, analyze_matrix, decompose_and_plot


def fetch_nbu_rates(currencies, start_date: str, end_date: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    records = []
    dt = start_dt
    while dt <= end_dt:
        date_str_api = dt.strftime("%Y%m%d")
        row = {"date": dt.date()}
        for cur in currencies:
            url = (
                f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
                f"?valcode={cur}&date={date_str_api}&json"
            )
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                rate = float(data[0]["rate"]) if data else np.nan
            except Exception:
                rate = np.nan
            row[cur] = rate
        records.append(row)
        dt += timedelta(days=1)
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def pipeline_real_nbu():
    if os.path.exists(NBU_CSV_PATH):
        df = pd.read_csv(NBU_CSV_PATH, parse_dates=["date"])
    else:
        df = fetch_nbu_rates(NBU_CURRENCIES, NBU_START_DATE, NBU_END_DATE)
        os.makedirs(os.path.dirname(NBU_CSV_PATH), exist_ok=True)
        df.to_csv(NBU_CSV_PATH, index=False)

    df = df.dropna(subset=NBU_CURRENCIES)
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    cut = int(NBU_TRAIN_RATIO * n)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)

    x_train = np.arange(len(train), dtype=float)
    x_test = np.arange(len(train), len(df), dtype=float)

    rows_metrics = []
    anomaly_rows = []
    all_series = {}

    for cur in NBU_CURRENCIES:
        y_train_raw = train[cur].astype(float).values
        y_test = test[cur].astype(float).values

        detector = EntropyAnomalyDetector(
            window=ANOMALY_WINDOW,
            base_k=ANOMALY_BASE_K,
            alpha=ANOMALY_ALPHA,
            bins=ANOMALY_BINS,
        )
        y_train_clean, anomalies_count = detector.clean(y_train_raw)

        flt = AdaptiveAlphaBetaGammaFilter(ABG_ALPHA, ABG_BETA, ABG_GAMMA, DT)
        y_train_smooth = run_filter_series(flt, y_train_clean)

        all_series[cur] = y_train_smooth

        anomaly_rows.append(
            {
                "currency": cur,
                "metric": "total_anomalies",
                "value": detector.total_anomalies,
            }
        )
        anomaly_rows.append(
            {
                "currency": cur,
                "metric": "avg_entropy",
                "value": detector.avg_entropy,
            }
        )
        anomaly_rows.append(
            {
                "currency": cur,
                "metric": "avg_k_local",
                "value": detector.avg_k_local,
            }
        )

        dates_train = train["date"].values
        plt.figure(figsize=(12, 6))
        plt.plot(dates_train, y_train_raw, label=f"{cur} raw", alpha=0.3)
        plt.plot(
            dates_train,
            y_train_smooth,
            label=f"{cur} cleaned+smoothed (Entropy + ABG_adaptive)",
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "real_nbu_filters"), exist_ok=True)
        plt.savefig(
            os.path.join(FIGURES_DIR, "real_nbu_filters", f"{cur}_cleaned_smoothed.png")
        )
        plt.close()

        models = Models(device="cpu")
        models.fit_all(x_train, y_train_smooth)

        for name, m in models.models.items():
            y_pred_train = m.predict(x_train)
            y_pred_test = m.predict(x_test)
            train_metrics = metrics_regression(y_train_smooth, y_pred_train)
            test_metrics = metrics_regression(y_test, y_pred_test)
            for dataset, mm in [("train", train_metrics), ("test", test_metrics)]:
                for k, v in mm.items():
                    rows_metrics.append(
                        {
                            "currency": cur,
                            "model": name,
                            "dataset": dataset,
                            "metric": k,
                            "value": v,
                        }
                    )

        dates_full = df["date"].values
        y_full = df[cur].astype(float).values
        if "Poly2" in models.models:
            m_plot = models.models["Poly2"]
        else:
            first_key = list(models.models.keys())[0]
            m_plot = models.models[first_key]
        y_pred_full = m_plot.predict(np.arange(len(df), dtype=float))

        plt.figure(figsize=(12, 6))
        plt.plot(dates_full, y_full, label=f"{cur} real", alpha=0.7)
        plt.plot(
            dates_full, y_pred_full, label=f"{cur} Poly2 regression", linestyle="--"
        )
        plt.axvline(dates_full[cut - 1], color="black", linewidth=1.0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "real_nbu"), exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "real_nbu", f"{cur}_regression.png"))
        plt.close()

        decompose_and_plot(
            y=y_full,
            dates=pd.to_datetime(df["date"].values),
            title=f"NBU decomposition {cur}",
            fig_path=os.path.join(FIGURES_DIR, "real_nbu", f"{cur}_decomposition.png"),
            model=TS_DECOMP_MODEL,
            period=TS_DECOMP_PERIOD_NBU,
        )

    os.makedirs(os.path.join(REPORTS_DIR, "real_nbu"), exist_ok=True)
    df_metrics = pd.DataFrame(rows_metrics)
    df_metrics.to_csv(
        os.path.join(REPORTS_DIR, "real_nbu", "nbu_model_metrics.csv"), index=False
    )

    df_anomaly = pd.DataFrame(anomaly_rows)
    df_anomaly.to_csv(
        os.path.join(REPORTS_DIR, "real_nbu", "real_nbu_anomaly_report.csv"),
        index=False,
    )

    if all_series:
        data_matrix = pd.DataFrame(all_series).T
        data_matrix.columns = train["date"].values
        analyze_matrix(
            data=data_matrix,
            name="real_nbu",
            model=TS_DECOMP_MODEL,
            period=TS_DECOMP_PERIOD_NBU,
            synthetic_years=TS_SYNTHETIC_YEARS,
            noise_scale=TS_NOISE_SCALE,
            reports_dir=os.path.join(REPORTS_DIR, "ts_real_nbu"),
            figures_dir=os.path.join(FIGURES_DIR, "ts_real_nbu"),
        )
