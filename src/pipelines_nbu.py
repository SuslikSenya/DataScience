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
    TS_NOISE_SCALE, FORECAST_HORIZONS,
)
from src.data_preproccesing import normalize_series, make_windows, denormalize_series, rollout_forecast

from src.filters import (
    EntropyAnomalyDetector,
    run_filter_series, AlphaBetaGammaFilter,
)
from src.models import NNModels
from src.ts_analysis import metrics_regression, analyze_matrix, decompose_and_plot, generate_extrapolation_x

WINDOW_NBU = 20


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

    rows_metrics = []
    anomaly_rows = []
    all_series = {}

    extr_dir = os.path.join(REPORTS_DIR, "real_nbu_extrapolation")
    extr_fig_dir = os.path.join(FIGURES_DIR, "real_nbu_nn_extrapolation")

    os.makedirs(extr_dir, exist_ok=True)
    os.makedirs(extr_fig_dir, exist_ok=True)

    for cur in NBU_CURRENCIES:
        y_full = df[cur].astype(float).values

        y_train_raw = train[cur].astype(float).values

        detector = EntropyAnomalyDetector(
            window=ANOMALY_WINDOW,
            base_k=ANOMALY_BASE_K,
            alpha=ANOMALY_ALPHA,
            bins=ANOMALY_BINS,
        )
        y_train_clean, _ = detector.clean(y_train_raw)

        flt = AlphaBetaGammaFilter(ABG_ALPHA, ABG_BETA, ABG_GAMMA, DT)
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
            label=f"{cur} cleaned+smoothed (Entropy + ABG)",
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "real_nbu_filters"), exist_ok=True)
        plt.savefig(
            os.path.join(FIGURES_DIR, "real_nbu_filters", f"{cur}_cleaned_smoothed.png")
        )
        plt.close()

        y_full_norm, mean_y, std_y = normalize_series(y_full)
        X_all, y_all_target, idx_all = make_windows(y_full_norm, WINDOW_NBU)
        if len(X_all) == 0:
            continue

        mask_train = idx_all < cut
        mask_test_idx = idx_all >= cut

        X_train_w = X_all[mask_train]
        y_train_w = y_all_target[mask_train]
        X_test_w = X_all[mask_test_idx]
        y_test_w = y_all_target[mask_test_idx]

        nn_models = NNModels()
        nn_models.fit_all(X_train_w, y_train_w)

        for name, m in nn_models.models.items():
            y_pred_all_norm = m.predict(X_all)
            y_pred_all = denormalize_series(y_pred_all_norm, mean_y, std_y)

            y_pred_train = y_pred_all[mask_train]
            y_pred_test = y_pred_all[mask_test_idx]

            y_train_real = denormalize_series(y_train_w, mean_y, std_y)
            y_test_real = denormalize_series(y_test_w, mean_y, std_y)

            train_metrics = metrics_regression(y_train_real, y_pred_train)
            test_metrics = metrics_regression(y_test_real, y_pred_test)

            for dataset, mm in [("train", train_metrics), ("test", test_metrics)]:
                for k, v in mm.items():
                    rows_metrics.append(
                        {
                            "currency": cur,
                            "family": "nn",
                            "model": name,
                            "dataset": dataset,
                            "metric": k,
                            "value": v,
                        }
                    )

        dates_full = df["date"].values
        if "MLP" in nn_models.models:
            m_plot = nn_models.models["MLP"]
        else:
            first_key = list(nn_models.models.keys())[0]
            m_plot = nn_models.models[first_key]

        y_pred_all_norm_plot = m_plot.predict(X_all)
        y_pred_all_plot = denormalize_series(y_pred_all_norm_plot, mean_y, std_y)
        y_pred_full = np.full_like(y_full, np.nan, dtype=float)
        y_pred_full[idx_all] = y_pred_all_plot

        plt.figure(figsize=(12, 6))
        plt.plot(dates_full, y_full, label=f"{cur} real", alpha=0.7)
        plt.plot(dates_full, y_pred_full, label=f"{cur} NN (MLP)", linestyle="--")
        plt.axvline(dates_full[cut - 1], color="black", linewidth=1.0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "real_nbu_nn"), exist_ok=True)
        plt.savefig(
            os.path.join(FIGURES_DIR, "real_nbu_nn", f"{cur}_nn_regression.png")
        )
        plt.close()

        decompose_and_plot(
            y=y_full,
            dates=pd.to_datetime(df["date"].values),
            title=f"NBU decomposition {cur}",
            fig_path=os.path.join(
                FIGURES_DIR, "real_nbu_nn", f"{cur}_decomposition.png"
            ),
            model=TS_DECOMP_MODEL,
            period=TS_DECOMP_PERIOD_NBU,
        )

        last_train_date = train["date"].iloc[-1]
        last_test_date = test["date"].iloc[-1] if len(test) > 0 else None

        for model_name, m in nn_models.models.items():
            for h in FORECAST_HORIZONS:
                n_steps = max(1, int(len(train) * float(h)))
                y_hist = y_full[:cut]
                y_future = rollout_forecast(
                    m,
                    y_history=y_hist,
                    mean=mean_y,
                    std=std_y,
                    window=WINDOW_NBU,
                    n_steps=n_steps,
                )

                future_dates = pd.date_range(
                    last_train_date + timedelta(days=1),
                    periods=n_steps,
                    freq="D",
                )

                df_extr = pd.DataFrame(
                    {
                        "currency": cur,
                        "model": model_name,
                        "horizon": h,
                        "date": future_dates,
                        "y_pred": y_future,
                    }
                )
                df_extr.to_csv(
                    os.path.join(
                        extr_dir,
                        f"{cur}_{model_name}_h{h}.csv",
                    ),
                    index=False,
                )

                plt.figure(figsize=(12, 6))
                plt.plot(df["date"], y_full, label="real")
                plt.axvline(last_train_date, color="black", linewidth=1.0)

                if last_test_date is not None:
                    mask_overlap = future_dates <= last_test_date
                else:
                    mask_overlap = np.zeros_like(y_future, dtype=bool)

                if mask_overlap.any():
                    plt.plot(
                        future_dates[mask_overlap],
                        y_future[mask_overlap],
                        label=f"{model_name} forecast (overlap, h={h})",
                    )
                if (~mask_overlap).any():
                    plt.plot(
                        future_dates[~mask_overlap],
                        y_future[~mask_overlap],
                        linestyle="--",
                        label=f"{model_name} forecast (beyond, h={h})",
                    )

                plt.title(f"NBU {cur}: NN {model_name}, horizon={h}")
                plt.xlabel("date")
                plt.ylabel("rate")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        extr_fig_dir,
                        f"{cur}_{model_name}_h{h}.png",
                    )
                )
                plt.close()

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
