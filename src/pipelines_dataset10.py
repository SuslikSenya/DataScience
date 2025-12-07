import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    DATASET10_PATH,
    DATASET10_REGION_COL,
    TS_DECOMP_MODEL,
    TS_DECOMP_PERIOD_DS10,
    TS_SYNTHETIC_YEARS,
    TS_NOISE_SCALE, FORECAST_HORIZONS, DATASET10_DATE_COL,
    DATASET10_VALUE_COL, DATASET10_TRAIN_RATIO,
)
from src.data_preproccesing import normalize_series, make_windows, denormalize_series, rollout_forecast
from src.models import NNModels

from src.ts_analysis import (
    metrics_regression,
    analyze_matrix,
    decompose_and_plot,
)

WINDOW_DS10 = 10


def load_and_prepare_dataset10(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    df[DATASET10_REGION_COL] = df[DATASET10_REGION_COL].astype(str).str.strip()

    df[DATASET10_DATE_COL] = pd.to_datetime(
        df[DATASET10_DATE_COL], infer_datetime_format=True, dayfirst=False
    )

    df[DATASET10_VALUE_COL] = (
        df[DATASET10_VALUE_COL]
        .astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[DATASET10_VALUE_COL] = pd.to_numeric(df[DATASET10_VALUE_COL], errors="coerce")

    df = df.dropna(
        subset=[DATASET10_REGION_COL, DATASET10_DATE_COL, DATASET10_VALUE_COL]
    )

    return df


def build_region_ts_dataset10(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["YearMonth"] = df[DATASET10_DATE_COL].dt.to_period("M").dt.to_timestamp()

    region_ts = (
        df.groupby([DATASET10_REGION_COL, "YearMonth"])[DATASET10_VALUE_COL]
        .sum()
        .unstack("YearMonth")
        .sort_index(axis=1)
    )

    region_ts = region_ts.fillna(0.0)
    return region_ts


def statistical_learning_dataset10(region_ts: pd.DataFrame) -> pd.DataFrame:
    records = []
    dates = region_ts.columns
    n = len(dates)
    cut = int(DATASET10_TRAIN_RATIO * n)

    months = pd.date_range("2025-01-01", periods=n, freq="M")

    extr_dir = os.path.join(REPORTS_DIR, "dataset10_nn_extrapolation")
    extr_fig_dir = os.path.join(FIGURES_DIR, "dataset10_nn_extrapolation")
    os.makedirs(extr_dir, exist_ok=True)
    os.makedirs(extr_fig_dir, exist_ok=True)

    for region, row in region_ts.iterrows():
        y_full = row.values.astype(float)

        y_norm_full, mean_y, std_y = normalize_series(y_full)
        X_all, y_all_target, idx_all = make_windows(y_norm_full, WINDOW_DS10)
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

            m_train = metrics_regression(y_train_real, y_pred_train)
            if len(y_test_real) > 0:
                m_test = metrics_regression(y_test_real, y_pred_test)
            else:
                m_test = {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}

            for dataset, mm in [("train", m_train), ("test", m_test)]:
                records.append(
                    {
                        "region": region,
                        "family": "nn",
                        "model": name,
                        "dataset": dataset,
                        "mse": mm["mse"],
                        "mae": mm["mae"],
                        "r2": mm["r2"],
                    }
                )

        if "MLP" in nn_models.models:
            m_plot = nn_models.models["MLP"]
        else:
            first_key = list(nn_models.models.keys())[0]
            m_plot = nn_models.models[first_key]

        y_pred_all_norm_plot = m_plot.predict(X_all)
        y_pred_all_plot = denormalize_series(y_pred_all_norm_plot, mean_y, std_y)
        y_pred_full = np.full_like(y_full, np.nan, dtype=float)
        y_pred_full[idx_all] = y_pred_all_plot

        plt.figure(figsize=(9, 5))
        plt.plot(months, y_full, marker="o", label="Sales")
        plt.plot(months, y_pred_full, linestyle="--", label="NN (MLP)")
        plt.title(f"NN Regression {region}")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "dataset10_nn_regression"), exist_ok=True)
        plt.savefig(
            os.path.join(
                FIGURES_DIR, "dataset10_nn_regression", f"regression_nn_{region}.png"
            )
        )
        plt.close()

        train_months = months[:cut]
        test_months = months[cut:] if cut < n else pd.DatetimeIndex([])

        last_train_month = train_months[-1]

        for model_name, m in nn_models.models.items():
            for h in FORECAST_HORIZONS:
                n_steps = max(1, int(len(train_months) * float(h)))
                y_hist = y_full[:cut]
                y_future = rollout_forecast(
                    m,
                    y_history=y_hist,
                    mean=mean_y,
                    std=std_y,
                    window=WINDOW_DS10,
                    n_steps=n_steps,
                )

                future_months = pd.date_range(
                    last_train_month + pd.offsets.MonthEnd(1),
                    periods=n_steps,
                    freq="M",
                )

                df_extr = pd.DataFrame(
                    {
                        "region": region,
                        "model": model_name,
                        "horizon": h,
                        "date": future_months,
                        "y_pred": y_future,
                    }
                )
                df_extr.to_csv(
                    os.path.join(
                        extr_dir,
                        f"{region}_{model_name}_h{h}.csv",
                    ),
                    index=False,
                )

                plt.figure(figsize=(9, 5))
                plt.plot(train_months, y_full[:cut], marker="o", label="train")
                if len(test_months) > 0:
                    plt.plot(test_months, y_full[cut:], marker="o", label="test")
                plt.axvline(last_train_month, color="black", linewidth=1.0)

                plt.plot(
                    future_months,
                    y_future,
                    marker="o",
                    linestyle="--",
                    label=f"{model_name} forecast (h={h})",
                )

                plt.title(f"{region}: NN {model_name}, horizon={h}")
                plt.xlabel("Month")
                plt.ylabel("Sales")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        extr_fig_dir,
                        f"{region}_{model_name}_h{h}.png",
                    )
                )
                plt.close()

    df_metrics = pd.DataFrame(records)
    os.makedirs(os.path.join(REPORTS_DIR, "dataset10"), exist_ok=True)
    df_metrics.to_csv(
        os.path.join(REPORTS_DIR, "dataset10", "dataset10_model_metrics.csv"),
        index=False,
    )
    return df_metrics


def pipeline_dataset10():
    df_raw = load_and_prepare_dataset10(DATASET10_PATH)
    region_ts = build_region_ts_dataset10(df_raw)

    dates = region_ts.columns

    plt.figure(figsize=(10, 6))
    for region, row in region_ts.iterrows():
        plt.plot(dates, row.values, marker="o", label=region)
    plt.title("DataSet_10: monthly sales by region")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(FIGURES_DIR, "dataset10"), exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "dataset10", "dataset10_time_series.png"))
    plt.close()

    data = region_ts.copy()
    analyze_matrix(
        data=data,
        name="dataset10",
        model=TS_DECOMP_MODEL,
        period=TS_DECOMP_PERIOD_DS10,
        synthetic_years=TS_SYNTHETIC_YEARS,
        noise_scale=TS_NOISE_SCALE,
        reports_dir=os.path.join(REPORTS_DIR, "ts_dataset10"),
        figures_dir=os.path.join(FIGURES_DIR, "ts_dataset10"),
    )

    for region, row in region_ts.iterrows():
        y_region = row.values.astype(float)
        decompose_and_plot(
            y=y_region,
            dates=dates,
            title=f"Sales_decompose (DataSet_10, {region})",
            fig_path=os.path.join(
                FIGURES_DIR, "dataset10", f"dataset10_decomposition_{region}.png"
            ),
            model=TS_DECOMP_MODEL,
            period=TS_DECOMP_PERIOD_DS10,
        )

    statistical_learning_dataset10(region_ts)
