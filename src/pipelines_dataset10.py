import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    DATASET10_PATH,
    DATASET10_REGION_COL,
    TS_DECOMP_MODEL,
    TS_DECOMP_PERIOD_DS10,
    TS_SYNTHETIC_YEARS,
    TS_NOISE_SCALE, DATASET10_DATE_COL, DATASET10_VALUE_COL, DATASET10_TRAIN_RATIO,
)

from src.models import Models
from src.ts_analysis import (
    metrics_regression,
    analyze_matrix,
    decompose_and_plot,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


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
    x = np.arange(n, dtype=float)
    cut = int(DATASET10_TRAIN_RATIO * n)

    for region, row in region_ts.iterrows():
        y = row.values.astype(float)
        x_train = x[:cut]
        y_train = y[:cut]
        x_test = x[cut:]
        y_test = y[cut:]

        models = Models(device=device)
        models.fit_all(x_train, y_train)

        for name, m in models.models.items():
            y_pred_train = m.predict(x_train)
            m_train = metrics_regression(y_train, y_pred_train)
            if len(y_test) > 0:
                y_pred_test = m.predict(x_test)
                m_test = metrics_regression(y_test, y_pred_test)
            else:
                m_test = {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            for dataset, mm in [("train", m_train), ("test", m_test)]:
                records.append(
                    {
                        "region": region,
                        "model": name,
                        "dataset": dataset,
                        "mse": mm["mse"],
                        "mae": mm["mae"],
                        "r2": mm["r2"],
                    }
                )

        if "Poly2" in models.models:
            m_plot = models.models["Poly2"]
        else:
            first_key = list(models.models.keys())[0]
            m_plot = models.models[first_key]
        y_pred_full = m_plot.predict(x)

        plt.figure(figsize=(9, 5))
        plt.plot(dates, y, marker="o", label="Sales")
        plt.plot(dates, y_pred_full, linestyle="--", label="Poly2")
        plt.title(f"PolyRegression {region} (DataSet_10)")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "dataset10_regression"), exist_ok=True)
        plt.savefig(
            os.path.join(
                FIGURES_DIR, "dataset10_regression", f"regression_{region}.png"
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
