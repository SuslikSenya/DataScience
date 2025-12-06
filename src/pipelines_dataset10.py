import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    DATASET10_PATH,
    DATASET10_MONTH_COLUMNS,
    DATASET10_REGION_COL,
    LINEAR_TRAIN_RATIO_DS10,
    TS_DECOMP_MODEL,
    TS_DECOMP_PERIOD_DS10,
    TS_SYNTHETIC_YEARS,
    TS_NOISE_SCALE,
)

from src.models import Models
from src.ts_analysis import (
    metrics_regression,
    analyze_matrix,
    decompose_and_plot,
)


def load_and_clean_dataset10(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df[DATASET10_REGION_COL] = df[DATASET10_REGION_COL].astype(str).str.strip()
    df.replace(
        {
            "n.a.": np.nan,
            "not avilable": np.nan,
            -1.0: np.nan,
            -1: np.nan,
        },
        inplace=True,
    )
    for col in DATASET10_MONTH_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[DATASET10_MONTH_COLUMNS] = (
        df[DATASET10_MONTH_COLUMNS].T.interpolate(limit_direction="both").T
    )
    df[DATASET10_MONTH_COLUMNS] = df[DATASET10_MONTH_COLUMNS].fillna(
        df[DATASET10_MONTH_COLUMNS].mean()
    )
    return df


def build_region_ts_dataset10(df: pd.DataFrame) -> pd.DataFrame:
    region_ts = df.groupby(DATASET10_REGION_COL)[DATASET10_MONTH_COLUMNS].mean()
    return region_ts


def statistical_learning_dataset10(region_ts: pd.DataFrame) -> pd.DataFrame:
    records = []
    n = len(DATASET10_MONTH_COLUMNS)
    x = np.arange(n, dtype=float)
    cut = int(LINEAR_TRAIN_RATIO_DS10 * n)
    device = "cuda" if False else "cpu"

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

        months = pd.date_range("2025-01-01", periods=n, freq="M")
        if "Poly2" in models.models:
            m_plot = models.models["Poly2"]
        else:
            first_key = list(models.models.keys())[0]
            m_plot = models.models[first_key]
        y_pred_full = m_plot.predict(x)

        plt.figure(figsize=(9, 5))
        plt.plot(months, y, marker="o", label="Sales")
        plt.plot(months, y_pred_full, linestyle="--", label="Poly2")
        plt.title(f"PolyRegression {region}")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "dataset10_regression"), exist_ok=True)
        plt.savefig(
            os.path.join(FIGURES_DIR, "dataset10_regression", f"regression_{region}.png")
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
    df_raw = load_and_clean_dataset10(DATASET10_PATH)
    region_ts = build_region_ts_dataset10(df_raw)

    dates = pd.date_range("2025-01-01", periods=len(DATASET10_MONTH_COLUMNS), freq="M")
    plt.figure(figsize=(10, 6))
    for region, row in region_ts.iterrows():
        plt.plot(dates, row.values, marker="o", label=region)
    plt.title("DataSet_10")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(FIGURES_DIR, "dataset10"), exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "dataset10", "dataset10_time_series.png"))
    plt.close()

    data = pd.DataFrame(
        {region: region_ts.loc[region].values for region in region_ts.index},
        index=DATASET10_MONTH_COLUMNS,
    ).T
    data.columns = dates

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
