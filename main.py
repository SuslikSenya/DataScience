import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# ==============================
# CONFIG
# ==============================

RANDOM_STATE = 42

DATA_DIR = "data"
FIGURES_DIR = "figures"
REPORTS_DIR = "reports"

DATA_FILE = "4_dataset_POPULATION_MARRIAGE_DIVORCE.xlsx"
FORECAST_HORIZON = 5

np.random.seed(RANDOM_STATE)


# ==============================
# UTILS
# ==============================


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def forecast_series_es(series: pd.Series, horizon: int) -> pd.Series:
    """Прогноз для одного часового ряду (експоненціальне згладжування)."""
    series = series.dropna().astype(float)
    if series.empty:
        raise ValueError("Порожній часовий ряд для прогнозування.")

    last_year = int(series.index.max())

    if len(series) < 3:
        idx_future = np.arange(last_year + 1, last_year + 1 + horizon, dtype=int)
        vals = np.repeat(series.iloc[-1], horizon)
        return pd.Series(vals, index=idx_future)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = ExponentialSmoothing(series, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
        fcst = fit.forecast(horizon)
        new_index = np.arange(last_year + 1, last_year + 1 + horizon, dtype=int)
        fcst.index = new_index
        return fcst
    except Exception:
        idx_future = np.arange(last_year + 1, last_year + 1 + horizon, dtype=int)
        vals = np.repeat(series.iloc[-1], horizon)
        return pd.Series(vals, index=idx_future)


# ==============================
# DATA LOADING & PREPROCESSING
# ==============================


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не знайдено файл датасету за шляхом: {path}")
    df = pd.read_excel(path)
    return df


def map_indicator_to_kind(text: str) -> str | None:
    """Мапа 'Показник' -> ['marriage', 'divorce']."""
    s = str(text).lower()
    # шлюбність
    if "шлюб" in s and "розлуч" not in s and "розірван" not in s:
        return "marriage"
    # розлучуваність
    if "розлуч" in s or "розірван" in s:
        return "divorce"
    return None


def preprocess_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    З твого формату робимо таблицю:
    колонки: ['region', 'year', 'marriage', 'divorce'].
    """
    indicator_col = "Показник"
    region_col = "Територіальний розріз"

    if indicator_col not in df_raw.columns or region_col not in df_raw.columns:
        raise ValueError(
            "Не знайдено колонки 'Показник' або 'Територіальний розріз' у файлі."
        )

    # вибираємо колонки-роки
    year_cols: list[str] = []
    for col in df_raw.columns:
        try:
            y = int(str(col))
        except ValueError:
            continue
        if 1900 <= y <= 2100:
            year_cols.append(col)

    if not year_cols:
        raise ValueError("Не знайдено жодної колонки-року (1989, 1990, ...).")

    # long формат: одна строка = (Показник, регіон, рік, значення)
    df_long = df_raw.melt(
        id_vars=[indicator_col, region_col],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )

    df_long["year"] = df_long["year"].astype(int)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # мапимо показник на тип (шлюби / розлучення)
    df_long["kind"] = df_long[indicator_col].apply(map_indicator_to_kind)
    df_long = df_long[df_long["kind"].notna()].copy()

    # агрегуємо по регіону, року та виду показника
    df_group = (
        df_long.groupby([region_col, "year", "kind"], as_index=False)["value"]
        .sum()
        .rename(columns={region_col: "region"})
    )

    # pivot: окремі колонки 'marriage' і 'divorce'
    df_pivot = df_group.pivot_table(
        index=["region", "year"], columns="kind", values="value"
    ).reset_index()

    # щоб були колонки завжди
    if "marriage" not in df_pivot.columns:
        df_pivot["marriage"] = np.nan
    if "divorce" not in df_pivot.columns:
        df_pivot["divorce"] = np.nan

    # на всяк випадок заповнюємо відсутні 0, щоб модель не падала
    df_pivot[["marriage", "divorce"]] = df_pivot[["marriage", "divorce"]].fillna(0.0)

    df_pivot = df_pivot[["region", "year", "marriage", "divorce"]].sort_values(
        ["region", "year"]
    )

    return df_pivot


def split_region_country(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Відділяємо області від країни.
    Рядки з 'region' == 'Україна' вважаємо країною.
    """
    df_country = df_all[df_all["region"] == "Україна"].copy()
    if df_country.empty:
        # fallback – агрегуємо всі регіони
        df_country = (
            df_all.groupby("year", as_index=False)[["marriage", "divorce"]]
            .sum()
            .assign(region="COUNTRY_TOTAL")
        )
    else:
        df_country = df_country.sort_values("year").reset_index(drop=True)
        df_country["region"] = "COUNTRY_TOTAL"

    df_regions = df_all[df_all["region"] != "Україна"].copy()
    return df_regions, df_country


# ==============================
# EXPLORATORY ANALYSIS
# ==============================


def exploratory_analysis(df_regions: pd.DataFrame, df_country: pd.DataFrame) -> None:
    print("\n=== REGIONS (HEAD) ===")
    print(df_regions.head())
    print("\n=== COUNTRY (HEAD) ===")
    print(df_country.head())

    print("\n=== DESCRIBE (REGION-LEVEL) ===")
    print(df_regions[["marriage", "divorce"]].describe())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df_country["year"],
        df_country["marriage"],
        marker="o",
        label="Marriage (country)",
    )
    ax.plot(
        df_country["year"],
        df_country["divorce"],
        marker="o",
        label="Divorce (country)",
    )
    ax.set_title("Динаміка шлюбності та розлучуваності (країна, історія)")
    ax.set_xlabel("Рік")
    ax.set_ylabel("Кількість")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "country_history_marriage_divorce.png"))
    plt.close(fig)


# ==============================
# HIERARCHICAL FORECASTS
# ==============================


def bottom_up_forecast(
    df_region_year: pd.DataFrame,
    target_col: str,
    horizon: int,
) -> pd.DataFrame:
    rows = []

    # 1. прогнози по регіонах
    for region, g in df_region_year.groupby("region"):
        g = g.sort_values("year")
        series = pd.Series(g[target_col].values, index=g["year"].values)
        if series.dropna().empty:
            continue
        fcst = forecast_series_es(series, horizon)

        for year, value in fcst.items():
            rows.append(
                {
                    "approach": "bottom_up",
                    "level": "region",
                    "region": region,
                    "year": int(year),
                    "target": target_col,
                    "value": float(value),
                }
            )

    df_region_fcst = pd.DataFrame(rows)

    # 2. агрегуємо до країни
    if not df_region_fcst.empty:
        df_country_fcst = df_region_fcst.groupby(
            ["approach", "year", "target"], as_index=False
        )["value"].sum()
        df_country_fcst["level"] = "country"
        df_country_fcst["region"] = "COUNTRY_TOTAL"
    else:
        df_country_fcst = pd.DataFrame(
            columns=["approach", "level", "region", "year", "target", "value"]
        )

    df_region_fcst = df_region_fcst[
        ["approach", "level", "region", "year", "target", "value"]
    ]
    df_country_fcst = df_country_fcst[
        ["approach", "level", "region", "year", "target", "value"]
    ]

    df_all = pd.concat([df_region_fcst, df_country_fcst], ignore_index=True)
    return df_all


def top_down_forecast(
    df_region_year: pd.DataFrame,
    df_country: pd.DataFrame,
    target_col: str,
    horizon: int,
) -> pd.DataFrame:
    rows = []

    # 1. прогноз по країні (COUNTRY_TOTAL)
    g_country = df_country.sort_values("year")
    series_country = pd.Series(
        g_country[target_col].values, index=g_country["year"].values
    )
    fcst_country = forecast_series_es(series_country, horizon)

    for year, value in fcst_country.items():
        rows.append(
            {
                "approach": "top_down",
                "level": "country",
                "region": "COUNTRY_TOTAL",
                "year": int(year),
                "target": target_col,
                "value": float(value),
            }
        )

    # 2. частки регіонів за останній історичний рік
    last_hist_year = int(df_region_year["year"].max())
    df_last = df_region_year[df_region_year["year"] == last_hist_year].copy()
    df_last = df_last.sort_values("region")

    s = df_last.set_index("region")[target_col].astype(float)
    if s.sum() <= 0 or s.isna().all():
        shares = pd.Series(
            np.repeat(1.0 / len(df_last), len(df_last)),
            index=df_last["region"].values,
        )
    else:
        shares = (s / s.sum()).fillna(1.0 / len(s))

    # 3. розподіл по регіонах
    for year, country_val in fcst_country.items():
        for region, share in shares.items():
            rows.append(
                {
                    "approach": "top_down",
                    "level": "region",
                    "region": region,
                    "year": int(year),
                    "target": target_col,
                    "value": float(country_val * share),
                }
            )

    df_td = pd.DataFrame(rows)
    df_td = df_td[["approach", "level", "region", "year", "target", "value"]]
    return df_td


def generate_all_forecasts(
    df_regions: pd.DataFrame,
    df_country: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    parts = []
    for target_col in ["marriage", "divorce"]:
        bu = bottom_up_forecast(df_regions, target_col, horizon)
        td = top_down_forecast(df_regions, df_country, target_col, horizon)
        parts.extend([bu, td])
    df_forecasts = pd.concat(parts, ignore_index=True)
    return df_forecasts


# ==============================
# TABLES & PLOTS
# ==============================


def save_forecast_tables(df_forecasts: pd.DataFrame) -> None:
    tidy_path = os.path.join(REPORTS_DIR, "forecasts_tidy.csv")
    df_forecasts.to_csv(tidy_path, index=False)

    df_region = df_forecasts[df_forecasts["level"] == "region"].copy()
    if not df_region.empty:
        pivot_region = df_region.pivot_table(
            index=["region", "approach", "target"],
            columns="year",
            values="value",
        )
        pivot_region_path = os.path.join(REPORTS_DIR, "forecasts_region_wide.csv")
        pivot_region.to_csv(pivot_region_path)

    df_country = df_forecasts[df_forecasts["level"] == "country"].copy()
    if not df_country.empty:
        pivot_country = df_country.pivot_table(
            index=["region", "approach", "target"],
            columns="year",
            values="value",
        )
        pivot_country_path = os.path.join(REPORTS_DIR, "forecasts_country_wide.csv")
        pivot_country.to_csv(pivot_country_path)


def plot_country_forecasts(
    df_country_hist: pd.DataFrame,
    df_forecasts: pd.DataFrame,
) -> None:
    for target_col in ["marriage", "divorce"]:
        fig, ax = plt.subplots(figsize=(10, 5))

        df_h = df_country_hist.sort_values("year")
        ax.plot(
            df_h["year"],
            df_h[target_col],
            marker="o",
            label=f"Historical {target_col}",
        )

        for approach in ["bottom_up", "top_down"]:
            mask = (
                (df_forecasts["level"] == "country")
                & (df_forecasts["approach"] == approach)
                & (df_forecasts["target"] == target_col)
            )
            df_a = df_forecasts[mask].sort_values("year")
            if df_a.empty:
                continue
            ax.plot(
                df_a["year"],
                df_a["value"],
                marker="o",
                linestyle="--",
                label=f"{approach} forecast ({target_col})",
            )

        ax.set_title(f"Країна: історія та прогнози ({target_col})")
        ax.set_xlabel("Рік")
        ax.set_ylabel("Кількість")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(FIGURES_DIR, f"country_forecast_{target_col}.png")
        fig.savefig(out_path)
        plt.close(fig)


def plot_region_forecasts(
    df_regions: pd.DataFrame,
    df_forecasts: pd.DataFrame,
) -> None:
    regions = sorted(df_regions["region"].unique())
    for region in regions:
        df_hist_r = df_regions[df_regions["region"] == region].sort_values("year")
        for target_col in ["marriage", "divorce"]:
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(
                df_hist_r["year"],
                df_hist_r[target_col],
                marker="o",
                label=f"Historical {target_col}",
            )

            for approach in ["bottom_up", "top_down"]:
                mask = (
                    (df_forecasts["level"] == "region")
                    & (df_forecasts["approach"] == approach)
                    & (df_forecasts["region"] == region)
                    & (df_forecasts["target"] == target_col)
                )
                df_a = df_forecasts[mask].sort_values("year")
                if df_a.empty:
                    continue
                ax.plot(
                    df_a["year"],
                    df_a["value"],
                    marker="o",
                    linestyle="--",
                    label=f"{approach} forecast ({target_col})",
                )

            ax.set_title(f"Регіон: {region} – {target_col}")
            ax.set_xlabel("Рік")
            ax.set_ylabel("Кількість")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            safe_region = (
                str(region).replace("/", "_").replace("\\", "_").replace(" ", "_")
            )
            out_path = os.path.join(
                FIGURES_DIR, f"region_{safe_region}_forecast_{target_col}.png"
            )
            fig.savefig(out_path)
            plt.close(fig)


# ==============================
# MAIN
# ==============================


def main():
    ensure_dirs()

    data_path = os.path.join(DATA_DIR, DATA_FILE)
    df_raw = load_dataset(data_path)

    df_all = preprocess_dataset(df_raw)
    df_regions, df_country = split_region_country(df_all)

    exploratory_analysis(df_regions, df_country)

    df_forecasts = generate_all_forecasts(
        df_regions=df_regions,
        df_country=df_country,
        horizon=FORECAST_HORIZON,
    )

    save_forecast_tables(df_forecasts)
    plot_country_forecasts(df_country_hist=df_country, df_forecasts=df_forecasts)
    plot_region_forecasts(df_regions=df_regions, df_forecasts=df_forecasts)

    print("\nГотово. Таблиці у 'reports', графіки у 'figures'.")


if __name__ == "__main__":
    main()
