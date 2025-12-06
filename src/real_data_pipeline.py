import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
import os

from requests.adapters import HTTPAdapter
from urllib3 import Retry

from .data import Config, EntropyAnomalyDetector
from .filters import AdaptiveAlphaBetaGammaFilter, AlphaBetaGammaFilter, AlphaBetaFilter, run_filter_series
from .metrics_plot import ModelMetrics, Plot
from .reports import ReportGenerator


class NBUScrapper:
    def __init__(self, currencies: List[str] = None):
        self.currencies = currencies or ["USD", "EUR", "RUB"]
        self.session = requests.Session()

    def _fetch_single(self, currency: str, date_str: str) -> Optional[float]:
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

        url = (
            f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
            f"?valcode={currency}&date={date_str}&json"
        )

        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return float(data[0]["rate"]) if data else None

        except Exception as e:
            print("NBU fetch error: %s", e)
            return None

    def fetch_range(self, start: str, end: str) -> pd.DataFrame:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        records = []
        dt = start_dt

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df


class DataLoader:
    def __init__(self, save_path: str):
        self.save_path = save_path

    def save(self, df: pd.DataFrame):
        df.to_csv(self.save_path, index=False)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.save_path, parse_dates=["date"])

    def split(self, df: pd.DataFrame, ratio: float):
        n = len(df)
        cut = int(n * ratio)
        return df[:cut], df[cut:]


# !=============================================================================
# ! Real Data Pipeline
# !=============================================================================


def pipeline_real(cfg: Config):
    scraper = NBUScrapper(cfg.currencies)
    loader = DataLoader(cfg.save_path)

    if not os.path.exists(cfg.save_path):
        df = scraper.fetch_range(cfg.start_date, cfg.end_date)
        loader.save(df)

    df = loader.load()
    df = df.dropna(subset=cfg.currencies)
    train, test = loader.split(df, cfg.train_ratio)

    x_train = np.arange(len(train), dtype=float)
    x_test = np.arange(len(train), len(df), dtype=float)

    reports = []
    filter_reports = []

    for cur in cfg.currencies:
        y_train_raw = train[cur].astype(float).values
        y_test = test[cur].astype(float).values

        detector = EntropyAnomalyDetector(
            window=cfg.anomaly_window,
            base_k=cfg.anomaly_threshold,
            alpha=1.2,
            bins=30,
        )
        y_train_clean, anomalies_count = detector.clean(y_train_raw)

        ReportGenerator.save_anomaly_report(
            detector,
            os.path.join(
                cfg.save_report_path,
                f"real_anomaly_report_{cur}.csv",
            ),
        )

        filters = {
            "AB": AlphaBetaFilter(cfg.ab_alpha, cfg.ab_beta, cfg.dt),
            "ABG": AlphaBetaGammaFilter(
                cfg.abg_alpha, cfg.abg_beta, cfg.abg_gamma, cfg.dt
            ),
            "ABG_adaptive": AdaptiveAlphaBetaGammaFilter(
                cfg.abg_alpha, cfg.abg_beta, cfg.abg_gamma, cfg.dt
            ),
        }

        filtered_outputs = {}
        filter_reports_cur = []

        for name, flt in filters.items():
            y_filt = run_filter_series(flt, y_train_clean)
            filtered_outputs[name] = y_filt

            metrics = ModelMetrics.compute(y_train_clean, y_filt)
            metrics["bias"] = float(np.mean(y_train_clean - y_filt))
            metrics["residual_std"] = float(np.std(y_train_clean - y_filt))

            record = {
                "filter": name,
                "dataset": f"real_{cur}_train",
                "metrics": metrics,
            }
            filter_reports.append(record)
            filter_reports_cur.append(record)

        best_report = min(
            filter_reports_cur,
            key=lambda r: r["metrics"]["residual_std"],
        )
        best_filter_name = best_report["filter"]
        best_filter_output = filtered_outputs[best_filter_name]

        Plot.plot_best_filter_real(
            x_train,
            y_train_clean,
            best_filter_name,
            best_filter_output,
            os.path.join(
                cfg.save_plot_path,
                "filters",
                f"{cur}_{best_filter_name}_real.png",
            ),
        )

        for name, model in cfg.models.items():
            model.fit(x_train, y_train_clean)

            pred_train = model.predict(x_train)
            pred_test = model.predict(x_test)

            train_metrics = ModelMetrics.compute(y_train_clean, pred_train)
            test_metrics = ModelMetrics.compute(y_test, pred_test)

            rep = {
                "model": f"{cur}_{name}",
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
            reports.append(rep)

            plot_path = os.path.join(
                cfg.save_plot_path,
                name,
                f"{cur}_real.png",
            )
            Plot.plot_data(
                x_train,
                y_train_clean,
                y_train_clean,
                pred_train,
                x_test,
                pred_test,
                y_test,
                fname=plot_path,
            )

    ReportGenerator.save_model_report(
        reports, os.path.join(cfg.save_report_path, "real_model_report.csv")
    )
    ReportGenerator.save_filter_report(
        filter_reports, os.path.join(cfg.save_report_path, "real_filter_report.csv")
    )
