import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config


# !=============================================================================
# ! Logging
# !=============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# !=============================================================================
# ! Trend Generator
# !=============================================================================


class TrendGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, x: np.ndarray, trend_type: str) -> np.ndarray:
        params = self.cfg.trend_cfg[trend_type]

        if trend_type == "linear":
            return params["a"] * x + params["b"]

        if trend_type == "quadratic":
            return params["a"] * x**2 + params["b"] * x + params["c"]

        raise ValueError(f"Unsupported trend type: {trend_type}")


# !=============================================================================
# ! Noise Generator
# !=============================================================================


class NoiseGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, noise_type: str, n: int, add_anomalies: bool) -> np.ndarray:
        params = self.cfg.noise_cfg[noise_type]

        if noise_type == "normal":
            noise = np.random.normal(params["mu"], params["sigma"], n)

        elif noise_type == "exponential":
            noise = np.random.exponential(1.0 / params["rate"], n)

        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        return self._inject_anomalies(noise) if add_anomalies else noise

    def _inject_anomalies(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        k = int(len(arr) * self.cfg.anomaly_percentage / 100)

        if k == 0:
            return arr

        idx = np.random.choice(len(arr), k, replace=False)
        factors = np.random.choice([-4, -3, 3, 4], size=k)
        arr[idx] *= factors
        return arr


# !=============================================================================
# ! Entropy-Based Anomaly Detector
# !=============================================================================


class EntropyAnomalyDetector:
    def __init__(self, window: int, base_k: float, alpha: float = 1.5, bins: int = 20):
        self.window = window
        self.base_k = base_k
        self.alpha = alpha
        self.bins = bins

        self.total_anomalies: int = 0
        self.avg_entropy: float = 0.0
        self.avg_k_local: float = 0.0

    @staticmethod
    def _entropy(arr: np.ndarray, bins: int) -> float:
        hist, _ = np.histogram(arr, bins=bins, density=True)
        hist = hist + 1e-12
        return entropy(hist)

    def clean(self, arr: np.ndarray):
        arr = arr.copy()
        n = len(arr)

        clean = arr.copy()
        anomalies = np.zeros(n, dtype=bool)

        H_global = self._entropy(arr, self.bins) + 1e-12
        med_global = np.median(arr)
        mad_global = np.median(np.abs(arr - med_global)) + 1e-12

        hard_thr = self.base_k * mad_global

        entropy_list = []
        k_list = []

        for i in range(n):
            left = max(0, i - self.window)
            right = min(n, i + self.window + 1)
            window_vals = arr[left:right]

            H_local = self._entropy(window_vals, self.bins)
            entropy_list.append(H_local)

            diff = (H_local - H_global) / H_global

            k_local = self.base_k * (1.0 - self.alpha * diff)

            k_local = float(np.clip(k_local, 0.3 * self.base_k, 1.2 * self.base_k))
            k_list.append(k_local)

            med = np.median(window_vals)
            mad = np.median(np.abs(window_vals - med)) + 1e-12

            dev_local = abs(arr[i] - med)
            dev_global = abs(arr[i] - med_global)

            if (dev_local > k_local * mad) or (dev_global > hard_thr):
                clean[i] = med
                anomalies[i] = True

        self.total_anomalies = int(anomalies.sum())
        self.avg_entropy = float(np.mean(entropy_list))
        self.avg_k_local = float(np.mean(k_list))

        return clean, self.total_anomalies


# !=============================================================================
# ! Filters
# !=============================================================================


class AlphaBetaFilter:
    def __init__(self, alpha: float, beta: float, dt: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.dt = dt

        self.x = None
        self.v = None

    def initialize(self, x0: float):
        self.x = x0
        self.v = 0.0

    def update(self, z: float) -> float:
        if self.x is None:
            self.initialize(z)
            return z

        x_pred = self.x + self.v * self.dt
        v_pred = self.v

        e = z - x_pred

        self.x = x_pred + self.alpha * e
        self.v = v_pred + (self.beta * e) / self.dt

        return self.x


class AlphaBetaGammaFilter:
    def __init__(self, alpha: float, beta: float, gamma: float, dt: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt = dt

        self.x = None
        self.v = None
        self.a = None

    def initialize(self, x0: float):
        self.x = x0
        self.v = 0.0
        self.a = 0.0

    def update(self, z: float) -> float:
        if self.x is None:
            self.initialize(z)
            return z

        dt = self.dt

        x_pred = self.x + self.v * dt + 0.5 * self.a * dt * dt
        v_pred = self.v + self.a * dt
        a_pred = self.a

        e = z - x_pred

        self.x = x_pred + self.alpha * e
        self.v = v_pred + (self.beta * e) / dt
        self.a = a_pred + (self.gamma * e) / (0.5 * dt * dt)

        return self.x


class AdaptiveAlphaBetaGammaFilter:
    def __init__(self, alpha: float, beta: float, gamma: float, dt: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt = dt

        self.x = None
        self.v = None
        self.a = None

        self.innov_history = []

    def initialize(self, x0: float):
        self.x = x0
        self.v = 0.0
        self.a = 0.0

    def _adapt_parameters(self, e: float):
        self.innov_history.append(e)
        if len(self.innov_history) > 50:
            self.innov_history.pop(0)

        std_e = np.std(self.innov_history) + 1e-6

        if abs(e) > 2 * std_e:
            self.alpha *= 1.05
            self.beta *= 1.05
            self.gamma *= 1.05
        else:
            self.alpha *= 0.995
            self.beta *= 0.995
            self.gamma *= 0.995

        self.alpha = float(np.clip(self.alpha, 0.01, 1.0))
        self.beta = float(np.clip(self.beta, 0.001, 1.0))
        self.gamma = float(np.clip(self.gamma, 0.0001, 1.0))

    def update(self, z: float) -> float:
        if self.x is None:
            self.initialize(z)
            return z

        dt = self.dt

        x_pred = self.x + self.v * dt + 0.5 * self.a * dt * dt
        v_pred = self.v + self.a * dt
        a_pred = self.a

        e = z - x_pred

        self._adapt_parameters(e)

        self.x = x_pred + self.alpha * e
        self.v = v_pred + self.beta * e / dt
        self.a = a_pred + self.gamma * e / (0.5 * dt * dt)

        return self.x


def run_filter_series(flt, y: np.ndarray) -> np.ndarray:
    out = []
    for z in y:
        out.append(flt.update(float(z)))
    return np.asarray(out)


# !=============================================================================
# ! Data Scraper: NBU
# !=============================================================================


class NBUScraper:
    def __init__(self, currencies: List[str]):
        self.currencies = currencies
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
            logger.error("NBU fetch error: %s", e)
            return None

    def fetch_range(self, start: str, end: str) -> pd.DataFrame:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        records = []
        dt = start_dt

        for _ in tqdm(range((end_dt - start_dt).days + 1)):
            date_str = dt.strftime("%Y%m%d")
            row = {"date": dt.date()}

            for cur in self.currencies:
                row[cur] = self._fetch_single(cur, date_str)

            records.append(row)
            dt += timedelta(days=1)

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df


# !=============================================================================
# ! Data Loader
# !=============================================================================


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def save(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        df.to_csv(self.path, index=False)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, parse_dates=["date"])

    def split(self, df: pd.DataFrame, ratio: float):
        n = len(df)
        cut = int(ratio * n)
        return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(
            drop=True
        )


# !=============================================================================
# ! Models
# !=============================================================================


class SklearnModel:
    def __init__(self, trend_type: str):
        self.degree = 1 if trend_type == "linear" else 2
        self.poly = PolynomialFeatures(self.degree, include_bias=False)
        self.model = LinearRegression()

    def fit(self, x, y):
        X = self.poly.fit_transform(x.reshape(-1, 1))
        self.model.fit(X, y)

    def predict(self, x):
        X = self.poly.transform(x.reshape(-1, 1))
        return self.model.predict(X)


class TorchNNRegressor(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TorchNNModel:
    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 500,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = device

        self.model = TorchNNRegressor(
            input_dim=1, hidden_dim=self.hidden_dim, output_dim=1
        ).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x, y):
        self.model.train()
        x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(x_t)
            loss = self.loss_fn(y_pred, y_t)
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        self.model.eval()
        x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(self.device)
        with torch.no_grad():
            y_pred = self.model(x_t).cpu().numpy().reshape(-1)
        return y_pred


# !=============================================================================
# ! Metrics
# !=============================================================================


class ModelMetrics:
    @staticmethod
    def compute(y_true, y_pred) -> Dict[str, float]:
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = (
            float(1 - np.sum((y_true - y_pred) ** 2) / denom)
            if denom > 0
            else float("nan")
        )
        return {"mse": mse, "mae": mae, "r2": r2}


# !=============================================================================
# ! Reporting
# !=============================================================================


class ReportGenerator:
    @staticmethod
    def save_data_report(metrics: Dict[str, float], path: str):
        df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Data report saved: %s", path)

    @staticmethod
    def save_model_report(reports: List[dict], path: str):
        rows = []
        for rep in reports:
            model = rep["model"]
            for phase in ["train_metrics", "test_metrics"]:
                if phase not in rep:
                    continue
                dataset = "train" if phase == "train_metrics" else "test"
                for k, v in rep[phase].items():
                    rows.append(
                        {"model": model, "dataset": dataset, "metric": k, "value": v}
                    )

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Model report saved: %s", path)

    @staticmethod
    def save_anomaly_report(detector: EntropyAnomalyDetector, path: str):
        df = pd.DataFrame(
            [
                {"metric": "total_anomalies", "value": detector.total_anomalies},
                {"metric": "avg_entropy", "value": detector.avg_entropy},
                {"metric": "avg_k_local", "value": detector.avg_k_local},
            ]
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Anomaly report saved: %s", path)

    @staticmethod
    def save_filter_report(reports: List[dict], path: str):
        rows = []
        for rep in reports:
            flt = rep["filter"]
            dataset = rep.get("dataset", "")
            for k, v in rep["metrics"].items():
                rows.append(
                    {"filter": flt, "dataset": dataset, "metric": k, "value": v}
                )
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Filter report saved: %s", path)


# !=============================================================================
# ! Plotting
# !=============================================================================


class Plot:
    @staticmethod
    def plot_data(
        x_train,
        y_train,
        trend_train,
        pred_train,
        x_test,
        pred_test,
        trend_test,
        filename,
    ):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(x_train, y_train, label="train data", linewidth=1)
        plt.plot(x_train, trend_train, label="train trend", linestyle="--")
        plt.plot(x_train, pred_train, label="train pred")

        split_x = x_train.max()
        plt.axvline(x=split_x, color="black", linewidth=1.0)
        plt.text(split_x, plt.ylim()[1], "split", ha="right", va="top")

        plt.plot(x_test, pred_test, label="test pred")
        plt.plot(x_test, trend_test, label="test trend", linestyle="--")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_noise(noise_raw, noise_clean, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        x = np.arange(len(noise_raw))
        plt.figure(figsize=(12, 4))
        plt.plot(x, noise_raw, label="raw noise", alpha=0.7)
        plt.plot(x, noise_clean, label="clean noise", alpha=0.7)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_best_filter_synthetic(
        x,
        y_noisy,
        trend,
        best_name: str,
        best_arr: np.ndarray,
        filename: str,
    ):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(x, y_noisy, label="noisy train", alpha=0.5)
        plt.plot(x, trend, label="true trend", linestyle="--")
        plt.plot(x, best_arr, label=f"best filter: {best_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_best_filter_real(
        x,
        y,
        best_name: str,
        best_arr: np.ndarray,
        filename: str,
    ):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label="raw series", alpha=0.5)
        plt.plot(x, best_arr, label=f"best filter: {best_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


# !=============================================================================
# ! Synthetic Pipeline
# !=============================================================================


def pipeline_synthetic(cfg: Config):
    logger.info("Starting synthetic pipeline.")

    trend_gen = TrendGenerator(cfg)
    noise_gen = NoiseGenerator(cfg)

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

        plot_path = os.path.join(cfg.save_plot_path, name, "synthetic.png")
        Plot.plot_data(
            x_train,
            y_train,
            trend_train,
            pred_train,
            x_test,
            pred_test,
            trend_test,
            filename=plot_path,
        )

    ReportGenerator.save_model_report(
        reports, os.path.join(cfg.save_report_path, "synthetic_model_report.csv")
    )


# !=============================================================================
# ! Real Data Pipeline
# !=============================================================================


def pipeline_real(cfg: Config):
    logger.info("Starting real-data pipeline.")

    scraper = NBUScraper(cfg.currencies)
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
                filename=plot_path,
            )

    ReportGenerator.save_model_report(
        reports, os.path.join(cfg.save_report_path, "real_model_report.csv")
    )
    ReportGenerator.save_filter_report(
        filter_reports, os.path.join(cfg.save_report_path, "real_filter_report.csv")
    )


# !=============================================================================
# ! Main
# !=============================================================================


def main():
    cfg = Config()

    cfg.models["Sklearn"] = SklearnModel(cfg.trend_type)
    cfg.models["TorchNN"] = TorchNNModel(
        hidden_dim=64,
        lr=1e-3,
        epochs=1000,
        device="cpu",
    )

    os.makedirs(cfg.save_report_path, exist_ok=True)
    os.makedirs(cfg.save_plot_path, exist_ok=True)

    print("Choose pipeline:")
    print("1 - Synthetic")
    print("2 - Real")
    print("None - both")

    ch = input("> ").strip()
    if ch == "1":
        pipeline_synthetic(cfg)
    elif ch == "2":
        pipeline_real(cfg)
    elif ch == "":
        pipeline_synthetic(cfg)
        pipeline_real(cfg)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
