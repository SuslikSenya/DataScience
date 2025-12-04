import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
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
    """
    Generates deterministic trend components for synthetic time series
    using linear or quadratic mathematical models.
    """

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
    """
    Produces synthetic noise of different statistical distributions
    and optionally injects controlled anomalies into the dataset.
    """

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
        """
        Multiplies random entries by large factors to simulate outliers.
        """
        arr = arr.copy()
        k = int(len(arr) * self.cfg.anomaly_percentage / 100)

        if k == 0:
            return arr

        idx = np.random.choice(len(arr), k, replace=False)
        factors = np.random.choice([-4, -3, 3, 4], size=k)
        arr[idx] *= factors
        return arr


# !=============================================================================
# ! Entropy-Based R&D Anomaly Detector
# !=============================================================================


class EntropyAnomalyDetector:
    """
    Detects and corrects anomalies in time series using a custom R&D method
    based on local Shannon entropy and adaptive statistical thresholds.
    """

    def __init__(self, window: int, base_k: float, alpha: float = 1.2, bins: int = 20):
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

    def clean(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        arr = arr.copy()
        n = len(arr)

        clean = arr.copy()
        anomalies = np.zeros(n, dtype=bool)

        H_global = self._entropy(arr, self.bins)

        entropy_list = []
        k_list = []

        for i in range(n):
            left = max(0, i - self.window)
            right = min(n, i + self.window + 1)
            window_vals = arr[left:right]

            H_local = self._entropy(window_vals, self.bins)
            entropy_list.append(H_local)

            k_local = self.base_k * (1 + self.alpha * (H_local / (H_global + 1e-12)))
            k_list.append(k_local)

            med = np.median(window_vals)
            mad = np.median(np.abs(window_vals - med)) + 1e-12

            if abs(arr[i] - med) > k_local * mad:
                clean[i] = med
                anomalies[i] = True

        self.total_anomalies = int(anomalies.sum())
        self.avg_entropy = float(np.mean(entropy_list))
        self.avg_k_local = float(np.mean(k_list))

        return clean, self.total_anomalies


# !=============================================================================
# ! Data Scraper: NBU
# !=============================================================================


class NBUScraper:
    """
    Downloads real-world currency exchange rate time series
    from the National Bank of Ukraine API with retry and error handling.
    """

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
    """
    Handles saving, loading, and splitting time series datasets
    for both synthetic and real-data processing pipelines.
    """

    def __init__(self, path: str):
        self.path = path

    def save(self, df: pd.DataFrame):
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


class LSMModel:
    """
    Implements a classical polynomial regression model
    trained via the least-squares method (LSM).
    """

    def __init__(self, trend_type: str):
        self.degree = 1 if trend_type == "linear" else 2
        self.coef = None

    def fit(self, x, y):
        self.coef = np.polyfit(x, y, self.degree)

    def predict(self, x):
        return np.polyval(self.coef, x)


class SklearnModel:
    """
    Provides polynomial regression using sklearn tools
    for comparison with manual LSM implementations.
    """

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


# !=============================================================================
# ! Metrics
# !=============================================================================


class ModelMetrics:
    """
    Computes key evaluation metrics for regression models
    including MSE, MAE, and determination coefficient R².
    """

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
    """
    Generates structured CSV-based reports for data metrics,
    model performance metrics, and anomaly detection statistics.
    """

    @staticmethod
    def save_data_report(metrics: Dict[str, float], path: str):
        df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
        df.to_csv(path, index=False)
        logger.info("Data report saved: %s", path)

    @staticmethod
    def save_model_report(reports: List[dict], path: str):
        rows = []
        for rep in reports:
            model = rep["model"]
            for phase in ["train_metrics", "test_metrics"]:
                dataset = "train" if phase == "train_metrics" else "test"
                for k, v in rep[phase].items():
                    rows.append(
                        {"model": model, "dataset": dataset, "metric": k, "value": v}
                    )

        df = pd.DataFrame(rows)
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
        df.to_csv(path, index=False)
        logger.info("Anomaly report saved: %s", path)


# !=============================================================================
# ! Plotting
# !=============================================================================


class Plot:
    """
    Produces visualization plots for time series data, model predictions,
    and comparative analysis of training and forecasting quality.
    """

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
        plt.figure(figsize=(12, 6))
        plt.plot(x_train, y_train, label="train data")
        plt.plot(x_train, trend_train, label="trend", linestyle="--")
        plt.plot(x_train, pred_train, label="train pred")

        split_x = x_train.max()
        plt.axvline(x=split_x, color="black", linewidth=1.3)
        plt.text(split_x, plt.ylim()[1], "split", ha="right", va="top")

        plt.plot(x_test, pred_test, label="test pred")
        plt.plot(x_test, trend_test, label="test trend", linestyle="--")

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
        detector, f"{cfg.save_report_path}/anomaly_report.csv"
    )

    # Save data metrics
    data_metrics = {
        "noise_mean": float(noise_raw.mean()),
        "noise_std": float(noise_raw.std()),
        "noise_clean_std": float(noise_clean.std()),
        "anomalies_removed": anomalies_count,
    }
    ReportGenerator.save_data_report(
        data_metrics, f"{cfg.save_report_path}/data_report.csv"
    )

    # Fit models
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
            filename=f"{cfg.save_plot_path}/{name}_synthetic.png",
        )

    ReportGenerator.save_model_report(
        reports, f"{cfg.save_report_path}/model_report.csv"
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
    train, test = loader.split(df, cfg.train_ratio)

    x_train = np.arange(len(train), dtype=float)
    x_test = np.arange(len(train), len(df), dtype=float)

    reports = []

    for cur in cfg.currencies:
        y_train = train[cur].astype(float).values
        y_test = test[cur].astype(float).values

        for name, model in cfg.models.items():
            model.fit(x_train, y_train)
            pred_train = model.predict(x_train)
            pred_test = model.predict(x_test)

            rep = {
                "model": f"{cur}_{name}",
                "train_metrics": ModelMetrics.compute(y_train, pred_train),
                "test_metrics": ModelMetrics.compute(y_test, pred_test),
            }

            reports.append(rep)

            Plot.plot_data(
                x_train,
                y_train,
                y_train,
                pred_train,
                x_test,
                pred_test,
                y_test,
                filename=f"{cfg.save_plot_path}/{cur}_{name}_real.png",
            )

    ReportGenerator.save_model_report(
        reports, f"{cfg.save_report_path}/real_model_report.csv"
    )


# !=============================================================================
# ! Main
# !=============================================================================


def main():
    cfg = Config()

    cfg.models["LSM"] = LSMModel(cfg.trend_type)
    cfg.models["Sklearn"] = SklearnModel(cfg.trend_type)

    os.makedirs(cfg.save_report_path, exist_ok=True)
    os.makedirs(cfg.save_plot_path, exist_ok=True)

    print("Choose pipeline:")
    print("1 - Synthetic")
    print("2 - Real")

    ch = input("> ").strip()
    if ch == "1":
        pipeline_synthetic(cfg)
    elif ch == "2":
        pipeline_real(cfg)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
