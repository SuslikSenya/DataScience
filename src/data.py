import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from scipy.stats import entropy


@dataclass
class Config:
    """
    Main config dataclass with default values.
    """

    # ===== Synthetic data =====
    n_train: int = 500
    n_test: int = 500
    x_range_train: Tuple[float, float] = (1.0, 50)
    x_range_test: Tuple[float, float] = (51, 100.0)
    add_anomalies: bool = True
    anomaly_percentage: float = 15.0
    trend_cfg: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "linear": {"a": 1.5, "b": 10.0},
            "cubic": {"a": 0.02, "b": 0.2, "c": 5.0, "d": 2.0},
        }
    )
    noise_cfg: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "normal": {"mu": 0.0, "sigma": 2.0},
            "uniform": {},
        }
    )
    trend_type: str = "cubic"
    noise_type: str = "normal"
    anomaly_window: int = 3
    anomaly_threshold: float = 1.0

    # ===== Real data =====
    save_path: str = "multi_currency_data.csv"
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-01"
    train_ratio: float = 0.5
    currencies: List[str] = field(default_factory=lambda: ["USD", "EUR", "RUB"])

    # ===== All models =====
    models: Dict[str, object] = field(default_factory=dict)

    # ===== Reports/plots =====
    save_plot_path: str = "plots"
    save_report_path: str = "report"


class BaseGenerator:
    def __init__(self, config: Config):
        self.config = config


class TrendGenerator(BaseGenerator):
    def generate(self, x: np.ndarray, trend_type: str) -> np.ndarray:
        """Trend generator function"""
        params = self.config.trend_cfg[trend_type]
        if trend_type == "linear":
            return params["a"] * x + params["b"]
        elif trend_type == "cubic":
            return params["a"] * x ** 3 + params["b"] * x ** 2 + params["c"] * x + params["d"]
        raise ValueError("Unknown trend")


# !=============================================================================
# ! Noise Generator
# !=============================================================================


class NoiseGenerator(BaseGenerator):
    def generate(self,
                 noise_type: str,
                 n: int,
                 add_anomalies: bool) -> np.ndarray:
        """Noise generator function"""
        params = self.config.noise_cfg[noise_type]
        if noise_type == "normal":
            noise = np.random.normal(params["mu"], params["sigma"], n)
        elif noise_type == "uniform":
            noise = np.random.uniform(low=-10, high=10, size=n)
        else:
            raise ValueError("Unknown noise")

        return self.add_anomalies(noise) if add_anomalies else noise

    def add_anomalies(self,
                      noise: np.ndarray) -> np.ndarray:
        """NOISE IMPLEMENTOR FUNCTION"""
        noise = noise.copy()
        k = int(len(noise) * self.config.anomaly_percentage / 100)
        idx = np.random.choice(len(noise), k, replace=False)
        factors = np.random.choice([-4, -3, 3, 4], size=k)
        noise[idx] = noise[idx] * factors
        return noise


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
