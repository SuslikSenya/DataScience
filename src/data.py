import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Config:
    """ Config """
    n_train: int = 500
    n_test: int = 200
    x_range_train: Tuple[float, float] = (1.0, 45.0)
    x_range_test: Tuple[float, float] = (46.0, 50.0)
    anomaly_percentage: float = 15.0

    random_seed: int = 7

    trend_config: Dict[str, Dict[str, float]] = None
    noise_config: Dict[str, Dict[str, float]] = None

    save_path: str = "multi_currency_data.csv"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    train_ratio: float = 0.8
    is_linear: bool = True


class BaseGenerator:
    def __init__(self,
                 config: Config):
        self.config = config


class TrendGenerator(BaseGenerator):
    def generate(self,
                 x: np.ndarray,
                 trend_type: str) -> np.ndarray:
        """Trend generator function"""
        params = self.config.trend_config[trend_type]
        if trend_type == "linear":
            return params["a"] * x + params["b"]
        elif trend_type == "cubic":
            return params["a"] * x ** 3 + params["b"] * x ** 2 + params["c"] * x + params["d"]
        raise ValueError("Unknown trend")


class NoiseGenerator(BaseGenerator):
    def generate(self,
                 noise_type: str,
                 n: int,
                 add_anomalies: bool) -> np.ndarray:
        """Noise generator function"""
        params = self.config.noise_config[noise_type]
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
