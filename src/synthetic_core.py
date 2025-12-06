#syntetic_core.py
import numpy as np

from src.config import TREND_CFG, NOISE_CFG


def generate_trend(x: np.ndarray, trend_type: str) -> np.ndarray:
    params = TREND_CFG[trend_type]
    if trend_type == "linear":
        return params["a"] * x + params["b"]
    if trend_type == "cubic":
        return params["a"] * x**3 + params["b"] * x**2 + params["c"] * x + params["d"]
    raise ValueError("Unsupported trend type")


def generate_noise(noise_type: str, n: int) -> np.ndarray:
    params = NOISE_CFG[noise_type]
    if noise_type == "normal":
        return np.random.normal(params["mu"], params["sigma"], n)
    if noise_type == "uniform":
        return np.random.uniform(-10, 10, n)
    raise ValueError("Unsupported noise type")


def inject_anomalies(arr: np.ndarray, percentage: float) -> np.ndarray:
    arr = arr.copy()
    k = int(len(arr) * percentage)
    if k <= 0:
        return arr
    idx = np.random.choice(len(arr), k, replace=False)
    factors = np.random.choice([-4, -3, 3, 4], size=k)
    arr[idx] *= factors
    return arr
