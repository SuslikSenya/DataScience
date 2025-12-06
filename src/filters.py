#filters.py

import numpy as np
from scipy.stats import entropy


class EntropyAnomalyDetector:
    def __init__(self, window: int, base_k: float, alpha: float = 1.5, bins: int = 20):
        self.window = window
        self.base_k = base_k
        self.alpha = alpha
        self.bins = bins
        self.total_anomalies = 0
        self.avg_entropy = 0.0
        self.avg_k_local = 0.0

    @staticmethod
    def _entropy(arr: np.ndarray, bins: int) -> float:
        hist, _ = np.histogram(arr, bins=bins, density=True)
        hist = hist + 1e-12
        return float(entropy(hist))

    def clean(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=float)
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
    return np.asarray(out, dtype=float)
