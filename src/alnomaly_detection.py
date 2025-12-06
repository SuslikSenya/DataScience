import numpy as np
from scipy.stats import entropy


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

