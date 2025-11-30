import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


class ModelMetrics:
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        return {
            "mse": float(np.mean((y_true - y_pred) ** 2)),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "r2": float("nan") if denom == 0 else float(1 - np.sum((y_true - y_pred) ** 2) / denom)
        }


class DataMetrics:
    @staticmethod
    def calculate(arr: np.ndarray, name: str) -> Dict[str, float]:
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_min": float(np.min(arr)),
            f"{name}_max": float(np.max(arr))
        }


class Plot:
    @staticmethod
    def plot_data(x_train,
                  y_train,
                  trend_train,
                  predictions_train,
                  x_test=None,
                  predictions_test=None,
                  trend_test=None,
                  fname: str = None, ):
        """2D plotiing function"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        ax1.plot(x_train, y_train, label="data", alpha=0.6)
        ax1.plot(x_train, trend_train, label="trend", linestyle="--")
        ax1.plot(x_train, predictions_train, label="train pred")
        ax1.set_title("Train")
        ax1.legend()

        if x_test is not None:
            ax2.plot(x_test, predictions_test, label="test pred")
            if trend_test is not None:
                ax2.plot(x_test, trend_test, linestyle="--", label="trend")
            ax2.legend()
            ax2.set_title("Test")

        plt.tight_layout()
        plt.savefig(f"{fname}_data")

    @staticmethod
    def plot_hist(fname: str,
                  arr: np.ndarray,
                  n: int = 50):
        """Histogram plotting function"""
        plt.figure(figsize=(10, 7))
        plt.hist(arr, bins=n, color="b")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid()
        plt.savefig(f"{fname}_hist")
        plt.show()
        plt.close()
