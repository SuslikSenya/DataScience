import os

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


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
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(x_train, y_train, label="train data", linewidth=1)
        plt.plot(x_train, trend_train, label="train trend", linestyle="--")
        plt.plot(x_train, predictions_train, label="train pred")

        split_x = x_train.max()
        plt.axvline(x=split_x, color="black", linewidth=1.0)
        plt.text(split_x, plt.ylim()[1], "split", ha="right", va="top")

        plt.plot(x_test, predictions_test, label="test pred")
        plt.plot(x_test, trend_test, label="test trend", linestyle="--")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fname)
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

