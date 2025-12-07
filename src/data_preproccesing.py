import numpy as np

from src.models import BaseTSModel


def normalize_series(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std <= 1e-8:
        std = 1.0
    y_norm = (y - mean) / std
    return y_norm, mean, std


def denormalize_series(y_norm: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(y_norm, dtype=float) * std + mean


def make_windows(y_norm: np.ndarray, window: int):
    y_norm = np.asarray(y_norm, dtype=float)
    X = []
    targets = []
    idx = []
    for t in range(window, len(y_norm)):
        X.append(y_norm[t - window: t])
        targets.append(y_norm[t])
        idx.append(t)
    if not X:
        return np.zeros((0, window)), np.zeros((0,)), np.zeros((0,), dtype=int)
    return (
        np.asarray(X, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(idx, dtype=int),
    )


def rollout_forecast(
        model: BaseTSModel,
        y_history: np.ndarray,
        mean: float,
        std: float,
        window: int,
        n_steps: int,
) -> np.ndarray:
    y_history = np.asarray(y_history, dtype=float)
    if len(y_history) < window:
        raise ValueError("Not enough history for given window size")
    y_norm = (y_history - mean) / std
    window_norm = y_norm[-window:].copy()
    preds_norm = []
    for _ in range(n_steps):
        x_win = window_norm.reshape(1, -1)
        y_next_norm = float(model.predict(x_win)[0])
        preds_norm.append(y_next_norm)
        window_norm[:-1] = window_norm[1:]
        window_norm[-1] = y_next_norm
    preds_norm = np.asarray(preds_norm, dtype=float)
    return denormalize_series(preds_norm, mean, std)
