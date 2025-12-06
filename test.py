import os
from datetime import datetime, timedelta
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from scipy.stats import entropy
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    #! dirs
    DATA_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    #! basic
    RANDOM_STATE,
    N_CLUSTERS,
    #! synthetic
    SYN_N_TRAIN,
    SYN_N_TEST,
    SYN_X_RANGE_TRAIN,
    SYN_X_RANGE_TEST,
    SYN_TREND_TYPE,
    TREND_CFG,
    NOISE_CFG,
    SYN_NOISE_TYPE,
    SYN_ADD_ANOMALIES,
    SYN_ANOMALY_PERCENTAGE,
    TS_DECOMP_MODEL,
    TS_DECOMP_PERIOD_SYN,
    TS_SYNTHETIC_YEARS,
    TS_NOISE_SCALE,
    ANOMALY_WINDOW,
    ANOMALY_BASE_K,
    ANOMALY_ALPHA,
    ANOMALY_BINS,
    AB_ALPHA,
    AB_BETA,
    ABG_ALPHA,
    ABG_BETA,
    ABG_GAMMA,
    DT,
    #! NBU
    NBU_CURRENCIES,
    NBU_START_DATE,
    NBU_END_DATE,
    NBU_TRAIN_RATIO,
    NBU_CSV_PATH,
    TS_DECOMP_PERIOD_NBU,
    #! DataSet_6
    DATASET6_PATH,
    DATASET6_MONTH_COLUMNS,
    DATASET6_REGION_COL,
    LINEAR_TRAIN_RATIO_DS6,
    TS_DECOMP_PERIOD_DS6,
    #! MA / ARIMA / EXTRAPOLATION
    FORECAST_HORIZONS,
    ARIMA_P_RANGE,
    ARIMA_D_RANGE,
    ARIMA_Q_RANGE,
    MA_WINDOW_CANDIDATES,
    DEFAULT_MA_WINDOW,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


device = "cpu"
print(device)


#! ==============================
#!  TS analysis
#! ==============================


def compute_series_features(y: np.ndarray) -> dict:
    s = pd.Series(y)
    mean = float(s.mean())
    std = float(s.std())
    min_v = float(s.min())
    max_v = float(s.max())
    cv = float(std / mean) if mean != 0 else float("nan")
    lag1 = float(s.autocorr(lag=1))
    return {
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "cv": cv,
        "lag1_autocorr": lag1,
    }


def decompose_and_plot(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    title: str,
    fig_path: str,
    model: str,
    period: int,
):
    ts = pd.Series(y, index=dates)

    if len(ts) < 2 * period:
        period = max(2, len(ts) // 2)
        if period < 2:
            print(
                f"[WARN] decompose_and_plot: series too short for decomposition (len={len(ts)})"
            )
            return

    try:
        dec = seasonal_decompose(
            ts, model=model, period=period, extrapolate_trend="freq"
        )
    except ValueError as e:
        print(f"[ERROR] decompose_and_plot failed: {e}")
        return

    fig = dec.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close(fig)


def generate_synthetic_like(
    y: np.ndarray, years: int, noise_scale: float
) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    n = len(values)
    mean_val = float(values.mean())
    seasonal_pattern = values - mean_val
    base_std = float(values.std())
    noise_std = base_std * noise_scale
    total = n * years
    out = []
    for t in range(total):
        m = t % n
        base = mean_val + seasonal_pattern[m]
        noise = np.random.normal(0.0, noise_std)
        out.append(base + noise)
    return np.asarray(out, dtype=float)


def compare_real_vs_synth(y_real: np.ndarray, y_synth: np.ndarray, n_real: int) -> dict:
    real = np.asarray(y_real, dtype=float)
    synth = np.asarray(y_synth[:n_real], dtype=float)
    return {
        "real_mean": float(real.mean()),
        "real_std": float(real.std()),
        "synthetic_mean": float(synth.mean()),
        "synthetic_std": float(synth.std()),
        "corr_real_synth": float(pd.Series(real).corr(pd.Series(synth))),
    }


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom > 0:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    else:
        r2 = float("nan")

    return {"mse": mse, "mae": mae, "r2": r2}


def split_train_val_series(y: np.ndarray, val_ratio: float = 0.2):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y, np.array([], dtype=float)
    cut = max(1, int((1.0 - val_ratio) * n))
    return y[:cut], y[cut:]


def select_best_ma_window(y: np.ndarray, candidate_windows, val_ratio: float = 0.2):
    from math import inf

    y_train, y_val = split_train_val_series(y, val_ratio)
    if len(y_val) == 0:
        return DEFAULT_MA_WINDOW, float("nan")

    best_w = None
    best_mse = inf

    for w in candidate_windows:
        if w <= 0:
            continue
        history = np.asarray(y_train, dtype=float)
        preds = []
        hist_list = history.tolist()
        for _ in range(len(y_val)):
            if len(hist_list) < w:
                preds.append(float(np.mean(hist_list)))
            else:
                preds.append(float(np.mean(hist_list[-w:])))
            hist_list.append(preds[-1])
        m = metrics_regression(y_val, np.asarray(preds, dtype=float))
        if m["mse"] < best_mse:
            best_mse = m["mse"]
            best_w = w

    if best_w is None:
        best_w = DEFAULT_MA_WINDOW

    return best_w, best_mse


def select_best_arima_order(
    y: np.ndarray,
    p_range,
    d_range,
    q_range,
    val_ratio: float = 0.2,
):
    from math import inf

    y_train, y_val = split_train_val_series(y, val_ratio)
    best_order = None
    best_aic = inf
    best_mse = inf

    use_full_for_mse = len(y_val) == 0

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(y_train, order=(p, d, q)).fit()
                except Exception:
                    continue

                aic = model.aic
                if use_full_for_mse:
                    forecast = model.predict(start=0, end=len(y_train) - 1)
                    m = metrics_regression(y_train, np.asarray(forecast, dtype=float))
                else:
                    steps = len(y_val)
                    forecast = model.forecast(steps=steps)
                    m = metrics_regression(y_val, np.asarray(forecast, dtype=float))

                mse_val = m["mse"]

                if (aic < best_aic) or (aic == best_aic and mse_val < best_mse):
                    best_aic = aic
                    best_mse = mse_val
                    best_order = (p, d, q)

    return best_order, best_aic, best_mse


def generate_extrapolation_x(x_train: np.ndarray, horizons) -> dict:
    x_train = np.asarray(x_train, dtype=float)
    if len(x_train) < 2:
        return {}

    x_min, x_max = x_train[0], x_train[-1]
    interval = x_max - x_min
    if interval <= 0:
        interval = float(len(x_train))

    out = {}
    for h in horizons:
        n_steps = max(1, int(len(x_train) * float(h)))
        x_future = np.linspace(
            x_max, x_max + float(h) * interval, n_steps, endpoint=False
        )
        out[h] = x_future
    return out


def analyze_matrix(
    data: pd.DataFrame,
    name: str,
    model: str,
    period: int,
    synthetic_years: int,
    noise_scale: float,
    reports_dir: str,
    figures_dir: str,
):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    index = data.index
    cols = data.columns
    try:
        dates = pd.to_datetime(cols)
    except Exception:
        dates = None

    feat_records = []
    cmp_records = []

    for label in index:
        y = data.loc[label].values.astype(float)
        if isinstance(dates, pd.DatetimeIndex):
            dt_index = dates
        else:
            dt_index = pd.date_range("2024-01-01", periods=len(y), freq="D")

        decompose_and_plot(
            y,
            dt_index,
            title=f"{name}: {label}",
            fig_path=os.path.join(figures_dir, f"{name}_decomposition_{label}.png"),
            model=model,
            period=period,
        )

        feats = compute_series_features(y)
        row_f = {"series": label}
        row_f.update(feats)
        feat_records.append(row_f)

        y_synth = generate_synthetic_like(y, synthetic_years, noise_scale)
        cmp = compare_real_vs_synth(y, y_synth, len(y))
        row_c = {"series": label}
        row_c.update(cmp)
        cmp_records.append(row_c)

    if not feat_records:
        return pd.DataFrame(), pd.DataFrame()

    feats_df = pd.DataFrame(feat_records).set_index("series")
    feats_df.to_csv(os.path.join(reports_dir, f"{name}_features.csv"))

    cmp_df = pd.DataFrame(cmp_records).set_index("series")
    cmp_df.to_csv(os.path.join(reports_dir, f"{name}_real_vs_synth.csv"))

    if len(index) >= 2:
        corr = data.T.corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5
        )
        plt.title(f"Correlation matrix: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{name}_correlation.png"))
        plt.close()
        corr.to_csv(os.path.join(reports_dir, f"{name}_correlation.csv"))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feats_df.values)
        kmeans = KMeans(
            n_clusters=min(N_CLUSTERS, len(index)), random_state=RANDOM_STATE, n_init=10
        )
        clusters = kmeans.fit_predict(X_scaled)
        feats_df["cluster"] = clusters
        feats_df.to_csv(os.path.join(reports_dir, f"{name}_features_with_clusters.csv"))
    else:
        feats_df["cluster"] = 0
        feats_df.to_csv(os.path.join(reports_dir, f"{name}_features_with_clusters.csv"))

    return feats_df, cmp_df


#! ==============================
#!  ENTROPY/FILTERS
#! ==============================


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


#! ==============================
#!  MODELS
#! ==============================


class BaseTSModel:
    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SklearnPolyTSModel(BaseTSModel):
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree, include_bias=False)
        self.model = LinearRegression()

    def fit(self, x: np.ndarray, y: np.ndarray):
        X = self.poly.fit_transform(x.reshape(-1, 1))
        self.model.fit(X, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        X = self.poly.transform(x.reshape(-1, 1))
        return self.model.predict(X)


class MovingAverageTSModel(BaseTSModel):
    def __init__(self, window: int = DEFAULT_MA_WINDOW):
        self.window = int(window)
        self.history = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.history = np.asarray(y, dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.history is None:
            return np.zeros_like(x, dtype=float)
        preds = []
        hist = self.history.tolist()
        for _ in range(len(x)):
            if len(hist) < self.window:
                preds.append(float(np.mean(hist)))
            else:
                preds.append(float(np.mean(hist[-self.window :])))
            hist.append(preds[-1])
        return np.asarray(preds, dtype=float)


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


class TorchNNTSModel(BaseTSModel):
    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 500,
        device: str = device,
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

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.train()
        x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(self.device)
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(x_t)
            loss = self.loss_fn(y_pred, y_t)
            loss.backward()
            self.optimizer.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(x_t).cpu().numpy().reshape(-1)
        return y_pred


class Models:
    def __init__(self, device: str = "cpu", ma_window: int = DEFAULT_MA_WINDOW):
        self.models = {
            "MA": MovingAverageTSModel(window=ma_window),
            "Poly2": SklearnPolyTSModel(degree=2),
            "TorchNN": TorchNNTSModel(
                hidden_dim=32, lr=1e-3, epochs=500, device=device
            ),
        }

    def fit_all(self, x: np.ndarray, y: np.ndarray):
        for m in self.models.values():
            m.fit(x, y)

    def predict_all(self, x: np.ndarray) -> dict:
        out = {}
        for name, m in self.models.items():
            out[name] = m.predict(x)
        return out


#! ==============================
#!  GENERATORS: TREND/NOISE/ANOMALIES
#! ==============================


def generate_trend(x: np.ndarray, trend_type: str) -> np.ndarray:
    params = TREND_CFG[trend_type]
    if trend_type == "linear":
        return params["a"] * x + params["b"]
    if trend_type == "quadratic":
        return params["a"] * x**2 + params["b"] * x + params["c"]
    raise ValueError("Unsupported trend type")


def generate_noise(noise_type: str, n: int) -> np.ndarray:
    params = NOISE_CFG[noise_type]
    if noise_type == "normal":
        return np.random.normal(params["mu"], params["sigma"], n)
    if noise_type == "exponential":
        return np.random.exponential(1.0 / params["rate"], n)
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


#! ==============================
#!  PIPELINE: SYNTHETIC
#! ==============================


def pipeline_synthetic():
    x_train = np.linspace(*SYN_X_RANGE_TRAIN, SYN_N_TRAIN)
    x_test = np.linspace(*SYN_X_RANGE_TEST, SYN_N_TEST)

    trend_train = generate_trend(x_train, SYN_TREND_TYPE)
    trend_test = generate_trend(x_test, SYN_TREND_TYPE)

    noise_raw = generate_noise(SYN_NOISE_TYPE, SYN_N_TRAIN)
    if SYN_ADD_ANOMALIES:
        noise_raw = inject_anomalies(noise_raw, SYN_ANOMALY_PERCENTAGE)

    detector = EntropyAnomalyDetector(
        window=ANOMALY_WINDOW,
        base_k=ANOMALY_BASE_K,
        alpha=ANOMALY_ALPHA,
        bins=ANOMALY_BINS,
    )
    noise_clean, anomalies_count = detector.clean(noise_raw)
    y_train = trend_train + noise_clean

    anomaly_report = pd.DataFrame(
        [
            {"metric": "total_anomalies", "value": detector.total_anomalies},
            {"metric": "avg_entropy", "value": detector.avg_entropy},
            {"metric": "avg_k_local", "value": detector.avg_k_local},
        ]
    )
    anomaly_report.to_csv(
        os.path.join(REPORTS_DIR, "synthetic_anomaly_report.csv"), index=False
    )

    noise_stats = pd.DataFrame(
        [
            {"metric": "noise_raw_mean", "value": float(noise_raw.mean())},
            {"metric": "noise_raw_std", "value": float(noise_raw.std())},
            {"metric": "noise_clean_std", "value": float(noise_clean.std())},
            {"metric": "anomalies_removed", "value": anomalies_count},
        ]
    )
    noise_stats.to_csv(
        os.path.join(REPORTS_DIR, "synthetic_noise_report.csv"), index=False
    )

    #! --- ПІДБІР ПАРАМЕТРІВ MA ТА ARIMA ДЛЯ SYNTHETIC ---
    ma_best_window, ma_best_mse = select_best_ma_window(
        y_train, MA_WINDOW_CANDIDATES, val_ratio=0.2
    )
    arima_best_order, arima_best_aic, arima_best_mse = select_best_arima_order(
        y_train, ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE, val_ratio=0.2
    )

    sel_rows = [
        {
            "method": "MA",
            "param": "window",
            "value": ma_best_window,
            "metric": "mse_val",
            "score": ma_best_mse,
        },
        {
            "method": "ARIMA",
            "param": "order",
            "value": str(arima_best_order),
            "metric": "mse_val",
            "score": arima_best_mse,
        },
        {
            "method": "ARIMA",
            "param": "order",
            "value": str(arima_best_order),
            "metric": "aic",
            "score": arima_best_aic,
        },
    ]
    pd.DataFrame(sel_rows).to_csv(
        os.path.join(REPORTS_DIR, "synthetic_ma_arima_selection.csv"), index=False
    )

    filters = {
        "AB": AlphaBetaFilter(AB_ALPHA, AB_BETA, DT),
        "ABG": AlphaBetaGammaFilter(ABG_ALPHA, ABG_BETA, ABG_GAMMA, DT),
        "ABG_adaptive": AdaptiveAlphaBetaGammaFilter(
            ABG_ALPHA, ABG_BETA, ABG_GAMMA, DT
        ),
    }

    filter_rows = []
    filter_outputs = {}
    for name, flt in filters.items():
        y_filt = run_filter_series(flt, y_train)
        filter_outputs[name] = y_filt
        m = metrics_regression(trend_train, y_filt)
        m["bias"] = float(np.mean(trend_train - y_filt))
        for k, v in m.items():
            filter_rows.append(
                {"filter": name, "dataset": "synthetic_train", "metric": k, "value": v}
            )

    df_filter = pd.DataFrame(filter_rows)
    df_filter.to_csv(
        os.path.join(REPORTS_DIR, "synthetic_filter_report.csv"), index=False
    )

    best_filter_name = (
        df_filter[df_filter["metric"] == "mse"].sort_values("value").iloc[0]["filter"]
    )
    best_output = filter_outputs[best_filter_name]

    plt.figure(figsize=(12, 6))
    plt.plot(x_train, y_train, label="train noisy+clean", alpha=0.6)
    plt.plot(x_train, trend_train, label="true trend", linestyle="--")
    plt.plot(x_train, best_output, label=f"best filter: {best_filter_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.join(FIGURES_DIR, "synthetic_filters"), exist_ok=True)
    plt.savefig(
        os.path.join(
            FIGURES_DIR, "synthetic_filters", f"{best_filter_name}_synthetic.png"
        )
    )
    plt.close()

    models = Models(device="cpu", ma_window=ma_best_window)
    models.fit_all(x_train, y_train)

    metrics_rows = []
    for name, m in models.models.items():
        y_pred_train = m.predict(x_train)
        y_pred_test = m.predict(x_test)

        plt.figure(figsize=(12, 6))
        plt.plot(x_train, y_train, label="train noisy", alpha=0.6)
        plt.plot(x_train, trend_train, label="train trend", linestyle="--")
        plt.plot(x_train, y_pred_train, label=f"train pred ({name})", linewidth=2)
        plt.axvline(x_train[-1], color="black", linewidth=1.0)
        plt.plot(x_test, trend_test, label="test trend", linestyle="--")
        plt.plot(x_test, y_pred_test, label=f"test pred ({name})", linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "synthetic_models"), exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "synthetic_models", f"{name}.png"))
        plt.close()

        train_metrics = metrics_regression(trend_train, y_pred_train)
        test_metrics = metrics_regression(trend_test, y_pred_test)
        for dataset, mm in [("train", train_metrics), ("test", test_metrics)]:
            for k, v in mm.items():
                metrics_rows.append(
                    {"model": name, "dataset": dataset, "metric": k, "value": v}
                )

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(
        os.path.join(REPORTS_DIR, "synthetic_model_metrics.csv"), index=False
    )

    x_horizons = generate_extrapolation_x(x_train, FORECAST_HORIZONS)
    extr_dir = os.path.join(REPORTS_DIR, "synthetic_extrapolation")
    os.makedirs(extr_dir, exist_ok=True)

    for model_name, m in models.models.items():
        for h, x_future in x_horizons.items():
            y_future = m.predict(x_future)
            df_extr = pd.DataFrame({"x": x_future, "y_pred": y_future})
            df_extr.to_csv(
                os.path.join(extr_dir, f"{model_name}_h{h}.csv"),
                index=False,
            )

    if arima_best_order is not None:
        try:
            arima_model = ARIMA(y_train, order=arima_best_order).fit()
            arima_fig_dir = os.path.join(FIGURES_DIR, "synthetic_arima")
            os.makedirs(arima_fig_dir, exist_ok=True)

            for h, x_future in x_horizons.items():
                n_steps = len(x_future)
                y_future_arima = arima_model.forecast(steps=n_steps)
                y_future_arima = np.asarray(y_future_arima, dtype=float)

                trend_future = generate_trend(x_future, SYN_TREND_TYPE)

                df_extr = pd.DataFrame(
                    {
                        "step": np.arange(len(y_train), len(y_train) + n_steps),
                        "x": x_future,
                        "y_pred": y_future_arima,
                        "y_true_trend": trend_future,
                    }
                )
                df_extr.to_csv(
                    os.path.join(extr_dir, f"ARIMA_h{h}.csv"),
                    index=False,
                )

                plt.figure(figsize=(12, 6))
                plt.plot(x_train, y_train, label="train noisy", alpha=0.5)
                plt.plot(x_train, trend_train, "--", label="train trend")

                plt.axvline(x_train[-1], color="black", linewidth=1.0)

                plt.plot(x_future, trend_future, "r--", label=f"true trend (h={h})")
                plt.plot(
                    x_future,
                    y_future_arima,
                    label=f"ARIMA forecast (h={h})",
                )

                plt.title(f"ARIMA extrapolation, horizon={h}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(arima_fig_dir, f"arima_extrapolation_h{h}.png")
                )
                plt.close()

        except Exception as e:
            print(f"[WARN] ARIMA extrapolation failed: {e}")

    data = pd.DataFrame(
        {"trend": trend_train, "noisy_clean": y_train},
        index=pd.date_range("2025-01-01", periods=len(y_train), freq="D"),
    ).T

    analyze_matrix(
        data=data,
        name="synthetic_pipeline",
        model=TS_DECOMP_MODEL,
        period=TS_DECOMP_PERIOD_SYN,
        synthetic_years=TS_SYNTHETIC_YEARS,
        noise_scale=TS_NOISE_SCALE,
        reports_dir=os.path.join(REPORTS_DIR, "ts_synthetic"),
        figures_dir=os.path.join(FIGURES_DIR, "ts_synthetic"),
    )

    dates_train = pd.date_range("2025-01-01", periods=len(y_train), freq="D")
    decompose_and_plot(
        y=y_train,
        dates=dates_train,
        title="Synthetic decompose (noisy_clean)",
        fig_path=os.path.join(
            FIGURES_DIR, "synthetic_pipeline_decomposition_noisy_clean.png"
        ),
        model=TS_DECOMP_MODEL,
        period=TS_DECOMP_PERIOD_SYN,
    )


#! ==============================
#!  PIPELINE: REAL NBU
#! ==============================


def fetch_nbu_rates(currencies, start_date: str, end_date: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    records = []
    dt = start_dt
    while dt <= end_dt:
        date_str_api = dt.strftime("%Y%m%d")
        row = {"date": dt.date()}
        for cur in currencies:
            url = (
                f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
                f"?valcode={cur}&date={date_str_api}&json"
            )
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                rate = float(data[0]["rate"]) if data else np.nan
            except Exception:
                rate = np.nan
            row[cur] = rate
        records.append(row)
        dt += timedelta(days=1)
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def pipeline_real_nbu():
    if os.path.exists(NBU_CSV_PATH):
        df = pd.read_csv(NBU_CSV_PATH, parse_dates=["date"])
    else:
        df = fetch_nbu_rates(NBU_CURRENCIES, NBU_START_DATE, NBU_END_DATE)
        os.makedirs(os.path.dirname(NBU_CSV_PATH), exist_ok=True)
        df.to_csv(NBU_CSV_PATH, index=False)

    df = df.dropna(subset=NBU_CURRENCIES)
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    cut = int(NBU_TRAIN_RATIO * n)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)

    x_train = np.arange(len(train), dtype=float)
    x_test = np.arange(len(train), len(df), dtype=float)

    rows_metrics = []
    anomaly_rows = []

    all_series = {}

    extr_dir = os.path.join(REPORTS_DIR, "real_nbu_extrapolation")
    os.makedirs(extr_dir, exist_ok=True)

    for cur in NBU_CURRENCIES:
        y_train_raw = train[cur].astype(float).values
        y_test = test[cur].astype(float).values

        detector = EntropyAnomalyDetector(
            window=ANOMALY_WINDOW,
            base_k=ANOMALY_BASE_K,
            alpha=ANOMALY_ALPHA,
            bins=ANOMALY_BINS,
        )
        y_train_clean, anomalies_count = detector.clean(y_train_raw)

        flt = AlphaBetaGammaFilter(ABG_ALPHA, ABG_BETA, ABG_GAMMA, DT)
        y_train_smooth = run_filter_series(flt, y_train_clean)

        all_series[cur] = y_train_smooth

        anomaly_rows.append(
            {
                "currency": cur,
                "metric": "total_anomalies",
                "value": detector.total_anomalies,
            }
        )
        anomaly_rows.append(
            {
                "currency": cur,
                "metric": "avg_entropy",
                "value": detector.avg_entropy,
            }
        )
        anomaly_rows.append(
            {
                "currency": cur,
                "metric": "avg_k_local",
                "value": detector.avg_k_local,
            }
        )

        dates_train = train["date"].values
        plt.figure(figsize=(12, 6))
        plt.plot(dates_train, y_train_raw, label=f"{cur} raw", alpha=0.3)
        plt.plot(
            dates_train,
            y_train_smooth,
            label=f"{cur} cleaned+smoothed (Entropy + ABG_adaptive)",
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "real_nbu_filters"), exist_ok=True)
        plt.savefig(
            os.path.join(FIGURES_DIR, "real_nbu_filters", f"{cur}_cleaned_smoothed.png")
        )
        plt.close()

        models = Models(device="cpu")
        models.fit_all(x_train, y_train_smooth)

        for name, m in models.models.items():
            y_pred_train = m.predict(x_train)
            y_pred_test = m.predict(x_test)
            train_metrics = metrics_regression(y_train_smooth, y_pred_train)
            test_metrics = metrics_regression(y_test, y_pred_test)
            for dataset, mm in [("train", train_metrics), ("test", test_metrics)]:
                for k, v in mm.items():
                    rows_metrics.append(
                        {
                            "currency": cur,
                            "model": name,
                            "dataset": dataset,
                            "metric": k,
                            "value": v,
                        }
                    )

        x_horizons = generate_extrapolation_x(x_train, FORECAST_HORIZONS)
        for model_name, m in models.models.items():
            for h, x_future in x_horizons.items():
                y_future = m.predict(x_future)
                df_extr = pd.DataFrame(
                    {
                        "currency": cur,
                        "model": model_name,
                        "horizon": h,
                        "step": np.arange(len(x_train), len(x_train) + len(x_future)),
                        "y_pred": y_future,
                    }
                )
                df_extr.to_csv(
                    os.path.join(
                        extr_dir,
                        f"{cur}_{model_name}_h{h}.csv",
                    ),
                    index=False,
                )

        dates_full = df["date"].values
        y_full = df[cur].astype(float).values
        if "Poly2" in models.models:
            m_plot = models.models["Poly2"]
        else:
            first_key = list(models.models.keys())[0]
            m_plot = models.models[first_key]
        y_pred_full = m_plot.predict(np.arange(len(df), dtype=float))

        plt.figure(figsize=(12, 6))
        plt.plot(dates_full, y_full, label=f"{cur} real", alpha=0.7)
        plt.plot(
            dates_full, y_pred_full, label=f"{cur} Poly2 regression", linestyle="--"
        )
        plt.axvline(dates_full[cut - 1], color="black", linewidth=1.0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "real_nbu"), exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, "real_nbu", f"{cur}_regression.png"))
        plt.close()

        decompose_and_plot(
            y=y_full,
            dates=pd.to_datetime(df["date"].values),
            title=f"NBU decomposition {cur}",
            fig_path=os.path.join(FIGURES_DIR, "real_nbu", f"{cur}_decomposition.png"),
            model=TS_DECOMP_MODEL,
            period=TS_DECOMP_PERIOD_NBU,
        )

        arima_order, arima_aic, arima_mse = select_best_arima_order(
            y_train_smooth, ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE, val_ratio=0.2
        )

        if arima_order is not None:
            try:
                arima_model = ARIMA(y_train_smooth, order=arima_order).fit()
                arima_fig_dir = os.path.join(FIGURES_DIR, "real_nbu_arima")
                os.makedirs(arima_fig_dir, exist_ok=True)

                x_horizons_cur = generate_extrapolation_x(x_train, FORECAST_HORIZONS)

                for h, x_future in x_horizons_cur.items():
                    n_steps = len(x_future)
                    y_future = np.asarray(arima_model.forecast(steps=n_steps), float)

                    last_train_date = train["date"].iloc[-1]
                    future_dates = pd.date_range(
                        last_train_date + timedelta(days=1),
                        periods=n_steps,
                        freq="D",
                    )

                    df_extr_arima = pd.DataFrame(
                        {
                            "currency": cur,
                            "horizon": h,
                            "date": future_dates,
                            "y_pred": y_future,
                        }
                    )
                    df_extr_arima.to_csv(
                        os.path.join(extr_dir, f"{cur}_ARIMA_h{h}.csv"),
                        index=False,
                    )

                    plt.figure(figsize=(12, 6))
                    plt.plot(train["date"], y_train_smooth, label="train smoothed")
                    plt.plot(test["date"], y_test, label="test real", alpha=0.5)
                    plt.axvline(train["date"].iloc[-1], color="black", linewidth=1.0)

                    if len(test) > 0:
                        last_test_date = test["date"].iloc[-1]
                        mask_overlap = future_dates <= last_test_date
                    else:
                        mask_overlap = np.zeros_like(y_future, dtype=bool)

                    if mask_overlap.any():
                        plt.plot(
                            future_dates[mask_overlap],
                            y_future[mask_overlap],
                            label=f"ARIMA forecast (overlap, h={h})",
                        )
                    if (~mask_overlap).any():
                        plt.plot(
                            future_dates[~mask_overlap],
                            y_future[~mask_overlap],
                            linestyle="--",
                            label=f"ARIMA forecast (beyond, h={h})",
                        )

                    plt.title(
                        f"NBU {cur}: ARIMA extrapolation, horizon={h}, order={arima_order}"
                    )
                    plt.xlabel("date")
                    plt.ylabel("rate")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            arima_fig_dir, f"{cur}_arima_extrapolation_h{h}.png"
                        )
                    )
                    plt.close()

            except Exception as e:
                print(f"[WARN] NBU ARIMA failed for {cur}: {e}")

    os.makedirs(os.path.join(REPORTS_DIR, "real_nbu"), exist_ok=True)
    df_metrics = pd.DataFrame(rows_metrics)
    df_metrics.to_csv(
        os.path.join(REPORTS_DIR, "real_nbu", "nbu_model_metrics.csv"), index=False
    )

    df_anomaly = pd.DataFrame(anomaly_rows)
    df_anomaly.to_csv(
        os.path.join(REPORTS_DIR, "real_nbu", "real_nbu_anomaly_report.csv"),
        index=False,
    )

    if all_series:
        data_matrix = pd.DataFrame(all_series).T
        data_matrix.columns = train["date"].values
        analyze_matrix(
            data=data_matrix,
            name="real_nbu",
            model=TS_DECOMP_MODEL,
            period=TS_DECOMP_PERIOD_NBU,
            synthetic_years=TS_SYNTHETIC_YEARS,
            noise_scale=TS_NOISE_SCALE,
            reports_dir=os.path.join(REPORTS_DIR, "ts_real_nbu"),
            figures_dir=os.path.join(FIGURES_DIR, "ts_real_nbu"),
        )


#! ==============================
#!  PIPELINE: DataSet_6
#! ==============================


def load_and_clean_dataset6(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df[DATASET6_REGION_COL] = df[DATASET6_REGION_COL].astype(str).str.strip()
    df.replace(
        {
            "n.a.": np.nan,
            "not avilable": np.nan,
            -1.0: np.nan,
            -1: np.nan,
        },
        inplace=True,
    )
    for col in DATASET6_MONTH_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[DATASET6_MONTH_COLUMNS] = (
        df[DATASET6_MONTH_COLUMNS].T.interpolate(limit_direction="both").T
    )
    df[DATASET6_MONTH_COLUMNS] = df[DATASET6_MONTH_COLUMNS].fillna(
        df[DATASET6_MONTH_COLUMNS].mean()
    )
    return df


def build_region_ts_dataset6(df: pd.DataFrame) -> pd.DataFrame:
    region_ts = df.groupby(DATASET6_REGION_COL)[DATASET6_MONTH_COLUMNS].mean()
    return region_ts


def statistical_learning_dataset6(region_ts: pd.DataFrame) -> pd.DataFrame:
    records = []
    n = len(DATASET6_MONTH_COLUMNS)
    x = np.arange(n, dtype=float)
    cut = int(LINEAR_TRAIN_RATIO_DS6 * n)
    for region, row in region_ts.iterrows():
        y = row.values.astype(float)
        x_train = x[:cut]
        y_train = y[:cut]
        x_test = x[cut:]
        y_test = y[cut:]

        models = Models(device=device)
        models.fit_all(x_train, y_train)

        for name, m in models.models.items():
            y_pred_train = m.predict(x_train)
            m_train = metrics_regression(y_train, y_pred_train)
            if len(y_test) > 0:
                y_pred_test = m.predict(x_test)
                m_test = metrics_regression(y_test, y_pred_test)
            else:
                m_test = {"mse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            for dataset, mm in [("train", m_train), ("test", m_test)]:
                records.append(
                    {
                        "region": region,
                        "model": name,
                        "dataset": dataset,
                        "mse": mm["mse"],
                        "mae": mm["mae"],
                        "r2": mm["r2"],
                    }
                )

        months = pd.date_range("2025-01-01", periods=n, freq="M")
        if "Poly2" in models.models:
            m_plot = models.models["Poly2"]
        else:
            first_key = list(models.models.keys())[0]
            m_plot = models.models[first_key]
        y_pred_full = m_plot.predict(x)

        plt.figure(figsize=(9, 5))
        plt.plot(months, y, marker="o", label="Sales")
        plt.plot(months, y_pred_full, linestyle="--", label="Poly2")
        plt.title(f"PolyRegression {region}")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.join(FIGURES_DIR, "dataset6_regression"), exist_ok=True)
        plt.savefig(
            os.path.join(FIGURES_DIR, "dataset6_regression", f"regression_{region}.png")
        )
        plt.close()

        x_horizons = generate_extrapolation_x(x_train, FORECAST_HORIZONS)
        extr_dir = os.path.join(REPORTS_DIR, "dataset6_extrapolation")
        os.makedirs(extr_dir, exist_ok=True)
        for model_name, m in models.models.items():
            for h, x_future in x_horizons.items():
                y_future = m.predict(x_future)
                df_extr = pd.DataFrame(
                    {
                        "region": region,
                        "model": model_name,
                        "horizon": h,
                        "step": np.arange(len(x_train), len(x_train) + len(x_future)),
                        "y_pred": y_future,
                    }
                )
                df_extr.to_csv(
                    os.path.join(
                        extr_dir,
                        f"{region}_{model_name}_h{h}.csv",
                    ),
                    index=False,
                )

        arima_order, arima_aic, arima_mse = select_best_arima_order(
            y_train, ARIMA_P_RANGE, ARIMA_D_RANGE, ARIMA_Q_RANGE, val_ratio=0.2
        )

        if arima_order is not None:
            try:
                arima_model = ARIMA(y_train, order=arima_order).fit()
                arima_fig_dir = os.path.join(FIGURES_DIR, "dataset6_arima")
                os.makedirs(arima_fig_dir, exist_ok=True)

                months_all = pd.date_range("2025-01-01", periods=n, freq="M")
                train_months = months_all[:cut]
                test_months = months_all[cut:]

                x_horizons_reg = generate_extrapolation_x(x_train, FORECAST_HORIZONS)

                for h, x_future in x_horizons_reg.items():
                    n_steps = len(x_future)
                    y_future = np.asarray(arima_model.forecast(steps=n_steps), float)

                    last_train_month = train_months[-1]
                    future_months = pd.date_range(
                        last_train_month + pd.offsets.MonthEnd(1),
                        periods=n_steps,
                        freq="M",
                    )

                    df_extr_arima = pd.DataFrame(
                        {
                            "region": region,
                            "horizon": h,
                            "date": future_months,
                            "y_pred": y_future,
                        }
                    )
                    df_extr_arima.to_csv(
                        os.path.join(
                            extr_dir,
                            f"{region}_ARIMA_h{h}.csv",
                        ),
                        index=False,
                    )

                    plt.figure(figsize=(9, 5))
                    plt.plot(train_months, y_train, marker="o", label="train")
                    if len(y_test) > 0:
                        plt.plot(test_months, y_test, marker="o", label="test")
                    plt.axvline(train_months[-1], color="black", linewidth=1.0)

                    plt.plot(
                        future_months,
                        y_future,
                        marker="o",
                        label=f"ARIMA forecast (h={h})",
                    )

                    plt.title(
                        f"{region}: ARIMA extrapolation, horizon={h}, order={arima_order}"
                    )
                    plt.xlabel("Month")
                    plt.ylabel("Sales")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            arima_fig_dir,
                            f"{region}_arima_extrapolation_h{h}.png",
                        )
                    )
                    plt.close()

            except Exception as e:
                print(f"[WARN] Dataset6 ARIMA failed for {region}: {e}")

    df_metrics = pd.DataFrame(records)
    os.makedirs(os.path.join(REPORTS_DIR, "dataset6"), exist_ok=True)
    df_metrics.to_csv(
        os.path.join(REPORTS_DIR, "dataset6", "dataset6_model_metrics.csv"),
        index=False,
    )
    return df_metrics


def pipeline_dataset6():
    df_raw = load_and_clean_dataset6(DATASET6_PATH)
    region_ts = build_region_ts_dataset6(df_raw)

    dates = pd.date_range("2025-01-01", periods=len(DATASET6_MONTH_COLUMNS), freq="M")
    plt.figure(figsize=(10, 6))
    for region, row in region_ts.iterrows():
        plt.plot(dates, row.values, marker="o", label=region)
    plt.title("DataSet_6")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(FIGURES_DIR, "dataset6"), exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "dataset6", "dataset6_time_series.png"))
    plt.close()

    data = pd.DataFrame(
        {region: region_ts.loc[region].values for region in region_ts.index},
        index=DATASET6_MONTH_COLUMNS,
    ).T
    data.columns = dates

    analyze_matrix(
        data=data,
        name="dataset6",
        model=TS_DECOMP_MODEL,
        period=TS_DECOMP_PERIOD_DS6,
        synthetic_years=TS_SYNTHETIC_YEARS,
        noise_scale=TS_NOISE_SCALE,
        reports_dir=os.path.join(REPORTS_DIR, "ts_dataset6"),
        figures_dir=os.path.join(FIGURES_DIR, "ts_dataset6"),
    )

    for region, row in region_ts.iterrows():
        y_region = row.values.astype(float)
        decompose_and_plot(
            y=y_region,
            dates=dates,
            title=f"Sales_decompose (DataSet_6, {region})",
            fig_path=os.path.join(
                FIGURES_DIR, "dataset6", f"dataset6_decomposition_{region}.png"
            ),
            model=TS_DECOMP_MODEL,
            period=TS_DECOMP_PERIOD_DS6,
        )

    statistical_learning_dataset6(region_ts)


#! ==============================
#!  DIRECTORIES + MAIN
#! ==============================


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    ensure_dirs()

    print("Choose pipeline:")
    print("1 - Synthetic")
    print("2 - Real NBU")
    print("3 - DataSet_6")
    print("Enter - all")

    choice = input("> ").strip()

    if choice == "1":
        pipeline_synthetic()
    elif choice == "2":
        pipeline_real_nbu()
    elif choice == "3":
        pipeline_dataset6()
    elif choice == "":
        pipeline_synthetic()
        pipeline_real_nbu()
        pipeline_dataset6()
    else:
        print("[ERROR]")


if __name__ == "__main__":
    main()
