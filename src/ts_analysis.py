import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.config import RANDOM_STATE, N_CLUSTERS, DEFAULT_MA_WINDOW


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
