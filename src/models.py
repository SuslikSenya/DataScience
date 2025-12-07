# models.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import DEFAULT_MA_WINDOW


class BaseTSModel:
    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SklearnPolyTSModel(BaseTSModel):
    def __init__(self, degree: int = 2):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

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
                preds.append(float(np.mean(hist[-self.window:])))
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
            device: str = "cpu",
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
    def __init__(self, device: str = "cpu"):
        self.models = {
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
