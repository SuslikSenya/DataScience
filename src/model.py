import numpy as np
import logging

import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch import nn, optim

logger = logging.getLogger(__name__)


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


class TorchNNModel:
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

    def fit(self, x, y):
        self.model.train()
        x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(x_t)
            loss = self.loss_fn(y_pred, y_t)
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        self.model.eval()
        x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(self.device)
        with torch.no_grad():
            y_pred = self.model(x_t).cpu().numpy().reshape(-1)
        return y_pred


class SklearnModel:
    """
    Provides polynomial regression using sklearn tools
    for comparison with manual LSM implementations.
    """

    def __init__(self, trend_type: str):
        self.degree = 1 if trend_type == "linear" else 2
        self.poly = PolynomialFeatures(self.degree, include_bias=False)
        self.model = LinearRegression()

    def fit(self, x, y):
        X = self.poly.fit_transform(x.reshape(-1, 1))
        self.model.fit(X, y)

    def predict(self, x):
        X = self.poly.transform(x.reshape(-1, 1))
        return self.model.predict(X)
