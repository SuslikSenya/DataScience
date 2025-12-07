# models.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BaseTSModel:
    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


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


class TorchRNNRegressor(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out  # (batch, seq_len, 1)


class TorchLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out  # (batch, seq_len, 1)


class TorchNNTSModel(BaseTSModel):
    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 500,
        input_dim: int | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.input_dim = input_dim
        self.model = None
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def _build_if_needed(self, in_dim: int):
        if self.model is None:
            if self.input_dim is None:
                self.input_dim = in_dim
            self.model = TorchNNRegressor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=1,
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if x.ndim == 1:
            X = x.reshape(-1, 1)
        elif x.ndim == 2:
            X = x
        else:
            raise ValueError("TorchNNTSModel.fit: x must be 1D or 2D")

        self._build_if_needed(X.shape[1])

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(X_t)
        loss = self.loss_fn(y_pred, y_t)
        loss.backward()
        self.optimizer.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            X = x.reshape(-1, 1)
        elif x.ndim == 2:
            X = x
        else:
            raise ValueError("TorchNNTSModel.predict: x must be 1D or 2D")

        X_t = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_t).cpu().numpy().reshape(-1)
        return y_pred


class TorchRNNTsModel(BaseTSModel):
    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 500,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.model = TorchRNNRegressor(
            input_dim=1, hidden_dim=self.hidden_dim, num_layers=1
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self.model.train()

        if x.ndim == 1:
            x_seq = torch.tensor(x.reshape(1, -1, 1), dtype=torch.float32)
            y_seq = torch.tensor(y.reshape(1, -1, 1), dtype=torch.float32)
            self.optimizer.zero_grad()
            y_pred = self.model(x_seq)
            loss = self.loss_fn(y_pred, y_seq)
            loss.backward()
            self.optimizer.step()
        elif x.ndim == 2:
            X_seq = torch.tensor(
                x.reshape(x.shape[0], x.shape[1], 1), dtype=torch.float32)
            y_seq = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
            self.optimizer.zero_grad()
            y_pred_full = self.model(X_seq)  # (batch, L, 1)
            y_pred_last = y_pred_full[:, -1, :]  # (batch, 1)
            loss = self.loss_fn(y_pred_last, y_seq)
            loss.backward()
            self.optimizer.step()
        else:
            raise ValueError("TorchRNNTsModel.fit: x must be 1D or 2D")

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self.model.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x_seq = torch.tensor(x.reshape(1, -1, 1), dtype=torch.float32)
                y_pred_full = self.model(x_seq).cpu().numpy().reshape(-1)
                return y_pred_full
            elif x.ndim == 2:
                X_seq = torch.tensor(
                    x.reshape(x.shape[0], x.shape[1], 1), dtype=torch.float32
                )
                y_pred_full = self.model(X_seq).cpu().numpy()  # (batch, L, 1)
                y_pred_last = y_pred_full[:, -1, 0]
                return y_pred_last
            else:
                raise ValueError("TorchRNNTsModel.predict: x must be 1D or 2D")


class TorchLSTMTSModel(BaseTSModel):
    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 500,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.model = TorchLSTMRegressor(
            input_dim=1, hidden_dim=self.hidden_dim, num_layers=1
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self.model.train()

        if x.ndim == 1:
            x_seq = torch.tensor(x.reshape(1, -1, 1), dtype=torch.float32)
            y_seq = torch.tensor(y.reshape(1, -1, 1), dtype=torch.float32)
            self.optimizer.zero_grad()
            y_pred = self.model(x_seq)
            loss = self.loss_fn(y_pred, y_seq)
            loss.backward()
            self.optimizer.step()
        elif x.ndim == 2:
            X_seq = torch.tensor(
                x.reshape(x.shape[0], x.shape[1], 1), dtype=torch.float32
            )
            y_seq = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
            self.optimizer.zero_grad()
            y_pred_full = self.model(X_seq)
            y_pred_last = y_pred_full[:, -1, :]  # (batch, 1)
            loss = self.loss_fn(y_pred_last, y_seq)
            loss.backward()
            self.optimizer.step()
        else:
            raise ValueError("TorchLSTMTSModel.fit: x must be 1D or 2D")

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self.model.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x_seq = torch.tensor(x.reshape(1, -1, 1), dtype=torch.float32) 
                y_pred_full = self.model(x_seq).cpu().numpy().reshape(-1)
                return y_pred_full
            elif x.ndim == 2:
                X_seq = torch.tensor(
                    x.reshape(x.shape[0], x.shape[1], 1), dtype=torch.float32
                )
                y_pred_full = self.model(X_seq).cpu().numpy()
                y_pred_last = y_pred_full[:, -1, 0]
                return y_pred_last
            else:
                raise ValueError("TorchLSTMTSModel.predict: x must be 1D or 2D")


class NNModels:
    def __init__(self):
        self.models = {
            "MLP": TorchNNTSModel(hidden_dim=32, lr=1e-3, epochs=500),
            "RNN": TorchRNNTsModel(hidden_dim=32, lr=1e-3, epochs=500),
            "LSTM": TorchLSTMTSModel(hidden_dim=32, lr=1e-3, epochs=500),
        }

    def fit_all(self, x: np.ndarray, y: np.ndarray):
        for m in self.models.values():
            m.fit(x, y)

    def predict_all(self, x: np.ndarray) -> dict:
        out = {}
        for name, m in self.models.items():
            out[name] = m.predict(x)
        return out
