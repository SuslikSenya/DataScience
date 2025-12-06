import numpy as np


# !=============================================================================
# ! Filters
# !=============================================================================


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
    return np.asarray(out)
