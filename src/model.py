import numpy as np
import logging

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class LSMModel:
    def __init__(self, trend_type: str):
        self.degree = 1 if trend_type == "linear" else 2
        self.pred_coef = None

    def fit(self, 
            x: np.ndarray, 
            y: np.ndarray):
        """Model training(fitting) function"""
        self.pred_coef = np.polyfit(x, y, self.degree)
        logger.info("Fitted polynomial degree %d", self.degree)
        return self

    def predict(self, 
                x: np.ndarray) -> np.ndarray:
        """Model testing function"""
        return np.polyval(self.pred_coef, x)


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
