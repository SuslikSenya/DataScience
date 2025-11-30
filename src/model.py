import numpy as np
import logging

logger = logging.getLogger(__name__)


class LSMModel:
    def __init__(self, 
                 is_linear: bool = True):
        """Model initialization function"""
        self.degree = 1 if is_linear else 3
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
