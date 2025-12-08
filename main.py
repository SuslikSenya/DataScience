import warnings

import torch
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.dirs import ensure_dirs
from src.pipelines_nbu import pipeline_real_nbu

"""
Білет 23: Дослідження алгоритмів обробки аномалій у часових рядах
Аналіз алгоритмів виявлення та обробки аномалій типу "грубі значення"
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


def main():
    ensure_dirs()
    try:
        pipeline_real_nbu()
    except Exception as e:
        print(f"Error {e} ")


if __name__ == "__main__":
    main()
