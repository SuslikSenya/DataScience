import warnings

import torch
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.dirs import ensure_dirs
from src.pipelines_dataset10 import pipeline_dataset10

"""
Виконав: Слободенюк О.А.
Lab_work_8, варіант 10, III рівень складності:
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


def main():
    ensure_dirs()

    pipeline_dataset10()


if __name__ == "__main__":
    main()
