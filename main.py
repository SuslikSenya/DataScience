import warnings

import torch
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.dirs import ensure_dirs
from src.pipelines_dataset10 import pipeline_dataset10
from src.pipelines_nbu import pipeline_real_nbu
from src.pipelines_synthetic import pipeline_synthetic

"""
Виконав: Слободенюк О.А.
Lab_work_7, варіант 10, IV рівень складності:
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


def main():
    ensure_dirs()

    print("Choose pipeline:")
    print("1 - Synthetic")
    print("2 - Real NBU")
    print("3 - DataSet_10")
    print("Enter - all")

    choice = input("> ").strip()

    if choice == "1":
        pipeline_synthetic()
    elif choice == "2":
        pipeline_real_nbu()
    elif choice == "3":
        pipeline_dataset10()
    elif choice == "":
        pipeline_synthetic()
        pipeline_real_nbu()
        pipeline_dataset10()
    else:
        print("[ERROR]")


if __name__ == "__main__":
    main()
