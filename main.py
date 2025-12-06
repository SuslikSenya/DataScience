import os

from src.data import Config
from src.model import SklearnModel, TorchNNModel
from src.synthetic_data_pipeline import pipeline_synthetic
from src.real_data_pipeline import pipeline_real

"""
Виконав: Слободенюк О.А.
Lab_work_3, варіант 10, III рівень складності:
"""


def main():
    cfg = Config()

    cfg.models["Sklearn"] = SklearnModel(cfg.trend_type)
    cfg.models["TorchNN"] = TorchNNModel(
        hidden_dim=64,
        lr=1e-3,
        epochs=1000,
        device="cpu",
    )

    os.makedirs(cfg.save_report_path, exist_ok=True)
    os.makedirs(cfg.save_plot_path, exist_ok=True)

    print("Choose pipeline:")
    print("1 - Synthetic")
    print("2 - Real")
    print("None - both")

    ch = input("> ").strip()
    if ch == "1":
        pipeline_synthetic(cfg)
    elif ch == "2":
        pipeline_real(cfg)
    elif ch == "":
        pipeline_synthetic(cfg)
        pipeline_real(cfg)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
