import os

from src.data import Config
from src.model import LSMModel, SklearnModel
from src.synthetic_data_pipeline import SyntheticDataPipeline, pipeline_synthetic
from src.real_data_pipeline import RealDataPipeline, pipeline_real

"""
Виконав: Слободенюк О.А.
Lab_work_1, варіант 10, III рівень складності:

Закон зміни похибки – рівномірний, нормальний.
Закон зміни досліджуваного процесу – кубічний, лінійний.
Реальні дані – 3 показники на вибір.
"""


# def main():
#     print("Choose pipeline:")
#     print("1 - Synthetic")
#     print("2 - Real NBU data")
#
#     choice = input("Enter 1 or 2: ")
#
#     if choice == "1":
#         default_trend_config = {
#             "linear": {"a": 1.5, "b": 10.0},
#             "cubic": {"a": 0.02, "b": 0.2, "c": 5.0, "d": 2.0},
#         }
#         default_noise_config = {
#             "normal": {"mu": 0.0, "sigma": 2.0},
#             "uniform": {},
#         }
#         cfg = Config(trend_config=default_trend_config,
#                      noise_config=default_noise_config)
#         pipe = SyntheticDataPipeline(cfg, add_anomalies=True)
#         plot_dict, log_dict = pipe.forward("cubic", "uniform", False)
#         pipe.save_plot(plot_dict=plot_dict)
#
#         d = log_dict['data_metrics']
#         train = log_dict['train_metrics']
#         test = log_dict['test_metrics']
#         coef = log_dict['pred_coef']
#
#         print(f" - Дані: mean={d['uniform_mean']:.3f} std={d['uniform_std']:.3f}")
#         print(f" - Навчання: MSE={train['mse']:.3f} MAE={train['mae']:.3f} R²={train['r2']:.6f}")
#         print(f" - Тест: MSE={test['mse']:.3f} MAE={test['mae']:.3f} R²={test['r2']:.6f}")
#         print(f" - Коефіцієнти: {[f'{c:.3f}' for c in coef]}")
#
#     elif choice == "2":
#         cfg = Config(
#             save_path="multi_currency_data.csv",
#             start_date="2024-01-01",
#             end_date="2025-01-01",
#             train_ratio=0.8,
#             is_linear=False
#         )
#
#         pipeline = RealDataPipeline(cfg)
#
#         pipeline.scrape_and_save()
#
#         def print_currency_results(currency, results):
#             print(f"\n {currency} РЕЗУЛЬТАТИ:")
#             print(
#                 f" - Навчання: MSE={results['train_metrics']['mse']:.3f} | MAE={results['train_metrics']['mae']:.3f} | R²={results['train_metrics']['r2']:.3f}")
#             print(
#                 f" - Тест:     MSE={results['test_metrics']['mse']:.3f} | MAE={results['test_metrics']['mae']:.3f} | R²={results['test_metrics']['r2']:.3f}")
#
#         print("Analyzing USD...")
#         plot_usd, log_usd = pipeline.forward("USD")
#         pipeline.save_plot(plot_usd, "usd_plot")
#         print_currency_results("USD", log_usd)
#
#         print("Analyzing EUR...")
#         plot_eur, log_eur = pipeline.forward("EUR")
#         pipeline.save_plot(plot_eur, "eur_plot")
#         print_currency_results("EUR", log_eur)
#
#         print("Analyzing RUB...")
#         plot_rub, log_rub = pipeline.forward("RUB")
#         pipeline.save_plot(plot_rub, "rub_plot")
#         print_currency_results("RUB", log_rub)
#
#         correlation_matrix = pipeline.get_currency_correlation()
#         print("Currency correlation:")
#         print(correlation_matrix)
#
#     else:
#         print("Invalid choice")

def main():
    cfg = Config()

    cfg.models["LSM"] = LSMModel(cfg.trend_type)
    cfg.models["Sklearn"] = SklearnModel(cfg.trend_type)

    os.makedirs(cfg.save_report_path, exist_ok=True)
    os.makedirs(cfg.save_plot_path, exist_ok=True)

    print("Choose pipeline:")
    print("1 - Synthetic")
    print("2 - Real")

    ch = input("> ").strip()
    if ch == "1":
        pipeline_synthetic(cfg)
    elif ch == "2":
        pipeline_real(cfg)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
