from src.data import Config
from src.synthetic_data_pipeline import SyntheticDataPipeline
from src.real_data_pipeline import RealDataPipeline

"""
Виконав: Слободенюк О.А.
Lab_work_1, варіант 10, III рівень складності:

Закон зміни похибки – рівномірний, нормальний.
Закон зміни досліджуваного процесу – кубічний, лінійний.
Реальні дані – 3 показники на вибір.
"""


def main():
    print("Choose pipeline:")
    print("1 - Synthetic")
    print("2 - Real NBU data")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        default_trend_config = {
            "linear": {"a": 1.5, "b": 10.0},
            "cubic": {"a": 0.02, "b": 0.2, "c": 5.0, "d": 2.0},
        }
        default_noise_config = {
            "normal": {"mu": 0.0, "sigma": 2.0},
            "uniform": {},
        }
        cfg = Config(trend_config=default_trend_config,
                     noise_config=default_noise_config)
        pipe = SyntheticDataPipeline(cfg, add_anomalies=True)
        plot_dict, log_dict = pipe.forward("cubic", "uniform", False)
        pipe.save_plot(plot_dict=plot_dict)

        d = log_dict['data_metrics']
        train = log_dict['train_metrics']
        test = log_dict['test_metrics']
        coef = log_dict['pred_coef']

        print(f" - Дані: mean={d['uniform_mean']:.3f} std={d['uniform_std']:.3f}")
        print(f" - Навчання: MSE={train['mse']:.3f} MAE={train['mae']:.3f} R²={train['r2']:.6f}")
        print(f" - Тест: MSE={test['mse']:.3f} MAE={test['mae']:.3f} R²={test['r2']:.6f}")
        print(f" - Коефіцієнти: {[f'{c:.3f}' for c in coef]}")

    elif choice == "2":
        cfg = Config(
            save_path="multi_currency_data.csv",
            start_date="2024-01-01",
            end_date="2025-01-01",
            train_ratio=0.8,
            is_linear=False
        )

        pipeline = RealDataPipeline(cfg)

        pipeline.scrape_and_save()

        print("Analyzing USD...")
        plot_usd, log_usd = pipeline.forward("USD")
        pipeline.save_plot(plot_usd, "usd_plot")
        print("USD results:", log_usd)

        print("Analyzing EUR...")
        plot_eur, log_eur = pipeline.forward("EUR")
        pipeline.save_plot(plot_eur, "eur_plot")
        print("EUR results:", log_eur)

        correlation_matrix = pipeline.get_currency_correlation()
        print("Currency correlation:")
        print(correlation_matrix)

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
