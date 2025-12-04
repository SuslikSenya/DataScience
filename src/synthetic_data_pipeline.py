import numpy as np
from typing import Dict, Any

from .data import Config, TrendGenerator, NoiseGenerator, EntropyAnomalyDetector
from .metrics_plot import ModelMetrics, DataMetrics, Plot
from .model import LSMModel
from .reports import ReportGenerator


# !=============================================================================
# ! Synthetic Pipeline
# !=============================================================================


def pipeline_synthetic(cfg: Config):
    print("Starting synthetic pipeline.")

    trend_gen = TrendGenerator(cfg)
    noise_gen = NoiseGenerator(cfg)

    # Generate synthetic data
    x_train = np.linspace(*cfg.x_range_train, cfg.n_train)
    x_test = np.linspace(*cfg.x_range_test, cfg.n_test)

    trend_train = trend_gen.generate(x_train, cfg.trend_type)
    trend_test = trend_gen.generate(x_test, cfg.trend_type)

    noise_raw = noise_gen.generate(cfg.noise_type, cfg.n_train, cfg.add_anomalies)

    detector = EntropyAnomalyDetector(
        window=cfg.anomaly_window,
        base_k=cfg.anomaly_threshold,
        alpha=1.2,
        bins=30,
    )

    noise_clean, anomalies_count = detector.clean(noise_raw)
    y_train = trend_train + noise_clean

    # Save anomaly report
    ReportGenerator.save_anomaly_report(
        detector, f"{cfg.save_report_path}/anomaly_report.csv"
    )

    data_metrics = {
        "noise_mean": float(noise_raw.mean()),
        "noise_std": float(noise_raw.std()),
        "noise_clean_std": float(noise_clean.std()),
        "anomalies_removed": anomalies_count,
    }

    ReportGenerator.save_data_report(
        data_metrics, f"{cfg.save_report_path}/data_report.csv"
    )

    reports = []
    for name, model in cfg.models.items():
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        rep = {
            "model": name,
            "train_metrics": ModelMetrics.compute(trend_train, pred_train),
            "test_metrics": ModelMetrics.compute(trend_test, pred_test),
        }
        reports.append(rep)

        Plot.plot_data(
            x_train,
            y_train,
            trend_train,
            pred_train,
            x_test,
            pred_test,
            trend_test,
            fname=f"{cfg.save_plot_path}/{name}_synthetic.png",
        )

    ReportGenerator.save_model_report(
        reports, f"{cfg.save_report_path}/model_report.csv"
    )

class SyntheticDataPipeline:
    def __init__(
            self, config: Config, add_anomalies: bool = False
    ):
        """Pipeline initialization function"""

        print("Synthetic data pipeline initialized")
        self.cfg = config
        self.add_anomalies = add_anomalies
        self.trends = TrendGenerator(config)
        self.noises = NoiseGenerator(config)

    def forward(self,
                trend_type: str,
                noise_type: str,
                model_linear: bool) -> Dict[str, Any]:
        """Pipeline forward method"""
        print("Generating data...\n")
        x_train = np.linspace(*self.cfg.x_range_train, self.cfg.n_train)
        x_test = np.linspace(*self.cfg.x_range_test, self.cfg.n_test)

        trend_train = self.trends.generate(x_train, trend_type)
        trend_test = self.trends.generate(x_test, trend_type)

        noise_train = self.noises.generate(noise_type, self.cfg.n_train, self.add_anomalies)
        y_train = trend_train + noise_train

        Plot.plot_hist(fname='histogram', arr=noise_train)
        print("Fitting the model...\n")
        model = LSMModel(model_linear).fit(x_train, y_train)

        print("Predicting...\n")
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        plot_data_dict = {
            "x_train": x_train,
            "y_train": y_train,
            "trend_train": trend_train,
            "x_test": x_test,
            "trend_test": trend_test,
            "pred_train": pred_train,
            "pred_test": pred_test,
        }

        log_dict = {"data_metrics": DataMetrics.calculate(name=noise_type, arr=noise_train),
                    "train_metrics": ModelMetrics.calculate(trend_train, pred_train),
                    "test_metrics": ModelMetrics.calculate(trend_test, pred_test),
                    "pred_coef": model.pred_coef.tolist(), }

        return plot_data_dict, log_dict

    @staticmethod
    def save_plot(plot_dict: Dict[str, Any]):
        """Creating and saving the plot"""
        Plot.plot_data(x_train=plot_dict["x_train"],
                       y_train=plot_dict["y_train"], trend_train=plot_dict["trend_train"],
                       predictions_train=plot_dict["pred_train"],
                       x_test=plot_dict["x_test"],
                       predictions_test=plot_dict["pred_test"],
                       trend_test=plot_dict["trend_test"],
                       fname='Data')
