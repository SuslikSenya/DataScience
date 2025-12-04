from typing import Dict, List

import pandas as pd

from src.data import EntropyAnomalyDetector


# !=============================================================================
# ! Reporting
# !=============================================================================


class ReportGenerator:
    """
    Generates structured CSV-based reports for data metrics,
    model performance metrics, and anomaly detection statistics.
    """

    @staticmethod
    def save_data_report(metrics: Dict[str, float], path: str):
        df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
        df.to_csv(path, index=False)

    @staticmethod
    def save_model_report(reports: List[dict], path: str):
        rows = []
        for rep in reports:
            model = rep["model"]
            for phase in ["train_metrics", "test_metrics"]:
                dataset = "train" if phase == "train_metrics" else "test"
                for k, v in rep[phase].items():
                    rows.append(
                        {"model": model, "dataset": dataset, "metric": k, "value": v}
                    )

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    @staticmethod
    def save_anomaly_report(detector: EntropyAnomalyDetector, path: str):
        df = pd.DataFrame(
            [
                {"metric": "total_anomalies", "value": detector.total_anomalies},
                {"metric": "avg_entropy", "value": detector.avg_entropy},
                {"metric": "avg_k_local", "value": detector.avg_k_local},
            ]
        )
        df.to_csv(path, index=False)
