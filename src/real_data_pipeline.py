import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
import os

from requests.adapters import HTTPAdapter
from urllib3 import Retry

from .data import Config
from .model import LSMModel
from .metrics_plot import ModelMetrics, Plot


class NBUScrapper:
    def __init__(self, currencies: List[str] = None):
        self.currencies = currencies or ["USD", "EUR", "RUB"]
        self.session = requests.Session()

        # Add retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch_single_currency(self, currency: str, date_str: str) -> float:

        url = f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?valcode={currency}&date={date_str}&json"

        try:
            r = self.session.get(url, timeout=15)
            if r.status_code == 200:
                day = r.json()
                if day:  # Check if list is not empty
                    return float(day[0]["rate"])
            else:
                print(f"HTTP {r.status_code} for {currency} on {date_str}")
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            print(f"Error fetching {currency} for {date_str}: {e}")

        return None

    def fetch_all_currencies(self, date_str: str) -> Dict[str, float]:
        rates = {}
        for currency in self.currencies:
            rate = self.fetch_single_currency(currency, date_str)
            rates[currency] = rate
        return rates

    def fetch_range(self, start: str, end: str) -> pd.DataFrame:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")

        records = []
        current_date = s

        while current_date <= e:
            date_str = current_date.strftime("%Y%m%d")
            print(f"Fetching data for {current_date.date()}")

            rates = self.fetch_all_currencies(date_str)

            if any(rates.values()):
                record = {"date": current_date.date()}
                record.update(rates)
                records.append(record)

            current_date += timedelta(days=1)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("date")
        return df


class DataLoader:
    def __init__(self, save_path: str):
        self.save_path = save_path

    def save(self, df: pd.DataFrame):
        df.to_csv(self.save_path, index=False)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.save_path, parse_dates=["date"])

    def split(self, df: pd.DataFrame, ratio: float):
        n = len(df)
        cut = int(n * ratio)
        return df[:cut], df[cut:]


class RealDataPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scraper = NBUScrapper()
        self.loader = DataLoader(cfg.save_path)

    def scrape_and_save(self):
        if os.path.exists(self.cfg.save_path):
            print(f"Файл {self.cfg.save_path} вже існує. Використовую існуючі дані.")
            return

        print(f"Файл {self.cfg.save_path} не знайдено. Завантажую нові дані...")
        df = self.scraper.fetch_range(self.cfg.start_date, self.cfg.end_date)
        self.loader.save(df)
        print(f"Збережено дані для валют: {list(df.columns[1:])}")

    def forward(self, target_currency: str = "USD") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        df = self.loader.load()
        train, test = self.loader.split(df, self.cfg.train_ratio)

        x_train = np.arange(len(train), dtype=float)
        x_test = np.arange(len(train), len(df), dtype=float)

        y_train = train[target_currency].values
        y_test = test[target_currency].values

        model = LSMModel(self.cfg.is_linear).fit(x_train, y_train)
        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        plot_data = {
            "x_train": x_train,
            "y_train": y_train,
            "trend_train": y_train,
            "x_test": x_test,
            "trend_test": y_test,
            "pred_train": pred_train,
            "pred_test": pred_test,
            "currency": target_currency
        }

        logging_data = {
            "config": self.cfg,
            "currency": target_currency,
            "train_metrics": ModelMetrics.calculate(y_train, pred_train),
            "test_metrics": ModelMetrics.calculate(y_test, pred_test),
            "pred_coef": model.pred_coef.tolist(),
            "available_currencies": list(df.columns[1:])  # список доступних валют
        }

        return plot_data, logging_data

    def get_currency_correlation(self) -> pd.DataFrame:
        """Отримати кореляцію між валютами"""
        df = self.loader.load()
        currency_columns = [col for col in df.columns if col != 'date']
        return df[currency_columns].corr()

    def save_plot(self, plot_data: Dict[str, Any], fname: str):
        Plot.plot_data(
            x_train=plot_data["x_train"],
            y_train=plot_data["y_train"],
            trend_train=plot_data["trend_train"],
            predictions_train=plot_data["pred_train"],
            x_test=plot_data["x_test"],
            predictions_test=plot_data["pred_test"],
            trend_test=plot_data["trend_test"],
            fname=fname,
        )
