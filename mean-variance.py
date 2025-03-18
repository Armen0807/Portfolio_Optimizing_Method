import pandas as pd
import scipy.optimize as sco
import datetime as dt
import numpy as np
from typing import Dict

from inar_strat_types import AbstractStratConfig, StratHistory
from grt_lib_orchestrator import AbstractBacktestStrategy
from pandas import DataFrame

class MeanVarianceConfig(AbstractStratConfig):
    class Config:
        arbitrary_types_allowed = True

    risk_free_rate: float


class MeanVarianceDetails(StratHistory):
    portfolio_risk: Dict[dt.date, float] = {}
    synthetic_tc: Dict[dt.date, float] = {}
    strat_level: Dict[dt.date, float] = {}


class MeanVarianceIntermediate(StratHistory):
    volatility: Dict[str, Dict[int, Dict[dt.date, float]]] = {}
    underlying_spot_price: Dict[str, Dict[dt.date, float]] = {}
    returns: Dict[str, Dict[dt.date, float]] = {}
    weights: Dict[str, Dict[dt.date, float]] = {}


class MeanVarianceHistory(StratHistory):
    details: MeanVarianceDetails = MeanVarianceDetails()
    intermediate: MeanVarianceIntermediate = MeanVarianceIntermediate()


class MeanVarianceBacktestStrategy(AbstractBacktestStrategy):

    config: MeanVarianceConfig
    history: MeanVarianceHistory

    def __init__(self, config: MeanVarianceConfig, data_file: str):
        self.config = config
        self.history = MeanVarianceHistory()

        self.data: DataFrame = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        self.mean_returns = self.data.mean()
        self.cov_matrix = self.data.cov()
        self.risk_free_rate = config.risk_free_rate

    def portfolio_performance(self, weights):
        """Calculate portfolio statistics: risk, return, Sharpe ratio."""
        returns = np.sum(self.mean_returns * weights)
        std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (returns - self.risk_free_rate) / std_dev
        return std_dev, returns, sharpe_ratio

    def negative_sharpe_ratio(self, weights):
        """Objective function to minimize: negative Sharpe ratio."""
        return -self.portfolio_performance(weights)[2]

    def optimize_portfolio(self):
        """Run the optimization to maximize the Sharpe ratio."""
        num_assets = len(self.mean_returns)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        bounds = tuple((None, None) for _ in range(num_assets))

        initial_weights = np.array([1. / num_assets] * num_assets)

        optimized = sco.minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = optimized.x
        std_dev, returns, sharpe_ratio = self.portfolio_performance(optimal_weights)

        return optimal_weights, std_dev, returns, sharpe_ratio

    def run(self):
        """Run the backtest and display results."""
        weights, risk, expected_return, sharpe = self.optimize_portfolio()

        print("Pondérations optimales :", weights)
        print("Rendement attendu du portefeuille :", expected_return)
        print("Risque (écart-type) du portefeuille :", risk)
        print("Ratio de Sharpe :", sharpe)

if __name__ == "__main__":
    config = MeanVarianceConfig(risk_free_rate=0.0455)  # Example risk-free rate
    strategy = MeanVarianceBacktestStrategy(config=config, data_file='chemin_vers_votre_fichier.csv')
    strategy.run()
