import pandas as pd
import scipy.optimize as sco
import datetime as dt
import numpy as np
from typing import Dict

from inar_strat_types import AbstractStratConfig, StratHistory
from grt_lib_orchestrator import AbstractBacktestStrategy
from pandas import DataFrame

class MinVarianceConfig(AbstractStratConfig):
    class Config:
        arbitrary_types_allowed = True


class MinVarianceDetails(StratHistory):
    portfolio_risk: Dict[dt.date, float] = {}
    synthetic_tc: Dict[dt.date, float] = {}
    strat_level: Dict[dt.date, float] = {}


class MinVarianceIntermediate(StratHistory):
    volatility: Dict[str, Dict[int, Dict[dt.date, float]]] = {}
    underlying_spot_price: Dict[str, Dict[dt.date, float]] = {}
    returns: Dict[str, Dict[dt.date, float]] = {}
    weights: Dict[str, Dict[dt.date, float]] = {}


class MinVarianceHistory(StratHistory):
    details: MinVarianceDetails = MinVarianceDetails()
    intermediate: MinVarianceIntermediate = MinVarianceIntermediate()


class MinVarianceBacktestStrategy(AbstractBacktestStrategy):

    config: MinVarianceConfig
    history: MinVarianceHistory

    def __init__(self, config: MinVarianceConfig, data_file: str):
        self.config = config
        self.history = MinVarianceHistory()

        # Load data from CSV
        self.data: DataFrame = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        self.mean_returns = self.data.mean()
        self.cov_matrix = self.data.cov()

    def portfolio_risk(self, weights):
        """Calculate portfolio risk (standard deviation)."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def optimize_portfolio(self):
        """Run the optimization to minimize the portfolio risk."""
        num_assets = len(self.mean_returns)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        bounds = tuple((None, None) for _ in range(num_assets))

        initial_weights = np.array([1. / num_assets] * num_assets)

        optimized = sco.minimize(
            self.portfolio_risk,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = optimized.x
        min_risk = self.portfolio_risk(optimal_weights)

        return optimal_weights, min_risk

    def run(self):
        """Run the backtest and display results."""
        weights, risk = self.optimize_portfolio()

        print("Pondérations optimales pour la variance minimale :", weights)
        print("Risque minimum (écart-type) du portefeuille :", risk)

# Example usage
if __name__ == "__main__":
    config = MinVarianceConfig()
    strategy = MinVarianceBacktestStrategy(config=config, data_file='chemin_vers_votre_fichier.csv')
    strategy.run()
