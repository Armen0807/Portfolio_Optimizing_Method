import pandas as pd
import scipy.optimize as sco
import datetime as dt
import numpy as np
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.common.instruments import Instrument
from nautilus_trader.common.timeframes import Timeframe
from nautilus_trader.common.periods import Period
from nautilus_trader.model.data import Bar
from nautilus_trader.trading.position import Position
from nautilus_trader.common.quantity import Quantity

class MinVarianceConfig(BaseModel):
    assets: List[str] = Field(..., description="List of asset tickers for portfolio optimization")
    optimization_frequency: str = Field("M", description="Frequency of portfolio rebalancing (e.g., 'M' for monthly)")


class MinVarianceDetails:
    portfolio_risk: Dict[dt.date, float] = {}
    synthetic_tc: Dict[dt.date, float] = {}
    strat_level: Dict[dt.date, float] = {}


class MinVarianceIntermediate:
    volatility: Dict[str, Dict[int, Dict[dt.date, float]]] = {}
    underlying_spot_price: Dict[str, Dict[dt.date, float]] = {}
    returns: Dict[str, Dict[dt.date, float]] = {}
    weights: Dict[str, Dict[dt.date, float]] = {}


class MinVarianceHistory:
    details: MinVarianceDetails = MinVarianceDetails()
    intermediate: MinVarianceIntermediate = MinVarianceIntermediate()


class MinVarianceBacktestStrategy(Strategy):

    config: MinVarianceConfig
    history: MinVarianceHistory
    instruments: Dict[str, Instrument] = {}
    last_optimization_date: Optional[dt.date] = None
    optimal_weights: Dict[str, float] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = MinVarianceConfig(**self.config)
        self.history = MinVarianceHistory()
        self.instruments = {
            ticker: self.engine.get_instrument(ticker) for ticker in self.config.assets
        }
        if not all(self.instruments.values()):
            raise ValueError(f"Could not find all specified assets: {self.config.assets}")
        self.last_optimization_date = None
        self.optimal_weights = {ticker: 1.0 / len(self.config.assets) for ticker in self.config.assets} # Initial equal weights

    def get_historical_prices(self, instrument: Instrument, start_date: dt.date, end_date: dt.date) -> Optional[pd.DataFrame]:
        history = self.engine.get_historical_data(
            instrument=instrument,
            timeframe=Timeframe.DAILY,
            period=Period.range(start_date, end_date)
        )
        if not history:
            return None
        df = pd.DataFrame([{'ts': bar.ts, 'close': bar.close} for bar in history])
        df['ts'] = pd.to_datetime(df['ts']).dt.date
        df.set_index('ts', inplace=True)
        return df

    def portfolio_risk(self, weights, cov_matrix):
        """Calculate portfolio risk (standard deviation)."""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def optimize_portfolio(self, returns_data: pd.DataFrame):
        """Run the optimization to minimize the portfolio risk."""
        cov_matrix = returns_data.cov()
        num_assets = len(returns_data.columns)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        bounds = tuple((0, 1) for _ in range(num_assets)) # Weights between 0 and 1

        initial_weights = np.array([1. / num_assets] * num_assets)

        optimized = sco.minimize(
            self.portfolio_risk,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if optimized.success:
            optimal_weights = optimized.x
            min_risk = self.portfolio_risk(optimal_weights, cov_matrix)
            return dict(zip(returns_data.columns, optimal_weights)), min_risk
        else:
            self.logger.warn(f"Optimization failed: {optimized.message}")
            return self.optimal_weights, np.nan

    def on_bar(self, bar: Bar):
        current_date = bar.trade_date.date()

        # Determine if it's time to rebalance the portfolio
        if self.last_optimization_date is None or self.should_rebalance(self.last_optimization_date, current_date):
            self.logger.info(f"Rebalancing portfolio on {current_date}")
            end_date = current_date
            start_date = end_date - pd.DateOffset(years=1) # Look back one year for optimization

            historical_data = {}
            for ticker, instrument in self.instruments.items():
                df_prices = self.get_historical_prices(instrument, start_date, end_date)
                if df_prices is not None and not df_prices.empty:
                    historical_data[ticker] = df_prices['close']
                else:
                    self.logger.warn(f"Could not retrieve historical data for {ticker}")
                    return

            if len(historical_data) == len(self.instruments):
                prices_df = pd.DataFrame(historical_data)
                returns_df = prices_df.pct_change().dropna()
                if not returns_df.empty:
                    optimal_weights, min_risk = self.optimize_portfolio(returns_df)
                    self.optimal_weights = optimal_weights
                    self.history.details.portfolio_risk[current_date] = min_risk
                    # Store other history if needed
                    self.logger.info(f"Optimal weights for min variance: {self.optimal_weights}")
                    self.logger.info(f"Minimum portfolio risk: {min_risk:.4f}")
                    self.last_optimization_date = current_date
                else:
                    self.logger.warn("No return data available for optimization.")
            else:
                self.logger.warn("Could not retrieve historical data for all assets.")

        # You would typically implement order placement logic here based on the optimal weights
        # and your current portfolio holdings. This example focuses on the optimization step.
        pass

    def should_rebalance(self, last_rebalance_date: dt.date, current_date: dt.date) -> bool:
        if self.config.optimization_frequency == "M":
            return last_rebalance_date.month < current_date.month or last_rebalance_date.year < current_date.year
        elif self.config.optimization_frequency == "Q":
            return (current_date.month - last_rebalance_date.month) % 3 == 0 or current_date.year > last_rebalance_date.year
        elif self.config.optimization_frequency == "Y":
            return current_date.year > last_rebalance_date.year
        elif self.config.optimization_frequency == "W":
            return current_date.isocalendar()[1] > last_rebalance_date.isocalendar()[1] or current_date.year > last_rebalance_date.year
        elif self.config.optimization_frequency == "D":
            return current_date > last_rebalance_date
        return False

    def on_start(self):
        self.logger.info("Minimum Variance Optimization Strategy Started")
        if not self.config.assets:
            self.logger.error("No assets specified in the configuration.")

    def on_stop(self):
        self.logger.info("Minimum Variance Optimization Strategy Stopped")
