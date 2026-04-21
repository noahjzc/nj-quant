"""
Fitness Evaluator: Simplified backtest for GA optimization.

Simplified vs full backtest:
- No daily ATR/stop-loss/take-profit (too slow for GA)
- Weekly rebalancing with equal-weight portfolio
- Uses real price data to compute actual returns
- Fast enough for thousands of GA evaluations
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import timedelta
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_loader import FactorLoader
from back_testing.factors.factor_config import get_factor_directions
from back_testing.data.data_provider import DataProvider


class FitnessEvaluator:
    """
    Fitness evaluator using simplified weekly-rebalancing backtest.

    Optimization target: Calmar ratio = annual_return / max_drawdown
    Constraint: max_drawdown <= 20%
    """

    def __init__(self, data_path: str = None,
                 max_drawdown_constraint: float = 0.20,
                 n_stocks: int = 5):
        """
        Args:
            data_path: Path to data directory (Parquet/CSV)
            max_drawdown_constraint: Maximum allowed drawdown
            n_stocks: Number of stocks to hold
        """
        self.data_path = data_path
        self.max_drawdown_constraint = max_drawdown_constraint
        self.n_stocks = n_stocks

        self.data_provider = DataProvider(use_db=False, data_dir=data_path)
        self.factor_loader = FactorLoader(data_provider=self.data_provider)

    def evaluate(self, weights: Dict[str, float],
                start_date: pd.Timestamp,
                end_date: pd.Timestamp) -> float:
        """
        Evaluate fitness of a weight configuration.

        Args:
            weights: Factor weights dict
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Calmar ratio (0 if constraint violated)
        """
        try:
            result = self._run_backtest(weights, start_date, end_date)

            annual_return = result.get('annual_return', 0)
            max_drawdown = result.get('max_drawdown', 0)

            if max_drawdown > self.max_drawdown_constraint:
                return 0.0

            # Prevent division by zero
            calmar = annual_return / max(max_drawdown, 0.01)

            return calmar

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.0

    def _run_backtest(self, weights: Dict[str, float],
                     start_date: pd.Timestamp,
                     end_date: pd.Timestamp) -> Dict:
        """
        Run simplified backtest.

        Weekly flow:
        1. Get rebalance dates (Fridays)
        2. Each Friday: select stocks by factor scoring
        3. Next Friday: calculate holding period return
        4. Accumulate portfolio value curve
        5. Calculate performance metrics
        """
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        if len(rebalance_dates) < 2:
            return self._empty_result()

        factor_directions = get_factor_directions()

        portfolio_values = [1.0]

        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            factor_list = list(weights.keys())
            factor_data = self.factor_loader.load_all_stock_factors(
                current_date, factor_list
            )

            if len(factor_data) == 0:
                portfolio_values.append(portfolio_values[-1])
                continue

            selector = MultiFactorSelector(
                weights=weights,
                directions=factor_directions
            )
            selected_stocks = selector.select_top_stocks(
                data=factor_data,
                n=self.n_stocks
            )

            if not selected_stocks:
                portfolio_values.append(portfolio_values[-1])
                continue

            period_return = self._calculate_period_return(
                selected_stocks, current_date, next_date
            )

            new_value = portfolio_values[-1] * (1 + period_return)
            portfolio_values.append(new_value)

        portfolio_values = np.array(portfolio_values)
        total_return = portfolio_values[-1] / portfolio_values[0] - 1

        n_weeks = len(portfolio_values) - 1
        if n_weeks > 0:
            annual_return = (1 + total_return) ** (52 / n_weeks) - 1
        else:
            annual_return = 0

        max_drawdown = self._calculate_max_drawdown(portfolio_values)

        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'n_weeks': n_weeks
        }

    def _get_rebalance_dates(self, start_date: pd.Timestamp,
                            end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """Get all Fridays between start and end dates"""
        dates = []
        current = pd.Timestamp(start_date)

        while current.weekday() != 4:
            current += timedelta(days=1)

        while current <= end_date:
            dates.append(current)
            current += timedelta(days=7)

        return dates

    def _calculate_period_return(self, stocks: List[str],
                                 current_date: pd.Timestamp,
                                 next_date: pd.Timestamp) -> float:
        """
        Calculate equal-weighted return of selected stocks over period.

        Args:
            stocks: List of stock codes
            current_date: Rebalance date
            next_date: Next rebalance date

        Returns:
            Equal-weighted portfolio return
        """
        returns = []

        for stock in stocks:
            try:
                df = self.data_provider.get_stock_data(
                    stock,
                    start_date=current_date.strftime('%Y-%m-%d'),
                    end_date=next_date.strftime('%Y-%m-%d')
                )

                if len(df) < 2:
                    continue

                price_col = None
                for col in ['后复权价', 'close', '收盘价']:
                    if col in df.columns:
                        price_col = col
                        break

                if price_col is None:
                    continue

                df = df.sort_index()
                prices = df[price_col].values

                if len(prices) >= 2:
                    period_return = (prices[-1] / prices[0]) - 1
                    returns.append(period_return)

            except Exception:
                continue

        if not returns:
            return 0.0

        return np.mean(returns)

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown from value sequence"""
        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _empty_result(self) -> Dict:
        """Return empty result for invalid evaluation"""
        return {
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'n_weeks': 0
        }