"""Multi-factor stock selector using weighted scoring."""
from typing import Dict, List, Optional

import pandas as pd

from back_testing.factors.factor_utils import FactorProcessor


class MultiFactorSelector:
    """Multi-factor stock selector based on weighted scoring.

    This class implements a multi-factor selection model that:
    1. Standardizes factors using rank or z-score method
    2. Adjusts for factor direction (positive = larger is better, negative = smaller is better)
    3. Computes weighted composite scores
    4. Selects top N stocks by composite score

    Args:
        weights: Factor weight dictionary, e.g., {'PB': 0.3, 'ROE': 0.2, ...}
        directions: Factor direction, 1 means larger is better, -1 means smaller is better
        method: 'rank' or 'zscore' standardization method
    """

    def __init__(self, weights: Dict[str, float], directions: Dict[str, int], method: str = 'rank', data_provider=None, neutralize: bool = False):
        """Initialize MultiFactorSelector.

        Args:
            weights: Factor weight dictionary.
            directions: Factor direction dictionary.
            method: Standardization method - 'rank' or 'zscore'.
            data_provider: Data provider for turnover filtering.
            neutralize: Whether to apply market neutralization (regress each factor on log market cap).

        Raises:
            ValueError: If method is not 'rank' or 'zscore'.
        """
        if method not in ('rank', 'zscore'):
            raise ValueError(f"Invalid method '{method}'. Use 'rank' or 'zscore'.")

        self.weights = weights
        self.directions = directions
        self.method = method
        self.data_provider = data_provider
        self.neutralize = neutralize
        self._factor_processor = FactorProcessor()

    def calculate_factor_scores(self, data: pd.DataFrame) -> pd.Series:
        """Calculate composite factor scores for all stocks.

        Args:
            data: DataFrame with factor columns as keys and stock data as values.
                  Index should be stock codes.

        Returns:
            Series with stock codes as index and composite scores as values.
        """
        if data.empty:
            return pd.Series(dtype=float)

        # Get factor columns that exist in data
        factor_columns = [f for f in self.weights.keys() if f in data.columns]
        if not factor_columns:
            return pd.Series(0.0, index=data.index)

        # Handle single stock case - return neutral score of 0.5
        if len(data) == 1:
            return pd.Series([0.5], index=data.index)

        # === 市值中性化 ===
        if self.neutralize and 'LN_MCAP' in data.columns:
            market_cap = data['LN_MCAP']
            for factor in factor_columns:
                if factor != 'LN_MCAP':
                    # 对每个因子做市值中性化
                    data = data.copy()  # avoid SettingWithCopyWarning
                    data[factor] = self._factor_processor.neutralize(data[factor], market_cap)

        composite_scores = pd.Series(0.0, index=data.index)
        total_weight = sum(self.weights[f] for f in factor_columns)

        for factor in factor_columns:
            # Get factor direction (default to 1 if not specified)
            direction = self.directions.get(factor, 1)

            # For direction=-1 (smaller is better):
            #   ascending=True so lower values get lower percentile (no flip needed)
            # For direction=1 (larger is better):
            #   ascending=False so lower percentile goes to higher values (flip needed)
            ascending = direction == -1

            # Standardize factor using specified method
            if self.method == 'rank':
                processed = self._factor_processor.rank_percentile(
                    data[factor], ascending=ascending
                )
                # For direction=1, flip so higher original values get higher scores
                if direction == 1:
                    processed = 1 - processed
            else:  # zscore
                # Negate before z-score for direction=-1 so lower values get higher scores
                if direction == -1:
                    processed = self._factor_processor.z_score(-data[factor])
                else:
                    processed = self._factor_processor.z_score(data[factor])

            # Add weighted contribution to composite score
            composite_scores += processed * self.weights[factor] / total_weight

        return composite_scores

    def select_top_stocks(self, data: pd.DataFrame, n: int = 5, excluded: List[str] = None, date: pd.Timestamp = None) -> List[str]:
        """Select top N stocks by composite factor score.

        Args:
            data: DataFrame with factor columns and stock data.
            n: Number of stocks to select.
            excluded: List of stock codes to exclude from selection.
            date: Date for turnover filtering.

        Returns:
            List of selected stock codes sorted by score (highest first).
        """
        if excluded is None:
            excluded = []

        # Calculate scores
        scores = self.calculate_factor_scores(data)

        # Filter out excluded stocks
        available = scores.drop(index=[s for s in excluded if s in scores.index], errors='ignore')

        if available.empty:
            return []

        # === 主板优先逻辑 ===
        main_board = []
        chi_next = []
        for code in available.index:
            if code.startswith(('sh600', 'sh000', 'sz001', 'sz002')):
                main_board.append(code)
            else:
                chi_next.append(code)

        # 主板优先：70% 选主板，30% 选科创/创业
        # 多选一些候选（3倍），后续进行成交额过滤
        n_main_candidate = min(int(n * 0.7 * 3), len(main_board))
        n_chi_candidate = min(int(n * 0.3 * 3) + 2, len(chi_next))  # 多选一些

        # 按分数排序并选取候选
        main_scores = available[main_board].sort_values(ascending=False) if main_board else pd.Series(dtype=float)
        chi_scores = available[chi_next].sort_values(ascending=False) if chi_next else pd.Series(dtype=float)

        main_candidates = list(main_scores.head(n_main_candidate).index)
        chi_candidates = list(chi_scores.head(n_chi_candidate).index)

        # === 成交额过滤（只对候选股票检查） ===
        if self.data_provider is not None:
            MIN_TURNOVER = 50_000_000  # 5000万
            all_candidates = main_candidates + chi_candidates
            turnover = self._get_turnover_data(all_candidates, date)
            if len(turnover) > 0:
                # 过滤成交额不足的股票
                main_candidates = [c for c in main_candidates if c in turnover.index and turnover[c] >= MIN_TURNOVER]
                chi_candidates = [c for c in chi_candidates if c in turnover.index and turnover[c] >= MIN_TURNOVER]

        # 重新计算最终选取数量
        n_main = min(int(n * 0.7), len(main_candidates))
        n_chi = n - n_main

        # 按分数排序并选取最终结果
        main_final = available[main_candidates].sort_values(ascending=False) if main_candidates else pd.Series(dtype=float)
        chi_final = available[chi_candidates].sort_values(ascending=False) if chi_candidates else pd.Series(dtype=float)

        result = list(main_final.head(n_main).index)
        result.extend(chi_final.head(n_chi).index)

        return result

    def _get_turnover_data(self, stock_codes: List[str], date: pd.Timestamp) -> pd.Series:
        """获取股票成交额数据"""
        if self.data_provider is None:
            return pd.Series(dtype=float)
        from back_testing.factors.factor_loader import FactorLoader
        loader = FactorLoader(data_provider=self.data_provider)
        return loader.load_stock_turnover(list(stock_codes), date)

    def get_factor_contribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get each factor's contribution to the composite score.

        Args:
            data: DataFrame with factor columns and stock data.

        Returns:
            DataFrame with factor names as columns and stock codes as index.
            Each cell contains the weighted contribution of that factor.
        """
        if data.empty:
            return pd.DataFrame()

        # Get factor columns that exist in data
        factor_columns = [f for f in self.weights.keys() if f in data.columns]
        if not factor_columns:
            return pd.DataFrame(index=data.index)

        # Handle single stock case - return neutral contributions summing to 0.5
        if len(data) == 1:
            contributions = pd.DataFrame(0.0, index=data.index, columns=factor_columns)
            total_weight = sum(self.weights[f] for f in factor_columns)
            for factor in factor_columns:
                # Single stock gets neutral score of 0.5, distributed by normalized weight
                contributions.loc[data.index[0], factor] = 0.5 * self.weights[factor] / total_weight
            return contributions

        total_weight = sum(self.weights[f] for f in factor_columns)
        contributions = pd.DataFrame(0.0, index=data.index, columns=factor_columns)

        for factor in factor_columns:
            direction = self.directions.get(factor, 1)

            # For direction=-1 (smaller is better):
            #   ascending=True so lower values get lower percentile (no flip needed)
            # For direction=1 (larger is better):
            #   ascending=False so lower percentile goes to higher values (flip needed)
            ascending = direction == -1

            # Standardize factor
            if self.method == 'rank':
                processed = self._factor_processor.rank_percentile(
                    data[factor], ascending=ascending
                )
                # For direction=1, flip so higher original values get higher scores
                if direction == 1:
                    processed = 1 - processed
            else:
                # For zscore with direction=-1, negate before z_score
                if direction == -1:
                    processed = self._factor_processor.z_score(-data[factor])
                else:
                    processed = self._factor_processor.z_score(data[factor])

            # Store weighted contribution
            contributions[factor] = processed * self.weights[factor] / total_weight

        return contributions
