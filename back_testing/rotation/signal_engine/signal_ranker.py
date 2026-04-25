"""第二层信号排序器 — 多因子加权评分"""
import pandas as pd
import numpy as np
from typing import List, Dict
from back_testing.factors.factor_utils import FactorProcessor


class SignalRanker:
    """
    第二层排序器：对候选股按多因子加权评分排序

    使用 zscore 标准化 + 因子方向调整 + 加权求和
    """

    def __init__(self, factor_weights: Dict[str, float], factor_directions: Dict[str, int]):
        """
        Args:
            factor_weights: 因子权重，如 {'RSI_1': 0.2, 'RET_20': 0.25}
            factor_directions: 因子方向，1=越大越好，-1=越小越好
        """
        self.factor_weights = factor_weights
        self.factor_directions = factor_directions
        self._processor = FactorProcessor()

    def rank(self, factor_data: pd.DataFrame, top_n: int = 5) -> List[str]:
        """
        对候选股排序，返回 top_n 只股票代码

        Args:
            factor_data: DataFrame，index=股票代码，columns=因子值
            top_n: 返回前 N 只

        Returns:
            排序后的股票代码列表
        """
        if factor_data.empty:
            return []

        scores = self._calculate_scores(factor_data)
        sorted_codes = scores.sort_values(ascending=False).head(top_n)
        return sorted_codes.index.tolist()

    def _calculate_scores(self, factor_data: pd.DataFrame) -> pd.Series:
        """计算每只股票的加权综合得分"""
        total_weight = sum(self.factor_weights.values())
        if total_weight == 0:
            return pd.Series(0.0, index=factor_data.index)

        composite = pd.Series(0.0, index=factor_data.index)

        for factor, weight in self.factor_weights.items():
            if factor not in factor_data.columns:
                continue

            raw = factor_data[factor].copy()
            # zscore 标准化
            normalized = self._processor.z_score(raw)
            # 方向调整：-1 则反转
            direction = self.factor_directions.get(factor, 1)
            if direction == -1:
                normalized = 1 - normalized
            # 加权
            composite += normalized * weight / total_weight

        return composite
