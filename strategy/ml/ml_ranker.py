"""ML 排名器 — LightGBM 模型推理，替代 SignalRanker"""
import logging
from typing import List

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class MLRanker:
    """基于 LightGBM 的股票排名器。

    与 SignalRanker 接口完全一致，可直接替换。
    """

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self._feature_names = self.model.feature_name_

    @property
    def required_features(self) -> List[str]:
        """返回模型训练时使用的因子名列表"""
        return self._feature_names

    def rank(self, factor_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """对股票按预测收益排序，返回 top_n 只。

        Args:
            factor_df: DataFrame，index=stock_code，columns=因子值
            top_n: 返回前 N 只

        Returns:
            排序后的股票代码列表
        """
        if factor_df.empty:
            return []

        # 特征缺失警告
        available = set(factor_df.columns)
        expected = set(self._feature_names)
        missing = expected - available
        if len(missing) > len(expected) * 0.3:
            logger.warning(
                "factor_df 缺少 %d/%d 个模型特征，预测可能不可靠。缺失示例: %s",
                len(missing), len(expected),
                list(missing)[:5]
            )

        # 构建完整特征矩阵：缺失列填 0，保留列名避免 sklearn 警告
        X = pd.DataFrame(0.0, index=factor_df.index, columns=self._feature_names, dtype=np.float32)
        for feat in self._feature_names:
            if feat in factor_df.columns:
                X[feat] = factor_df[feat].fillna(0.0).astype(np.float32)

        predictions = self.model.predict(X)

        # 按预测值降序排列
        order = np.argsort(-predictions)
        return factor_df.index[order[:top_n]].tolist()
