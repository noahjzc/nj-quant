"""TemporalMLRanker — Encoder + LightGBM 联合推理"""
import logging
from collections import deque
from typing import List

import numpy as np
import pandas as pd
import torch
import joblib

from strategy.ml.temporal.encoder import TemporalEncoder

logger = logging.getLogger(__name__)


class TemporalMLRanker:
    """带时序编码的 ML 排名器。

    接口与 MLRanker 完全一致: rank(factor_df, top_n) → List[str]。
    内部维护每只股票的因子历史缓存 (deque, maxlen=20)。
    """

    def __init__(self, model_path: str, encoder_path: str, seq_len: int = 20):
        self.model = joblib.load(model_path)

        checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        self.encoder = TemporalEncoder(
            n_features=cfg['n_features'],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            max_len=cfg.get('max_len', seq_len),
        )
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.encoder.eval()

        self._feature_names = cfg['factor_columns']
        self._seq_len = seq_len
        self._n_features = len(self._feature_names)
        self._d_model = cfg['d_model']

        self._history: dict = {}

    @property
    def required_features(self) -> List[str]:
        return self._feature_names

    def rank(self, factor_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        if factor_df.empty:
            return []

        stocks = factor_df.index.tolist()
        M = self._n_features

        # 1. 更新历史缓存
        for stock in stocks:
            if stock not in self._history:
                self._history[stock] = deque(maxlen=self._seq_len)
                for _ in range(self._seq_len):
                    self._history[stock].append(np.zeros(M, dtype=np.float32))
            values = np.array(
                [float(factor_df.loc[stock].get(f, 0.0)) for f in self._feature_names],
                dtype=np.float32
            )
            self._history[stock].append(values)

        # 2. 构建时序 tensor: (N, seq_len, M)
        history_array = np.array([list(self._history[s]) for s in stocks], dtype=np.float32)
        history_tensor = torch.tensor(history_array)

        # 3. Encoder → (N, d_model)
        with torch.no_grad():
            temporal_feats = self.encoder(history_tensor).numpy()

        # 4. 截面因子: (N, M)
        cross_features = np.array([
            [float(factor_df.loc[s].get(f, 0.0)) for f in self._feature_names]
            for s in stocks
        ], dtype=np.float32)

        # 5. 拼接 → LightGBM 预测
        X = np.concatenate([cross_features, temporal_feats], axis=1)
        predictions = self.model.predict(X)

        # 6. 排序返回
        order = np.argsort(-predictions)
        return [stocks[i] for i in order[:top_n]]
