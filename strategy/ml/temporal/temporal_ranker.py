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

        # 加载归一化参数
        mean = cfg.get('factor_mean')
        std = cfg.get('factor_std')
        if mean is not None and std is not None:
            self._factor_mean = np.array(mean, dtype=np.float32)
            self._factor_std = np.array(std, dtype=np.float32)
        else:
            logger.warning("checkpoint 不含归一化参数，推理结果可能不可靠")
            self._factor_mean = np.zeros(self._n_features, dtype=np.float32)
            self._factor_std = np.ones(self._n_features, dtype=np.float32)

        self._history: dict = {}
        self._history_hits: dict = {}  # 用于 LRU 淘汰

    @property
    def required_features(self) -> List[str]:
        return self._feature_names

    def rank(self, factor_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        if factor_df.empty:
            return []

        stocks = factor_df.index.tolist()
        M = self._n_features

        # 1. 更新历史缓存（加 LRU 淘汰）
        MAX_CACHE = 6000
        if len(self._history) > MAX_CACHE:
            stale = sorted(self._history_hits.items(), key=lambda x: x[1])[:500]
            for s, _ in stale:
                del self._history[s]
                del self._history_hits[s]

        for stock in stocks:
            self._history_hits[stock] = self._history_hits.get(stock, 0) + 1
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

        # 3. z-score 归一化 (与预训练一致)
        history_array = (history_array - self._factor_mean) / self._factor_std
        history_array = np.nan_to_num(history_array, nan=0.0, posinf=0.0, neginf=0.0)
        history_tensor = torch.tensor(history_array)

        # 4. Encoder → (N, d_model)
        with torch.no_grad():
            temporal_feats = self.encoder(history_tensor).numpy()

        # 5. 截面因子: (N, M)
        cross_features = np.array([
            [float(factor_df.loc[s].get(f, 0.0)) for f in self._feature_names]
            for s in stocks
        ], dtype=np.float32)

        # 6. 拼接 → LightGBM 预测
        X = np.concatenate([cross_features, temporal_feats], axis=1)
        predictions = self.model.predict(X)

        # 7. 排序返回
        order = np.argsort(-predictions)
        return [stocks[i] for i in order[:top_n]]
