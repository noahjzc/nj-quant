import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import os
import tempfile
import pandas as pd
import numpy as np
import torch
import joblib
import lightgbm as lgb
from strategy.ml.temporal.temporal_ranker import TemporalMLRanker
from strategy.ml.temporal.encoder import TemporalEncoder


def test_temporal_ranker_rank():
    """基本 rank 测试: 接口与 MLRanker 一致"""
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = TemporalEncoder(n_features=3, d_model=16, n_heads=2, n_layers=1)
        encoder_path = os.path.join(tmpdir, 'encoder.pt')
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': {'n_features': 3, 'd_model': 16, 'n_heads': 2,
                       'n_layers': 1, 'max_len': 20, 'factor_columns': ['a', 'b', 'c']},
        }, encoder_path)

        X = np.random.randn(100, 3 + 16).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        model = lgb.LGBMRegressor(n_estimators=10, verbosity=-1)
        model.fit(X, y)
        model_path = os.path.join(tmpdir, 'model.pkl')
        joblib.dump(model, model_path)

        ranker = TemporalMLRanker(model_path, encoder_path)

        stocks = [f'sh00000{i}' for i in range(10)]
        for _ in range(25):
            factor_df = pd.DataFrame(
                np.random.randn(10, 3).astype(np.float32),
                index=stocks, columns=['a', 'b', 'c']
            )
            result = ranker.rank(factor_df, top_n=5)

        assert len(result) == 5
        assert all(s in stocks for s in result)


def test_temporal_ranker_history_warmup():
    """历史不足 20 天时零填充，不崩溃"""
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = TemporalEncoder(n_features=2, d_model=8, n_heads=2, n_layers=1)
        encoder_path = os.path.join(tmpdir, 'encoder.pt')
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': {'n_features': 2, 'd_model': 8, 'n_heads': 2,
                       'n_layers': 1, 'max_len': 20, 'factor_columns': ['x', 'y']},
        }, encoder_path)

        X = np.random.randn(50, 2 + 8).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        model = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
        model.fit(X, y)
        model_path = os.path.join(tmpdir, 'model.pkl')
        joblib.dump(model, model_path)

        ranker = TemporalMLRanker(model_path, encoder_path)

        stocks = [f'sh00000{i}' for i in range(5)]
        for _ in range(5):
            factor_df = pd.DataFrame(
                np.random.randn(5, 2).astype(np.float32),
                index=stocks, columns=['x', 'y']
            )
            result = ranker.rank(factor_df, top_n=3)
        assert len(result) == 3
