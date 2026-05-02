import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import tempfile
import numpy as np
import pandas as pd
import joblib
import pytest
from strategy.ml.ml_ranker import MLRanker
from strategy.ml.trainer import MLRankerTrainer


class TestMLRanker:
    def setup_method(self):
        """训练一个微型 LightGBM 模型用于测试"""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            np.random.normal(0, 1, (n, 5)),
            columns=['F1', 'F2', 'F3', 'F4', 'F5']
        )
        true_weights = np.array([0.3, -0.2, 0.1, 0.0, -0.1])
        noise = np.random.normal(0, 0.1, n)
        y = X @ true_weights + noise

        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbosity=-1)
        self.model.fit(X, y)

        self.tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        joblib.dump(self.model, self.tmp.name)
        self.tmp.close()

        self.ranker = MLRanker(self.tmp.name)

    def teardown_method(self):
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_required_features(self):
        assert self.ranker.required_features == ['F1', 'F2', 'F3', 'F4', 'F5']

    def test_rank_returns_list(self):
        df = pd.DataFrame({
            'F1': [0.5, -0.3, 1.2],
            'F2': [-0.1, 0.8, 0.3],
            'F3': [0.2, -0.5, 0.1],
            'F4': [0.1, 0.1, 0.1],
            'F5': [0.0, 0.0, 0.0],
        }, index=['A', 'B', 'C'])

        result = self.ranker.rank(df, top_n=2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(r in ['A', 'B', 'C'] for r in result)

    def test_rank_sorted_descending(self):
        """高分在前的应排在前面"""
        df = pd.DataFrame({
            'F1': [2.0, -2.0, 0.0],
            'F2': [-1.0, 1.0, 0.0],
            'F3': [0.0, 0.0, 0.0],
            'F4': [0.0, 0.0, 0.0],
            'F5': [0.0, 0.0, 0.0],
        }, index=['HIGH', 'LOW', 'MID'])

        result = self.ranker.rank(df, top_n=3)
        # F1 权重 0.3 最高，F2 权重 -0.2，所以 HIGH 应该排最前
        assert result[0] == 'HIGH'

    def test_rank_top_n_limit(self):
        df = pd.DataFrame({
            'F1': np.random.normal(0, 1, 10),
            'F2': np.random.normal(0, 1, 10),
            'F3': np.random.normal(0, 1, 10),
            'F4': np.random.normal(0, 1, 10),
            'F5': np.random.normal(0, 1, 10),
        }, index=[f'stock_{i}' for i in range(10)])

        result = self.ranker.rank(df, top_n=3)
        assert len(result) == 3

    def test_rank_empty_returns_empty(self):
        df = pd.DataFrame({'F1': [], 'F2': [], 'F3': [], 'F4': [], 'F5': []})
        result = self.ranker.rank(df, top_n=5)
        assert result == []

    def test_missing_columns_handled(self):
        """缺少部分特征列不应崩溃"""
        df = pd.DataFrame({
            'F1': [0.5, -0.3],
            'F2': [-0.1, 0.8],
        }, index=['A', 'B'])

        result = self.ranker.rank(df, top_n=2)
        assert len(result) == 2
