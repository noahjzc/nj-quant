# strategy/ml/temporal/temporal_trainer.py
"""Phase 2: 用预训练 Encoder 提取特征 → LightGBM 训练"""
import logging
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import joblib
import lightgbm as lgb

from strategy.ml.temporal.encoder import TemporalEncoder

logger = logging.getLogger(__name__)


class TemporalTrainer:
    """用预训练 Encoder 从训练数据提取时序特征，训练 LightGBM。"""

    def __init__(self, cache_dir: str, encoder_path: str):
        self.cache_dir = Path(cache_dir)
        self.daily_dir = self.cache_dir / 'daily'

        ckpt = torch.load(encoder_path, map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        self.factor_columns = cfg['factor_columns']
        self.seq_len = cfg.get('max_len', 20)
        self.d_model = cfg['d_model']

        self.encoder = TemporalEncoder(
            n_features=cfg['n_features'], d_model=cfg['d_model'],
            n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            max_len=self.seq_len,
        )
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.encoder.eval()

    def train(
        self,
        train_start: str,
        train_end: str,
        output_path: str,
        params: dict = None,
        purge_days: int = 5,
    ) -> str:
        """联合训练: 提取时序特征 → 拼截面因子 → 训练 LightGBM。

        Returns:
            模型保存路径
        """
        all_dates = sorted([f.stem for f in self.daily_dir.glob('*.parquet')])
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        dates = [d for d in all_dates if train_start <= d <= train_end]

        if purge_days > 0:
            end_idx = date_to_idx[dates[-1]]
            cutoff_idx = max(0, end_idx - purge_days)
            dates = [d for d in dates if d <= all_dates[cutoff_idx]]

        if len(dates) < 30:
            raise ValueError(f"训练日期不足: {len(dates)} 天")

        logger.info(f"训练日期: {dates[0]} ~ {dates[-1]}, 共 {len(dates)} 天")

        history: dict = {}
        M = len(self.factor_columns)
        X_chunks = []
        y_chunks = []

        for i, date_str in enumerate(dates):
            idx = date_to_idx[date_str]
            if idx < self.seq_len - 1:
                continue
            target_idx = idx + 5
            if target_idx >= len(all_dates):
                continue

            daily_path = self.daily_dir / f'{date_str}.parquet'
            target_path = self.daily_dir / f'{all_dates[target_idx]}.parquet'
            if not target_path.exists():
                continue

            df = pd.read_parquet(daily_path).set_index('stock_code')
            target_df = pd.read_parquet(target_path).set_index('stock_code')

            cols = [c for c in self.factor_columns if c in df.columns]
            if len(cols) < len(self.factor_columns) * 0.5:
                continue
            common = df.index.intersection(target_df.index)
            if len(common) < 100:
                continue

            # Update history per stock
            for s in common:
                if s not in history:
                    history[s] = deque(maxlen=self.seq_len)
                history[s].append(np.array(
                    [float(df.loc[s].get(c, 0.0)) for c in self.factor_columns],
                    dtype=np.float32
                ))

            # Build temporal tensor
            stock_list = list(common)
            hist_array = np.zeros((len(stock_list), self.seq_len, M), dtype=np.float32)
            for j, s in enumerate(stock_list):
                h = list(history[s])
                for t_idx in range(min(len(h), self.seq_len)):
                    hist_array[j, self.seq_len - len(h) + t_idx] = h[t_idx]

            # Encoder extract
            with torch.no_grad():
                temporal_feats = self.encoder(torch.tensor(hist_array)).numpy()

            # Cross-sectional factors
            cross_feats = np.array([
                [float(df.loc[s].get(c, 0.0)) for c in self.factor_columns]
                for s in stock_list
            ], dtype=np.float32)

            # Concatenate
            X = np.concatenate([cross_feats, temporal_feats], axis=1)

            # Labels: 5-day forward return
            y = np.array([
                (target_df.loc[s, 'close'] - df.loc[s, 'close']) / df.loc[s, 'close']
                for s in stock_list
            ], dtype=np.float32)

            # Filter extreme labels
            mask = (y > -0.5) & (y < 0.5)
            if mask.sum() < 100:
                continue
            X_chunks.append(X[mask])
            y_chunks.append(y[mask])

            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i+1}/{len(dates)}, 样本累计: {sum(len(c) for c in X_chunks)}")

        if not X_chunks:
            raise ValueError("未生成任何训练样本")

        X_all = np.vstack(X_chunks)
        y_all = np.concatenate(y_chunks)
        logger.info(f"训练集: X={X_all.shape}, y_mean={y_all.mean():.6f}, y_std={y_all.std():.6f}")

        # Train LightGBM
        default_params = {
            'objective': 'regression', 'metric': 'rmse',
            'boosting_type': 'gbdt', 'num_leaves': 63,
            'learning_rate': 0.05, 'n_estimators': 500,
            'early_stopping_rounds': 50, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1,
            'reg_lambda': 0.1, 'min_child_samples': 100,
            'verbosity': -1, 'random_state': 42,
        }
        if params:
            default_params.update(params)

        split_idx = int(len(X_all) * 0.8)
        purge_size = max(1, int(len(X_all) * 0.02))
        val_start = min(split_idx + purge_size, len(X_all) - 1)

        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_all[:split_idx], y_all[:split_idx],
            eval_set=[(X_all[val_start:], y_all[val_start:])],
        )

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(output_dir / 'best_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"模型已保存: {model_path}")

        # Print feature importance
        importance = model.feature_importances_
        n_cross = M
        top_idx = np.argsort(-importance)[:10]
        top_features = [(f"factor_{i}" if i < n_cross else f"temp_{i-n_cross}",
                         importance[i]) for i in top_idx]
        logger.info(f"Top 10 特征: {top_features}")

        return model_path
