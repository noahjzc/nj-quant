"""ML 排名器训练模块 — 从 Parquet 缓存构造训练数据 + 训练 LightGBM"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from strategy.factors.alpha158 import Alpha158Calculator

logger = logging.getLogger(__name__)

_ALPHA_COLUMNS = None


def _get_alpha_columns() -> list:
    global _ALPHA_COLUMNS
    if _ALPHA_COLUMNS is None:
        calc = Alpha158Calculator()
        dummy = pd.DataFrame({
            'open': [10.0, 10.1, 10.2], 'high': [10.3, 10.2, 10.4],
            'low': [9.8, 9.9, 9.7], 'close': [10.0, 10.1, 10.0],
            'volume': [1e6, 1.1e6, 0.9e6],
        })
        _ALPHA_COLUMNS = list(calc.compute(dummy).columns)
    return _ALPHA_COLUMNS


class MLRankerTrainer:
    """从 Parquet 缓存构造训练数据并训练 LightGBM 模型"""

    def __init__(self, cache_dir: str, daily_subdir: str = 'daily'):
        self.cache_dir = Path(cache_dir)
        self.daily_dir = self.cache_dir / daily_subdir
        self.alpha_columns = _get_alpha_columns()

    def build_dataset(
        self,
        train_start: str,
        train_end: str,
        purge_days: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """构造训练集: 逐日读取 Parquet，拼接 X(因子) + y(5日收益)。

        Args:
            train_start: 训练开始日期 'YYYY-MM-DD'
            train_end: 训练截止日期 'YYYY-MM-DD'
            purge_days: 训练截止日前的缓冲交易日数，防止标签延伸到回测期

        Returns:
            (X, y): 特征 DataFrame 和标签 Series
        """
        all_dates = sorted([f.stem for f in self.daily_dir.glob('*.parquet')])
        if not all_dates:
            raise ValueError(f"缓存目录为空: {self.daily_dir}")

        # 日期→索引映射，避免 O(m) 的 list.index()
        date_to_idx: Dict[str, int] = {d: i for i, d in enumerate(all_dates)}
        logger.info(f"缓存日期范围: {all_dates[0]} ~ {all_dates[-1]}")

        # 筛选训练期内的日期
        train_dates = [d for d in all_dates if train_start <= d <= train_end]
        if not train_dates:
            raise ValueError(f"训练期内无缓存数据: {train_start} ~ {train_end}")

        # purge_days: 训练截止日向前推，防止标签延伸到回测期
        if purge_days > 0:
            end_idx = date_to_idx[train_dates[-1]]
            cutoff_idx = max(0, end_idx - purge_days)
            train_dates = [d for d in train_dates if d <= all_dates[cutoff_idx]]

        if len(train_dates) < 10:
            raise ValueError(f"训练日期不足: {len(train_dates)} 天")

        logger.info(f"训练日期: {train_dates[0]} ~ {train_dates[-1]}, "
                    f"共 {len(train_dates)} 天")

        # 预加载所有文件（减少重复 I/O）
        daily_cache: Dict[str, pd.DataFrame] = {}
        needed_dates = set(train_dates)
        for d in train_dates:
            target_idx = date_to_idx[d] + 5
            if target_idx < len(all_dates):
                needed_dates.add(all_dates[target_idx])

        for d in needed_dates:
            path = self.daily_dir / f'{d}.parquet'
            if path.exists():
                daily_cache[d] = pd.read_parquet(path)

        # 取第一个有效日期的列作为基准，保证所有 chunk 列一致
        first_df = None
        for d in train_dates:
            if d in daily_cache and not daily_cache[d].empty:
                first_df = daily_cache[d]
                break
        if first_df is None:
            raise ValueError("无有效训练数据")

        canonical_cols = [c for c in self.alpha_columns if c in first_df.columns]
        if len(canonical_cols) < 50:
            raise ValueError(f"Alpha 因子列不足: {len(canonical_cols)}")

        X_chunks = []
        y_chunks = []

        for i, date_str in enumerate(train_dates):
            df = daily_cache.get(date_str)
            if df is None or df.empty:
                continue

            target_idx = date_to_idx[date_str] + 5
            if target_idx >= len(all_dates):
                continue

            target_date = all_dates[target_idx]
            target_df = daily_cache.get(target_date)
            if target_df is None or target_df.empty:
                continue

            # 用 merge 对齐，避免依赖排序后的位置一致性
            merged = df[['stock_code', 'close'] + canonical_cols].merge(
                target_df[['stock_code', 'close']],
                on='stock_code', suffixes=('', '_target'), how='inner'
            )
            if len(merged) < 100:
                continue

            X = merged[canonical_cols].values.astype(np.float32)
            close_today = merged['close'].values.astype(np.float32)
            close_target = merged['close_target'].values.astype(np.float32)

            # y = (close_t+5 - close_t) / close_t
            y = (close_target - close_today) / close_today

            # 过滤极端标签（新股/复牌导致的异常跳变）
            mask = (y > -0.5) & (y < 0.5)
            if mask.sum() < 100:
                continue

            X_chunks.append(X[mask])
            y_chunks.append(y[mask])

            if (i + 1) % 50 == 0:
                logger.info(f"  处理 {i + 1}/{len(train_dates)}: {date_str}, "
                           f"样本累计 {sum(len(c) for c in X_chunks)}")

        if not X_chunks:
            raise ValueError("未生成任何训练样本")

        X_all = np.vstack(X_chunks)
        y_all = np.concatenate(y_chunks)
        X_df = pd.DataFrame(X_all, columns=canonical_cols)

        logger.info(f"训练集: X={X_df.shape}, y={len(y_all)}, "
                    f"y_mean={y_all.mean():.6f}, y_std={y_all.std():.6f}")

        return X_df, y_all

    def train(
        self,
        train_start: str,
        train_end: str,
        model_path: str = None,
        params: dict = None,
        purge_days: int = 5,
    ) -> str:
        """训练 LightGBM 并保存模型。

        Args:
            train_start: 训练开始日期
            train_end: 训练截止日期
            model_path: 模型保存路径，默认 strategy/ml/model.pkl
            params: LightGBM 参数
            purge_days: 训练截止日前缓冲交易日数

        Returns:
            模型文件路径
        """
        X, y = self.build_dataset(train_start, train_end, purge_days=purge_days)

        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'early_stopping_rounds': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 100,
            'verbosity': -1,
            'random_state': 42,
        }
        if params:
            default_params.update(params)

        logger.info(f"开始训练 LightGBM: X={X.shape}, 参数={default_params}")

        # 时间序列分割：最后 20% 做验证（加 purge 防止标签重叠）
        purge_size = max(1, int(len(X) * 0.02))  # 2% purge gap
        split_idx = int(len(X) * 0.8)
        val_start = min(split_idx + purge_size, len(X) - 1)

        X_train, X_val = X.iloc[:split_idx], X.iloc[val_start:]
        y_train, y_val = y[:split_idx], y[val_start:]

        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
        )

        if model_path is None:
            model_path = str(Path(__file__).parent / 'model.pkl')

        joblib.dump(model, model_path)
        logger.info(f"模型已保存: {model_path}")

        # 打印特征重要性 Top 10
        importance = model.feature_importances_
        top_idx = np.argsort(-importance)[:10]
        feature_names = X.columns.tolist()
        top_features = [(feature_names[i], importance[i])
                        for i in top_idx if i < len(feature_names)]
        logger.info(f"Top 10 因子: {top_features}")

        return model_path
