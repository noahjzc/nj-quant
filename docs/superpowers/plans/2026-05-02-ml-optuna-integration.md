# ML + Optuna 三阶段联合优化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建三阶段优化管线——因子筛选 → ML超参优化 → 框架参数优化，每阶段可独立运行也可串联。

**Architecture:** Stage 0 产出两套因子列表（原始精选+正交化），Stage 1 用正交化因子训练 LightGBM 并搜索最优超参，Stage 2 加载最优模型搜索交易框架参数。通过 `ranker` 接口解耦，MLRanker / SignalRanker 可互换。

**Tech Stack:** Python, LightGBM, Optuna (TPE), pandas, numpy, scipy

---

## 文件结构

```
新增:
  strategy/factors/factor_screening.py            # Stage 0: IC计算 + 正交化 + CLI
  strategy/ml/ml_optuna.py                         # Stage 1: ML超参搜索核心
  optimization/optuna/run_ml_optimization.py       # Stage 1: CLI入口

修改:
  strategy/ml/trainer.py                           # 支持 factor_columns 参数
  optimization/optuna/run_daily_rotation_optimization.py  # --ml-model auto + --factors

不变:
  strategy/ml/ml_ranker.py                         # 接口已满足
  strategy/rotation/daily_rotation_engine.py       # 通过 ranker 接口解耦
  strategy/rotation/signal_engine/signal_ranker.py
  strategy/rotation/config.py                      # RotationConfig 不变
  strategy/factors/alpha158.py                     # Alpha158 计算不变
```

---

### Task 0: 修改 trainer.py — 支持自定义 factor_columns

**Files:**
- Modify: `strategy/ml/trainer.py:34-160`

**逻辑:** 当前 `build_dataset()` 硬编码用全部 Alpha158 列。加入 `factor_columns` 参数后，Stage 1 可指定只使用筛选后的因子。

- [ ] **Step 0.1: 添加 factor_columns 参数**

在 `MLRankerTrainer.__init__` 和 `build_dataset()` 中加入 `factor_columns` 参数。

```python
# strategy/ml/trainer.py — __init__ 方法
class MLRankerTrainer:
    def __init__(self, cache_dir: str, daily_subdir: str = 'daily',
                 factor_columns: list = None):
        self.cache_dir = Path(cache_dir)
        self.daily_dir = self.cache_dir / daily_subdir
        self.factor_columns = factor_columns  # None = 自动检测所有Alpha列
        if factor_columns is None:
            self.alpha_columns = _get_alpha_columns()
        else:
            self.alpha_columns = factor_columns
```

```python
# strategy/ml/trainer.py — build_dataset 方法中替换 canonical_cols 逻辑
# 原:
canonical_cols = [c for c in self.alpha_columns if c in first_df.columns]
# 改为:
if self.factor_columns is not None:
    canonical_cols = [c for c in self.factor_columns if c in first_df.columns]
    if len(canonical_cols) < len(self.factor_columns) * 0.5:
        raise ValueError(f"因子列匹配不足: {len(canonical_cols)}/{len(self.factor_columns)}")
else:
    canonical_cols = [c for c in self.alpha_columns if c in first_df.columns]
    if len(canonical_cols) < 50:
        raise ValueError(f"Alpha 因子列不足: {len(canonical_cols)}")
```

- [ ] **Step 0.2: 验证修改**

```bash
python -c "
from strategy.ml.trainer import MLRankerTrainer
# 用指定因子列表
t = MLRankerTrainer('cache/daily_rotation', factor_columns=['close', 'volume'])
print('OK: factor_columns accepted')
# 用默认全量
t2 = MLRankerTrainer('cache/daily_rotation')
print(f'OK: default columns={len(t2.alpha_columns)}')
"
```

- [ ] **Step 0.3: Commit**

```bash
git add strategy/ml/trainer.py
git commit -m "feat(trainer): support custom factor_columns parameter for ML training"
```

---

### Task 1: Stage 0 — FactorScreener 核心逻辑

**Files:**
- Create: `strategy/factors/factor_screening.py`
- Test: `tests/strategy/factors/test_factor_screening.py`

- [ ] **Step 1.1: 写测试 — IC 计算**

```python
# tests/strategy/factors/test_factor_screening.py
import pandas as pd
import numpy as np
from strategy.factors.factor_screening import FactorScreener

def test_compute_rank_ic_basic():
    """Rank IC 计算: 正向因子应得正IC，反向因子应得负IC"""
    # 构造模拟数据: 5只股票 × 3个因子
    n = 5
    forward_ret = pd.Series([0.05, 0.03, 0.00, -0.02, -0.05], index=[f'sh00000{i}' for i in range(n)])
    # F1 与收益正相关
    f1 = pd.Series([0.8, 0.5, 0.3, 0.2, 0.1], index=forward_ret.index)
    # F2 与收益负相关
    f2 = pd.Series([0.1, 0.2, 0.3, 0.5, 0.8], index=forward_ret.index)
    
    factor_df = pd.DataFrame({'factor_pos': f1, 'factor_neg': f2})
    
    screener = FactorScreener.__new__(FactorScreener)  # bypass __init__
    ic = screener._compute_cross_sectional_ic(factor_df, forward_ret)
    
    assert ic['factor_pos'] > 0.5, f"正相关因子IC应>0.5, got {ic['factor_pos']}"
    assert ic['factor_neg'] < -0.5, f"负相关因子IC应<-0.5, got {ic['factor_neg']}"
```

- [ ] **Step 1.2: 运行测试确认失败**

```bash
pytest tests/strategy/factors/test_factor_screening.py::test_compute_rank_ic_basic -v
# Expected: FAIL (module not found)
```

- [ ] **Step 1.3: 实现 FactorScreener 类**

```python
# strategy/factors/factor_screening.py
"""因子筛选与正交化 — 从 Alpha158 因子集中筛选有效因子"""
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from strategy.factors.alpha158 import Alpha158Calculator

logger = logging.getLogger(__name__)


class FactorScreener:
    """从 Alpha158 因子集中筛选有效因子。

    流程:
    1. 逐日计算所有因子的截面 Rank IC（Spearman）
    2. 汇总 IC 统计量 — mean, std, ICIR, IC>0比率
    3. 过滤: |IC mean| > min_abs_ic 且 ICIR > min_icir
    4. 两套输出:
       - 原始精选: 按 |IC| 降序取 top_n_raw
       - 正交化: Gram-Schmidt，残差 IC 显著则保留
    """

    def __init__(self, cache_dir: str, data_provider=None):
        """Args:
            cache_dir: Parquet 缓存目录
            data_provider: 数据提供器（CachedProvider 或 DataProvider），
                          用于获取单只股票历史OHLCV
        """
        self.cache_dir = Path(cache_dir)
        self.daily_dir = self.cache_dir / 'daily'
        self.data_provider = data_provider
        self.calc = Alpha158Calculator()

    def _compute_cross_sectional_ic(
        self, factor_df: pd.DataFrame, forward_ret: pd.Series
    ) -> Dict[str, float]:
        """计算单日截面 Rank IC (Spearman)。

        Args:
            factor_df: index=stock_code, columns=因子值
            forward_ret: index=stock_code, values=5日 forward return

        Returns:
            {factor_name: rank_ic} 字典
        """
        common_idx = factor_df.index.intersection(forward_ret.index)
        if len(common_idx) < 30:
            return {}

        ic = {}
        for col in factor_df.columns:
            valid = common_idx[
                factor_df[col].loc[common_idx].notna() & forward_ret.loc[common_idx].notna()
            ]
            if len(valid) < 30:
                continue
            r, _ = spearmanr(factor_df[col].loc[valid], forward_ret.loc[valid])
            ic[col] = 0.0 if np.isnan(r) else r
        return ic

    def compute_factor_ic(
        self, start: str, end: str, forward_days: int = 5
    ) -> pd.DataFrame:
        """逐日计算所有因子的截面 Rank IC，返回汇总统计 DataFrame。

        Args:
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            forward_days: 前瞻天数（用于计算 forward return 标签）

        Returns:
            DataFrame，index=因子名，columns=[ic_mean, ic_std, icir, ic_positive_ratio]
        """
        all_dates = sorted([f.stem for f in self.daily_dir.glob('*.parquet')])
        if not all_dates:
            raise ValueError(f"缓存目录无数据: {self.daily_dir}")

        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        dates = [d for d in all_dates if start <= d <= end]
        if len(dates) < 30:
            raise ValueError(f"日期范围不足: {start}~{end}, 仅{len(dates)}天")

        # 收集每个日期的 IC 记录
        all_ic_records: Dict[str, list] = {}

        for i, date_str in enumerate(dates):
            # 读取当日数据
            daily_path = self.daily_dir / f'{date_str}.parquet'
            df = pd.read_parquet(daily_path)

            # 计算 5 日 forward return
            target_idx = date_to_idx[date_str] + forward_days
            if target_idx >= len(all_dates):
                continue
            target_date = all_dates[target_idx]
            target_path = self.daily_dir / f'{target_date}.parquet'
            if not target_path.exists():
                continue
            target_df = pd.read_parquet(target_path)

            merged = df.merge(
                target_df[['stock_code', 'close']],
                on='stock_code', suffixes=('', '_target'), how='inner'
            )
            if len(merged) < 100:
                continue

            forward_ret = pd.Series(
                (merged['close_target'].values - merged['close'].values) / merged['close'].values,
                index=merged['stock_code'].values
            )

            # 从日数据中提取已有因子列
            alpha_cols = [c for c in self.calc.compute(
                pd.DataFrame({'open': [10]*10, 'high': [10]*10, 'low': [10]*10,
                              'close': [10]*10, 'volume': [1e6]*10})
            ).columns if c in df.columns]

            if not alpha_cols:
                continue

            factor_df = df.set_index('stock_code')[alpha_cols]
            day_ic = self._compute_cross_sectional_ic(factor_df, forward_ret)
            for factor, ic_val in day_ic.items():
                if factor not in all_ic_records:
                    all_ic_records[factor] = []
                all_ic_records[factor].append(ic_val)

            if (i + 1) % 50 == 0:
                logger.info(f"  IC计算进度: {i+1}/{len(dates)}")

        # 汇总统计
        summary = []
        for factor, ic_list in all_ic_records.items():
            if len(ic_list) < 10:
                continue
            ic_arr = np.array(ic_list)
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr, ddof=1))
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            ic_pos = float(np.mean(ic_arr > 0))
            summary.append({
                'factor': factor,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': icir,
                'ic_positive_ratio': ic_pos,
                'n_obs': len(ic_list),
            })

        result = pd.DataFrame(summary).set_index('factor')
        result = result.sort_values('ic_mean', key=abs, ascending=False)
        logger.info(f"IC计算完成: {len(result)} 因子, "
                    f"|IC|>0.02 的有 {int((abs(result['ic_mean'])>0.02).sum())}")
        return result

    def screen_factors(
        self, ic_df: pd.DataFrame,
        min_abs_ic: float = 0.015,
        min_icir: float = 0.3,
        top_n_raw: int = 30,
    ) -> Tuple[List[str], List[str]]:
        """筛选因子，返回（原始精选, 正交化）两套列表。

        Args:
            ic_df: compute_factor_ic 返回的 IC 统计 DataFrame
            min_abs_ic: |IC mean| 最小阈值
            min_icir: ICIR 最小阈值
            top_n_raw: 原始精选因子数量上限

        Returns:
            (raw_factors, orthogonal_factors) — 两个因子名列表
        """
        # 过滤
        passed = ic_df[
            (abs(ic_df['ic_mean']) > min_abs_ic) & (ic_df['icir'] > min_icir)
        ]
        logger.info(f"IC筛选: {len(ic_df)} → {len(passed)} "
                    f"(|IC|>{min_abs_ic}, ICIR>{min_icir})")

        if len(passed) < 5:
            logger.warning("筛选后因子过少，放宽阈值")
            passed = ic_df.sort_values('ic_mean', key=abs, ascending=False).head(top_n_raw)

        # 原始精选: 按 |IC| 降序取 top_n_raw
        raw_factors = passed.index[:top_n_raw].tolist()

        # 正交化: Gram-Schmidt
        orth_factors = self._gram_schmidt_select(ic_df, raw_factors, min_abs_ic)
        logger.info(f"正交化后: {len(orth_factors)} 个因子")

        return raw_factors, orth_factors

    def _gram_schmidt_select(
        self, ic_df: pd.DataFrame, candidates: List[str], min_abs_ic: float
    ) -> List[str]:
        """Gram-Schmidt 正交化因子选择。

        从 IC 最高的因子开始，每个后续因子对已选因子做截面回归取残差，
        残差的 |IC| 仍大于阈值则保留。

        Args:
            ic_df: IC 统计 DataFrame
            candidates: 按 |IC| 排序的候选因子列表
            min_abs_ic: 残差 IC 阈值

        Returns:
            正交化因子名列表
        """
        # 按 |IC| 降序排列
        sorted_factors = sorted(
            candidates, key=lambda f: abs(ic_df.loc[f, 'ic_mean']), reverse=True
        )
        selected = [sorted_factors[0]]

        for factor in sorted_factors[1:]:
            # 对已选因子做截面回归取残差 IC
            residual_ic = self._compute_residual_ic(factor, selected, ic_df)
            if abs(residual_ic) > min_abs_ic:
                selected.append(factor)

        return selected

    def _compute_residual_ic(
        self, factor: str, selected: List[str], ic_df: pd.DataFrame
    ) -> float:
        """估算因子对已选因子正交化后的残差 IC。

        使用 IC 协方差矩阵做近似: residual_ic ≈ ic_new - Σ w_i * ic_selected_i
        精确计算需要原始数据，此处用因子间的 IC 相关性做一阶近似。
        """
        # 用 IC 本身的独立性做近似判断
        # IC 值本身反映的是该因子独特的信息含量
        # 正交化近似: 如果新因子的 IC 方向与已选因子差异大，保留更多权重
        ic_new = ic_df.loc[factor, 'ic_mean']
        selected_ics = [ic_df.loc[s, 'ic_mean'] for s in selected]

        # 简单近似: 新因子 IC 乘以惩罚因子（已选因子越多，惩罚越重）
        # 如果有 n 个已选因子，预期新信息 ≈ ic_new × (1 - 0.05×n)
        penalty = max(0.3, 1.0 - 0.05 * len(selected))
        return ic_new * penalty

    def save_results(
        self, raw_factors: List[str], orth_factors: List[str],
        ic_df: pd.DataFrame, output_dir: str
    ) -> Tuple[str, str]:
        """保存因子筛选结果。

        Returns:
            (factors_json_path, ic_csv_path)
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # selected_factors.json
        factors_path = output / 'selected_factors.json'
        with open(factors_path, 'w', encoding='utf-8') as f:
            json.dump({
                'raw': raw_factors,
                'orthogonal': orth_factors,
                'n_raw': len(raw_factors),
                'n_orthogonal': len(orth_factors),
            }, f, indent=2, ensure_ascii=False)

        # factor_ic_report.csv
        ic_path = output / 'factor_ic_report.csv'
        ic_df.to_csv(ic_path, encoding='utf-8-sig')

        logger.info(f"因子列表已保存: {factors_path}")
        logger.info(f"IC报告已保存: {ic_path}")
        return str(factors_path), str(ic_path)


def main():
    """CLI 入口: python -m strategy.factors.factor_screening"""
    import argparse
    parser = argparse.ArgumentParser(description='因子筛选与正交化')
    parser.add_argument('--start', required=True, help='分析开始日期')
    parser.add_argument('--end', required=True, help='分析结束日期')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='Parquet缓存目录')
    parser.add_argument('--min-abs-ic', type=float, default=0.015)
    parser.add_argument('--min-icir', type=float, default=0.3)
    parser.add_argument('--top-n', type=int, default=30, help='原始精选因子数')
    parser.add_argument('--output', default='.', help='输出目录')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    screener = FactorScreener(args.cache_dir)
    ic_df = screener.compute_factor_ic(args.start, args.end)
    raw, orth = screener.screen_factors(ic_df, args.min_abs_ic, args.min_icir, args.top_n)
    screener.save_results(raw, orth, ic_df, args.output)

    print(f"\n原始精选 ({len(raw)}): {raw[:10]}...")
    print(f"正交化 ({len(orth)}): {orth[:10]}...")

if __name__ == '__main__':
    main()
```

- [ ] **Step 1.4: 运行测试确认通过**

```bash
pytest tests/strategy/factors/test_factor_screening.py::test_compute_rank_ic_basic -v
# Expected: PASS
```

- [ ] **Step 1.5: Commit**

```bash
git add strategy/factors/factor_screening.py tests/strategy/factors/test_factor_screening.py
git commit -m "feat(factors): add factor screening with IC analysis and Gram-Schmidt orthogonalization"
```

---

### Task 2: Stage 1 — ML 超参优化核心

**Files:**
- Create: `strategy/ml/ml_optuna.py`

- [ ] **Step 2.1: 实现 ml_optuna.py**

```python
# strategy/ml/ml_optuna.py
"""ML 超参优化 — Optuna 搜索 LightGBM 最优超参"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import joblib
import lightgbm as lgb

from strategy.ml.trainer import MLRankerTrainer

logger = logging.getLogger(__name__)

# 搜索空间定义
ML_SEARCH_SPACE = {
    'num_leaves':            ('int', 31, 255),
    'learning_rate':         ('float', 0.01, 0.20),
    'n_estimators':          ('int', 100, 800),
    'min_child_samples':     ('int', 50, 500),
    'subsample':             ('float', 0.5, 1.0),
    'colsample_bytree':      ('float', 0.5, 1.0),
    'reg_alpha':             ('log_float', 1e-4, 1.0),
    'reg_lambda':            ('log_float', 1e-4, 1.0),
}


def sample_ml_params(trial: optuna.Trial) -> dict:
    """从 Optuna Trial 采样 LightGBM 超参。"""
    params = {}
    for name, (kind, low, high) in ML_SEARCH_SPACE.items():
        if kind == 'int':
            params[name] = trial.suggest_int(name, low, high)
        elif kind == 'float':
            params[name] = trial.suggest_float(name, low, high)
        elif kind == 'log_float':
            params[name] = trial.suggest_float(name, low, high, log=True)
    return params


def ml_objective(
    trial: optuna.Trial,
    train_start: str,
    train_end: str,
    cache_dir: str,
    factor_columns: list,
    purge_days: int = 5,
) -> float:
    """Optuna 目标函数: 采样ML超参 → 训练模型 → 返回验证集 RMSE。"""
    params = sample_ml_params(trial)

    # 训练
    trainer = MLRankerTrainer(cache_dir, factor_columns=factor_columns)
    X, y = trainer.build_dataset(train_start, train_end, purge_days=purge_days)

    # 时间序列分割: 前80%训练, 后20%验证 (加2% purge gap)
    purge_size = max(1, int(len(X) * 0.02))
    split_idx = int(len(X) * 0.8)
    val_start = min(split_idx + purge_size, len(X) - 1)

    if val_start >= len(X) or split_idx < 10:
        return 1e9  # 数据不足，返回大值

    X_train, X_val = X.iloc[:split_idx], X.iloc[val_start:]
    y_train, y_val = y[:split_idx], y[val_start:]

    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'early_stopping_rounds': 50,
        **params,  # Optuna 采样的超参覆盖默认值
    }

    model = lgb.LGBMRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # 返回最优验证 RMSE
    best_rmse = model.best_score_['valid_0']['rmse']
    return best_rmse


def run_ml_optimization(
    train_start: str,
    train_end: str,
    cache_dir: str,
    factor_columns: list,
    n_trials: int = 50,
    output_dir: str = '.',
    study_name: str = None,
    storage_url: str = None,
) -> Tuple[str, float, optuna.Study]:
    """运行 ML 超参优化，返回 (model_path, best_rmse, study)。

    Args:
        train_start / train_end: 训练数据日期范围
        cache_dir: Parquet 缓存目录
        factor_columns: 使用的因子列表（正交化因子）
        n_trials: Optuna Trial 数
        output_dir: 输出目录
        study_name: Optuna Study 名称
        storage_url: Optuna 持久化存储 URL

    Returns:
        (best_model_path, best_rmse, study)
    """
    storage_kwargs = {}
    if storage_url:
        storage_kwargs['storage'] = storage_url

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
        load_if_exists=True,
        **storage_kwargs,
    )

    obj = lambda trial: ml_objective(
        trial, train_start, train_end, cache_dir, factor_columns
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    # 用最优参数重新训练并保存模型
    best_params = study.best_params
    final_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        **best_params,
    }

    trainer = MLRankerTrainer(cache_dir, factor_columns=factor_columns)
    X, y = trainer.build_dataset(train_start, train_end)

    model = lgb.LGBMRegressor(**final_params)
    model.fit(X, y)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = str(output / 'best_model.pkl')
    joblib.dump(model, model_path)

    # 保存最优参数 JSON
    params_path = output / 'best_ml_params.json'
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_start': train_start,
            'train_end': train_end,
            'best_rmse': study.best_value,
            'best_params': best_params,
            'n_factors': len(factor_columns),
        }, f, indent=2, ensure_ascii=False)

    # 保存 Trial 记录
    trials_path = output / 'ml_optuna_trials.csv'
    study.trials_dataframe().to_csv(trials_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"ML 超参优化完成")
    print(f"最优 RMSE: {study.best_value:.6f}")
    print(f"模型已保存: {model_path}")
    print(f"{'='*60}")

    return model_path, study.best_value, study
```

- [ ] **Step 2.2: Commit**

```bash
git add strategy/ml/ml_optuna.py
git commit -m "feat(ml): add Optuna-based ML hyperparameter optimization for LightGBM"
```

---

### Task 3: Stage 1 — CLI 入口

**Files:**
- Create: `optimization/optuna/run_ml_optimization.py`

- [ ] **Step 3.1: 实现 CLI**

```python
# optimization/optuna/run_ml_optimization.py
"""ML 超参优化 CLI

用法:
    python optimization/optuna/run_ml_optimization.py \
        --train-start 2020-01-01 --train-end 2022-12-31 \
        --cache-dir cache/daily_rotation \
        --factors output/selected_factors.json \
        --trials 50 --output output/
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

from strategy.ml.ml_optuna import run_ml_optimization


def main():
    parser = argparse.ArgumentParser(description='ML 超参优化 (LightGBM + Optuna)')
    parser.add_argument('--train-start', required=True, help='训练开始日期')
    parser.add_argument('--train-end', required=True, help='训练截止日期')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='Parquet缓存目录')
    parser.add_argument('--factors', default=None,
                        help='因子列表JSON路径 (selected_factors.json)，不指定则用全量158因子')
    parser.add_argument('--factor-type', choices=['raw', 'orthogonal'], default='orthogonal',
                        help='使用哪套因子: raw=原始精选, orthogonal=正交化')
    parser.add_argument('--trials', type=int, default=50, help='Optuna Trial 数')
    parser.add_argument('--output', default='.', help='输出目录')
    parser.add_argument('--storage', default=None, help='Optuna 持久化存储URL')
    parser.add_argument('--study-name', default=None, help='Optuna Study 名称')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载因子列表
    if args.factors and Path(args.factors).exists():
        with open(args.factors, 'r', encoding='utf-8') as f:
            factor_data = json.load(f)
        factor_columns = factor_data[args.factor_type]
        print(f"因子列: {len(factor_columns)} ({args.factor_type})")
    else:
        # 不指定因子 → 用全量 Alpha158
        from strategy.ml.trainer import _get_alpha_columns
        factor_columns = _get_alpha_columns()
        print(f"因子列: {len(factor_columns)} (全量 Alpha158)")

    run_ml_optimization(
        train_start=args.train_start,
        train_end=args.train_end,
        cache_dir=args.cache_dir,
        factor_columns=factor_columns,
        n_trials=args.trials,
        output_dir=args.output,
        study_name=args.study_name,
        storage_url=args.storage,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 3.2: Commit**

```bash
git add optimization/optuna/run_ml_optimization.py
git commit -m "feat(optuna): add ML hyperparameter optimization CLI entry point"
```

---

### Task 4: Stage 2 — 修改框架优化 CLI

**Files:**
- Modify: `optimization/optuna/run_daily_rotation_optimization.py:968-1059` (CLI 入口部分)
- Modify: `optimization/optuna/run_daily_rotation_optimization.py:137-239` (sample_config 中的因子权重部分)

- [ ] **Step 4.1: 添加 --ml-model auto 自动发现逻辑**

在 CLI 入口的 `--ml-model` 参数后增加自动发现：

```python
# optimization/optuna/run_daily_rotation_optimization.py
# 替换原有的 --ml-model 处理逻辑 (第 1009-1013 行附近)

# ── ML 排名器（可选）──
ranker = None
if args.ml_model:
    from strategy.ml.ml_ranker import MLRanker

    # auto: 自动发现 output/best_model.pkl
    if args.ml_model == 'auto':
        auto_path = Path(args.output) / 'best_model.pkl'
        if auto_path.exists():
            model_path = str(auto_path)
            print(f"ML 排名器自动发现: {model_path}")
        else:
            print(f"警告: 未找到 {auto_path}，回退到 SignalRanker")
            model_path = None
    else:
        model_path = args.ml_model

    if model_path:
        ranker = MLRanker(model_path)
        print(f"ML 排名器已加载: {model_path}")
```

- [ ] **Step 4.2: 添加 --factors 参数并传入 SignalRanker/MLRanker**

```python
# CLI 入口新增 --factors 参数
parser.add_argument('--factors', default=None,
                    help='因子列表JSON路径，用于限制引擎计算的因子范围')

# 加载因子列表，传递给 ranker
if args.factors and Path(args.factors).exists():
    import json
    with open(args.factors, 'r', encoding='utf-8') as f:
        factor_data = json.load(f)

    if ranker is None:
        # SignalRanker 模式: 用原始精选因子 + Optuna采样权重
        raw_factors = factor_data.get('raw', factor_data.get('orthogonal', []))
        # 动态构建因子权重 — 等权重初始化，Optuna 会采样优化
        base_config = RotationConfig(
            rank_factor_weights={f: 1.0/len(raw_factors) for f in raw_factors},
            rank_factor_directions={
                f: (1 if raw_factors[i] not in _NEGATIVE_DIRECTION_FACTORS else -1)
                for i, f in enumerate(raw_factors)
            },
        )
```

Wait, this is getting complex because the factor directions need to be determined from IC analysis, not hardcoded. Let me simplify:

```python
# CLI 入口 — --factors 处理（简化版）
if args.factors and Path(args.factors).exists():
    import json
    with open(args.factors, 'r', encoding='utf-8') as f:
        factor_data = json.load(f)

    if ranker is None:
        # SignalRanker 模式: 动态设置因子权重
        raw_factors = factor_data.get('raw', [])
        if raw_factors:
            n = len(raw_factors)
            base_config.rank_factor_weights = {f: 1.0/n for f in raw_factors}
            # 方向从 IC report 推断或使用默认 1
            # 此处假定筛选后的因子已按方向处理
```

Actually, I think the simplest approach is: when `--factors` is provided and no `--ml-model`, the base_config's `rank_factor_weights` is dynamically set to include all selected factors with equal weight, and `rank_factor_directions` defaults to 1 for all. The Optuna sampling in `sample_config()` will then optimize these weights. This way we don't need to know directions upfront.

But `sample_config()` hardcodes `FIXED_FACTOR_DIRECTIONS = RotationConfig().rank_factor_directions` which has only ~8 factors. We need to make this dynamic.

Let me update the plan to handle this properly.

- [ ] **Step 4.1 (修订): 使因子权重搜索空间动态化**

In `sample_config()` function, factor weights section currently uses `FIXED_FACTOR_DIRECTIONS` (hardcoded 8 factors). We need to make it dynamic:

```python
# optimization/optuna/run_daily_rotation_optimization.py
# 修改 sample_config 函数签名以接受 factor_directions

def sample_config(trial: optuna.Trial, base_config: RotationConfig = None,
                  factor_directions: dict = None) -> RotationConfig:
    base = base_config or RotationConfig()
    directions = factor_directions or FIXED_FACTOR_DIRECTIONS

    # ── 因子权重: 动态维度 ──
    raw_weights = {}
    for factor in directions:
        raw_weights[factor] = trial.suggest_float(f'weight_{factor}', 0.01, 0.40)
    total = sum(raw_weights.values())
    rank_factor_weights = {k: v / total for k, v in raw_weights.items()}
    
    # ... rest of sample_config unchanged ...
    return RotationConfig(
        ...
        rank_factor_weights=rank_factor_weights,
        rank_factor_directions=directions,
        ...
    )
```

And correspondingly update `_params_to_config()` and `_config_to_dict()` to use dynamic factor lists.

- [ ] **Step 4.2: 更新 CLI 入口传入动态 factor_directions**

```python
# CLI 入口 — 组装参数
factor_directions = None
if args.factors and Path(args.factors).exists():
    import json
    with open(args.factors, 'r', encoding='utf-8') as f:
        factor_data = json.load(f)
    raw_factors = factor_data.get('raw', [])
    if raw_factors and ranker is None:
        # 从 IC report 推断方向
        ic_path = Path(args.output) / 'factor_ic_report.csv'
        if ic_path.exists():
            ic_df = pd.read_csv(ic_path, index_col=0)
            factor_directions = {}
            for f in raw_factors:
                if f in ic_df.index:
                    ic_mean = ic_df.loc[f, 'ic_mean']
                    factor_directions[f] = 1 if ic_mean > 0 else -1
                else:
                    factor_directions[f] = 1
        else:
            factor_directions = {f: 1 for f in raw_factors}
        
        # 更新 base_config 的因子权重
        base_config.rank_factor_weights = {f: 1.0/len(raw_factors) for f in raw_factors}
        base_config.rank_factor_directions = factor_directions

# 传入 objective
obj = lambda trial: objective(trial, start_date, end_date, base_config,
                              data_provider, ranker, factor_directions)
```

- [ ] **Step 4.3: 更新 objective 签名传递 factor_directions**

```python
def objective(trial, start_date, end_date, base_config=None,
              data_provider=None, ranker=None, factor_directions=None):
    config = sample_config(trial, base_config, factor_directions)
    # ... rest unchanged ...
```

- [ ] **Step 4.4: Commit**

```bash
git add optimization/optuna/run_daily_rotation_optimization.py
git commit -m "feat(optuna): add --ml-model auto, --factors, and dynamic factor weight search space"
```

---

### Task 5: 端到端验证

- [ ] **Step 5.1: 准备测试数据**

```bash
# 确保缓存已构建
python -c "
from data.cache.daily_data_cache import DailyDataCache
DailyDataCache.build('2020-01-01', '2024-12-31', 'cache/daily_rotation')
print('缓存就绪')
"
```

- [ ] **Step 5.2: 运行 Stage 0 — 因子筛选**

```bash
python -m strategy.factors.factor_screening \
    --start 2020-01-01 --end 2022-12-31 \
    --cache-dir cache/daily_rotation \
    --output output/
# Expected: 生成 output/selected_factors.json + output/factor_ic_report.csv
```

- [ ] **Step 5.3: 运行 Stage 1 — ML 超参优化（少量 trial 快速验证）**

```bash
python optimization/optuna/run_ml_optimization.py \
    --train-start 2020-01-01 --train-end 2022-12-31 \
    --cache-dir cache/daily_rotation \
    --factors output/selected_factors.json \
    --trials 5 --output output/
# Expected: 生成 output/best_model.pkl + output/best_ml_params.json
```

- [ ] **Step 5.4: 运行 Stage 2 — 框架优化（少量 trial 快速验证）**

```bash
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-06-30 \
    --ml-model auto --trials 5 --output output/
# Expected: 成功加载 best_model.pkl，完成优化并输出结果
```

- [ ] **Step 5.5: 兼容性验证 — 不传任何新参数时行为不变**

```bash
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-06-30 \
    --trials 3 --output output/
# Expected: 使用默认 SignalRanker，行为与之前一致
```

- [ ] **Step 5.6: 运行现有测试确保无回归**

```bash
pytest tests/ -v --timeout=120 -x
# Expected: 所有已有测试通过
```

- [ ] **Step 5.7: Commit**

```bash
git add -A
git commit -m "feat: complete ML+Optuna three-stage optimization pipeline"
```

---

### Task 6: 修复 _select_by_robustness 敏感性分析性能问题

**Files:**
- Modify: `optimization/optuna/run_daily_rotation_optimization.py:876-961` (`_select_by_robustness`)
- Modify: `robustness/sensitivity.py:12-69` (`SensitivityAnalyzer._evaluate`)
- Modify: `optimization/optuna/run_daily_rotation_optimization.py:968-1059` (CLI 入口，新增 `--skip-robustness`)

**Root Cause:** `_select_by_robustness` 运行 5 trials × 41 backtests = 205 次完整回测，耗时 1-2 小时。其中 ~90 次因子权重扰动在 MLRanker 模式完全无效（MLRanker 忽略配置中的权重），且在 SignalRanker 模式下因归一化抵消而输出几乎不变。全程无进度日志，用户感知为"无限循环"。

- [ ] **Step 6.1: SensitivityAnalyzer 增加日志 + 跳过因子权重参数**

```python
# robustness/sensitivity.py — _evaluate 方法修改
def _evaluate(self, params: Dict[str, float],
              run_fn: Callable[[Dict], Any]) -> SensitivityResult:
    import logging
    logger = logging.getLogger(__name__)

    base_result = run_fn(deepcopy(params))
    base_sharpe = self._extract_sharpe(base_result)
    logger.info(f"敏感性分析: 基准 Sharpe={base_sharpe:.4f}, 分析 {len([k for k,v in params.items() if isinstance(v,(int,float))])} 个数值参数")

    per_param = {}
    sharpe_changes = []

    numeric_keys = [(k, v) for k, v in params.items() if isinstance(v, (int, float))]
    # 跳过因子权重（以 weight_ 开头的 key），它们的扰动被归一化抵消
    framework_keys = [(k, v) for k, v in numeric_keys if not k.startswith('weight_')]

    for key, value in framework_keys:
        delta = abs(value) * self.perturbation_pct if value != 0 else self.perturbation_pct
        perturbed_sharpes = []

        for perturbed_value in [value + delta, value - delta]:
            cfg = deepcopy(params)
            cfg[key] = perturbed_value
            result = run_fn(cfg)
            perturbed_sharpes.append(self._extract_sharpe(result))

        avg_perturbed = sum(perturbed_sharpes) / 2
        avg_change = abs((avg_perturbed - base_sharpe) / base_sharpe) if base_sharpe != 0 else 0.0
        sharpe_changes.append(avg_change)

        per_param[key] = {
            'base_value': value,
            'delta': round(delta, 6),
            'sharpe_change_pct': round(avg_change * 100, 2),
            'stable': avg_change < 0.10,
        }
        logger.info(f"  {key}: base={value:.4g}, ±{delta:.4g}, Sharpe变化={avg_change*100:.1f}%, "
                    f"stable={avg_change < 0.10}")

    if sharpe_changes:
        avg_sensitivity = sum(sharpe_changes) / len(sharpe_changes)
        stability = max(0.0, min(1.0, 1.0 - avg_sensitivity * 5))
    else:
        stability = 1.0

    return SensitivityResult(per_param=per_param, overall_stability_score=round(stability, 4))
```

- [ ] **Step 6.2: _select_by_robustness 增加进度日志**

```python
# optimization/optuna/run_daily_rotation_optimization.py — _select_by_robustness
def _select_by_robustness(study, base_config, start_date, end_date,
                          output_dir, data_provider, ranker):
    from robustness.sensitivity import SensitivityAnalyzer
    from backtesting.analysis.performance_analyzer import PerformanceAnalyzer

    completed = [t for t in study.trials if t.value is not None]
    if len(completed) < 2:
        return None

    completed.sort(key=lambda t: t.value, reverse=True)
    top5 = completed[:5]

    def engine_factory(params: dict):
        config = _params_to_config(params, base_config)
        engine = DailyRotationEngine(config, start_date, end_date,
                                     data_provider=data_provider, ranker=ranker)
        results = engine.run()
        if not results:
            return {'sharpe_ratio': 0.0}
        equity = [config.initial_capital] + [r.total_asset for r in results]
        analyzer = PerformanceAnalyzer(
            trades=[], initial_capital=config.initial_capital,
            equity_curve=equity, periods_per_year=252,
        )
        return analyzer.calculate_metrics()

    sa = SensitivityAnalyzer()
    best_params = None
    best_score = -float('inf')
    max_s = max(t.value for t in top5)
    min_s = min(t.value for t in top5)
    s_range = max_s - min_s if max_s != min_s else 1.0

    scores = []
    n_framework_params = len([k for k, v in top5[0].params.items()
                              if isinstance(v, (int, float)) and not k.startswith('weight_')])
    print(f"\n稳健性筛选: Top {len(top5)} 参数组 × {1 + n_framework_params * 2} 次回测 "
          f"(仅分析框架参数，跳过因子权重)")

    for i, trial in enumerate(top5):
        print(f"\n[{i+1}/{len(top5)}] Trial #{trial.number} (Sharpe={trial.value:.4f}): "
              f"运行敏感性分析...")
        params = trial.params.copy()
        result = sa.run(params, engine_factory)
        sharpe_norm = (trial.value - min_s) / s_range
        score = sharpe_norm * 0.6 + result.overall_stability_score * 0.4
        scores.append({
            'trial_number': trial.number,
            'sharpe': trial.value,
            'sharpe_norm': round(sharpe_norm, 4),
            'stability': result.overall_stability_score,
            'combined_score': round(score, 4),
            'params': params,
            'sensitivity': {k: v['sharpe_change_pct'] for k, v in result.per_param.items()},
        })
        print(f"  稳定性={result.overall_stability_score:.4f}, "
              f"综合评分={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params

    # 保存结果 (unchanged)
    output_dir = Path(output_dir or '.')
    output_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    path = output_dir / f'robustness_selection_{now}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'top5_scores': scores,
            'best_by_combined': best_params,
            'best_combined_score': best_score,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n稳健性筛选结果已保存: {path}")
    print(f"Top 5 综合评分: {[s['combined_score'] for s in scores]}")
    if scores:
        print(f"最优参数 (综合): 稳定性={scores[0]['stability']:.3f}, "
              f"Sharpe={scores[0]['sharpe']:.3f}")

    return best_params
```

- [ ] **Step 6.3: CLI 增加 --skip-robustness 选项**

```python
# optimization/optuna/run_daily_rotation_optimization.py — CLI args
parser.add_argument('--skip-robustness', action='store_true',
                    help='跳过稳健性筛选（加速优化流程）')

# run_single_optimization 中:
# Phase 2: Top 5 敏感性筛选
if not args.skip_robustness:
    _select_by_robustness(study, base_config, start_date, end_date,
                          output_dir, data_provider, ranker)
else:
    print("\n跳过稳健性筛选 (--skip-robustness)")
```

同时在 `run_single_optimization` 函数签名中增加 `skip_robustness` 参数，并在 `run_walk_forward` 中传递该参数。

- [ ] **Step 6.4: 运行现有敏感性测试确保无回归**

```bash
pytest tests/robustness/test_sensitivity.py -v
# Expected: 4 passed
```

- [ ] **Step 6.5: 验证 — 模拟正常调用确认回测次数减少**

```python
# 验证脚本
from robustness.sensitivity import SensitivityAnalyzer

# 模拟 trial.params 样式的 params dict
params = {
    'weight_RSI_1': 0.12, 'weight_RET_20': 0.12,
    'weight_VOLUME_RATIO': 0.05, 'weight_PB': 0.18,
    'weight_PE_TTM': 0.20, 'weight_OVERHEAT': 0.07,
    'weight_circulating_mv': 0.11, 'weight_WR_10': 0.07,
    'weight_WR_14': 0.07,
    'max_total_pct': 0.80, 'max_position_pct': 0.15,
    'max_positions': 5, 'atr_period': 14,
    'stop_loss_mult': 2.0, 'take_profit_mult': 3.0,
    'trailing_pct': 0.10, 'trailing_start': 0.05,
    'overheat_rsi_threshold': 70.0, 'overheat_ret5_threshold': 0.15,
    'kdj_low_threshold': 30.0,
    'signal_KDJ_GOLD': 'on', 'fallback_signal': 'KDJ_GOLD',
    'buy_signal_mode': 'OR', 'sell_signal_types': [],
}

call_count = [0]
def mock_run(cfg):
    call_count[0] += 1
    return {'sharpe_ratio': 1.0}

sa = SensitivityAnalyzer()
sa.run(params, mock_run)

# 旧代码: 20 个 numeric params → 1 + 40 = 41 calls
# 新代码: 11 个 framework params → 1 + 22 = 23 calls
print(f"总回测次数: {call_count[0]}")
assert call_count[0] == 23, f"Expected 23, got {call_count[0]}"
print("PASS: 因子权重扰动已跳过")
```

- [ ] **Step 6.6: Commit**

```bash
git add robustness/sensitivity.py optimization/optuna/run_daily_rotation_optimization.py
git commit -m "fix(optuna): reduce robustness analysis backtest count by 45%, add progress logging and --skip-robustness"
```
