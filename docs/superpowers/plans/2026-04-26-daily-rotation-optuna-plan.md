# Daily Rotation Optuna 参数优化 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 Optuna 贝叶斯优化入口，对 daily_rotation 的混合类型参数空间（float/int/categorical）进行单期和 Walk-Forward 优化。

**Architecture:** 新建独立文件 `run_daily_rotation_optimization.py`，包含 `sample_config`（Trial → RotationConfig）、`objective`（回测 → Sharpe）、窗口生成和两种运行模式。不改动现有 GA 框架。

**Tech Stack:** Python, Optuna, DailyRotationEngine, RotationConfig, PerformanceAnalyzer

---

### Task 1: 绩效计算工具函数

**Files:**
- Create: `back_testing/optimization/run_daily_rotation_optimization.py`
- Test: `tests/back_testing/optimization/test_daily_rotation_optuna.py`

- [ ] **Step 1: 写失败的测试**

```python
"""daily_rotation Optuna 优化 单元测试"""
import numpy as np
from back_testing.optimization.run_daily_rotation_optimization import (
    compute_sharpe,
    max_drawdown,
)


def test_max_drawdown_simple():
    equity = [1.0, 1.1, 0.9, 1.0, 0.8, 1.05]
    dd = max_drawdown(equity)
    # peak=1.1, trough=0.8 → dd=0.3/1.1≈0.2727
    assert abs(dd - 0.2727) < 0.001


def test_max_drawdown_no_drawdown():
    equity = [1.0, 1.1, 1.2, 1.3]
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_empty():
    assert max_drawdown([1.0]) == 0.0


def test_compute_sharpe_positive():
    # 构造收益率序列：每天固定 +0.001，年化 Sharpe ≈ 252 * 0.001 / 0 = 无穷
    # 用两个不同值：0.001 和 0.002，std≈0.000707
    equity = [1.0 + i * 0.001 if i % 2 == 0 else 1.0 + i * 0.001 + 0.0005
              for i in range(253)]
    sharpe = compute_sharpe(equity)
    assert sharpe > 1.0


def test_compute_sharpe_constant():
    equity = [1.0] * 253
    assert compute_sharpe(equity) == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/back_testing/optimization/test_daily_rotation_optuna.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: 实现工具函数**

```python
"""Daily Rotation 参数优化 — Optuna 贝叶斯优化

用法:
    # 单期优化
    python back_testing/optimization/run_daily_rotation_optimization.py \
        --mode single --start 2024-01-01 --end 2024-12-31 --trials 100

    # Walk-Forward 优化
    python back_testing/optimization/run_daily_rotation_optimization.py \
        --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,  # Optuna 日志太多，默认只显示警告
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 绩效计算
# ═══════════════════════════════════════════════

def max_drawdown(equity: List[float]) -> float:
    """从净值序列计算最大回撤"""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe(equity: List[float], periods_per_year: int = 252) -> float:
    """从净值序列计算年化 Sharpe Ratio（无风险利率=0）"""
    if len(equity) < 2:
        return 0.0
    values = np.array(equity)
    returns = np.diff(values) / values[:-1]
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/back_testing/optimization/test_daily_rotation_optuna.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tests/back_testing/optimization/test_daily_rotation_optuna.py back_testing/optimization/run_daily_rotation_optimization.py
git commit -m "feat: add sharpe/max_drawdown utilities for Optuna optimization"
```

---

### Task 2: sample_config — Trial → RotationConfig 参数采样

**Files:**
- Modify: `back_testing/optimization/run_daily_rotation_optimization.py`
- Modify: `tests/back_testing/optimization/test_daily_rotation_optuna.py`

- [ ] **Step 1: 写测试**

在 `tests/back_testing/optimization/test_daily_rotation_optuna.py` 追加：

```python
import optuna
from back_testing.optimization.run_daily_rotation_optimization import sample_config
from back_testing.rotation.config import RotationConfig


def test_sample_config_basic():
    study = optuna.create_study()
    trial = study.ask()

    config = sample_config(trial)

    # 信号至少有一个开启
    assert any(s in config.buy_signal_types for s in [
        'KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD', 'BOLL_BREAK', 'HIGH_BREAK'
    ]), "at least one buy signal must be on"

    # 因子权重和为 1（允许浮点误差）
    w = sum(config.rank_factor_weights.values())
    assert abs(w - 1.0) < 0.001, f"weights sum to {w}"

    # 连续参数在范围内
    assert 0.30 <= config.max_total_pct <= 1.00
    assert 0.05 <= config.max_position_pct <= 0.30
    assert 3 <= config.max_positions <= 10
    assert 7 <= config.atr_period <= 21
    assert 1.0 <= config.stop_loss_mult <= 3.5
    assert 2.0 <= config.take_profit_mult <= 5.0
    assert 0.05 <= config.trailing_pct <= 0.20
    assert 0.02 <= config.trailing_start <= 0.10
    assert 60.0 <= config.overheat_rsi_threshold <= 90.0
    assert 0.05 <= config.overheat_ret5_threshold <= 0.30

    assert config.buy_signal_mode in ('OR', 'AND')

    # 固定参数不变
    assert config.initial_capital == 1_000_000.0
    assert config.benchmark_index == 'sh000300'
    assert config.exclude_st is True


def test_sample_config_overrides_base():
    base = RotationConfig(
        max_total_pct=0.50,
        initial_capital=500_000,
    )
    study = optuna.create_study()
    trial = study.ask()
    config = sample_config(trial, base_config=base)
    assert config.initial_capital == 500_000
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/back_testing/optimization/test_daily_rotation_optuna.py::test_sample_config_basic -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: 实现 sample_config**

在 `run_daily_rotation_optimization.py` 的 import 区追加：

```python
import optuna
from back_testing.rotation.config import RotationConfig
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
```

在文件末尾追加：

```python
# ═══════════════════════════════════════════════
# 参数采样
# ═══════════════════════════════════════════════

# 可优化信号的完整列表
ALL_SIGNAL_TYPES = ['KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD', 'BOLL_BREAK', 'HIGH_BREAK']

# 固定参数（不参与优化）
FIXED_SELL_SIGNALS = ['KDJ_DEATH', 'MACD_DEATH', 'MA_DEATH', 'VOL_DEATH',
                      'BOLL_BREAK_DOWN', 'HIGH_BREAK_DOWN']
FIXED_FACTOR_DIRECTIONS = {
    'RSI_1': 1, 'RET_20': 1, 'VOLUME_RATIO': 1,
    'PB': -1, 'PE_TTM': -1, 'OVERHEAT': -1,
}


def sample_config(trial: optuna.Trial, base_config: RotationConfig = None) -> RotationConfig:
    """从 Optuna Trial 采样一个 RotationConfig

    Args:
        trial: Optuna trial 对象
        base_config: 基础配置（固定参数从此继承），None 则用默认值

    Returns:
        采样后的 RotationConfig
    """
    base = base_config or RotationConfig()

    # --- 因子权重：独立采样后归一化 ---
    raw_weights = {}
    for factor in ['RSI_1', 'RET_20', 'VOLUME_RATIO', 'PB', 'PE_TTM', 'OVERHEAT']:
        raw_weights[factor] = trial.suggest_float(f'weight_{factor}', 0.01, 0.40)
    total = sum(raw_weights.values())
    rank_factor_weights = {k: v / total for k, v in raw_weights.items()}

    # --- 买入信号开关（至少保留一个） ---
    active_signals: List[str] = []
    for sig in ALL_SIGNAL_TYPES:
        on = trial.suggest_categorical(f'signal_{sig}', ['on', 'off'])
        if on == 'on':
            active_signals.append(sig)
    if not active_signals:
        # 极少数情况，随机选一个开启
        fallback = trial.suggest_categorical('_signal_fallback', ALL_SIGNAL_TYPES)
        active_signals.append(fallback)

    # --- 信号逻辑模式 ---
    buy_signal_mode = trial.suggest_categorical('buy_signal_mode', ['OR', 'AND'])

    # --- 连续参数 ---
    max_total_pct = trial.suggest_float('max_total_pct', 0.30, 1.00)
    max_position_pct = trial.suggest_float('max_position_pct', 0.05, 0.30)
    overheat_rsi_threshold = trial.suggest_float('overheat_rsi_threshold', 60.0, 90.0)
    overheat_ret5_threshold = trial.suggest_float('overheat_ret5_threshold', 0.05, 0.30)
    stop_loss_mult = trial.suggest_float('stop_loss_mult', 1.0, 3.5)
    take_profit_mult = trial.suggest_float('take_profit_mult', 2.0, 5.0)
    trailing_pct = trial.suggest_float('trailing_pct', 0.05, 0.20)
    trailing_start = trial.suggest_float('trailing_start', 0.02, 0.10)

    # --- 整数参数 ---
    max_positions = trial.suggest_int('max_positions', 3, 10)
    atr_period = trial.suggest_int('atr_period', 7, 21)

    return RotationConfig(
        initial_capital=base.initial_capital,
        max_total_pct=max_total_pct,
        max_position_pct=max_position_pct,
        max_positions=max_positions,
        buy_signal_types=active_signals,
        buy_signal_mode=buy_signal_mode,
        sell_signal_types=base.sell_signal_types,
        rank_factor_weights=rank_factor_weights,
        rank_factor_directions=FIXED_FACTOR_DIRECTIONS,
        market_regime=base.market_regime,
        exclude_st=base.exclude_st,
        exclude_limit_up=base.exclude_limit_up,
        exclude_limit_down=base.exclude_limit_down,
        exclude_suspended=base.exclude_suspended,
        benchmark_index=base.benchmark_index,
        atr_period=atr_period,
        stop_loss_mult=stop_loss_mult,
        take_profit_mult=take_profit_mult,
        trailing_pct=trailing_pct,
        trailing_start=trailing_start,
        overheat_rsi_threshold=overheat_rsi_threshold,
        overheat_ret5_threshold=overheat_ret5_threshold,
    )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/back_testing/optimization/test_daily_rotation_optuna.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add tests/back_testing/optimization/test_daily_rotation_optuna.py back_testing/optimization/run_daily_rotation_optimization.py
git commit -m "feat: add sample_config for Optuna trial to RotationConfig mapping"
```

---

### Task 3: objective 函数 — 回测 → Sharpe

**Files:**
- Modify: `back_testing/optimization/run_daily_rotation_optimization.py`

- [ ] **Step 1: 实现 objective 函数**

在 `run_daily_rotation_optimization.py` 末尾追加：

```python
# ═══════════════════════════════════════════════
# 目标函数
# ═══════════════════════════════════════════════

MAX_DRAWDOWN_LIMIT = 0.30


def objective(trial: optuna.Trial,
              start_date: str,
              end_date: str,
              base_config: RotationConfig = None) -> float:
    """Optuna 目标函数：给定 Trial，运行回测，返回年化 Sharpe

    Args:
        trial: Optuna trial
        start_date: 回测开始日期 'YYYY-MM-DD'
        end_date: 回测结束日期 'YYYY-MM-DD'
        base_config: 基础配置

    Returns:
        年化 Sharpe Ratio（max_drawdown > 30% 时返回 0）
    """
    config = sample_config(trial, base_config)

    try:
        engine = DailyRotationEngine(config, start_date, end_date)
        results = engine.run()

        if not results or len(results) < 2:
            return 0.0

        equity = [config.initial_capital] + [r.total_asset for r in results]

        dd = max_drawdown(equity)
        if dd > MAX_DRAWDOWN_LIMIT:
            return 0.0

        sharpe = compute_sharpe(equity, periods_per_year=252)
        return sharpe if not math.isnan(sharpe) else 0.0

    except Exception:
        logger.debug(f"Trial {trial.number} failed", exc_info=True)
        return 0.0
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/optimization/run_daily_rotation_optimization.py
git commit -m "feat: add objective function for Optuna daily rotation optimization"
```

---

### Task 4: 窗口生成 + 单期优化 + Walk-Forward

**Files:**
- Modify: `back_testing/optimization/run_daily_rotation_optimization.py`

- [ ] **Step 1: 实现窗口生成函数**

在 `run_daily_rotation_optimization.py` 末尾追加：

```python
# ═══════════════════════════════════════════════
# Walk-Forward 窗口生成
# ═══════════════════════════════════════════════

def generate_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_months: int = 12,
    test_months: int = 6,
    step_months: int = 3,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """生成 Walk-Forward 滚动窗口

    Returns:
        List of (train_start, train_end, test_start, test_end)
    """
    windows = []
    train_start = start_date
    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)

        if test_end > end_date:
            break

        windows.append((train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(months=step_months)

    return windows
```

- [ ] **Step 2: 实现 run_single_optimization**

在文件末尾追加：

```python
# ═══════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════

def run_single_optimization(
    start_date: str,
    end_date: str,
    n_trials: int = 100,
    base_config: RotationConfig = None,
    study_name: str = None,
    output_dir: str = None,
) -> Tuple[RotationConfig, float, optuna.Study]:
    """单期优化：在固定日期范围内搜索最优参数

    Returns:
        (best_config, best_sharpe, study)
    """
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
    )

    obj = lambda trial: objective(trial, start_date, end_date, base_config)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    best_config = sample_config(study.best_trial, base_config)
    # 用 best_trial 的参数再采样一次确保一致性；直接用 study.best_params
    best_sharpe = study.best_value

    print(f"\n{'=' * 60}")
    print(f"单期优化完成")
    print(f"区间: {start_date} ~ {end_date}")
    print(f"最优 Sharpe: {best_sharpe:.4f}")
    print(f"{'=' * 60}")

    # 保存结果
    _save_results(study, best_config, best_sharpe, start_date, end_date, output_dir)

    return best_config, best_sharpe, study


def run_walk_forward(
    start_date: str,
    end_date: str,
    n_trials: int = 100,
    base_config: RotationConfig = None,
    train_months: int = 12,
    test_months: int = 6,
    step_months: int = 3,
    output_dir: str = None,
) -> List[Dict]:
    """Walk-Forward 滚动优化

    Returns:
        List of dicts，每个窗口一条记录
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    windows = generate_windows(start_ts, end_ts, train_months, test_months, step_months)

    if not windows:
        print("错误: 日期范围内无法生成 Walk-Forward 窗口")
        return []

    print(f"\nWalk-Forward 优化: {len(windows)} 个窗口")
    print(f"训练={train_months}月 测试={test_months}月 步进={step_months}月")

    records = []
    for i, (train_s, train_e, test_s, test_e) in enumerate(windows):
        print(f"\n{'─' * 50}")
        print(f"窗口 {i + 1}/{len(windows)}")
        print(f"训练: {train_s.strftime('%Y-%m-%d')} ~ {train_e.strftime('%Y-%m-%d')}")
        print(f"测试: {test_s.strftime('%Y-%m-%d')} ~ {test_e.strftime('%Y-%m-%d')}")

        # 训练期优化
        train_start_str = train_s.strftime('%Y-%m-%d')
        train_end_str = train_e.strftime('%Y-%m-%d')
        best_config, train_sharpe, study = run_single_optimization(
            train_start_str, train_end_str, n_trials, base_config,
            study_name=f"wf_window_{i}",
            output_dir=output_dir,
        )

        # 测试期评估
        test_start_str = test_s.strftime('%Y-%m-%d')
        test_end_str = test_e.strftime('%Y-%m-%d')
        test_sharpe = _evaluate_on_test(best_config, test_start_str, test_end_str)

        record = {
            'window': i,
            'train_start': train_start_str,
            'train_end': train_end_str,
            'test_start': test_start_str,
            'test_end': test_end_str,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'best_params': _config_to_dict(best_config),
        }
        records.append(record)

    # 汇总
    _print_wf_summary(records)
    _save_wf_results(records, output_dir)

    return records


def _evaluate_on_test(config: RotationConfig, start: str, end: str) -> float:
    """在测试集上评估给定配置，返回 Sharpe"""
    try:
        engine = DailyRotationEngine(config, start, end)
        results = engine.run()
        if not results or len(results) < 2:
            return 0.0
        equity = [config.initial_capital] + [r.total_asset for r in results]
        return compute_sharpe(equity)
    except Exception:
        return 0.0


def _config_to_dict(config: RotationConfig) -> Dict:
    """将 RotationConfig 序列化为可 JSON 化的 dict"""
    return {
        'max_total_pct': config.max_total_pct,
        'max_position_pct': config.max_position_pct,
        'max_positions': config.max_positions,
        'buy_signal_types': config.buy_signal_types,
        'buy_signal_mode': config.buy_signal_mode,
        'rank_factor_weights': config.rank_factor_weights,
        'atr_period': config.atr_period,
        'stop_loss_mult': config.stop_loss_mult,
        'take_profit_mult': config.take_profit_mult,
        'trailing_pct': config.trailing_pct,
        'trailing_start': config.trailing_start,
        'overheat_rsi_threshold': config.overheat_rsi_threshold,
        'overheat_ret5_threshold': config.overheat_ret5_threshold,
    }


def _save_results(study: optuna.Study, best_config: RotationConfig,
                  best_sharpe: float, start: str, end: str,
                  output_dir: str = None):
    """保存优化结果到 JSON"""
    output_dir = Path(output_dir or '.')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 最优参数
    best_path = output_dir / 'best_params.json'
    best_data = {
        'start_date': start,
        'end_date': end,
        'best_sharpe': best_sharpe,
        'best_params': _config_to_dict(best_config),
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best_data, f, indent=2, ensure_ascii=False)
    print(f"最优参数已保存: {best_path}")

    # Trial 记录
    trials_path = output_dir / 'optuna_trials.csv'
    df = study.trials_dataframe()
    df.to_csv(trials_path, index=False, encoding='utf-8-sig')
    print(f"Trial 记录已保存: {trials_path} ({len(df)} 条)")


def _print_wf_summary(records: List[Dict]):
    """打印 Walk-Forward 汇总"""
    print(f"\n{'=' * 60}")
    print("Walk-Forward 汇总")
    print(f"{'=' * 60}")
    test_sharpes = [r['test_sharpe'] for r in records if r['test_sharpe'] > 0]
    if test_sharpes:
        print(f"窗口数: {len(records)}")
        print(f"测试 Sharpe 均值: {np.mean(test_sharpes):.4f}")
        print(f"测试 Sharpe 中位数: {np.median(test_sharpes):.4f}")
        print(f"测试 Sharpe 标准差: {np.std(test_sharpes):.4f}")
        print(f"正 Sharpe 窗口: {sum(1 for s in test_sharpes if s > 0)}/{len(test_sharpes)}")


def _save_wf_results(records: List[Dict], output_dir: str = None):
    """保存 Walk-Forward 结果"""
    output_dir = Path(output_dir or '.')
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'walkforward_results.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Walk-Forward 结果已保存: {path}")
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/optimization/run_daily_rotation_optimization.py
git commit -m "feat: add walk-forward windows, single/wf optimization runners"
```

---

### Task 5: CLI 入口

**Files:**
- Modify: `back_testing/optimization/run_daily_rotation_optimization.py`

- [ ] **Step 1: 添加 CLI main 函数**

在 `run_daily_rotation_optimization.py` 末尾追加：

```python
# ═══════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Daily Rotation 参数优化（Optuna）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单期优化
  python back_testing/optimization/run_daily_rotation_optimization.py \\
      --mode single --start 2024-01-01 --end 2024-12-31 --trials 100

  # Walk-Forward 优化
  python back_testing/optimization/run_daily_rotation_optimization.py \\
      --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50
        """
    )
    parser.add_argument('--mode', choices=['single', 'walkforward'], default='single',
                        help='优化模式: single=单期, walkforward=滚动窗口')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--trials', type=int, default=100, help='每次优化的 Trial 数')
    parser.add_argument('--output', default='.', help='结果输出目录')
    parser.add_argument('--train-months', type=int, default=12, help='WF 训练期（月）')
    parser.add_argument('--test-months', type=int, default=6, help='WF 测试期（月）')
    parser.add_argument('--step-months', type=int, default=3, help='WF 步进（月）')
    parser.add_argument('--verbose', action='store_true', help='详细日志')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('back_testing.rotation').setLevel(logging.DEBUG)

    base_config = RotationConfig()

    if args.mode == 'single':
        run_single_optimization(
            start_date=args.start,
            end_date=args.end,
            n_trials=args.trials,
            base_config=base_config,
            output_dir=args.output,
        )
    else:
        run_walk_forward(
            start_date=args.start,
            end_date=args.end,
            n_trials=args.trials,
            base_config=base_config,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            output_dir=args.output,
        )
```

- [ ] **Step 2: Commit**

```bash
git add back_testing/optimization/run_daily_rotation_optimization.py
git commit -m "feat: add CLI entry point for Optuna optimization"
```

---

### Task 6: 集成冒烟测试

**Files:**
- Test: `tests/back_testing/optimization/test_daily_rotation_optuna.py` (追加)

- [ ] **Step 1: 写冒烟测试**

在 `tests/back_testing/optimization/test_daily_rotation_optuna.py` 末尾追加：

```python
import tempfile
import os


def test_run_single_optimization_smoke():
    """冒烟测试：3 trials 跑通无报错"""
    from back_testing.optimization.run_daily_rotation_optimization import (
        run_single_optimization,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        config, sharpe, study = run_single_optimization(
            start_date='2024-01-01',
            end_date='2024-02-28',
            n_trials=3,
            output_dir=tmpdir,
        )

        assert isinstance(sharpe, float)
        assert len(study.trials) == 3

        # 验证输出文件
        assert os.path.exists(os.path.join(tmpdir, 'best_params.json'))
        assert os.path.exists(os.path.join(tmpdir, 'optuna_trials.csv'))


def test_generate_windows():
    from back_testing.optimization.run_daily_rotation_optimization import (
        generate_windows,
    )
    import pandas as pd

    windows = generate_windows(
        pd.Timestamp('2022-01-01'),
        pd.Timestamp('2024-12-31'),
        train_months=12,
        test_months=6,
        step_months=3,
    )
    assert len(windows) > 0
    for train_s, train_e, test_s, test_e in windows:
        assert train_s < train_e < test_s < test_e
        # 训练期约 12 个月
        delta = (train_e - train_s).days
        assert 300 < delta < 400, f"train window {delta} days"
```

- [ ] **Step 2: 运行冒烟测试**

```bash
pytest tests/back_testing/optimization/test_daily_rotation_optuna.py -v
```
Expected: 9 passed (5 单元 + 2 sample_config + 2 冒烟)

- [ ] **Step 3: 运行 CLI 命令确认无 ImportError**

```bash
python back_testing/optimization/run_daily_rotation_optimization.py --help
```
Expected: 显示 usage 和参数列表

- [ ] **Step 4: Commit**

```bash
git add tests/back_testing/optimization/test_daily_rotation_optuna.py
git commit -m "test: add integration smoke test for Optuna daily rotation optimization"
```

---

## 验证清单

1. `python back_testing/optimization/run_daily_rotation_optimization.py --mode single --start 2024-01-01 --end 2024-06-30 --trials 5` 跑通无报错
2. 检查 `best_params.json` 中各参数在 spec 定义的范围内
3. 检查 `optuna_trials.csv` 包含 5 行 trial 记录
4. 运行 `pytest tests/back_testing/optimization/test_daily_rotation_optuna.py -v` 全部通过
