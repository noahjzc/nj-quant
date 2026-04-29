"""Daily Rotation 参数优化 — Optuna 贝叶斯优化

用法:
    # 单期优化
    python back_testing/optimization/run_daily_rotation_optimization.py \
        --mode single --start 2024-01-01 --end 2024-12-31 --trials 100

    # Walk-Forward 优化
    python back_testing/optimization/run_daily_rotation_optimization.py \
        --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50

架构概览:
    本模块使用 Optuna TPE (Tree-structured Parzen Estimator) 对 DailyRotationEngine
    的 14 个超参数进行贝叶斯优化，目标是最大化年化 Sharpe Ratio。

    优化模式:
    - single: 在一个固定日期范围内搜索最优参数，适合快速验证
    - walkforward: 滚动窗口优化，每个窗口的训练期 >= 测试期起始日，
      模拟实盘中"用历史最优参数跑未来"的真实场景

    数据流:
    1. DailyDataCache.build() — 从 PostgreSQL 逐日读取，写入 Parquet（一次性构建）
    2. CachedProvider — 从 Parquet 读取，替代 DataProvider（跨 Trial 复用）
    3. preload.parquet — 引擎所需的前 30 天历史打包为单文件，避免每个 Trial 读 30+ 文件
"""
import argparse
import datetime
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import optuna
from back_testing.rotation.config import RotationConfig
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.data.daily_data_cache import DailyDataCache, CachedProvider

# 默认日志级别设为 WARNING，避免 Optuna 和引擎的 INFO 日志刷屏
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 禁用 Optuna 的默认日志（Trial 级别信息由进度条替代）
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════
# 绩效计算
# ═══════════════════════════════════════════════

def max_drawdown(equity: List[float]) -> float:
    """从净值序列计算最大回撤

    算法: 遍历净值序列，维护历史最高点 peak，每个时点计算
    (peak - 当前值) / peak 作为当前回撤，取全程最大值。

    例如: [1.0, 1.1, 0.9, 1.0] → peak 升至 1.1 → 跌至 0.9 时回撤 = (1.1-0.9)/1.1 ≈ 18.2%

    Args:
        equity: 净值序列（含初始资金），如 [1_000_000, 1_010_000, 995_000, ...]

    Returns:
        最大回撤比率（0.0 ~ 1.0），序列不足 2 个点时返回 0.0
    """
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
    """从净值序列计算年化 Sharpe Ratio（无风险利率=0）

    计算步骤:
    1. 日收益率 = (当日净值 - 前日净值) / 前日净值
    2. 剔除 NaN（避免停牌等导致的异常值）
    3. 年化 Sharpe = (日均收益率 / 日收益率标准差) × sqrt(年交易日数)

    选择从净值而非收益率计算的原因: 回测引擎输出的是每日总资产，
    比直接从交易收益计算更准确，因为包含了持仓浮动盈亏。

    Args:
        equity: 净值序列（含初始资金）
        periods_per_year: 年交易日数，A 股默认 252

    Returns:
        年化 Sharpe Ratio，无法计算时返回 0.0
    """
    if len(equity) < 2:
        return 0.0
    values = np.array(equity)
    # 日收益率 = (V_t - V_{t-1}) / V_{t-1}
    returns = np.diff(values) / values[:-1]
    # 过滤 NaN（极少数情况下停牌股平仓可能产生 0/0）
    returns = returns[~np.isnan(returns)]
    # 需要至少 2 个有效收益率，且标准差不能为 0（全部同收益说明数据异常）
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


# ═══════════════════════════════════════════════
# 参数采样
# ═══════════════════════════════════════════════

# 可优化信号的完整列表（固定的搜索空间，不在运行时修改）
# 每个信号对应 daily_rotation_engine 中 _scan_buy_candidates 的一个信号 mask
ALL_SIGNAL_TYPES = ['KDJ_GOLD', 'MACD_GOLD', 'MA_GOLD', 'VOL_GOLD', 'BOLL_BREAK', 'HIGH_BREAK', 'KDJ_GOLD_LOW', 'PSY_BUY']

# 因子方向不参与优化，直接使用 RotationConfig 默认值
# 原因: 方向由金融逻辑决定（如波动率越大越不好），优化方向反而会引入噪声
FIXED_FACTOR_DIRECTIONS = RotationConfig().rank_factor_directions


def sample_config(trial: optuna.Trial, base_config: RotationConfig = None) -> RotationConfig:
    """从 Optuna Trial 采样一个完整的 RotationConfig

    这是参数搜索的核心: 将 Optuna 的 suggest_* API 映射到策略配置的每个字段。
    共采样 14 个可优化参数 + 8 个因子权重。

    采样策略:
    - 因子权重: 8 个独立采样 [0.01, 0.40]，然后归一化使总和 = 1.0
      为什么先独立采样再归一化而非用 Dirichlet 分布?
      → suggest_float 独立采样 + 归一化等价于 Dirichlet(1,1,...,1) 均匀分布，
        且 Optuna 对独立参数的超参空间建模更准确
    - 买入信号: 每个信号独立 on/off 开关，至少保证一个激活
      fallback_signal 的存在是为了保持参数空间一致性（TPE 需要固定维度）
    - 连续参数: 在合理的业务范围内均匀采样
    - 整数参数: max_positions [3,10], atr_period [7,21]

    Args:
        trial: Optuna trial 对象，提供 suggest_* 采样接口
        base_config: 基础配置（固定参数从此继承），None 则用默认值

    Returns:
        采样后的 RotationConfig，可直接传入 DailyRotationEngine
    """
    base = base_config or RotationConfig()

    # ── 因子权重: 8 个因子独立采样 [0.01, 0.40]，然后归一化 ──
    # 每个因子权重有独立的下限 0.01（避免某因子被完全淘汰）和上限 0.40（避免单因子过重）
    raw_weights = {}
    for factor in FIXED_FACTOR_DIRECTIONS:
        raw_weights[factor] = trial.suggest_float(f'weight_{factor}', 0.01, 0.40)
    # 归一化: 确保所有权重之和 = 1.0，使不同 Trial 的权重具有可比性
    total = sum(raw_weights.values())
    rank_factor_weights = {k: v / total for k, v in raw_weights.items()}

    # ── 买入信号开关: 8 个信号各自独立 on/off ──
    active_signals = []
    for sig in ALL_SIGNAL_TYPES:
        on = trial.suggest_categorical(f'signal_{sig}', ['on', 'off'])
        if on == 'on':
            active_signals.append(sig)
    # fallback 信号: 当所有信号都被关掉时，用此信号兜底
    # TPE 要求参数空间维度固定，即使实际不使用也必须采样
    fallback_signal = trial.suggest_categorical('fallback_signal', ALL_SIGNAL_TYPES)
    if not active_signals:
        active_signals.append(fallback_signal)

    # ── 信号逻辑模式 ──
    # OR: 任一信号触发即买入（宽松，候选多但质量参差）
    # AND: 所有信号同时触发才买入（严格，候选少但质量高）
    buy_signal_mode = trial.suggest_categorical('buy_signal_mode', ['OR', 'AND'])

    # ── 连续参数 ──
    # 总仓位上限: 30%~100%，市场好时可以满仓，差时空仓
    max_total_pct = trial.suggest_float('max_total_pct', 0.30, 1.00)
    # 单只股票仓位上限: 5%~30%，分散风险
    max_position_pct = trial.suggest_float('max_position_pct', 0.05, 0.30)
    # 过热度 RSI 阈值: 60~90，超过此值且短期涨幅大则触发过热惩罚
    overheat_rsi_threshold = trial.suggest_float('overheat_rsi_threshold', 60.0, 90.0)
    # 过热度 5 日涨幅阈值: 5%~30%，避免追高
    overheat_ret5_threshold = trial.suggest_float('overheat_ret5_threshold', 0.05, 0.30)
    # ATR 止损倍数: 1.0~3.5，买入价 - N×ATR 触发止损
    stop_loss_mult = trial.suggest_float('stop_loss_mult', 1.0, 3.5)
    # ATR 止盈倍数: 2.0~5.0，买入价 + N×ATR 触发止盈
    take_profit_mult = trial.suggest_float('take_profit_mult', 2.0, 5.0)
    # 移动止损回撤比例: 5%~20%，从最高点回落此比例触发卖出
    trailing_pct = trial.suggest_float('trailing_pct', 0.05, 0.20)
    # 移动止损启动阈值: 2%~10%，涨幅超过此值才开始跟踪最高点
    trailing_start = trial.suggest_float('trailing_start', 0.02, 0.10)

    # ── 整数参数 ──
    # 最大持仓数: 3~10 只，影响集中度
    max_positions = trial.suggest_int('max_positions', 3, 10)
    # ATR 计算周期: 7~21 天，短周期更敏感，长周期更平滑
    atr_period = trial.suggest_int('atr_period', 7, 21)
    # KDJ 低位金叉阈值: 20~40，K 值低于此阈值且金叉时触发 KDJ_GOLD_LOW 信号
    kdj_low_threshold = trial.suggest_float('kdj_low_threshold', 20.0, 40.0)

    # 组装配置: 固定参数从 base 继承，优化参数从 Trial 采样
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
        kdj_low_threshold=kdj_low_threshold,
    )


# ═══════════════════════════════════════════════
# 目标函数
# ═══════════════════════════════════════════════

# 最大回撤上限: 超过此值的 Trial 直接返回 0，引导 Optuna 避开高风险区域
# 设置为 30% 而非更低的理由: A 股波动大，过低会导致可用参数空间过窄
MAX_DRAWDOWN_LIMIT = 0.30


def objective(trial: optuna.Trial,
              start_date: str,
              end_date: str,
              base_config: RotationConfig = None,
              data_provider=None) -> float:
    """Optuna 目标函数: 采样参数 → 运行回测 → 返回年化 Sharpe

    这是 Optuna 优化的核心回调。每个 Trial 的完整生命周期:
    1. sample_config(trial) → 采样一组超参数
    2. DailyRotationEngine.run() → 用这组参数跑完整回测
    3. max_drawdown() + compute_sharpe() → 评估绩效
    4. 返回 Sharpe（Optuna 最大化此值）

    惩罚机制:
    - max_drawdown > 30% → 返回 0.0（强惩罚，直接淘汰）
    - 异常/无结果 → 返回 0.0
    Sharpe 为 0 在 TPE 采样中是明确的不良信号，会被逐渐排除。

    Args:
        trial: Optuna trial 对象
        start_date: 回测开始日期 'YYYY-MM-DD'
        end_date: 回测结束日期 'YYYY-MM-DD'
        base_config: 基础配置（固定参数从此继承）
        data_provider: 数据提供器（CachedProvider 或 DataProvider），None 则每次新建

    Returns:
        年化 Sharpe Ratio（供 Optuna 最大化）
    """
    # 1. 从 Trial 采样参数 → 构建本次回测的配置
    config = sample_config(trial, base_config)

    try:
        # 2. 创建引擎并运行完整回测
        engine = DailyRotationEngine(config, start_date, end_date,
                                     data_provider=data_provider)
        results = engine.run(trial=trial)

        # 4. 结果不足 → 返回 0（数据不够或回测失败）
        if not results or len(results) < 2:
            return 0.0

        # 5. 构建净值序列: [初始资金, 第1天总资产, 第2天总资产, ...]
        equity = [config.initial_capital] + [r.total_asset for r in results]

        # 6. 回撤惩罚: 超过 30% 直接淘汰
        dd = max_drawdown(equity)
        if dd > MAX_DRAWDOWN_LIMIT:
            return 0.0

        # 7. 计算目标值: 年化 Sharpe Ratio
        sharpe = compute_sharpe(equity, periods_per_year=252)
        return sharpe if not math.isnan(sharpe) else 0.0

    except Exception:
        # Trial 异常不中断整个优化，记录后返回 0
        # 常见原因: 数据窗口不足、参数组合导致除零等
        logger.debug(f"Trial {trial.number} failed", exc_info=True)
        return 0.0


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

    Walk-Forward 是量化策略验证的黄金标准: 模拟"用历史数据找到最优参数，
    然后在未来数据上验证"的真实场景，评估参数的稳定性和过拟合程度。

    窗口生成逻辑:
    - 窗口 1: 训练[2019-01-01, 2020-01-01), 测试[2020-01-01, 2020-07-01)
    - 窗口 2: 训练[2019-04-01, 2020-04-01), 测试[2020-04-01, 2020-10-01)
    - ...
    - 当测试期结束日超过 end_date 时停止

    典型配置:
    - train_months=12, test_months=6, step_months=3
      每个窗口训练 1 年数据，测试未来半年，每 3 个月滚动一次

    Args:
        start_date: 整体回测起始日
        end_date: 整体回测结束日
        train_months: 每个窗口的训练期长度（月）
        test_months: 每个窗口的测试期长度（月）
        step_months: 窗口滚动步长（月）

    Returns:
        窗口列表，每个元素为 (train_start, train_end, test_start, test_end)
    """
    windows = []
    # 训练起始日从回测起始日开始，每次迭代向前滚动 step_months
    train_start = start_date
    while True:
        # 训练期结束日 = 训练起始日 + train_months 个月 - 1 天
        # 例如: 1月1日 + 12个月 - 1天 = 12月31日（覆盖一个完整年度）
        train_end = train_start + pd.DateOffset(months=train_months) - pd.DateOffset(days=1)
        # 测试期起始日 = 训练期结束日的下一个交易日
        test_start = train_end + pd.DateOffset(days=1)
        # 测试期结束日 = 测试起始日 + test_months 个月 - 1 天
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)

        # 测试期超出整体区间 → 停止生成窗口
        if test_end > end_date:
            break

        windows.append((train_start, train_end, test_start, test_end))
        # 向前滚动: 训练起始日前进 step_months
        train_start = train_start + pd.DateOffset(months=step_months)

    return windows


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
    n_jobs: int = 1,
    data_provider=None,
    storage_url: str = None,
) -> Tuple[RotationConfig, float, optuna.Study]:
    """单期优化: 在固定日期范围内搜索最优参数

    这是最基本的优化模式。在 [start_date, end_date] 区间内运行 n_trials 次回测，
    每次用不同的参数组合，Optuna TPE 根据历史结果逐步收敛到最优区域。

    执行流程:
    1. 创建 Optuna Study（TPE 采样器，最大化 Sharpe）
    2. 构建预加载缓存（将前 30 天数据打包为单个 Parquet）
    3. 运行 n_trials 次 objective()，每轮采样 → 回测 → 评估
    4. 从 best_params 还原 best_config
    5. 保存结果到 JSON + CSV

    并行化说明:
    当 n_jobs > 1 时，Optuna 使用多进程并行运行 Trial。
    因为 CachedProvider 只做只读 Parquet 操作（无数据库连接），
    多进程间共享文件系统即可，无需进程间通信。

    Args:
        start_date: 回测开始日期 'YYYY-MM-DD'
        end_date: 回测结束日期 'YYYY-MM-DD'
        n_trials: 优化 Trial 总数
        base_config: 基础配置（固定参数从此继承）
        study_name: Optuna Study 名称（用于持久化存储的标识）
        output_dir: 结果输出目录（JSON + CSV）
        n_jobs: 并行 Trial 数（1=串行，-1=全部核心）
        data_provider: 数据提供器（CachedProvider 推荐），None 则每次新建 DataProvider
        storage_url: Optuna 持久化存储 URL（如 sqlite:///optuna.db），
                     None 则使用内存存储（进程退出后丢失）

    Returns:
        (best_config, best_sharpe, study)
    """
    # 创建 Optuna Study
    # - direction='maximize': 目标是最大化 Sharpe
    # - TPESampler(seed=42): 固定随机种子，保证相同参数空间可复现
    # - storage: 持久化存储（可选），用于中断后恢复或分布式优化
    storage_kwargs = {}
    if storage_url:
        storage_kwargs['storage'] = f"{storage_url}"
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
        **storage_kwargs,
    )

    # 创建目标函数的偏应用版本，闭包捕获固定参数
    obj = lambda trial: objective(trial, start_date, end_date, base_config, data_provider)
    # 启动优化: show_progress_bar=True 显示 tqdm 进度条
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)

    # 从 best_params（字典）重建 best_config（RotationConfig 对象）
    # 因为 Optuna 只能存储原始值（float/int/str），无法存储自定义对象
    best_config = _params_to_config(study.best_params, base_config)
    best_sharpe = study.best_value

    # 打印优化结果摘要
    print(f"\n{'=' * 60}")
    print(f"单期优化完成")
    print(f"区间: {start_date} ~ {end_date}")
    print(f"最优 Sharpe: {best_sharpe:.4f}")
    print(f"{'=' * 60}")

    # 保存结果: best_params.json + optuna_trials.csv
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
    n_jobs: int = 1,
    data_provider=None,
    storage_url: str = None,
) -> List[Dict]:
    """Walk-Forward 滚动优化

    滚动优化的完整流程:
    ┌──────────────────────────────────────────────────────┐
    │ 窗口 1: [训练 2020-01 ~ 2020-12] → [测试 2021-01 ~ 2021-06] │
    │ 窗口 2:     [训练 2020-04 ~ 2021-03] → [测试 2021-04 ~ 2021-09] │
    │ 窗口 3:         [训练 2020-07 ~ 2021-06] → [测试 2021-07 ~ 2021-12] │
    │ ...                                                  │
    └──────────────────────────────────────────────────────┘

    每个窗口:
    1. 在训练期上运行单期优化（run_single_optimization）
    2. 用在训练期找到的最优参数，在测试期上评估（_evaluate_on_test）
    3. 记录: 训练 Sharpe + 测试 Sharpe + 最优参数

    为什么 Walk-Forward 重要:
    - 单期优化找到的参数可能在样本外失效（过拟合）
    - 如果多个窗口的测试 Sharpe 都为正且稳定，说明参数具有泛化能力
    - 如果训练 Sharpe 远高于测试 Sharpe，说明存在过拟合

    汇总指标解读:
    - 测试 Sharpe 均值: 策略在样本外的平均表现
    - 测试 Sharpe 标准差: 表现稳定性，越小越好
    - 正 Sharpe 窗口比例: 策略一致性的直观指标

    Args:
        start_date: 整体回测起始日
        end_date: 整体回测结束日
        n_trials: 每个窗口的优化 Trial 数
        base_config: 基础配置
        train_months: 训练期长度（月）
        test_months: 测试期长度（月）
        step_months: 窗口滚动步长（月）
        output_dir: 结果输出目录
        n_jobs: 每个窗口内的并行 Trial 数
        data_provider: 数据提供器
        storage_url: Optuna 持久化存储 URL

    Returns:
        每个窗口一条记录: {window, train_start, train_end, test_start, test_end,
                          train_sharpe, test_sharpe, best_params}
    """
    # 1. 生成滚动窗口列表
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
        # 2. 打印当前窗口信息
        print(f"\n{'─' * 50}")
        print(f"窗口 {i + 1}/{len(windows)}")
        print(f"训练: {train_s.strftime('%Y-%m-%d')} ~ {train_e.strftime('%Y-%m-%d')}")
        print(f"测试: {test_s.strftime('%Y-%m-%d')} ~ {test_e.strftime('%Y-%m-%d')}")

        # 3. 训练期优化: 在训练期上搜索最优参数
        train_start_str = train_s.strftime('%Y-%m-%d')
        train_end_str = train_e.strftime('%Y-%m-%d')
        best_config, train_sharpe, study = run_single_optimization(
            train_start_str, train_end_str, n_trials, base_config,
            study_name=f"wf_window_{i}",  # 每个窗口独立命名，便于持久化区分
            output_dir=output_dir,
            n_jobs=n_jobs,
            data_provider=data_provider,
            storage_url=storage_url,
        )

        # 4. 测试期评估: 用训练期最优参数在测试期上跑一遍（不优化，纯评估）
        #    这是 Walk-Forward 的核心: 验证参数在样本外的表现
        test_start_str = test_s.strftime('%Y-%m-%d')
        test_end_str = test_e.strftime('%Y-%m-%d')
        test_sharpe = _evaluate_on_test(best_config, test_start_str, test_end_str, data_provider)

        # 5. 记录窗口结果
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

    # 6. 打印汇总统计
    _print_wf_summary(records)
    # 7. 保存完整结果
    _save_wf_results(records, output_dir)

    return records


# ═══════════════════════════════════════════════
# 内部辅助函数
# ═══════════════════════════════════════════════

def _params_to_config(params: Dict, base_config: RotationConfig = None) -> RotationConfig:
    """从 Optuna best_params 字典重建 RotationConfig

    为什么需要这个函数:
    Optuna 的 study.best_params 是一个纯字典 {参数名: 值}，不包含
    Trial 对象。我们需要从字典反向构建完整的 RotationConfig。

    与 sample_config() 的关系:
    两者是互逆操作:
    - sample_config: Trial.suggest → RotationConfig
    - _params_to_config: best_params dict → RotationConfig

    关键: 因子权重需要重新归一化，因为 best_params 中存储的是归一化后的值，
    但重建时仍需确保 weight 总和 = 1.0（浮点精度可能导致微小偏差）。

    Args:
        params: study.best_params 字典
        base_config: 基础配置（提供固定参数默认值）

    Returns:
        重建的 RotationConfig
    """
    base = base_config or RotationConfig()

    # 从 params 中提取原始权重并重新归一化
    # 虽然 best_params 中存的是归一化值，但重新归一化保证精度
    raw_weights = {factor: params[f'weight_{factor}']
                   for factor in FIXED_FACTOR_DIRECTIONS}
    total = sum(raw_weights.values())
    rank_factor_weights = {k: v / total for k, v in raw_weights.items()}

    # 重建信号列表: 遍历所有信号，收集状态为 'on' 的
    active_signals = [sig for sig in ALL_SIGNAL_TYPES
                      if params.get(f'signal_{sig}') == 'on']
    # 兜底: 如果没有激活信号，用 fallback_signal
    if not active_signals:
        active_signals = [params.get('fallback_signal', ALL_SIGNAL_TYPES[0])]

    # 构建 RotationConfig: 优化参数从 params 取，固定参数从 base 继承
    return RotationConfig(
        initial_capital=base.initial_capital,
        max_total_pct=params['max_total_pct'],
        max_position_pct=params['max_position_pct'],
        max_positions=params['max_positions'],
        buy_signal_types=active_signals,
        buy_signal_mode=params['buy_signal_mode'],
        sell_signal_types=params.get('sell_signal_types', base.sell_signal_types),
        rank_factor_weights=rank_factor_weights,
        rank_factor_directions=FIXED_FACTOR_DIRECTIONS,
        market_regime=base.market_regime,
        exclude_st=base.exclude_st,
        exclude_limit_up=base.exclude_limit_up,
        exclude_limit_down=base.exclude_limit_down,
        exclude_suspended=base.exclude_suspended,
        benchmark_index=base.benchmark_index,
        atr_period=params['atr_period'],
        stop_loss_mult=params['stop_loss_mult'],
        take_profit_mult=params['take_profit_mult'],
        trailing_pct=params['trailing_pct'],
        trailing_start=params['trailing_start'],
        overheat_rsi_threshold=params['overheat_rsi_threshold'],
        overheat_ret5_threshold=params['overheat_ret5_threshold'],
        kdj_low_threshold=params['kdj_low_threshold'],
    )


def _build_preload_cache(data_provider, base_config, start_date: str, end_date: str) -> Optional[str]:
    """构建预加载缓存: 将引擎初始化所需的 30 天历史打包为单个 Parquet

    性能意义:
    引擎构造函数中的 _preload_histories() 需要读取回测首日前 30 天的
    全市场数据。如果没有预加载缓存，每个 Trial 需要:
    - 遍历 30+ 个 Parquet 日文件
    - 过滤出 ~4760 只股票
    - concat 成 master DataFrame
    这个过程每个 Trial 耗时约 3-5 秒，100 个 Trial 就浪费 5-8 分钟。

    预加载缓存将上述操作降为单次 pd.read_parquet()（~0.5 秒）。

    限制条件:
    - 仅对 CachedProvider 生效（需要有 cache_dir 存放 preload.parquet）
    - DataProvider 模式不适用（数据来自 PostgreSQL，无法预打包）

    Args:
        data_provider: 数据提供器（需为 CachedProvider，否则跳过）
        base_config: 基础配置（提供 benchmark_index）
        start_date: 回测开始日期
        end_date: 回测结束日期

    Returns:
        preload.parquet 的路径，不可用时返回 None
    """
    # 检查是否有 cache 对象（只有 CachedProvider 有 cache 属性）
    cache = getattr(data_provider, 'cache', None)
    if cache is None:
        return None

    # 获取指数数据以确定第一个交易日
    base = base_config or RotationConfig()
    index_df = data_provider.get_index_data(base.benchmark_index, start_date, end_date)
    if index_df is None or index_df.empty:
        return None

    # 以第一个交易日为基准，生成前 30 天的预加载缓存
    first_date = index_df.index[0].strftime('%Y-%m-%d')
    output_path = str(cache.cache_dir / 'preload.parquet')
    # write_preload_cache 内部会用 get_histories 读取 30 天数据并合并写入
    cache.write_preload_cache(first_date, output_path)
    return output_path


def _evaluate_on_test(config: RotationConfig, start: str, end: str,
                      data_provider=None) -> float:
    """在测试集上评估给定配置（纯评估，不优化参数）

    用于 Walk-Forward 的测试期评估: 在训练期找到最优参数后，
    在测试期上跑完整回测，评估样本外表现。

    与 objective() 的区别:
    - objective: 参数来自 Trial 采样 → 用于优化
    - _evaluate_on_test: 参数固定 → 用于评估
    两者都返回 Sharpe，但后者不做任何参数搜索。

    注意: 这里不设 max_drawdown 惩罚，因为测试集上需要看到真实表现，
    包括可能出现的大回撤。惩罚只在优化阶段使用，避免过拟合被美化。

    Args:
        config: 待评估的完整配置（通常来自训练期最优参数）
        start: 测试期开始日期
        end: 测试期结束日期
        data_provider: 数据提供器

    Returns:
        年化 Sharpe Ratio
    """
    try:
        engine = DailyRotationEngine(config, start, end,
                                     data_provider=data_provider)
        results = engine.run()
        if not results or len(results) < 2:
            return 0.0
        equity = [config.initial_capital] + [r.total_asset for r in results]
        return compute_sharpe(equity)
    except Exception:
        return 0.0


def _config_to_dict(config: RotationConfig) -> Dict:
    """将 RotationConfig 序列化为与 best_params 兼容的字典格式

    序列化目的:
    1. 将最优配置保存为 JSON（_save_results）
    2. Walk-Forward 记录中包含每个窗口的最优参数（records['best_params']）

    设计原则:
    输出格式必须与 sample_config() 的 suggest_* 参数名一致，
    这样 _params_to_config() 可以无损还原。具体约定:
    - 因子权重: weight_{factor} 存储归一化后的值
    - 信号开关: signal_{sig} 存储 'on' 或 'off'
    - fallback_signal: 存储首选信号（用于当所有信号都 off 时的兜底）
    - 其他参数: 直接存储原始值

    Args:
        config: 待序列化的 RotationConfig

    Returns:
        与 study.best_params 格式兼容的字典
    """
    result: Dict = {}

    # 因子权重: 输出当前归一化值
    # 因为归一化后 sum=1.0，_params_to_config 二次归一化后值不变
    for factor, direction in FIXED_FACTOR_DIRECTIONS.items():
        result[f'weight_{factor}'] = config.rank_factor_weights.get(
            factor, 0.01 if factor == 'OVERHEAT' else 0.05
        )

    # 买入信号开关: 8 个信号各自标记 on/off
    for sig in ALL_SIGNAL_TYPES:
        result[f'signal_{sig}'] = 'on' if sig in config.buy_signal_types else 'off'
    # fallback 信号: 取 buy_signal_types 的第一个（当所有信号 off 时的兜底）
    result['fallback_signal'] = config.buy_signal_types[0] if config.buy_signal_types else ALL_SIGNAL_TYPES[0]

    # 连续 / 整数 / 分类参数: 直接映射
    result['buy_signal_mode'] = config.buy_signal_mode
    result['max_total_pct'] = config.max_total_pct
    result['max_position_pct'] = config.max_position_pct
    result['max_positions'] = config.max_positions
    result['atr_period'] = config.atr_period
    result['stop_loss_mult'] = config.stop_loss_mult
    result['take_profit_mult'] = config.take_profit_mult
    result['trailing_pct'] = config.trailing_pct
    result['trailing_start'] = config.trailing_start
    result['overheat_rsi_threshold'] = config.overheat_rsi_threshold
    result['overheat_ret5_threshold'] = config.overheat_ret5_threshold
    result['kdj_low_threshold'] = config.kdj_low_threshold
    result['sell_signal_types'] = list(config.sell_signal_types)

    return result


def _save_results(study: optuna.Study, best_config: RotationConfig,
                  best_sharpe: float, start: str, end: str,
                  output_dir: str = None):
    """保存单期优化结果到文件

    输出两个文件:
    1. best_params_{timestamp}.json — 最优参数 + 日期范围 + 最佳 Sharpe
       可用于: 手动复现、参数分享、后续评估
    2. optuna_trials_{timestamp}.csv — 所有 Trial 的完整记录
       可用于: 可视化优化过程、分析参数重要性、诊断收敛情况

    文件名包含时间戳: 避免多次运行互相覆盖，便于对比不同批次的优化结果。

    Args:
        study: 完成的 Optuna Study 对象
        best_config: 最优参数配置
        best_sharpe: 最优 Sharpe 值
        start: 回测开始日期
        end: 回测结束日期
        output_dir: 输出目录，默认当前目录
    """
    output_dir = Path(output_dir or '.')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存最优参数 JSON
    now = time.time()
    best_path = output_dir / f'best_params_{now}.json'
    best_data = {
        'start_date': start,
        'end_date': end,
        'best_sharpe': best_sharpe,
        'best_params': _config_to_dict(best_config),
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best_data, f, indent=2, ensure_ascii=False)
    print(f"最优参数已保存: {best_path}")

    # 保存所有 Trial 记录 CSV
    trials_path = output_dir / 'optuna_trials_{now}.csv'
    df = study.trials_dataframe()
    df.to_csv(trials_path, index=False, encoding='utf-8-sig')
    print(f"Trial 记录已保存: {trials_path} ({len(df)} 条)")


def _print_wf_summary(records: List[Dict]):
    """打印 Walk-Forward 汇总统计

    输出关键指标:
    - 窗口数: 有效窗口的总数
    - 测试 Sharpe 均值: 样本外平均表现，越高越好
    - 测试 Sharpe 中位数: 排除极端值后的中心趋势，用于判断策略是否稳定
    - 测试 Sharpe 标准差: 表现波动程度，越小说明越稳定（不同市场环境下表现一致）
    - 正 Sharpe 窗口比例: 策略在多少比例的窗口中是盈利的

    好的 Walk-Forward 结果:
    - 测试 Sharpe 均值 > 0.5 且标准差 < 0.5
    - 正 Sharpe 窗口 > 70%
    - 训练 Sharpe 和测试 Sharpe 接近（不过拟合）

    Args:
        records: run_walk_forward 返回的记录列表
    """
    print(f"\n{'=' * 60}")
    print("Walk-Forward 汇总")
    print(f"{'=' * 60}")
    test_sharpes = [r['test_sharpe'] for r in records]
    positive_count = sum(1 for s in test_sharpes if s > 0)
    print(f"窗口数: {len(records)}")
    print(f"测试 Sharpe 均值: {np.mean(test_sharpes):.4f}")
    print(f"测试 Sharpe 中位数: {np.median(test_sharpes):.4f}")
    print(f"测试 Sharpe 标准差: {np.std(test_sharpes):.4f}")
    print(f"正 Sharpe 窗口: {positive_count}/{len(records)}")


def _save_wf_results(records: List[Dict], output_dir: str = None):
    """保存 Walk-Forward 结果到 JSON

    输出文件: walkforward_results_{timestamp}.json
    内容: 每个窗口的训练/测试 Sharpe + 最优参数，完整可复现。

    用途:
    - 分析参数随时间的演变趋势
    - 检查是否存在"窗口特有参数"（过拟合信号）
    - 作为策略文档的量化支撑

    Args:
        records: run_walk_forward 返回的记录列表
        output_dir: 输出目录
    """
    now = time.time()
    output_dir = Path(output_dir or '.')
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'walkforward_results_{now}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Walk-Forward 结果已保存: {path}")


# ═══════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Daily Rotation 参数优化（Optuna + Parquet 缓存）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单期优化（4 进程并行，自动缓存）
  python back_testing/optimization/run_daily_rotation_optimization.py \\
      --mode single --start 2024-01-01 --end 2024-12-31 --trials 100 --n-jobs 4

  # Walk-Forward 优化
  python back_testing/optimization/run_daily_rotation_optimization.py \\
      --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50 --n-jobs 4
        """
    )
    parser.add_argument('--mode', choices=['single', 'walkforward'], default='single',
                        help='优化模式: single=单期, walkforward=滚动窗口')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--trials', type=int, default=100, help='每次优化的 Trial 数')
    parser.add_argument('--n-jobs', type=int, default=1, help='并行 Trial 数（-1=全部核心）')
    parser.add_argument('--output', default='.', help='结果输出目录')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='Parquet 缓存目录')
    parser.add_argument('--train-months', type=int, default=12, help='WF 训练期（月）')
    parser.add_argument('--test-months', type=int, default=6, help='WF 测试期（月）')
    parser.add_argument('--step-months', type=int, default=3, help='WF 步进（月）')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存（每次从 DB 查询）')
    parser.add_argument('--storage', default=None, help='Optuna 持久化存储（如 sqlite:///optuna.db），默认内存模式')
    parser.add_argument('--verbose', action='store_true', help='详细日志')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('back_testing.rotation').setLevel(logging.DEBUG)

    base_config = RotationConfig()

    # ── 构建 / 读取 Parquet 缓存 ──
    # DailyDataCache.build() 是增量构建: 已存在的 daily 文件自动跳过
    # 首次运行: 从 PostgreSQL 逐日查询写入 Parquet（一次性开销）
    # 后续运行: 跳过已缓存日期，只构建新日期（接近零开销）
    cached_provider = None
    if not args.no_cache:
        cache_path = DailyDataCache.build(
            start_date=args.start,
            end_date=args.end,
            cache_dir=args.cache_dir,
        )
        cache = DailyDataCache(cache_path)
        cached_provider = CachedProvider(cache)
        print(f"数据缓存就绪: {cache_path}")
        print(f"  {len(cache.stock_codes)} 只股票, {len(cache.trading_dates)} 个交易日")

    # 根据模式分发
    if args.mode == 'single':
        run_single_optimization(
            start_date=args.start,
            end_date=args.end,
            n_trials=args.trials,
            base_config=base_config,
            output_dir=args.output,
            n_jobs=args.n_jobs,
            data_provider=cached_provider,
            storage_url=args.storage,
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
            n_jobs=args.n_jobs,
            data_provider=cached_provider,
            storage_url=args.storage,
        )
