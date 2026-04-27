# 代码优化任务计划

## 目标
解决代码审查中发现的全部问题（编码优化 + 逻辑 bug），分 5 个阶段实施。

## 阶段总览

| 阶段 | 内容 | 优先级 | 涉及文件 | 状态 |
|------|------|--------|----------|------|
| 1 | **严重逻辑 Bug**（结果完全错误） | 🔴 紧急 | 3 | ✅ 完成 |
| 2 | **中等逻辑 Bug**（偏离正确结果） | 🔴 高 | 4 | ✅ 完成 |
| 3 | 编码质量与一致性 | 🟡 中 | 4 | ✅ 完成 |
| 4 | 架构整合 | 🟡 中 | 5 | ✅ 完成 |
| 5 | **清理（死代码/测试分离）** | 🟢 低 | 3 | ✅ 完成 |

---

## Phase 1: 严重逻辑 Bug（导致结果完全错误）

### ✅ 1.1 MultiFactorSelector 方向=-1 因子评分反转
- **文件**: `back_testing/selectors/multi_factor_selector.py`
- **修复**: `ascending = direction == -1` → `ascending = direction == 1`，移除 direction=1 的 `1 - processed` 翻转
- **影响**: `calculate_factor_scores()` 和 `get_factor_contribution()` 均已修复
- **验证**: 全部 10 个 MultiFactorSelector 测试通过

### ✅ 1.2 PerformanceAnalyzer 复利相乘个股收益
- **文件**: `back_testing/analysis/performance_analyzer.py`
- **修复**: 添加可选 `equity_curve` 参数，有 equity_curve 时从净值序列计算总收益/年化/Sharpe/Sortino/Calmar/最大回撤
- **影响**: `run_composite_backtest.py` 传入周频 equity_curve；win_rate/profit_loss_ratio 仍从交易计算
- **验证**: 全部 25 个 PerformanceAnalyzer 测试通过（无 equity_curve 时保持向后兼容）

### ✅ 1.3 GA 适应度评估剔除亏损股票（幸存者偏差）
- **文件**: `back_testing/optimization/genetic_optimizer/fitness.py`
- **修复**: `_calculate_period_return()` 中 `filtered_returns` 逻辑替换为 `np.mean(returns)`

---

## Phase 2: 中等逻辑 Bug（偏离正确结果）

### ✅ 2.1 PerformanceAnalyzer 年化收益率公式
- **文件**: `back_testing/analysis/performance_analyzer.py`
- **修复**: 与 1.2 联动解决，equity_curve 模式用 `(1+total_return)^(periods_per_year/n_periods)-1`

### ✅ 2.2 RiskManager/PositionManager 忽略传入的 total_capital
- **文件**: `back_testing/risk/position_manager.py`, `back_testing/risk/risk_manager.py`
- **修复**: `calculate_buy_shares` 和 `can_buy` 增加 `total_capital` 参数，`RiskManager.calculate_position_size` 透传

### ✅ 2.3 run_rotator_backtest.py 伪回测
- **文件**: `back_testing/backtest/run_rotator_backtest.py`
- **修复**: 实现现金+持仓跟踪、交易成本、等权分配资金

### ✅ 2.4 BacktestEngine 基准加载改 DataProvider
- **文件**: `back_testing/core/backtest_engine.py`
- **修复**: `load_benchmark()` 从 CSV 改为 `self.data_provider.get_index_data()`

---

## Phase 3: 编码质量与一致性

### ✅ 3.1 trade_date 列名一致性
- **文件**: `back_testing/core/backtest_engine.py`
- **修复**: `load_benchmark` 修复后使用 `data.index` 访问日期

### ✅ 3.2 mutation.py 归一化复用 Chromosome._normalize
- **文件**: `back_testing/optimization/genetic_optimizer/mutation.py`, `chromosome.py`
- **修复**: `_normalize` 改为 @staticmethod，mutation 中调用 `Chromosome._normalize(mutated.genes)`

### ✅ 3.3 FactorLoader.load_stock_turnover 简化列名查找
- **文件**: `back_testing/factors/factor_loader.py`
- **修复**: 列名模糊匹配 → 直接使用 `turnover_amount`

### ✅ 3.4 消除 BacktestEngine.data_path 默认 None 的隐患
- **文件**: `back_testing/core/backtest_engine.py`
- **修复**: load_benchmark 不再使用 data_path，标记为已废弃

---

## Phase 4: 架构整合

### ✅ 4.1 提取 BaseRotator 基类
- **文件**: `base_rotator.py`, `portfolio_rotator.py`, `composite_rotator.py`
- **方案**: 提取共享的 `rebalance()`、`run_weekly()` 模板方法，子类只实现 `select_stocks()`
- **验证**: 所有已有测试通过

### ✅ 4.2 StockSelector 合并+批量查询改造
- **文件**: `selectors/stock_selector.py`, `selectors/composite_selector.py`, `composite_rotator.py`
- **方案**: 
  - StockSelector 统一 CompositeSelector/StockSelector，支持 CompositeScorer 和 SignalScorer 两种模式
  - 使用 `DataProvider.get_batch_latest()` 替代逐只查询（N+1 → 1）
  - CompositeSelector 保留为薄包装
  - 给 `get_batch_latest()` 补上 macd_dif, macd_dea, boll_mid 字段

---

## Phase 5: 清理

### ✅ 5.1 移除 stop_loss_strategies.py 中的内嵌测试
### ✅ 5.2 PortfolioBacktest 使用中，保留
### ✅ 5.3 删除 niching.py（死代码，从未被引用）

---

## 决策记录

| 日期 | 决定 | 原因 |
|------|------|------|
| 2026-04-24 | 新增 Phase 1-2 逻辑 bug 修复 | 代码深入审查后发现 |
| 2026-04-24 | Bug 1.1 的测试需同步修正 | 旧测试用例匹配 buggy 行为 |

## 错误记录

| 文件 | 错误 | 处理 |
|------|------|------|
| `tests/back_testing/test_multi_factor_selector.py` | 测试预期基于 buggy 代码 | 已修正所有 5 个受影响的测试 |

---

## Phase 6: Daily Rotation 代码审查修复（新增）

> 来自 2026-04-25 代码审查

### 🔴 CRITICAL（必须修复）

#### 6.1 信号过滤器 4 个信号缺失
- **文件**: `back_testing/rotation/signal_engine/signal_filter.py`
- **问题**: `HIGH_BREAK`、`HIGH_BREAK_DOWN`、`VOL_DEATH`、`DMI_GOLD/DMI_DEATH` 未在 `_SIGNAL_MAP` 中注册
- **影响**: 用户配置这些信号类型会被静默忽略
- **修复**: 实现缺失的信号检测器类并加入 `_SIGNAL_MAP`

| 信号 | 状态 | 备注 |
|------|------|------|
| `VOL_DEATH` | ❌ 缺失 | 类已定义但未注册 |
| `DMI_GOLD` | ❌ 缺失 | 需要 `dmi_plus_di`/`dmi_minus_di` 列（数据中暂无） |
| `DMI_DEATH` | ❌ 缺失 | 同上 |
| `HIGH_BREAK` | ❌ 缺失 | 需要 N 日高点列（`high_N`），数据中暂无 |
| `HIGH_BREAK_DOWN` | ❌ 缺失 | 同上 |

#### 6.2 布林带信号永远返回 False
- **文件**: `back_testing/rotation/signal_engine/signal_filter.py`
- **问题**: `BollBreakSignal` 检查 `boll_upper` 列，但 `get_batch_latest` 只返回 `boll_mid`
- **修复**: 用 `boll_mid` 计算布林带上下轨：`boll_upper = boll_mid + 2*std`，`boll_lower = boll_mid - 2*std`

#### 6.3 涨跌停过滤失效
- **文件**: `back_testing/rotation/daily_rotation_engine.py`
- **问题**: `_filter_stock_pool` 检查 `limit_up`/`limit_down`，但 `get_batch_latest` 未返回这两列
- **修复**: 
  - 方案 A：修改 `get_batch_latest` 返回 `limit_up`/`limit_down`（涉及 db 层）
  - 方案 B：在 engine 层用 `change_pct >= 9.9%` 近似判断涨停

### 🟡 IMPORTANT（应该修复）

#### 6.4 VOLGoldSignal 实现与 spec 不符
- **文件**: `back_testing/rotation/signal_engine/signal_filter.py`
- **问题**: Spec 要求 `VOL_MA5 上穿 VOL_MA20`，实际是比较 `volume_ratio` 与其自身 5 日均值
- **修复**: 使用 `volume` 列计算 MA5/MA20，而非 `volume_ratio`

#### 6.5 `capital_after` 字段名误导
- **文件**: `back_testing/rotation/trade_executor.py`, `daily_rotation_engine.py`
- **问题**: `TradeRecord.capital_after` 存的是交易前资金，但字段名暗示交易后
- **修复**: 重命名为 `capital_before` 或计算交易后实际现金

### 🟢 MINOR（可选修复）

#### 6.6 未使用 import
- **文件**: `back_testing/rotation/daily_rotation_engine.py` 第 4 行
- **问题**: `from dataclasses import dataclass, field` 中 `field` 未使用
- **修复**: 删除 `field` import

#### 6.7 SignalFilter 静默忽略未知信号
- **文件**: `back_testing/rotation/signal_engine/signal_filter.py`
- **问题**: `__init__` 中未知信号类型被 try/except 吞掉，无警告
- **修复**: 记录 warning log 或抛出异常

---

### 阶段总览更新

| 阶段 | 内容 | 优先级 | 涉及文件 | 状态 |
|------|------|--------|----------|------|
| 1-5 | 既有优化任务 | - | - | ✅ 完成 |
| **6** | **Daily Rotation 审查修复** | 🔴 高 | 4 | ✅ 完成 |
