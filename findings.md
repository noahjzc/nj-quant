# 代码审查发现记录

## 项目概览
- 量化选股回测系统（A股），已迁移 PostgreSQL
- ~50 个 Python 文件，2 套独立的选股/轮动系统并存

## 编码风格发现

### 两套选股系统
1. **PortfolioRotator** (老): 策略轮动，用 `SignalScorer` + `StockSelector`
2. **CompositeRotator** (新): 多因子/综合评分，用 `CompositeScorer` + `CompositeSelector` / `MultiFactorSelector`

### 数据访问层
- `DataProvider` 统一从 PostgreSQL 读取数据
- `FactorLoader` 使用批量查询（`get_batch_latest`），避免 N+1
- `CompositeSelector`/`StockSelector` 仍用逐只查询（N+1）

### GA 优化器
- 完整的遗传算法实现（Chromosome → Population → Selection → Crossover → Mutation → Elite + Niching）
- `niching.py` 定义了适应度共享算法但未被引用
- `mutation.py` 归一化逻辑与 `Chromosome._normalize` 重复

### 其他
- `stop_loss_strategies.py` 嵌入了 ~120 行测试代码
- `PerformanceVisualizer` 大量手动 HTML/CSS 模板代码
- `data_column_names.py` 常量与 DataProvider 列名对应正确

---

## 逻辑 Bug 发现（新增 2026-04-24）

### 🔴 Bug 1: MultiFactorSelector 方向=-1 评分反转
- **影响**: 影响所有 `direction=-1` 因子的选股结果（如 LN_MCAP、PB、PE_TTM 若被设置为 -1）
- **严重性**: 方向完全反了
- **根因**: `line 89: ascending = direction == -1` 逻辑反了，且 `direction=1` 的翻转也反了
- **修复**: 改为 `ascending = direction == 1`，取消 direction=1 的额外翻转

### 🔴 Bug 2: PerformanceAnalyzer 个股收益复利相乘
- **影响**: 所有调用 `PerformanceAnalyzer` 的绩效报告数据全部失真
- **根因**: `_calculate_total_return` 对个股收益率做 `∏(1+ri)-1`，但组合是多股同时持仓
- **示例**: 5 只各赚 10% 算成 61% 而非 10%
- **连带影响**: Sharpe、Calmar、Sortino 都基于这个 total_return 计算

### 🔴 Bug 3: GA fitness 剔除亏损股
- **影响**: GA 优化结果基于虚高适应度，最优权重可能不是真正最优
- **根因**: `_calculate_period_return` 中 `filtered_returns` 剔除亏损 >5% 的股票
- **示例**: 3 只亏 10% + 2 只赚 5% = 真实 -4%，但被算成 +5%

### 🟡 Bug 4: PerformanceAnalyzer 年化公式用交易次数当年数
- **影响**: 年化收益率几乎为 0
- **根因**: `n_years = len(returns)` 而不是实际时间
- **关联**: Bug 2 修复后此问题可联动解决

### 🟡 Bug 5: PositionManager.total_capital 静态不变
- **影响**: 仓位管理不随盈亏调整，回测失准
- **根因**: `calculate_buy_shares` 用 `self.total_capital`（固定值）忽略传入参数

### 🟡 Bug 6: run_rotator_backtest.py 伪回测
- **影响**: 回测结果无实际参考价值
- **根因**: 无复利、无交易成本、硬编码 10% 现金、每期重算
