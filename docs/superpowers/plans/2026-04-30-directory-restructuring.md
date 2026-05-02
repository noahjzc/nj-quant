# 目录结构整理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目目录从混合结构重排为按功能域划分的清晰结构

**Architecture:** 按功能域划分顶层目录，数据同步统一到 `data/sync/`，策略独立为 `strategy/rotation/`，回测引擎通用部分在 `backtesting/`

**Tech Stack:** 纯文件迁移 + import 语句更新，无技术栈变化

---

## 任务概览

| 阶段 | 任务数 | 描述 |
|------|--------|------|
| 阶段 1 | 1-3 | 创建新目录结构 |
| 阶段 2 | 4-15 | 迁移文件到新位置 |
| 阶段 3 | 16-17 | 更新 import 语句 |
| 阶段 4 | 18 | 更新 CLAUDE.md |
| 阶段 5 | 19 | 删除 legacy 代码 |
| 阶段 6 | 20 | 运行测试验证 |

---

## 阶段 1：创建新目录结构

### Task 1: 创建顶层目录

**Files:**
- Create: `backtesting/`
- Create: `strategy/rotation/`
- Create: `strategy/signals/`
- Create: `strategy/ranking/`
- Create: `strategy/factors/`
- Create: `data/providers/`
- Create: `data/cache/`
- Create: `data/db/`
- Create: `data/sync/`
- Create: `optimization/optuna/`
- Create: `signal_pipeline/generators/`
- Create: `signal_pipeline/backfill/`

- [ ] **Step 1: 创建所有新目录**

Run:
```bash
mkdir -p backtesting/analysis backtesting/risk
mkdir -p strategy/rotation/signal_engine strategy/signals strategy/ranking strategy/factors
mkdir -p data/providers data/cache data/db data/sync
mkdir -p optimization/optuna
mkdir -p signal_pipeline/generators signal_pipeline/backfill
```

---

## 阶段 2：迁移文件

### Task 2: 迁移 backtesting/analysis

**Files:**
- Copy: `back_testing/analysis/performance_analyzer.py` → `backtesting/analysis/`
- Copy: `back_testing/analysis/visualizer.py` → `backtesting/analysis/`

- [ ] **Step 1: 复制 analysis 文件**

Run:
```bash
cp back_testing/analysis/performance_analyzer.py backtesting/analysis/
cp back_testing/analysis/visualizer.py backtesting/analysis/
```

---

### Task 3: 迁移 backtesting/risk

**Files:**
- Copy: `back_testing/risk/risk_manager.py` → `backtesting/risk/`
- Copy: `back_testing/risk/position_manager.py` → `backtesting/risk/`
- Copy: `back_testing/risk/stop_loss_strategies.py` → `backtesting/risk/`

- [ ] **Step 1: 复制 risk 文件**

Run:
```bash
cp back_testing/risk/risk_manager.py backtesting/risk/
cp back_testing/risk/position_manager.py backtesting/risk/
cp back_testing/risk/stop_loss_strategies.py backtesting/risk/
```

---

### Task 4: 迁移 backtesting 入口脚本

**Files:**
- Copy: `back_testing/backtest/run_daily_rotation.py` → `backtesting/`
- Copy: `back_testing/backtest/run_composite_backtest.py` → `backtesting/`
- Copy: `back_testing/backtest/run_backtest.py` → `backtesting/`
- Copy: `back_testing/backtest/run_full_backtest.py` → `backtesting/`
- Copy: `back_testing/backtest/run_rotator_backtest.py` → `backtesting/`
- Copy: `back_testing/backtest/timing_back_testing.py` → `backtesting/`

- [ ] **Step 1: 复制 backtest 入口脚本**

Run:
```bash
cp back_testing/backtest/run_daily_rotation.py backtesting/
cp back_testing/backtest/run_composite_backtest.py backtesting/
cp back_testing/backtest/run_backtest.py backtesting/
cp back_testing/backtest/run_full_backtest.py backtesting/
cp back_testing/backtest/run_rotator_backtest.py backtesting/
cp back_testing/backtest/timing_back_testing.py backtesting/
```

---

### Task 5: 迁移 strategy/rotation

**Files:**
- Copy: `back_testing/rotation/config.py` → `strategy/rotation/`
- Copy: `back_testing/rotation/daily_rotation_engine.py` → `strategy/rotation/`
- Copy: `back_testing/rotation/market_regime.py` → `strategy/rotation/`
- Copy: `back_testing/rotation/position_manager.py` → `strategy/rotation/`
- Copy: `back_testing/rotation/trade_executor.py` → `strategy/rotation/`
- Copy: `back_testing/rotation/strategy.py` → `strategy/rotation/`
- Copy: `back_testing/rotation/__init__.py` → `strategy/rotation/`
- Copy: `back_testing/rotation/signal_engine/base_signal.py` → `strategy/rotation/signal_engine/`
- Copy: `back_testing/rotation/signal_engine/signal_filter.py` → `strategy/rotation/signal_engine/`
- Copy: `back_testing/rotation/signal_engine/signal_ranker.py` → `strategy/rotation/signal_engine/`
- Copy: `back_testing/rotation/signal_engine/__init__.py` → `strategy/rotation/signal_engine/`

- [ ] **Step 1: 复制 rotation 策略文件**

Run:
```bash
cp back_testing/rotation/config.py strategy/rotation/
cp back_testing/rotation/daily_rotation_engine.py strategy/rotation/
cp back_testing/rotation/market_regime.py strategy/rotation/
cp back_testing/rotation/position_manager.py strategy/rotation/
cp back_testing/rotation/trade_executor.py strategy/rotation/
cp back_testing/rotation/strategy.py strategy/rotation/
cp back_testing/rotation/__init__.py strategy/rotation/
```

- [ ] **Step 2: 复制 signal_engine 子目录**

Run:
```bash
cp back_testing/rotation/signal_engine/base_signal.py strategy/rotation/signal_engine/
cp back_testing/rotation/signal_engine/signal_filter.py strategy/rotation/signal_engine/
cp back_testing/rotation/signal_engine/signal_ranker.py strategy/rotation/signal_engine/
cp back_testing/rotation/signal_engine/__init__.py strategy/rotation/signal_engine/
```

---

### Task 6: 迁移 strategy/factors

**Files:**
- Copy: `back_testing/factors/factor_utils.py` → `strategy/factors/`
- Copy: `back_testing/factors/factor_loader.py` → `strategy/factors/`

- [ ] **Step 1: 复制 factors 文件**

Run:
```bash
cp back_testing/factors/factor_utils.py strategy/factors/
cp back_testing/factors/factor_loader.py strategy/factors/
```

---

### Task 7: 迁移 data/providers

**Files:**
- Copy: `back_testing/data/data_provider.py` → `data/providers/`
- Copy: `back_testing/data/daily_data_cache.py` → `data/cache/`
- Copy: `back_testing/data/build_daily_cache.py` → `data/cache/`
- Copy: `back_testing/data/index_data_provider.py` → `data/providers/`

- [ ] **Step 1: 复制 data providers**

Run:
```bash
cp back_testing/data/data_provider.py data/providers/
cp back_testing/data/daily_data_cache.py data/cache/
cp back_testing/data/build_daily_cache.py data/cache/
cp back_testing/data/index_data_provider.py data/providers/
```

- [ ] **Step 2: 创建 CachedProvider 软链接或复制**

`data/providers/cached_provider.py` 需要从 `daily_data_cache.py` 中的 `CachedProvider` 类拆分出来，或者直接在 `data/providers/` 下创建。具体需要读取源码确认。
（建议先读取 `back_testing/data/daily_data_cache.py` 确认 CachedProvider 是否已独立，如未独立则暂不处理，后续 Task 16 更新 import 时处理）

---

### Task 8: 迁移 data/db

**Files:**
- Copy: `back_testing/data/db/models.py` → `data/db/`
- Copy: `back_testing/data/db/connection.py` → `data/db/`
- Copy: `back_testing/data/db/__init__.py` → `data/db/`

- [ ] **Step 1: 复制 db 文件**

Run:
```bash
cp back_testing/data/db/models.py data/db/
cp back_testing/data/db/connection.py data/db/
cp back_testing/data/db/__init__.py data/db/
```

---

### Task 9: 迁移 data/sync（含 signal_pipeline/data_sources）

**Files:**
- Copy: `back_testing/data/sync/akshare_client.py` → `data/sync/`
- Copy: `back_testing/data/sync/baostock_client.py` → `data/sync/`
- Copy: `back_testing/data/sync/daily_update.py` → `data/sync/`
- Copy: `back_testing/data/sync/import_index_from_csv.py` → `data/sync/`
- Copy: `back_testing/data/sync/import_overview_data.py` → `data/sync/`
- Copy: `back_testing/data/sync/init_history.py` → `data/sync/`
- Copy: `back_testing/data/sync/overview_client.py` → `data/sync/`
- Copy: `back_testing/data/sync/sync_overview.py` → `data/sync/`
- Copy: `back_testing/data/sync/backfill_overview.py` → `data/sync/`
- Copy: `back_testing/data/sync/__init__.py` → `data/sync/`
- Copy: `signal_pipeline/data_sources/tushare_client.py` → `data/sync/`
- Copy: `signal_pipeline/data_sources/akshare_client.py` → `data/sync/`
- Copy: `signal_pipeline/data_sources/__init__.py` → `data/sync/`

- [ ] **Step 1: 复制 back_testing/data/sync 文件**

Run:
```bash
cp back_testing/data/sync/akshare_client.py data/sync/
cp back_testing/data/sync/baostock_client.py data/sync/
cp back_testing/data/sync/daily_update.py data/sync/
cp back_testing/data/sync/import_index_from_csv.py data/sync/
cp back_testing/data/sync/import_overview_data.py data/sync/
cp back_testing/data/sync/init_history.py data/sync/
cp back_testing/data/sync/overview_client.py data/sync/
cp back_testing/data/sync/sync_overview.py data/sync/
cp back_testing/data/sync/backfill_overview.py data/sync/
cp back_testing/data/sync/__init__.py data/sync/
```

- [ ] **Step 2: 复制 signal_pipeline/data_sources 文件到 data/sync/**

Run:
```bash
cp signal_pipeline/data_sources/tushare_client.py data/sync/
cp signal_pipeline/data_sources/akshare_client.py data/sync/
cp signal_pipeline/data_sources/__init__.py data/sync/
```

**注意：** 如果 `signal_pipeline/data_sources/akshare_client.py` 和 `back_testing/data/sync/akshare_client.py` 同时存在，需要合并或保留一份（检查两者是否完全相同）。

---

### Task 10: 迁移 optimization/optuna

**Files:**
- Copy: `back_testing/optimization/run_daily_rotation_optimization.py` → `optimization/optuna/`

- [ ] **Step 1: 复制优化入口**

Run:
```bash
cp back_testing/optimization/run_daily_rotation_optimization.py optimization/optuna/
```

---

### Task 11: 迁移 signal_pipeline/generators

**Files:**
- Copy: `signal_pipeline/signal_generator.py` → `signal_pipeline/generators/`
- Copy: `signal_pipeline/intraday_signal.py` → `signal_pipeline/generators/`
- Copy: `signal_pipeline/indicator_calculator.py` → `signal_pipeline/generators/`
- Copy: `signal_pipeline/data_merger.py` → `signal_pipeline/backfill/`

- [ ] **Step 1: 复制 generators**

Run:
```bash
cp signal_pipeline/signal_generator.py signal_pipeline/generators/
cp signal_pipeline/intraday_signal.py signal_pipeline/generators/
cp signal_pipeline/indicator_calculator.py signal_pipeline/generators/
```

- [ ] **Step 2: 复制 backfill**

Run:
```bash
cp signal_pipeline/night_backfill.py signal_pipeline/backfill/
cp signal_pipeline/batch_backfill.py signal_pipeline/backfill/
cp signal_pipeline/data_merger.py signal_pipeline/backfill/
```

---

### Task 12: 迁移 tests

**Files:**
- Create: `tests/backtesting/analysis/`、`tests/backtesting/risk/`、`tests/strategy/rotation/`、`tests/strategy/factors/`、`tests/data/sync/`、`tests/signal_pipeline/`、`tests/optimization/optuna/`
- Copy: 各测试文件到对应新目录

- [ ] **Step 1: 读取现有测试文件结构**

Run:
```bash
find tests/ -name "*.py" ! -path "*/__pycache__/*" ! -name "conftest.py"
```

- [ ] **Step 2: 创建测试子目录**

Run:
```bash
mkdir -p tests/backtesting/analysis tests/backtesting/risk
mkdir -p tests/strategy/rotation tests/strategy/factors
mkdir -p tests/data/sync
mkdir -p tests/signal_pipeline/generators tests/signal_pipeline/backfill
mkdir -p tests/optimization/optuna
```

- [ ] **Step 3: 根据 Task 1 的 find 结果，将测试文件复制到新位置**

---

## 阶段 3：更新 import 语句

### Task 13: 收集所有需要更新的 import 引用

**Files:**
- Grep 所有 import 语句，找出引用旧路径的文件

- [ ] **Step 1: 收集 import 引用**

Run:
```bash
grep -r "from back_testing" --include="*.py" . | grep -v ".venv" | grep -v "__pycache__"
grep -r "import back_testing" --include="*.py" . | grep -v ".venv" | grep -v "__pycache__"
grep -r "from signal_pipeline.data_sources" --include="*.py" . | grep -v ".venv" | grep -v "__pycache__"
```

记录所有需要更新的文件列表。

---

### Task 14: 更新 backtesting 目录下文件的 import

**Files:**
- Modify: `backtesting/run_daily_rotation.py`
- Modify: `backtesting/analysis/*.py`
- Modify: `backtesting/risk/*.py`

- [ ] **Step 1: 更新 backtesting 下文件的 import**

根据 Task 13 的结果，将 `backtesting/run_daily_rotation.py` 中的 `from back_testing.rotation` → `from strategy.rotation`，`from back_testing.data` → `from data`，`from back_testing.factors` → `from strategy.factors` 等。

---

### Task 15: 更新 strategy 目录下文件的 import

**Files:**
- Modify: `strategy/rotation/*.py`
- Modify: `strategy/rotation/signal_engine/*.py`
- Modify: `strategy/factors/*.py`

- [ ] **Step 1: 更新 strategy 下文件的 import**

将 `strategy/rotation/daily_rotation_engine.py` 中的 `from back_testing.data` → `from data`，`from back_testing.factors` → `from strategy.factors` 等。

---

### Task 16: 更新 data 目录下文件的 import

**Files:**
- Modify: `data/providers/*.py`
- Modify: `data/cache/*.py`
- Modify: `data/db/*.py`
- Modify: `data/sync/*.py`

- [ ] **Step 1: 更新 data 下文件的 import**

将 `data/sync/tushare_client.py` 等文件中的 `from signal_pipeline` 相关 import 移除或更新。

---

### Task 17: 更新 optimization/optuna 和 signal_pipeline 下文件的 import

**Files:**
- Modify: `optimization/optuna/run_daily_rotation_optimization.py`
- Modify: `signal_pipeline/generators/*.py`
- Modify: `signal_pipeline/backfill/*.py`

- [ ] **Step 1: 更新 optimization/optuna 的 import**

将 `optimization/optuna/run_daily_rotation_optimization.py` 中的 `from back_testing` → `from backtesting`/`from strategy`/`from data` 等。

- [ ] **Step 2: 更新 signal_pipeline 下文件的 import**

将 `signal_pipeline/generators/signal_generator.py` 等文件中的 `from signal_pipeline.data_sources` → `from data.sync`。

---

### Task 18: 更新 web/server 和 tests 下的 import

**Files:**
- Modify: `web/server/**/*.py`
- Modify: `tests/**/*.py`

- [ ] **Step 1: 更新 web/server 的 import**

检查 `web/server/main.py`、`web/server/api/*.py` 等文件的 import 并更新。

- [ ] **Step 2: 更新 tests 的 import**

检查所有 `tests/` 下的测试文件的 import 并更新。

---

## 阶段 4：更新文档

### Task 19: 更新 CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 更新 CLAUDE.md 中的架构描述**

将 CLAUDE.md 中关于目录结构的描述更新为新结构。关键修改：
- `back_testing/rotation/` → `strategy/rotation/`
- `back_testing/analysis/` → `backtesting/analysis/`
- `back_testing/risk/` → `backtesting/risk/`
- `back_testing/factors/` → `strategy/factors/`
- `back_testing/data/` → `data/`
- `back_testing/optimization/` → `optimization/optuna/`
- 添加 `signal_pipeline/generators/` 和 `signal_pipeline/backfill/` 的说明

---

## 阶段 5：删除 legacy 代码

### Task 20: 删除 back_testing 目录下的 legacy 文件和目录

**Files:**
- Delete: `back_testing/rotation/`
- Delete: `back_testing/analysis/`
- Delete: `back_testing/risk/`
- Delete: `back_testing/factors/`
- Delete: `back_testing/backtest/`
- Delete: `back_testing/optimization/`
- Delete: `back_testing/core/`
- Delete: `back_testing/selectors/`
- Delete: `back_testing/strategies/`
- Delete: `back_testing/data/`
- Delete: `back_testing/base_rotator.py`
- Delete: `back_testing/composite_rotator.py`
- Delete: `back_testing/composite_scorer.py`
- Delete: `back_testing/portfolio_backtest.py`
- Delete: `back_testing/portfolio_rotator.py`
- Delete: `back_testing/signal_scorer.py`
- Delete: `back_testing/strategy_evaluator.py`

- [ ] **Step 1: 确认所有迁移文件已正确复制到新位置**

Run:
```bash
# 验证关键文件存在
ls strategy/rotation/daily_rotation_engine.py
ls backtesting/analysis/performance_analyzer.py
ls data/sync/tushare_client.py
```

- [ ] **Step 2: 删除 back_testing 下的旧文件**

Run:
```bash
rm -rf back_testing/rotation back_testing/analysis back_testing/risk back_testing/factors
rm -rf back_testing/backtest back_testing/optimization back_testing/core
rm -rf back_testing/selectors back_testing/strategies back_testing/data
rm -f back_testing/base_rotator.py back_testing/composite_rotator.py
rm -f back_testing/composite_scorer.py back_testing/portfolio_backtest.py
rm -f back_testing/portfolio_rotator.py back_testing/signal_scorer.py
rm -f back_testing/strategy_evaluator.py
```

---

### Task 21: 删除 signal_pipeline 下的迁移后目录

**Files:**
- Delete: `signal_pipeline/data_sources/`

- [ ] **Step 1: 删除 signal_pipeline/data_sources**

Run:
```bash
rm -rf signal_pipeline/data_sources/
```

---

## 阶段 6：测试验证

### Task 22: 运行测试验证

**Files:**
- Run: pytest

- [ ] **Step 1: 运行所有测试**

Run:
```bash
pytest tests/ -v --tb=short 2>&1 | head -100
```

Expected: 测试应该通过（或有明确的 import 错误指向未更新的 import 语句）

- [ ] **Step 2: 如果有 import 错误，记录并修复**

回到 Task 14-18 继续更新 import。

---

### Task 23: 验证入口脚本可执行

**Files:**
- Run: `python backtesting/run_daily_rotation.py --help` 或类似命令验证路径正确

- [ ] **Step 1: 验证回测入口**

Run:
```bash
python backtesting/run_daily_rotation.py --help
```

- [ ] **Step 2: 验证优化入口**

Run:
```bash
python optimization/optuna/run_daily_rotation_optimization.py --help
```

---

## 实施检查清单

- [ ] 所有新目录已创建
- [ ] 所有文件已复制到新位置
- [ ] 所有 import 语句已更新
- [ ] CLAUDE.md 已更新
- [ ] legacy 目录和文件已删除
- [ ] 测试通过
- [ ] 入口脚本可执行

---

## 风险点

1. **import 更新遗漏** — 需要 grep 全面搜索，可能有多处遗漏
2. **signal_pipeline/data_sources/akshare_client.py 和 back_testing/data/sync/akshare_client.py 冲突** — 需要确认是否相同或需要合并
3. **CachedProvider** — 需要确认是否需要独立文件或从 daily_data_cache 导入
4. **根目录散落文件**（`main.py`、`data_processor/`、`fetch_data/`）— 暂不处理，如需整理可后续添加

