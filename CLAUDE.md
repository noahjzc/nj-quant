# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **quantitative stock trading** system for Chinese A-share markets (A股). It implements a daily rotation backtesting framework, Optuna-based parameter optimization, multi-factor stock ranking (Alpha158 + custom factors), ML-based ranking (LightGBM), signal pipeline for live trading, FastAPI web dashboard, and robustness testing.

## Running the Project

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows

# Single daily rotation backtest (uses Parquet cache by default)
python backtesting/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31

# Backtest with DB query (no cache)
python backtesting/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31 --no-cache

# Backtest with config file (from optimization output)
python backtesting/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31 --config output/best_params_xxx.json

# Backtest with ML ranker
python backtesting/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31 --ml-model models/lightgbm_model.pkl

# Optuna optimization — single-period
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 --trials 100

# Optuna optimization — walk-forward
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50

# Walk-forward with robustness-based selection
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50 --select-by-robustness

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/strategy/rotation/test_overheat.py -v
pytest tests/optimization/optuna/test_daily_rotation_optuna.py -v
pytest tests/strategy/factors/test_multi_factor_selector.py -v

# Train ML ranker
python strategy/ml/train.py --start 2020-01-01 --end 2023-12-31 --output models/lightgbm_model.pkl

# Start web dashboard
python -m web.server.main  # or: uvicorn web.server.main:app --reload
```

## Architecture

```
nj-quant/
├── backtesting/                  # 回测框架
│   ├── run_daily_rotation.py    # 入口脚本（CLI + 结果导出 + 稳健性检验）
│   ├── analysis/                # 绩效分析与可视化
│   │   ├── performance_analyzer.py  # 总收益/年化/Sharpe/Sortino/Calmar/胜率/IR/alpha-beta/偏度峰度/滚动Sharpe/月度收益
│   │   └── visualizer.py        # Matplotlib 图表 (净值/回撤/收益分布) + HTML 报告
│   ├── risk/                    # 风险管理
│   │   ├── risk_manager.py      # 统一协调止损止盈+仓位管理
│   │   ├── position_manager.py  # 仓位计算 (单只上限20%, 总仓90%, 整手买入)
│   │   └── stop_loss_strategies.py  # ATR止损/止盈/移动止损 (优先级: 止损 > 移动止损 > 正常)
│   └── costs/                   # 交易成本
│       ├── cost_model.py        # 印花税(0.1%卖)/过户费/佣金(0.03%, 最低5元)/滑点/平方根冲击
│       └── market_constraints.py  # 涨跌停板/ST/停牌/成交额过滤 + T+1约束
│
├── strategy/                     # 策略定义
│   ├── rotation/                # 每日轮转策略 (核心)
│   │   ├── daily_rotation_engine.py  # 核心引擎: Master DataFrame缓存, 向量化信号
│   │   ├── config.py            # RotationConfig, MarketRegimeConfig, Alpha158方向表
│   │   ├── market_regime.py     # 市场状态检测 (strong/neutral/weak)
│   │   ├── position_manager.py  # 策略层仓位管理 (融合市场状态感知)
│   │   ├── trade_executor.py    # 交易执行 + TradeRecord
│   │   ├── strategy.py          # 策略接口抽象
│   │   └── signal_engine/       # 信号引擎
│   │       ├── signal_filter.py # 买入/卖出信号检测 (14种信号类型)
│   │       └── signal_ranker.py # 多因子加权排名 (z-score + 方向调整加权和)
│   ├── factors/                 # 因子计算
│   │   ├── alpha158.py          # Alpha158 计算器 (156个因子: KBar 9 + Price 2 + Rolling 145)
│   │   ├── factor_utils.py      # FactorProcessor (rank, zscore, winsorize)
│   │   └── factor_loader.py     # 加载股票因子数据
│   ├── ml/                      # ML 排名器
│   │   ├── ml_ranker.py         # LightGBM 推理 (替代 SignalRanker, 接口兼容)
│   │   ├── ml_ranker_trainer.py # LightGBM 训练流水线
│   │   └── train.py             # 训练入口脚本
│   ├── signals/                 # 信号类型定义 (预留，当前为空)
│   └── ranking/                 # 多因子排名 (预留，当前为空)
│
├── data/                        # 数据层
│   ├── providers/               # 数据提供者
│   │   ├── data_provider.py     # 直连 PostgreSQL (SQLAlchemy)，含 get_stock_data / get_all_stock_codes / get_batch_histories / get_stocks_for_date
│   │   └── index_data_provider.py  # 指数数据 (sh000001等)
│   ├── cache/                   # Parquet 缓存
│   │   ├── daily_data_cache.py  # DailyDataCache + CachedProvider (跨trial复用)
│   │   └── build_daily_cache.py # 从DB批量构建缓存
│   ├── db/                      # 数据库模型
│   │   ├── models.py            # SQLAlchemy ORM: StockDaily, StockMeta, IndexDaily
│   │   └── connection.py        # 引擎+session工厂, 读取 config/database.ini
│   └── sync/                    # 数据同步客户端
│       ├── akshare_client.py    # AKShare 数据源
│       ├── baostock_client.py   # Baostock 数据源
│       ├── tushare_client.py    # Tushare 数据源
│       └── init_history.py      # 历史数据初始化
│
├── optimization/                 # 参数优化
│   └── optuna/                  # Optuna 优化框架
│       └── run_daily_rotation_optimization.py  # CLI (single/walkforward) + 14参数采样 + 稳健性筛选
│
├── signal_pipeline/             # 实盘信号生成管线
│   ├── generators/              # 信号生成器
│   │   ├── signal_generator.py  # 主信号生成器
│   │   ├── intraday_signal.py   # 日内信号
│   │   ├── indicator_calculator.py  # 指标计算
│   │   ├── roi_pipeline.py      # 投资回报率管线
│   │   ├── trade_war_pipeline.py   # 贸易战主题管线
│   │   ├── dce_pipeline.py      # 大宗商品管线
│   │   └── news_pipeline.py     # 新闻情绪管线
│   └── backfill/               # 数据回填
│       ├── night_backfill.py    # 夜间回填
│       ├── batch_backfill.py    # 批量回填
│       └── data_merger.py       # 数据合并
│
├── robustness/                   # 稳健性检验
│   ├── robustness_analyzer.py   # 统一入口 (Monte Carlo + CSCV + Deflated Sharpe)
│   ├── monte_carlo.py           # 蒙特卡洛模拟
│   ├── cscv.py                  # Combinatorial Symmetrical Cross-Validation
│   ├── sensitivity.py           # 参数敏感性分析
│   ├── statistics.py            # 统计工具
│   └── metrics.py               # 稳健性指标
│
├── web/                        # 前端 + API
│   ├── server/                 # FastAPI 后端
│   │   ├── main.py             # 应用入口 + CORS + 4路由挂载
│   │   ├── config.py           # Web配置
│   │   ├── schemas.py          # Pydantic 模型
│   │   └── api/                # API 路由
│   │       ├── signals.py      # 信号查询
│   │       ├── positions.py    # 持仓管理
│   │       ├── data_browser.py # 数据浏览器
│   │       └── cron_status.py  # 定时任务状态
│   └── frontend/               # 前端 (Vite/React)
│
├── scripts/                     # 运维脚本 (数据库升级、数据同步等)
├── tests/                       # 测试 (~27个测试文件, 与源码结构对齐)
├── config/                      # 配置文件
│   └── database.ini             # PostgreSQL 连接配置 (需自行创建)
├── docs/                        # 项目文档 (设计规格、实施计划)
├── cache/                       # 缓存数据 (Parquet文件, 运行时生成)
├── results/                     # 回测结果输出 (按时间戳命名)
└── .venv/                       # Python 虚拟环境 (Windows)
```

## Daily Rotation Engine

The engine (`DailyRotationEngine`) processes one trading day at a time with a rolling-window Master DataFrame cache:

1. **Preload (30+ days of history)**: Queries all stocks for the lookback window into a single Master DataFrame (indexed by `trade_date` + `stock_code`), avoiding per-stock I/O on each day
2. **Per-day loop**:
   - Update master cache with today's data (incremental, no repeated DB queries)
   - Detect market regime (strong/neutral/weak)
   - Filter candidate pool (exclude ST, limit up/down, suspended, low liquidity)
   - Build signal features — vectorized via groupby rolling transforms on `tail(21)` per stock
   - Check exits: sell signals → ATR stop-loss → trailing stop → take-profit monitoring
   - Rank remaining candidates (multi-factor z-score weighted sum or ML model)
   - Allocate buys to top-ranked candidates (two-phase: sell first, then buy)
3. **Two-phase execution**: Sell phase first (free up cash), then buy phase (allocate remaining cash to top candidates)
4. **Signal pipeline**: Layer 1 = binary signal detection (14 signal types), Layer 2 = multi-factor weighted ranking
5. **Cost deduction**: Every trade applies stamp duty, transfer fee, brokerage, slippage, and impact cost

Key performance optimizations:
- `_build_signal_features()`: Vectorized via groupby rolling transforms on the master cache, producing a feature matrix for ALL candidates in a single operation
- `_cache_df` updated incrementally per day (no repeated queries)
- `CachedProvider` reads from pre-built Parquet files (zero DB queries during optimization trials)

## Configuration

`RotationConfig` is a Python dataclass defined in `strategy/rotation/config.py`. **All fields must have type annotations** — unannotated fields become class variables, not instance fields, and won't appear in `__init__`.

Key config groups:
- **Position sizing**: `max_total_pct`, `max_position_pct`, `max_positions`, `initial_capital`
- **Buy signals**: `buy_signal_types` (list of signal names), `buy_signal_mode` (`'AND'`/`'OR'`)
- **Sell signals**: `sell_signal_types`, `sell_signal_mode`
- **Rank factors**: `rank_factor_weights` (dict: factor_name → weight), `rank_factor_directions` (dict: factor_name → 1 or -1)
- **Market regime**: `MarketRegimeConfig` — strong/neutral/weak thresholds for index return, MA alignment, volume ratio
- **Stops**: ATR period, stop-loss multiplier, take-profit multiplier, trailing stop %, trailing start %
- **Overheat penalty**: RSI threshold + 5-day return threshold to avoid chasing overbought stocks
- **Alpha158 integration**: `config.py` exports `ALPHA158_DIRECTIONS` (158 factor direction defaults) and `add_alpha158_factors()` helper to bulk-add all Alpha158 factors with sensible defaults
- **Benchmark**: `benchmark_index` (default `'sh000001'`)

## Data Access: Two Providers

**`DataProvider`** (direct DB, `data/providers/data_provider.py`): Queries PostgreSQL via SQLAlchemy. Used for single backtests.
- `get_stock_data(code, date, start_date, end_date)` → DataFrame
- `get_all_stock_codes()` → list
- `get_batch_histories(codes, end_date, start_date)` → `{code: DataFrame}`
- `get_stocks_for_date(codes, date)` → `{code: dict}` — **dicts do NOT contain `stock_code` key**
- `get_index_data(index_code, start_date, end_date)` → DataFrame

**`CachedProvider`** (Parquet, `data/cache/daily_data_cache.py`): Reads from pre-built Parquet cache. Used by Optuna trials for cross-trial data reuse.
- Same interface as DataProvider for the three main methods
- Extra: `get_daily_dataframe(date)` → full market DataFrame (used by engine fast path)

Build cache once before optimization: `DailyDataCache.build(start, end, cache_dir)` (built from DB via `DataProvider`).

Database connection is configured in `config/database.ini` (must be created — not checked in).

## Alpha158 Factors

`Alpha158Calculator` (`strategy/factors/alpha158.py`) computes 156 factors without Qlib dependency:
- 9 KBar factors: OHLC candle patterns (KMID, KLEN, KUP, KLOW, KSFT, etc.)
- 2 Price factors: OPEN0 (open/close), HIGH0 (high/close), LOW0 (low/close)
- 145 Rolling factors: 5 windows [5,10,20,30,60] × 29 operators (momentum, volatility, trend, volume, correlation, etc.)
- Input: DataFrame with `open/high/low/close/volume` columns
- Output: (n_rows, 156) factor DataFrame
- Uses Qlib semantics: `Greater` = maximum, `Less` = minimum, `Ref` = lag, all rolling with `min_periods=1`

Factor directions (`config.py`): 1 = larger is better (momentum, trend strength), -1 = smaller is better (volatility, downside risk).

## ML Ranking

`MLRanker` (`strategy/ml/ml_ranker.py`) replaces `SignalRanker` with a trained LightGBM model:
- Same interface: `rank(factor_df, top_n) → List[str]`
- Loads model via `joblib`, auto-detects required feature names
- Warns if >30% of expected features are missing
- Trained via `MLRankerTrainer` → `strategy/ml/train.py` entry script

## Optimization

Uses **Optuna TPE** (Tree-structured Parzen Estimator) for Bayesian optimization, implemented in `optimization/optuna/run_daily_rotation_optimization.py` (~1060 lines).

14 sampled parameters via `sample_config()`:
- Position sizing: `max_position_pct`, `max_total_pct`
- Signal mode: `buy_signal_mode` (AND/OR)
- Factor weights: Alpha158 group weights with direction awareness
- ATR stops: stop_loss_mult, take_profit_mult, trailing_stop_pct, trailing_start_pct
- Overheat thresholds: `rsi_threshold`, `ret5_threshold`
- Data: `preload_days`, `min_days_listed`

Modes:
- **`single`**: Optimize over one date range, minimize `-Sharpe` (with optional `--select-by-robustness`)
- **`walkforward`**: Rolling windows with parameter carry-forward, reports per-window stats
- Robustness-based selection (`--select-by-robustness`): Runs Monte Carlo bootstrapping + CSCV on top-N trials, selects the most robust config (not just highest Sharpe)

Dependencies: `optuna`, `scipy` are required but NOT listed in `requirements.txt`.

## Robustness Testing

`RobustnessAnalyzer` (`robustness/robustness_analyzer.py`) runs automatically at the end of every backtest (`run_daily_rotation.py`):
- **Monte Carlo**: Bootstraps trade returns to estimate Sharpe 95% CI and max drawdown distribution
- **CSCV** (Combinatorial Symmetrical Cross-Validation): Estimates overfit probability and rank decay
- **Deflated Sharpe**: Adjusts Sharpe for multiple testing
- Results saved to `results/*/robustness.json` alongside `metrics.json`

## Web Dashboard

FastAPI app at `web/server/main.py`:
- **`/signals`** — signal generation status and results
- **`/positions`** — current positions and P&L
- **`/data`** — data browser for factor/price exploration
- **`/cron`** — scheduled task monitoring
- Swagger docs at `/docs`, CORS enabled for local dev
- Frontend in `web/frontend/` (Vite/React)

## Testing

27 test files organized mirroring the source tree:
```
tests/
├── strategy/rotation/   # Engine, overheat penalty, signal filter/ranker tests
├── strategy/factors/    # Multi-factor selector, Alpha158 tests
├── strategy/ml/         # ML ranker tests
├── data/providers/      # Data provider integration tests
├── data/sync/           # Data sync client tests
├── optimization/optuna/ # Optuna optimization tests
├── backtesting/         # Backtesting integration tests
├── robustness/          # Robustness analyzer tests
├── signal_pipeline/     # Signal pipeline tests
└── data/cache/          # Cache verification tests
```

Run with `pytest tests/ -v`. Note: `pytest.ini` sets `pythonpath` to a worktree path — you may need to adjust for your environment.

## Known Pitfalls

1. **Dataclass fields need type annotations**: `field_name = value` creates a class variable, not an instance field. Use `field_name: type = value`. Missing annotations on `RotationConfig` caused all Optuna trials to fail with `unexpected keyword argument`.

2. **`get_stocks_for_date()` missing `stock_code`**: Row dicts from `DataProvider.get_stocks_for_date()` don't include `stock_code`. Engine's `_advance_to_date` must add it: `row_data['stock_code'] = stock_code`. Missing this caused zero trades.

3. **Signal feature index alignment**: `groupby.last()` returns `stock_code` index; `groupby.nth(-2)` returns original index. Both must be aligned to the same index before combining into a feature DataFrame.

4. **`config/database.ini` not checked in**: Database connection config must be created manually. Format: standard INI with `[postgresql]` section containing `host`, `port`, `database`, `user`, `password`.

5. **`pytest.ini` worktree path**: The `pythonpath` in `pytest.ini` points to a `.worktrees/live-signal` path that may not exist on your machine. Update to the actual project root if tests fail with import errors.

6. **`optuna` not in `requirements.txt`**: Both `optuna` and `scipy` must be installed separately for optimization to work.

## Development Notes

- **Stock codes**: Shanghai `sh` prefix (e.g., `sh600519`), Shenzhen `sz` prefix (e.g., `sz000001`)
- **Factor directions**: `1` = larger is better (typically momentum/trend), `-1` = smaller is better (typically volatility/downside)
- **T+1 rule**: Stocks bought on day T cannot be sold on the same day
- **Lot size**: A-shares trade in lots of 100 shares (整手), position sizes floor to nearest 100
- **Parquet cache**: Built once, reused across trials — dramatically speeds up optimization (minutes → seconds per trial)
- **Do not use git**: The project does not use git for version control
- **Virtual environment**: `.venv/` is standard, activate with `.venv\Scripts\activate` on Windows
- **Results output**: Each backtest run creates `results/{timestamp}_performance/` with equity curve CSV, trades CSV, positions CSV, metrics JSON, robustness JSON, and HTML report with charts


## Developer Rules

- **Discussed First**: Don't start writing the code first. Any issue should be discussed first, and only after it is recorded in a document can the coding process begin.
- **Brain Storming**: Complex tasks require the use of "brainstorming" skills first.
- **Code Review**: After completing each code writing task or document, a code review must be conducted using the "requesting code review" and "code simplifier" skills.