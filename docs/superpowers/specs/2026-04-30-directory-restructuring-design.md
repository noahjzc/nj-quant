# 目录结构整理方案

**日期：** 2026/04/30
**状态：** 已批准，待实施

---

## 目标

将项目从按模块混合组织改为按功能域清晰划分，解决以下问题：

- `back_testing/` 臃肿（回测引擎 + 策略 + 数据 + 优化混杂）
- `signal_pipeline/` 结构混乱（信号生成 + 数据回填 + 第三方客户端混合）
- legacy 代码未清理
- 数据同步相关目录分散在两处

---

## 整理后的目录结构

```
nj-quant/
├── backtesting/              # 回测引擎（通用框架，不含策略）
│   ├── run_daily_rotation.py # 入口脚本
│   ├── run_composite_backtest.py
│   ├── analysis/             # 性能分析、可视化
│   │   ├── performance_analyzer.py
│   │   └── visualizer.py
│   └── risk/                 # 风险管理
│       ├── risk_manager.py
│       └── stop_loss_strategies.py
│
├── strategy/                  # 策略定义
│   ├── rotation/              # 每日轮转策略
│   │   ├── config.py
│   │   ├── daily_rotation_engine.py
│   │   ├── market_regime.py
│   │   ├── position_manager.py
│   │   ├── trade_executor.py
│   │   └── signal_engine/
│   │       ├── base_signal.py
│   │       ├── signal_filter.py
│   │       └── signal_ranker.py
│   ├── signals/               # 信号类型定义（通用）
│   ├── ranking/               # 多因子排名
│   └── factors/               # 因子工具
│       ├── factor_utils.py
│       └── factor_loader.py
│
├── data/                      # 数据层统一管理
│   ├── providers/             # 数据提供者
│   │   ├── data_provider.py
│   │   ├── cached_provider.py
│   │   └── index_data_provider.py
│   ├── cache/                 # Parquet 缓存
│   │   ├── daily_data_cache.py
│   │   └── build_daily_cache.py
│   ├── db/                    # 数据库模型
│   │   ├── models.py
│   │   └── connection.py
│   └── sync/                  # 数据同步客户端
│       ├── akshare_client.py
│       ├── baostock_client.py
│       ├── tushare_client.py
│       ├── overview_client.py
│       └── sync_overview.py
│
├── optimization/              # 参数优化
│   └── optuna/                # Optuna 优化框架
│       └── run_daily_rotation_optimization.py
│
├── signal_pipeline/           # 信号生成管线
│   ├── generators/            # 信号生成
│   │   ├── signal_generator.py
│   │   ├── intraday_signal.py
│   │   └── indicator_calculator.py
│   └── backfill/              # 数据回填
│       ├── night_backfill.py
│       ├── batch_backfill.py
│       └── data_merger.py
│
├── web/                       # 前端 + API
│   ├── frontend/
│   └── server/
│
├── scripts/                    # 运维脚本
│   ├── deploy/
│   └── *.sh / *.sql
│
└── tests/                     # 测试
    ├── backtesting/
    ├── strategy/
    ├── data/
    └── signal_pipeline/
```

---

## 文件迁移映射

| 原路径 | 新路径 |
|--------|--------|
| `back_testing/rotation/` | `strategy/rotation/` |
| `back_testing/analysis/` | `backtesting/analysis/` |
| `back_testing/risk/` | `backtesting/risk/` |
| `back_testing/backtest/run_daily_rotation.py` | `backtesting/run_daily_rotation.py` |
| `back_testing/backtest/run_composite_backtest.py` | `backtesting/run_composite_backtest.py` |
| `back_testing/optimization/run_daily_rotation_optimization.py` | `optimization/optuna/run_daily_rotation_optimization.py` |
| `back_testing/factors/` | `strategy/factors/` |
| `back_testing/rotation/signal_engine/` | `strategy/rotation/signal_engine/` |
| `signal_pipeline/data_sources/` | `data/sync/` |
| `signal_pipeline/generators/` | `signal_pipeline/generators/`（已存在） |
| `signal_pipeline/backfill/` | `signal_pipeline/backfill/`（已存在） |

---

## 需要删除的 legacy 代码

以下目录和文件在整理后应删除（不迁移）：

- `back_testing/core/`
- `back_testing/selectors/`
- `back_testing/strategies/`
- `back_testing/base_rotator.py`
- `back_testing/composite_rotator.py`
- `back_testing/composite_scorer.py`
- `back_testing/portfolio_backtest.py`
- `back_testing/portfolio_rotator.py`
- `back_testing/signal_scorer.py`
- `back_testing/strategy_evaluator.py`
- `back_testing/strategies/`（目录）

---

## 实施步骤

1. 创建新目录结构
2. 迁移文件到新位置
3. 更新所有 import 语句
4. 更新 `CLAUDE.md` 中的架构描述
5. 更新 `requirements.txt` 中的模块引用（如有）
6. 删除 legacy 目录和文件
7. 运行测试验证

---

## 测试目录结构

与源码目录结构对齐：

```
tests/
├── backtesting/
│   ├── rotation/
│   ├── analysis/
│   └── risk/
├── strategy/
│   ├── rotation/
│   ├── signals/
│   └── factors/
├── data/
│   ├── providers/
│   └── sync/
├── signal_pipeline/
│   ├── generators/
│   └── backfill/
└── optimization/
    └── optuna/
```

---

## 待确认

- [x] 测试目录结构已确认：与源码目录结构对齐
