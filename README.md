# nj-quant

A股日频轮动量化回测系统 —— 基于多因子排序与信号检测的每日调仓策略，支持 Optuna 贝叶斯参数优化。

## 核心特性

- **日频轮动引擎** — 逐日处理：市场状态检测 → 信号特征构建 → 卖出检查 → 候选排序 → 买入分配，两阶段执行（先卖后买）
- **14 种买卖信号** — MA、MACD、RSI、KDJ、PSY 等技术指标信号检测，支持 AND/OR 组合模式
- **多因子加权排名** — z-score 标准化 + 方向调整的加权求和，因子权重可通过 Optuna 自动寻优
- **市场状态感知** — 强/中/弱三档市场分型，动态调整仓位上限与风控参数
- **风险管理** — ATR 止损/止盈、移动止损、过热惩罚（RSI + 5日涨幅阈值）
- **Optuna 优化** — TPE 贝叶斯搜索，支持单期优化与滚动窗口（walk-forward）验证
- **Parquet 缓存** — 优化阶段数据预构建为 Parquet，多 trial 零数据库查询

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 如需 Optuna 优化（可选）
pip install optuna
```

### 单次回测

```bash
python back_testing/backtest/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31
```

### 参数优化

```bash
# 单期优化
python back_testing/optimization/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 --trials 100

# 滚动窗口优化
python back_testing/optimization/run_daily_rotation_optimization.py \
    --mode walkforward --start 2022-01-01 --end 2024-12-31 --trials 50
```

### 运行测试

```bash
pytest tests/back_testing/ -v
```

## 项目结构

```
back_testing/
├── rotation/                  # 日频轮动引擎（核心）
│   ├── daily_rotation_engine.py  # 主引擎：向量化信号 + 两阶段执行
│   ├── config.py                 # RotationConfig 配置数据类
│   ├── market_regime.py          # 市场状态检测
│   ├── signal_engine/
│   │   ├── signal_filter.py      # 买卖信号检测（14种）
│   │   └── signal_ranker.py      # 多因子加权排名
│   └── trade_executor.py         # 交易执行与记录
├── optimization/              # 参数优化
│   └── run_daily_rotation_optimization.py  # Optuna CLI
├── data/                      # 数据层
│   ├── data_provider.py       # PostgreSQL 直连（SQLAlchemy）
│   └── daily_data_cache.py    # Parquet 缓存（优化用）
├── analysis/                  # 绩效分析
│   ├── performance_analyzer.py   # Sharpe/Calmar/回撤/胜率
│   └── visualizer.py
├── risk/                      # 风险管理
│   └── stop_loss_strategies.py   # ATR止损/移动止损
├── factors/                   # 因子工具
│   └── factor_utils.py        # 排名/z-score/缩尾处理
├── backtest/                  # 入口脚本
└── core/                      # 遗留回测引擎
```

## 技术栈

- **Python 3.12+**
- **Pandas / NumPy** — 数据处理与向量化计算
- **SQLAlchemy + PostgreSQL** — 行情数据存储
- **Optuna** — 贝叶斯超参数优化
- **AkShare / Tushare** — 数据获取
- **Matplotlib / Seaborn** — 可视化

## 配置

数据库连接配置位于 `config/database.ini`，格式如下：

```ini
[database]
host = localhost
port = 5432
database = your_db
username = your_user
password = your_pass
```
