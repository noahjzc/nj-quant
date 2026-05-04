# nj-quant

A 股量化交易系统 —— 日频轮动回测、多因子排名、机器学习选股、Optuna 超参优化、时序特征学习。

## 核心特性

- **日频轮动引擎** — 向量化信号生成、两阶段交易执行（先卖后买）、市场状态感知（强/中/弱三档）
- **Alpha158 因子** — 156 个金融因子（K 线形态 9 个、价格比率 2 个、5 窗口 × 29 种滚动算子 = 145 个）
- **因子筛选与正交化** — Rank IC 分析、Gram-Schmidt 去相关、双输出（原始精选因子 + 正交化因子）
- **ML 排名** — LightGBM 模型训练与推理、Optuna 超参搜索
- **时序特征层** — Transformer Encoder 自监督预训练（遮罩因子预测）+ LightGBM 联合排名
- **Optuna 优化** — TPE 贝叶斯搜索、单期优化 + 滚动窗口（Walk-Forward）验证、稳健性择优
- **稳健性检验** — 蒙特卡洛模拟、CSCV 过拟合检测、Deflated Sharpe、参数敏感性分析
- **Web 看板** — FastAPI 后端 + React 前端，支持信号监控与数据浏览

## 快速开始

```bash
# 克隆项目
git clone <repo-url> && cd nj-quant

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
pip install -e ".[dev]"  # 含测试依赖

# 配置数据库
cp config/database.example.ini config/database.ini
# 编辑 database.ini，填入 PostgreSQL 连接信息

# 构建数据缓存（从数据库读取行情数据，写入 Parquet）
python -c "
from data.cache.daily_data_cache import DailyDataCache
DailyDataCache.build('2023-01-01', '2024-12-31', 'cache/daily_rotation')
"

# 运行一次回测
python backtesting/run_daily_rotation.py --start 2024-01-01 --end 2024-12-31
```

## 三阶段优化管线

```bash
# ═══════════ Stage 0: 因子筛选与正交化 ═══════════
# 从 158 个 Alpha 因子中筛选有效因子，产出原始精选 + 正交化两套因子集
python -m strategy.factors.factor_screening \
    --start 2020-01-01 --end 2022-12-31 \
    --output output/

# ═══════════ Stage 1: ML 模型训练 ═══════════
# Step 1a: 时序 Encoder 预训练（自监督，无需收益标签）
python -m strategy.ml.temporal.pretrain \
    --start 2020-01-01 --end 2022-12-31 \
    --factors output/selected_factors.json \
    --epochs 50 --output output/

# Step 1b: ML 排名模型训练（Encoder 提取时序特征 + LightGBM 训练 + Optuna 调参）
python optimization/optuna/run_ml_optimization.py \
    --train-start 2020-01-01 --train-end 2022-12-31 \
    --factors output/selected_factors.json \
    --encoder output/temporal_encoder.pt \
    --trials 50 --output output/

# ═══════════ Stage 2: 交易框架参数优化 ═══════════
# --ml-model auto 自动发现并加载 Stage 1 产出的最优模型
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 \
    --ml-model auto --trials 100 --output output/

# 跳过稳健性筛选以加速（开发迭代时推荐）
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 \
    --ml-model auto --trials 100 --skip-robustness

# 单独运行参数稳定性筛选
python -m robustness.sensitivity_report \
    --params output/best_params_xxx.json \
    --start 2024-01-01 --end 2024-12-31 \
    --ml-model output/best_model.pkl
```

## CLI 命令速查

### 安装后命令（pip install -e . 后可用）

| 命令 | 功能 |
|------|------|
| `nj-quant-backtest` | 运行单次日频轮动回测 |
| `nj-quant-optimize` | 运行交易框架参数优化 (Optuna) |
| `nj-quant-ml-optimize` | 运行 ML 模型训练与超参优化 |
| `nj-quant-factor-screen` | 运行因子筛选与正交化 |
| `nj-quant-pretrain` | 运行时序 Encoder 自监督预训练 |

### 直接运行

| 脚本 | 功能 |
|------|------|
| `python backtesting/run_daily_rotation.py` | 单次回测（含绩效分析、可视化报告） |
| `python optimization/optuna/run_daily_rotation_optimization.py` | 交易框架参数 Optuna 优化 |
| `python optimization/optuna/run_ml_optimization.py` | ML 模型训练 + 超参优化 |
| `python -m strategy.factors.factor_screening` | 因子 IC 分析与正交化筛选 |
| `python -m strategy.ml.temporal.pretrain` | 时序 Encoder 自监督预训练 |
| `python -m robustness.sensitivity_report` | 参数稳定性筛选（敏感性分析） |

## 架构

```
nj-quant/
├── backtesting/                  # 回测框架
│   ├── run_daily_rotation.py    # CLI 入口 + 结果导出
│   ├── analysis/                # 绩效分析 (Sharpe/Calmar/回撤/IR) + 可视化
│   ├── risk/                    # 风险管理 (ATR止损/止盈/移动止损/仓位管理)
│   └── costs/                   # 交易成本 (印花税/佣金/滑点/冲击成本)
│
├── strategy/                     # 策略层
│   ├── rotation/                # 日频轮动引擎（核心）
│   │   ├── daily_rotation_engine.py  # 主引擎: Master DF 缓存 + 向量化信号
│   │   ├── config.py            # RotationConfig 配置数据类
│   │   ├── market_regime.py     # 市场状态检测 (强/中/弱)
│   │   ├── position_manager.py  # 仓位管理（融合市场状态）
│   │   ├── trade_executor.py    # 交易执行 + TradeRecord
│   │   └── signal_engine/       # 信号引擎
│   │       ├── signal_filter.py # 买卖信号检测 (14种信号类型)
│   │       └── signal_ranker.py # 多因子加权排名 (z-score + 方向调整)
│   ├── factors/                 # 因子层
│   │   ├── alpha158.py          # Alpha158 因子计算器 (156因子)
│   │   ├── factor_utils.py      # FactorProcessor (rank/zscore/winsorize)
│   │   └── factor_screening.py  # 因子筛选 (IC分析 + Gram-Schmidt正交化)
│   └── ml/                      # ML 排名
│       ├── ml_ranker.py         # LightGBM 推理排名器
│       ├── trainer.py           # MLRankerTrainer（训练数据构造 + 模型训练）
│       ├── ml_optuna.py         # ML 超参 Optuna 优化
│       └── temporal/            # 时序特征层
│           ├── encoder.py       # TemporalEncoder (轻量 Transformer)
│           ├── pretrain.py      # 自监督预训练（遮罩因子预测）
│           ├── temporal_ranker.py    # TemporalMLRanker (Encoder + LightGBM)
│           └── temporal_trainer.py   # Phase 2 联合训练
│
├── optimization/                 # 参数优化
│   └── optuna/
│       ├── run_daily_rotation_optimization.py  # 框架参数优化 CLI
│       └── run_ml_optimization.py              # ML 超参优化 CLI
│
├── data/                        # 数据层
│   ├── providers/               # 数据提供者 (PostgreSQL 直连)
│   ├── cache/                   # Parquet 缓存 (DailyDataCache + CachedProvider)
│   └── db/                      # SQLAlchemy ORM 模型
│
├── robustness/                   # 稳健性检验
│   ├── robustness_analyzer.py   # 统一入口 (Monte Carlo + CSCV + Deflated Sharpe)
│   ├── sensitivity.py           # 参数敏感性分析器
│   ├── sensitivity_report.py    # 敏感性分析 CLI
│   ├── monte_carlo.py           # 蒙特卡洛模拟
│   └── cscv.py                  # CSCV 过拟合检测
│
├── signal_pipeline/             # 实盘信号管线
├── web/                         # Web 看板 (FastAPI + React)
├── docs/superpowers/            # 设计文档 + 实施计划
├── tests/                       # 测试套件
├── config/                      # 配置文件
├── pyproject.toml               # 项目元数据 + 依赖
└── requirements.txt             # pip 依赖列表
```

## 日频轮动引擎

引擎 (`DailyRotationEngine`) 逐日处理，核心流程：

1. **预加载**: 引擎启动前加载 30+ 天全市场历史数据到 Master DataFrame（索引 `trade_date × stock_code`），避免逐股 I/O
2. **每日循环**:
   - 增量更新 Master 缓存（当天新数据追加，无重复 DB 查询）
   - 检测市场状态（强/中/弱，基于指数收益率 + 均线排列 + 成交量比）
   - 过滤候选池（排除 ST/涨跌停/停牌/低流动性）
   - 构建信号特征（向量化 groupby rolling transform，全候选股一次计算）
   - 卖出检查（卖出信号 → ATR 止损 → 移动止损 → 止盈监控）
   - 排名买入候选（多因子加权 z-score 或 ML 模型预测）
   - 两阶段执行：先卖后买
3. **交易成本**: 每次交易扣除印花税（0.1% 卖）、过户费、佣金（0.03% 最低 5 元）、滑点、冲击成本

## 数据访问

两种数据提供者，接口一致：

| 提供者 | 数据源 | 适用场景 |
|--------|--------|----------|
| `DataProvider` | PostgreSQL 直连 (SQLAlchemy) | 单次回测 |
| `CachedProvider` | Parquet 文件 | Optuna 优化 (跨 Trial 复用) |

构建缓存一次，优化阶段零数据库查询：

```python
from data.cache.daily_data_cache import DailyDataCache
DailyDataCache.build('2023-01-01', '2024-12-31', 'cache/daily_rotation')
```

## 技术栈

| 类别 | 核心包 | 用途 |
|------|--------|------|
| ML | lightgbm, scikit-learn, joblib | 股票排名模型训练与推理 |
| 深度学习 | torch | TemporalEncoder (时序特征提取) |
| 优化 | optuna, scipy | 贝叶斯超参搜索、IC 计算 |
| 数据处理 | pandas, numpy | 向量化计算、因子工程 |
| 数据库 | sqlalchemy, psycopg2 | PostgreSQL 行情读写 |
| 数据源 | akshare, tushare | 行情数据获取 |
| Web | fastapi, uvicorn, pydantic | API 服务 |
| 可视化 | matplotlib, seaborn | 绩效图表、HTML 报告 |

Python ≥ 3.10。完整依赖列表见 `pyproject.toml` 或 `requirements.txt`。

## 测试

```bash
pytest tests/ -v                             # 运行全部测试
pytest tests/strategy/ml/temporal/ -v        # 时序特征层测试
pytest tests/robustness/ -v                  # 稳健性检验测试
pytest tests/strategy/factors/ -v            # 因子筛选测试
```

## 设计文档

关键设计规格位于 `docs/superpowers/specs/`：

| 文档 | 内容 |
|------|------|
| `2026-04-25-daily-rotation-design.md` | 日频轮动引擎设计 |
| `2026-04-26-daily-rotation-optuna-design.md` | Optuna 优化设计 |
| `2026-04-30-robustness-and-cost-design.md` | 稳健性分析与交易成本 |
| `2026-05-02-ml-optuna-integration-design.md` | ML + Optuna 三阶段联合优化 |
| `2026-05-02-qlib-llm-research-and-roadmap.md` | Qlib/RD-Agent 调研与路线图 |
| `2026-05-03-temporal-feature-layer-design.md` | 时序特征层 (TemporalEncoder) 设计 |

## 配置

数据库连接配置 `config/database.ini`：

```ini
[postgresql]
host = localhost
port = 5432
database = your_db
user = your_user
password = your_pass
```

## 常见问题

**股票代码格式**: 上海 `sh` 前缀（如 `sh600519`），深圳 `sz` 前缀（如 `sz000001`）。

**因子方向**: `1` = 越大越好（动量/趋势类），`-1` = 越小越好（波动/估值类）。

**T+1 规则**: 当日买入的股票当日不能卖出，回测引擎已内置此约束。

**交易单位**: A 股以手为单位（100 股），仓位计算自动取整到整手。

**最小交易日要求**: 回测区间至少需要 30 个交易日，否则数据不足以计算滚动指标。
