# 实验追踪系统设计

## 概述

将实验追踪集成到现有 Web 看板中，支持自动记录每次回测/优化的参数与结果，提供可视化对比界面。

## 存储设计

```
output/
└── experiments/
    ├── index.json                  # 所有实验索引 [{id, timestamp, type, metrics, ranker}, ...]
    ├── exp_20260504_001.json       # 实验完整记录
    ├── exp_20260504_002.json
    └── ...
```

`index.json` 为列表，每元素是实验摘要（供列表页快速加载）。单个 `exp_xxx.json` 包含完整信息。

### 实验记录字段

```json
{
  "experiment_id": "exp_20260504_001",
  "timestamp": "2026-05-04T10:30:00",
  "type": "backtest | optimization_single | optimization_wf | sensitivity",
  "ranker": "TemporalMLRanker | MLRanker | SignalRanker",
  "ranker_config": {
    "model": "output/best_model.pkl",
    "encoder": "output/temporal_encoder.pt",
    "factors": "output/selected_factors.json",
    "factor_count": 21
  },
  "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
  "metrics": {
    "sharpe": 1.52,
    "annual_return": 0.18,
    "max_drawdown": 0.12,
    "calmar": 1.5,
    "win_rate": 0.58,
    "ic": 0.048,
    "total_return": 0.22
  },
  "config": {
    "max_positions": 5,
    "max_total_pct": 0.80,
    "stop_loss_mult": 2.0,
    "...": "..."
  }
}
```

## 后端 API

新增 `web/server/api/experiments.py`，挂载到 `/api/experiments`。

| 端点 | 功能 |
|------|------|
| `GET /` | 实验列表 `?ranker=&type=&sort=sharpe&order=desc&limit=20&offset=0` |
| `GET /{id}` | 单个实验完整详情 |
| `POST /compare` | 对比多个实验 `{"ids": ["exp_001", "exp_002"]}` |
| `GET /stats` | 聚合统计 `?group_by=ranker&metric=sharpe` |

纯文件 I/O，不依赖 PostgreSQL。

## 前端页面

在现有 Ant Design 布局中新增"实验追踪"Tab。

### 列表视图

Ant Design Table，默认按时间倒序。所有指标列可点击表头排序。

| 列 | 说明 | 排序 | 宽度 |
|----|------|------|------|
| ☐ | 多选 checkbox | — | 40px |
| ID | 实验编号 (exp_xxx) | — | 120px |
| 时间 | 运行时间戳 | 默认降序 | 140px |
| 类型 | 回测 / 优化 / WF / 敏感性 | — | 80px |
| 模型 | SignalRanker / MLRanker / TemporalMLRanker | — | 140px |
| Sharpe | 年化 Sharpe Ratio | ✓ | 80px |
| 年化收益 | 年化收益率 (百分比) | ✓ | 90px |
| 最大回撤 | Max Drawdown (百分比) | ✓ | 90px |
| Calmar | Calmar Ratio | ✓ | 80px |
| 胜率 | 交易胜率 (百分比) | ✓ | 70px |
| IC | 信息系数 | ✓ | 70px |
| 因字数 | 使用的因子数量 | — | 60px |

筛选栏：
- 模型类型下拉：全部 / SignalRanker / MLRanker / TemporalMLRanker
- 实验类型下拉：全部 / 回测 / 单期优化 / Walk-Forward / 敏感性分析
- 日期范围选择器
- "对比选中 (N)" 按钮，选中 ≥2 个时激活

### 对比视图

选中 2-5 个实验后展开。每个对比类别一个表格，**全部行都展示**：
- 值相同的行：白色/浅灰背景
- 值不同的行：浅黄色高亮背景 (`#fff7e6`)，醒目但不刺眼

**指标对比表**：行=指标，列=实验

```
指标                   exp_001      exp_002      exp_003
─────────────────────────────────────────────────────
Sharpe         ░░░░    1.52         1.31 ░░░░    1.48
年化收益       ░░░░   18.2%        15.8% ░░░░   17.1%
最大回撤                12.3%        12.1%        11.9%
胜率           ░░░░     58%          52% ░░░░     55%
Calmar                  1.50         1.42         1.48
IC                      0.048        0.042        0.046
```

**参数对比表**：行=参数，列=实验

```
参数                   exp_001      exp_002      exp_003
─────────────────────────────────────────────────────
max_positions   ░░░░      5            7 ░░░░      5
max_total_pct   ░░░░    0.80         0.90 ░░░░    0.85
stop_loss_mult            2.0          2.0          2.0  (无高亮)
atr_period                14           14           14   (无高亮)
trailing_pct    ░░░░    0.10         0.15 ░░░░    0.12
... (所有参数，包括因子权重)
```

**配置信息区**（表头上方）：
- 一行标签展示每个实验的模型类型、因子集、日期范围
- 最优值用绿色加粗标注

## 自动记录

新增 `experiments/recorder.py`，提供 `record_experiment()` 函数。在以下位置调用：

- `backtesting/run_daily_rotation.py` — 回测结束时
- `optimization/optuna/run_daily_rotation_optimization.py` — 优化结束时
- `robustness/sensitivity_report.py` — 敏感性分析结束时

## 兼容性

- `output/experiments/` 目录不存在时自动创建
- index.json 文件损坏或不存在时自动重建
- 不改变任何现有 CLI 接口

## 文件变更

```
新增:
  experiments/__init__.py
  experiments/recorder.py                          # 实验记录器
  web/server/api/experiments.py                    # 后端 API
  web/frontend/src/pages/Experiments/index.tsx     # 前端页面

修改:
  web/server/main.py                               # 注册路由
  web/frontend/src/App.tsx                         # 导航 + 路由
  backtesting/run_daily_rotation.py                # 自动记录
  optimization/optuna/run_daily_rotation_optimization.py  # 自动记录
```
