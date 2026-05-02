# ML + Optuna 三阶段联合优化设计

## 概述

将 ML 模型训练纳入 Optuna 优化体系，分三个阶段：因子筛选 → ML 超参优化 → 框架参数优化。每个阶段输出明确，可独立运行或串联执行。

## 现状

- **ML 训练**: `strategy/ml/train_model.py` 手动训练 LightGBM，参数写死在代码里
- **Optuna 优化**: `optimization/optuna/run_daily_rotation_optimization.py` 只优化 14 个框架参数
- **ML 模型参与度低**: 通过 `--ml-model` 加载预训练模型，Optuna 不知道模型的存在
- **158 因子全部使用**: 包含低信噪比、高度共线的因子，拖慢训练和推理

## 目标

构建三阶段优化管线，每阶段可独立执行，也可一键串联：

```
Stage 0: 因子筛选 + 正交化      ~5 分钟
  158 → 两套输出（原始精选 + 正交化）
  ↓
Stage 1: ML 超参优化             ~10 分钟
  Optuna 搜索 LightGBM 超参
  ↓
Stage 2: 框架参数优化             ~80 分钟
  Optuna 搜索交易参数 + 稳健性筛选
```

## Stage 0: 因子筛选 + 正交化

### 职责

从 158 个 Alpha 因子中筛选有效因子，产出两套因子集给不同 Ranker 使用。

### 流程

1. 逐日计算 158 个 Alpha 因子（复用 `Alpha158Calculator`）
2. 每个因子计算截面 Rank IC（与 5 日 forward return 的 Spearman 相关系数）
3. 汇总统计：IC mean、IC std、ICIR、IC>0 比例
4. 过滤：`|IC mean| > 0.015` 且 `ICIR > 0.3`
5. 按 `|IC mean|` 降序排列

### 两套输出

| 输出 | 方法 | 用途 | 因子数 |
|------|------|------|--------|
| 原始精选 | 直接取过滤后 Top 30 | SignalRanker（保持可解释性） | ~30 |
| 正交化 | Gram-Schmidt，从 IC 最高开始，残差 IC 显著则保留 | MLRanker（最大化信号纯度） | ~20-30 |

### 实现

新文件 `strategy/factors/factor_screening.py`，提供：

- `compute_factor_ic(cache_dir, start, end)` → IC DataFrame
- `screen_factors(ic_df)` → `(raw_factors, orthogonal_factors)`
- `save_results(raw, orth, output_dir)` → `selected_factors.json`
- CLI: `python strategy/factors/factor_screening.py --start ... --end ...`

### 输出文件

```
output/
├── selected_factors.json         # {raw: [...], orthogonal: [...]}
└── factor_ic_report.csv          # 每个因子的 IC mean/std/ICIR
```

## Stage 1: ML 超参优化

### 职责

用筛选后的正交化因子训练 LightGBM，Optuna 搜索最优超参。

### 搜索空间（8 个 LightGBM 超参）

| 参数 | 范围 | 类型 | 说明 |
|------|------|------|------|
| num_leaves | [31, 255] | int | 树复杂度 |
| learning_rate | [0.01, 0.20] | float | 学习率 |
| n_estimators | [100, 800] | int | 树数量 |
| min_child_samples | [50, 500] | int | 叶子最小样本 |
| subsample | [0.5, 1.0] | float | 行采样 |
| colsample_bytree | [0.5, 1.0] | float | 列采样 |
| reg_alpha | [1e-4, 1.0] | log float | L1 正则 |
| reg_lambda | [1e-4, 1.0] | log float | L2 正则 |

### 目标函数

每个 Trial：
1. 采样超参 → 训练 LightGBM（用正交化因子）
2. 时间序列分割：前 80% 训练，后 20% 验证（2% purge gap）
3. Early stopping（50 轮无改善则停止）
4. 返回验证集 RMSE（Optuna 最小化）

### 实现

新增 `strategy/ml/ml_optuna.py`（核心逻辑）和 `optimization/optuna/run_ml_optimization.py`（CLI）。

复用 `MLRankerTrainer.build_dataset()`，但不硬编码所有 158 因子——改为支持 `factor_columns` 参数。

### CLI

```bash
python optimization/optuna/run_ml_optimization.py \
    --train-start 2020-01-01 --train-end 2022-12-31 \
    --factors output/selected_factors.json \
    --trials 50 --output output/
```

### 输出

```
output/
├── best_model.pkl                # 最优 LightGBM 模型
├── best_ml_params.json           # 最优超参
└── ml_optuna_trials.csv          # Trial 记录
```

## Stage 2: 框架参数优化

### 职责

用 Stage 0 因子集和 Stage 1 最优模型，Optuna 搜索交易框架参数。

### 改动

相比于现有流程，Stage 2 三处小改动：

1. **`--ml-model auto`**: 自动发现 `output/best_model.pkl`，无需手动指定路径
2. **`--factors` 参数**: 指定因子列表文件，引擎只计算被筛选的因子（而非全部 158）
3. **SignalRanker 权重自适应**: 权重搜索空间维度跟随因子数量，不再硬编码 8 个

### 两条路径

| 路径 | Ranker | 因子来源 | 权重搜索 |
|------|--------|----------|----------|
| A: ML 模式 | MLRanker | 正交化因子 ~25 | 由模型学习，无需搜索 |
| B: 传统模式 | SignalRanker | 原始精选因子 ~30 | Optuna 采样（30 维独立+归一化） |

### 框架参数（不变）

买入信号 × 8、信号模式、仓位上限 × 2、过热度 × 2、ATR 止损/止盈 × 2 + 周期、移动止损 × 2、KDJ 阈值、最大持仓数。共 14 + 因子权重。

### CLI

```bash
# ML 模式
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 \
    --ml-model auto --factors output/selected_factors.json \
    --trials 100 --output output/
```

## 文件变更汇总

```
新增:
  strategy/factors/factor_screening.py          # Stage 0: IC计算 + 正交化
  strategy/ml/ml_optuna.py                       # Stage 1: ML超参搜索核心
  optimization/optuna/run_ml_optimization.py     # Stage 1: CLI

修改:
  strategy/ml/trainer.py           # 支持指定 factor_columns（不写死158）
  optimization/optuna/run_daily_rotation_optimization.py
    - 新增 --ml-model auto 自动发现
    - 新增 --factors 参数
    - SignalRanker 权重维度跟随因子数

不变:
  strategy/ml/ml_ranker.py                      # MLRanker 接口已满足
  strategy/rotation/daily_rotation_engine.py    # 通过 ranker 接口解耦
  strategy/rotation/signal_engine/signal_ranker.py
  strategy/rotation/config.py                   # RotationConfig 结构不变
  strategy/factors/alpha158.py                  # Alpha158 计算不变
```

## 数据隔离

```
训练/因子分析期          回测/优化期
├─ 2020-01-01 ─────────── 2022-12-31 ─── 2024-01-01 ─── 2024-12-31 ──→
│                                                        │
└── Stage 0 (因子筛选) ──┐                               │
└── Stage 1 (ML 训练)  ──┘                               │
                                  └── Stage 2 (框架优化) ──┘
```

无 lookahead 泄露：Stage 0/1 的数据窗口完全在 Stage 2 之前。

## 兼容性

- 不传 `--ml-model` 和 `--factors` 时，行为与现有完全一致
- `selected_factors.json` 不存在时，Stage 1 退化为使用全部 158 因子
- Stage 0/1/2 均可独立运行，不强依赖前面阶段

## 替代方案（已否决）

- **全联合优化**: 每个 Trial 重训模型 + 回测。100 trial × 3min = 5h+，不可行
- **仅去重不筛选**: 保留 100+ 因子，O(n²) 共线性矩阵计算耗时且收益低
