# 时序特征层 + LightGBM 联合排名设计

## 概述

在现有 LightGBM 排名器之前加入一个轻量 Transformer Encoder，从因子时序（过去 N 天）提取时序动态特征，弥补当前 ML 管线只使用截面因子快照的短板。

## 动机

- **当前问题**: LightGBM 只看 t 时刻因子的截面值，无法感知因子的变化趋势（"波动率在上升" vs "波动率很高"在截面视角下完全一样）
- **数据结构**: 引擎的 `_cache_df` 已缓存最近 30+ 天全市场数据（`trade_date × stock_code` 多索引），包含时序信息，未被利用
- **参考**: Qlib/RD-Agent 论文结论——时序模式 + 因子交互是 alpha 的主要来源

## 核心架构

```
输入: N 只股票 × 20 天 × 30 个筛选后因子
  │
TemporalEncoder (PyTorch, ~2万参数)
  ├─ Linear Projection: 30 → 64
  ├─ Positional Encoding (可学习)
  ├─ TransformerEncoderLayer × 2 (4 heads, d_model=64)
  └─ AdaptiveAvgPool1d → (N, 64维时序特征)
  │
  ├─ 拼接截面因子 (N, 30)
  ├─ LightGBM.predict() → (N,) 预测收益
  │
输出: Top N 股票排序
```

## 关键设计决策

### 模型规模小

2层、4头、64维——参数量约 2 万。金融数据信噪比极低（~1:100），大模型会记住噪声。小模型天然正则化，训练快（epoch < 1秒），推理快（< 10ms per 5000 stocks）。

### 与 MLRanker 接口兼容

`TemporalMLRanker.rank(factor_df, top_n)` 签名与 `MLRanker` 完全一致：
- 内部维护 `deque(maxlen=20)` 滚动缓存每只股票的因子历史
- 每次 `rank()` 调用自动更新缓存并提取时序特征
- 引擎零改动，可通过 `--ml-model` 直接替换

### 两阶段训练

| 阶段 | 目标 | 数据 | 产出 |
|------|------|------|------|
| Phase 1: 预训练 | Encoder 学会"看懂"因子时序模式 | 因子历史 + 自监督（遮罩预测） | `temporal_encoder.pt` |
| Phase 2: 联合 | LightGBM 学会使用时序特征做排名 | 时序特征(Encoder输出) + 截面因子 + 5日收益率 | `best_model.pkl` |

Phase 1 自监督预训练不需要收益标签，数据量大、不过拟合。Phase 2 用预训练 Encoder 提取的特征 + 截面因子训练 LightGBM，不再更新 Encoder 权重。

## 文件规划

```
新增:
  strategy/ml/temporal/
    __init__.py
    encoder.py              # TemporalEncoder (PyTorch nn.Module)
    pretrain.py             # Phase 1: 自监督预训练
    temporal_ranker.py      # TemporalMLRanker (encoder + LightGBM)
    temporal_trainer.py     # Phase 2: 联合训练

修改:
  strategy/ml/trainer.py    # 新增 _build_temporal_dataset() 生成 3D tensor

不变:
  strategy/ml/ml_ranker.py
  strategy/rotation/daily_rotation_engine.py
  strategy/rotation/signal_engine/signal_ranker.py
```

## 数据流

### 训练数据构造

```
MLRankerTrainer.build_dataset() → (X_截面, y)

TemporalTrainer._build_temporal_dataset():
  对于每个训练日 T:
    从 daily parquet 读取 [T-19, T] 共 20 天的因子值
    每只股票构建 (20, M) 的时序矩阵 + (M,) 的截面向量
    标签: T+5 日收益率
  返回: (X_seq, X_cross, y)
    X_seq: (n_samples, 20, M)  float32
    X_cross: (n_samples, M)    float32
    y: (n_samples,)            float32
```

### 推理数据流

```
DailyRotationEngine._rank_candidates()
  └─ ranker.rank(factor_df, top_n)
       │
       ├─ TemporalMLRanker:
       │   1. 更新 _history_cache[stock] ← deque(maxlen=20)
       │   2. 构建 (N, 20, M) tensor
       │   3. encoder.forward() → (N, 64)
       │   4. concat(factor_df, temporal_feats) → X
       │   5. model.predict(X) → predictions
       │   6. argsort(-predictions)[:top_n]
```

## 依赖

- `torch >= 2.0`（新增）
- 其余依赖已有（lightgbm, optuna, pandas, numpy, joblib）

## CLI 串联

```bash
# Phase 1: Encoder 预训练
python -m strategy.ml.temporal.pretrain \
    --start 2020-01-01 --end 2022-12-31 \
    --cache-dir cache/daily_rotation \
    --factors output/selected_factors.json \
    --epochs 50 --output output/

# Phase 2: 联合训练 + Optuna 搜索 ML 超参
python optimization/optuna/run_ml_optimization.py \
    --train-start 2020-01-01 --train-end 2022-12-31 \
    --factors output/selected_factors.json \
    --encoder output/temporal_encoder.pt \
    --trials 50 --output output/

# 回测: 零改动接入
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2024-01-01 --end 2024-12-31 \
    --ml-model auto --trials 100 --output output/
```

## 风险与假设

- **PyTorch 新增依赖**: `pip install torch`，体积较大（~2GB），但量化开发标配
- **Encoder 预训练效果**: 遮罩预测任务是否能学到有意义时序模式？先用简单统计特征（斜率、波动率、拐点）做 baseline 对比
- **20 天窗口假设**: 如果最佳窗口不是 20，后续可加入 Optuna 搜索
- **GPU 需求**: CPU 训练即可（模型仅 2 万参数），有 GPU 更快
