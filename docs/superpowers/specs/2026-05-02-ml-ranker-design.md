# ML 排名器 设计文档

## 目标

用 LightGBM 替代现有的线性加权排名（SignalRanker），让模型自动学习 166 个因子与未来收益之间的非线性关系。

## 架构

```
strategy/ml/
├── __init__.py
├── ml_ranker.py        MLRanker（加载模型 + 推理）
└── trainer.py          Trainer（构造训练数据 + 训练 + 保存）

strategy/rotation/
├── daily_rotation_engine.py  ✏️ 集成 MLRanker
├── run_daily_rotation.py     ✏️ 新增 --ml-model 参数
```

## 接口设计

### MLRanker

```python
class MLRanker:
    """与 SignalRanker 接口完全一致，可直接替换"""

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def rank(self, factor_df: pd.DataFrame, top_n: int) -> list[str]:
        predictions = self.model.predict(factor_df)
        return factor_df.index[np.argsort(-predictions)[:top_n]].tolist()
```

签名 `rank(factor_df, top_n) -> list[str]` 保持不变，引擎 `_execute_buy` 零改动切换。

### Trainer

```python
class MLRankerTrainer:
    """从 Parquet 缓存构造训练数据，训练 LightGBM"""

    def __init__(self, cache_dir: str, model_path: str = None):
        ...

    def build_dataset(self, train_start: str, train_end: str,
                      purge_days: int = 5) -> tuple:
        """构造 X(因子) + y(5日收益)，返回 (X, y)
        
        purge_days: 训练截止日与回测开始日之间的缓冲，防止标签延伸到回测期
        """

    def train(self, train_start: str, train_end: str,
              params: dict = None) -> str:
        """训练并保存模型，返回模型路径"""
```

## 标签设计

```
y = close(t+5) / close(t) - 1   # 未来5日收益率
```

### 为什么选 5 日而非 1 日

- 单日收益信噪比极低（IC ~0.02），模型学不到有效信号
- 5 日信号累积，噪声部分抵消，IC 通常更高
- 与策略平均持仓天数匹配

## 训练流程

```
2020-2023 年 Parquet 缓存
  → 逐日遍历，每天每只股票一条样本
  → X = 166 个因子, y = close(t+5)/close(t) - 1
  → LightGBM 回归训练 → model_20240101.pkl
```

### 防泄漏

```
训练截止日 train_end → 留 5 个交易日 purge → 回测开始日
```

确保标签 close(t+5) 不会延伸到回测期。

### 定期重训

| 回测年 | 训练数据 |
|--------|---------|
| 2024 | 2021-2023 |
| 2025 | 2022-2024 |
| ... | 自动推进 |

## 引擎集成

```python
# _execute_buy 中，rank 调用不变:
self.ranker = SignalRanker(...)      # 旧
self.ranker = MLRanker(model_path)   # 新

ranked = self.ranker.rank(factor_df, top_n=x)  # 不变
```

引擎通过 `--ml-model` 参数选择排名器。

## 实施步骤

| 步骤 | 内容 | 文件 |
|------|------|------|
| 1 | Trainer: 数据构造 + 训练 | `strategy/ml/trainer.py` |
| 2 | MLRanker: 模型加载 + 推理 | `strategy/ml/ml_ranker.py` |
| 3 | 引擎集成 | `daily_rotation_engine.py` |
| 4 | CLI 参数 | `run_daily_rotation.py` |

## Otuna 集成（后续）

| 之前 | 之后 |
|------|------|
| 调因子权重 | — |
| — | 调 LightGBM 超参（learning_rate, num_leaves, n_estimators） |
| 仓位/止损/信号 | 不变 |
