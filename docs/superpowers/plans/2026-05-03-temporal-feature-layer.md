# 时序特征层 + LightGBM 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建轻量 Transformer Encoder 提取因子时序特征，增强 LightGBM 排名器，弥补当前截面因子快照的短板。

**Architecture:** TemporalEncoder(2层Transformer, ~2万参数) 预训练后与 LightGBM 组成 TemporalMLRanker，rank() 接口与 MLRanker 一致，引擎零改动。

**Tech Stack:** PyTorch 2.x, LightGBM, Optuna, pandas, numpy

---

## 文件结构

```
新增:
  strategy/ml/temporal/__init__.py
  strategy/ml/temporal/encoder.py          # TemporalEncoder (nn.Module)
  strategy/ml/temporal/pretrain.py         # 自监督预训练
  strategy/ml/temporal/temporal_ranker.py  # TemporalMLRanker
  strategy/ml/temporal/temporal_trainer.py # Phase 2 联合训练

修改:
  optimization/optuna/run_ml_optimization.py   # 新增 --encoder
  optimization/optuna/run_daily_rotation_optimization.py  # 支持 TemporalMLRanker

不变:
  strategy/ml/ml_ranker.py
  strategy/ml/trainer.py
  strategy/rotation/daily_rotation_engine.py
  strategy/rotation/signal_engine/signal_ranker.py
```

---

### Task 0: 安装 PyTorch

- [ ] **Step 0.1: Install PyTorch (CPU 版本)**

```bash
cd D:/workspace/code/mine/quant/nj-quant
.venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- [ ] **Step 0.2: 验证**

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# Expected: PyTorch 2.x, CUDA: False (or True if GPU)
```

---

### Task 1: TemporalEncoder (PyTorch Module)

**Files:**
- Create: `strategy/ml/temporal/__init__.py`
- Create: `strategy/ml/temporal/encoder.py`
- Test: `tests/strategy/ml/temporal/test_encoder.py`

- [ ] **Step 1.1: 写测试**

```python
# tests/strategy/ml/temporal/test_encoder.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from strategy.ml.temporal.encoder import TemporalEncoder, LearnablePositionalEncoding


def test_encoder_output_shape():
    """Encoder 输出形状正确: (batch, seq_len, n_features) → (batch, d_model)"""
    encoder = TemporalEncoder(n_features=30, d_model=64, n_heads=4, n_layers=2, max_len=20)
    x = torch.randn(16, 20, 30)  # 16 stocks, 20 days, 30 factors
    out = encoder(x)
    assert out.shape == (16, 64), f"Expected (16, 64), got {out.shape}"


def test_encoder_masked_forward():
    """Encoder 支持 src_key_padding_mask 处理 pad 或 mask"""
    encoder = TemporalEncoder(n_features=30, d_model=64, n_heads=4, n_layers=2, max_len=20)
    x = torch.randn(16, 20, 30)
    mask = torch.zeros(16, 20, dtype=torch.bool)
    mask[:, 15:] = True  # 最后5天被遮住
    out = encoder(x, src_key_padding_mask=mask)
    assert out.shape == (16, 64)


def test_encoder_deterministic():
    """eval 模式下输出可复现"""
    encoder = TemporalEncoder(n_features=30, d_model=32, n_heads=2, n_layers=1, max_len=20)
    encoder.eval()
    x = torch.randn(8, 20, 30)
    with torch.no_grad():
        out1 = encoder(x)
        out2 = encoder(x)
    assert torch.allclose(out1, out2)
```

- [ ] **Step 2.1: 运行测试确认失败**

```bash
cd D:/workspace/code/mine/quant/nj-quant
pytest tests/strategy/ml/temporal/test_encoder.py -v
# Expected: 3 FAIL (module not found)
```

- [ ] **Step 3.1: 实现 TemporalEncoder**

```python
# strategy/ml/temporal/__init__.py
from strategy.ml.temporal.encoder import TemporalEncoder
```

```python
# strategy/ml/temporal/encoder.py
"""时序因子编码器 — 轻量 Transformer Encoder 提取因子时序特征"""
import math
import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码（天数固定为 20，学习最优的位置表示）"""

    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class TemporalEncoder(nn.Module):
    """从因子时序中提取定长时序特征向量。

    Args:
        n_features: 因子数量 (M)
        d_model: Transformer 隐藏维度
        n_heads: 多头注意力头数
        n_layers: Encoder 层数
        max_len: 最大序列长度
        dropout: Dropout 比例
    """

    def __init__(
        self,
        n_features: int = 30,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # 因子投影: M → d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # 位置编码
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN: 训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 时序池化: 自适应平均 → (batch, d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) — 因子时序
            src_key_padding_mask: (batch, seq_len) — True=忽略该位置

        Returns:
            (batch, d_model) — 时序特征向量
        """
        # 投影: (B, T, M) → (B, T, d_model)
        x = self.input_proj(x)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 池化: (B, T, d_model) → (B, d_model, T) → (B, d_model, 1) → (B, d_model)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)

        return x
```

- [ ] **Step 4.1: 运行测试确认通过**

```bash
pytest tests/strategy/ml/temporal/test_encoder.py -v
# Expected: 3 PASS
```

- [ ] **Step 5.1: Commit**

```bash
git add strategy/ml/temporal/__init__.py strategy/ml/temporal/encoder.py tests/strategy/ml/temporal/test_encoder.py
git commit -m "feat(ml): add lightweight Transformer Encoder for factor temporal feature extraction"
```

---

### Task 2: Encoder 预训练 (Self-Supervised)

**Files:**
- Create: `strategy/ml/temporal/pretrain.py`

- [ ] **Step 1.2: 实现自监督预训练**

```python
# strategy/ml/temporal/pretrain.py
"""Encoder 自监督预训练 — 遮罩预测任务"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from strategy.ml.temporal.encoder import TemporalEncoder

logger = logging.getLogger(__name__)


class MaskedFactorDataset(Dataset):
    """自监督数据集: 随机遮罩因子值，让模型预测被遮住的值"""

    def __init__(self, cache_dir: str, start: str, end: str,
                 factor_columns: List[str], seq_len: int = 20,
                 mask_ratio: float = 0.15):
        self.daily_dir = Path(cache_dir) / 'daily'
        self.factor_columns = factor_columns
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio

        all_dates = sorted([f.stem for f in self.daily_dir.glob('*.parquet')])
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        dates = [d for d in all_dates if start <= d <= end]

        # 收集所有序列: 对于每个日期 T，取 [T-seq_len+1, T] 的因子值
        self.samples = []
        for date_str in dates:
            idx = date_to_idx[date_str]
            if idx < seq_len - 1:
                continue
            window_dates = all_dates[idx - seq_len + 1: idx + 1]

            # 读取窗口内所有日期的数据
            daily_dfs = {}
            stock_sets = []
            for d in window_dates:
                p = self.daily_dir / f'{d}.parquet'
                if not p.exists():
                    break
                df = pd.read_parquet(p).set_index('stock_code')
                cols = [c for c in self.factor_columns if c in df.columns]
                if cols:
                    daily_dfs[d] = df[cols]
                    stock_sets.append(set(df.index))
            else:
                # 取所有日期都有的股票
                common_stocks = list(set.intersection(*stock_sets))
                if len(common_stocks) < 100:
                    continue
                # 随机采样最多 500 只
                if len(common_stocks) > 500:
                    common_stocks = list(np.random.choice(common_stocks, 500, replace=False))

                for stock in common_stocks:
                    seq = []
                    for d in window_dates:
                        if stock in daily_dfs[d].index:
                            seq.append(daily_dfs[d].loc[stock].values.astype(np.float32))
                        else:
                            seq.append(np.zeros(len(cols), dtype=np.float32))
                    self.samples.append(np.stack(seq))  # (seq_len, M)

        logger.info(f"预训练数据集: {len(self.samples)} 条序列, "
                     f"{len(cols)} 个因子, {seq_len} 天窗口")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # (seq_len, M)
        # 随机遮罩
        mask = torch.rand(x.shape) < self.mask_ratio
        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x, x, mask


def pretrain(
    cache_dir: str,
    factor_columns: List[str],
    start: str,
    end: str,
    output_path: str,
    seq_len: int = 20,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """预训练 TemporalEncoder。

    遮罩预测: 随机遮住 15% 的因子值，让 Encoder 预测原始值。
    Encoder 学会从周围未遮罩的时序信息推断被遮因子——等价于学习因子的时序依赖关系。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    dataset = MaskedFactorDataset(
        cache_dir, start, end, factor_columns, seq_len=seq_len
    )
    if len(dataset) == 0:
        raise ValueError("无训练数据，检查因子列是否在 daily parquet 中存在")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    M = len(factor_columns)
    encoder = TemporalEncoder(
        n_features=M, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_len=seq_len,
    ).to(device)

    # 预测头: 从 d_model 还原到 M 维因子值
    pred_head = nn.Linear(d_model, M).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(pred_head.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    encoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for masked_x, target, mask in loader:
            masked_x, target, mask = masked_x.to(device), target.to(device), mask.to(device)
            # 编码
            features = encoder(masked_x)  # (B, d_model)
            # 预测被遮住的值: broadcast features 到每个时间步
            pred = pred_head(features)  # (B, M)
            # 只计算被遮位置的 loss
            mask_flat = mask.any(dim=1)  # (B, M) → 是否有任何位置被遮
            if mask_flat.sum() == 0:
                continue
            # 简单方案: 对所有位置计算 loss，被遮位置的权重更大
            loss = loss_fn(pred * mask_flat.float().unsqueeze(1),
                          target.mean(dim=1) * mask_flat.float().unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.6f}")

    # 保存
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': {
            'n_features': M, 'd_model': d_model, 'n_heads': n_heads,
            'n_layers': n_layers, 'max_len': seq_len,
            'factor_columns': factor_columns,
        },
    }, output_path)
    logger.info(f"预训练完成，模型已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='TemporalEncoder 自监督预训练')
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--cache-dir', default='cache/daily_rotation')
    parser.add_argument('--factors', default=None, help='selected_factors.json')
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output', default='output/temporal_encoder.pt')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    if args.factors and Path(args.factors).exists():
        with open(args.factors) as f:
            data = json.load(f)
        factor_columns = data.get('orthogonal', data.get('raw', []))
    else:
        from strategy.factors.alpha158 import Alpha158Calculator
        import pandas as pd
        calc = Alpha158Calculator()
        dummy = pd.DataFrame({
            'open': [10.0]*10, 'high': [10.0]*10, 'low': [10.0]*10,
            'close': [10.0]*10, 'volume': [1e6]*10,
        })
        factor_columns = list(calc.compute(dummy).columns)

    pretrain(
        cache_dir=args.cache_dir,
        factor_columns=factor_columns,
        start=args.start, end=args.end,
        output_path=args.output,
        seq_len=args.seq_len, d_model=args.d_model,
        epochs=args.epochs,
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 2.2: 验证可导入**

```bash
cd D:/workspace/code/mine/quant/nj-quant
python -c "
from strategy.ml.temporal.pretrain import MaskedFactorDataset, pretrain
print('OK: pretrain module imports successfully')
"
```

- [ ] **Step 3.2: Commit**

```bash
git add strategy/ml/temporal/pretrain.py
git commit -m "feat(ml): add self-supervised pretraining for TemporalEncoder (masked factor prediction)"
```

---

### Task 3: TemporalMLRanker

**Files:**
- Create: `strategy/ml/temporal/temporal_ranker.py`
- Test: `tests/strategy/ml/temporal/test_temporal_ranker.py`

- [ ] **Step 1.3: 写测试**

```python
# tests/strategy/ml/temporal/test_temporal_ranker.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import torch
import joblib
import lightgbm as lgb
import tempfile
import os
from strategy.ml.temporal.temporal_ranker import TemporalMLRanker
from strategy.ml.temporal.encoder import TemporalEncoder


def test_temporal_ranker_rank():
    """基本 rank 测试: 接口与 MLRanker 一致"""
    # 创建临时 encoder + model
    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存一个 dummy encoder
        encoder = TemporalEncoder(n_features=3, d_model=16, n_heads=2, n_layers=1)
        encoder_path = os.path.join(tmpdir, 'encoder.pt')
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': {'n_features': 3, 'd_model': 16, 'n_heads': 2,
                       'n_layers': 1, 'max_len': 20, 'factor_columns': ['a', 'b', 'c']},
        }, encoder_path)

        # 保存一个 dummy lightgbm
        X = np.random.randn(100, 3 + 16)  # 3 factors + 16 temporal
        y = np.random.randn(100)
        model = lgb.LGBMRegressor(n_estimators=10, verbosity=-1)
        model.fit(X, y)
        model_path = os.path.join(tmpdir, 'model.pkl')
        joblib.dump(model, model_path)

        # 创建 ranker
        ranker = TemporalMLRanker(model_path, encoder_path)

        # 多次调用 rank 填充历史缓存
        stocks = [f'sh00000{i}' for i in range(10)]
        for _ in range(25):
            factor_df = pd.DataFrame(
                np.random.randn(10, 3), index=stocks, columns=['a', 'b', 'c']
            )
            result = ranker.rank(factor_df, top_n=5)

        assert len(result) == 5
        assert all(s in stocks for s in result)


def test_temporal_ranker_history_warmup():
    """历史不足 20 天时零填充，不崩溃"""
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = TemporalEncoder(n_features=2, d_model=8, n_heads=2, n_layers=1)
        encoder_path = os.path.join(tmpdir, 'encoder.pt')
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': {'n_features': 2, 'd_model': 8, 'n_heads': 2,
                       'n_layers': 1, 'max_len': 20, 'factor_columns': ['x', 'y']},
        }, encoder_path)

        X = np.random.randn(50, 2 + 8)
        y = np.random.randn(50)
        model = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
        model.fit(X, y)
        model_path = os.path.join(tmpdir, 'model.pkl')
        joblib.dump(model, model_path)

        ranker = TemporalMLRanker(model_path, encoder_path)

        # 只有 5 天历史
        stocks = [f'sh00000{i}' for i in range(5)]
        for _ in range(5):
            factor_df = pd.DataFrame(
                np.random.randn(5, 2), index=stocks, columns=['x', 'y']
            )
            result = ranker.rank(factor_df, top_n=3)
        assert len(result) == 3
```

- [ ] **Step 2.3: 运行测试确认失败**

```bash
pytest tests/strategy/ml/temporal/test_temporal_ranker.py -v
# Expected: 2 FAIL
```

- [ ] **Step 3.3: 实现 TemporalMLRanker**

```python
# strategy/ml/temporal/temporal_ranker.py
"""TemporalMLRanker — Encoder + LightGBM 联合推理"""
import logging
from collections import deque
from typing import List

import numpy as np
import pandas as pd
import torch
import joblib

from strategy.ml.temporal.encoder import TemporalEncoder

logger = logging.getLogger(__name__)


class TemporalMLRanker:
    """带时序编码的 ML 排名器。

    接口与 MLRanker 完全一致: rank(factor_df, top_n) → List[str]。
    内部维护每只股票的因子历史缓存 (deque, maxlen=20)。

    Args:
        model_path: LightGBM 模型路径 (.pkl)
        encoder_path: 预训练 TemporalEncoder 路径 (.pt)
        seq_len: 时序窗口长度（需与预训练一致）
    """

    def __init__(self, model_path: str, encoder_path: str, seq_len: int = 20):
        self.model = joblib.load(model_path)

        checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        self.encoder = TemporalEncoder(
            n_features=cfg['n_features'],
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            max_len=cfg.get('max_len', seq_len),
        )
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.encoder.eval()

        self._feature_names = cfg['factor_columns']
        self._seq_len = seq_len
        self._n_features = len(self._feature_names)
        self._d_model = cfg['d_model']

        # 因子值历史缓存: {stock_code: deque(maxlen=seq_len)}
        self._history: dict = {}

    @property
    def required_features(self) -> List[str]:
        return self._feature_names

    def rank(self, factor_df: pd.DataFrame, top_n: int = 5) -> List[str]:
        if factor_df.empty:
            return []

        stocks = factor_df.index.tolist()
        M = self._n_features

        # 1. 更新历史缓存
        for stock in stocks:
            if stock not in self._history:
                # 新股票: 零填充到 seq_len
                self._history[stock] = deque(maxlen=self._seq_len)
                for _ in range(self._seq_len):
                    self._history[stock].append(np.zeros(M, dtype=np.float32))
            row = factor_df.loc[stock]
            values = np.array([float(row.get(f, 0.0)) for f in self._feature_names],
                              dtype=np.float32)
            self._history[stock].append(values)

        # 2. 构建时序 tensor: (N, seq_len, M)
        history_array = np.array([list(self._history[s]) for s in stocks], dtype=np.float32)
        history_tensor = torch.tensor(history_array)

        # 3. Encoder 提取时序特征: (N, d_model)
        with torch.no_grad():
            temporal_feats = self.encoder(history_tensor).numpy()

        # 4. 构建截面因子矩阵: (N, M)
        cross_features = np.array([
            [float(factor_df.loc[s].get(f, 0.0)) for f in self._feature_names]
            for s in stocks
        ], dtype=np.float32)

        # 5. 拼接 → LightGBM 预测
        X = np.concatenate([cross_features, temporal_feats], axis=1)
        predictions = self.model.predict(X)

        # 6. 排序返回
        order = np.argsort(-predictions)
        return [stocks[i] for i in order[:top_n]]
```

- [ ] **Step 4.3: 运行测试确认通过**

```bash
pytest tests/strategy/ml/temporal/test_temporal_ranker.py -v
# Expected: 2 PASS
```

- [ ] **Step 5.3: Commit**

```bash
git add strategy/ml/temporal/temporal_ranker.py tests/strategy/ml/temporal/test_temporal_ranker.py
git commit -m "feat(ml): add TemporalMLRanker combining Transformer Encoder with LightGBM"
```

---

### Task 4: TemporalTrainer (Phase 2 联合训练)

**Files:**
- Create: `strategy/ml/temporal/temporal_trainer.py`

- [ ] **Step 1.4: 实现 Phase 2 联合训练**

```python
# strategy/ml/temporal/temporal_trainer.py
"""Phase 2: 用预训练 Encoder 提取特征 → LightGBM 训练"""
import json
import logging
from pathlib import Path
from typing import List, Optional
from collections import deque

import numpy as np
import pandas as pd
import torch
import joblib
import lightgbm as lgb

from strategy.ml.temporal.encoder import TemporalEncoder

logger = logging.getLogger(__name__)


class TemporalTrainer:
    """用预训练 Encoder 从训练数据提取时序特征，训练 LightGBM。

    流程:
    1. 加载预训练 Encoder
    2. 遍历训练日期，构建因子时序 → Encoder 提取特征 → 缓存
    3. 拼接截面因子 + 时序特征 → 训练 LightGBM
    """

    def __init__(self, cache_dir: str, encoder_path: str):
        self.cache_dir = Path(cache_dir)
        self.daily_dir = self.cache_dir / 'daily'

        ckpt = torch.load(encoder_path, map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        self.factor_columns = cfg['factor_columns']
        self.seq_len = cfg.get('max_len', 20)
        self.d_model = cfg['d_model']

        self.encoder = TemporalEncoder(
            n_features=cfg['n_features'], d_model=cfg['d_model'],
            n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            max_len=self.seq_len,
        )
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.encoder.eval()

    def extract_features(self, start: str, end: str) -> np.ndarray:
        """用 Encoder 对所有训练日期提取时序特征，返回 numpy 数组缓存。

        Returns:
            (n_dates, n_stocks, d_model) — 可保存为 .npy
        """
        all_dates = sorted([f.stem for f in self.daily_dir.glob('*.parquet')])
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        dates = [d for d in all_dates if start <= d <= end]

        features_by_date = {}
        history: dict = {}  # stock → deque

        for i, date_str in enumerate(dates):
            idx = date_to_idx[date_str]
            df = pd.read_parquet(self.daily_dir / f'{date_str}.parquet')
            df = df.set_index('stock_code')

            stocks = df.index.tolist()
            cols = [c for c in self.factor_columns if c in df.columns]
            if len(cols) < len(self.factor_columns) * 0.5:
                continue

            # 更新历史
            M = len(self.factor_columns)
            for s in stocks:
                if s not in history:
                    history[s] = deque(maxlen=self.seq_len)
                values = np.array([float(df.loc[s].get(c, 0.0)) for c in self.factor_columns],
                                  dtype=np.float32)
                history[s].append(values)

            # 构建 tensor
            stock_list = list(stocks)
            hist_array = np.zeros((len(stock_list), self.seq_len, M), dtype=np.float32)
            for j, s in enumerate(stock_list):
                h = list(history[s])
                for t_idx in range(min(len(h), self.seq_len)):
                    hist_array[j, self.seq_len - len(h) + t_idx] = h[t_idx]

            with torch.no_grad():
                feats = self.encoder(torch.tensor(hist_array)).numpy()

            features_by_date[date_str] = feats
            stock_list_snapshot = stock_list

            if (i + 1) % 50 == 0:
                logger.info(f"  特征提取: {i+1}/{len(dates)}")

            # 存储每日期望的 stock 列表以备后用
            setattr(self, f'_stocks_{date_str}', stock_list)

        return features_by_date

    def train(
        self,
        train_start: str,
        train_end: str,
        output_path: str,
        params: dict = None,
        purge_days: int = 5,
    ) -> str:
        """联合训练: 提取时序特征 → 拼截面因子 → 训练 LightGBM。

        Returns:
            模型保存路径
        """
        all_dates = sorted([f.stem for f in self.daily_dir.glob('*.parquet')])
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        dates = [d for d in all_dates if train_start <= d <= train_end]

        # purge
        if purge_days > 0:
            end_idx = date_to_idx[dates[-1]]
            cutoff_idx = max(0, end_idx - purge_days)
            dates = [d for d in dates if d <= all_dates[cutoff_idx]]

        logger.info(f"训练日期: {dates[0]} ~ {dates[-1]}, 共 {len(dates)} 天")

        # 逐日提取特征并构建训练集
        history: dict = {}
        M = len(self.factor_columns)
        X_chunks = []
        y_chunks = []

        for i, date_str in enumerate(dates):
            idx = date_to_idx[date_str]
            if idx < self.seq_len - 1:
                continue
            target_idx = idx + 5
            if target_idx >= len(all_dates):
                continue

            df = pd.read_parquet(self.daily_dir / f'{date_str}.parquet').set_index('stock_code')
            target_df = pd.read_parquet(
                self.daily_dir / f'{all_dates[target_idx]}.parquet'
            ).set_index('stock_code')

            cols = [c for c in self.factor_columns if c in df.columns]
            common = df.index.intersection(target_df.index)
            if len(common) < 100:
                continue

            # 更新历史
            for s in common:
                if s not in history:
                    history[s] = deque(maxlen=self.seq_len)
                history[s].append(np.array(
                    [float(df.loc[s].get(c, 0.0)) for c in self.factor_columns],
                    dtype=np.float32
                ))

            # Encoder 提取时序特征
            stock_list = list(common)
            hist_array = np.zeros((len(stock_list), self.seq_len, M), dtype=np.float32)
            for j, s in enumerate(stock_list):
                h = list(history[s])
                for t_idx in range(min(len(h), self.seq_len)):
                    hist_array[j, self.seq_len - len(h) + t_idx] = h[t_idx]

            with torch.no_grad():
                temporal_feats = self.encoder(torch.tensor(hist_array)).numpy()

            # 截面因子
            cross_feats = np.array([
                [float(df.loc[s].get(c, 0.0)) for c in self.factor_columns]
                for s in stock_list
            ], dtype=np.float32)

            # 拼接
            X = np.concatenate([cross_feats, temporal_feats], axis=1)

            # 标签
            y = np.array([
                (target_df.loc[s, 'close'] - df.loc[s, 'close']) / df.loc[s, 'close']
                for s in stock_list
            ], dtype=np.float32)

            mask = (y > -0.5) & (y < 0.5)
            X_chunks.append(X[mask])
            y_chunks.append(y[mask])

            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i+1}/{len(dates)}, 样本: {sum(len(c) for c in X_chunks)}")

        X_all = np.vstack(X_chunks)
        y_all = np.concatenate(y_chunks)
        logger.info(f"训练集: X={X_all.shape}, y={len(y_all)}")

        # 训练 LightGBM
        default_params = {
            'objective': 'regression', 'metric': 'rmse',
            'boosting_type': 'gbdt', 'num_leaves': 63,
            'learning_rate': 0.05, 'n_estimators': 500,
            'early_stopping_rounds': 50, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1,
            'reg_lambda': 0.1, 'min_child_samples': 100,
            'verbosity': -1, 'random_state': 42,
        }
        if params:
            default_params.update(params)

        # 时间序列分割
        split_idx = int(len(X_all) * 0.8)
        purge_size = max(1, int(len(X_all) * 0.02))
        val_start = min(split_idx + purge_size, len(X_all) - 1)

        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_all[:split_idx], y_all[:split_idx],
            eval_set=[(X_all[val_start:], y_all[val_start:])],
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        model_path = str(output / 'best_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"模型已保存: {model_path}")

        return model_path
```

- [ ] **Step 2.4: 验证可导入**

```bash
cd D:/workspace/code/mine/quant/nj-quant
python -c "
from strategy.ml.temporal.temporal_trainer import TemporalTrainer
print('OK: TemporalTrainer imports')
"
```

- [ ] **Step 3.4: Commit**

```bash
git add strategy/ml/temporal/temporal_trainer.py
git commit -m "feat(ml): add TemporalTrainer for Phase 2 joint feature extraction and LightGBM training"
```

---

### Task 5: 修改 run_ml_optimization.py 支持 --encoder

**Files:**
- Modify: `optimization/optuna/run_ml_optimization.py`

- [ ] **Step 1.5: 添加 --encoder 参数和集成逻辑**

在 `main()` 中添加:
```python
parser.add_argument('--encoder', default=None,
                    help='预训练 TemporalEncoder 路径 (.pt)，用于时序特征增强')
```

在调用 `run_ml_optimization` 之前，如果指定了 `--encoder`，则使用 `TemporalTrainer` 替代默认的 `MLRankerTrainer`：

```python
if args.encoder and Path(args.encoder).exists():
    # 时序模式: Encoder 提取特征 + LightGBM 训练
    from strategy.ml.temporal.temporal_trainer import TemporalTrainer
    trainer = TemporalTrainer(args.cache_dir, args.encoder)
    model_path = trainer.train(
        train_start=args.train_start,
        train_end=args.train_end,
        output_path=args.output,
    )
    print(f"\n时序增强模型已保存: {model_path}")
else:
    # 标准模式: 纯 LightGBM Optuna 优化（现有逻辑）
    run_ml_optimization(...)
```

- [ ] **Step 2.5: 验证**

```bash
cd D:/workspace/code/mine/quant/nj-quant
python optimization/optuna/run_ml_optimization.py --help
# Expected: --encoder 参数出现在帮助中
```

- [ ] **Step 3.5: Commit**

```bash
git add optimization/optuna/run_ml_optimization.py
git commit -m "feat(optuna): add --encoder flag for temporal feature-enhanced ML training"
```

---

### Task 6: 修改 run_daily_rotation_optimization.py 支持 TemporalMLRanker

**Files:**
- Modify: `optimization/optuna/run_daily_rotation_optimization.py` (CLI 入口)

- [ ] **Step 1.6: --ml-model auto 自动发现 temporal 模型**

在 `--ml-model auto` 逻辑中，除了检查 `best_model.pkl`，也检查 `temporal_encoder.pt`：

```python
if args.ml_model == 'auto':
    auto_model = Path(args.output) / 'best_model.pkl'
    auto_encoder = Path(args.output) / 'temporal_encoder.pt'
    if auto_model.exists() and auto_encoder.exists():
        # 时序模式
        from strategy.ml.temporal.temporal_ranker import TemporalMLRanker
        ranker = TemporalMLRanker(str(auto_model), str(auto_encoder))
        print(f"时序增强排名器已加载: encoder={auto_encoder}, model={auto_model}")
    elif auto_model.exists():
        # 纯 LightGBM 模式
        ranker = MLRanker(str(auto_model))
        print(f"ML 排名器已加载: {auto_model}")
    else:
        print(f"警告: 未找到模型，回退到 SignalRanker")
```

- [ ] **Step 2.6: 验证**

```bash
cd D:/workspace/code/mine/quant/nj-quant
python optimization/optuna/run_daily_rotation_optimization.py --help
# Expected: --ml-model 说明包含 auto 自动发现逻辑
```

- [ ] **Step 3.6: Commit**

```bash
git add optimization/optuna/run_daily_rotation_optimization.py
git commit -m "feat(optuna): auto-detect TemporalMLRanker when encoder+model both exist"
```

---

### Task 7: 端到端验证

- [ ] **Step 1.7: Encoder 快速预训练验证（少量 epoch）**

```bash
cd D:/workspace/code/mine/quant/nj-quant

# 用筛选后的因子快速预训练
python -m strategy.ml.temporal.pretrain \
    --start 2020-01-01 --end 2020-06-30 \
    --cache-dir cache/daily_rotation \
    --factors output/selected_factors.json \
    --epochs 5 --output output/test_encoder.pt
# Expected: 预训练完成，生成 test_encoder.pt
```

- [ ] **Step 2.7: Phase 2 联合训练验证**

```bash
python optimization/optuna/run_ml_optimization.py \
    --train-start 2020-01-01 --train-end 2020-06-30 \
    --cache-dir cache/daily_rotation \
    --factors output/selected_factors.json \
    --encoder output/test_encoder.pt \
    --output output/
# Expected: 用 Encoder 提取特征并训练 LightGBM
```

- [ ] **Step 3.7: 回测验证（少量 trial）**

```bash
python optimization/optuna/run_daily_rotation_optimization.py \
    --mode single --start 2023-01-01 --end 2023-06-30 \
    --ml-model auto --trials 3 --skip-robustness --output output/
# Expected: 自动加载 TemporalMLRanker，回测正常
```

- [ ] **Step 4.7: 现有测试无回归**

```bash
pytest tests/robustness/test_sensitivity.py tests/strategy/factors/test_factor_screening.py tests/strategy/ml/temporal/ -v
# Expected: 全部 PASS
```

- [ ] **Step 5.7: Commit**

```bash
git add -A
git commit -m "feat: complete temporal feature layer integration with Transformer Encoder + LightGBM"
```
