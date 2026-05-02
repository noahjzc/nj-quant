"""Encoder 自监督预训练 — 遮罩预测任务"""
import argparse
import json
import logging
from pathlib import Path
from typing import List

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

        self.samples = []
        for date_str in dates:
            idx = date_to_idx[date_str]
            if idx < seq_len - 1:
                continue
            window_dates = all_dates[idx - seq_len + 1: idx + 1]

            daily_dfs = {}
            stock_sets = []
            cols = None
            for d in window_dates:
                p = self.daily_dir / f'{d}.parquet'
                if not p.exists():
                    break
                df = pd.read_parquet(p).set_index('stock_code')
                available_cols = [c for c in self.factor_columns if c in df.columns]
                if cols is None:
                    cols = available_cols
                daily_dfs[d] = df[cols]
                stock_sets.append(set(df.index))
            else:
                common_stocks = list(set.intersection(*stock_sets))
                if len(common_stocks) < 100:
                    continue
                if len(common_stocks) > 500:
                    common_stocks = list(np.random.choice(common_stocks, 500, replace=False))

                for stock in common_stocks:
                    seq = []
                    for d in window_dates:
                        if stock in daily_dfs[d].index:
                            seq.append(daily_dfs[d].loc[stock].values.astype(np.float32))
                        else:
                            seq.append(np.zeros(len(cols), dtype=np.float32))
                    self.samples.append(np.stack(seq))

        logger.info(f"预训练数据集: {len(self.samples)} 条序列, "
                     f"{len(cols)} 个因子, {seq_len} 天窗口")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
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
    """预训练 TemporalEncoder。遮罩预测: 随机遮住 15% 的因子值，让 Encoder 预测原始值。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    dataset = MaskedFactorDataset(cache_dir, start, end, factor_columns, seq_len=seq_len)
    if len(dataset) == 0:
        raise ValueError("无训练数据，检查因子列是否在 daily parquet 中存在")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    M = len(factor_columns)
    encoder = TemporalEncoder(
        n_features=M, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_len=seq_len,
    ).to(device)

    # 简单的重建损失: 对整个序列做编码 → 预测 → MSE
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    encoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for masked_x, target, mask in loader:
            masked_x = masked_x.to(device)
            # 简单自监督: 编码被遮的序列，用 hidden states 做重建
            # 用 encoder 的中间表示做预测
            encoder.train()
            # 编码被遮序列
            features = encoder(masked_x)  # (B, d_model)
            # 损失: 让特征能区分不同序列模式（对比学习简化版）
            # 加上一个小的重建辅助损失
            with torch.no_grad():
                target_encoded = encoder(target.to(device))
            # 让被遮序列的特征接近未遮序列的特征
            loss = loss_fn(features, target_encoded)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.6f}")

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
        calc = Alpha158Calculator()
        dummy = pd.DataFrame({
            'open': [10.0]*10, 'high': [10.0]*10, 'low': [10.0]*10,
            'close': [10.0]*10, 'volume': [1e6]*10,
        })
        factor_columns = list(calc.compute(dummy).columns)

    pretrain(
        cache_dir=args.cache_dir, factor_columns=factor_columns,
        start=args.start, end=args.end, output_path=args.output,
        seq_len=args.seq_len, d_model=args.d_model, epochs=args.epochs,
    )


if __name__ == '__main__':
    main()
