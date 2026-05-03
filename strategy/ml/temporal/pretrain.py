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
        cols = None
        for date_str in dates:
            idx = date_to_idx[date_str]
            if idx < seq_len - 1:
                continue
            window_dates = all_dates[idx - seq_len + 1: idx + 1]

            daily_dfs = {}
            stock_sets = []
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

        # 计算 per-factor 归一化参数
        all_data = np.concatenate([s.reshape(-1, len(cols)) for s in self.samples], axis=0)
        self.factor_mean = np.nanmean(all_data, axis=0).astype(np.float32)
        self.factor_std = np.nanstd(all_data, axis=0).astype(np.float32)
        self.factor_std[self.factor_std < 1e-8] = 1.0  # 避免除零

        logger.info(f"预训练数据集: {len(self.samples)} 条序列, "
                     f"{len(cols)} 个因子, {seq_len} 天窗口")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        # z-score 归一化
        x = (x - torch.tensor(self.factor_mean)) / torch.tensor(self.factor_std)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
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

    # 预测头: 从 d_model 特征空间映射回 M 维因子值
    # 对每个时间步做预测 → 只计算被遮位置的 MSE
    pred_head = nn.Sequential(
        nn.Linear(d_model, d_model * 2),
        nn.GELU(),
        nn.Linear(d_model * 2, M),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(pred_head.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    encoder.train()
    pred_head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for masked_x, target, mask in loader:
            masked_x = masked_x.to(device)        # (B, T, M)
            target_val = target.to(device)         # (B, T, M) — 原始因子值
            mask = mask.to(device)                 # (B, T, M) — True=被遮

            # 编码 → 每个时间步的特征: (B, T, d_model)
            # TransformerEncoder 输出 (B, T, d_model)，不加全局池化
            features_seq = encoder.forward_sequence(masked_x)  # (B, T, d_model)

            # 预测每个时间步的因子值: (B, T, M)
            pred = pred_head(features_seq)

            # 只计算被遮位置的 MSE
            masked_pred = pred[mask]
            masked_target = target_val[mask]

            if masked_pred.numel() == 0:
                continue

            loss = nn.functional.mse_loss(masked_pred, masked_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(pred_head.parameters()), max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            logger.info(f"  Epoch {epoch+1}/{epochs}, loss={avg_loss:.6f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': {
            'n_features': M, 'd_model': d_model, 'n_heads': n_heads,
            'n_layers': n_layers, 'max_len': seq_len,
            'factor_columns': factor_columns,
            'factor_mean': dataset.factor_mean.tolist(),
            'factor_std': dataset.factor_std.tolist(),
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
