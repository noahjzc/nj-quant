"""时序因子编码器 — 轻量 Transformer Encoder 提取因子时序特征"""
import math
import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码（天数固定，学习最优的位置表示）"""

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

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

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
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(-1)
        return x

    def forward_sequence(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """不池化版本，保留每个时间步的特征，用于序列级预测任务（如遮罩重建）。

        Returns:
            (batch, seq_len, d_model)
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x
