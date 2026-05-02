import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from strategy.ml.temporal.encoder import TemporalEncoder, LearnablePositionalEncoding


def test_encoder_output_shape():
    """Encoder 输出形状: (batch, seq_len, n_features) → (batch, d_model)"""
    encoder = TemporalEncoder(n_features=30, d_model=64, n_heads=4, n_layers=2, max_len=20)
    x = torch.randn(16, 20, 30)
    out = encoder(x)
    assert out.shape == (16, 64), f"Expected (16, 64), got {out.shape}"


def test_encoder_masked_forward():
    """Encoder 支持 src_key_padding_mask"""
    encoder = TemporalEncoder(n_features=30, d_model=64, n_heads=4, n_layers=2, max_len=20)
    x = torch.randn(16, 20, 30)
    mask = torch.zeros(16, 20, dtype=torch.bool)
    mask[:, 15:] = True
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
