"""Tests for RotationConfig default values with new signals/factors."""
from strategy.rotation.config import RotationConfig


def test_new_signal_types_in_defaults():
    config = RotationConfig()
    assert "KDJ_GOLD_LOW" in config.buy_signal_types
    assert "PSY_SELL" in config.sell_signal_types


def test_dmi_not_in_defaults():
    config = RotationConfig()
    assert "DMI_GOLD" not in config.buy_signal_types
    assert "DMI_DEATH" not in config.sell_signal_types


def test_new_factors_in_defaults():
    config = RotationConfig()
    assert "circulating_mv" in config.rank_factor_weights
    assert "WR_10" in config.rank_factor_weights
    assert "WR_14" in config.rank_factor_weights
    assert config.rank_factor_directions["circulating_mv"] == -1
    assert config.rank_factor_directions["WR_10"] == -1
    assert config.rank_factor_directions["WR_14"] == -1


def test_kdj_low_threshold_default():
    config = RotationConfig()
    assert config.kdj_low_threshold == 30.0


def test_existing_factors_unchanged():
    config = RotationConfig()
    for factor in ['RSI_1', 'RET_20', 'VOLUME_RATIO', 'PB', 'PE_TTM', 'OVERHEAT']:
        assert factor in config.rank_factor_weights
        assert factor in config.rank_factor_directions
