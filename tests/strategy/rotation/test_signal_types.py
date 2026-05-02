"""Tests for SignalType enum values."""
from strategy.rotation.signal_engine.base_signal import SignalType


def test_kdj_gold_low_is_buy():
    assert SignalType.KDJ_GOLD_LOW.is_buy is True
    assert SignalType.KDJ_GOLD_LOW.is_sell is False


def test_psy_buy_is_buy():
    assert SignalType.PSY_BUY.is_buy is True
    assert SignalType.PSY_BUY.is_sell is False


def test_psy_sell_is_sell():
    assert SignalType.PSY_SELL.is_buy is False
    assert SignalType.PSY_SELL.is_sell is True


def test_existing_signals_unchanged():
    assert SignalType.KDJ_GOLD.is_buy is True
    assert SignalType.KDJ_DEATH.is_sell is True
    assert SignalType.BOLL_BREAK.is_buy is True
    assert SignalType.BOLL_BREAK_DOWN.is_sell is True
    assert SignalType.HIGH_BREAK.is_buy is True
