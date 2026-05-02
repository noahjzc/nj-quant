"""Tests for signal detectors: KDJGoldLowSignal, PSYBuySignal, PSYSellSignal."""
import pandas as pd
from strategy.rotation.signal_engine.signal_filter import (
    KDJGoldLowSignal, PSYBuySignal, PSYSellSignal, SignalFilter
)


def _make_kdj_df(k_values, d_values):
    """Build a minimal DataFrame for KDJ signal testing."""
    return pd.DataFrame({'kdj_k': k_values, 'kdj_d': d_values})


def _make_psy_df(psy_val, psyma_val):
    """Build a minimal DataFrame for PSY signal testing."""
    return pd.DataFrame({'psy': [psy_val], 'psyma': [psyma_val]})


class TestKDJGoldLowSignal:
    def test_triggers_on_cross_up_and_low_k(self):
        detector = KDJGoldLowSignal(k_threshold=30.0)
        # k crosses above d: prev k(17) <= d(17), now k(20) > d(18), and k=20 < 30
        df = _make_kdj_df([16, 17, 20], [17, 17, 18])
        result = detector.detect(df, 'sh600001')
        assert result.triggered is True

    def test_no_trigger_when_k_too_high(self):
        detector = KDJGoldLowSignal(k_threshold=30.0)
        # k crosses above d, but k is 35 >= 30
        df = _make_kdj_df([30, 32, 35], [31, 31, 33])
        result = detector.detect(df, 'sh600001')
        assert result.triggered is False

    def test_no_trigger_when_no_cross(self):
        detector = KDJGoldLowSignal(k_threshold=30.0)
        # k is low but no cross (k stays above d)
        df = _make_kdj_df([25, 26, 26], [24, 25, 25])
        result = detector.detect(df, 'sh600001')
        assert result.triggered is False

    def test_respects_threshold_parameter(self):
        detector_low = KDJGoldLowSignal(k_threshold=20.0)
        detector_high = KDJGoldLowSignal(k_threshold=40.0)
        # k at 25, crosses up from 22→25 above d 23→24
        df = _make_kdj_df([22, 22, 25], [21, 23, 24])
        assert detector_low.detect(df, 'sh600001').triggered is False  # 25 >= 20
        assert detector_high.detect(df, 'sh600001').triggered is True   # 25 < 40


class TestPSYBuySignal:
    def test_triggers_when_psy_low_and_above_psyma(self):
        # PSY < 25 (oversold) AND PSY > PSYMA (turning up)
        df = _make_psy_df(20, 18)
        result = PSYBuySignal().detect(df, 'sh600001')
        assert result.triggered is True

    def test_no_trigger_when_psy_not_low_enough(self):
        df = _make_psy_df(30, 28)
        result = PSYBuySignal().detect(df, 'sh600001')
        assert result.triggered is False

    def test_no_trigger_when_psy_below_psyma(self):
        df = _make_psy_df(20, 22)
        result = PSYBuySignal().detect(df, 'sh600001')
        assert result.triggered is False


class TestPSYSellSignal:
    def test_triggers_when_psy_high_and_below_psyma(self):
        # PSY > 75 (overbought) AND PSY < PSYMA (turning down)
        df = _make_psy_df(80, 82)
        result = PSYSellSignal().detect(df, 'sh600001')
        assert result.triggered is True

    def test_no_trigger_when_psy_not_high_enough(self):
        df = _make_psy_df(70, 72)
        result = PSYSellSignal().detect(df, 'sh600001')
        assert result.triggered is False

    def test_no_trigger_when_psy_above_psyma(self):
        df = _make_psy_df(80, 78)
        result = PSYSellSignal().detect(df, 'sh600001')
        assert result.triggered is False


class TestSignalFilterWithNewSignals:
    def test_signal_map_contains_new_detectors(self):
        sf = SignalFilter(['KDJ_GOLD_LOW', 'PSY_BUY'], mode='OR')
        detector_types = [type(d).__name__ for d in sf.detectors]
        assert 'KDJGoldLowSignal' in detector_types
        assert 'PSYBuySignal' in detector_types

    def test_kdj_low_threshold_passed_through(self):
        sf = SignalFilter(['KDJ_GOLD_LOW'], mode='OR', kdj_low_threshold=25.0)
        kdj_detector = next(d for d in sf.detectors if isinstance(d, KDJGoldLowSignal))
        assert kdj_detector.k_threshold == 25.0
