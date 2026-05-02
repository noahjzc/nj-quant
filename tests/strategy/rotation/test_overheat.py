"""测试 OVERHEAT 过热度计算"""
import pytest
from strategy.rotation.daily_rotation_engine import compute_overheat


class TestComputeOverheat:
    def test_no_overheat_when_below_thresholds(self):
        """RSI 和 RET_5 都在阈值以下，返回 0"""
        assert compute_overheat(rsi=70, ret5=0.10) == 0.0
        assert compute_overheat(rsi=80, ret5=0.10) == 0.0  # RSI 超标但涨幅不够
        assert compute_overheat(rsi=70, ret5=0.20) == 0.0  # 涨幅超标但 RSI 不够

    def test_overheat_when_both_above_thresholds(self):
        """双高时返回正数过热度"""
        result = compute_overheat(rsi=80, ret5=0.20)
        assert result > 0.0

    def test_overheat_max_value(self):
        """极端过热接近 1.0"""
        result = compute_overheat(rsi=100, ret5=0.50)
        assert 0.9 <= result <= 1.0

    def test_overheat_zero_at_exact_threshold(self):
        """恰好在阈值时过热度为 0（不触发）"""
        assert compute_overheat(rsi=75, ret5=0.20) == 0.0  # RSI 不满足 >
        assert compute_overheat(rsi=80, ret5=0.15) == 0.0  # RET5 不满足 >

    def test_custom_thresholds(self):
        """自定义阈值生效"""
        result = compute_overheat(rsi=80, ret5=0.10, rsi_threshold=80, ret5_threshold=0.05)
        assert result == 0.0  # RSI 不满足 >

    def test_negative_ret5(self):
        """负涨幅不触发过热"""
        assert compute_overheat(rsi=85, ret5=-0.10) == 0.0
