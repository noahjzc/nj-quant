import pytest
from datetime import date

from signal_pipeline.batch_backfill import _generate_trading_days


class TestGenerateTradingDays:
    def test_filters_weekends(self):
        """Thu-Fri-Mon range should include Thu/Fri, skip Sat/Sun, include Mon."""
        days = _generate_trading_days(date(2025, 4, 24), date(2025, 4, 28))
        day_strs = [str(d.date()) for d in days]
        assert "2025-04-24" in day_strs  # Thursday
        assert "2025-04-25" in day_strs  # Friday
        assert "2025-04-26" not in day_strs  # Saturday
        assert "2025-04-27" not in day_strs  # Sunday
        assert "2025-04-28" in day_strs  # Monday

    def test_single_day_weekday(self):
        days = _generate_trading_days(date(2025, 4, 23), date(2025, 4, 23))
        assert len(days) == 1

    def test_single_day_weekend(self):
        days = _generate_trading_days(date(2025, 4, 26), date(2025, 4, 26))  # Saturday
        assert len(days) == 0
