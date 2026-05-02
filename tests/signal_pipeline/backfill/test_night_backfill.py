"""Tests for night_backfill.py"""
import os
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure the module path is correct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "signal_pipeline"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConvertTsCode:
    """Tests for _convert_ts_code helper."""

    def test_sz_code(self):
        from signal_pipeline.night_backfill import _convert_ts_code
        assert _convert_ts_code("000001.SZ") == "sz000001"

    def test_sh_code(self):
        from signal_pipeline.night_backfill import _convert_ts_code
        assert _convert_ts_code("600519.SH") == "sh600519"

    def test_bj_code(self):
        from signal_pipeline.night_backfill import _convert_ts_code
        assert _convert_ts_code("920017.BJ") == "bj920017"

    def test_unknown_code_raises(self):
        from signal_pipeline.night_backfill import _convert_ts_code
        with pytest.raises(ValueError, match="Unknown Tushare code suffix"):
            _convert_ts_code("123456.XG")


class TestNaToNone:
    """Tests for _na_to_none helper."""

    def test_none_passthrough(self):
        from signal_pipeline.night_backfill import _na_to_none
        assert _na_to_none(None) is None

    def test_np_integer(self):
        from signal_pipeline.night_backfill import _na_to_none
        val = np.int64(42)
        result = _na_to_none(val)
        assert result == 42
        assert isinstance(result, int)

    def test_np_floating(self):
        from signal_pipeline.night_backfill import _na_to_none
        val = np.float64(3.14)
        result = _na_to_none(val)
        assert result == 3.14
        assert isinstance(result, float)

    def test_np_nan_becomes_none(self):
        from signal_pipeline.night_backfill import _na_to_none
        assert _na_to_none(np.nan) is None
        assert _na_to_none(np.float64("nan")) is None

    def test_np_inf_becomes_none(self):
        from signal_pipeline.night_backfill import _na_to_none
        assert _na_to_none(np.inf) is None
        assert _na_to_none(-np.inf) is None

    def test_pd_na_becomes_none(self):
        from signal_pipeline.night_backfill import _na_to_none
        assert _na_to_none(pd.NA) is None

    def test_normal_float(self):
        from signal_pipeline.night_backfill import _na_to_none
        assert _na_to_none(2.71) == 2.71

    def test_normal_int(self):
        from signal_pipeline.night_backfill import _na_to_none
        assert _na_to_none(100) == 100


class TestGetPreviousTradingDay:
    """Tests for _get_previous_trading_day helper."""

    def test_weekday_returns_yesterday(self):
        from signal_pipeline.night_backfill import _get_previous_trading_day
        # Mock date.today() to a Monday → yesterday is Sunday → should return Friday
        with patch("signal_pipeline.night_backfill.date") as mock_date:
            # Monday 2025-04-28
            mock_date.today.return_value = date(2025, 4, 28)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            result = _get_previous_trading_day()
            assert result == date(2025, 4, 25)  # Friday

    def test_friday_returns_thursday(self):
        from signal_pipeline.night_backfill import _get_previous_trading_day
        with patch("signal_pipeline.night_backfill.date") as mock_date:
            # Friday 2025-04-25
            mock_date.today.return_value = date(2025, 4, 25)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            result = _get_previous_trading_day()
            assert result == date(2025, 4, 24)  # Thursday

    def test_saturday_returns_friday(self):
        from signal_pipeline.night_backfill import _get_previous_trading_day
        with patch("signal_pipeline.night_backfill.date") as mock_date:
            # Saturday 2025-04-26
            mock_date.today.return_value = date(2025, 4, 26)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            result = _get_previous_trading_day()
            assert result == date(2025, 4, 25)  # Friday

    def test_sunday_returns_friday(self):
        from signal_pipeline.night_backfill import _get_previous_trading_day
        with patch("signal_pipeline.night_backfill.date") as mock_date:
            # Sunday 2025-04-27
            mock_date.today.return_value = date(2025, 4, 27)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            result = _get_previous_trading_day()
            assert result == date(2025, 4, 25)  # Friday


class TestNightBackfillIntegration:
    """Integration tests for the full backfill pipeline using mocks."""

    @pytest.fixture
    def mock_tushare_data(self):
        """Return mock Tushare DataFrames."""
        daily = pd.DataFrame({
            "ts_code": ["000001.SZ", "600519.SH"],
            "trade_date": ["20250428", "20250428"],
            "open": [10.0, 1800.0],
            "high": [10.5, 1850.0],
            "low": [9.8, 1790.0],
            "close": [10.2, 1830.0],
            "volume": [1_000_000, 500_000],
            "amount": [10_200_000.0, 915_000_000.0],
            "pct_chg": [2.0, 1.67],
        })
        basic = pd.DataFrame({
            "ts_code": ["000001.SZ", "600519.SH"],
            "turnover_rate": [0.5, 0.3],
            "volume_ratio": [1.2, 0.9],
            "circulating_mv": [1e9, 2e10],
            "total_mv": [2e9, 3e10],
            "pe_ttm": [8.5, 30.2],
            "ps_ttm": [1.1, 10.5],
            "pcf_ttm": [5.0, 20.1],
            "pb": [1.2, 5.0],
        })
        adj = pd.DataFrame({
            "ts_code": ["000001.SZ", "600519.SH"],
            "trade_date": ["20250428", "20250428"],
            "adj_factor": [1.02, 1.05],
        })
        return daily, basic, adj

    def test_full_pipeline_with_mocks(self, mock_tushare_data, tmp_path):
        """Test that the pipeline runs end-to-end with mocked Tushare and DB."""
        daily, basic, adj = mock_tushare_data

        # Patch environment and Tushare client
        with patch.dict(os.environ, {"TUSHARE_TOKEN": "mock_token"}):
            with patch("signal_pipeline.night_backfill.TushareClient") as MockClient:
                mock_client_instance = MagicMock()
                mock_client_instance.get_daily_all.return_value = daily
                mock_client_instance.get_daily_basic_all.return_value = basic
                mock_client_instance.get_adj_factor_all.return_value = adj
                MockClient.return_value = mock_client_instance

                with patch("signal_pipeline.night_backfill.get_engine") as mock_engine:
                    with patch("signal_pipeline.night_backfill.get_session") as mock_session:
                        mock_conn = MagicMock()
                        mock_engine.return_value.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
                        mock_engine.return_value.begin.return_value.__exit__ = MagicMock()

                        mock_session_instance = MagicMock()
                        mock_session_instance.bind.func.now.return_value = None
                        mock_session.return_value = mock_session_instance

                        # Patch _cron_start / _cron_finish
                        with patch("signal_pipeline.night_backfill._cron_start") as mock_start:
                            with patch("signal_pipeline.night_backfill._cron_finish") as mock_finish:
                                mock_start.return_value = 1

                                # Patch sys.argv so main() uses _get_previous_trading_day instead of CLI arg
                                with patch("sys.argv", ["night_backfill.py"]):
                                    # Patch CACHE_DIR to tmp_path
                                    with patch("signal_pipeline.night_backfill.CACHE_DIR", tmp_path):
                                        with patch("signal_pipeline.night_backfill.DAILY_CACHE_DIR", tmp_path / "daily"):
                                            with patch("signal_pipeline.night_backfill._get_previous_trading_day") as mock_date:
                                                mock_date.return_value = date(2025, 4, 28)

                                                from signal_pipeline.night_backfill import main

                                                # Should not raise
                                                main()

                                            # Verify Tushare was called with correct date
                                            mock_client_instance.get_daily_all.assert_called_once_with("20250428")
                                            mock_client_instance.get_daily_basic_all.assert_called_once_with("20250428")
                                            mock_client_instance.get_adj_factor_all.assert_called_once_with("20250428")

                                            # Verify cron_finish was called with success
                                            mock_finish.assert_called_once()
                                            call_args = mock_finish.call_args
                                            assert call_args[0][2] == "success"  # status


class TestImports:
    """Verify all imports work correctly."""

    def test_imports_succeed(self):
        """Verify the module can be imported without errors."""
        import signal_pipeline.night_backfill
        assert hasattr(signal_pipeline.night_backfill, "_convert_ts_code")
        assert hasattr(signal_pipeline.night_backfill, "_na_to_none")
        assert hasattr(signal_pipeline.night_backfill, "_cron_start")
        assert hasattr(signal_pipeline.night_backfill, "_cron_finish")
        assert hasattr(signal_pipeline.night_backfill, "_get_previous_trading_day")
        assert hasattr(signal_pipeline.night_backfill, "main")

    def test_token_check_on_missing(self):
        """Verify script exits when TUSHARE_TOKEN is missing."""
        import subprocess
        import sys

        # Run with empty env
        env = os.environ.copy()
        env.pop("TUSHARE_TOKEN", None)
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, 'signal_pipeline'); from night_backfill import main; main()"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode != 0
        assert "TUSHARE_TOKEN" in result.stderr or "not set" in result.stderr
