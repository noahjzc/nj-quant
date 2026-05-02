"""Tests for sync_overview.py"""
import os
import tempfile
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestProcessStockDf:
    """测试 _process_stock_df 字段映射和类型转换。"""

    def test_columns_renamed(self):
        from data.sync.sync_overview import _process_stock_df

        raw = pd.DataFrame({
            "股票代码": ["sh600519", "sz000001"],
            "股票名称": ["贵州茅台", "平安银行"],
            "交易日期": ["2025-04-23", "2025-04-23"],
            "开盘价": [1800.0, 12.0],
            "最高价": [1850.0, 12.5],
            "最低价": [1790.0, 11.8],
            "收盘价": [1830.0, 12.2],
            "后复权价": [1830.0, 12.2],
            "前复权价": [1800.0, 12.0],
            "涨跌幅": [1.5, 0.8],
            "成交量": [100000.0, 500000.0],
            "成交额": [1000000.0, 5000000.0],
            "换手率": [0.5, 0.3],
            "流通市值": [1e9, 2e9],
            "总市值": [2e9, 3e9],
            "是否涨停": ["否", "是"],
            "是否跌停": ["否", "否"],
            "市盈率TTM": [30.0, 8.0],
            "市销率TTM": [10.0, 5.0],
            "市现率TTM": [20.0, 15.0],
            "市净率": [5.0, 1.2],
            "MA_5": [1820.0, 12.1],
            "MA_10": [1810.0, 12.0],
            "MA_20": [1800.0, 11.9],
            "MA_30": [1795.0, 11.8],
            "MA_60": [1780.0, 11.5],
            "MA金叉死叉": ["金叉", "死叉"],
            "MACD_DIF": [1.5, 0.3],
            "MACD_DEA": [1.2, 0.2],
            "MACD_MACD": [0.6, 0.2],
            "MACD_金叉死叉": ["金叉", "死叉"],
            "KDJ_K": [60.0, 50.0],
            "KDJ_D": [55.0, 48.0],
            "KDJ_J": [70.0, 54.0],
            "KDJ_金叉死叉": ["金叉", "死叉"],
            "布林线中轨": [1800.0, 12.0],
            "布林线上轨": [1850.0, 12.5],
            "布林线下轨": [1750.0, 11.5],
            "psy": [50.0, 45.0],
            "psyma": [48.0, 44.0],
            "rsi1": [60.0, 55.0],
            "rsi2": [58.0, 53.0],
            "rsi3": [56.0, 51.0],
            "振幅": [3.0, 5.0],
            "量比": [1.2, 0.9],
        })

        df = _process_stock_df(raw)

        assert list(df.columns) == [
            "stock_code", "stock_name", "trade_date", "open", "high", "low", "close",
            "adj_close", "prev_adj_close", "change_pct", "volume", "turnover_amount",
            "turnover_rate", "circulating_mv", "total_mv", "limit_up", "limit_down",
            "pe_ttm", "ps_ttm", "pcf_ttm", "pb",
            "ma_5", "ma_10", "ma_20", "ma_30", "ma_60", "ma_cross",
            "macd_dif", "macd_dea", "macd_hist", "macd_cross",
            "kdj_k", "kdj_d", "kdj_j", "kdj_cross",
            "boll_mid", "boll_upper", "boll_lower",
            "psy", "psyma", "rsi_1", "rsi_2", "rsi_3",
            "amplitude", "volume_ratio",
        ]
        assert df.iloc[0]["stock_code"] == "sh600519"
        assert df.iloc[0]["limit_up"] == False
        assert df.iloc[1]["limit_up"] == True
        assert df.iloc[0]["trade_date"] == date(2025, 4, 23)


class TestNaToNone:
    """测试 _na_to_none。"""

    def test_normal_passthrough(self):
        from data.sync.sync_overview import _na_to_none
        assert _na_to_none(42) == 42
        assert _na_to_none(3.14) == 3.14

    def test_nan_becomes_none(self):
        from data.sync.sync_overview import _na_to_none
        assert _na_to_none(float("nan")) is None
        assert _na_to_none(float("inf")) is None
        assert _na_to_none(-float("inf")) is None

    def test_pd_na(self):
        from data.sync.sync_overview import _na_to_none
        assert _na_to_none(pd.NA) is None


class TestOverviewClientParse:
    """测试 OverviewClient.parse_zip。"""

    def test_parse_zip(self):
        from data.sync.overview_client import OverviewClient

        # 构造模拟 ZIP
        stock_csv = "股票代码,股票名称,交易日期,开盘价,最高价,最低价,收盘价\nsh600519,贵州茅台,2025-04-23,1800,1850,1790,1830"
        index_csv = "index_code,date,open,close,low,high,volume,money,change\nsh000001,2025-04-23,3400,3500,3380,3510,100000,100000000,0.02"

        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            z.writestr("overview-push-20250423/stock overview.csv", stock_csv.encode("gbk"))
            z.writestr("overview-push-20250423/index data.csv", index_csv.encode("gbk"))

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(zip_buf.getvalue())
            tmp_path = f.name

        try:
            stock_df, index_df = OverviewClient.parse_zip(tmp_path)
            assert len(stock_df) == 1
            assert len(index_df) == 1
            assert index_df.iloc[0]["index_code"] == "sh000001"
        finally:
            os.unlink(tmp_path)