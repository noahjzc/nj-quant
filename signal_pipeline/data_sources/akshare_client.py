import time
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class AKShareClient:
    """AKShare real-time data client for intraday snapshots."""

    COLUMN_MAP = {
        '代码': 'stock_code',
        '名称': 'stock_name',
        '最新价': 'close',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
        '成交额': 'turnover_amount',
        '振幅': 'amplitude',
        '涨跌幅': 'change_pct',
        '换手率': 'turnover_rate',
        '量比': 'volume_ratio',
        '市盈率-动态': 'pe_ttm',
        '市净率': 'pb',
        '流通市值': 'circulating_mv',
        '总市值': 'total_mv',
    }

    def __init__(self, max_retries: int = 3, retry_delay: int = 120):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_spot_all(self) -> pd.DataFrame:
        """Get real-time spot data for all A-share stocks."""
        df = self._call_with_retry()
        df = df.rename(columns=self.COLUMN_MAP)
        keep_cols = [c for c in self.COLUMN_MAP.values() if c in df.columns]
        return df[keep_cols]

    def _call_with_retry(self) -> pd.DataFrame:
        import akshare as ak
        last_error = None
        for attempt in range(self.max_retries):
            try:
                df = ak.stock_zh_a_spot_em()
                if not df.empty:
                    return df
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"AKShare error: {e}, retry in {self.retry_delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)

        raise RuntimeError(f"AKShare call failed after {self.max_retries} attempts: {last_error}")
