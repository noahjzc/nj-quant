# signal_pipeline/data_sources/tushare_client.py
import time
import logging
import pandas as pd
from typing import Callable

import tushare as ts

logger = logging.getLogger(__name__)


class TushareClient:
    """Tushare Pro data client with retry logic."""

    def __init__(self, token: str, max_retries: int = 3, retry_delay: int = 120):
        self.token = token
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._pro = None

    @property
    def pro(self):
        if self._pro is None:
            self._pro = ts.pro_api(self.token)
        return self._pro

    def get_daily_all(self, trade_date: str) -> pd.DataFrame:
        return self._call_with_retry(lambda: self.pro.daily(trade_date=trade_date))

    def get_daily_basic_all(self, trade_date: str) -> pd.DataFrame:
        return self._call_with_retry(lambda: self.pro.daily_basic(trade_date=trade_date))

    def get_adj_factor_all(self, trade_date: str) -> pd.DataFrame:
        return self._call_with_retry(lambda: self.pro.adj_factor(trade_date=trade_date))

    def _call_with_retry(self, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = fn()
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
                if attempt < self.max_retries - 1:
                    logger.warning(f"Empty result, retry {attempt + 2}/{self.max_retries}")
                    time.sleep(self.retry_delay)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"Tushare error: {e}, retry in {self.retry_delay}s ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)

        raise RuntimeError(f"Tushare call failed after {self.max_retries} attempts: {last_error}")
