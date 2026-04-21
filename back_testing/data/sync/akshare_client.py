"""akshare API 封装层"""
import os
import time
import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class AkshareClient:
    """akshare API 封装，提供限流和错误处理"""

    def __init__(self, rate_limit: int = 10, retry_times: int = 3, timeout: int = 30, disable_proxy: bool = True):
        """
        Args:
            rate_limit: 每秒请求数限制
            retry_times: 重试次数
            timeout: 超时秒数
            disable_proxy: 是否禁用代理（默认 True，解决代理连接问题）
        """
        self.rate_limit = rate_limit
        self.retry_times = retry_times
        self.timeout = timeout
        self._min_interval = 1.0 / rate_limit
        self._last_request_time = 0.0
        self._disable_proxy = disable_proxy

        # 保存原始代理设置
        self._original_http_proxy = os.environ.get('http_proxy')
        self._original_https_proxy = os.environ.get('https_proxy')
        self._original_http_proxy = os.environ.get('HTTP_PROXY', self._original_http_proxy)
        self._original_https_proxy = os.environ.get('HTTPS_PROXY', self._original_https_proxy)

        # 禁用代理以解决连接问题
        if disable_proxy:
            self._disable_proxies()

    def _disable_proxies(self):
        """禁用代理设置"""
        for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY']:
            os.environ.pop(key, None)
        # 确保 requests 使用无代理 session
        if hasattr(requests, 'Session'):
            # 全局禁用
            requests.session().trust_env = False

    def _restore_proxies(self):
        """恢复原始代理设置"""
        if self._original_http_proxy:
            os.environ['http_proxy'] = self._original_http_proxy
            os.environ['HTTP_PROXY'] = self._original_http_proxy
        if self._original_https_proxy:
            os.environ['https_proxy'] = self._original_https_proxy
            os.environ['HTTPS_PROXY'] = self._original_https_proxy

    def _rate_limit_wait(self):
        """等待直到可以发送请求"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _retry_request(self, func, *args, **kwargs):
        """带重试的请求"""
        last_error = None
        for attempt in range(self.retry_times):
            try:
                self._rate_limit_wait()
                # 再次确保代理被禁用
                if self._disable_proxy:
                    self._disable_proxies()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"请求失败（第 {attempt + 1}/{self.retry_times} 次）: {e}")
                if attempt < self.retry_times - 1:
                    time.sleep(2 ** attempt)  # 指数退避
        raise last_error

    def get_stock_list(self) -> pd.DataFrame:
        """获取全市场A股列表"""
        df = self._retry_request(ak.stock_zh_a_spot_em)
        # 过滤只需要的基本信息
        result = df[['代码', '名称', '板块', '市值', '上市时间']].copy()
        result.columns = ['stock_code', 'stock_name', 'industry', 'market_cap', 'list_date']
        # 添加市场前缀
        result['stock_code'] = result['stock_code'].apply(
            lambda x: f'sh{x}' if str(x).startswith('6') else f'sz{x}'
        )
        return result

    def get_stock_daily(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取股票日线数据（后复权）

        Args:
            stock_code: 股票代码，如 sh600519
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            adjust: 复权类型，qfq=后复权，hfq=前复权，None=不复权

        Returns:
            DataFrame with columns: stock_code, trade_date, open, high, low, close, volume, turnover, amplitude, change_pct
        """
        # akshare 需要不带前缀的代码
        plain_code = stock_code[2:] if stock_code.startswith(('sh', 'sz')) else stock_code

        df = self._retry_request(
            ak.stock_zh_a_hist,
            symbol=plain_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # 重命名列
        df = df.rename(columns={
            '日期': 'trade_date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'turnover',
            '振幅': 'amplitude',
            '涨跌幅': 'change_pct'
        })

        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['stock_code'] = stock_code
        df['is_trading'] = True

        return df[['stock_code', 'trade_date', 'open', 'high', 'low', 'close',
                   'volume', 'turnover', 'amplitude', 'change_pct', 'is_trading']]

    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取指数日线数据

        Args:
            index_code: 指数代码，如 sh000001
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
        """
        df = self._retry_request(
            ak.index_zh_a_hist,
            symbol=index_code[2:],
            period="daily",
            start_date=start_date,
            end_date=end_date
        )

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            '日期': 'trade_date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume',
            '成交额': 'turnover'
        })

        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['index_code'] = index_code

        return df[['index_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover']]

    def get_stock_financial(
        self,
        stock_code: str,
        start_year: int = None,
        end_year: int = None
    ) -> pd.DataFrame:
        """
        获取股票财务数据

        Args:
            stock_code: 股票代码
            start_year: 起始年份
            end_year: 结束年份
        """
        plain_code = stock_code[2:] if stock_code.startswith(('sh', 'sz')) else stock_code

        try:
            df = self._retry_request(
                ak.stock_financial_analysis_indicator,
                symbol=plain_code
            )
        except Exception as e:
            logger.warning(f"获取财务数据失败 {stock_code}: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 只保留需要的列
        cols = ['股票代码', '报告日期', '净资产收益率', '资产报酬率', '销售毛利率',
                '销售净利率', '资产负债率', '流动比率', '速动比率',
                '市盈率(TTM)', '市净率', '市销率(TTM)']
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]

        df = df.rename(columns={
            '股票代码': 'stock_code',
            '报告日期': 'report_period',
            '净资产收益率': 'roe',
            '资产报酬率': 'roa',
            '销售毛利率': 'gross_margin',
            '销售净利率': 'net_margin',
            '资产负债率': 'debt_ratio',
            '流动比率': 'current_ratio',
            '速动比率': 'quick_ratio',
            '市盈率(TTM)': 'pe_ttm',
            '市净率': 'pb',
            '市销率(TTM)': 'ps_ttm'
        })

        df['stock_code'] = stock_code
        df['report_period'] = pd.to_datetime(df['report_period']).dt.date

        # 尝试获取发布日期（使用报告期后推2个月作为预估）
        df['publish_date'] = df['report_period'].apply(
            lambda x: date(x.year + (x.month // 12), (x.month % 12) + 1, 1) if x.month < 12
            else date(x.year + 1, 1, 1)
        )

        return df
