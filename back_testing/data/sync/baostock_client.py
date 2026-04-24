"""baostock API 封装层"""
import logging
import threading
from datetime import date, datetime
from typing import List, Optional

import baostock as bs
import pandas as pd

logger = logging.getLogger(__name__)

# 主要指数代码
INDEX_CODES = ['sh000001', 'sh000300', 'sz399001', 'sz399006']  # 上证, 沪深300, 深证, 创业板


class BaostockClient:
    """baostock API 封装，提供数据获取接口"""

    def __init__(self):
        """初始化 baostock 客户端"""
        self._logged_in = False

    def _ensure_login(self):
        """确保已登录"""
        if not self._logged_in:
            lg = bs.login()
            if lg.error_code != '0':
                raise ConnectionError(f"baostock login failed: {lg.error_msg}")
            self._logged_in = True

    def _logout(self):
        """登出"""
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def __enter__(self):
        self._ensure_login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._logout()
        return False

    def _query_with_timeout(self, bs_code: str, fields: str, start_date: str, end_date: str,
                            adjustflag: str = "2", timeout: int = 30, max_retries: int = 3):
        """
        带超时和重试的查询

        Args:
            bs_code: baostock格式代码，如 sh.600519
            fields: 查询字段
            start_date: 开始日期
            end_date: 结束日期
            adjustflag: 复权类型 ("1"=前复权, "2"=后复权, "3"=不复权)
            timeout: 超时秒数
            max_retries: 最大重试次数

        Returns:
            ResultSet 或 None（超时或重试耗尽）
        """
        import time as time_module

        # 需要重试的网络错误
        retry_errors = (
            '10054',  # 远程主机强迫关闭了一个现有的连接
            '远程主机强迫关闭',
            '连接被关闭',
            'Connection reset',
            '接收数据异常',
        )

        for attempt in range(max_retries):
            result = {'rs': None, 'error': None}

            def _query():
                try:
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        fields,
                        start_date=start_date,
                        end_date=end_date,
                        frequency="d",
                        adjustflag=adjustflag
                    )
                    result['rs'] = rs
                except Exception as e:
                    result['error'] = e

            thread = threading.Thread(target=_query)
            thread.daemon = True
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                logger.warning(f"查询 {bs_code} 第{attempt + 1}次超时（{timeout}秒）")
                time_module.sleep(1)
                continue

            if result['error']:
                error_str = str(result['error'])
                # 检查是否是网络错误，需要重试
                if any(e in error_str for e in retry_errors):
                    logger.warning(f"查询 {bs_code} 第{attempt + 1}次遇到网络错误: {result['error']}")
                    time_module.sleep(2)  # 等待后重试
                    continue
                else:
                    raise result['error']

            return result['rs']

        # 所有重试都失败
        logger.warning(f"查询 {bs_code} 失败，已重试{max_retries}次")
        return None

    @staticmethod
    def _format_date(date_str: str) -> str:
        """转换日期格式 YYYYMMDD -> YYYY-MM-DD"""
        if date_str and len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取全市场A股列表

        Returns:
            DataFrame with columns: stock_code, stock_name, industry, list_date
        """
        self._ensure_login()

        # 获取所有股票
        rs = bs.query_all_stock(day=datetime.now().strftime('%Y-%m-%d'))
        if rs.error_code != '0':
            raise ConnectionError(f"query_all_stock failed: {rs.error_msg}")

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 重命名列
        df = df.rename(columns={
            'code': 'stock_code',
            'code_name': 'stock_name',
            'tradeStatus': 'trade_status'
        })

        # 只保留 A 股（sh.6xxxxx, sz.0xxxxx, sz.3xxxxx）
        # baostock 格式是 sh.600000, sz.000001 等
        # 转换为 sh600000, sz000001 格式
        df['stock_code'] = df['stock_code'].str.replace('.', '')

        # 过滤只保留 A 股
        df = df[df['stock_code'].str.match(r'^(sh6|sz0|sz3)')]

        # 判断市场
        df['market'] = df['stock_code'].apply(
            lambda x: '沪' if x.startswith('sh') else '深'
        )

        return df[['stock_code', 'stock_name', 'market', 'trade_status']]

    def get_stock_daily(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取股票日线数据

        Args:
            stock_code: 股票代码，如 sh600519
            start_date: 开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
            end_date: 结束日期，格式 YYYYMMDD 或 YYYY-MM-DD

        Returns:
            DataFrame with columns: stock_code, trade_date, open, high, low, close, volume, turnover, amplitude, change_pct, is_trading
        """
        self._ensure_login()

        # 转换日期格式 YYYYMMDD -> YYYY-MM-DD
        start_date = self._format_date(start_date)
        end_date = self._format_date(end_date)

        # baostock 需要 sh.600519 格式
        bs_code = f"{stock_code[:2]}.{stock_code[2:]}"

        rs = self._query_with_timeout(
            bs_code,
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date, end_date,
            adjustflag="2",
            timeout=30
        )

        if rs is None:
            return pd.DataFrame()

        if rs.error_code != '0':
            logger.warning(f"获取 {stock_code} 日线数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 重命名列
        df = df.rename(columns={
            'date': 'trade_date',
            'amount': 'turnover',
            'turn': 'turnover_rate',
            'pctChg': 'change_pct'
        })

        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        df['stock_code'] = stock_code
        df['volume'] = df['volume'] * 100  # 转换为手数
        df['amplitude'] = ((df['high'] - df['low']) / df['low'] * 100).fillna(0)
        df['is_trading'] = df['close'] > 0

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
        self._ensure_login()

        # 转换日期格式 YYYYMMDD -> YYYY-MM-DD
        start_date = self._format_date(start_date)
        end_date = self._format_date(end_date)

        # baostock 格式
        bs_code = f"{index_code[:2]}.{index_code[2:]}"

        rs = self._query_with_timeout(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date, end_date,
            adjustflag="3",
            timeout=30
        )

        if rs is None:
            return pd.DataFrame()

        if rs.error_code != '0':
            logger.warning(f"获取 {index_code} 指数数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.rename(columns={
            'date': 'trade_date',
            'amount': 'turnover'
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
        self._ensure_login()

        # 转换年份为日期格式
        start_date = f"{start_year}-01-01" if start_year else None
        end_date = f"{end_year}-12-31" if end_year else None

        # baostock 格式
        bs_code = f"{stock_code[:2]}.{stock_code[2:]}"

        # 获取财务数据
        rs = bs.query_financial_data(bs_code, start_date, end_date)
        if rs.error_code != '0':
            logger.warning(f"获取 {stock_code} 财务数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 需要的列映射
        col_map = {
            'code': 'stock_code',
            'pubDate': 'publish_date',
            'statDate': 'report_period',
            'roeAvg': 'roe',
            'roa': 'roa',
            'grossProfitMargin': 'gross_margin',
            'netProfitMargin': 'net_margin',
            'debtEquityRatio': 'debt_ratio',
            'currentRatio': 'current_ratio',
            'quickRatio': 'quick_ratio',
            'peTTM': 'pe_ttm',
            'pbMRQ': 'pb',
            'psTTM': 'ps_ttm'
        }

        # 只保留存在的列
        available_cols = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)

        # 转换数据类型
        numeric_cols = ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio',
                       'current_ratio', 'quick_ratio', 'pe_ttm', 'pb', 'ps_ttm']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 转换日期
        if 'publish_date' in df.columns:
            df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce').dt.date
        if 'report_period' in df.columns:
            df['report_period'] = pd.to_datetime(df['report_period'], errors='coerce').dt.date

        df['stock_code'] = stock_code

        return df

    def get_dividend_data(self, stock_code: str) -> pd.DataFrame:
        """
        获取股票分红数据

        Args:
            stock_code: 股票代码
        """
        self._ensure_login()

        bs_code = f"{stock_code[:2]}.{stock_code[2:]}"

        rs = bs.query_dividend(bs_code)
        if rs.error_code != '0':
            logger.warning(f"获取 {stock_code} 分红数据失败: {rs.error_msg}")
            return pd.DataFrame()

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        return df
