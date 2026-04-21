"""
数据提供器 - 统一管理股票数据读取

支持两种数据源：
- Parquet格式（推荐）：读取速度快10-50倍
- CSV格式（兼容）：原始格式

使用方式：
    from back_testing.data.data_provider import DataProvider

    provider = DataProvider()  # 默认使用Parquet
    # 或者
    provider = DataProvider(use_parquet=False)  # 使用CSV

    df = provider.get_stock_data('sh600519')
    df = provider.get_stock_data('sh600519', date='2024-01-15')  # 筛选到指定日期
"""
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class DataProvider:
    """
    统一数据提供器
    """

    def __init__(
        self,
        use_parquet: bool = True,
        data_dir: Optional[str] = None
    ):
        """
        Args:
            use_parquet: 是否使用Parquet格式（默认True）
            data_dir: 数据目录，默认使用项目中的data/daily_ycz
        """
        self.use_parquet = use_parquet

        # 确定数据目录
        if data_dir is None:
            # 默认使用项目中的data/daily_ycz
            project_root = Path(__file__).parent.parent
            self.data_dir = project_root / 'data' / 'daily_ycz'
        else:
            self.data_dir = Path(data_dir)

        # 备选CSV目录（用于Parquet不存在时降级）
        self.csv_dir = Path(r'D:\workspace\code\mine\quant\data\metadata\daily_ycz')

    def _get_file_path(self, stock_code: str) -> Path:
        """获取数据文件路径"""
        if self.use_parquet:
            parquet_path = self.data_dir / f'{stock_code}.parquet'
            if parquet_path.exists():
                return parquet_path
            # Parquet不存在，尝试降级到CSV
            csv_path = self.csv_dir / f'{stock_code}.csv'
            if csv_path.exists():
                return csv_path
            raise FileNotFoundError(f"找不到数据文件: {stock_code}")
        else:
            csv_path = self.csv_dir / f'{stock_code}.csv'
            if csv_path.exists():
                return csv_path
            raise FileNotFoundError(f"找不到数据文件: {stock_code}")

    def get_stock_data(
        self,
        stock_code: str,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取股票数据

        Args:
            stock_code: 股票代码，如 'sh600519'
            date: 筛选到指定日期之前的数据（不含当天）
            start_date: 筛选开始日期
            end_date: 筛选结束日期

        Returns:
            DataFrame，按交易日期排序
        """
        file_path = self._get_file_path(stock_code)

        # 读取数据
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, encoding='gbk')

        # 确保日期列存在
        date_col = '交易日期'
        if date_col not in df.columns:
            # 尝试其他可能的列名
            for col in ['date', 'Date', 'DATE', '交易日期']:
                if col in df.columns:
                    date_col = col
                    break

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        # 按日期筛选
        if date is not None:
            date_ts = pd.to_datetime(date)
            df = df[df[date_col] < date_ts]

        if start_date is not None:
            df = df[df[date_col] >= pd.to_datetime(start_date)]

        if end_date is not None:
            df = df[df[date_col] <= pd.to_datetime(end_date)]

        return df

    def get_stock_price(
        self,
        stock_code: str,
        date: Union[str, pd.Timestamp]
    ) -> Optional[float]:
        """
        获取指定日期的收盘价（后复权价）

        Args:
            stock_code: 股票代码
            date: 日期

        Returns:
            收盘价，如果不存在返回None
        """
        date_ts = pd.to_datetime(date)

        file_path = self._get_file_path(stock_code)

        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, encoding='gbk')

            date_col = '交易日期'
            if date_col not in df.columns:
                for col in ['date', 'Date', 'DATE']:
                    if col in df.columns:
                        date_col = col
                        break

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)

            # 找指定日期之前或当天的最近交易日
            hist = df[df[date_col] <= date_ts]
            if len(hist) == 0:
                return None

            return hist.iloc[-1].get('后复权价')
        except Exception:
            return None

    def get_latest_price(self, stock_code: str) -> Optional[float]:
        """获取最近收盘价"""
        df = self.get_stock_data(stock_code)
        if len(df) == 0:
            return None
        return df.iloc[-1].get('后复权价')

    def get_all_stock_codes(self) -> list:
        """获取所有股票代码（排除BJ开头的北交所股票）"""
        if self.use_parquet and self.data_dir.exists():
            files = list(self.data_dir.glob('*.parquet'))
            return [f.stem for f in files if not f.stem.startswith('bj')]
        elif self.csv_dir.exists():
            files = list(self.csv_dir.glob('*.csv'))
            return [f.stem for f in files if not f.stem.startswith('index') and not f.stem.startswith('bj')]
        return []


# 全局默认实例
_default_provider: Optional[DataProvider] = None


def get_provider() -> DataProvider:
    """获取默认数据提供器（单例）"""
    global _default_provider
    if _default_provider is None:
        _default_provider = DataProvider()
    return _default_provider
