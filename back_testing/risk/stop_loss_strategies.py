"""
止损止盈策略模块 - ATR和移动止损计算

提供基于ATR的动态止损止盈和移动止损功能。

参数配置：
    ATR周期: 14天（标准值）
    止损倍数: 2.0×ATR
    止盈倍数: 3.0×ATR
    移动止损幅度: 10%
    移动止损启动条件: 5%盈利
"""
import numpy as np
import pandas as pd
from typing import Optional


class StopLossStrategies:
    """止损止盈策略计算器"""

    # 默认参数
    DEFAULT_ATR_PERIOD = 14
    DEFAULT_STOP_LOSS_MULT = 2.0
    DEFAULT_TAKE_PROFIT_MULT = 3.0
    DEFAULT_TRAILING_PCT = 0.10
    DEFAULT_TRAILING_START = 0.05  # 5% profit threshold

    @staticmethod
    def _get_column(df: pd.DataFrame, *names) -> str:
        """获取DataFrame中存在的列名"""
        for name in names:
            if name in df.columns:
                return name
        raise ValueError(f"None of the columns {names} found in DataFrame")

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """
        计算ATR（Average True Range）

        ATR = MA(TR, period)

        其中 TR = max(H-L, |H-PC|, |L-PC|)
        H: 当日最高价
        L: 当日最低价
        PC: 前一日收盘价

        Args:
            df: 股票数据DataFrame，必须包含 HIGH_PRICE, LOW_PRICE, CLOSE_PRICE 列
                 如果没有前收盘价列，则使用收盘价的前一行作为前收盘价
            period: ATR计算周期，默认14

        Returns:
            float: 最新一根K线的ATR值
        """
        if len(df) < period + 1:
            raise ValueError(f"数据长度不足{period + 1}根K线，无法计算ATR")

        # 获取列名（兼容中文、英文大写和英文小写列名）
        high_col = StopLossStrategies._get_column(df, 'high', 'HIGH_PRICE', '最高价')
        low_col = StopLossStrategies._get_column(df, 'low', 'LOW_PRICE', '最低价')
        close_col = StopLossStrategies._get_column(df, 'close', 'CLOSE_PRICE', '收盘价')

        # 尝试获取前收盘价，如果没有则用收盘价前一值
        prev_close_col = None
        for col in ['prev_adj_close', 'prev_close', 'PREVIOUS_CLOSE_PRICE', '前收盘价']:
            if col in df.columns:
                prev_close_col = col
                break

        # 计算 True Range
        high = df[high_col].values
        low = df[low_col].values
        close = df[close_col].values

        # 使用前收盘价列或使用收盘价前一行
        if prev_close_col and prev_close_col in df.columns:
            prev_close = df[prev_close_col].values
        else:
            # 使用收盘价的前一行作为前收盘价（从1开始，0位置是当前收盘价）
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]  # 第一个值用自己

        # TR = max(H-L, |H-PC|, |L-PC|)
        tr1 = high - low  # 当日高低点差
        tr2 = np.abs(high - prev_close)  # 当日最高与前收盘价差
        tr3 = np.abs(low - prev_close)  # 当日最低与前收盘价差

        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # ATR = MA(TR, period) - 使用最近period个TR值
        atr = np.mean(tr[-period:])

        return float(atr)

    @staticmethod
    def calculate_stop_loss(buy_price: float, atr: float, mult: float = 2.0) -> float:
        """
        计算止损价格线

        止损线 = 买入价 - (倍数 × ATR)

        Args:
            buy_price: 买入价格
            atr: ATR值
            mult: 止损倍数，默认2.0

        Returns:
            float: 止损价格线
        """
        return buy_price - (mult * atr)

    @staticmethod
    def calculate_take_profit(buy_price: float, atr: float, mult: float = 3.0) -> float:
        """
        计算止盈价格线

        止盈线 = 买入价 + (倍数 × ATR)

        Args:
            buy_price: 买入价格
            atr: ATR值
            mult: 止盈倍数，默认3.0

        Returns:
            float: 止盈价格线
        """
        return buy_price + (mult * atr)

    @staticmethod
    def calculate_trailing_stop(highest_price: float, trailing_pct: float = 0.10) -> float:
        """
        计算移动止损价格线

        移动止损线 = 最高价 × (1 - 跟踪幅度)

        Args:
            highest_price: 持仓期间最高价
            trailing_pct: 跟踪幅度，默认0.10（10%）

        Returns:
            float: 移动止损价格线
        """
        return highest_price * (1 - trailing_pct)

    @staticmethod
    def check_exit(
        position: dict,
        current_price: float,
        atr: float,
        highest_price: float,
        stop_loss_mult: float = 2.0,
        take_profit_mult: float = 3.0,
        trailing_pct: float = 0.10,
        trailing_start: float = 0.05
    ) -> dict:
        """
        检查是否触发止损/止盈/移动止损

        执行优先级（根据spec第54-58行）：
        1. 止损（无条件触发）
        2. 移动止损（只在浮盈≥trailing_start或已触及止盈线时生效）
        3. 正常持仓

        止盈逻辑（根据spec第34行）：
        如果当前价 >= 止盈线 --> 进入移动止损监控模式
        即：止盈线被触及后，以止盈价作为新的最高价，启动移动止损

        Args:
            position: 持仓信息字典，包含:
                - 'buy_price': 买入价格
            current_price: 当前价格
            atr: ATR值
            highest_price: 持仓期间最高价
            stop_loss_mult: 止损倍数，默认2.0
            take_profit_mult: 止盈倍数，默认3.0
            trailing_pct: 移动止损幅度，默认0.10（10%）
            trailing_start: 移动止损启动盈利门槛，默认0.05（5%）

        Returns:
            dict: {
                'action': 'stop_loss' | 'trailing_stop' | None,
                'price': 触发价格,
                'reason': 触发原因,
                'take_profit_hit': bool  # 是否已触及止盈线（进入移动止损监控模式）
            }
        """
        buy_price = position.get('buy_price', current_price)

        # 计算各价格线
        stop_loss_price = StopLossStrategies.calculate_stop_loss(buy_price, atr, stop_loss_mult)
        take_profit_price = StopLossStrategies.calculate_take_profit(buy_price, atr, take_profit_mult)

        # 计算当前盈亏
        profit_pct = (current_price - buy_price) / buy_price

        # 判断是否已触及止盈线（进入移动止损监控模式）
        take_profit_hit = current_price >= take_profit_price

        # 如果已触及止盈线，以止盈价作为新的最高价计算移动止损
        # 否则使用持仓期间最高价
        effective_highest = take_profit_price if take_profit_hit else highest_price
        trailing_stop_price = StopLossStrategies.calculate_trailing_stop(effective_highest, trailing_pct)

        # 1. 止损检查（无条件）
        if current_price <= stop_loss_price:
            return {
                'action': 'stop_loss',
                'price': current_price,
                'reason': f'止损触发：当前价{current_price:.2f} ≤ 止损线{stop_loss_price:.2f} '
                          f'(买入价{buy_price:.2f} - {stop_loss_mult}×ATR{atr:.2f})',
                'take_profit_hit': take_profit_hit
            }

        # 2. 移动止损检查（只在浮盈≥trailing_start或已触及止盈线时生效）
        # 根据spec：止盈线被触及后进入移动止损监控模式
        if profit_pct >= trailing_start or take_profit_hit:
            if current_price <= trailing_stop_price:
                return {
                    'action': 'trailing_stop',
                    'price': current_price,
                    'reason': f'移动止损触发：当前价{current_price:.2f} ≤ 移动止损线{trailing_stop_price:.2f} '
                              f'(参考价{effective_highest:.2f} × (1-{trailing_pct}))，盈利{profit_pct*100:.1f}%',
                    'take_profit_hit': take_profit_hit
                }

        # 3. 正常持仓
        return {
            'action': None,
            'price': current_price,
            'reason': '正常持仓',
            'take_profit_hit': take_profit_hit
        }


def calculate_atr_simple(df: pd.DataFrame, period: int = 14) -> float:
    """
    便捷函数：计算ATR（使用后复权价计算）

    Args:
        df: 股票数据DataFrame
        period: ATR计算周期，默认14

    Returns:
        float: 最新ATR值
    """
    return StopLossStrategies.calculate_atr(df, period)
