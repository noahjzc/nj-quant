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


if __name__ == '__main__':

    def run_tests():
        """运行所有单元测试"""
        print("StopLossStrategies Unit Tests")
        print("=" * 50)

        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        test_data = pd.DataFrame({
            'high': [10.0 + i * 0.1 for i in range(20)],
            'low': [9.0 + i * 0.1 for i in range(20)],
            'close': [9.8 + i * 0.1 for i in range(20)],
            'prev_adj_close': [9.5 + i * 0.1 for i in range(20)],
        }, index=dates)

        # ========== Test 1: ATR Calculation ==========
        print("\n[Test 1] ATR Calculation")
        atr = StopLossStrategies.calculate_atr(test_data, period=14)
        assert isinstance(atr, float), "ATR should be float"
        assert atr > 0, "ATR should be positive"
        print(f"  ATR = {atr:.4f} - PASS")

        # ========== Test 2: Stop Loss / Take Profit Price Calculation ==========
        print("\n[Test 2] Stop Loss / Take Profit Calculation")
        buy_price = 10.0
        stop_loss = StopLossStrategies.calculate_stop_loss(buy_price, atr)
        take_profit = StopLossStrategies.calculate_take_profit(buy_price, atr)
        trailing_stop = StopLossStrategies.calculate_trailing_stop(15.0)

        assert stop_loss < buy_price, "Stop loss should be below buy price"
        assert take_profit > buy_price, "Take profit should be above buy price"
        assert trailing_stop < 15.0, "Trailing stop should be below high price"
        print(f"  Stop={stop_loss:.4f}, TakeProfit={take_profit:.4f}, Trail={trailing_stop:.4f} - PASS")

        # ========== Test 3: Stop Loss Trigger (Unconditional) ==========
        print("\n[Test 3] Stop Loss Trigger (Unconditional)")
        # With atr=1.0 (calculated), stop_loss=10.0-2.0*1.0=8.0, so price<=7.5 triggers
        position = {'buy_price': buy_price}
        result = StopLossStrategies.check_exit(
            position, current_price=7.5, atr=atr, highest_price=15.0
        )
        assert result['action'] == 'stop_loss', f"Should trigger stop loss, got: {result['action']}"
        assert 'take_profit_hit' in result, "Result should contain take_profit_hit field"
        print(f"  Stop loss triggered - PASS")

        # ========== Test 4: Take Profit Hit -> Enter Trailing Stop Monitoring (Key Fix) ==========
        print("\n[Test 4] Take Profit Hit -> Enter Trailing Stop Monitoring (No Immediate Exit)")
        # Buy price 10.0, ATR=0.1, take_profit_mult=3.0 -> take_profit = 10 + 0.3 = 10.3
        # When current_price = 10.3 (hits take profit), should enter trailing_stop mode
        # NOT return take_profit action immediately
        # Stop loss = 10.0 - 2.0*0.1 = 9.8, so current_price=10.3 > 9.8, no stop loss
        result = StopLossStrategies.check_exit(
            position, current_price=10.3, atr=0.1, highest_price=10.0,
            stop_loss_mult=2.0, take_profit_mult=3.0, trailing_pct=0.10, trailing_start=0.05
        )
        assert result['action'] != 'take_profit', "Should NOT immediately exit on take profit hit"
        assert result['take_profit_hit'] == True, "Should mark as take profit hit"
        # Trailing stop = 10.3 * 0.9 = 9.27, current_price=10.3 > 9.27, so hold
        assert result['action'] is None, f"current_price=10.3 > trailing_stop=9.27, should hold, got: {result['action']}"
        print(f"  Take profit hit -> monitoring mode entered - PASS")

        # ========== Test 5: After Take Profit, Price Drops and Triggers Trailing Stop ==========
        print("\n[Test 5] After Take Profit Hit, Trailing Stop Triggers on Price Drop")
        # Buy price 10.0, ATR=0.5, stop_loss=9.0, take_profit=11.5
        # With trailing_pct=0.05 (5%), trailing_stop = 11.5 * 0.95 = 10.925
        # When current_price=10.9: take_profit_hit=False (10.9<11.5), profit_pct=0.09>=0.05, trailing_stop=10.9<=10.925=True
        result = StopLossStrategies.check_exit(
            position, current_price=10.9, atr=0.5, highest_price=11.5,
            stop_loss_mult=2.0, take_profit_mult=3.0, trailing_pct=0.05, trailing_start=0.05
        )
        assert result['action'] == 'trailing_stop', f"Should trigger trailing stop, got: {result['action']}"
        print(f"  Trailing stop triggered after profit >= 5% - PASS")

        # ========== Test 6: Profit >= 5% Without Take Profit Hit, Trailing Stop Active ==========
        print("\n[Test 6] Profit >= 5%, Trailing Stop Active")
        # Buy price 9.3, current 9.8 (~5.4% profit), highest 10.6
        # stop_loss = 9.3 - 2.0*0.5 = 8.3
        # With trailing_pct=0.05: trailing_stop = 10.6 * 0.95 = 10.07
        # Price 9.8 > 8.3 (no stop loss), 9.8 <= 10.07 (trailing stop triggers)
        position_6 = {'buy_price': 9.3}
        result = StopLossStrategies.check_exit(
            position_6, current_price=9.8, atr=0.5, highest_price=10.6,
            stop_loss_mult=2.0, take_profit_mult=3.0, trailing_pct=0.05, trailing_start=0.05
        )
        assert result['action'] == 'trailing_stop', f"Should trigger trailing stop, got: {result['action']}"
        print(f"  Profit >= 5% trailing stop - PASS")

        # ========== Test 7: Profit < 5% and No Take Profit Hit, Trailing Stop Inactive ==========
        print("\n[Test 7] Profit < 5% and No Take Profit Hit, Trailing Stop Inactive")
        # Buy price 10.0, current 10.3 (3% < 5%), highest 10.3
        # trailing_stop = 10.3 * 0.9 = 9.27
        # Price 9.5 > 9.0 (no stop loss), 9.5 > 9.27 (no trailing stop)
        result = StopLossStrategies.check_exit(
            position, current_price=9.5, atr=0.5, highest_price=10.3,
            stop_loss_mult=2.0, take_profit_mult=3.0, trailing_pct=0.10, trailing_start=0.05
        )
        assert result['action'] is None, f"Profit<5% and no take profit, should hold, got: {result['action']}"
        assert result['take_profit_hit'] == False, "Should not mark as take profit hit"
        print(f"  Profit < 5% trailing stop inactive - PASS")

        # ========== Test 8: Normal Holding ==========
        print("\n[Test 8] Normal Holding")
        result = StopLossStrategies.check_exit(
            position, current_price=10.5, atr=atr, highest_price=11.0
        )
        assert result['action'] is None, "Normal holding action should be None"
        assert result['reason'] == '正常持仓', "Normal holding reason should match"
        print(f"  Normal holding - PASS")

        # ========== Test 9: Stop Loss / Take Profit Formula Verification ==========
        print("\n[Test 9] Stop Loss / Take Profit Formula Verification")
        test_buy = 20.0
        test_atr = 0.5
        assert StopLossStrategies.calculate_stop_loss(test_buy, test_atr, 2.0) == 19.0, "Stop loss formula error"
        assert StopLossStrategies.calculate_take_profit(test_buy, test_atr, 3.0) == 21.5, "Take profit formula error"
        print(f"  Stop/Take profit formula - PASS")

        # ========== Test 10: Trailing Stop Formula Verification ==========
        print("\n[Test 10] Trailing Stop Formula Verification")
        assert StopLossStrategies.calculate_trailing_stop(100.0, 0.10) == 90.0, "Trailing stop formula error"
        assert StopLossStrategies.calculate_trailing_stop(50.0, 0.20) == 40.0, "Trailing stop formula error"
        print(f"  Trailing stop formula - PASS")

        print("\n" + "=" * 50)
        print("All 10 unit tests passed!")

    run_tests()
