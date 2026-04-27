"""风险管理器 - 协调止损止盈和仓位管理组件"""

from typing import Dict, Optional

from back_testing.risk.stop_loss_strategies import StopLossStrategies
from back_testing.risk.position_manager import PositionManager


class RiskManager:
    """风险管理器"""

    def __init__(self, config: dict = None):
        """
        config = {
            'atr_period': 14,
            'stop_loss_mult': 2.0,
            'take_profit_mult': 3.0,
            'trailing_pct': 0.10,
            'trailing启动条件': 0.05,
            'max_position_pct': 0.20,
            'max_total_pct': 0.90,
            'total_capital': 100000.0,
        }
        """
        if config is None:
            config = {}

        self.atr_period = config.get('atr_period', 14)
        self.stop_loss_mult = config.get('stop_loss_mult', 2.0)
        self.take_profit_mult = config.get('take_profit_mult', 3.0)
        self.trailing_pct = config.get('trailing_pct', 0.10)
        self.trailing启动条件 = config.get('trailing启动条件', 0.05)
        self.max_position_pct = config.get('max_position_pct', 0.20)
        self.max_total_pct = config.get('max_total_pct', 0.90)
        self.total_capital = config.get('total_capital', 100000.0)

        # 初始化组件
        self.position_manager = PositionManager(
            total_capital=self.total_capital,
            max_position_pct=self.max_position_pct,
            max_total_pct=self.max_total_pct
        )

    def check_exit(self, position: dict, current_price: float, atr: float) -> dict:
        """
        检查是否触发止损/止盈/移动止损

        Args:
            position: 持仓信息字典，包含:
                - 'buy_price': 买入价格
                - 'highest_price': 持仓期间最高价（可选，默认为当前价）
            current_price: 当前价格
            atr: ATR值

        Returns:
            dict: {
                'action': 'stop_loss' | 'trailing_stop' | None,
                'price': 触发价格,
                'reason': 触发原因
            }
        """
        # 获取持仓期间最高价
        highest_price = position.get('highest_price', current_price)

        # 调用止损策略检查
        result = StopLossStrategies.check_exit(
            position=position,
            current_price=current_price,
            atr=atr,
            highest_price=highest_price,
            stop_loss_mult=self.stop_loss_mult,
            take_profit_mult=self.take_profit_mult,
            trailing_pct=self.trailing_pct,
            trailing_start=self.trailing启动条件
        )

        # 返回所有字段，包括take_profit_hit标志
        return result

    def calculate_position_size(self, total_capital: float, current_price: float,
                                existing_positions: float) -> int:
        """
        计算买入股数

        Args:
            total_capital: 总资金
            current_price: 当前价格
            existing_positions: 已有持仓股数（整手）

        Returns:
            int: 买入股数（整手，100的倍数）
        """
        if existing_positions <= 0:
            positions_dict = {}
            prices_dict = None
        else:
            # 将existing_positions作为同一只股票的持仓处理
            # 使用current_price估算已有持仓的市值
            positions_dict = {'POSITION': int(existing_positions)}
            prices_dict = {'POSITION': current_price}

        shares = self.position_manager.calculate_buy_shares(
            stock_code='POSITION',
            current_price=current_price,
            existing_positions=positions_dict,
            prices=prices_dict,
            total_capital=total_capital
        )

        return shares
