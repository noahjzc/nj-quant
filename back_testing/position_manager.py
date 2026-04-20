"""仓位管理器 - 负责持仓控制和资金分配"""

import math
from typing import Dict, Optional


class PositionManager:
    """仓位管理器"""

    def __init__(self, total_capital: float, max_position_pct: float = 0.20,
                 max_total_pct: float = 0.90):
        """
        Args:
            total_capital: 总资金
            max_position_pct: 单只股票最大占比，默认20%
            max_total_pct: 总仓位上限，默认90%
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_total_pct = max_total_pct

    def calculate_buy_shares(self, stock_code: str, current_price: float,
                            existing_positions: Dict[str, int],
                            prices: Optional[Dict[str, float]] = None) -> int:
        """
        计算买入股数

        Args:
            stock_code: 股票代码
            current_price: 当前价格
            existing_positions: 已持仓字典 {stock_code: shares}
            prices: 当前持仓的市值单价字典 {stock_code: price}，用于计算已用资金

        Returns:
            int: 买入股数（整手，100的倍数）
        """
        if current_price <= 0:
            return 0

        if not self.can_buy(stock_code, current_price, existing_positions, prices):
            return 0

        # 已用资金
        used_capital = self._calculate_used_capital(existing_positions, prices)

        # 可用资金 = min(总资金 × 90% - 已用资金, 总资金 × 20%)
        max_total_position = self.total_capital * self.max_total_pct
        available_by_total = max_total_position - used_capital

        max_single_position = self.total_capital * self.max_position_pct

        available_capital = min(available_by_total, max_single_position)

        if available_capital <= 0:
            return 0

        # 买入股数 = floor(可用资金 / 当前价 / 100) × 100
        shares = math.floor(available_capital / current_price / 100) * 100

        return shares

    def can_buy(self, stock_code: str, current_price: float,
                existing_positions: Dict[str, int],
                prices: Optional[Dict[str, float]] = None) -> bool:
        """
        检查是否可以买入

        Args:
            stock_code: 股票代码
            current_price: 当前价格
            existing_positions: 已持仓字典 {stock_code: shares}
            prices: 当前持仓的市值单价字典 {stock_code: price}，用于计算已用资金

        Returns:
            bool: 是否可以买入
        """
        if current_price <= 0:
            return False

        # 已用资金
        used_capital = self._calculate_used_capital(existing_positions, prices)

        # 检查总仓位是否已达到上限
        max_total_position = self.total_capital * self.max_total_pct
        if used_capital >= max_total_position:
            return False

        # 检查单只股票是否已达到上限
        max_single_position = self.total_capital * self.max_position_pct

        # 该股票已用资金
        existing_shares = existing_positions.get(stock_code, 0)
        existing_value = existing_shares * (prices.get(stock_code, 0) if prices else 0)

        # 如果加上100股（最小买入单位）会超过单只上限，则不能买
        if existing_value + current_price * 100 > max_single_position:
            return False

        return True

    def get_position_value(self, positions: Dict[str, int], prices: Dict[str, float]) -> float:
        """计算持仓总市值"""
        total_value = 0.0
        for stock_code, shares in positions.items():
            if stock_code in prices and shares > 0:
                total_value += shares * prices[stock_code]
        return total_value

    def get_available_capital(self, positions: Dict[str, int], prices: Dict[str, float]) -> float:
        """计算可用资金"""
        used_capital = self.get_position_value(positions, prices)
        return self.total_capital - used_capital

    def _calculate_used_capital(self, positions: Dict[str, int],
                                prices: Optional[Dict[str, float]] = None) -> float:
        """
        计算已用资金

        Args:
            positions: 已持仓字典 {stock_code: shares}
            prices: 市价字典 {stock_code: price}

        Returns:
            float: 已用资金（持仓市值）
        """
        if not positions:
            return 0.0

        if prices is None:
            # 如果没有价格信息，假设已持仓为空（保守估计）
            return 0.0

        total = 0.0
        for stock_code, shares in positions.items():
            if shares > 0 and stock_code in prices:
                total += shares * prices[stock_code]
        return total