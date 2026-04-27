"""轮动策略持仓管理器 — 复用 PositionManager 整手计算逻辑"""
import math
from typing import Dict, Optional


class RotationPositionManager:
    """
    持仓管理器 — 每日轮动专用

    资金分配逻辑：
    - 总仓位上限 = 总资产 × max_total_pct
    - 单只上限 = 总资产 × max_position_pct
    - 买入股数 = floor(min(单只上限, 剩余可用) / 单价 / 100) × 100
    """

    def __init__(self, total_capital: float, max_total_pct: float = 0.90,
                 max_position_pct: float = 0.20):
        self.total_capital = total_capital
        self.max_total_pct = max_total_pct
        self.max_position_pct = max_position_pct

    def update_capital(self, total_capital: float):
        """更新总资产（随每日盈亏变化）"""
        self.total_capital = total_capital

    def calculate_buy_shares(
        self,
        stock_code: str,
        current_price: float,
        existing_positions: Dict[str, int],
        prices: Optional[Dict[str, float]] = None
    ) -> int:
        """
        计算买入股数（整手，100的倍数）
        """
        if current_price <= 0:
            return 0

        used_capital = self._calculate_used_capital(existing_positions, prices)
        max_total_position = self.total_capital * self.max_total_pct
        available_by_total = max_total_position - used_capital

        max_single_position = self.total_capital * self.max_position_pct
        available_capital = min(available_by_total, max_single_position)

        if available_capital <= 0:
            return 0

        shares = math.floor(available_capital / current_price / 100) * 100
        return shares

    def can_buy(
        self,
        stock_code: str,
        current_price: float,
        existing_positions: Dict[str, int],
        prices: Optional[Dict[str, float]] = None
    ) -> bool:
        """检查是否可以买入"""
        if current_price <= 0:
            return False

        used_capital = self._calculate_used_capital(existing_positions, prices)
        max_total_position = self.total_capital * self.max_total_pct
        if used_capital >= max_total_position:
            return False

        max_single_position = self.total_capital * self.max_position_pct
        existing_shares = existing_positions.get(stock_code, 0)
        existing_value = existing_shares * (prices.get(stock_code, 0) if prices else 0)
        if existing_value + current_price * 100 > max_single_position:
            return False

        return True

    def get_position_value(self, positions: Dict[str, int], prices: Dict[str, float]) -> float:
        """计算持仓总市值"""
        total = 0.0
        for code, shares in positions.items():
            if shares > 0 and code in prices:
                total += shares * prices[code]
        return total

    def get_available_capital(self, positions: Dict[str, int], prices: Dict[str, float]) -> float:
        """计算可用资金"""
        used = self.get_position_value(positions, prices)
        return self.total_capital * self.max_total_pct - used

    def _calculate_used_capital(
        self,
        positions: Dict[str, int],
        prices: Optional[Dict[str, float]] = None
    ) -> float:
        """计算已用资金"""
        if not positions or prices is None:
            return 0.0
        total = 0.0
        for code, shares in positions.items():
            if shares > 0 and code in prices:
                total += shares * prices[code]
        return total
