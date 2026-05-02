"""交易执行器 — 买卖操作和成本计算"""
import math
from dataclasses import dataclass
from typing import Optional

from backtesting.costs.cost_model import CostModel


@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    stock_code: str
    action: str          # 'BUY' or 'SELL'
    price: float
    shares: int
    cost: float          # 手续费+印花税+过户费
    capital_before: float  # 交易前现金（便于追踪资金变化）


class TradeExecutor:
    """
    交易执行器

    成本设置（沿用 BacktestEngine）：
    - 印花税：0.1%（卖出时收取）
    - 过户费：0.002%（买卖都收取）
    - 券商佣金：0.03%，最低5元
    """

    STAMP_DUTY = 0.001
    TRANSFER_FEE = 0.00002
    BROKERAGE = 0.0003
    MIN_BROKERAGE = 5.0

    def __init__(self, cost_model=None):
        self.cost_model = cost_model or CostModel()

    def execute_buy(self, stock_code: str, price: float, cash: float) -> tuple[int, float]:
        """
        模拟买入

        Args:
            stock_code: 股票代码
            price: 买入价格
            cash: 可用资金

        Returns:
            (买入股数, 手续费总额)
        """
        if price <= 0 or cash <= 0:
            return 0, 0.0

        # 按可用资金计算最大可买股数
        max_shares = math.floor(cash / price / 100) * 100
        if max_shares == 0:
            return 0, 0.0

        buy_value = max_shares * price
        cost_breakdown = self.cost_model.buy_cost(price, max_shares)
        total_cost = cost_breakdown.total
        actual_cost = total_cost + buy_value

        if actual_cost > cash:
            # 钱不够，降低股数
            available = cash - total_cost
            max_shares = math.floor(available / price / 100) * 100
            if max_shares == 0:
                return 0, 0.0
            buy_value = max_shares * price
            cost_breakdown = self.cost_model.buy_cost(price, max_shares)
            total_cost = cost_breakdown.total

        return max_shares, total_cost

    def execute_sell(self, stock_code: str, price: float, shares: int) -> tuple[int, float]:
        """
        模拟卖出

        Args:
            stock_code: 股票代码
            price: 卖出价格
            shares: 卖出股数

        Returns:
            (实际卖出股数, 手续费总额)
        """
        if price <= 0 or shares <= 0:
            return 0, 0.0

        cost_breakdown = self.cost_model.sell_cost(price, shares)
        total_cost = cost_breakdown.total

        return shares, total_cost