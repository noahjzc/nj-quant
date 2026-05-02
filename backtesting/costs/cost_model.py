from dataclasses import dataclass
import math


@dataclass
class CostBreakdown:
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    brokerage: float = 0.0
    slippage: float = 0.0
    impact: float = 0.0

    @property
    def total(self) -> float:
        return self.stamp_duty + self.transfer_fee + self.brokerage + self.slippage + self.impact


class CostModel:
    """统一交易成本模型：费率 + 滑点 + 平方根冲击"""

    STAMP_DUTY = 0.001
    TRANSFER_FEE = 0.00002
    BROKERAGE = 0.0003
    MIN_BROKERAGE = 5.0

    def __init__(self, slippage_bps: float = 1.0, impact_model: str = 'sqrt'):
        self.slippage_bps = slippage_bps
        self.impact_model = impact_model

    def buy_cost(self, price: float, shares: int,
                 amount_today: float = None, volatility: float = None) -> CostBreakdown:
        return self._calc_cost(price, shares, is_buy=True,
                               amount_today=amount_today, volatility=volatility)

    def sell_cost(self, price: float, shares: int,
                  amount_today: float = None, volatility: float = None) -> CostBreakdown:
        return self._calc_cost(price, shares, is_buy=False,
                               amount_today=amount_today, volatility=volatility)

    def _calc_cost(self, price: float, shares: int, is_buy: bool,
                   amount_today: float = None, volatility: float = None) -> CostBreakdown:
        value = price * shares

        stamp = 0.0 if is_buy else value * self.STAMP_DUTY
        transfer = value * self.TRANSFER_FEE
        brokerage = max(value * self.BROKERAGE, self.MIN_BROKERAGE)
        slippage = value * (self.slippage_bps / 10000.0)
        impact = self._calc_impact(value, amount_today, volatility)

        return CostBreakdown(
            stamp_duty=stamp,
            transfer_fee=transfer,
            brokerage=brokerage,
            slippage=slippage,
            impact=impact,
        )

    def _calc_impact(self, order_value: float, amount_today: float = None,
                     volatility: float = None) -> float:
        if self.impact_model == 'none':
            return 0.0
        if self.impact_model == 'fixed':
            return order_value * 0.0005
        if amount_today is None or amount_today <= 0:
            return 0.0
        if volatility is None:
            volatility = 0.02

        # sqrt model
        q_v = order_value / amount_today
        return volatility * math.sqrt(q_v) * order_value
