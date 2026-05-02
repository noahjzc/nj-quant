import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
from backtesting.costs.cost_model import CostModel, CostBreakdown


class TestCostModelDefaults:
    def setup_method(self):
        self.model = CostModel()

    def test_buy_cost_no_impact(self):
        cost = self.model.buy_cost(price=10.0, shares=10000)
        # buy_value = 100000
        # transfer_fee = 100000 * 0.00002 = 2.0
        # brokerage = max(100000 * 0.0003, 5.0) = 30.0
        # slippage = 100000 * 0.0001 = 10.0
        # impact = 0 (no volume info)
        # total = 2.0 + 30.0 + 10.0 = 42.0
        assert pytest.approx(cost.total, 0.01) == 42.0
        assert cost.stamp_duty == 0.0

    def test_sell_cost_with_stamp(self):
        cost = self.model.sell_cost(price=10.0, shares=10000)
        assert pytest.approx(cost.stamp_duty, 0.01) == 100.0
        assert cost.stamp_duty > 0

    def test_min_brokerage(self):
        cost = self.model.buy_cost(price=10.0, shares=100)
        assert cost.brokerage == 5.0

    def test_slippage_default(self):
        cost = self.model.buy_cost(price=10.0, shares=10000)
        expected_slippage = 100000 * 0.0001
        assert pytest.approx(cost.slippage, 0.01) == expected_slippage

    def test_slippage_disabled(self):
        model = CostModel(slippage_bps=0)
        cost = model.buy_cost(price=10.0, shares=10000)
        assert cost.slippage == 0.0


class TestCostModelImpact:
    def setup_method(self):
        self.model = CostModel(impact_model='sqrt')

    def test_sqrt_impact_zero_when_no_volume(self):
        cost = self.model.buy_cost(price=10.0, shares=10000)
        assert cost.impact == 0.0

    def test_sqrt_impact_small_order(self):
        # Small order (10k RMB) relative to large daily volume (1B) => impact < slippage
        cost = self.model.buy_cost(price=10.0, shares=1000, amount_today=1e9, volatility=0.02)
        assert cost.impact < cost.slippage

    def test_sqrt_impact_large_order(self):
        cost = self.model.buy_cost(price=10.0, shares=50000, amount_today=5e6, volatility=0.03)
        assert cost.impact > cost.slippage

    def test_impact_model_fixed(self):
        model = CostModel(impact_model='fixed')
        cost = model.buy_cost(price=10.0, shares=10000)
        # fixed: 100000 * 0.0005 = 50.0
        assert pytest.approx(cost.impact, 0.01) == 50.0

    def test_impact_model_none(self):
        model = CostModel(impact_model='none')
        cost = model.buy_cost(price=10.0, shares=50000, amount_today=5e6, volatility=0.03)
        assert cost.impact == 0.0
