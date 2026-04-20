import pytest
from back_testing.position_manager import PositionManager


class TestPositionManager:
    """仓位管理器测试"""

    def test_init_default_values(self):
        """测试默认参数初始化"""
        pm = PositionManager(total_capital=100000)
        assert pm.total_capital == 100000
        assert pm.max_position_pct == 0.20
        assert pm.max_total_pct == 0.90

    def test_init_custom_values(self):
        """测试自定义参数初始化"""
        pm = PositionManager(total_capital=500000, max_position_pct=0.15, max_total_pct=0.80)
        assert pm.total_capital == 500000
        assert pm.max_position_pct == 0.15
        assert pm.max_total_pct == 0.80

    def test_get_position_value_empty(self):
        """测试空持仓的市值计算"""
        pm = PositionManager(total_capital=100000)
        value = pm.get_position_value({}, {'sh600519': 100.0})
        assert value == 0.0

    def test_get_position_value_with_positions(self):
        """测试有持仓的市值计算"""
        pm = PositionManager(total_capital=100000)
        positions = {'sh600519': 1000, 'sz000001': 500}
        prices = {'sh600519': 180.0, 'sz000001': 12.0}
        # 1000 * 180 + 500 * 12 = 180000 + 6000 = 186000
        value = pm.get_position_value(positions, prices)
        assert value == 186000.0

    def test_get_available_capital(self):
        """测试可用资金计算"""
        pm = PositionManager(total_capital=100000)
        positions = {'sh600519': 1000}
        prices = {'sh600519': 50.0}  # 市值 50000
        available = pm.get_available_capital(positions, prices)
        assert available == 50000.0

    def test_can_buy_no_positions(self):
        """测试无持仓时可以买入"""
        pm = PositionManager(total_capital=100000)
        assert pm.can_buy('sh600519', 100.0, {}, None) is True

    def test_can_buy_zero_price(self):
        """测试零价格不能买入"""
        pm = PositionManager(total_capital=100000)
        assert pm.can_buy('sh600519', 0.0, {}, None) is False

    def test_can_buy_negative_price(self):
        """测试负价格不能买入"""
        pm = PositionManager(total_capital=100000)
        assert pm.can_buy('sh600519', -10.0, {}, None) is False

    def test_can_buy_exceeds_total_limit(self):
        """测试超过总仓位上限（90%）不能买入"""
        pm = PositionManager(total_capital=100000)
        # 已有持仓 95000市值（超过90%）
        positions = {'sz000001': 1000}
        prices = {'sz000001': 95.0}
        # 即使只有100股也要95*100=9500，总仓位会超过90%
        assert pm.can_buy('sh600519', 100.0, positions, prices) is False

    def test_can_buy_exceeds_single_limit(self):
        """测试超过单只股票上限（20%）不能买入"""
        pm = PositionManager(total_capital=100000)
        # 已有持仓 21000市值（超过20%单只上限20000）
        positions = {'sz000001': 1000}
        prices = {'sz000001': 21.0}
        # 注意：20%限制是针对单只股票的。
        # sz000001已有21%是它自己的问题，不影响买其他股票
        # 只要买sh600519不超过20%就可以买
        # 买100股需要2100，现有0 + 2100 = 2100 < 20000，可以买
        assert pm.can_buy('sh600519', 21.0, positions, prices) is True
        # 但如果买sh600519会超过它自己的20%上限，就不能买
        assert pm.can_buy('sh600519', 201.0, positions, prices) is False

    def test_can_buy_within_limits(self):
        """测试在限制范围内可以买入"""
        pm = PositionManager(total_capital=100000)
        # 已有持仓 5000市值（5%）
        positions = {'sz000001': 500}
        prices = {'sz000001': 10.0}
        # 买入100股需要1000，不会超过任何限制
        assert pm.can_buy('sh600519', 10.0, positions, prices) is True

    def test_can_buy_same_stock_accumulates(self):
        """测试同一只股票累加持仓"""
        pm = PositionManager(total_capital=100000)
        # 已持有 sh600519 1000股@50 = 50000市值（50%，超过20%单只上限）
        positions = {'sh600519': 1000}
        prices = {'sh600519': 50.0}
        # 不能再买这只股票
        assert pm.can_buy('sh600519', 50.0, positions, prices) is False

        # 但可以买其他股票
        assert pm.can_buy('sz000001', 10.0, positions, prices) is True

    def test_calculate_buy_shares_empty(self):
        """测试空仓时计算买入股数"""
        pm = PositionManager(total_capital=100000)
        # 可用资金 = min(90000, 20000) = 20000
        # 买入股数 = floor(20000 / 100 / 100) * 100 = 200
        shares = pm.calculate_buy_shares('sh600519', 100.0, {}, None)
        assert shares == 200

    def test_calculate_buy_shares_limited_by_single_position(self):
        """测试被单只上限限制的买入股数"""
        pm = PositionManager(total_capital=100000)
        # 总仓位90%=90000，但单只上限20%=20000
        # 可用资金 = min(90000 - 0, 20000) = 20000
        # 买入股数 = floor(20000 / 80 / 100) * 100 = 200
        shares = pm.calculate_buy_shares('sh600519', 80.0, {}, None)
        assert shares == 200  # 200 * 80 = 16000，正好不超过20000

    def test_calculate_buy_shares_limited_by_total_position(self):
        """测试被总仓位上限限制的买入股数"""
        pm = PositionManager(total_capital=100000)
        # 已有持仓70000市值
        positions = {'sz000001': 1000}
        prices = {'sz000001': 70.0}
        # 已用70000，剩余可用 = min(90000-70000=20000, 20000) = 20000
        # 买入股数 = floor(20000 / 50 / 100) * 100 = 400
        shares = pm.calculate_buy_shares('sh600519', 50.0, positions, prices)
        assert shares == 400  # 400 * 50 = 20000

    def test_calculate_buy_shares_no_capacity(self):
        """测试没有可用资金时返回0"""
        pm = PositionManager(total_capital=100000)
        # 已有持仓 95000市值（超过90%）
        positions = {'sz000001': 1000}
        prices = {'sz000001': 95.0}
        shares = pm.calculate_buy_shares('sh600519', 100.0, positions, prices)
        assert shares == 0

    def test_calculate_buy_shares_zero_price(self):
        """测试零价格返回0"""
        pm = PositionManager(total_capital=100000)
        shares = pm.calculate_buy_shares('sh600519', 0.0, {}, None)
        assert shares == 0

    def test_calculate_buy_shares_round_to_lot(self):
        """测试买入股数取整为100的倍数"""
        pm = PositionManager(total_capital=100000)
        # 可用资金 = 20000
        # floor(20000 / 33.33 / 100) = floor(6.0) = 6
        # 6 * 100 = 600
        shares = pm.calculate_buy_shares('sh600519', 33.33, {}, None)
        assert shares == 600

    def test_calculate_buy_shares_without_prices(self):
        """测试没有价格信息时保守返回0"""
        pm = PositionManager(total_capital=100000)
        # 已有持仓但没有价格，无法计算已用资金
        positions = {'sz000001': 1000}
        # 没有prices传入
        shares = pm.calculate_buy_shares('sh600519', 100.0, positions, None)
        # 没有价格信息时，已用资金按0计算
        # 可用资金 = min(90000, 20000) = 20000
        # floor(20000 / 100 / 100) * 100 = 200
        assert shares == 200

    def test_multiple_positions_total_limit(self):
        """测试多只股票总仓位限制"""
        pm = PositionManager(total_capital=100000)
        # 已有两只股票各持仓
        positions = {'sh600519': 400, 'sz000001': 400}
        prices = {'sh600519': 100.0, 'sz000001': 100.0}
        # 已用 400*100 + 400*100 = 80000
        # 可用 = min(90000-80000=10000, 20000) = 10000
        shares = pm.calculate_buy_shares('sh000001', 50.0, positions, prices)
        # floor(10000 / 50 / 100) * 100 = 200
        assert shares == 200

    def test_edge_case_price_at_limit(self):
        """测试价格刚好触及限制边界"""
        pm = PositionManager(total_capital=100000)
        # 单只上限20000，每股价格200，买入100股需要20000
        # 这是边界情况，应该允许
        shares = pm.calculate_buy_shares('sh600519', 200.0, {}, None)
        # floor(20000 / 200 / 100) * 100 = floor(1.0) * 100 = 100
        assert shares == 100

    def test_edge_case_price_above_limit(self):
        """测试价格略高于限制"""
        pm = PositionManager(total_capital=100000)
        # 单只上限20000，每股价格201，买入100股需要20100
        # 这会超过限制，应该被拒绝
        # can_buy会检查：existing_value + current_price*100 > max_single_position
        # 0 + 201*100 = 20100 > 20000，应该返回False
        assert pm.can_buy('sh600519', 201.0, {}, None) is False


class TestPositionManagerIntegrations:
    """仓位管理器集成测试"""

    def test_full_trading_scenario(self):
        """测试完整交易场景"""
        pm = PositionManager(total_capital=100000)

        # 初始状态
        positions = {}
        prices = {}

        # 第一次买入 sh600519
        assert pm.can_buy('sh600519', 100.0, positions, prices) is True
        shares1 = pm.calculate_buy_shares('sh600519', 100.0, positions, prices)
        assert shares1 == 200  # 使用20000资金

        # 更新持仓
        positions['sh600519'] = shares1
        prices['sh600519'] = 100.0

        # 第二次买入 sz000001
        assert pm.can_buy('sz000001', 50.0, positions, prices) is True
        shares2 = pm.calculate_buy_shares('sz000001', 50.0, positions, prices)
        # 已用20000，可用=min(90000-20000=70000, 20000)=20000
        # floor(20000/50/100)*100 = 400
        assert shares2 == 400

        # 更新持仓
        positions['sz000001'] = shares2
        prices['sz000001'] = 50.0

        # 第三次买入 sh600519（加仓）
        # 此时sh600519已有200股市值20000（20%），已达上限，不能再加
        assert pm.can_buy('sh600519', 100.0, positions, prices) is False
        shares3 = pm.calculate_buy_shares('sh600519', 100.0, positions, prices)
        assert shares3 == 0

        # sz000001也已到上限20000（20%），也不能再加
        assert pm.can_buy('sz000001', 50.0, positions, prices) is False

        # 总持仓市值验证
        total_value = pm.get_position_value(positions, prices)
        assert total_value == 200 * 100 + 400 * 50  # 20000 + 20000 = 40000

        # 可用资金验证
        available = pm.get_available_capital(positions, prices)
        assert available == 100000 - 40000  # 60000
