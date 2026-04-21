import pytest
from back_testing.analysis.performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    """绩效分析器测试"""

    def test_total_return_single_profitable_trade(self):
        """测试单笔盈利交易的总收益率"""
        # 100000 initial capital, sell with 25% return
        # profit = 25000, total_return = 25000/100000 = 0.25
        trades = [{'action': 'sell', 'return': 0.25, 'shares': 100, 'price': 125}]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert abs(metrics['total_return'] - 0.25) < 0.001

    def test_total_return_single_losing_trade(self):
        """测试单笔亏损交易的总收益率"""
        trades = [{'action': 'sell', 'return': -0.10, 'shares': 100, 'price': 90}]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert abs(metrics['total_return'] - (-0.10)) < 0.001

    def test_total_return_multiple_trades(self):
        """测试多笔交易的总收益率"""
        # Trade 1: 100000 -> 100000 * (1 + 0.10) = 110000
        # Trade 2: 110000 -> 110000 * (1 + 0.15) = 126500
        # Trade 3: 126500 -> 126500 * (1 - 0.05) = 120175
        # total_return = 120175/100000 - 1 = 0.20175
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.15, 'shares': 100, 'price': 126.5},
            {'action': 'sell', 'return': -0.05, 'shares': 100, 'price': 120.175},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert abs(metrics['total_return'] - 0.20175) < 0.001

    def test_empty_trades(self):
        """测试空交易列表"""
        analyzer = PerformanceAnalyzer([], initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert metrics['total_return'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['sortino_ratio'] == 0.0
        assert metrics['max_drawdown'] == 0.0

    def test_annual_return_calculation(self):
        """测试年化收益率计算"""
        # Simulate 1 year with 20% return
        trades = [{'action': 'sell', 'return': 0.20, 'shares': 100, 'price': 120}]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        # Assuming 1 trade cycle = 1 year
        assert abs(metrics['annual_return'] - 0.20) < 0.001

    def test_max_drawdown_calculation(self):
        """测试最大回撤计算"""
        # Simulate equity curve: 100000 -> 120000 -> 90000 -> 110000
        # Drawdowns: 0 -> 0 -> (120000-90000)/120000=25% -> 0
        # Max drawdown = 25%
        trades = [
            {'action': 'sell', 'return': 0.20, 'shares': 100, 'price': 120},
            {'action': 'sell', 'return': -0.25, 'shares': 100, 'price': 90},
            {'action': 'sell', 'return': 0.2222, 'shares': 100, 'price': 110},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert abs(metrics['max_drawdown'] - 0.25) < 0.01

    def test_max_drawdown_no_drawdown(self):
        """测试无回撤情况"""
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.15, 'shares': 100, 'price': 126.5},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert metrics['max_drawdown'] == 0.0

    def test_win_rate_all_wins(self):
        """测试全胜率的胜率"""
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.20, 'shares': 100, 'price': 120},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert metrics['win_rate'] == 1.0

    def test_win_rate_all_losses(self):
        """测试全失败的胜率"""
        trades = [
            {'action': 'sell', 'return': -0.10, 'shares': 100, 'price': 90},
            {'action': 'sell', 'return': -0.05, 'shares': 100, 'price': 95},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert metrics['win_rate'] == 0.0

    def test_win_rate_mixed(self):
        """测试混合交易的胜率"""
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': -0.05, 'shares': 100, 'price': 95},
            {'action': 'sell', 'return': 0.15, 'shares': 100, 'price': 115},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert metrics['win_rate'] == pytest.approx(2/3)

    def test_win_rate_empty_trades(self):
        """测试空交易的胜率"""
        analyzer = PerformanceAnalyzer([], initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert metrics['win_rate'] == 0.0

    def test_profit_loss_ratio_profitable(self):
        """测试盈亏比计算（盈利情况）"""
        # wins: 10%, 20% avg=15%; losses: -5% only
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.20, 'shares': 100, 'price': 120},
            {'action': 'sell', 'return': -0.05, 'shares': 100, 'price': 95},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        # avg_win = 0.15, avg_loss = 0.05, pl_ratio = 0.15/0.05 = 3.0
        assert abs(metrics['profit_loss_ratio'] - 3.0) < 0.01

    def test_profit_loss_ratio_no_losses(self):
        """测试无亏损交易的盈亏比"""
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.20, 'shares': 100, 'price': 120},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        # No losses, should return infinity or a large number
        import math
        assert math.isinf(metrics['profit_loss_ratio']) or metrics['profit_loss_ratio'] > 1000

    def test_profit_loss_ratio_no_wins(self):
        """测试无盈利交易的盈亏比"""
        trades = [
            {'action': 'sell', 'return': -0.10, 'shares': 100, 'price': 90},
            {'action': 'sell', 'return': -0.05, 'shares': 100, 'price': 95},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        # No wins, should return 0
        assert metrics['profit_loss_ratio'] == 0.0

    def test_sharpe_ratio_insufficient_data(self):
        """测试Sharpe比率数据不足情况"""
        # Need at least 2 trades to calculate volatility
        trades = [{'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110}]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert metrics['sharpe_ratio'] == 0.0

    def test_sharpe_ratio_calculation(self):
        """测试Sharpe比率计算"""
        # With returns [0.10, 0.15, 0.05], mean = 0.10, std ≈ 0.05
        # sharpe = (0.10 - 0.025) / 0.05 ≈ 1.5
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.15, 'shares': 100, 'price': 115},
            {'action': 'sell', 'return': 0.05, 'shares': 100, 'price': 105},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000, risk_free_rate=0.025)
        metrics = analyzer.calculate_metrics()
        # Sharpe should be positive and reasonable
        assert 1.0 < metrics['sharpe_ratio'] < 2.0

    def test_sortino_ratio_calculation(self):
        """测试Sortino比率计算"""
        # Returns [0.10, -0.05, 0.15], mean = 0.0667
        # downside returns: [-0.05], downside_std = 0.05
        # sortino = (0.0667 - 0.025) / 0.05 ≈ 0.83
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': -0.05, 'shares': 100, 'price': 95},
            {'action': 'sell', 'return': 0.15, 'shares': 100, 'price': 115},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000, risk_free_rate=0.025)
        metrics = analyzer.calculate_metrics()
        # Sortino should be positive and less than sharpe (since only downside deviations count)
        assert 0.5 < metrics['sortino_ratio'] < 1.5

    def test_sortino_ratio_insufficient_data(self):
        """测试Sortino比率数据不足情况"""
        trades = [{'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110}]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        assert metrics['sortino_ratio'] == 0.0

    def test_sortino_ratio_all_positive_returns(self):
        """测试Sortino无下行波动情况"""
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.15, 'shares': 100, 'price': 115},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000, risk_free_rate=0.025)
        metrics = analyzer.calculate_metrics()
        # No downside, should return infinity or very large
        import math
        assert math.isinf(metrics['sortino_ratio']) or metrics['sortino_ratio'] > 100

    def test_calmar_ratio_calculation(self):
        """测试Calmar比率计算"""
        # 3 trades, overall return ~20%, max_drawdown ~25%
        # calmar = 0.20 / 0.25 = 0.8
        trades = [
            {'action': 'sell', 'return': 0.20, 'shares': 100, 'price': 120},
            {'action': 'sell', 'return': -0.25, 'shares': 100, 'price': 90},
            {'action': 'sell', 'return': 0.3333, 'shares': 100, 'price': 120},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        # Calmar ratio should be around 0.8
        assert 0.7 < metrics['calmar_ratio'] < 0.9

    def test_calmar_ratio_zero_drawdown(self):
        """测试Calmar比率零回撤情况"""
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 121},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        # Zero drawdown, calmar should be infinity
        import math
        assert math.isinf(metrics['calmar_ratio']) or metrics['calmar_ratio'] > 1000

    def test_default_initial_capital(self):
        """测试默认初始资金"""
        trades = [{'action': 'sell', 'return': 0.50, 'shares': 100, 'price': 150}]
        analyzer = PerformanceAnalyzer(trades)
        metrics = analyzer.calculate_metrics()
        # Default capital is 1000000, 50% return
        assert abs(metrics['total_return'] - 0.50) < 0.001

    def test_default_benchmark_index(self):
        """测试默认基准指数"""
        trades = [{'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110}]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        # Should not raise any errors
        assert analyzer.benchmark_index == 'sh000001'

    def test_custom_risk_free_rate(self):
        """测试自定义无风险利率"""
        trades = [
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 121},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000, risk_free_rate=0.03)
        metrics = analyzer.calculate_metrics()
        # Sharpe should use custom risk free rate
        assert metrics['annual_return'] == pytest.approx(0.10)

    def test_buy_action_not_counted(self):
        """测试买入操作不计入收益"""
        trades = [
            {'action': 'buy', 'price': 100, 'shares': 100},
            {'action': 'sell', 'return': 0.10, 'shares': 100, 'price': 110},
        ]
        analyzer = PerformanceAnalyzer(trades, initial_capital=100000)
        metrics = analyzer.calculate_metrics()
        # Only sell trades count
        assert abs(metrics['total_return'] - 0.10) < 0.001
