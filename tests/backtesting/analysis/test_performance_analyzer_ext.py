import pytest
import numpy as np
from backtesting.analysis.performance_analyzer import PerformanceAnalyzer


class TestNewMetrics:
    """新增指标测试"""

    def setup_method(self):
        n = 252
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n)
        equity = 1_000_000 * np.cumprod(1 + returns)
        equity = np.insert(equity, 0, 1_000_000)

        benchmark_returns = np.random.normal(0.0005, 0.015, n)

        self.analyzer = PerformanceAnalyzer(
            trades=[],
            initial_capital=1_000_000,
            equity_curve=equity.tolist(),
            periods_per_year=252,
            benchmark_returns=benchmark_returns,
        )
        self.analyzer.calculate_metrics()

    def test_daily_returns_exposed(self):
        """验证 daily_returns 属性已暴露"""
        assert hasattr(self.analyzer, 'daily_returns')
        assert self.analyzer.daily_returns is not None
        assert len(self.analyzer.daily_returns) == 252

    def test_information_ratio(self):
        """验证信息比率计算"""
        ir = self.analyzer.information_ratio()
        assert isinstance(ir, float)
        # IR should be a finite number
        assert np.isfinite(ir)

    def test_alpha_beta(self):
        """验证 alpha/beta 回归"""
        ab = self.analyzer.alpha_beta()
        assert 'alpha' in ab
        assert 'beta' in ab
        assert 'r_squared' in ab
        assert isinstance(ab['alpha'], float)
        assert isinstance(ab['beta'], float)
        assert isinstance(ab['r_squared'], float)

    def test_alpha_beta_synthetic(self):
        """合成数据验证: 策略收益 = 0.002 + 1.5 * 基准收益"""
        np.random.seed(99)
        benchmark = np.random.normal(0.0005, 0.015, 252)
        noise = np.random.normal(0, 0.005, 252)
        strategy = 0.002 + 1.5 * benchmark + noise
        equity = 1_000_000 * np.cumprod(1 + strategy)
        equity = np.insert(equity, 0, 1_000_000)

        analyzer = PerformanceAnalyzer(
            trades=[],
            initial_capital=1_000_000,
            equity_curve=equity.tolist(),
            periods_per_year=252,
            benchmark_returns=benchmark,
        )
        analyzer.calculate_metrics()
        ab = analyzer.alpha_beta()

        # beta should be close to 1.5
        assert 1.0 < ab['beta'] < 2.0, f"Expected beta ~1.5, got {ab['beta']}"
        # alpha should be close to 0.002
        assert 0.001 < ab['alpha'] < 0.003, f"Expected alpha ~0.002, got {ab['alpha']}"

    def test_skewness_kurtosis(self):
        """验证偏度和峰度"""
        sk = self.analyzer.skewness_kurtosis()
        assert 'skewness' in sk
        assert 'kurtosis' in sk
        assert isinstance(sk['skewness'], float)
        assert isinstance(sk['kurtosis'], float)

    def test_rolling_sharpe(self):
        """验证滚动 Sharpe"""
        rs = self.analyzer.rolling_sharpe(window=60)
        assert len(rs) > 0
        assert len(rs) == 252 - 60 + 1
        # All values should be finite
        assert np.all(np.isfinite(rs))

    def test_rolling_sharpe_short_data(self):
        """数据不足时应返回空数组"""
        analyzer = PerformanceAnalyzer(
            trades=[],
            equity_curve=[1_000_000, 1_001_000],
            periods_per_year=252,
        )
        analyzer.calculate_metrics()
        rs = analyzer.rolling_sharpe(window=60)
        assert len(rs) == 0

    def test_monthly_returns(self):
        """验证月度收益"""
        mr = self.analyzer.monthly_returns()
        assert isinstance(mr, dict)
        # 252 days / 21 = 12 months
        assert len(mr) == 12
        assert all(isinstance(v, float) for v in mr.values())

    def test_monthly_returns_empty(self):
        """空数据时返回空字典"""
        analyzer = PerformanceAnalyzer(trades=[], equity_curve=[1_000_000], periods_per_year=252)
        analyzer.calculate_metrics()
        mr = analyzer.monthly_returns()
        assert mr == {}

    def test_max_drawdown_duration(self):
        """验证最大回撤持续天数"""
        dur = self.analyzer.max_drawdown_duration()
        assert dur >= 0
        assert isinstance(dur, int)

    def test_max_drawdown_duration_synthetic(self):
        """合成数据验证回撤持续天数"""
        # 创建已知回撤序列: 峰值在 index 5 (1.05), index 9 仍在峰值之下
        # 因此回撤从 index 6 持续到 index 9, 共 4 天
        equity = np.array([
            1.00, 1.01, 1.02, 1.03, 1.04, 1.05,  # 0-5: 上涨
            1.04, 1.03, 1.02,  # 6-8: 持续下跌
            1.04, 1.06,  # 9: 仍低于峰值, 10: 创新高恢复
        ])
        analyzer = PerformanceAnalyzer(
            trades=[],
            initial_capital=1.0,
            equity_curve=equity.tolist(),
            periods_per_year=252,
        )
        analyzer.calculate_metrics()
        dur = analyzer.max_drawdown_duration()
        assert dur == 4, f"Expected 4, got {dur}"

    def test_max_drawdown_duration_no_equity(self):
        """无净值曲线时返回 0"""
        analyzer = PerformanceAnalyzer(trades=[], initial_capital=1_000_000)
        analyzer.calculate_metrics()
        dur = analyzer.max_drawdown_duration()
        assert dur == 0

    def test_no_benchmark_graceful(self):
        """无基准收益时，基准相关指标应返回默认值"""
        analyzer = PerformanceAnalyzer(
            trades=[],
            initial_capital=1_000_000,
            equity_curve=[1_000_000, 1_010_000, 1_020_000],
            periods_per_year=252,
        )
        analyzer.calculate_metrics()
        assert analyzer.information_ratio() == 0.0
        ab = analyzer.alpha_beta()
        assert ab['alpha'] == 0.0
        assert ab['beta'] == 0.0
        assert ab['r_squared'] == 0.0

    def test_daily_returns_set_in_equity_path(self):
        """验证 daily_returns 在 equity_curve 分支中被赋值"""
        analyzer = PerformanceAnalyzer(
            trades=[],
            equity_curve=[1_000_000, 1_010_000, 1_020_000, 1_015_000],
            periods_per_year=252,
        )
        analyzer.calculate_metrics()
        assert analyzer.daily_returns is not None
        assert len(analyzer.daily_returns) == 3
        # period returns: (1010000-1000000)/1000000 = 0.01, etc.
        np.testing.assert_allclose(
            analyzer.daily_returns,
            [0.01, 0.00990099, -0.00490196],
            rtol=1e-3,
        )

    def test_daily_returns_none_without_equity(self):
        """无 equity_curve 时 daily_returns 为 None"""
        analyzer = PerformanceAnalyzer(
            trades=[{'action': 'sell', 'return': 0.1, 'shares': 100, 'price': 110}],
            initial_capital=1_000_000,
        )
        analyzer.calculate_metrics()
        assert analyzer.daily_returns is None
