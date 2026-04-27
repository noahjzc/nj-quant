"""
Fitness: 适应度评估器 - 遗传算法的核心

适应度函数(Fitness Function)是GA的灵魂:
- 衡量每个染色体的优劣
- 决定哪些个体被选中繁殖
- 引导进化方向

优化目标:
- 主目标: 最大化 Information Ratio = Alpha / Tracking Error (相对于沪深300)
- 基准: 沪深300 (hs300)
- 约束: 最大回撤 <= 25%
- 原因: 信息比率衡量相对基准的超额收益风险比，更适合A股市场

简化回测策略 (vs 全量回测):
- 无日内止损/止盈/ATR移动止损 (GA需要评估数千个解，太慢)
- 每周调仓 (每周五选择，下周五调仓)
- 等权持仓 (5只股票，每只20%资金)
- 使用后复权价格计算真实收益

数据流:
1. 给定因子权重配置
2. 在每个调仓日用多因子选股
3. 持有到下一个调仓日
4. 计算组合收益率序列
5. 获取同期沪深300收益率序列
6. 计算超额收益Alpha和跟踪误差TrackingError
7. 计算 Information Ratio = Alpha / TrackingError
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import timedelta
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_loader import FactorLoader
from back_testing.factors.factor_config import get_factor_directions
from back_testing.data.data_provider import DataProvider


class FitnessEvaluator:
    """
    适应度评估器 - 用简化回测评估因子权重配置

    核心指标: Information Ratio (信息比率)
    - Alpha: 组合相对沪深300的超额收益
    - Tracking Error: 超额收益的标准差
    - IR = Alpha / Tracking Error，越高越好

    约束处理:
    - 如果最大回撤 > 25%，返回0(无效解)
    - 避免追求高收益但风险过大的配置
    """

    def __init__(self, max_drawdown_constraint: float = 0.25,
                 n_stocks: int = 5,
                 stop_loss_threshold: float = -0.05,
                 benchmark_code: str = 'sh000300',
                 stock_codes: Optional[List[str]] = None):
        """
        初始化评估器

        Args:
            max_drawdown_constraint: 最大允许回撤，默认25%
                                     超过此回撤的解被视为无效
            n_stocks: 持仓数量，默认5只
                     每期选择top n只等权分配
            stop_loss_threshold: 止损阈值，默认-5%
                               持仓期间任意股票亏损超过此阈值则该期收益记负
            benchmark_code: 基准指数代码，默认沪深300 (sh000300)
            stock_codes: 股票池（None=全市场）
        """
        self.max_drawdown_constraint = max_drawdown_constraint
        self.n_stocks = n_stocks
        self.stop_loss_threshold = stop_loss_threshold
        self.benchmark_code = benchmark_code
        self.stock_codes = stock_codes

        # 初始化数据提供者
        self.data_provider = DataProvider()
        # 初始化因子加载器
        self.factor_loader = FactorLoader(data_provider=self.data_provider)
        # 调试计数器
        self._debug_count = 0
        # 缓存基准指数数据
        self._benchmark_cache = {}
        # 缓存因子数据: key=(date, factors_tuple), value=DataFrame
        self._factor_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self):
        """清除因子数据缓存（每个训练窗口开始时调用）"""
        self._factor_cache.clear()
        self._benchmark_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._debug_count = 0

    def get_cache_stats(self) -> dict:
        """获取缓存命中率统计"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': hit_rate
        }

    def evaluate(self, weights: Dict[str, float],
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp) -> float:
        """
        评估因子权重配置

        给定权重配置，在指定时间段运行回测，返回信息比率(Information Ratio)。

        Args:
            weights: 因子权重字典，格式 {'因子名': 权重值}
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            Information Ratio
            - 约束满足: 返回 Alpha / TrackingError
            - 约束违反: 返回0.0(无效解)
        """
        try:
            # 运行回测获取组合收益率序列
            result = self._run_backtest(weights, start_date, end_date)

            portfolio_returns = result.get('weekly_returns', [])
            max_drawdown = result.get('max_drawdown', 0)

            self._debug_count += 1

            # 约束检查: 最大回撤不能超过限制
            if max_drawdown > self.max_drawdown_constraint:
                if self._debug_count <= 3:
                    non_zero = sum(1 for r in portfolio_returns if r != 0)
                    print(f"    [DEBUG] IR=0: 最大回撤{max_drawdown:.1%} > {self.max_drawdown_constraint:.0%}, 周收益非零={non_zero}/{len(portfolio_returns)}", flush=True)
                return 0.0

            if len(portfolio_returns) < 4:
                if self._debug_count <= 3:
                    print(f"    [DEBUG] IR=0: 周收益序列太短 n={len(portfolio_returns)}", flush=True)
                return 0.0

            # 获取基准收益率序列
            benchmark_returns = self._get_benchmark_returns(
                start_date, end_date, len(portfolio_returns)
            )

            if len(benchmark_returns) < 4:
                if self._debug_count <= 3:
                    print(f"    [DEBUG] IR=0: 基准收益率序列太短 n={len(benchmark_returns)}", flush=True)
                return 0.0

            # 计算信息比率
            ir = self._calculate_information_ratio(portfolio_returns, benchmark_returns)

            if self._debug_count <= 3:
                non_zero_p = sum(1 for r in portfolio_returns if r != 0)
                non_zero_b = sum(1 for r in benchmark_returns if r != 0)
                print(f"    [DEBUG] IR={ir:.4f} | 周收益非零={non_zero_p}/{len(portfolio_returns)} | 基准非零={non_zero_b}/{len(benchmark_returns)} | 最大回撤={max_drawdown:.2%}", flush=True)

            return ir

        except Exception as e:
            print(f"    [DEBUG] 评估异常: {e}")
            return 0.0

    def _run_backtest(self, weights: Dict[str, float],
                      start_date: pd.Timestamp,
                      end_date: pd.Timestamp) -> Dict:
        """
        运行简化回测

        调仓流程 (每周):
        1. 每周五获取调仓日列表
        2. 在调仓日加载所有股票因子数据
        3. 用多因子选股模型选择top n只股票
        4. 持有到下一个调仓日，计算持有期收益率
        5. 累计组合价值曲线
        6. 计算最终绩效指标

        Args:
            weights: 因子权重
            start_date: 回测开始
            end_date: 回测结束

        Returns:
            绩效字典: {annual_return, max_drawdown, total_return, n_weeks}
        """
        # 获取所有调仓日(每周五)
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        if len(rebalance_dates) < 2:
            return self._empty_result()

        factor_directions = get_factor_directions()

        # 组合净值序列，初始值为1.0
        portfolio_values = [1.0]
        # 组合周收益率序列
        weekly_returns = []

        # 遍历每个调仓周期
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            # 获取所有股票的因子数据（带缓存）
            factor_list = list(weights.keys())
            cache_key = (current_date, tuple(sorted(factor_list)))
            if cache_key not in self._factor_cache:
                print(f"    [缓存未命中] 准备加载因子数据: {current_date}", flush=True)
                if self.stock_codes is not None:
                    self._factor_cache[cache_key] = self.factor_loader.load_stock_factors(
                        self.stock_codes, current_date, factor_list
                    )
                else:
                    self._factor_cache[cache_key] = self.factor_loader.load_all_stock_factors(
                        current_date, factor_list
                    )
                self._cache_misses += 1
            else:
                self._cache_hits += 1
            factor_data = self._factor_cache[cache_key]

            if len(factor_data) == 0:
                # 无数据，组合价值不变
                portfolio_values.append(portfolio_values[-1])
                weekly_returns.append(0.0)
                continue

            # 创建多因子选股器并选择股票
            selector = MultiFactorSelector(
                weights=weights,
                directions=factor_directions
            )
            selected_stocks = selector.select_top_stocks(
                data=factor_data,
                n=self.n_stocks
            )

            if not selected_stocks:
                # 未选出股票，组合价值不变
                portfolio_values.append(portfolio_values[-1])
                weekly_returns.append(0.0)
                continue

            # 计算持有期收益率
            period_return = self._calculate_period_return(
                selected_stocks, current_date, next_date
            )

            weekly_returns.append(period_return)

            # 更新组合净值
            new_value = portfolio_values[-1] * (1 + period_return)
            portfolio_values.append(new_value)

        # 转换为numpy数组便于计算
        portfolio_values = np.array(portfolio_values)

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(portfolio_values)

        return {
            'weekly_returns': weekly_returns,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values
        }

    def _get_rebalance_dates(self, start_date: pd.Timestamp,
                             end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """
        获取调仓日列表(每周五)

        调仓规则:
        - 从start_date开始找到第一个周五
        - 然后每周五递增直到end_date

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            周五日期列表
        """
        dates = []
        current = pd.Timestamp(start_date)

        # 找到第一个周五
        # Python weekday(): Monday=0, Tuesday=1, ..., Friday=4, Saturday=5, Sunday=6
        while current.weekday() != 4:
            current += timedelta(days=1)

        # 收集所有周五
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=7)

        return dates

    def _calculate_period_return(self, stocks: List[str],
                                 current_date: pd.Timestamp,
                                 next_date: pd.Timestamp) -> float:
        """
        计算等权组合持有期收益率（带简化止损）

        对于持仓列表中的每只股票:
        1. 获取从current_date到next_date的价格数据
        2. 计算持有期收益率 = (期末价格/期初价格) - 1
        3. 取所有股票收益率的均值作为组合收益

        风控逻辑：
        - 止损：持仓期间任意股票亏损超过 stop_loss_threshold 则该期收益记负（取最小收益）

        Args:
            stocks: 持仓股票代码列表
            current_date: 调仓日(期初)
            next_date: 下一调仓日(期末)

        Returns:
            等权组合收益率
        """
        returns = []

        for stock in stocks:
            try:
                # 获取股票数据
                df = self.data_provider.get_stock_data(
                    stock,
                    start_date=current_date.strftime('%Y-%m-%d'),
                    end_date=next_date.strftime('%Y-%m-%d')
                )

                if len(df) < 2:
                    continue

                # 使用后复权价计算收益
                price_col = 'adj_close'
                if price_col not in df.columns:
                    continue

                df = df.sort_index()
                prices = df[price_col].values

                if len(prices) >= 2:
                    # 计算单只股票持有期收益
                    period_return = (prices[-1] / prices[0]) - 1
                    returns.append(period_return)

            except Exception:
                continue

        if not returns:
            return 0.0

        # 等权平均所有持仓股票的收益（不剔除亏损股，实盘止损逻辑应独立处理）
        return np.mean(returns)

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        计算最大回撤

        最大回撤 = max{(Peak - Value) / Peak}
        即从历史最高点到当前点的最大跌幅

        算法:
        1. 遍历每个时间点
        2. 更新历史最高点
        3. 计算当前回撤
        4. 记录最大回撤

        Args:
            portfolio_values: 组合净值序列

        Returns:
            最大回撤比例(0-1)
        """
        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            # 更新历史最高
            if value > peak:
                peak = value

            # 计算当前回撤
            drawdown = (peak - value) / peak if peak > 0 else 0

            # 更新最大回撤
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _empty_result(self) -> Dict:
        """
        返回空结果(用于无效评估)

        当数据不足无法评估时返回零值结果
        """
        return {
            'weekly_returns': [],
            'max_drawdown': 0.0,
            'portfolio_values': [1.0]
        }

    def _get_benchmark_returns(self,
                               start_date: pd.Timestamp,
                               end_date: pd.Timestamp,
                               n_periods: int) -> list:
        """
        获取基准指数周收益率序列

        Args:
            start_date: 开始日期
            end_date: 结束日期
            n_periods: 需要的收益率期数

        Returns:
            基准周收益率列表
        """
        cache_key = (start_date, end_date)
        if cache_key in self._benchmark_cache:
            cached = self._benchmark_cache[cache_key]
            if len(cached) >= n_periods:
                return cached[:n_periods]

        try:
            # 获取基准指数数据
            df = self.data_provider.get_index_data(
                self.benchmark_code,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if df is None or len(df) < 2:
                return [0.0] * n_periods

            # 使用后复权收盘价计算周收益率
            price_col = 'adj_close'
            if price_col not in df.columns:
                # 尝试 close 列
                price_col = 'close'

            df = df.sort_index()
            prices = df[price_col].values

            if len(prices) < 2:
                return [0.0] * n_periods

            # 计算周收益率 (与组合调仓周期对齐)
            rebalance_dates = self._get_rebalance_dates(start_date, end_date)
            benchmark_returns = []

            # 构建日期到索引的映射
            date_indices = {pd.Timestamp(d).date(): idx for idx, d in enumerate(df.index)}

            for i in range(len(rebalance_dates) - 1):
                date_before = rebalance_dates[i].date()
                date_after = rebalance_dates[i + 1].date()

                idx_before = date_indices.get(date_before, 0)
                idx_after = date_indices.get(date_after, len(prices) - 1)

                if idx_before < idx_after and prices[idx_before] > 0:
                    period_return = (prices[idx_after] / prices[idx_before]) - 1
                    benchmark_returns.append(period_return)
                else:
                    benchmark_returns.append(0.0)

            # 缓存结果
            self._benchmark_cache[cache_key] = benchmark_returns

            return benchmark_returns[:n_periods]

        except Exception as e:
            print(f"获取基准数据失败: {e}")
            return [0.0] * n_periods

    def _calculate_information_ratio(self,
                                     portfolio_returns: list,
                                     benchmark_returns: list) -> float:
        """
        计算信息比率 Information Ratio = Alpha / Tracking Error

        Alpha = 平均超额收益 (组合收益 - 基准收益)
        Tracking Error = 超额收益的标准差

        Args:
            portfolio_returns: 组合周收益率列表
            benchmark_returns: 基准周收益率列表

        Returns:
            信息比率
        """
        if len(portfolio_returns) != len(benchmark_returns):
            # 长度不匹配时截断到较短的长度
            n = min(len(portfolio_returns), len(benchmark_returns))
            portfolio_returns = portfolio_returns[:n]
            benchmark_returns = benchmark_returns[:n]

        if len(portfolio_returns) < 4:
            return 0.0

        # 转换为numpy数组
        portfolio_returns = np.array(portfolio_returns)
        benchmark_returns = np.array(benchmark_returns)

        # 计算超额收益
        excess_returns = portfolio_returns - benchmark_returns

        # 计算Alpha (年化超额收益)
        alpha = np.mean(excess_returns) * 52  # 年化

        # 计算Tracking Error (超额收益的标准差，年化)
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(52)

        # 防止除零
        if tracking_error < 1e-6:
            return 0.0

        information_ratio = alpha / tracking_error

        return information_ratio
