"""
Fitness: 适应度评估器 - 遗传算法的核心

适应度函数(Fitness Function)是GA的灵魂:
- 衡量每个染色体的优劣
- 决定哪些个体被选中繁殖
- 引导进化方向

优化目标:
- 主目标: 最大化 Calmar比率 = 年化收益率 / 最大回撤
- 约束: 最大回撤 <= 20%
- 原因: Calmar比率同时考虑了收益和风险，是衡量策略质量的好指标

简化回测策略 (vs 全量回测):
- 无日内止损/止盈/ATR移动止损 (GA需要评估数千个解，太慢)
- 每周调仓 (每周五选择，下周五调仓)
- 等权持仓 (5只股票，每只20%资金)
- 使用后复权价格计算真实收益

数据流:
1. 给定因子权重配置
2. 在每个调仓日用多因子选股
3. 持有到下一个调仓日
4. 计算组合收益率曲线
5. 计算Calmar比率
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import timedelta
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_loader import FactorLoader
from back_testing.factors.factor_config import get_factor_directions
from back_testing.data.data_provider import DataProvider


class FitnessEvaluator:
    """
    适应度评估器 - 用简化回测评估因子权重配置

    核心指标: Calmar比率
    - 年化收益率: 衡量盈利能力
    - 最大回撤: 衡量风险
    - Calmar = 年化收益 / 最大回撤，越高越好

    约束处理:
    - 如果最大回撤 > 20%，返回0(无效解)
    - 避免追求高收益但风险过大的配置
    """

    def __init__(self, max_drawdown_constraint: float = 0.20,
                 n_stocks: int = 5,
                 stop_loss_threshold: float = -0.05):
        """
        初始化评估器

        Args:
            max_drawdown_constraint: 最大允许回撤，默认20%
                                     超过此回撤的解被视为无效
            n_stocks: 持仓数量，默认5只
                     每期选择top n只等权分配
            stop_loss_threshold: 止损阈值，默认-5%
                               持仓期间任意股票亏损超过此阈值则该期收益记负
        """
        self.max_drawdown_constraint = max_drawdown_constraint
        self.n_stocks = n_stocks
        self.stop_loss_threshold = stop_loss_threshold

        # 初始化数据提供者
        self.data_provider = DataProvider()
        # 初始化因子加载器
        self.factor_loader = FactorLoader(data_provider=self.data_provider)

    def evaluate(self, weights: Dict[str, float],
                start_date: pd.Timestamp,
                end_date: pd.Timestamp) -> float:
        """
        评估因子权重配置

        给定权重配置，在指定时间段运行回测，返回Calmar比率。

        Args:
            weights: 因子权重字典，格式 {'因子名': 权重值}
            start_date: 回测开始日期
            end_date: 回测结束日期

        Returns:
            Calmar比率
            - 约束满足: 返回年化收益/最大回撤
            - 约束违反: 返回0.0(无效解)
        """
        try:
            # 运行回测
            result = self._run_backtest(weights, start_date, end_date)

            annual_return = result.get('annual_return', 0)
            max_drawdown = result.get('max_drawdown', 0)

            # 约束检查: 最大回撤不能超过限制
            if max_drawdown > self.max_drawdown_constraint:
                return 0.0

            # 计算Calmar比率，防止除零
            # 假设最小回撤为1%，避免Calmar变成无穷大
            calmar = annual_return / max(max_drawdown, 0.01)

            return calmar

        except Exception as e:
            print(f"评估失败: {e}")
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

        # 遍历每个调仓周期
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            # 获取所有股票的因子数据
            factor_list = list(weights.keys())
            factor_data = self.factor_loader.load_all_stock_factors(
                current_date, factor_list
            )

            if len(factor_data) == 0:
                # 无数据，组合价值不变
                portfolio_values.append(portfolio_values[-1])
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
                continue

            # 计算持有期收益率
            period_return = self._calculate_period_return(
                selected_stocks, current_date, next_date
            )

            # 更新组合净值
            new_value = portfolio_values[-1] * (1 + period_return)
            portfolio_values.append(new_value)

        # 转换为numpy数组便于计算
        portfolio_values = np.array(portfolio_values)

        # 计算总收益率
        total_return = portfolio_values[-1] / portfolio_values[0] - 1

        # 计算年化收益率
        # 假设每年约52周
        n_weeks = len(portfolio_values) - 1
        if n_weeks > 0:
            annual_return = (1 + total_return) ** (52 / n_weeks) - 1
        else:
            annual_return = 0

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(portfolio_values)

        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'n_weeks': n_weeks
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

        # 检查止损：任意股票亏损超过阈值则该期收益记负
        min_return = min(returns)
        if min_return <= self.stop_loss_threshold:
            return min_return

        # 返回等权平均收益
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
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'n_weeks': 0
        }
