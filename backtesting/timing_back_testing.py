from pandas import DataFrame

from data_column_names import *
from strategy.abstract_strategy import AbstractStrategy


class TimingBackTesting:
    """
    择时回测
    1. 根据策略附加策略因子
    2. 设置因子买入、卖出阈值
    3. 标记“买入”信号
    4. 标记“持仓”信号
    5. 标记“卖出”信号
    6. 计算资金变化
    7. 输出结果


    """

    def __init__(self, stock_data: DataFrame, strategy: AbstractStrategy, buy_threshold, sell_threshold):
        self.data = stock_data
        self.strategy = strategy
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def run(self):
        self.data = self.strategy.fill_factor(self.data)

        self.make_signal()

        self.calculate_fund()
        

        return self.data


    def make_signal(self):
        condition1 = self.data[STRATEGY_FACTOR] >= self.buy_threshold
        condition2 = self.data[STRATEGY_FACTOR].shift(1) < self.sell_threshold
        self.data.loc[condition1 & condition2, TRADE_SIGNAL] = 1

        condition1 = self.data[STRATEGY_FACTOR] <= self.sell_threshold
        condition2 = self.data[STRATEGY_FACTOR].shift(1) > self.sell_threshold
        self.data.loc[condition1 & condition2, TRADE_SIGNAL] = 0


        pass


    def calculate_fund(self):
        pass