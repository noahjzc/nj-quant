from pandas import DataFrame

from strategy.abstract_strategy import AbstractStrategy


class FirstStrategy(AbstractStrategy):

    def fill_factor(self, data: DataFrame):
        pass
