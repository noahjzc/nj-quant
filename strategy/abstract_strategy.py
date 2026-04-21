from pandas import DataFrame


class AbstractStrategy:
    def __init__(self):
        pass

    def fill_factor(self, data: DataFrame):
        pass