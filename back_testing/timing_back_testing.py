from pandas import DataFrame



class StockBackTesting:
    """
    择时


    """

    data: DataFrame = None

    def __init__(self, stock_data: DataFrame):
        self.data = stock_data


