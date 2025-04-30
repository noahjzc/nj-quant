from data_column_names import *
import pandas as pd
from pandas import DataFrame


class PriceCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate_change(dataframe: DataFrame):
        # dataframe[CHANGE_AMOUNT] = dataframe[CLOSE_PRICE] - dataframe[PREVIOUS_CLOSE_PRICE]
        dataframe[CHANGE] = round(dataframe[CLOSE_PRICE] / dataframe[PREVIOUS_CLOSE_PRICE] - 1, 4)

        return dataframe

    @staticmethod
    def fill_previous_close_price(df: DataFrame):
        """
        填充上一个交易日的收盘价
        :param df:
        :return:
        """
        df[PREVIOUS_CLOSE_PRICE] = df[CLOSE_PRICE].shift(1)
        df[PREVIOUS_POST_ADJUSTED_CLOSE_PRICE] = df[POST_ADJUSTED_CLOSE_PRICE].shift(1)
        df[PREVIOUS_PRE_ADJUSTED_CLOSE_PRICE] = df[PRE_ADJUSTED_CLOSE_PRICE].shift(1)
        return df

    @staticmethod
    def calculate_adjusted_price(df: DataFrame):


        return df
