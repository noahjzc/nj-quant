# This is a sample Python script.
from datetime import datetime
from data_processor.price_calculator import *

import pandas as pd
from pandas import DataFrame

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def print_log(content):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{datetime.now()} => {content}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    base_path = r"D:\workspace\code\mine\quant\data\metadata\daily_ycz"
    stock_code = "sh688188"

    file_path = f"{base_path}\\{stock_code}.csv"

    df: DataFrame = pd.read_csv(file_path, encoding="gbk", parse_dates=[TRADE_DATE], index_col=[TRADE_DATE])

    # df.set_index(TRADE_DATE, inplace=True)
    df.sort_index(inplace=True, ascending=True)

    # print(df[CLOSE_PRICE].shift(1))
    df = PriceCalculator.fill_previous_close_price(df)



    # print(df[[PREVIOUS_CLOSE_PRICE, CLOSE_PRICE, PREVIOUS_POST_ADJUSTED_CLOSE_PRICE, POST_ADJUSTED_CLOSE_PRICE,
    #           PREVIOUS_PRE_ADJUSTED_CLOSE_PRICE, PRE_ADJUSTED_CLOSE_PRICE]])
    # df[PREVIOUS_CLOSE_PRICE] =
    # df = PriceCalculator.calculate_change(df)
    # df = PriceCalculator.calculate_adjusted_price(df)
    # df = PriceCalculator(df)
    # print(type(df))
    # print(df)

    # save_file_path = rf"D:\workspace\code\mine\quant\data\cleaned_data\daily\{stock_code}.csv"
    # df.to_csv(save_file_path)
    # print_log(file_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
