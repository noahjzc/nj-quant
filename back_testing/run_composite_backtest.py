"""
多策略综合评分量化选股系统 - 回测入口
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from back_testing.composite_rotator import CompositeRotator

DATA_PATH = r'D:\workspace\code\mine\quant\data\metadata\daily_ycz'
INITIAL_CAPITAL = 1000000.0
N_STOCKS = 5


def get_trading_fridays(start_date: str, end_date: str) -> list:
    """获取回测区间内所有周五"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    fridays = []
    current = start
    while current.weekday() != 4:
        current += timedelta(days=1)

    while current <= end:
        fridays.append(current)
        current += timedelta(days=7)

    return fridays


def load_stock_price(stock_code: str, date: pd.Timestamp, data_path: str) -> float:
    """加载指定日期的收盘价"""
    file_path = f"{data_path}\\{stock_code}.csv"
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        df = df.sort_values('交易日期')
        hist = df[df['交易日期'] <= date]
        if len(hist) == 0:
            return None
        return hist.iloc[-1]['后复权价']
    except Exception:
        return None


def load_stock_prices(stock_codes: list, date: pd.Timestamp, data_path: str) -> dict:
    """批量加载多只股票的价格"""
    prices = {}
    for code in stock_codes:
        price = load_stock_price(code, date, data_path)
        if price:
            prices[code] = price
    return prices


def run_backtest(start_date: str, end_date: str, initial_capital: float = INITIAL_CAPITAL):
    """运行综合评分策略回测"""
    print("=" * 60)
    print("多策略综合评分量化选股系统 - 回测")
    print("=" * 60)
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"持仓数量: {N_STOCKS}")
    print("=" * 60)

    rotator = CompositeRotator(
        data_path=DATA_PATH,
        initial_capital=initial_capital,
        n_stocks=N_STOCKS
    )

    fridays = get_trading_fridays(start_date, end_date)
    print(f"\n调仓日数量: {len(fridays)}")

    portfolio_value = initial_capital
    weekly_results = []

    for i, friday in enumerate(fridays):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{len(fridays)} 周: {friday.strftime('%Y-%m-%d')} (周五)")
        print("=" * 60)

        # 获取持仓股票价格
        prices = load_stock_prices(rotator.current_stocks, friday, DATA_PATH) if rotator.current_stocks else {}

        # 执行每周流程
        result = rotator.run_weekly(friday, prices)

        # 计算持仓价值
        stock_values = {}
        for code in rotator.current_stocks:
            price = prices.get(code)
            if price:
                shares = int(portfolio_value / N_STOCKS / price)
                stock_values[code] = shares * price
            else:
                stock_values[code] = 0

        portfolio_value = sum(stock_values.values())

        weekly_results.append({
            'date': friday,
            'stocks': result['stocks'],
            'portfolio_value': portfolio_value,
            'return': (portfolio_value - initial_capital) / initial_capital
        })

        print(f"\n持仓股票: {result['stocks']}")
        print(f"使用策略: {result['strategy']}")
        print(f"组合净值: {portfolio_value:,.2f}")
        print(f"累计收益: {(portfolio_value - initial_capital) / initial_capital:.2%}")

    # 汇总结果
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)

    df_weeks = pd.DataFrame(weekly_results)
    total_return = (portfolio_value - initial_capital) / initial_capital

    days = (fridays[-1] - fridays[0]).days if len(fridays) > 1 else 1
    years = days / 365
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    df_weeks['peak'] = df_weeks['portfolio_value'].cummax()
    df_weeks['drawdown'] = (df_weeks['peak'] - df_weeks['portfolio_value']) / df_weeks['peak']
    max_drawdown = df_weeks['drawdown'].max()

    print(f"回测区间: {fridays[0].strftime('%Y-%m-%d')} ~ {fridays[-1].strftime('%Y-%m-%d')}")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"调仓次数: {len(fridays)}")

    output_path = f"back_testing/results/composite_{start_date}_{end_date}.csv"
    df_weeks.to_csv(output_path, index=False)
    print(f"\n周报已保存: {output_path}")

    return df_weeks


def main():
    parser = argparse.ArgumentParser(description='多策略综合评分量化选股系统')
    parser.add_argument('--start', default='2020-01-01', help='回测开始日期')
    parser.add_argument('--end', default='2025-04-18', help='回测结束日期')
    parser.add_argument('--capital', type=float, default=1000000.0, help='初始资金')
    args = parser.parse_args()

    run_backtest(args.start, args.end, args.capital)


if __name__ == '__main__':
    main()