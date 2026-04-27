"""
策略轮动量化选股系统 - 回测入口
"""
import sys
import io

# 设置控制台输出为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from back_testing.portfolio_rotator import PortfolioRotator
from back_testing.data.data_provider import DataProvider

# Parquet数据目录（已废弃，数据现在从DB读取）
DATA_PATH = None
INITIAL_CAPITAL = 1000000.0  # 100万
N_STOCKS = 5
N_WEEKS = 4

# 交易成本参数
STAMP_DUTY = 0.001       # 印花税：卖出时，0.1%
TRANSFER_FEE = 0.00002   # 过户费：买卖都收取，0.002%
BROKERAGE = 0.0003       # 券商佣金：买卖都收取，0.03%，最低5元


def calculate_transaction_cost(amount: float, is_buy: bool) -> float:
    """计算交易成本"""
    if is_buy:
        brokerage = max(amount * BROKERAGE, 5)
        transfer = amount * TRANSFER_FEE
        return brokerage + transfer
    else:
        brokerage = max(amount * BROKERAGE, 5)
        transfer = amount * TRANSFER_FEE
        stamp = amount * STAMP_DUTY
        return brokerage + transfer + stamp


def get_trading_fridays(start_date: str, end_date: str) -> list:
    """获取回测区间内所有周五（调仓日）"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    fridays = []
    current = start
    # 找到第一个周五
    while current.weekday() != 4:  # Monday=0, Friday=4
        current += timedelta(days=1)

    while current <= end:
        fridays.append(current)
        current += timedelta(days=7)

    return fridays


def load_stock_price(data_provider: DataProvider, stock_code: str, date: pd.Timestamp) -> float:
    """加载指定日期的收盘价"""
    return data_provider.get_stock_price(stock_code, date)


def load_stock_data_batch(data_provider: DataProvider, stock_codes: list, date: pd.Timestamp) -> dict:
    """批量加载多只股票在指定日期的价格"""
    prices = {}
    for code in stock_codes:
        price = data_provider.get_stock_price(code, date)
        if price:
            prices[code] = price
    return prices


def run_backtest(start_date: str, end_date: str, initial_capital: float = INITIAL_CAPITAL):
    """运行回测"""
    print("=" * 60)
    print("策略轮动量化选股系统 - 回测")
    print("=" * 60)
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"持仓数量: {N_STOCKS}")
    print(f"调仓周期: 每周")
    print("=" * 60)

    rotator = PortfolioRotator(
        initial_capital=initial_capital,
        n_stocks=N_STOCKS,
        n_weeks=N_WEEKS
    )

    # 创建数据提供器
    data_provider = DataProvider()

    # 获取所有调仓日
    fridays = get_trading_fridays(start_date, end_date)
    print(f"\n调仓日数量: {len(fridays)}")

    cash = initial_capital
    holdings = {}  # {code: {'shares': int, 'buy_price': float}}
    total_cost = 0  # 累计交易成本
    weekly_results = []

    for i, friday in enumerate(fridays):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{len(fridays)} 周: {friday.strftime('%Y-%m-%d')} (周五)")
        print("=" * 60)

        # 执行每周流程
        result = rotator.run_weekly(friday)
        new_stocks = result['stocks']

        # 获取当前持仓股票的收盘价
        prices = load_stock_data_batch(data_provider, new_stocks, friday)
        for code in new_stocks:
            if code not in prices:
                price = data_provider.get_stock_price(code, friday)
                if price:
                    prices[code] = price

        # 计算当前持仓市值
        position_value = 0
        for code, pos in holdings.items():
            price = prices.get(code, 0)
            if price and price > 0:
                pos_value = pos['shares'] * price
                position_value += pos_value
                print(f"  持仓: {code} {pos['shares']}股 @ {price:.2f} = {pos_value:.2f}")

        portfolio_value = cash + position_value

        # 调仓：卖出不在新持仓列表中的股票
        sell_stocks = [code for code in holdings if code not in new_stocks]
        for code in sell_stocks:
            price = prices.get(code) or data_provider.get_stock_price(code, friday)
            if code in holdings and price and price > 0:
                pos = holdings[code]
                shares = pos['shares']
                revenue = shares * price
                cost = calculate_transaction_cost(revenue, is_buy=False)
                cash += revenue - cost
                total_cost += cost
                profit_pct = (price - pos['buy_price']) / pos['buy_price']
                profit_str = f"+{profit_pct*100:.1f}%" if profit_pct >= 0 else f"{profit_pct*100:.1f}%"
                print(f"  卖出: {code} {shares}股 @ {price:.2f} ({profit_str}, 手续费: {cost:.2f})")
                del holdings[code]

        # 调仓：买入新持仓列表中的股票
        buy_stocks = [code for code in new_stocks if code not in holdings]
        if cash > 0 and buy_stocks:
            # 按等权分配资金
            capital_per_stock = cash / len(buy_stocks)
            for code in buy_stocks:
                price = prices.get(code) or data_provider.get_stock_price(code, friday)
                if price and price > 0:
                    shares = int(capital_per_stock / price / 100) * 100
                    if shares > 0:
                        cost = shares * price
                        buy_cost = calculate_transaction_cost(cost, is_buy=True)
                        if cost + buy_cost <= cash:
                            holdings[code] = {'shares': shares, 'buy_price': price}
                            cash -= (cost + buy_cost)
                            total_cost += buy_cost
                            print(f"  买入: {code} {shares}股 @ {price:.2f} (手续费: {buy_cost:.2f})")

        # 重新计算调仓后的组合净值
        final_value = cash
        for code, pos in holdings.items():
            price = prices.get(code) or data_provider.get_stock_price(code, friday)
            if price and price > 0:
                final_value += pos['shares'] * price

        weekly_results.append({
            'date': friday,
            'strategy': result['strategy'],
            'stocks': new_stocks,
            'portfolio_value': final_value,
            'cash': cash,
            'return': (final_value - initial_capital) / initial_capital
        })

        print(f"\n持仓股票: {new_stocks}")
        print(f"使用策略: {result['strategy']}")
        print(f"现金: {cash:,.2f}")
        print(f"组合净值: {final_value:,.2f}")
        print(f"累计收益: {(final_value - initial_capital) / initial_capital:.2%}")

    # 汇总结果
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)

    df_weeks = pd.DataFrame(weekly_results)
    final_value = df_weeks['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

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
    print(f"累计交易成本: {total_cost:,.2f}")
    print(f"调仓次数: {len(fridays)}")

    # 策略使用统计
    strategy_counts = df_weeks['strategy'].value_counts()
    print(f"\n策略使用统计:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}周 ({count/len(fridays):.1%})")

    # 保存结果
    output_path = f"back_testing/results/rotator_{start_date}_{end_date}.csv"
    df_weeks.to_csv(output_path, index=False)
    print(f"\n周报已保存: {output_path}")

    return df_weeks


def main():
    parser = argparse.ArgumentParser(description='策略轮动量化选股系统')
    parser.add_argument('--start', default='2020-01-01', help='回测开始日期')
    parser.add_argument('--end', default='2025-04-18', help='回测结束日期')
    parser.add_argument('--capital', type=float, default=1000000.0, help='初始资金')
    args = parser.parse_args()

    run_backtest(args.start, args.end, args.capital)


if __name__ == '__main__':
    main()