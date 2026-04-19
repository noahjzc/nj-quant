"""
策略轮动量化选股系统 - 回测入口
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from back_testing.portfolio_rotator import PortfolioRotator

DATA_PATH = r'D:\workspace\code\mine\quant\data\metadata\daily_ycz'
INITIAL_CAPITAL = 1000000.0  # 100万
N_STOCKS = 5
N_WEEKS = 4


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


def load_stock_price(stock_code: str, date: pd.Timestamp, data_path: str) -> float:
    """加载指定日期的收盘价"""
    file_path = f"{data_path}\\{stock_code}.csv"
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        df = df.sort_values('交易日期')

        # 找当天或之前的最近交易日
        hist = df[df['交易日期'] <= date]
        if len(hist) == 0:
            return None
        return hist.iloc[-1]['后复权价']
    except Exception:
        return None


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
        data_path=DATA_PATH,
        initial_capital=initial_capital,
        n_stocks=N_STOCKS,
        n_weeks=N_WEEKS
    )

    # 获取所有调仓日
    fridays = get_trading_fridays(start_date, end_date)
    print(f"\n调仓日数量: {len(fridays)}")

    current_capital = initial_capital
    portfolio_value = initial_capital
    portfolio_history = []

    weekly_results = []

    for i, friday in enumerate(fridays):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{len(fridays)} 周: {friday.strftime('%Y-%m-%d')} (周五)")
        print("=" * 60)

        # 执行每周流程
        result = rotator.run_weekly(friday)

        # 计算持仓价值
        stock_values = {}
        for code in rotator.current_stocks:
            price = load_stock_price(code, friday, DATA_PATH)
            if price:
                shares = int(portfolio_value / N_STOCKS / price)
                stock_values[code] = shares * price
            else:
                stock_values[code] = 0

        # 估算组合净值
        portfolio_value = sum(stock_values.values()) + current_capital * 0.1  # 预留现金

        weekly_results.append({
            'date': friday,
            'strategy': result['strategy'],
            'stocks': result['stocks'],
            'portfolio_value': portfolio_value,
            'return': (portfolio_value - initial_capital) / initial_capital
        })

        # 打印周报
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

    # 计算年化收益率
    days = (fridays[-1] - fridays[0]).days if len(fridays) > 1 else 1
    years = days / 365
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 计算最大回撤
    df_weeks['peak'] = df_weeks['portfolio_value'].cummax()
    df_weeks['drawdown'] = (df_weeks['peak'] - df_weeks['portfolio_value']) / df_weeks['peak']
    max_drawdown = df_weeks['drawdown'].max()

    print(f"回测区间: {fridays[0].strftime('%Y-%m-%d')} ~ {fridays[-1].strftime('%Y-%m-%d')}")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
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