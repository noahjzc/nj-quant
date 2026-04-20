"""
多策略综合评分量化选股系统 - 回测入口
"""
import sys
import io
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from back_testing.composite_rotator import CompositeRotator
from back_testing.data_provider import DataProvider
from back_testing.risk_manager import RiskManager
from back_testing.stop_loss_strategies import StopLossStrategies

# Parquet数据目录
DATA_PATH = None  # 默认为 project_root/data/daily_ycz
INITIAL_CAPITAL = 1000000.0
N_STOCKS = 5

# Risk Manager configuration
RISK_CONFIG = {
    'atr_period': 14,
    'stop_loss_mult': 2.0,
    'take_profit_mult': 3.0,
    'trailing_pct': 0.10,
    'trailing启动条件': 0.05,
    'max_position_pct': 0.20,
    'max_total_pct': 0.90,
}


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


def load_stock_price(data_provider: DataProvider, stock_code: str, date: pd.Timestamp) -> float:
    """加载指定日期的收盘价"""
    return data_provider.get_stock_price(stock_code, date)


def load_stock_prices(data_provider: DataProvider, stock_codes: list, date: pd.Timestamp) -> dict:
    """批量加载多只股票的价格"""
    prices = {}
    for code in stock_codes:
        price = data_provider.get_stock_price(code, date)
        if price:
            prices[code] = price
    return prices


def calculate_transaction_cost(amount: float, is_buy: bool) -> float:
    """计算交易成本（印花税、佣金、过户费）"""
    STAMP_DUTY = 0.001     # 印花税：卖出时收取
    TRANSFER_FEE = 0.00002  # 过户费：买卖都收取
    BROKERAGE = 0.0003     # 券商佣金：买卖都收取，最低5元

    if is_buy:
        # 买入：券商佣金 + 过户费
        brokerage = max(amount * BROKERAGE, 5)
        transfer = amount * TRANSFER_FEE
        return brokerage + transfer
    else:
        # 卖出：券商佣金 + 过户费 + 印花税
        brokerage = max(amount * BROKERAGE, 5)
        transfer = amount * TRANSFER_FEE
        stamp = amount * STAMP_DUTY
        return brokerage + transfer + stamp


def get_stock_atr(data_provider: DataProvider, stock_code: str, date: pd.Timestamp, atr_period: int = 14) -> float:
    """
    获取指定日期的ATR值

    Args:
        data_provider: 数据提供器
        stock_code: 股票代码
        date: 日期
        atr_period: ATR周期，默认14

    Returns:
        float: ATR值，如果无法计算返回0.0
    """
    try:
        # 获取截至指定日期之前的数据（需要period+1条数据计算ATR）
        df = data_provider.get_stock_data(stock_code, date=date.strftime('%Y-%m-%d'))
        if len(df) < atr_period + 1:
            return 0.0

        atr = StopLossStrategies.calculate_atr(df, period=atr_period)
        return float(atr)
    except Exception:
        return 0.0


def check_daily_risk_exits(
    data_provider: DataProvider,
    risk_manager: RiskManager,
    holdings: dict,
    date: pd.Timestamp,
    atr_period: int = 14
) -> list:
    """
    检查持仓股票的每日止损/止盈/移动止损触发情况

    Args:
        data_provider: 数据提供器
        risk_manager: 风险管理器
        holdings: 持仓字典 {stock_code: {'shares': int, 'buy_price': float, 'highest_price': float}}
        date: 当前日期
        atr_period: ATR周期

    Returns:
        list: 需要卖出的股票列表，每个元素为 (stock_code, exit_reason)
    """
    sell_list = []

    for stock_code, position in list(holdings.items()):
        try:
            # 获取当前价格
            current_price = data_provider.get_stock_price(stock_code, date)
            if not current_price or current_price <= 0:
                continue

            # 获取ATR
            atr = get_stock_atr(data_provider, stock_code, date, atr_period)
            if atr <= 0:
                continue

            # 更新最高价
            highest_price = position.get('highest_price', position['buy_price'])
            if current_price > highest_price:
                highest_price = current_price
                position['highest_price'] = highest_price

            # 检查是否触发止损/止盈/移动止损
            exit_result = risk_manager.check_exit(
                position=position,
                current_price=current_price,
                atr=atr
            )

            if exit_result['action'] is not None:
                sell_list.append((stock_code, exit_result))
                print(f"    [风险止损] {stock_code} 触发{exit_result['action']}: {exit_result['reason']}")

        except Exception as e:
            # 如果出错，跳过这只股票（不强制卖出）
            print(f"    [警告] 检查 {stock_code} 风险时出错: {e}")
            continue

    return sell_list


def run_backtest(start_date: str, end_date: str, initial_capital: float = INITIAL_CAPITAL):
    """运行综合评分策略回测"""
    print("=" * 60)
    print("多策略综合评分量化选股系统 - 回测")
    print("=" * 60)
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"持仓数量: {N_STOCKS}")
    print("=" * 60)

    # 创建数据提供器
    data_provider = DataProvider(data_dir=DATA_PATH, use_parquet=True)

    # 初始化风险管理器
    risk_config = RISK_CONFIG.copy()
    risk_config['total_capital'] = initial_capital
    risk_manager = RiskManager(config=risk_config)

    rotator = CompositeRotator(
        data_path=DATA_PATH,
        initial_capital=initial_capital,
        n_stocks=N_STOCKS
    )

    fridays = get_trading_fridays(start_date, end_date)
    print(f"\n调仓日数量: {len(fridays)}")

    # 初始状态
    cash = initial_capital  # 现金
    holdings = {}  # 持仓 {stock_code: {'shares': int, 'buy_price': float, 'highest_price': float}}
    total_cost = 0  # 累计交易成本
    exit_records = []  # 记录止损/止盈触发的交易

    weekly_results = []

    for i, friday in enumerate(fridays):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{len(fridays)} 周: {friday.strftime('%Y-%m-%d')} (周五)")
        print("=" * 60)

        # 执行每日风险检查（从上周五到本周五之间）
        if i > 0:
            prev_friday = fridays[i - 1]

            # 检查每日持仓股票的风险触发
            daily_date = prev_friday + timedelta(days=1)
            while daily_date < friday:
                if holdings:
                    # 获取当日价格
                    for stock_code in list(holdings.keys()):
                        current_price = data_provider.get_stock_price(stock_code, daily_date)
                        if current_price and current_price > 0:
                            # 更新最高价
                            if 'highest_price' not in holdings[stock_code]:
                                holdings[stock_code]['highest_price'] = holdings[stock_code]['buy_price']
                            if current_price > holdings[stock_code]['highest_price']:
                                holdings[stock_code]['highest_price'] = current_price

                daily_date += timedelta(days=1)

            # 检查止损/止盈触发（使用上周五结算价）
            sell_list = check_daily_risk_exits(
                data_provider=data_provider,
                risk_manager=risk_manager,
                holdings=holdings,
                date=prev_friday,
                atr_period=risk_config['atr_period']
            )

            # 执行风险触发卖出
            for stock_code, exit_result in sell_list:
                if stock_code in holdings:
                    price = data_provider.get_stock_price(stock_code, prev_friday)
                    if price and price > 0:
                        shares = holdings[stock_code]['shares']
                        revenue = shares * price
                        cost = calculate_transaction_cost(revenue, is_buy=False)
                        cash += revenue - cost
                        total_cost += cost
                        exit_records.append({
                            'date': prev_friday,
                            'stock': stock_code,
                            'action': exit_result['action'],
                            'price': price,
                            'shares': shares,
                            'reason': exit_result['reason'],
                            'return': (price - holdings[stock_code]['buy_price']) / holdings[stock_code]['buy_price']
                        })
                        print(f"    [风险卖出] {stock_code} {shares}股 @ {price:.2f} (手续费: {cost:.2f})")
                        del holdings[stock_code]

        # 执行每周流程（选股）
        result = rotator.run_weekly(friday)
        new_stocks = rotator.current_stocks

        # 获取持仓股票周五价格
        prices = load_stock_prices(data_provider, new_stocks, friday) if new_stocks else {}

        # 计算当前持仓市值
        portfolio_value = cash
        position_value = 0
        for code, pos in holdings.items():
            price = prices.get(code, 0) if code in prices else data_provider.get_stock_price(code, friday)
            if price and price > 0:
                pos_value = pos['shares'] * price
                position_value += pos_value
                print(f"  持仓: {code} {pos['shares']}股 @ {price:.2f} = {pos_value:.2f}")

        portfolio_value += position_value

        # 调仓逻辑
        sell_stocks = [code for code in holdings if code not in new_stocks]
        buy_stocks = [code for code in new_stocks if code not in holdings]

        # 卖出不在新持仓列表中的股票（正常调仓）
        for code in sell_stocks:
            # 获取卖出价格（可能不在prices字典中，需要单独获取）
            price = prices.get(code) if code in prices else data_provider.get_stock_price(code, friday)
            if code in holdings:
                if price and price > 0:
                    shares = holdings[code]['shares']
                    revenue = shares * price
                    cost = calculate_transaction_cost(revenue, is_buy=False)
                    cash += revenue - cost
                    total_cost += cost
                    print(f"  卖出: {code} {shares}股 @ {price:.2f} (手续费: {cost:.2f})")
                else:
                    # 无法获取价格，报错并移除持仓（防止累积）
                    print(f"  警告: 无法获取 {code} 的价格，清除持仓")
                del holdings[code]

        # 买入新持仓股票（使用风险管理器的仓位计算）
        if cash > 0 and buy_stocks:
            # 计算已有持仓的市值
            existing_value = 0.0
            for code, pos in holdings.items():
                price = prices.get(code, 0) if code in prices else data_provider.get_stock_price(code, friday)
                if price and price > 0:
                    existing_value += pos['shares'] * price

            for code in buy_stocks:
                price = prices.get(code) if code in prices else data_provider.get_stock_price(code, friday)
                if price and price > 0:
                    # 使用风险管理器计算买入数量
                    shares = risk_manager.calculate_position_size(
                        total_capital=cash + existing_value,
                        current_price=price,
                        existing_positions=existing_value / price if price > 0 else 0
                    )
                    if shares > 0:
                        cost = shares * price
                        buy_cost = calculate_transaction_cost(cost, is_buy=True)
                        if cost + buy_cost <= cash:
                            holdings[code] = {
                                'shares': shares,
                                'buy_price': price,
                                'highest_price': price  # 初始化最高价
                            }
                            cash -= (cost + buy_cost)
                            total_cost += buy_cost
                            print(f"  买入: {code} {shares}股 @ {price:.2f} (手续费: {buy_cost:.2f})")

        # 重新计算调仓后的组合净值
        final_value = cash
        for code, pos in holdings.items():
            price = prices.get(code, 0) if code in prices else data_provider.get_stock_price(code, friday)
            if price and price > 0:
                final_value += pos['shares'] * price

        weekly_results.append({
            'date': friday,
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

    # 打印风险触发记录
    if exit_records:
        print(f"\n风险触发记录: {len(exit_records)}次")
        df_exits = pd.DataFrame(exit_records)
        print(df_exits.to_string(index=False))

    output_path = f"back_testing/results/composite_{start_date}_{end_date}_{time.time()}.csv"
    df_weeks.to_csv(output_path, index=False)
    print(f"\n周报已保存: {output_path}")

    return df_weeks


def main():
    parser = argparse.ArgumentParser(description='多策略综合评分量化选股系统')
    parser.add_argument('--start', default='2024-01-01', help='回测开始日期')
    parser.add_argument('--end', default='2024-12-31', help='回测结束日期')
    parser.add_argument('--capital', type=float, default=1000000.0, help='初始资金')
    args = parser.parse_args()

    run_backtest(args.start, args.end, args.capital)


if __name__ == '__main__':
    main()