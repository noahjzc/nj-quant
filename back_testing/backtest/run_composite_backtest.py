"""
多策略综合评分量化选股系统 - 回测入口
"""
import sys
import io
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from back_testing.composite_rotator import CompositeRotator
from back_testing.data.data_provider import DataProvider
from back_testing.data.index_data_provider import IndexDataProvider
from back_testing.analysis.performance_analyzer import PerformanceAnalyzer
from back_testing.risk.risk_manager import RiskManager
from back_testing.risk.stop_loss_strategies import StopLossStrategies
from back_testing.analysis.visualizer import PerformanceVisualizer

# Parquet数据目录（已废弃，数据现在从DB读取）
DATA_PATH = None
INITIAL_CAPITAL = 1000000.0
N_STOCKS = 5

# Exit reason display map
EXIT_REASON_MAP = {
    'stop_loss': '止损触发',
    'take_profit': '止盈触发',
    'trailing_stop': '移动止损触发'
}

# Risk Manager configuration
RISK_CONFIG = {
    'atr_period': 14,
    'stop_loss_mult': 1.5,      # 调整：止损2.0太松，1.0太紧
    'take_profit_mult': 2.5,    # 原: 3.0，降低止盈阈值
    'trailing_pct': 0.05,      # 原: 0.10，收紧移动止损
    'trailing启动条件': 0.03,   # 原: 0.05，更早启动
    'max_position_pct': 0.20,
    'max_total_pct': 0.90,
}

# Index data directory
INDEX_DATA_DIR = r'D:\workspace\code\mine\quant\data\metadata\daily_ycz\index'


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
                buy_price = position['buy_price']
                profit_pct = (current_price - buy_price) / buy_price * 100
                profit_str = f"+{profit_pct:.1f}%" if profit_pct >= 0 else f"{profit_pct:.1f}%"
                reason_display = EXIT_REASON_MAP.get(exit_result['action'], exit_result['action'])
                print(f"    [风险止损] {stock_code} 触发{reason_display}: {exit_result['reason']}")

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
    data_provider = DataProvider()

    # 创建指数数据提供器
    index_provider = IndexDataProvider(INDEX_DATA_DIR)

    # 初始化风险管理器
    risk_config = RISK_CONFIG.copy()
    risk_config['total_capital'] = initial_capital
    risk_manager = RiskManager(config=risk_config)

    rotator = CompositeRotator(
        initial_capital=initial_capital,
        n_stocks=N_STOCKS
    )

    fridays = get_trading_fridays(start_date, end_date)
    print(f"\n调仓日数量: {len(fridays)}")

    # 计算基准收益
    benchmark_return = index_provider.get_index_return('sh000001', start_date, end_date)

    # 初始状态
    cash = initial_capital  # 现金
    holdings = {}  # 持仓 {stock_code: {'shares': int, 'buy_price': float, 'highest_price': float}}
    total_cost = 0  # 累计交易成本
    exit_records = []  # 记录止损/止盈触发的交易
    rotation_sell_records = []  # 记录调仓卖出的交易
    buy_count = 0  # 买入次数（调仓产生的买入）

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
                    # 检查每日风险退出（止损/止盈/移动止损）
                    sell_list = check_daily_risk_exits(
                        data_provider=data_provider,
                        risk_manager=risk_manager,
                        holdings=holdings,
                        date=daily_date,
                        atr_period=risk_config['atr_period']
                    )

                    # 立即执行风险触发卖出
                    for stock_code, exit_result in sell_list:
                        if stock_code in holdings:
                            buy_price = holdings[stock_code]['buy_price']
                            price = data_provider.get_stock_price(stock_code, daily_date)
                            if price and price > 0:
                                shares = holdings[stock_code]['shares']
                                revenue = shares * price
                                cost = calculate_transaction_cost(revenue, is_buy=False)
                                profit_pct = (price - buy_price) / buy_price
                                cash += revenue - cost
                                total_cost += cost
                                exit_records.append({
                                    'date': daily_date,
                                    'stock': stock_code,
                                    'action': exit_result['action'],
                                    'price': price,
                                    'shares': shares,
                                    'buy_price': buy_price,
                                    'reason': exit_result['reason'],
                                    'return': profit_pct
                                })
                                profit_str = f"+{profit_pct*100:.1f}%" if profit_pct >= 0 else f"{profit_pct*100:.1f}%"
                                reason_display = EXIT_REASON_MAP.get(exit_result['action'], exit_result['action'])
                                print(f"    卖出: {stock_code} {shares}股 @ {price:.2f}")
                                print(f"      买入价: {buy_price:.2f}")
                                print(f"      盈亏: {profit_str}")
                                print(f"      原因: {reason_display}")
                                del holdings[stock_code]

                daily_date += timedelta(days=1)

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
                    buy_price = holdings[code]['buy_price']
                    shares = holdings[code]['shares']
                    revenue = shares * price
                    cost = calculate_transaction_cost(revenue, is_buy=False)
                    profit_pct = (price - buy_price) / buy_price
                    cash += revenue - cost
                    total_cost += cost
                    rotation_sell_records.append({
                        'date': friday,
                        'stock': code,
                        'action': 'rotation',
                        'price': price,
                        'shares': shares,
                        'buy_price': buy_price,
                        'reason': '调仓卖出',
                        'return': profit_pct
                    })
                    profit_str = f"+{profit_pct*100:.1f}%" if profit_pct >= 0 else f"{profit_pct*100:.1f}%"
                    print(f"  卖出: {code} {shares}股 @ {price:.2f}")
                    print(f"    买入价: {buy_price:.2f}")
                    print(f"    盈亏: {profit_str}")
                    print(f"    原因: 调仓")
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
                    # 新买入的股票不在持仓中，existing_positions为0
                    shares = risk_manager.calculate_position_size(
                        total_capital=cash + existing_value,
                        current_price=price,
                        existing_positions=0
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
                            buy_count += 1
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

    # 汇总所有交易记录
    all_trades = exit_records + rotation_sell_records
    sell_count = len(all_trades)  # 卖出次数

    # 统计盈利/亏损
    profitable_trades = [t for t in all_trades if t['return'] > 0]
    losing_trades = [t for t in all_trades if t['return'] <= 0]
    profitable_count = len(profitable_trades)
    losing_count = len(losing_trades)
    win_rate = (profitable_count / sell_count * 100) if sell_count > 0 else 0

    # 平均收益率
    avg_return = (sum(t['return'] for t in all_trades) / len(all_trades) * 100) if all_trades else 0

    # 最大单笔盈利/亏损
    max_profit = (max(t['return'] for t in all_trades) * 100) if all_trades else 0
    max_loss = (min(t['return'] for t in all_trades) * 100) if all_trades else 0

    # 止损/止盈/移动止损触发次数
    stop_loss_count = len([t for t in exit_records if t['action'] == 'stop_loss'])
    take_profit_count = len([t for t in exit_records if t['action'] == 'take_profit'])
    trailing_stop_count = len([t for t in exit_records if t['action'] == 'trailing_stop'])

    # 打印交易统计
    print("\n" + "=" * 50)
    print("交易统计")
    print("=" * 50)
    print(f"总买入次数: {buy_count}")
    print(f"总卖出次数: {sell_count}")
    print(f"盈利次数: {profitable_count}")
    print(f"亏损次数: {losing_count}")
    print(f"胜率: {win_rate:.1f}%")
    print(f"平均收益率: {avg_return:+.1f}%")
    print(f"最大单笔盈利: {max_profit:+.1f}%")
    print(f"最大单笔亏损: {max_loss:+.1f}%")
    print(f"总止损次数: {stop_loss_count}")
    print(f"总止盈次数: {take_profit_count}")
    print(f"移动止损触发: {trailing_stop_count}")
    print(f"累计交易成本: {total_cost:,.2f}")
    print("=" * 50)

    # 集成绩效分析
    all_trades = exit_records + rotation_sell_records
    # 构建完整净值序列（含初始资金）
    equity_curve = [initial_capital] + df_weeks['portfolio_value'].tolist()
    analyzer = PerformanceAnalyzer(
        trades=all_trades,
        initial_capital=initial_capital,
        benchmark_index='sh000001',
        equity_curve=equity_curve,
        periods_per_year=52  # 周频
    )
    metrics = analyzer.calculate_metrics()

    print("\n" + "=" * 50)
    print("绩效分析")
    print("=" * 50)
    print(f"绝对收益:")
    print(f"  总收益率: {metrics['total_return']:+.2%}")
    print(f"  年化收益率: {metrics['annual_return']:+.2%}")
    print(f"  最大回撤: {metrics['max_drawdown']:+.2%}")
    print(f"\n风险调整收益:")
    print(f"  Sharpe比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  Calmar比率: {metrics['calmar_ratio']:.2f}")
    print(f"  Sortino比率: {metrics['sortino_ratio']:.2f}")
    print(f"\n相对收益 (vs 沪深300):")
    print(f"  基准收益: {benchmark_return:+.2%}")
    print(f"  超额收益: {metrics['total_return'] - benchmark_return:+.2%}")
    print(f"\n交易分析:")
    print(f"  胜率: {metrics['win_rate']:.1%}")
    print(f"  盈亏比: {metrics['profit_loss_ratio']:.2f}")
    print("=" * 50)

    # 生成可视化报告
    try:
        df_weeks['portfolio_value_normalized'] = df_weeks['portfolio_value'] / df_weeks['portfolio_value'].iloc[0]
        equity_curve = df_weeks.set_index('date')['portfolio_value_normalized']
        benchmark_data = index_provider.get_index_data('sh000001', start_date, end_date)
        benchmark_data['normalized'] = benchmark_data['close'] / benchmark_data['close'].iloc[0]
        benchmark_curve = benchmark_data.set_index('date')['normalized']
        visualizer = PerformanceVisualizer(equity_curve, benchmark_curve)
        report_path = visualizer.generate_report(all_trades, save_dir='back_testing/results')
        print(f"\n绩效报告已生成: {report_path}")
    except Exception as e:
        print(f"\n生成报告失败: {e}")

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