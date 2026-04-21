"""
量化策略回测主程序

运行MA均线策略和MACD策略的回测
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 回测股票池
STOCK_POOL = [
    ('sh600519', '贵州茅台', '白酒'),
    ('sh600036', '招商银行', '银行'),
    ('sh601318', '中国平安', '保险'),
    ('sh688256', '寒武纪', '科创板'),
    ('sz300750', '宁德时代', '电池'),
    ('sz000001', '平安银行', '银行'),
    ('sh601899', '紫金矿业', '贵金属'),
    ('sz300059', '东方财富', '金融'),
]

# 数据路径
DATA_PATH = r"D:\workspace\code\mine\quant\data\metadata\daily_ycz"

# 基准指数
BENCHMARK_INDEX = "sh000001"  # 上证指数

# 初始资金
INITIAL_CAPITAL = 100000.0


def run_ma_strategy(stock_code: str, stock_name: str, industry: str):
    """运行MA均线策略回测"""
    from back_testing.strategies.ma_strategy import MAStrategy

    print(f"\n{'=' * 60}")
    print(f"开始回测: {stock_code} {stock_name} ({industry})")
    print(f"{'=' * 60}")

    try:
        engine = MAStrategy(
            stock_code=stock_code,
            data_path=DATA_PATH,
            initial_capital=INITIAL_CAPITAL,
            benchmark_index=BENCHMARK_INDEX
        )
        result = engine.run()
        engine.print_result(result, "MA均线策略(MA5上穿MA20买入,MA5下穿MA20卖出)")
        return result
    except Exception as e:
        print(f"  [错误] {e}")
        return None


def run_macd_strategy(stock_code: str, stock_name: str, industry: str):
    """运行MACD策略回测"""
    from back_testing.strategies.macd_strategy import MACDStrategy

    print(f"\n{'=' * 60}")
    print(f"开始回测: {stock_code} {stock_name} ({industry})")
    print(f"{'=' * 60}")

    try:
        engine = MACDStrategy(
            stock_code=stock_code,
            data_path=DATA_PATH,
            initial_capital=INITIAL_CAPITAL,
            benchmark_index=BENCHMARK_INDEX
        )
        result = engine.run()
        engine.print_result(result, "MACD策略(DIF上穿DEA)")
        return result
    except Exception as e:
        print(f"  [错误] {e}")
        return None


def run_rsi_strategy(stock_code: str, stock_name: str, industry: str):
    """运行RSI均值回归策略回测"""
    from back_testing.strategies.rsi_strategy import RSIReversalStrategy

    print(f"\n{'=' * 60}")
    print(f"开始回测: {stock_code} {stock_name} ({industry})")
    print(f"{'=' * 60}")

    try:
        engine = RSIReversalStrategy(
            stock_code=stock_code,
            data_path=DATA_PATH,
            initial_capital=INITIAL_CAPITAL,
            benchmark_index=BENCHMARK_INDEX
        )
        result = engine.run()
        engine.print_result(result, "RSI均值回归策略(RSI<30买入, RSI>50卖出)")
        return result
    except Exception as e:
        print(f"  [错误] {e}")
        return None


def print_summary(results: list, strategy_name: str):
    """打印汇总结果"""
    if not results:
        return

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return

    print(f"\n{'=' * 70}")
    print(f"【{strategy_name}】汇总统计 (共{len(valid_results)}只股票)")
    print(f"{'=' * 70}")

    # 计算平均指标
    avg_return = sum(r['total_return'] for r in valid_results) / len(valid_results)
    avg_annual = sum(r['annual_return'] for r in valid_results) / len(valid_results)
    avg_drawdown = sum(r['max_drawdown'] for r in valid_results) / len(valid_results)
    avg_win_rate = sum(r['win_rate'] for r in valid_results) / len(valid_results)
    avg_trades = sum(r['total_trades'] for r in valid_results) / len(valid_results)
    avg_benchmark = sum(r.get('benchmark_return', 0) for r in valid_results) / len(valid_results)

    # 盈利股票数
    profit_stocks = len([r for r in valid_results if r['total_return'] > 0])

    print(f"盈利股票: {profit_stocks}/{len(valid_results)} ({profit_stocks/len(valid_results):.1%})")
    print(f"平均总收益率: {avg_return:.2%}")
    print(f"平均年化收益率: {avg_annual:.2%}")
    print(f"平均最大回撤: {avg_drawdown:.2%}")
    print(f"平均基准收益: {avg_benchmark:.2%}")
    print(f"平均超额收益: {avg_return - avg_benchmark:+.2%}")
    print(f"平均胜率: {avg_win_rate:.2%}")
    print(f"平均交易次数: {avg_trades:.1f}")

    print(f"\n各股票表现:")
    print("-" * 80)
    print(f"{'代码':<12} {'名称':<8} {'总收益':<10} {'基准收益':<10} {'超额收益':<10} {'胜率':<8}")
    print("-" * 80)

    for r in valid_results:
        stock_info = next((s for s in STOCK_POOL if s[0] == r['stock_code']), (r['stock_code'], '', ''))
        benchmark = r.get('benchmark_return', 0)
        excess = r['total_return'] - benchmark
        print(f"{stock_info[0]:<12} {stock_info[1]:<8} {r['total_return']:>9.2%} {benchmark:>9.2%} {excess:>+9.2%} {r['win_rate']:>7.1%}")

    print("=" * 70)


def main():
    """主函数"""
    print(f"\n{'#' * 60}")
    print(f"# 量化策略回测系统")
    print(f"# 回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 数据路径: {DATA_PATH}")
    print(f"# 基准指数: {BENCHMARK_INDEX}")
    print(f"# 初始资金: {INITIAL_CAPITAL:,.2f}")
    print(f"{'#' * 60}")

    ma_results = []
    macd_results = []
    rsi_results = []

    # 运行MA策略回测
    print("\n\n" + "=" * 60)
    print(">>> 第一部分: MA均线策略回测")
    print("=" * 60)

    for stock_code, stock_name, industry in STOCK_POOL:
        result = run_ma_strategy(stock_code, stock_name, industry)
        ma_results.append(result)

    print_summary(ma_results, "MA均线策略")

    # 运行MACD策略回测
    print("\n\n" + "=" * 60)
    print(">>> 第二部分: MACD策略回测")
    print("=" * 60)

    for stock_code, stock_name, industry in STOCK_POOL:
        result = run_macd_strategy(stock_code, stock_name, industry)
        macd_results.append(result)

    print_summary(macd_results, "MACD策略")

    # 运行RSI均值回归策略
    print("\n\n" + "=" * 60)
    print(">>> 第三部分: RSI均值回归策略回测")
    print("=" * 60)

    for stock_code, stock_name, industry in STOCK_POOL:
        result = run_rsi_strategy(stock_code, stock_name, industry)
        rsi_results.append(result)

    print_summary(rsi_results, "RSI均值回归策略")

    # 策略对比
    print("\n\n" + "=" * 60)
    print(">>> 第四部分: 策略对比")
    print("=" * 60)

    ma_valid = [r for r in ma_results if r is not None]
    macd_valid = [r for r in macd_results if r is not None]
    rsi_valid = [r for r in rsi_results if r is not None]

    if ma_valid and macd_valid and rsi_valid:
        ma_avg_return = sum(r['total_return'] for r in ma_valid) / len(ma_valid)
        macd_avg_return = sum(r['total_return'] for r in macd_valid) / len(macd_valid)
        rsi_avg_return = sum(r['total_return'] for r in rsi_valid) / len(rsi_valid)

        ma_avg_win_rate = sum(r['win_rate'] for r in ma_valid) / len(ma_valid)
        macd_avg_win_rate = sum(r['win_rate'] for r in macd_valid) / len(macd_valid)
        rsi_avg_win_rate = sum(r['win_rate'] for r in rsi_valid) / len(rsi_valid)

        avg_benchmark = sum(r.get('benchmark_return', 0) for r in ma_valid) / len(ma_valid)

        print(f"\n策略表现对比:")
        print("-" * 55)
        print(f"{'指标':<15} {'MA策略':<15} {'MACD策略':<15} {'RSI策略':<15}")
        print("-" * 55)
        print(f"{'平均总收益率':<15} {ma_avg_return:>14.2%} {macd_avg_return:>14.2%} {rsi_avg_return:>14.2%}")
        print(f"{'平均胜率':<15} {ma_avg_win_rate:>14.1%} {macd_avg_win_rate:>14.1%} {rsi_avg_win_rate:>14.1%}")
        print(f"{'超额收益':<15} {ma_avg_return-avg_benchmark:>+14.2%} {macd_avg_return-avg_benchmark:>+14.2%} {rsi_avg_return-avg_benchmark:>+14.2%}")
        print("-" * 55)

        # 找出最佳策略
        strategies = [
            ("MA策略", ma_avg_return),
            ("MACD策略", macd_avg_return),
            ("RSI策略", rsi_avg_return)
        ]
        best_strategy = max(strategies, key=lambda x: x[1])
        print(f"\n结论: {best_strategy[0]} 在这批股票上表现最好")

    print("\n\n" + "#" * 60)
    print("# 回测完成!")
    print("#" * 60)


if __name__ == '__main__':
    main()
