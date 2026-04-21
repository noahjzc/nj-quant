import pandas as pd
import numpy as np
from pandas import DataFrame
from abc import ABC, abstractmethod
from datetime import datetime
from back_testing.data.data_provider import DataProvider


class BacktestEngine:
    """
    通用回测引擎
    流程：加载数据 → 预处理 → 生成信号 → 模拟交易 → 输出结果
    """

    # 交易成本设置
    STAMP_DUTY = 0.001     # 印花税：卖出时收取，0.1%
    TRANSFER_FEE = 0.00002  # 过户费：买卖都收取，0.002%
    BROKERAGE = 0.0003     # 券商佣金：买卖都收取，0.03%，最低5元

    def __init__(self, stock_code: str, data_path: str = None, initial_capital: float = 100000.0, benchmark_index: str = None,
                 stop_loss: float = None, take_profit: float = None, start_date: str = None, use_parquet: bool = True):
        self.stock_code = stock_code
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.data: DataFrame = None
        self.trades: list = []
        self.current_capital = initial_capital
        self.shares = 0  # 持有股数
        self.position = False  # 是否持仓
        self.total_costs = 0  # 累计手续费
        self.portfolio_value_history = []  # 组合市值历史（用于计算最大回撤）
        self.benchmark_index = benchmark_index  # 基准指数代码
        self.benchmark_return = 0  # 基准指数收益率
        self.stop_loss = stop_loss      # e.g., 0.10 means 10% stop loss
        self.take_profit = take_profit  # e.g., 0.20 means 20% take profit
        self.buy_price = None  # 持仓买入价格，用于计算浮动盈亏
        self.start_date = pd.to_datetime(start_date) if start_date else None  # 回测开始日期
        self.use_parquet = use_parquet

        # 创建数据提供器
        self.data_provider = DataProvider(data_dir=data_path, use_parquet=use_parquet)

    def load_data(self) -> DataFrame:
        """加载股票数据"""
        df = self.data_provider.get_stock_data(self.stock_code)

        # 按日期排序
        df = df.sort_values('交易日期')
        df = df.reset_index(drop=True)

        # 按开始日期筛选
        if self.start_date:
            df = df[df['交易日期'] >= self.start_date]

        self.data = df
        return df

    def load_benchmark(self) -> float:
        """加载基准指数数据，计算基准收益率"""
        if not self.benchmark_index:
            return 0

        # 尝试加载指数数据
        index_path = f"{self.data_path}\\index\\{self.benchmark_index}.csv"
        try:
            index_df = pd.read_csv(index_path)
            index_df['date'] = pd.to_datetime(index_df['date'])
            index_df = index_df.sort_values('date').reset_index(drop=True)

            # 获取回测区间的指数数据
            start_date = self.data.iloc[0]['交易日期']
            end_date = self.data.iloc[-1]['交易日期']

            index_df = index_df[(index_df['date'] >= start_date) & (index_df['date'] <= end_date)]

            if len(index_df) >= 2:
                start_price = index_df.iloc[0]['close']
                end_price = index_df.iloc[-1]['close']
                self.benchmark_return = (end_price - start_price) / start_price
            return self.benchmark_return
        except Exception as e:
            print(f"  [警告] 加载基准指数失败: {e}")
            return 0

    def generate_signals(self) -> DataFrame:
        """
        生成交易信号 - 子类实现
        """
        raise NotImplementedError("子类必须实现generate_signals方法")

    def run(self) -> dict:
        """运行回测"""
        # 加载数据
        self.load_data()

        # 加载基准指数
        self.load_benchmark()

        # 生成信号
        self.data = self.generate_signals()

        # 模拟交易
        self.simulate_trades()

        # 计算结果
        result = self.calculate_result()
        return result

    def calculate_transaction_cost(self, amount: float, is_buy: bool) -> float:
        """计算交易成本"""
        if is_buy:
            # 买入：券商佣金 + 过户费
            brokerage = max(amount * self.BROKERAGE, 5)  # 最低5元
            transfer = amount * self.TRANSFER_FEE
            return brokerage + transfer
        else:
            # 卖出：券商佣金 + 过户费 + 印花税
            brokerage = max(amount * self.BROKERAGE, 5)  # 最低5元
            transfer = amount * self.TRANSFER_FEE
            stamp = amount * self.STAMP_DUTY
            return brokerage + transfer + stamp

    def simulate_trades(self):
        """模拟交易"""
        df = self.data

        # 重置索引以确保可以使用iloc
        df = df.reset_index(drop=True)

        # 记录初始组合市值
        self.portfolio_value_history = [(pd.Timestamp(df.iloc[0]['交易日期']), self.initial_capital)]

        for i in range(len(df)):
            signal = df.loc[i, 'TRADE_SIGNAL']
            price = df.loc[i, '后复权价']  # 使用后复权价计算

            # Check stop loss / take profit
            if self.position and self.buy_price is not None:
                unrealized_pnl = (price - self.buy_price) / self.buy_price
                if self.stop_loss and unrealized_pnl <= -self.stop_loss:
                    # Trigger stop loss - sell
                    signal = 0
                elif self.take_profit and unrealized_pnl >= self.take_profit:
                    # Trigger take profit - sell
                    signal = 0

            if signal == 1 and not self.position:
                # 买入信号且当前未持仓
                self.shares = int(self.current_capital / price)
                cost = self.shares * price
                # 计算买入手续费
                buy_cost = self.calculate_transaction_cost(cost, is_buy=True)
                self.total_costs += buy_cost
                self.current_capital -= (cost + buy_cost)
                self.position = True
                self.buy_price = price

                self.trades.append({
                    'date': df.loc[i, '交易日期'],
                    'action': 'BUY',
                    'price': price,
                    'shares': self.shares,
                    'cost': cost,
                    'fee': buy_cost,
                    'capital_after': self.current_capital
                })

            elif signal == 0 and self.position:
                # 卖出信号且当前持仓
                revenue = self.shares * price
                # 计算卖出手续费（含印花税）
                sell_cost = self.calculate_transaction_cost(revenue, is_buy=False)
                self.total_costs += sell_cost
                self.current_capital += (revenue - sell_cost)
                profit_ratio = (revenue - sell_cost - self.buy_price * self.shares) / (self.buy_price * self.shares)
                self.position = False

                # Check if stop loss was triggered
                unrealized_pnl = (price - self.buy_price) / self.buy_price
                stop_loss_triggered = self.stop_loss is not None and unrealized_pnl <= -self.stop_loss
                take_profit_triggered = self.take_profit is not None and unrealized_pnl >= self.take_profit

                self.trades.append({
                    'date': df.loc[i, '交易日期'],
                    'action': 'SELL',
                    'price': price,
                    'shares': self.shares,
                    'revenue': revenue,
                    'fee': sell_cost,
                    'profit_ratio': profit_ratio,
                    'capital_after': self.current_capital,
                    'stop_loss_triggered': stop_loss_triggered,
                    'take_profit_triggered': take_profit_triggered
                })
                self.shares = 0
                self.buy_price = None

            # 记录当前组合市值（现金 + 持仓市值）
            if self.position:
                portfolio_value = self.current_capital + self.shares * price
            else:
                portfolio_value = self.current_capital
            self.portfolio_value_history.append((df.loc[i, '交易日期'], portfolio_value))

        # 如果最后还持仓，按最后价格平仓
        if self.position:
            price = df.iloc[-1]['后复权价']
            revenue = self.shares * price
            sell_cost = self.calculate_transaction_cost(revenue, is_buy=False)
            self.total_costs += sell_cost
            self.current_capital += (revenue - sell_cost)
            profit_ratio = (revenue - sell_cost - self.buy_price * self.shares) / (self.buy_price * self.shares)
            self.trades.append({
                'date': df.iloc[-1]['交易日期'],
                'action': 'SELL',
                'price': price,
                'shares': self.shares,
                'revenue': revenue,
                'fee': sell_cost,
                'profit_ratio': profit_ratio,
                'capital_after': self.current_capital,
                'note': 'final_position'
            })
            self.shares = 0
            self.position = False

    def calculate_result(self) -> dict:
        """计算回测结果"""
        final_capital = self.current_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # 计算年化收益率
        if len(self.data) > 0:
            start_date = self.data.iloc[0]['交易日期']
            end_date = self.data.iloc[-1]['交易日期']
            days = (end_date - start_date).days
            years = days / 365 if days > 0 else 1
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            years = 1
            annual_return = 0

        # 计算最大回撤
        max_drawdown = self.calculate_max_drawdown()

        # 统计交易
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        total_trades = len(sell_trades)
        win_trades = len([t for t in sell_trades if t.get('profit_ratio', 0) > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        # 平均收益
        avg_profit = np.mean([t.get('profit_ratio', 0) for t in sell_trades]) if sell_trades else 0

        return {
            'stock_code': self.stock_code,
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_trades': win_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_costs': self.total_costs,
            'benchmark_return': self.benchmark_return,
            'trades': self.trades,
            'start_date': self.data.iloc[0]['交易日期'] if len(self.data) > 0 else None,
            'end_date': self.data.iloc[-1]['交易日期'] if len(self.data) > 0 else None,
        }

    def calculate_max_drawdown(self) -> float:
        """计算最大回撤（基于组合市值历史）"""
        if not self.portfolio_value_history:
            return 0

        peak = self.initial_capital
        max_dd = 0

        for date, value in self.portfolio_value_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def print_result(self, result: dict, strategy_name: str):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print(f"【{result['stock_code']}】{strategy_name} 回测结果")
        print("=" * 60)
        print(f"回测区间: {result['start_date'].strftime('%Y-%m-%d') if result['start_date'] else 'N/A'} ~ "
              f"{result['end_date'].strftime('%Y-%m-%d') if result['end_date'] else 'N/A'}")
        print(f"初始资金: {result['initial_capital']:,.2f}")
        print(f"最终资金: {result['final_capital']:,.2f}")
        print(f"总收益率: {result['total_return']:.2%}")
        print(f"年化收益率: {result['annual_return']:.2%}")
        print(f"最大回撤: {result['max_drawdown']:.2%}")

        # 基准对比
        benchmark_return = result.get('benchmark_return', 0)
        excess_return = result['total_return'] - benchmark_return
        print(f"基准指数收益: {benchmark_return:.2%}")
        print(f"超额收益: {excess_return:+.2%}")

        print(f"交易次数: {result['total_trades']}")
        print(f"盈利次数: {result['win_trades']}")
        print(f"胜率: {result['win_rate']:.2%}")
        print(f"平均收益: {result['avg_profit']:.2%}")
        print(f"累计手续费: {result.get('total_costs', 0):,.2f}")
        print("-" * 60)

        # 打印交易明细
        if result['trades']:
            print("\n交易明细:")
            for trade in result['trades']:
                date_str = trade['date'].strftime('%Y-%m-%d') if hasattr(trade['date'], 'strftime') else str(trade['date'])
                if trade['action'] == 'BUY':
                    print(f"  {date_str} 买入  价格:{trade['price']:.2f}  股数:{trade['shares']}  "
                          f"成本:{trade['cost']:.2f} 手续费:{trade.get('fee', 0):.2f}  资金:{trade['capital_after']:.2f}")
                else:
                    note = f" 收益:{trade['profit_ratio']:.2%}" if 'profit_ratio' in trade else ""
                    print(f"  {date_str} 卖出  价格:{trade['price']:.2f}  股数:{trade['shares']}  "
                          f"收入:{trade.get('revenue', 0):.2f} 手续费:{trade.get('fee', 0):.2f}{note}  资金:{trade['capital_after']:.2f}")

        print("=" * 60)
