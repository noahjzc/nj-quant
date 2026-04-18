import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import DataFrame

class Visualizer:
    """
    回测结果可视化
    生成资金曲线、买卖点标记图
    """

    def __init__(self, result: dict, strategy_name: str):
        self.result = result
        self.strategy_name = strategy_name

    def plot_equity_curve(self, save_path=None):
        """画出资金曲线"""
        trades = self.result['trades']
        if not trades:
            return

        # Build equity timeline
        dates = []
        values = []
        capital = self.result['initial_capital']

        for trade in trades:
            dates.append(trade['date'])
            values.append(capital)
            if trade['action'] == 'BUY':
                capital = trade['capital_after']
            else:
                capital = trade['capital_after']

        plt.figure(figsize=(12, 6))
        plt.plot(dates, values)
        plt.title(f'{self.strategy_name} - {self.result["stock_code"]} 资金曲线')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_trades_with_prices(self, data: DataFrame, save_path=None):
        """画出价格图和买卖点"""
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot price
        ax.plot(data['交易日期'], data['后复权价'], label='价格', alpha=0.7)

        # Mark buy/sell points
        trades = self.result['trades']
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] == 'SELL']

        ax.scatter([t['date'] for t in buys], [t['price'] for t in buys],
                   marker='^', color='green', s=100, label='买入')
        ax.scatter([t['date'] for t in sells], [t['price'] for t in sells],
                   marker='v', color='red', s=100, label='卖出')

        ax.set_title(f'{self.strategy_name} - {self.result["stock_code"]} 交易记录')
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.legend()
        ax.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_strategy_comparison(self, results_dict: dict, save_path=None):
        """对比多个策略的资金曲线"""
        plt.figure(figsize=(12, 6))
        for name, result in results_dict.items():
            trades = result['trades']
            dates = [t['date'] for t in trades]
            values = []
            capital = result['initial_capital']
            for t in trades:
                values.append(capital)
                capital = t['capital_after']
            plt.plot(dates, values, label=name)

        plt.title('策略对比')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()
