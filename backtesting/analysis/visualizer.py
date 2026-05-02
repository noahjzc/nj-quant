import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PerformanceVisualizer:
    """
    回测结果可视化
    生成资金曲线、回撤曲线、收益分布和HTML报告
    """

    def __init__(self, equity_curve: pd.DataFrame, benchmark_curve: pd.DataFrame = None):
        """
        初始化可视化器

        Args:
            equity_curve: 组合净值序列，DataFrame或Series，index为日期
            benchmark_curve: 基准指数净值序列，DataFrame或Series，index为日期
        """
        if isinstance(equity_curve, pd.Series):
            self.equity_curve = equity_curve.to_frame(name='equity')
        else:
            self.equity_curve = equity_curve

        if benchmark_curve is not None:
            if isinstance(benchmark_curve, pd.Series):
                self.benchmark_curve = benchmark_curve.to_frame(name='benchmark')
            else:
                self.benchmark_curve = benchmark_curve
        else:
            self.benchmark_curve = None

    def plot_equity_curve(self, save_path: str = None):
        """
        绘制资金曲线

        Args:
            save_path: 保存路径，为None时不保存
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        dates = self.equity_curve.index
        equity = self.equity_curve.iloc[:, 0]

        ax.plot(dates, equity, label='组合净值', linewidth=1.5, color='#1f77b4')

        if self.benchmark_curve is not None:
            # 对齐基准数据到组合日期范围
            benchmark_aligned = self.benchmark_curve.reindex(self.equity_curve.index)
            ax.plot(dates, benchmark_aligned.iloc[:, 0], label='基准指数',
                    linewidth=1.2, color='#ff7f0e', alpha=0.8)

        ax.set_title('基金净值走势', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=11)
        ax.set_ylabel('净值', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 格式化x轴日期
        if isinstance(dates, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_drawdown(self, save_path: str = None):
        """
        绘制回撤曲线

        Args:
            save_path: 保存路径，为None时不保存
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        equity = self.equity_curve.iloc[:, 0]
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        dates = self.equity_curve.index

        ax.fill_between(dates, drawdown, 0, alpha=0.4, color='#d62728', label='回撤')
        ax.plot(dates, drawdown, linewidth=0.8, color='#d62728')

        ax.set_title('回撤曲线', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=11)
        ax.set_ylabel('回撤比例', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 格式化x轴日期
        if isinstance(dates, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_return_distribution(self, trades: list, save_path: str = None):
        """
        绘制收益分布直方图

        Args:
            trades: 交易记录列表，每条记录包含'return'字段
            save_path: 保存路径，为None时不保存
        """
        if not trades:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        returns = [t['return'] for t in trades if 'return' in t]

        if not returns:
            return

        ax.hist(returns, bins=30, alpha=0.7, color='#2ca02c', edgecolor='white')

        mean_return = np.mean(returns)
        ax.axvline(mean_return, color='#d62728', linestyle='--', linewidth=2,
                   label=f'均值: {mean_return:.2%}')
        ax.axvline(0, color='#1f77b4', linestyle='-', linewidth=1.5, label='零线')

        ax.set_title('收益分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('收益率', fontsize=11)
        ax.set_ylabel('频次', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self, trades: list, save_dir: str = None) -> str:
        """
        生成HTML报告

        Args:
            trades: 交易记录列表
            save_dir: 保存目录，为None时使用当前目录

        Returns:
            报告文件路径
        """
        if save_dir is None:
            save_dir = '.'

        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(save_dir, f'performance_report_{timestamp}.html')

        equity_path = os.path.join(save_dir, f'equity_{timestamp}.png')
        drawdown_path = os.path.join(save_dir, f'drawdown_{timestamp}.png')
        return_dist_path = os.path.join(save_dir, f'return_dist_{timestamp}.png')

        self.plot_equity_curve(equity_path)
        self.plot_drawdown(drawdown_path)
        self.plot_return_distribution(trades, return_dist_path)

        # 计算统计数据
        equity = self.equity_curve.iloc[:, 0]
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0
        running_max = equity.cummax()
        max_drawdown = ((equity - running_max) / running_max).min()

        returns = [t['return'] for t in trades if 'return' in t] if trades else []
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        avg_return = np.mean(returns) if returns else 0

        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>策略绩效报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .stats {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 15px 25px;
            border-radius: 5px;
            border-left: 4px solid #1f77b4;
        }}
        .stat-box .label {{
            color: #777;
            font-size: 12px;
        }}
        .stat-box .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-top: 30px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>策略绩效报告</h1>

        <h2>绩效统计</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="label">总收益率</div>
                <div class="value" style="color: {'#2ca02c' if total_return >= 0 else '#d62728'}">{total_return:.2%}</div>
            </div>
            <div class="stat-box">
                <div class="label">最大回撤</div>
                <div class="value" style="color: #d62728">{max_drawdown:.2%}</div>
            </div>
            <div class="stat-box">
                <div class="label">胜率</div>
                <div class="value">{win_rate:.2%}</div>
            </div>
            <div class="stat-box">
                <div class="label">平均收益</div>
                <div class="value" style="color: {'#2ca02c' if avg_return >= 0 else '#d62728'}">{avg_return:.2%}</div>
            </div>
        </div>

        <h2>净值走势</h2>
        <div class="chart">
            <img src="{os.path.basename(equity_path)}" alt="净值走势">
        </div>

        <h2>回撤曲线</h2>
        <div class="chart">
            <img src="{os.path.basename(drawdown_path)}" alt="回撤曲线">
        </div>

        <h2>收益分布</h2>
        <div class="chart">
            <img src="{os.path.basename(return_dist_path)}" alt="收益分布">
        </div>

        <div class="timestamp">
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return report_path
