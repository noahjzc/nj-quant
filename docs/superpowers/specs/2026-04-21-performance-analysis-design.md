# 绩效分析模块设计方案

## 背景

回测系统已实现风险管理功能，需要增加完整的绩效分析能力，包括风险调整收益指标、相对收益分析、收益归因和可视化。

## 绩效指标体系

### 1. 绝对收益指标

| 指标    | 计算公式                      | 说明   |
|-------|---------------------------|------|
| 总收益率  | (期末净值 - 期初净值) / 期初净值      | 最终收益 |
| 年化收益率 | (1 + 总收益率)^(365/持仓天数) - 1 | 年化收益 |
| 最大回撤  | max(Peak - Valley) / Peak | 最大跌幅 |

**详细含义：**
- **总收益率**：最终赚了多少%。比如投入100万，期末变成125万，总收益就是25%
- **年化收益率**：折算成每年的收益。方便比较不同投资周期。比如2年赚50%，年化是22.5%
- **最大回撤**：从最高点到最低点跌了多少%。比如一度涨到120万，后来跌到100万，回撤就是16.7%

### 2. 风险调整收益指标

| 指标        | 计算公式                    | 说明      |
|-----------|-------------------------|---------|
| Sharpe比率  | (年化收益率 - 无风险利率) / 年化波动率 | 超额收益/风险 |
| Calmar比率  | 年化收益率 / 最大回撤            | 收益/回撤比  |
| Sortino比率 | (年化收益率 - 目标收益) / 下行波动率  | 只考虑下行风险 |

**详细含义：**

- **Sharpe比率**：每承担1份风险能获得多少超额收益。公式：(年化收益 - 无风险利率) / 波动率。**理想值：> 1 为好，> 2 很好**
- **Calmar比率**：每承受1份回撤风险能获得多少收益。公式：年化收益 / 最大回撤。**理想值：> 1 为好，> 3 很好**
- **Sortino比率**：和Sharpe类似，但只计算下跌时的波动（好的向上波动不计）。公式：(年化收益 - 目标收益) / 下行波动率。**理想值：> 1 为好**

**为什么需要这些？** 光看收益率不够，同样赚20%，一个稳稳的一个大起大落的，显然稳稳的更好。这些指标衡量的是"性价比"。

**参数配置：**

- 无风险利率：2.5%（默认值，可配置）
- 目标收益：0%（用于Sortino）

### 3. 相对收益指标（需要基准指数）

| 指标    | 计算公式                        | 说明     |
|-------|-----------------------------|--------|
| Alpha | 组合收益 - β × 基准收益             | 超额收益   |
| Beta  | Cov(组合收益, 基准收益) / Var(基准收益) | 系统性风险  |
| 信息比率  | 超额收益 / 跟踪误差                 | 主动管理能力 |

**详细含义：**

- **Alpha (α)**：跑赢基准多少。Alpha=5%意味着比基准多赚了5%
- **Beta (β)**：跟着基准走的程度。β=1跟基准一样，β=1.5涨跌都比基准多50%，β=0.8涨跌都比基准少20%
- **信息比率**：主动收益的稳定性。持续稳定跑赢基准的策略信息比率高

### 4. 交易分析指标

| 指标     | 计算公式             | 说明    |
|--------|------------------|-------|
| 胜率     | 盈利交易次数 / 总交易次数   | 盈利比例  |
| 盈亏比    | 平均盈利金额 / 平均亏损金额  | 收益风险比 |
| 平均持仓天数 | 总持仓天数 / 交易次数     | 持仓周期  |
| 换手率    | 总交易金额 / 2 / 平均资产 | 交易频率  |

**详细含义：**

- **胜率**：100笔交易里有多少笔赚钱
- **盈亏比**：赚的时候平均赚多少，亏的时候平均亏多少的比值。盈亏比2:1意味着赚是亏的2倍
- **平均持仓天数**：股票平均持有多长时间
- **换手率**：一年交易多少次（按金额算）。换手率300%意味着一年把组合换了3遍

## 文件结构

```
back_testing/
├── performance_analyzer.py  # 新增：绩效分析器
├── index_data_provider.py   # 新增：指数数据读取器
└── visualizer.py           # 新增/修改：可视化模块
```

## 指数数据读取

### 数据格式

```
index_code, date, open, close, low, high, volume, money, change
sh000001, 2024-01-02, 3000.0, 3050.0, 2980.0, 3060.0, 1.5e9, 4.5e11, 0.017
```

### IndexDataProvider接口

```python
class IndexDataProvider:
    """指数数据提供器"""

    def __init__(self, data_dir: str):
        """
        data_dir: 指数数据目录
        例如: D:\workspace\code\mine\quant\data\metadata\daily_ycz\index
        """

    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数数据

        Returns:
            DataFrame with columns: date, open, close, high, low, volume
        """

    def get_index_return(self, index_code: str, start_date: str, end_date: str) -> float:
        """计算区间收益率"""
```

## 绩效分析器接口

```python
class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(self, trades: list, benchmark_index: str = 'sh000001'):
        """
        Args:
            trades: 交易记录列表，每条记录包含:
                - date: 交易日期
                - action: 'buy' | 'sell'
                - price: 成交价格
                - shares: 成交数量
                - return: 收益率（卖出时）
            benchmark_index: 基准指数代码，默认sh000001（沪深300）
        """

    def calculate_returns(self) -> pd.DataFrame:
        """
        计算每日收益率序列

        Returns:
            DataFrame with columns: date, portfolio_return, benchmark_return
        """

    def calculate_metrics(self) -> dict:
        """
        计算所有绩效指标

        Returns:
            dict: {
                'total_return': float,
                'annual_return': float,
                'max_drawdown': float,
                'sharpe_ratio': float,
                'calmar_ratio': float,
                'sortino_ratio': float,
                'alpha': float,
                'beta': float,
                'win_rate': float,
                'profit_loss_ratio': float,
                'avg_holding_days': float,
                'turnover_rate': float,
                ...
            }
        """

    def get_equity_curve(self) -> pd.DataFrame:
        """获取资金曲线"""

    def get_drawdown_curve(self) -> pd.DataFrame:
        """获取回撤曲线"""
```

## 可视化模块

### 图表类型

1. **资金曲线** - 组合净值 vs 基准指数
2. **回撤曲线** - 净值从高点的回撤百分比
3. **收益分布直方图** - 单笔收益分布
4. **月度收益热力图** - 各月收益表现

### 可视化接口

```python
class PerformanceVisualizer:
    """绩效可视化器"""

    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer

    def plot_equity_curve(self, save_path: str = None):
        """资金曲线图"""

    def plot_drawdown(self, save_path: str = None):
        """回撤曲线图"""

    def plot_return_distribution(self, save_path: str = None):
        """收益分布图"""

    def plot_monthly_returns(self, save_path: str = None):
        """月度收益热力图"""

    def generate_report(self, save_dir: str = None) -> str:
        """
        生成完整分析报告（HTML格式）

        Returns:
            str: 报告文件路径
        """
```

## 输出增强

### 回测输出新增

```
========================================
绩效分析
========================================
绝对收益:
  总收益率: +25.3%
  年化收益率: +18.7%
  最大回撤: -8.2%

风险调整收益:
  Sharpe比率: 1.45
  Calmar比率: 2.28
  Sortino比率: 1.82

相对收益:
  Alpha: +5.3%
  Beta: 0.85
  信息比率: 1.12

交易分析:
  胜率: 62.5%
  盈亏比: 1.85
  平均持仓天数: 12.5天
  换手率: 320%

========================================
```

## 实现步骤

1. 创建 `index_data_provider.py` - 指数数据读取
2. 创建 `performance_analyzer.py` - 绩效指标计算
3. 创建/修改 `visualizer.py` - 可视化模块
4. 修改 `run_composite_backtest.py` - 集成绩效分析输出
5. 生成HTML报告功能

## 默认参数

| 参数    | 默认值      | 说明          |
|-------|----------|-------------|
| 基准指数  | sh000001 | 沪深300       |
| 无风险利率 | 2.5%     | 用于Sharpe计算  |
| 目标收益  | 0%       | 用于Sortino计算 |
