# 选股流程改进实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复选股流程中的 4 个缺陷，提升多因子模型选股质量

**Architecture:** 四阶段递进实施：①修 TURNOVER bug → ②加动量因子 → ③市值中性化 → ④Fitness 加风控

**Tech Stack:** Python, pandas, SQLAlchemy, PostgreSQL

---

## 文件映射

| 文件 | 职责 |
|------|------|
| `back_testing/factors/factor_loader.py` | 因子数据加载，包含 FACTOR_COLUMNS 映射和 load_stock_factors() |
| `back_testing/factors/factor_config.py` | 因子权重和方向配置 |
| `back_testing/selectors/multi_factor_selector.py` | 多因子选股核心，计算综合分数 |
| `back_testing/optimization/genetic_optimizer/fitness.py` | GA 适应度函数，简化回测 |

---

## 阶段1：修复 TURNOVER 映射 bug

**Files:**
- Modify: `back_testing/factors/factor_loader.py`

### Task 1: 修复 FACTOR_COLUMNS 映射

- [ ] **Step 1: 检查当前代码**

```bash
grep -n "TURNOVER" back_testing/factors/factor_loader.py
```

预期输出：第 41 行 `'TURNOVER': 'turnover'`

- [ ] **Step 2: 修复映射**

```python
# factor_loader.py 第 41 行
'TURNOVER': 'turnover_amount',  # 数据库实际列名是 turnover_amount
```

- [ ] **Step 3: 检查数据库列名**

```python
# 验证数据库列名
from back_testing.data.data_provider import DataProvider
p = DataProvider()
df = p.get_stock_data('sh600519', end_date='2024-01-31')
print([c for c in df.columns if 'turnover' in c.lower()])
```

- [ ] **Step 4: 验证 load_stock_turnover 行为**

```python
from back_testing.factors.factor_loader import FactorLoader
loader = FactorLoader()
# 验证成交额能正确获取
```

- [ ] **Step 5: 提交**

```bash
git add back_testing/factors/factor_loader.py
git commit -m "fix(factor_loader): correct TURNOVER mapping to turnover_amount"
```

---

## 阶段2：增加动量因子

**Files:**
- Modify: `back_testing/factors/factor_loader.py`
- Modify: `back_testing/factors/factor_config.py`

### Task 2: 在 factor_loader.py 中新增 RET_20 和 RET_60 计算

- [ ] **Step 1: 在 FACTOR_COLUMNS 中添加新因子映射**

```python
# factor_loader.py FACTOR_COLUMNS 字典末尾添加
'RET_20': 'ret_20',      # 20日收益率（需计算，非数据库列）
'RET_60': 'ret_60',      # 60日收益率（需计算）
'LN_MCAP': 'ln_mcap',    # 对数市值
```

- [ ] **Step 2: 修改 load_stock_factors() 增加动量计算**

```python
def load_stock_factors(self, stock_codes, date, factors):
    """
    加载因子数据，对动量类因子需要在加载时计算
    """
    # 1. 先获取基本因子数据
    result_data = {}
    for code in stock_codes:
        df = self.data_provider.get_stock_data(code, date=date)
        if len(df) == 0:
            continue
        latest = df.iloc[-1]
        row = {}
        for factor in factors:
            col_name = self.FACTOR_COLUMNS.get(factor, factor)
            # 检查是否为需要计算的因子
            if factor in ('RET_20', 'RET_60'):
                row[factor] = self._calculate_return(code, date, factor)
            elif factor == 'LN_MCAP':
                row[factor] = self._calculate_ln_mcap(latest)
            else:
                if col_name in df.columns:
                    row[factor] = latest[col_name]
        result_data[code] = row
```

- [ ] **Step 3: 添加 _calculate_return() 方法**

```python
def _calculate_return(self, stock_code: str, date: pd.Timestamp, period: int) -> float:
    """
    计算过去N日收益率
    
    Args:
        stock_code: 股票代码
        date: 当前日期
        period: 回看天数 (20 或 60)
    
    Returns:
        收益率 (小数，如 0.15 表示 15%)
    """
    end_date = date.strftime('%Y-%m-%d')
    start_date = (date - pd.Timedelta(days=period * 3)).strftime('%Y-%m-%d')  # 多取一些天数确保够用
    
    df = self.data_provider.get_stock_data(stock_code, start_date=start_date, end_date=end_date)
    if len(df) < period + 1:
        return 0.0
    
    # 使用 adj_close 计算收益率
    prices = df['adj_close'].values
    if len(prices) < period + 1:
        return 0.0
    
    start_price = prices[-(period + 1)]
    end_price = prices[-1]
    if start_price == 0:
        return 0.0
    return (end_price - start_price) / start_price
```

- [ ] **Step 4: 添加 _calculate_ln_mcap() 方法**

```python
def _calculate_ln_mcap(self, row) -> float:
    """
    计算对数市值
    
    Returns:
        对数市值 (circulating_mv 或 total_mv 的自然对数)
    """
    mv = row.get('circulating_mv') or row.get('total_mv', 0)
    if mv and mv > 0:
        return np.log(mv)
    return 0.0
```

- [ ] **Step 5: 在 factor_config.py 中添加新因子配置**

```python
# factor_config.py DEFAULT_FACTOR_CONFIG 末尾添加
# 新增动量因子
'RET_20': {
    'weight': 0.10,
    'direction': 1,
    'description': '20日价格动量，越强越好'
},
'RET_60': {
    'weight': 0.10,
    'direction': 1,
    'description': '60日价格动量，越强越好'
},
'LN_MCAP': {
    'weight': 0.05,
    'direction': -1,
    'description': '对数市值，越小越好'
},
```

- [ ] **Step 6: 测试新因子加载**

```python
from back_testing.factors.factor_loader import FactorLoader
from back_testing.data.data_provider import DataProvider
import pandas as pd

loader = FactorLoader(data_provider=DataProvider())
date = pd.Timestamp('2024-01-05')
factors = ['RSI_1', 'RET_20', 'RET_60', 'LN_MCAP']
codes = ['sh600519', 'sh600000', 'sh600036']
df = loader.load_stock_factors(codes, date, factors)
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'RET_20 values: {df["RET_20"].values}')
```

- [ ] **Step 7: 提交**

```bash
git add back_testing/factors/factor_loader.py back_testing/factors/factor_config.py
git commit -m "feat(factors): add momentum factors RET_20, RET_60, and LN_MCAP"
```

---

## 阶段3：市值中性化

**Files:**
- Modify: `back_testing/selectors/multi_factor_selector.py`

### Task 3: 在 MultiFactorSelector 中加入 neutralize 参数

- [ ] **Step 1: 修改 __init__ 添加 neutralize 参数**

```python
def __init__(self, weights, directions, method='rank', data_provider=None,
             neutralize=False):
    # ...
    self.neutralize = neutralize
```

- [ ] **Step 2: 修改 calculate_factor_scores 加入中性化逻辑**

```python
def calculate_factor_scores(self, data: pd.DataFrame) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    
    factor_columns = [f for f in self.weights.keys() if f in data.columns]
    if not factor_columns:
        return pd.Series(0.0, index=data.index)
    
    if len(data) == 1:
        return pd.Series([0.5], index=data.index)
    
    # === 市值中性化 ===
    if self.neutralize and 'LN_MCAP' in data.columns:
        market_cap = data['LN_MCAP']
        for factor in factor_columns:
            if factor != 'LN_MCAP':
                # 对每个因子做市值中性化
                data[factor] = FactorProcessor.neutralize(data[factor], market_cap)
    
    # 后续计算综合分数逻辑保持不变...
```

- [ ] **Step 3: 修改 select_top_stocks 传入 neutralize 参数**

确保 select_top_stocks 创建 MultiFactorSelector 时传入 neutralize 参数（或暴露给调用方）。

- [ ] **Step 4: 测试中性化效果**

```python
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_loader import FactorLoader
from back_testing.data.data_provider import DataProvider
import pandas as pd

loader = FactorLoader(data_provider=DataProvider())
date = pd.Timestamp('2024-01-05')
factors = ['RSI_1', 'RET_20', 'PB', 'LN_MCAP']
codes = ['sh600519', 'sh600000', 'sh600036', 'sz300750']
df = loader.load_stock_factors(codes, date, factors)

weights = {'RSI_1': 0.3, 'RET_20': 0.2, 'PB': 0.3, 'LN_MCAP': 0.2}
directions = {'RSI_1': 1, 'RET_20': 1, 'PB': -1, 'LN_MCAP': -1}

# 不带中性化
selector1 = MultiFactorSelector(weights, directions, neutralize=False)
scores1 = selector1.calculate_factor_scores(df)

# 带中性化
selector2 = MultiFactorSelector(weights, directions, neutralize=True)
scores2 = selector2.calculate_factor_scores(df)

print("Without neutralization:", scores1.values)
print("With neutralization:", scores2.values)
```

- [ ] **Step 5: 提交**

```bash
git add back_testing/selectors/multi_factor_selector.py
git commit -m "feat(selector): add market neutralization option to MultiFactorSelector"
```

---

## 阶段4：Fitness 函数加入风控

**Files:**
- Modify: `back_testing/optimization/genetic_optimizer/fitness.py`

### Task 4: 在 _run_backtest() 中加入简化止损逻辑

- [ ] **Step 1: 在 __init__ 中添加风控参数**

```python
def __initself, max_drawdown_constraint=0.20, n_stocks=5,
             stop_loss_threshold=-0.05, max_holding_loss=-0.20):
    # stop_loss_threshold: 止损阈值 (如 -0.05 表示 -5% 止损)
    # max_holding_loss: 持仓期最大回撤阈值 (超过则该期记负收益)
```

- [ ] **Step 2: 修改 _calculate_period_return 加入风控检查**

```python
def _calculate_period_return(self, stocks, current_date, next_date) -> float:
    """
    计算持有期收益率，加入简化风控
    
    风控逻辑：
    1. 止损：持仓期间亏损超过 stop_loss_threshold 则该期收益为负
    2. 最大回撤：持仓期间最大回撤超过 max_holding_loss 则强制换仓
    """
    if not stocks:
        return 0.0
    
    # 获取每日收盘价
    all_prices = []
    for code in stocks:
        df = self.data_provider.get_stock_data(
            code, start_date=current_date.strftime('%Y-%m-%d'),
            end_date=next_date.strftime('%Y-%m-%d')
        )
        if len(df) > 0:
            all_prices.append(df['adj_close'].values)
    
    if not all_prices:
        return 0.0
    
    # 简单等权计算（实际可用更复杂的风控）
    returns = []
    for prices in all_prices:
        if len(prices) >= 2:
            ret = (prices[-1] - prices[0]) / prices[0]
            returns.append(ret)
    
    if not returns:
        return 0.0
    
    # 检查止损
    period_return = sum(returns) / len(returns)
    min_return = min(returns)
    if min_return <= self.stop_loss_threshold:
        # 触发止损，该期收益记负
        return min_return
    
    return period_return
```

- [ ] **Step 3: 测试风控效果**

```python
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator

evaluator = FitnessEvaluator(
    max_drawdown_constraint=0.20,
    n_stocks=5,
    stop_loss_threshold=-0.05
)

# 测试评估
result = evaluator.evaluate(
    {'RSI_1': 0.2, 'RSI_2': 0.1, 'RSI_3': 0.05, 'KDJ_K': 0.15, 'KDJ_D': 0.05,
     'MA_5': 0.15, 'MA_10': 0.1, 'MA_20': 0.1, 'MA_30': 0.1},
    pd.Timestamp('2023-01-01'),
    pd.Timestamp('2024-01-01')
)
print(f'Calmar ratio: {result}')
```

- [ ] **Step 4: 提交**

```bash
git add back_testing/optimization/genetic_optimizer/fitness.py
git commit -m "feat(fitness): add simplified stop-loss to GA fitness evaluation"
```

---

## 测试验证计划

所有阶段完成后，运行完整测试：

```bash
# 1. 测试 DataProvider 和因子加载
python -c "
from back_testing.data.data_provider import DataProvider
from back_testing.factors.factor_loader import FactorLoader
p = DataProvider()
loader = FactorLoader(data_provider=p)
print('DataProvider: OK')
print('FactorLoader: OK')
"

# 2. 测试多因子选股
python -c "
from back_testing.selectors.multi_factor_selector import MultiFactorSelector
from back_testing.factors.factor_loader import FactorLoader
from back_testing.data.data_provider import DataProvider
import pandas as pd

loader = FactorLoader(data_provider=DataProvider())
date = pd.Timestamp('2024-01-05')
factors = ['RSI_1', 'RSI_2', 'RSI_3', 'KDJ_K', 'KDJ_D', 'MA_5', 'MA_10', 'MA_20', 'MA_30', 'RET_20', 'RET_60']
codes = ['sh600519', 'sh600000', 'sh600036', 'sz300750']
df = loader.load_stock_factors(codes, date, factors)
print(f'Factor data shape: {df.shape}')

weights = {f: 1.0/len(factors) for f in factors}
directions = {f: 1 for f in factors}
selector = MultiFactorSelector(weights, directions, neutralize=True)
scores = selector.calculate_factor_scores(df)
print(f'Top score: {scores.max():.4f}')
print('MultiFactorSelector: OK')
"

# 3. 测试 GA fitness
python -c "
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator
import pandas as pd

evaluator = FitnessEvaluator()
result = evaluator.evaluate(
    {'RSI_1': 0.2, 'RSI_2': 0.1, 'RSI_3': 0.05, 'KDJ_K': 0.15, 'KDJ_D': 0.05,
     'MA_5': 0.15, 'MA_10': 0.1, 'MA_20': 0.1, 'MA_30': 0.1},
    pd.Timestamp('2023-01-01'),
    pd.Timestamp('2024-01-01')
)
print(f'Fitness result: {result:.4f}')
print('FitnessEvaluator: OK')
"
```

---

## 实施顺序

1. Task 1 → 阶段1：修复 TURNOVER bug
2. Task 2 → 阶段2：增加动量因子
3. Task 3 → 阶段3：市值中性化
4. Task 4 → 阶段4：Fitness 加入风控
