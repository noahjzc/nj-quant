# 新股过滤 + 资金分配修复实现计划

> **For agentic workers:** Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现新股过滤逻辑（成熟度≥20日、停牌清缓存、退市清缓存、因子缺失得0分），并修复 `_execute_buy` 资金分配顺序 bug。

**Architecture:** 在 `DailyRotationEngine` 中修改 `_preload_histories`（30日预加载窗口）、`_advance_to_date`（停牌检测）、`_get_daily_stock_data`（成熟度过滤）、`_execute_buy`（NaN→0分 + 两阶段资金分配）五个方法。

**Tech Stack:** Python 3.12, pandas, numpy, PostgreSQL

---

## Task 0: 修复 `_execute_buy` 资金分配顺序 bug（CRITICAL）

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:256-344`

**当前逻辑（BUG）：**
- 按排名顺序遍历候选股，每次调用 `can_buy`（只检查仓位限制，不检查现金）后立即 `execute_buy`
- `execute_buy` 立即扣减 `self.current_capital`
- 导致低排名股票可能耗尽现金，使高排名股票无法买入

**新逻辑（两阶段）：**

阶段一：预计算每个候选股的可用性
```python
# 阶段一：计算候选股所需资金，确定最终能买哪些
capital_remaining = self.current_capital
selected = []
for stock_code in ranked:
    price = current_prices.get(stock_code, 0.0)
    if price <= 0:
        continue
    if not self.position_manager.can_buy(stock_code, price, existing_positions, current_prices):
        continue

    # 计算可买股数
    max_shares_by_capital = int(capital_remaining / price)
    max_shares_by_position = int(total_asset * self.position_manager.max_position_pct / price)
    shares = min(max_shares_by_capital, max_shares_by_position)
    if shares == 0:
        continue

    cost = shares * price * self.trade_executor.TRADING_TAX * self.trade_executor.TRADING_COMMISSION
    capital_needed = shares * price + cost
    if capital_needed > capital_remaining:
        continue

    selected.append((stock_code, price, shares, cost, capital_needed))
    capital_remaining -= capital_needed
    existing_positions[stock_code] = shares
```

阶段二：执行选中股票的买入
```python
# 阶段二：执行买入（从阶段一结果中扣除现金）
for stock_code, price, shares, cost, capital_needed in selected:
    self.current_capital -= capital_needed
    # ... 记录 position 和 trade_history
```

**步骤：**

- [ ] **Step 1: 读取代修改范围代码**

Read: `back_testing/rotation/daily_rotation_engine.py:256-344`

- [ ] **Step 2: 重写 `_execute_buy` 方法**

用两阶段逻辑替换现有方法体（保持方法签名不变）。

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile back_testing/rotation/daily_rotation_engine.py`
Expected: OK

- [ ] **Step 4: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "fix(rotation): two-phase capital allocation in _execute_buy"
```

---

## Task 1: 修改 `_preload_histories` — 30日预加载窗口

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:360-378`

**当前逻辑（需修改）：**
- 预加载 25 日日历天，截断到 20 行

**新逻辑：**
- 预加载 30 日日历天
- 不截断到 20 行（让缓存自然积累）
- 所有股票数据都进缓存，不做成熟度过滤

**步骤：**

- [ ] **Step 1: 修改 `_preload_histories`**

将：
```python
start = (first_date - pd.Timedelta(days=25)).strftime('%Y-%m-%d')
```

改为：
```python
start = (first_date - pd.Timedelta(days=self.PRELOAD_DAYS)).strftime('%Y-%m-%d')
```

删除截断逻辑：
```python
# 删除这两行：
if len(df) > 20:
    df = df.tail(20)
```

最终 `_preload_histories` 应为：
```python
def _preload_histories(self, first_date: pd.Timestamp):
    """预加载初始窗口：回测首日前30个日历日的数据"""
    if not self._all_codes:
        return

    start = (first_date - pd.Timedelta(days=self.PRELOAD_DAYS)).strftime('%Y-%m-%d')
    end = first_date.strftime('%Y-%m-%d')

    histories = self.data_provider.get_batch_histories(
        self._all_codes, end_date=end, start_date=start
    )

    self._stock_cache: Dict[str, pd.DataFrame] = {}
    for code, df in histories.items():
        if not df.empty:
            self._stock_cache[code] = df.copy()
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile back_testing/rotation/daily_rotation_engine.py`
Expected: OK

- [ ] **Step 3: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "feat(rotation): preload 30 calendar days without truncating to 20 rows"
```

---

## Task 2: 修改 `_advance_to_date` — 停牌检测和缓存清空

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:380-409`

**新逻辑：**
- 获取当日有交易的股票集合：`set(day_data.keys())`
- 遍历 `_stock_cache` 中所有股票代码
- 若某股票在当日无交易（停牌）→ 删除该股缓存 `del self._stock_cache[code]`
- 退市同理（持续无数据会被自动清理）

**步骤：**

- [ ] **Step 1: 在 `_advance_to_date` 开头添加停牌检测逻辑**

找到 `_advance_to_date` 的方法开头，在 `day_data = ...` 之后添加：

```python
# 检测停牌：当日无交易的股票清空缓存（退市同理）
trading_codes = set(day_data.keys())
for code in list(self._stock_cache.keys()):
    if code not in trading_codes:
        del self._stock_cache[code]
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile back_testing/rotation/daily_rotation_engine.py`
Expected: OK

- [ ] **Step 3: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "feat(rotation): clear cache for suspended/delisted stocks on non-trading days"
```

---

## Task 3: 修改 `_get_daily_stock_data` — 成熟度过滤

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:411-420`

**当前逻辑（需修改）：**
```python
def _get_daily_stock_data(self, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    result = {}
    for code, df in self._stock_cache.items():
        if df.empty:
            continue
        if date in df.index:
            result[code] = df
    return result
```

**新逻辑：**
在返回前增加成熟度判断：
```python
if len(df) < self.MIN_TRADING_DAYS:
    continue
```

最终应为：
```python
def _get_daily_stock_data(self, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """获取当日全市场日线数据（仅返回缓存中≥20个交易日的成熟股）"""
    result = {}
    for code, df in self._stock_cache.items():
        if df.empty:
            continue
        if len(df) < self.MIN_TRADING_DAYS:
            continue
        if date in df.index:
            result[code] = df
    return result
```

**步骤：**

- [ ] **Step 1: 修改 `_get_daily_stock_data` 方法体**

将方法体替换为上述新逻辑，同时更新 docstring。

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile back_testing/rotation/daily_rotation_engine.py`
Expected: OK

- [ ] **Step 3: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "feat(rotation): filter stocks by MIN_TRADING_DAYS in _get_daily_stock_data"
```

---

## Task 4: 修改 `_execute_buy` — 因子缺失填0分

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:273-290`

**当前逻辑（需修改）：**
```python
for factor in self.ranker.factor_weights.keys():
    if factor == 'RET_20':
        if len(df) >= 20 and 'close' in df.columns:
            ret = row['close'] / df['close'].iloc[-20] - 1
            factor_row[factor] = ret
    elif factor in row.index:
        factor_row[factor] = row[factor]
if factor_row:
    factor_data_dict[stock_code] = factor_row
```

**问题：**
- 当因子数据缺失时，`factor_row` 中没有该因子，导致 `factor_data_dict` 可能为空而被跳过
- `RET_20` 计算失败时不设置该因子值

**新逻辑：**
- 所有配置的因子都必须进入 `factor_row`，缺失的设为 `np.nan`
- 最终 `factor_df` 中 NaN 的处理由 `fillna(0)` 处理

修改为：
```python
for factor in self.ranker.factor_weights.keys():
    if factor == 'RET_20':
        # 20日收益率 = 当日收盘 / 20日前收盘 - 1
        if len(df) >= 20 and 'close' in df.columns:
            factor_row[factor] = row['close'] / df['close'].iloc[-20] - 1
        else:
            factor_row[factor] = np.nan
    elif factor in row.index:
        val = row[factor]
        factor_row[factor] = val if val == val else np.nan  # NaN check
    else:
        factor_row[factor] = np.nan
factor_data_dict[stock_code] = factor_row
```

然后在 `factor_df = pd.DataFrame(factor_data_dict).T` 之后、`ranked = self.ranker.rank(...)` 之前，填充 NaN 为 0：
```python
factor_df = pd.DataFrame(factor_data_dict).T
factor_df = factor_df.fillna(0)
ranked = self.ranker.rank(factor_df, top_n=x)
```

**步骤：**

- [ ] **Step 1: 修改因子提取循环**

替换因子提取循环为上述新逻辑。

- [ ] **Step 2: 在 `ranked = self.ranker.rank(...)` 前添加 `fillna(0)`**

```python
factor_df = pd.DataFrame(factor_data_dict).T
factor_df = factor_df.fillna(0)
ranked = self.ranker.rank(factor_df, top_n=x)
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile back_testing/rotation/daily_rotation_engine.py`
Expected: OK

- [ ] **Step 4: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "feat(rotation): fill missing factors with 0 in _execute_buy"
```

---

## Task 5: 添加类级别常量 + 集成验证

**Files:**
- Modify: `back_testing/rotation/daily_rotation_engine.py` (常量)

**步骤：**

- [ ] **Step 1: 添加类级别常量**

在 `DailyRotationEngine` 类开头（`__init__` 之前）添加：
```python
PRELOAD_DAYS = 30
MIN_TRADING_DAYS = 20
```

- [ ] **Step 2: 功能验证**

```python
from back_testing.rotation.daily_rotation_engine import DailyRotationEngine
from back_testing.rotation.config import RotationConfig
import warnings
warnings.filterwarnings('ignore')

config = RotationConfig()
engine = DailyRotationEngine(config, '2024-01-02', '2024-01-10')
dates = engine._get_trading_dates()
engine._preload_histories(dates[0])

# 验证1：预加载后，缓存中应有数据（但还未到20日门槛）
print(f"Preload cache count: {len(engine._stock_cache)}")

# 验证2：推进到第一天，缓存应包含≥20日的成熟股
engine._advance_to_date(dates[0])
mature = {k: v for k, v in engine._stock_cache.items() if len(v) >= engine.MIN_TRADING_DAYS}
print(f"Mature stocks (>=20 days): {len(mature)}")

# 验证3：_get_daily_stock_data 只返回成熟股
sd = engine._get_daily_stock_data(dates[0])
for code, df in list(sd.items())[:3]:
    print(f"  {code}: {len(df)} rows")
print(f"Total stocks in _get_daily_stock_data: {len(sd)}")
```

Expected: 预加载有数据，mature > 0，`_get_daily_stock_data` 数量 ≤ mature

- [ ] **Step 3: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "feat(rotation): add PRELOAD_DAYS and MIN_TRADING_DAYS constants"
```

---

## 自检清单

- [ ] Task 0: 两阶段资金分配已修复（高排名股票优先于低排名）
- [ ] `PRELOAD_DAYS = 30` 已添加为类级别常量
- [ ] `MIN_TRADING_DAYS = 20` 已添加为类级别常量
- [ ] `_preload_histories`: 加载30日窗口，不截断到20行
- [ ] `_advance_to_date`: 停牌股清缓存，退市股自动清缓存
- [ ] `_get_daily_stock_data`: 仅返回 len(df) >= 20 的股票
- [ ] `_execute_buy`: 因子缺失填充为 0
- [ ] `fillna(0)` 在 `rank()` 调用之前
- [ ] 5个 task 共6次 commit 完成（Task 0单独，Task 1-4各1次，Task 5共2次：常量+验证）
