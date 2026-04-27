# Post-Filter 修复计划

> **For agentic workers:** Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复新股过滤实现后的遗留问题：total_asset 在买入时存在 1 日延迟、停牌股在 self.positions 中残留。

**Tech Stack:** Python 3.12, pandas, numpy

---

## Task 1: 修复 `_execute_buy` 中 `total_asset` 滞后问题

**文件:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:151-164`

**问题：**
`total_asset` 在 line 151 计算时，卖出阶段（line 158）尚未执行，因此 position_manager 的 total_capital 仍是昨日值。导致 `max_shares_by_position = int(total_asset * max_position_pct / price)` 使用的是昨日总资产。

**修复方案：**
在 `_run_single_day` 中，将 `total_asset` 的计算移至卖出阶段之后、买入阶段之前：

```python
# Step 1: 检查持仓卖出信号
sell_trades = self._check_and_sell(date_str, filtered_data, current_prices)

# 更新现金（卖出）
for trade in sell_trades:
    self.current_capital += trade.shares * trade.price - trade.cost

# Step 2: 扫描买入信号
buy_candidates = self._scan_buy_candidates(filtered_data)

# Step 3: 重新计算 total_asset（此时包含卖出后的现金更新）
total_asset = self.current_capital + self.position_manager.get_position_value(
    {p.stock_code: p.shares for p in self.positions.values()},
    current_prices
)
self.position_manager.update_capital(total_asset)

# Step 4: 多因子排序，买入 TOP X
buy_trades, top_stocks_info = self._execute_buy(date_str, filtered_data, buy_candidates, max_positions, current_prices, total_asset)
```

**步骤：**

- [ ] **Step 1: 读取当前 `_run_single_day` 代码**

Read `back_testing/rotation/daily_rotation_engine.py:141-186`

- [ ] **Step 2: 将 total_asset 计算移至卖出阶段之后**

将 line 151-155 的 `total_asset` 计算和 `update_capital` 移动到卖出逻辑之后（sell_trades 处理之后）。

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile back_testing/rotation/daily_rotation_engine.py`
Expected: OK

- [ ] **Step 4: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "fix(rotation): compute total_asset after sell phase to avoid 1-day lag"
```

---

## Task 2: 修复停牌股在 `self.positions` 中残留问题

**文件:**
- Modify: `back_testing/rotation/daily_rotation_engine.py:188-242`

**问题：**
当股票停牌时：
1. `_advance_to_date` 清除了该股缓存
2. `_get_daily_stock_data` 中该股不在 stock_data 中
3. `_check_and_sell` 中 `stock_code not in stock_data` → `continue`，从未卖出
4. 持仓永久残留在 `self.positions` 中

**修复方案：**
在 `_check_and_sell` 中，对持仓股票若当日不在 `stock_data` 中（即停牌），以最新可用价格强制平仓：

```python
for stock_code, position in self.positions.items():
    if stock_code not in stock_data:
        # 停牌股：无法获取当日价格，用缓存中最后一日收盘价平仓
        if stock_code in self._stock_cache and not self._stock_cache[stock_code].empty:
            df = self._stock_cache[stock_code]
            price = df['close'].iloc[-1]
            if price > 0:
                # 强制平仓
                shares, cost = self.trade_executor.execute_sell(stock_code, price, position.shares)
                if shares > 0:
                    # ... 记录 trade ...
                    del self.positions[stock_code]
        continue
```

**步骤：**

- [ ] **Step 1: 读取 `_check_and_sell` 方法**

Read `back_testing/rotation/daily_rotation_engine.py:188-242`

- [ ] **Step 2: 在方法开头添加停牌股处理逻辑**

在遍历 `self.positions` 时，若 `stock_code not in stock_data`，从缓存获取最新价格强制平仓。

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile back_testing/rotation/daily_rotation_engine.py`
Expected: OK

- [ ] **Step 4: 提交**

```bash
git add back_testing/rotation/daily_rotation_engine.py
git commit -m "fix(rotation): force sell suspended stocks to prevent position orphaning"
```

---

## 自检清单

- [ ] Task 1: `total_asset` 在卖出后计算，传入 `_execute_buy`
- [ ] Task 2: 停牌股从缓存获取最新价强制平仓，不在 positions 中残留
- [ ] 2 次 commit 完成
