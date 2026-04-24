# 多因子回测优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扭亏为盈，收益率 > 0%，最大回撤 < 20%

**Architecture:** 收紧风控参数 + 限制科创板比例 + 成交额过滤

**Tech Stack:** Python, pandas, numpy

---

## 涉及文件

| 文件 | 修改内容 |
|------|----------|
| `back_testing/backtest/run_composite_backtest.py` | 调整 RISK_CONFIG 参数 |
| `back_testing/selectors/multi_factor_selector.py` | 增加科创板比例限制、成交额过滤 |
| `back_testing/factors/factor_loader.py` | 增加成交额数据获取 |

---

## Task 1: 收紧风控参数

**Files:**
- Modify: `back_testing/backtest/run_composite_backtest.py:34-42`

- [ ] **Step 1: 修改 RISK_CONFIG 参数**

```python
RISK_CONFIG = {
    'atr_period': 14,
    'stop_loss_mult': 1.0,      # 原: 2.0，更敏感
    'take_profit_mult': 2.0,    # 原: 3.0，降低止盈阈值
    'trailing_pct': 0.05,      # 原: 0.10，收紧移动止损
    'trailing启动条件': 0.03,   # 原: 0.05，更早启动
    'max_position_pct': 0.20,
    'max_total_pct': 0.90,
}
```

- [ ] **Step 2: 运行回测验证**

```bash
python -u -m back_testing.backtest.run_composite_backtest --start=2023-1-1 --end=2024-1-1
```

预期：最大回撤应显著缩小

---

## Task 2: 增加科创板/创业板比例限制

**Files:**
- Modify: `back_testing/selectors/multi_factor_selector.py:98-125`

- [ ] **Step 1: 在 `select_top_stocks` 增加主板优先逻辑**

在 `select_top_stocks` 方法中，修改选股逻辑：

```python
def select_top_stocks(self, data: pd.DataFrame, n: int = 5, excluded: List[str] = None) -> List[str]:
    """Select top N stocks by composite factor score."""
    if excluded is None:
        excluded = []

    # Calculate scores
    scores = self.calculate_factor_scores(data)

    # Filter out excluded stocks
    available = scores.drop(index=[s for s in excluded if s in scores.index], errors='ignore')

    if available.empty:
        return []

    # === 新增: 主板优先逻辑 ===
    # 主板股票: sh600***, sh000***, sz002***, sz001***
    # 科创/创业: sh688***, sz300***, sz301***
    main_board = []
    chi_next = []
    for code in available.index:
        if code.startswith(('sh600', 'sh000', 'sz001', 'sz002')):
            main_board.append(code)
        else:
            chi_next.append(code)

    # 主板优先：70% 选主板，30% 选科创/创业
    n_main = min(int(n * 0.7), len(main_board))
    n_chi = n - n_main

    # 按分数排序
    main_scores = available[main_board].sort_values(ascending=False) if main_board else pd.Series()
    chi_scores = available[chi_next].sort_values(ascending=False) if chi_next else pd.Series()

    # 选取
    result = list(main_scores.head(n_main).index)
    result.extend(chi_scores.head(n_chi).index)

    return result
```

- [ ] **Step 2: 运行回测验证**

```bash
python -u -m back_testing.backtest.run_composite_backtest --start=2023-1-1 --end=2024-1-1
```

预期：持仓中应有更多主板股票

---

## Task 3: 增加成交额过滤

**Files:**
- Modify: `back_testing/factors/factor_loader.py`
- Modify: `back_testing/selectors/multi_factor_selector.py`

- [ ] **Step 1: 在 factor_loader 中增加成交额获取方法**

在 `FactorLoader` 类中增加：

```python
def load_stock_turnover(
    self,
    stock_codes: List[str],
    date: pd.Timestamp
) -> pd.Series:
    """
    获取指定日期的成交额

    Args:
        stock_codes: 股票代码列表
        date: 日期

    Returns:
        Series: index为股票代码，value为成交额（元）
    """
    result = {}
    for code in stock_codes:
        try:
            df = self.data_provider.get_stock_data(code, date=date)
            if len(df) == 0:
                continue
            # 成交额列（Parquet column 14，GBK编码为"成交额"）
            # 尝试多种可能的列名
            turnover_col = None
            for col in df.columns:
                if '成交额' in col or 'TURNOVER' in col.upper():
                    turnover_col = col
                    break
            if turnover_col and turnover_col in df.columns:
                result[code] = df[turnover_col].iloc[-1]
        except Exception:
            continue
    return pd.Series(result)
```

- [ ] **Step 2: 在 multi_factor_selector 中增加成交额过滤**

在 `select_top_stocks` 方法中，增加：

```python
# === 新增: 成交额过滤 ===
# 获取当日成交额
turnover = self._get_turnover(available.index, date)
# 过滤成交额低于5000万的股票
MIN_TURNOVER = 50_000_000  # 5000万
if len(turnover) > 0:
    available = available[turnover >= MIN_TURNOVER]
```

需要在类中增加 `_get_turnover` 方法调用 `FactorLoader.load_stock_turnover`。

- [ ] **Step 3: 运行回测验证**

```bash
python -u -m back_testing.backtest.run_composite_backtest --start=2023-1-1 --end=2024-1-1
```

预期：过滤掉低流动性股票后，回撤应进一步缩小

---

## Task 4: 综合验证

- [ ] **运行完整回测**

```bash
python -u -m back_testing.backtest.run_composite_backtest --start=2023-1-1 --end=2024-1-1
```

- [ ] **对比优化前后结果**

| 指标 | 优化前 | 目标 |
|------|--------|------|
| 总收益率 | -25.63% | > 0% |
| 最大回撤 | -45.17% | < -20% |

---

## 实施顺序

1. Task 1: 收紧风控参数（最简单，立即见效）
2. Task 2: 科创板比例限制
3. Task 3: 成交额过滤
4. Task 4: 综合验证
