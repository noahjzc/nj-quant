# 选股流程改进设计

> 日期：2026-04-23
> 目标：提升多因子模型选股质量，使回测从亏损转为盈利

## 背景

之前回测表现差（收益率 -25.63%，最大回撤 -45.17%），分析主要原因是选股模型问题。现有代码存在 4 个明确缺陷需要修复。

---

## 阶段1：修复 TURNOVER 映射 bug（P0）

### 问题

`factor_loader.py` 中：
```python
'TURNOVER': 'turnover',  # 错误：数据库列名是 turnover_amount
```

但 `MultiFactorSelector.select_top_stocks()` 里用成交额过滤时，调用的 `load_stock_turnover()` 用的是另一个逻辑（遍历列名找包含 'TURNOVER' 的列），两个逻辑不一致。

### 修复方案

1. 修正 `factor_loader.py` 的映射：
```python
'TURNOVER': 'turnover_amount',  # 数据库实际列名
```

2. 统一 `load_stock_turnover()` 使用 `factor_loader` 的列名映射

---

## 阶段2：增加动量因子（P1）

### 现有因子

| 因子类型 | 因子名 | 方向 | 说明 |
|---------|--------|------|------|
| 动量 | RSI_1/2/3 | +1 | 超卖反弹 |
| 趋势 | MA_5/10/20/30 | +1 | 多头排列 |
| KDJ | KDJ_K/D | +1 | 金叉 |
| 估值 | PB | -1 | 低估值 |

### 新增因子

| 因子类型 | 因子名 | 方向 | 计算方式 |
|---------|--------|------|----------|
| **价格动量** | RET_20 | +1 | 过去20日收益率 |
| **价格动量** | RET_60 | +1 | 过去60日收益率 |
| **盈利动量** | EARN_MOM | +1 | 单季度净利润增速 |
| **规模** | LN_MCAP | -1 | 对数市值（越小越好） |

### 数据获取

`factor_loader.py` 需要扩展：
```python
# 新增因子列名映射
FACTOR_COLUMNS = {
    ...
    'RET_20': 'ret_20',      # 20日收益率（需要在加载时计算）
    'RET_60': 'ret_60',      # 60日收益率
    'EARN_MOM': 'earn_mom',  # 盈利动量（从财务数据或计算）
    'LN_MCAP': 'ln_mcap',    # 对数市值
}
```

注意：动量因子需要在 `load_stock_factors()` 中根据历史价格计算，而不是直接从数据库读取。

---

## 阶段3：市值中性化（P2）

### 问题

现有 `FactorProcessor.neutralize()` 函数存在但从未被调用。因子排名受市值偏差影响大（大盘股牛市天然排名高）。

### 修复方案

在 `MultiFactorSelector.calculate_factor_scores()` 中，加入中性化选项：

```python
def calculate_factor_scores(self, data: pd.DataFrame,
                          neutralize: bool = False) -> pd.Series:
    """
    Args:
        data: 包含因子值的 DataFrame
        neutralize: 是否进行市值中性化
    """
    # 对每个因子做市值中性化
    if neutralize and 'ln_mcap' in data.columns:
        for factor in factor_columns:
            if factor != 'LN_MCAP':  # 市值本身不做中性化
                data[factor] = FactorProcessor.neutralize(
                    data[factor], data['ln_mcap']
                )
    # 然后再计算综合分数
    ...
```

默认开启中性化，减少市值偏差。

---

## 阶段4：Fitness 函数加入风控（P3）

### 问题

当前 GA 的 Fitness 函数使用简化回测（无止损/止盈/移动止损），优化出来的权重在实盘风控下表现可能完全不同。

### 修复方案

在 `FitnessEvaluator._run_backtest()` 中加入简化风控：

```python
def _run_backtest(self, weights, ...):
    # 调仓时不加止损止盈（保持简化）
    # 但在计算组合收益率时：
    # 1. 每周检查是否触发止损，触发则跳过该持仓期
    # 2. 计算持有期收益率时加入交易成本
    # 3. 如果持仓期间最大回撤超过阈值，该期收益记为负
```

简化风控（仅用于 GA 评估，不用于实盘）：
- 止损：-5% 止损（简化版，不频繁触发）
- 持有期超过 20% 回撤时强制换仓

---

## 实施文件

| 文件 | 修改内容 |
|------|----------|
| `back_testing/factors/factor_loader.py` | 修复 TURNOVER 映射，增加动量因子计算 |
| `back_testing/selectors/multi_factor_selector.py` | 加入中性化选项 |
| `back_testing/optimization/genetic_optimizer/fitness.py` | 加入简化风控逻辑 |

---

## 预期效果

| 阶段 | 修复内容 | 预期 |
|------|----------|------|
| 1 | 成交额过滤生效 | 剔除低流动性股票 |
| 2 | 动量因子加入 | 选股区分度提升 |
| 3 | 市值中性化 | 因子不受大盘影响 |
| 4 | Fitness 加入风控 | GA 优化更真实 |
