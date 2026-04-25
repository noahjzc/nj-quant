# 新股过滤设计 spec

## 目标

过滤掉上市时间短、数据不成熟的新股，避免其高波动性和不稳定指标影响交易决策。

## 规则

### 1. 成熟度判断
- 股票在 `_stock_cache` 中积累 **≥ 20 个交易日** 即视为成熟股
- 成熟股参与：候选股扫描（`_scan_buy_candidates`）、多因子排序（`_execute_buy`）
- 不成熟股：被排除，不参与任何后续逻辑

### 2. 数据加载

**预加载（初始化时一次性）**
- 回测首日前 **30 个日历日**的历史数据
- 通过 `DataProvider.get_batch_histories()` 批量加载
- 存入 `_stock_cache: Dict[str, pd.DataFrame]`

**每日推进（每个交易日）**
- `DataProvider.get_stocks_for_date()` 获取当日全市场数据（一条 SQL）
- 追加到对应股票的缓存中
- 所有股票都追加（不区分新旧）

### 3. 停牌处理
- 若某股票在当日 `get_stocks_for_date` 中无数据 → 该股**清空缓存**
- 停牌结束后，重新作为新股处理，需重新积累 20 个交易日

### 4. 退市处理
- 若某股票在回测期间退市 → 直接清除缓存，不再参与

### 5. 因子评分
- 成熟股在因子提取时：
  - 若某因子数据存在 → 正常参与评分
  - 若某因子数据缺失（NaN） → 该因子得 **0 分**（不排除该股，但排序竞争力弱）
- `RET_20`（20 日收益率）：由引擎根据 `close` 列计算，非原始列

## 数据流

```
初始化:
  get_batch_histories(回测首日-30天, 回测首日)
  → _stock_cache (所有股票, 但不区分成熟与否)

每日 (_advance_to_date):
  get_stocks_for_date(当日)
  → 追加到 _stock_cache
  → 若当日无该股数据 → 清空该股缓存

每日 (_get_daily_stock_data):
  遍历 _stock_cache
  → len(cache) >= 20 → 成熟股 → 返回 DataFrame
  → len(cache) < 20 → 不成熟 → 跳过

每日 (_scan_buy_candidates / _execute_buy):
  仅处理成熟股（由 _get_daily_stock_data 保证）
  → 因子缺失 → 该因子得 0 分
```

## 涉及文件

- `back_testing/rotation/daily_rotation_engine.py`:
  - `_preload_histories`: 改为加载 30 日窗口
  - `_advance_to_date`: 增加停牌检测和缓存清空逻辑
  - `_get_daily_stock_data`: 增加成熟度判断（≥ 20 日）
  - `_execute_buy`: 因子缺失时置 0
- `back_testing/rotation/config.py`: 无需修改
- `back_testing/data/data_provider.py`: 无需修改

## 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `PRELOAD_DAYS` | 30 | 预加载日历日窗口 |
| `MIN_TRADING_DAYS` | 20 | 最小交易天数门槛 |
