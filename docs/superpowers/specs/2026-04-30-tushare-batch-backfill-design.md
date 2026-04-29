# Tushare 批量数据补全工具设计

## 概述

`night_backfill.py` 每次只补一天。实现一个批量补全脚本 `batch_backfill.py`，传入起止日期，自动遍历并逐日补全数据。

## 使用方式

```bash
# 补全 4月22日到今天之间的所有缺失交易日
python signal_pipeline/batch_backfill.py --start 2025-04-23 --end 2025-04-30

# 仅补单日（与 night_backfill 等效）
python signal_pipeline/batch_backfill.py --start 2025-04-23 --end 2025-04-23
```

- 不传参数时报错退出
- 支持交易日判断（跳过周末/节假日）

## 行为

| 情况 | 处理 |
|------|------|
| 当日非交易日 | 跳过，不计为失败 |
| API 调用失败 | 打印错误、退出，整个任务标记失败 |
| DB 写入失败 | 打印错误、退出 |
| 成功 | 打印进度，继续下一天 |
| Tushare 返回空 | 打印警告、退出（Tushare 无数据可能是严重问题）|

退出码：全部成功为 0，有失败为 1。

## 输出格式

```
=== Batch Backfill 2025-04-23 ~ 2025-04-30 ===
2025-04-23  ✓  4200 rows
2025-04-24  ✓  4180 rows
2025-04-28  ✓  4210 rows
2025-04-29  ✓  4195 rows
2025-04-30  ✗  Tushare call failed: rate limit exceeded
=== FAILED: 2025-04-30 ===
Total: 4 succeeded, 1 failed ===
```

失败当日会输出完整错误信息到 stderr。

## 实现

**文件：** `signal_pipeline/batch_backfill.py`

复用 `night_backfill.py` 中的 `TushareClient`、`_convert_ts_code`、`_na_to_none` 等函数。

核心逻辑：

1. 解析 `--start` / `--end` 参数
2. 用 `pandas.tseries.offsets.BDay()` 生成交易日序列，过滤出 start~end 范围内的交易日
3. 逐日调用补全逻辑（复用 night_backfill 的处理流程，但跳过 cron_log 写入和 Parquet 更新，专注 DB 写入）
4. 失败时立即 `sys.exit(1)`

## Parquet 缓存处理

批量补全期间**不写 Parquet**（避免部分日期缺失导致缓存不完整）。每日夜间的 `night_backfill` 正常运行时再写入 Parquet，保证缓存完整性。

如需主动重建缓存，补全完成后单独运行：

```bash
python back_testing/data/build_daily_cache.py --start 2025-01-01 --end 2025-04-30
```

## 依赖

无新依赖。复用现有 `TushareClient`、`IndicatorCalculator`。
