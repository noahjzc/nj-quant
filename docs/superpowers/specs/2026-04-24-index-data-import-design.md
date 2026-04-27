# 指数数据导入设计

## 目标

将本地CSV文件中的指数数据批量导入PostgreSQL的`index_daily`表，数据源位于`D:\workspace\code\mine\quant\data\all-overview-data\index`。

## 数据源

路径：`D:\workspace\code\mine\quant\data\all-overview-data\index`

文件格式：62个CSV文件，命名规则为`{index_code}.csv`，如`sh000300.csv`、`sz399006.csv`。

CSV列结构：
```
index_code,date,open,close,low,high,volume,money,change
sh000300,2026-04-22,4751.28,4799.63,4750.35,4801.45,20448183000.0,619505670200.0,0.0066
```

## 目标表

表名：`index_daily`（已存在于`back_testing/data/db/models.py`）

字段映射：

| CSV列 | IndexDaily字段 |
|-------|---------------|
| index_code | index_code (PK) |
| date | trade_date (PK) |
| open | open |
| high | high |
| low | low |
| close | close |
| volume | volume |
| money | turnover |

主键：(index_code, trade_date)，使用`session.merge()`实现UPSERT语义（不存在则插入，存在则更新）。

## 导入脚本

新建 `back_testing/data/sync/import_index_from_csv.py`

功能：
1. 扫描数据源目录下所有`.csv`文件
2. 逐个解析CSV，使用`session.merge()`写入数据库
3. 输出导入统计（成功行数、跳过行数、失败行数）

不使用`IndexDaily`模型以外的ORM操作，直接复用现有模型和数据库连接。
