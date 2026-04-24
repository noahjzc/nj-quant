"""
历史全息数据导入脚本

将预测者网CSV数据导入PostgreSQL数据库

使用方法:
    python import_overview_data.py --data-dir D:/path/to/stock/csv --batch-size 1000

字段映射:
    CSV字段 -> 数据库字段
"""
import argparse
import os
import glob
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


# CSV字段 -> 数据库字段 映射
CSV_TO_DB_MAP = {
    '股票代码': 'stock_code',
    '股票名称': 'stock_name',
    '交易日期': 'trade_date',
    '新浪行业': 'industry',
    '新浪概念': 'concept',
    '新浪地域': 'area',
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '收盘价': 'close',
    '后复权价': 'adj_close',
    '前复权价': 'prev_adj_close',
    '涨跌幅': 'change_pct',
    '成交量': 'volume',
    '成交额': 'turnover_amount',
    '换手率': 'turnover_rate',
    '流通市值': 'circulating_mv',
    '总市值': 'total_mv',
    '是否涨停': 'limit_up',
    '是否跌停': 'limit_down',
    '市盈率TTM': 'pe_ttm',
    '市销率TTM': 'ps_ttm',
    '市现率TTM': 'pcf_ttm',
    '市净率': 'pb',
    'MA_5': 'ma_5',
    'MA_10': 'ma_10',
    'MA_20': 'ma_20',
    'MA_30': 'ma_30',
    'MA_60': 'ma_60',
    'MA金叉死叉': 'ma_cross',
    'MACD_DIF': 'macd_dif',
    'MACD_DEA': 'macd_dea',
    'MACD_MACD': 'macd_hist',
    'MACD_金叉死叉': 'macd_cross',
    'KDJ_K': 'kdj_k',
    'KDJ_D': 'kdj_d',
    'KDJ_J': 'kdj_j',
    'KDJ_金叉死叉': 'kdj_cross',
    '布林线中轨': 'boll_mid',
    '布林线上轨': 'boll_upper',
    '布林线下轨': 'boll_lower',
    'psy': 'psy',
    'psyma': 'psyma',
    'rsi1': 'rsi_1',
    'rsi2': 'rsi_2',
    'rsi3': 'rsi_3',
    '振幅': 'amplitude',
    '量比': 'volume_ratio',
}


def get_engine() -> Engine:
    """获取数据库连接"""
    from back_testing.data.db.connection import get_engine as get_db_engine
    return get_db_engine()


def process_file(filepath: str) -> pd.DataFrame:
    """
    处理单个CSV文件

    Args:
        filepath: CSV文件路径

    Returns:
        处理后的DataFrame
    """
    # 读取CSV
    df = pd.read_csv(filepath, encoding='gbk')

    # 重命名列
    df = df.rename(columns=CSV_TO_DB_MAP)

    # 确保 stock_code 是字符串
    df['stock_code'] = df['stock_code'].astype(str)

    # 转换日期格式
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date

    # 布尔字段处理
    # 是否涨停/跌停: '是' -> True, 其他 -> False
    if 'limit_up' in df.columns:
        df['limit_up'] = df['limit_up'].apply(lambda x: str(x).strip() == '是')
    if 'limit_down' in df.columns:
        df['limit_down'] = df['limit_down'].apply(lambda x: str(x).strip() == '是')

    # 空字符串转None
    df = df.replace('', None)

    return df


def import_data(
    data_dir: str,
    batch_size: int = 1000,
    table_name: str = 'stock_daily',
    verbose: bool = True
) -> dict:
    """
    导入数据到数据库

    Args:
        data_dir: CSV文件所在目录
        batch_size: 每批写入行数
        table_name: 目标表名
        verbose: 是否打印进度

    Returns:
        统计信息字典
    """
    engine = get_engine()

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    if verbose:
        print(f"找到 {len(csv_files)} 个CSV文件")

    total_rows = 0
    total_files = 0
    error_files = []
    start_time = datetime.now()

    # 按文件处理
    for i, filepath in enumerate(csv_files):
        try:
            # 处理文件
            df = process_file(filepath)

            # 写入数据库
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists='append',  # 追加模式
                index=False,
                chunksize=batch_size
            )

            total_rows += len(df)
            total_files += 1

            if verbose and (i + 1) % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"进度: {i+1}/{len(csv_files)} 文件, "
                      f"已导入 {total_rows} 行, "
                      f"耗时: {elapsed:.1f}s")

        except Exception as e:
            error_files.append((filepath, str(e)))
            if verbose:
                print(f"错误: {filepath}: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()

    result = {
        'total_files': total_files,
        'total_rows': total_rows,
        'error_files': error_files,
        'elapsed_seconds': elapsed,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"导入完成!")
        print(f"成功: {total_files} 文件, {total_rows} 行")
        print(f"失败: {len(error_files)} 文件")
        print(f"耗时: {elapsed:.1f}秒")

    return result


def verify_data(data_dir: str, sample_stocks: list = None) -> dict:
    """
    验证导入数据

    Args:
        data_dir: CSV数据目录
        sample_stocks: 抽样的股票代码列表

    Returns:
        验证结果字典
    """
    engine = get_engine()

    results = {}

    # 1. 统计验证：总行数对比
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    csv_total = 0
    for f in csv_files:
        df = pd.read_csv(f, encoding='gbk')
        csv_total += len(df)

    with engine.connect() as conn:
        db_total = conn.execute(
            engine.dialect.text("SELECT COUNT(*) FROM stock_daily")
        ).scalar()

    results['统计验证'] = {
        'CSV总行数': csv_total,
        '数据库总行数': db_total,
        '差异': csv_total - db_total,
    }

    # 2. 抽样验证：对比几条数据
    if sample_stocks is None:
        sample_stocks = ['sh600000', 'sz300750', 'bj430017']

    results['抽样验证'] = {}

    for stock_code in sample_stocks:
        # 读取CSV数据
        csv_path = os.path.join(data_dir, f'{stock_code}.csv')
        if not os.path.exists(csv_path):
            continue

        df_csv = pd.read_csv(csv_path, encoding='gbk')
        df_csv = df_csv.rename(columns=CSV_TO_DB_MAP)
        df_csv['trade_date'] = pd.to_datetime(df_csv['trade_date']).dt.date

        # 从数据库读取
        with engine.connect() as conn:
            query = f"""
                SELECT * FROM stock_daily
                WHERE stock_code = '{stock_code}'
                ORDER BY trade_date DESC
                LIMIT 5
            """
            df_db = pd.read_sql(query, conn)

        if len(df_db) > 0 and len(df_csv) > 0:
            # 对比最新一条
            csv_latest = df_csv.iloc[-1]
            db_latest = df_db.iloc[0]

            comparison = {
                'CSV行数': len(df_csv),
                'DB行数': len(df_db),
                'CSV最新日期': str(csv_latest['trade_date']),
                'DB最新日期': str(db_latest['trade_date']),
                'CSV收盘价': csv_latest.get('close'),
                'DB收盘价': float(db_latest['close']) if db_latest['close'] else None,
                'CSV后复权价': csv_latest.get('adj_close'),
                'DB后复权价': float(db_latest['adj_close']) if db_latest['adj_close'] else None,
            }
            results['抽样验证'][stock_code] = comparison

    return results


def main():
    parser = argparse.ArgumentParser(description='导入历史全息数据到数据库')
    parser.add_argument(
        '--data-dir', '-d',
        required=True,
        help='CSV文件所在目录'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1000,
        help='每批写入行数 (默认: 1000)'
    )
    parser.add_argument(
        '--verify-only', '-v',
        action='store_true',
        help='仅验证数据，不导入'
    )
    parser.add_argument(
        '--table-name', '-t',
        default='stock_daily',
        help='目标表名 (默认: stock_daily)'
    )
    parser.add_argument(
        '--db-url',
        default=None,
        help='数据库连接URL (默认: 从环境变量DATABASE_URL读取)'
    )

    args = parser.parse_args()

    if args.db_url:
        os.environ['DATABASE_URL'] = args.db_url

    if args.verify_only:
        # 仅验证
        print("开始验证数据...")
        results = verify_data(args.data_dir)

        print("\n" + "="*60)
        print("验证结果")
        print("="*60)

        print("\n统计验证:")
        stats = results['统计验证']
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print("\n抽样验证:")
        for stock, data in results['抽样验证'].items():
            print(f"\n  {stock}:")
            for k, v in data.items():
                print(f"    {k}: {v}")

    else:
        # 导入
        print(f"开始导入数据 from: {args.data_dir}")
        print(f"批次大小: {args.batch_size}")
        print(f"目标表: {args.table_name}")
        print("")

        result = import_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            table_name=args.table_name
        )

        if result['error_files']:
            print("\n错误文件列表:")
            for filepath, error in result['error_files']:
                print(f"  {filepath}: {error}")

        # 验证
        print("\n开始验证...")
        verify_results = verify_data(args.data_dir)

        print("\n" + "="*60)
        print("验证结果")
        print("="*60)

        stats = verify_results['统计验证']
        print(f"\n统计验证:")
        print(f"  CSV总行数: {stats['CSV总行数']}")
        print(f"  数据库总行数: {stats['数据库总行数']}")
        print(f"  差异: {stats['差异']}")

        if stats['差异'] == 0:
            print("  ✅ 行数完全匹配!")
        else:
            print(f"  ⚠️ 存在差异，可能有重复导入")


if __name__ == '__main__':
    main()
