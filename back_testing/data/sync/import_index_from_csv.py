"""
指数数据导入脚本 - 从CSV文件批量导入index_daily表

用法:
    python -u -m back_testing.data.sync.import_index_from_csv
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import logging
from pathlib import Path

import pandas as pd

# 添加项目根目录到 path
sys.path.insert(0, str(__file__).rsplit('back_testing', 1)[0])

from back_testing.data.db.connection import get_session
from back_testing.data.db.models import IndexDaily

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = r'D:\workspace\code\mine\quant\data\all-overview-data\index'


def import_index_from_csv(data_dir: str = DATA_DIR) -> dict:
    """从CSV文件导入指数数据

    Args:
        data_dir: CSV文件所在目录

    Returns:
        统计字典 {file_count, total_rows, skipped, failed}
    """
    stats = {'file_count': 0, 'total_rows': 0, 'skipped': 0, 'failed': 0}
    session = get_session()()

    csv_files = list(Path(data_dir).glob('*.csv'))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")

    for csv_file in csv_files:
        index_code = csv_file.stem  # 文件名即指数代码
        logger.info(f"正在导入 {index_code} ...")

        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                logger.warning(f"{index_code}: 文件为空，跳过")
                stats['skipped'] += 1
                continue

            # 检查必要列
            required_cols = ['index_code', 'date', 'open', 'close', 'low', 'high', 'volume', 'money']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"{index_code}: 缺少必要列，跳过")
                stats['skipped'] += 1
                continue

            for _, row in df.iterrows():
                try:
                    index_daily = IndexDaily(
                        index_code=row['index_code'],
                        trade_date=pd.to_datetime(row['date']).date(),
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        turnover=row['money']
                    )
                    session.merge(index_daily)
                    stats['total_rows'] += 1
                except Exception as e:
                    logger.warning(f"{index_code} 行导入失败: {e}")
                    stats['failed'] += 1

            session.commit()
            stats['file_count'] += 1
            logger.info(f"  {index_code} 导入完成: {len(df)} 行")

        except Exception as e:
            logger.error(f"{index_code} 文件处理失败: {e}")
            session.rollback()
            stats['failed'] += 1

    session.close()
    return stats


def main():
    logger.info("=" * 50)
    logger.info("指数数据导入开始")
    logger.info("=" * 50)

    stats = import_index_from_csv()

    logger.info("=" * 50)
    logger.info("导入完成")
    logger.info(f"  处理文件数: {stats['file_count']}")
    logger.info(f"  总行数: {stats['total_rows']}")
    logger.info(f"  跳过: {stats['skipped']}")
    logger.info(f"  失败: {stats['failed']}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()