"""独立缓存构建入口 — 预计算 + Parquet 写入

Usage:
    python back_testing/data/build_daily_cache.py --start 2020-01-01 --end 2025-12-31
    python back_testing/data/build_daily_cache.py --start 2024-01-01 --end 2024-12-31 --cache-dir cache/my_cache
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path (in case script is run directly)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from back_testing.data.daily_data_cache import DailyDataCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='构建 Daily Rotation 预计算缓存（含滚动指标）'
    )
    parser.add_argument('--start', required=True, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='缓存目录')
    parser.add_argument('--preload-days', type=int, default=30, help='预加载天数')
    parser.add_argument('--benchmark-index', default='sh000300', help='基准指数代码')

    args = parser.parse_args()

    logger.info(f"开始构建缓存: {args.start} ~ {args.end}")
    path = DailyDataCache.build(
        start_date=args.start,
        end_date=args.end,
        cache_dir=args.cache_dir,
        preload_days=args.preload_days,
        benchmark_index=args.benchmark_index,
    )
    logger.info(f"缓存构建完成: {path}")


if __name__ == '__main__':
    main()
