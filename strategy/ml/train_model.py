"""ML 模型训练入口

Usage:
    python -m strategy.ml.train_model --train-start 2020-01-01 --train-end 2022-12-31 --cache-dir cache/daily_rotation
    python -m strategy.ml.train_model --train-start 2021-01-01 --train-end 2023-12-31 --output model.pkl
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strategy.ml.trainer import MLRankerTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='训练 ML 排名模型')
    parser.add_argument('--train-start', required=True, help='训练开始日期 YYYY-MM-DD')
    parser.add_argument('--train-end', required=True, help='训练截止日期 YYYY-MM-DD')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='Parquet 缓存目录')
    parser.add_argument('--output', default=None, help='模型输出路径')

    args = parser.parse_args()

    trainer = MLRankerTrainer(args.cache_dir)
    path = trainer.train(args.train_start, args.train_end, args.output)

    logger.info(f"训练完成: {path}")


if __name__ == '__main__':
    main()
