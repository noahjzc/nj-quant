"""
历史数据初始化脚本

用法:
    python back_testing/data/sync/init_history.py --start 20210101 --end 20260420
"""
import argparse
import logging
import sys
import os
from datetime import datetime

# ============================================================
# 重要：在导入 akshare 之前清除代理设置
# 否则 akshare 内部会继承系统代理导致连接失败
# ============================================================
# 清除所有可能的代理环境变量
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY']:
    os.environ.pop(key, None)
# 显式设置为空字符串（确保 requests 不使用代理）
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# 设置 requests 库的代理为 None
import requests
requests.trust_env = False
# 清空 session 的代理
s = requests.Session()
s.trust_env = False
s.proxies = {'http': None, 'https': None}

# 设置 urllib3 禁用代理
import urllib3
urllib3.disable_warnings()

# 添加项目根目录到 path
sys.path.insert(0, str(__file__).rsplit('back_testing', 1)[0])

from back_testing.data.db.connection import get_engine, get_session
from back_testing.data.db.models import StockDaily, StockMeta, StockFinancial, IndexDaily
from back_testing.data.sync.akshare_client import AkshareClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoryInitializer:
    """历史数据初始化器"""

    def __init__(self, client: AkshareClient):
        self.client = client
        self.engine = get_engine()
        self.Session = get_session()

    def init_stock_meta(self) -> int:
        """初始化股票元数据"""
        logger.info("开始获取股票列表...")
        df = self.client.get_stock_list()

        if df.empty:
            logger.error("获取股票列表失败")
            return 0

        session = self.Session()
        count = 0

        try:
            for _, row in df.iterrows():
                stock_code = row['stock_code']

                # 判断市场
                if stock_code.startswith('sh'):
                    market = '沪'
                elif stock_code.startswith('sz'):
                    market = '深'
                else:
                    market = '北'

                meta = StockMeta(
                    stock_code=stock_code,
                    stock_name=row.get('stock_name'),
                    industry=row.get('industry'),
                    market=market,
                    is_active=True
                )
                session.merge(meta)  # 使用 merge 避免主键冲突
                count += 1

            session.commit()
            logger.info(f"股票元数据初始化完成，共 {count} 只")
        except Exception as e:
            session.rollback()
            logger.error(f"股票元数据初始化失败: {e}")
        finally:
            session.close()

        return count

    def init_stock_daily(self, stock_codes: list, start_date: str, end_date: str) -> int:
        """初始化股票日线数据"""
        total = len(stock_codes)
        success_count = 0
        fail_count = 0

        session = self.Session()

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                logger.info(f"进度: {i + 1}/{total}")

            try:
                df = self.client.get_stock_daily(code, start_date, end_date)

                if df.empty:
                    fail_count += 1
                    continue

                for _, row in df.iterrows():
                    daily = StockDaily(
                        stock_code=row['stock_code'],
                        trade_date=row['trade_date'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        turnover=row['turnover'],
                        amplitude=row['amplitude'],
                        change_pct=row['change_pct'],
                        is_trading=row.get('is_trading', True)
                    )
                    session.merge(daily)

                session.commit()
                success_count += 1

            except Exception as e:
                session.rollback()
                logger.warning(f"获取 {code} 日线数据失败: {e}")
                fail_count += 1

        session.close()
        logger.info(f"日线数据初始化完成: 成功 {success_count}, 失败 {fail_count}")
        return success_count

    def init_index_daily(self, index_codes: list, start_date: str, end_date: str) -> int:
        """初始化指数日线数据"""
        session = self.Session()
        success_count = 0

        for code in index_codes:
            try:
                df = self.client.get_index_daily(code, start_date, end_date)

                if df.empty:
                    continue

                for _, row in df.iterrows():
                    index_daily = IndexDaily(
                        index_code=row['index_code'],
                        trade_date=row['trade_date'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        turnover=row['turnover']
                    )
                    session.merge(index_daily)

                session.commit()
                success_count += 1

            except Exception as e:
                session.rollback()
                logger.warning(f"获取 {code} 指数数据失败: {e}")

        session.close()
        logger.info(f"指数数据初始化完成: {success_count}/{len(index_codes)}")
        return success_count

    def init_financial(self, stock_codes: list, years: list = None) -> int:
        """初始化财务数据"""
        if years is None:
            years = [2021, 2022, 2023, 2024, 2025]

        total = len(stock_codes)
        success_count = 0

        session = self.Session()

        for i, code in enumerate(stock_codes):
            if (i + 1) % 100 == 0:
                logger.info(f"财务数据进度: {i + 1}/{total}")

            try:
                df = self.client.get_stock_financial(code)

                if df.empty:
                    continue

                # 过滤指定年份
                if years:
                    df = df[df['report_period'].apply(lambda x: x.year in years)]

                for _, row in df.iterrows():
                    financial = StockFinancial(
                        stock_code=code,
                        report_period=row['report_period'],
                        publish_date=row.get('publish_date'),
                        roe=row.get('roe'),
                        roa=row.get('roa'),
                        gross_margin=row.get('gross_margin'),
                        net_margin=row.get('net_margin'),
                        debt_ratio=row.get('debt_ratio'),
                        current_ratio=row.get('current_ratio'),
                        quick_ratio=row.get('quick_ratio'),
                        pe_ttm=row.get('pe_ttm'),
                        pb=row.get('pb'),
                        ps_ttm=row.get('ps_ttm')
                    )
                    session.merge(financial)

                session.commit()
                success_count += 1

            except Exception as e:
                session.rollback()
                logger.warning(f"获取 {code} 财务数据失败: {e}")

        session.close()
        logger.info(f"财务数据初始化完成: {success_count}/{total}")
        return success_count

    def run(self, start_date: str, end_date: str):
        """执行完整初始化"""
        logger.info(f"=" * 50)
        logger.info(f"历史数据初始化开始")
        logger.info(f"日期范围: {start_date} ~ {end_date}")
        logger.info(f"=" * 50)

        # 1. 初始化股票元数据
        count = self.init_stock_meta()

        # 2. 获取股票列表
        session = self.Session()
        stock_codes = [r[0] for r in session.query(StockMeta.stock_code).filter(
            StockMeta.is_active == True
        ).all()]
        session.close()

        # 3. 初始化日线数据
        logger.info("开始初始化日线数据（这可能需要较长时间）...")
        self.init_stock_daily(stock_codes, start_date, end_date)

        # 4. 初始化指数数据
        logger.info("开始初始化指数数据...")
        index_codes = ['sh000001', 'sh000300', 'sz399001', 'sz399006']  # 沪深300, 上证, 深证, 创业板
        self.init_index_daily(index_codes, start_date, end_date)

        # 5. 初始化财务数据
        logger.info("开始初始化财务数据...")
        self.init_financial(stock_codes)

        logger.info(f"=" * 50)
        logger.info(f"历史数据初始化完成")
        logger.info(f"=" * 50)


def main():
    parser = argparse.ArgumentParser(description='历史数据初始化')
    parser.add_argument('--start', default='20210101', help='开始日期 YYYYMMDD')
    parser.add_argument('--end', default=None, help='结束日期 YYYYMMDD')
    args = parser.parse_args()

    end_date = args.end or datetime.now().strftime('%Y%m%d')

    client = AkshareClient(rate_limit=5)  # 降低速率避免限流
    initializer = HistoryInitializer(client)
    initializer.run(args.start, end_date)


if __name__ == '__main__':
    main()