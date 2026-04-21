"""
每日增量更新脚本

用法:
    # 盘中更新（持仓股票）
    python back_testing/data/sync/daily_update.py --mode intraday --portfolio sh600519,sh600036

    # 收盘后更新（全市场）
    python back_testing/data/sync/daily_update.py --mode close
"""
import argparse
import logging
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, str(__file__).rsplit('back_testing', 1)[0])

from back_testing.data.db.connection import get_session
from back_testing.data.db.models import StockDaily, StockMeta, IndexDaily
from back_testing.data.sync.akshare_client import AkshareClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyUpdater:
    """每日数据更新器"""

    def __init__(self, client: AkshareClient):
        self.client = client
        self.Session = get_session()

    def get_last_trade_date(self, stock_code: str) -> date:
        """获取某股票最近交易日期"""
        session = self.Session()
        try:
            result = session.query(StockDaily.trade_date).filter(
                StockDaily.stock_code == stock_code
            ).order_by(StockDaily.trade_date.desc()).first()
            return result[0] if result else None
        finally:
            session.close()

    def get_all_active_stocks(self) -> list:
        """获取所有活跃股票"""
        session = self.Session()
        try:
            codes = [r[0] for r in session.query(StockMeta.stock_code).filter(
                StockMeta.is_active == True
            ).all()]
            return codes
        finally:
            session.close()

    def update_stock_daily(self, stock_code: str, start_date: date) -> bool:
        """更新单只股票日线数据"""
        start_str = start_date.strftime('%Y%m%d') if start_date else '20210101'
        today_str = datetime.now().strftime('%Y%m%d')

        try:
            df = self.client.get_stock_daily(stock_code, start_str, today_str)

            if df.empty:
                return False

            session = self.Session()
            try:
                for _, row in df.iterrows():
                    if row['trade_date'] <= start_date:
                        continue  # 跳过已有数据

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
                return True
            except Exception as e:
                session.rollback()
                logger.warning(f"更新 {stock_code} 失败: {e}")
                return False
            finally:
                session.close()

        except Exception as e:
            logger.warning(f"获取 {stock_code} 数据失败: {e}")
            return False

    def update_portfolio(self, portfolio: list) -> dict:
        """盘中更新持仓股票"""
        logger.info(f"盘中更新: {len(portfolio)} 只股票")
        results = {'success': 0, 'fail': 0}

        for code in portfolio:
            last_date = self.get_last_trade_date(code)
            if last_date is None:
                last_date = date.today() - timedelta(days=30)

            if self.update_stock_daily(code, last_date):
                results['success'] += 1
            else:
                results['fail'] += 1

        logger.info(f"盘中更新完成: 成功 {results['success']}, 失败 {results['fail']}")
        return results

    def update_full_market(self) -> dict:
        """收盘后全市场更新"""
        logger.info("开始全市场收盘后更新...")

        # 获取所有活跃股票
        stocks = self.get_all_active_stocks()
        logger.info(f"全市场股票数量: {len(stocks)}")

        # 计算起始日期（获取最近30天数据，确保不遗漏）
        start_date = date.today() - timedelta(days=30)

        results = {'success': 0, 'fail': 0}

        for i, code in enumerate(stocks):
            if (i + 1) % 100 == 0:
                logger.info(f"进度: {i + 1}/{len(stocks)}")

            if self.update_stock_daily(code, start_date):
                results['success'] += 1
            else:
                results['fail'] += 1

        # 更新指数数据
        index_codes = ['sh000001', 'sh000300', 'sz399001', 'sz399006']
        self.update_index_daily(index_codes, start_date)

        logger.info(f"全市场更新完成: 成功 {results['success']}, 失败 {results['fail']}")
        return results

    def update_index_daily(self, index_codes: list, start_date: date) -> bool:
        """更新指数数据"""
        start_str = start_date.strftime('%Y%m%d')
        today_str = datetime.now().strftime('%Y%m%d')

        session = self.Session()
        try:
            for code in index_codes:
                try:
                    df = self.client.get_index_daily(code, start_str, today_str)

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
                    logger.info(f"指数 {code} 更新完成")

                except Exception as e:
                    session.rollback()
                    logger.warning(f"指数 {code} 更新失败: {e}")

            return True

        finally:
            session.close()

    def run_intraday(self, portfolio: list):
        """运行盘中更新"""
        logger.info(f"=" * 50)
        logger.info("盘中数据更新开始")
        logger.info(f"时间: {datetime.now()}")
        logger.info(f"=" * 50)

        self.update_portfolio(portfolio)

        logger.info(f"=" * 50)
        logger.info("盘中数据更新完成")
        logger.info(f"=" * 50)

    def run_close(self):
        """运行收盘后更新"""
        logger.info(f"=" * 50)
        logger.info("收盘后数据更新开始")
        logger.info(f"时间: {datetime.now()}")
        logger.info(f"=" * 50)

        self.update_full_market()

        logger.info(f"=" * 50)
        logger.info("收盘后数据更新完成")
        logger.info(f"=" * 50)


def main():
    parser = argparse.ArgumentParser(description='每日数据更新')
    parser.add_argument('--mode', choices=['intraday', 'close'], required=True,
                        help='更新模式: intraday=盘中, close=收盘后')
    parser.add_argument('--portfolio', type=str, default=None,
                        help='持仓股票列表，逗号分隔，如 sh600519,sh600036')
    args = parser.parse_args()

    client = AkshareClient(rate_limit=10)
    updater = DailyUpdater(client)

    if args.mode == 'intraday':
        portfolio = args.portfolio.split(',') if args.portfolio else []
        if not portfolio:
            logger.error("盘中模式需要指定 --portfolio 参数")
            sys.exit(1)
        updater.run_intraday(portfolio)
    else:
        updater.run_close()


if __name__ == '__main__':
    main()