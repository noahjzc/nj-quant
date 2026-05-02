"""信号生成器 — 基于 RotationConfig 的双层信号处理"""
import logging
from typing import List
import pandas as pd
from strategy.rotation.config import RotationConfig
from strategy.rotation.signal_engine.signal_filter import SignalFilter
from strategy.rotation.signal_engine.signal_ranker import SignalRanker

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    双层信号生成器

    Layer 1 (SignalFilter): 二值信号检测 — 技术指标金叉/死叉
    Layer 2 (SignalRanker): 多因子加权排序 — z-score + 因子方向调整

    Usage:
        config = RotationConfig()
        generator = SignalGenerator(config)
        buy_codes = generator.generate_buy_signals(df, date)
    """

    def __init__(self, config: RotationConfig):
        self.config = config
        self._signal_filter = SignalFilter(
            signal_types=config.buy_signal_types,
            mode=config.buy_signal_mode,
            kdj_low_threshold=config.kdj_low_threshold,
        )
        self._sell_filter = SignalFilter(
            signal_types=config.sell_signal_types,
            mode='OR',
            kdj_low_threshold=config.kdj_low_threshold,
        )
        self._ranker = SignalRanker(
            factor_weights=config.rank_factor_weights,
            factor_directions=config.rank_factor_directions,
        )

    def generate_buy_signals(self, df: pd.DataFrame, trade_date: pd.Timestamp, top_n: int = 10) -> List[str]:
        """
        在给定日期对全市场 DataFrame 生成买入信号

        Args:
            df: 包含多只股票历史数据的 DataFrame，必须包含 trade_date 列
            trade_date: 当前交易日
            top_n: 返回排序后的前 N 只股票

        Returns:
            按综合得分排序的股票代码列表（top_n）
        """
        # Layer 1: 二值信号过滤
        candidate_codes = self._scan_buy_candidates(df, trade_date)
        if not candidate_codes:
            logger.debug(f"[{trade_date.date()}] No buy signal candidates found")
            return []

        logger.debug(f"[{trade_date.date()}] Buy signal candidates: {candidate_codes}")

        # Layer 2: 多因子排序
        ranked = self._rank_candidates(df, candidate_codes, top_n)
        return ranked

    def generate_sell_signals(self, df: pd.DataFrame, trade_date: pd.Timestamp, position_codes: List[str]) -> List[dict]:
        """
        在给定日期对持仓股票生成卖出信号

        Args:
            df: 包含多只股票历史数据的 DataFrame，必须包含 trade_date 列
            trade_date: 当前交易日
            position_codes: 当前持仓股票代码列表

        Returns:
            触发卖出信号的股票列表，每项包含 stock_code 和 reason
        """
        sell_signals = []
        for stock_code in position_codes:
            stock_df = df[df['stock_code'] == stock_code].copy()
            if stock_df.empty:
                continue
            if self._sell_filter.filter_sell(stock_df, stock_code):
                sell_signals.append({'stock_code': stock_code, 'reason': 'sell_signal'})
        return sell_signals

    def _scan_buy_candidates(self, df: pd.DataFrame, trade_date: pd.Timestamp) -> List[str]:
        """
        Layer 1: 扫描全市场，过滤出有买入信号的股票
        """
        candidates = []
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            if self._signal_filter.filter_buy(stock_df, stock_code):
                candidates.append(stock_code)
        return candidates

    def _rank_candidates(self, df: pd.DataFrame, candidate_codes: List[str], top_n: int) -> List[str]:
        """
        Layer 2: 对候选股构建因子数据并排序
        """
        # 构建因子 DataFrame: index=stock_code, columns=因子值
        factor_cols = list(self.config.rank_factor_weights.keys())
        available_cols = [c for c in factor_cols if c in df.columns]

        rows = []
        for stock_code in candidate_codes:
            stock_df = df[df['stock_code'] == stock_code]
            if stock_df.empty:
                continue
            latest = stock_df.sort_values('trade_date').iloc[-1]
            row = {'stock_code': stock_code}
            for col in available_cols:
                row[col] = latest.get(col, 0.0)
            rows.append(row)

        if not rows:
            return []

        factor_df = pd.DataFrame(rows).set_index('stock_code')
        return self._ranker.rank(factor_df, top_n=top_n)
