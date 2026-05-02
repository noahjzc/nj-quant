import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataMerger:
    """Merge intraday snapshot with historical data for signal generation."""

    @staticmethod
    def merge(
        history: pd.DataFrame,
        intraday: pd.DataFrame,
        today_date: pd.Timestamp,
    ) -> pd.DataFrame:
        history = history[history['trade_date'] < today_date].copy()

        today_rows = intraday.copy()
        today_rows['trade_date'] = today_date

        for col in ['volume', 'turnover_amount', 'amplitude', 'change_pct',
                     'turnover_rate', 'volume_ratio', 'pe_ttm', 'pb',
                     'circulating_mv', 'total_mv', 'stock_name']:
            if col not in today_rows.columns:
                today_rows[col] = None

        history_codes = set(history['stock_code'].unique())
        intraday_codes = set(intraday['stock_code'].unique())
        common_codes = history_codes & intraday_codes
        if len(common_codes) < len(intraday_codes):
            missing = intraday_codes - common_codes
            logger.info(f"Skipping {len(missing)} stocks without history data")

        today_rows = today_rows[today_rows['stock_code'].isin(common_codes)]

        history_cols = [c for c in history.columns if c in today_rows.columns or c == 'trade_date']
        history_subset = history[history_cols]

        merged = pd.concat([history_subset, today_rows], ignore_index=True)
        merged = merged.sort_values(['stock_code', 'trade_date']).reset_index(drop=True)

        logger.info(f"Merged: {len(history_subset)} history + {len(today_rows)} today = {len(merged)} rows")

        return merged
