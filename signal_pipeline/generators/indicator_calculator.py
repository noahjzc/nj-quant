import pandas as pd
import numpy as np


class IndicatorCalculator:
    """Vectorized technical indicator calculator for all stocks."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['stock_code', 'trade_date']).copy()
        grouped = df.groupby('stock_code', sort=False)

        # Moving Averages
        for window in [5, 10, 20, 30, 60]:
            df[f'ma_{window}'] = grouped['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        df['ma_cross'] = IndicatorCalculator._detect_cross(df, 'ma_5', 'ma_20')

        # MACD (12, 26, 9)
        df['ema_12'] = grouped['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        df['ema_26'] = grouped['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['macd_dif'] = df['ema_12'] - df['ema_26']
        df['macd_dea'] = grouped['macd_dif'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_hist'] = 2 * (df['macd_dif'] - df['macd_dea'])
        df['macd_cross'] = IndicatorCalculator._detect_cross(df, 'macd_dif', 'macd_dea')
        df.drop(['ema_12', 'ema_26'], axis=1, inplace=True)

        # KDJ (9, 3, 3)
        low_9 = grouped['low'].transform(lambda x: x.rolling(9, min_periods=1).min())
        high_9 = grouped['high'].transform(lambda x: x.rolling(9, min_periods=1).max())
        rsv = ((df['close'] - low_9) / (high_9 - low_9 + 1e-10)) * 100

        k_vals, d_vals = [], []
        for idx in grouped.groups.keys():
            grp_idx = grouped.get_group(idx).index
            grp_rsv = rsv.loc[grp_idx].reset_index(drop=True)
            k = pd.Series(50.0, index=range(len(grp_rsv)))
            d = pd.Series(50.0, index=range(len(grp_rsv)))
            for i in range(1, len(grp_rsv)):
                k.iloc[i] = 2/3 * k.iloc[i-1] + 1/3 * grp_rsv.iloc[i]
                d.iloc[i] = 2/3 * d.iloc[i-1] + 1/3 * k.iloc[i]
            k.index = grp_idx
            d.index = grp_idx
            k_vals.append(k)
            d_vals.append(d)
        df['kdj_k'] = pd.concat(k_vals).sort_index()
        df['kdj_d'] = pd.concat(d_vals).sort_index()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        df['kdj_cross'] = IndicatorCalculator._detect_cross(df, 'kdj_k', 'kdj_d')

        # Bollinger Bands (20, 2)
        df['boll_mid'] = grouped['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        std_20 = grouped['close'].transform(lambda x: x.rolling(20, min_periods=1).std())
        df['boll_upper'] = df['boll_mid'] + 2 * std_20
        df['boll_lower'] = df['boll_mid'] - 2 * std_20

        # RSI (6, 12, 24)
        for period, col in [(6, 'rsi_1'), (12, 'rsi_2'), (24, 'rsi_3')]:
            delta = grouped['close'].transform(lambda x: x.diff())
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.groupby(df['stock_code'], sort=False).transform(
                lambda x: x.ewm(span=period, adjust=False).mean()
            )
            avg_loss = loss.groupby(df['stock_code'], sort=False).transform(
                lambda x: x.ewm(span=period, adjust=False).mean()
            )
            rs = avg_gain / (avg_loss + 1e-10)
            df[col] = 100 - (100 / (1 + rs))

        # PSY (12)
        up_days = grouped['close'].transform(lambda x: (x.diff() > 0).rolling(12, min_periods=1).sum())
        df['psy'] = up_days / 12 * 100
        df['psyma'] = df.groupby('stock_code', sort=False)['psy'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )

        # Volume MAs
        df['vol_ma5'] = grouped['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['vol_ma20'] = grouped['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())

        # Close std (Bollinger width)
        df['close_std_20'] = grouped['close'].transform(lambda x: x.rolling(20, min_periods=1).std())

        # 20-day high max (exclude today)
        df['high_20_max'] = grouped['high'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).max())

        # ATR (14)
        prev_close = df.groupby('stock_code', sort=False)['close'].transform(lambda x: x.shift(1))
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.groupby(df['stock_code'], sort=False).transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )

        # Williams %R (10, 14)
        for period in [10, 14]:
            high_n = grouped['high'].transform(lambda x: x.rolling(period, min_periods=1).max())
            low_n = grouped['low'].transform(lambda x: x.rolling(period, min_periods=1).min())
            denom = high_n - low_n
            wr = pd.Series(-50.0, index=df.index)
            mask = denom > 0
            wr.loc[mask] = (high_n.loc[mask] - df.loc[mask, 'close']) / denom.loc[mask] * -100
            df[f'wr_{period}'] = wr

        # Returns
        df['ret_5'] = grouped['close'].transform(lambda x: x / x.shift(5) - 1)
        df['ret_20'] = grouped['close'].transform(lambda x: x / x.shift(20) - 1)

        # Fill NaN for insufficient history
        fill_cols = [
            'vol_ma5', 'vol_ma20', 'close_std_20', 'high_20_max',
            'atr_14', 'wr_10', 'wr_14', 'ret_5', 'ret_20',
            'boll_upper', 'boll_lower', 'rsi_1', 'rsi_2', 'rsi_3',
            'psy', 'psyma', 'kdj_k', 'kdj_d', 'kdj_j',
        ]
        df[fill_cols] = df[fill_cols].fillna(0.0)

        return df

    @staticmethod
    def _detect_cross(df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.Series:
        result = pd.Series('none', index=df.index, dtype=str)
        fast = df[fast_col]
        slow = df[slow_col]
        fast_prev = fast.groupby(df['stock_code'], sort=False).transform(lambda x: x.shift(1))
        slow_prev = slow.groupby(df['stock_code'], sort=False).transform(lambda x: x.shift(1))

        golden = (fast_prev <= slow_prev) & (fast > slow)
        death = (fast_prev >= slow_prev) & (fast < slow)
        result[golden] = 'golden_cross'
        result[death] = 'death_cross'
        return result