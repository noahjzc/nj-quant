"""抽样验证缓存预计算字段的准确性。

对随机抽样的股票/日期，用原始数据运行时计算各项指标，
与 Parquet 缓存中的预计算值逐字段比对。
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cache.daily_data_cache import DailyDataCache
from data.providers.data_provider import DataProvider


def compute_expected(df: pd.DataFrame) -> pd.DataFrame:
    """运行时计算各项指标（与 _precompute_stock_indicators 逻辑一致）"""
    df = df.sort_values('trade_date').copy()

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    expected = pd.DataFrame(index=df.index)
    expected['trade_date'] = df['trade_date']

    # Volume MAs
    expected['vol_ma5'] = volume.rolling(5, min_periods=1).mean()
    expected['vol_ma20'] = volume.rolling(20, min_periods=1).mean()

    # Close std (Bollinger width)
    expected['close_std_20'] = close.rolling(20, min_periods=1).std()

    # 20-day high max (exclude today via shift)
    expected['high_20_max'] = high.shift(1).rolling(20, min_periods=1).max()

    # ATR (14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    expected['atr_14'] = tr.rolling(14, min_periods=1).mean()

    # Williams %R (10, 14)
    for period in [10, 14]:
        high_n = high.rolling(period, min_periods=1).max()
        low_n = low.rolling(period, min_periods=1).min()
        denom = high_n - low_n
        wr = pd.Series(-50.0, index=df.index)
        mask = denom > 0
        wr[mask] = (high_n[mask] - close[mask]) / denom[mask] * -100
        expected[f'wr_{period}'] = wr

    # Returns
    expected['ret_5'] = close / close.shift(5) - 1
    expected['ret_20'] = close / close.shift(20) - 1

    # Fill NaN (matches precompute logic)
    new_cols = ['vol_ma5', 'vol_ma20', 'close_std_20', 'high_20_max',
                'atr_14', 'wr_10', 'wr_14', 'ret_5', 'ret_20']
    expected[new_cols] = expected[new_cols].fillna(0.0)

    return expected


def verify_cache(cache_dir: str, sample_stocks: int = 5, sample_dates: int = 5):
    """验证缓存预计算字段准确性"""
    cache = DailyDataCache(cache_dir)
    db = DataProvider()

    all_codes = cache.stock_codes
    all_dates = cache.trading_dates

    # 抽样：选中间段的股票和日期（避免首尾窗口不足的问题）
    rng = np.random.RandomState(42)
    sampled_codes = rng.choice(all_codes, min(sample_stocks, len(all_codes)), replace=False).tolist()
    # 取最后 N 个日期（数据最完整）
    sampled_dates = [d.strftime('%Y-%m-%d') for d in all_dates[-sample_dates:]]

    print(f"验证配置: {len(sampled_codes)} 只股票 x {len(sampled_dates)} 个日期 = {len(sampled_codes) * len(sampled_dates)} 次检查")
    print(f"股票: {sampled_codes}")
    print(f"日期: {sampled_dates}")
    print()

    precomputed_fields = ['vol_ma5', 'vol_ma20', 'close_std_20', 'high_20_max',
                          'atr_14', 'wr_10', 'wr_14', 'ret_5', 'ret_20']

    total_checks = 0
    mismatches = []

    for stock_code in sampled_codes:
        print(f"--- {stock_code} ---")

        # 从 DB 加载完整历史（包含回测首日前 30 天）
        min_date = sampled_dates[0]
        start_date = (pd.Timestamp(min_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = sampled_dates[-1]

        histories = db.get_batch_histories([stock_code], end_date=end_date, start_date=start_date)
        if stock_code not in histories or histories[stock_code].empty:
            print(f"  [跳过] DB 无数据")
            continue

        raw_df = histories[stock_code].reset_index()
        expected = compute_expected(raw_df)

        for date_str in sampled_dates:
            target_date = pd.Timestamp(date_str)
            # 从缓存读取当日数据
            day_df = cache.get_daily(date_str)
            if day_df.empty:
                print(f"  {date_str}: 缓存无数据")
                continue

            stock_row = day_df[day_df['stock_code'] == stock_code]
            if stock_row.empty:
                print(f"  {date_str}: 股票不在当日数据中")
                continue

            cached_row = stock_row.iloc[0]
            expected_row = expected[expected['trade_date'] == target_date]
            if expected_row.empty:
                print(f"  {date_str}: 预期数据无此日期")
                continue
            expected_row = expected_row.iloc[0]

            for field in precomputed_fields:
                total_checks += 1
                cached_val = cached_row.get(field, np.nan)
                expected_val = expected_row[field]

                # 处理 NaN 比较
                if pd.isna(cached_val) and pd.isna(expected_val):
                    continue
                if pd.isna(cached_val) or pd.isna(expected_val):
                    mismatches.append((stock_code, date_str, field, cached_val, expected_val))
                    continue

                if not np.isclose(float(cached_val), float(expected_val), rtol=1e-5, atol=1e-8):
                    mismatches.append((stock_code, date_str, field, cached_val, expected_val))

    # 报告
    print()
    print("=" * 60)
    print(f"验证完成: {total_checks} 次字段比对")

    if mismatches:
        print(f"不匹配: {len(mismatches)} 处")
        for stock, date, field, cached, expected in mismatches[:20]:
            print(f"  {stock} {date} {field}: 缓存={cached}, 预期={expected}")
        return False
    else:
        print("全部匹配!")
        return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default='cache/daily_rotation')
    parser.add_argument('--stocks', type=int, default=5)
    parser.add_argument('--dates', type=int, default=5)
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    # 新格式缓存: cache_dir/daily/ 直接存在
    # 旧格式: cache_dir/YYYY-MM-DD_YYYY-MM-DD/daily/
    if (cache_path / 'daily').is_dir():
        actual_cache = args.cache_dir
    else:
        subdirs = [d for d in cache_path.iterdir() if d.is_dir() and (d / 'daily').is_dir()]
        if subdirs:
            subdirs.sort(key=lambda d: d.name, reverse=True)
            actual_cache = str(subdirs[0])
        else:
            actual_cache = args.cache_dir
    print(f"使用缓存: {actual_cache}")

    ok = verify_cache(actual_cache, args.stocks, args.dates)
    sys.exit(0 if ok else 1)
