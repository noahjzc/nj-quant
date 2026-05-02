"""
Alpha158 因子计算器

基于 Microsoft Qlib Alpha158 因子定义，用 numpy/pandas 直接实现。
无需安装 Qlib 依赖。

因子分类:
- KBar (9):  单日 OHLC 蜡烛形态
- Price (2):  价格比率 (open/close, high/close, low/close)
              注: vwap/close 因 nj-quant 无 vwap 字段暂跳过
- Rolling (145): 5窗口 [5,10,20,30,60] × 29种算子

总计: 156 个因子

Qlib 关键语义:
- Greater(a, b) = np.maximum(a, b), NOT a > b
- Less(a, b)    = np.minimum(a, b), NOT a < b
- Ref(x, N): N>0 滞后, N<0 前瞻 (仅用于 label)
- 所有 rolling 操作 min_periods=1
- 因子值已除以 close 或 volume 做归一化，消除量纲

用法:
    from strategy.factors.alpha158 import Alpha158Calculator
    calc = Alpha158Calculator()
    factors_df = calc.compute(df)  # df 需含 open/high/low/close/volume 列
"""

import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20, 30, 60]


class Alpha158Calculator:
    """Alpha158 因子计算器 — 纯 numpy/pandas 实现"""

    def __init__(self, windows: list = None):
        self.windows = windows or WINDOWS

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有 Alpha158 因子。

        Args:
            df: 单只股票的 OHLCV DataFrame，需含 open/high/low/close/volume 列

        Returns:
            (n_rows, 156) 因子 DataFrame，index 与 df 对齐
        """
        o, c, h, l, v = df['open'], df['close'], df['high'], df['low'], df['volume']

        factors = {}

        # ── KBar (9) ──
        self._compute_kbar(o, c, h, l, factors)

        # ── Price (3) ──
        factors['OPEN0'] = o / c
        factors['HIGH0'] = h / c
        factors['LOW0'] = l / c

        # ── Rolling (144) ──
        delta = c.diff()
        for d in self.windows:
            self._compute_rolling(d, c, h, l, v, delta, factors)

        result = pd.DataFrame(factors, index=df.index)
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        return result

    # ═══════════════════════════════════════════════════════════
    # KBar
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _compute_kbar(o, c, h, l, factors: dict):
        hl_range = h - l + 1e-12
        factors['KMID'] = (c - o) / o
        factors['KLEN'] = (h - l) / o
        factors['KMID2'] = (c - o) / hl_range
        factors['KUP'] = (h - np.maximum(o, c)) / o
        factors['KUP2'] = (h - np.maximum(o, c)) / hl_range
        factors['KLOW'] = (np.minimum(o, c) - l) / o
        factors['KLOW2'] = (np.minimum(o, c) - l) / hl_range
        factors['KSFT'] = (2 * c - h - l) / o
        factors['KSFT2'] = (2 * c - h - l) / hl_range

    # ═══════════════════════════════════════════════════════════
    # Rolling
    # ═══════════════════════════════════════════════════════════

    def _compute_rolling(self, d: int, c, h, l, v, delta, factors: dict):
        su = str(d)

        # ── ROC / MA / STD ──
        factors[f'ROC{su}'] = c.shift(d) / c
        factors[f'MA{su}'] = c.rolling(d, min_periods=1).mean() / c
        factors[f'STD{su}'] = c.rolling(d, min_periods=1).std(ddof=0) / c

        # ── BETA (Slope) / RSQR / RESI ──
        slope, rsqr, resi = self._rolling_ols(c, d)
        factors[f'BETA{su}'] = slope / c
        factors[f'RSQR{su}'] = rsqr
        factors[f'RESI{su}'] = resi / c

        # ── MAX / MIN ──
        factors[f'MAX{su}'] = h.rolling(d, min_periods=1).max() / c
        factors[f'MIN{su}'] = l.rolling(d, min_periods=1).min() / c

        # ── QTLU / QTLD ──
        factors[f'QTLU{su}'] = c.rolling(d, min_periods=1).quantile(0.8) / c
        factors[f'QTLD{su}'] = c.rolling(d, min_periods=1).quantile(0.2) / c

        # ── RANK ──
        factors[f'RANK{su}'] = self._rolling_rank(c, d)

        # ── RSV ──
        lo_min = l.rolling(d, min_periods=1).min()
        hi_max = h.rolling(d, min_periods=1).max()
        factors[f'RSV{su}'] = (c - lo_min) / (hi_max - lo_min + 1e-12)

        # ── IMAX / IMIN / IMXD ──
        imax = h.rolling(d, min_periods=1).apply(lambda x: float(x.argmax() + 1), raw=True)
        imin = l.rolling(d, min_periods=1).apply(lambda x: float(x.argmin() + 1), raw=True)
        factors[f'IMAX{su}'] = imax / d
        factors[f'IMIN{su}'] = imin / d
        factors[f'IMXD{su}'] = (imax - imin) / d

        # ── CORR / CORD ──
        c_std = c.rolling(d, min_periods=1).std(ddof=0)
        v_log = np.log(v + 1)
        v_log_std = v_log.rolling(d, min_periods=1).std(ddof=0)
        corr_val = c.rolling(d, min_periods=1).corr(v_log)
        corr_val[corr_val.isna()] = np.nan
        factors[f'CORR{su}'] = corr_val

        ret = c / c.shift(1)
        vol_ret = np.log(v / v.shift(1) + 1)
        cord_val = ret.rolling(d, min_periods=1).corr(vol_ret)
        cord_val[cord_val.isna()] = np.nan
        factors[f'CORD{su}'] = cord_val

        # ── CNTP / CNTN / CNTD ──
        up = (c > c.shift(1)).rolling(d, min_periods=1).mean()
        down = (c < c.shift(1)).rolling(d, min_periods=1).mean()
        factors[f'CNTP{su}'] = up
        factors[f'CNTN{su}'] = down
        factors[f'CNTD{su}'] = up - down

        # ── SUMP / SUMN / SUMD ──
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        total_abs = np.abs(delta).rolling(d, min_periods=1).sum() + 1e-12
        factors[f'SUMP{su}'] = gain.rolling(d, min_periods=1).sum() / total_abs
        factors[f'SUMN{su}'] = loss.rolling(d, min_periods=1).sum() / total_abs
        factors[f'SUMD{su}'] = (gain.rolling(d, min_periods=1).sum() -
                                loss.rolling(d, min_periods=1).sum()) / total_abs

        # ── VMA / VSTD ──
        factors[f'VMA{su}'] = v.rolling(d, min_periods=1).mean() / (v + 1e-12)
        factors[f'VSTD{su}'] = v.rolling(d, min_periods=1).std(ddof=0) / (v + 1e-12)

        # ── WVMA ──
        wc = np.abs(c / c.shift(1) - 1) * v
        wc_mean = wc.rolling(d, min_periods=1).mean() + 1e-12
        wc_std = wc.rolling(d, min_periods=1).std(ddof=0)
        factors[f'WVMA{su}'] = wc_std / wc_mean

        # ── VSUMP / VSUMN / VSUMD ──
        dvol = v.diff()
        vgain = np.maximum(dvol, 0)
        vloss = np.maximum(-dvol, 0)
        vtotal = np.abs(dvol).rolling(d, min_periods=1).sum() + 1e-12
        factors[f'VSUMP{su}'] = vgain.rolling(d, min_periods=1).sum() / vtotal
        factors[f'VSUMN{su}'] = vloss.rolling(d, min_periods=1).sum() / vtotal
        factors[f'VSUMD{su}'] = (vgain.rolling(d, min_periods=1).sum() -
                                 vloss.rolling(d, min_periods=1).sum()) / vtotal

    # ═══════════════════════════════════════════════════════════
    # OLS helpers (Slope / Rsquare / Resi)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _rolling_ols(series: pd.Series, window: int):
        """回归 series 对位置索引 [1..window]，返回 (slope, rsquare, residual)"""
        n = len(series)
        slope = np.full(n, np.nan)
        rsqr = np.full(n, np.nan)
        resi = np.full(n, np.nan)

        x = np.arange(1, window + 1, dtype=float)

        for i in range(window - 1, n):
            y = series.iloc[i - window + 1:i + 1].values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                continue
            yv, xv = y[mask], x[mask]
            N = len(yv)
            sx = xv.sum()
            sy = yv.sum()
            sxx = (xv * xv).sum()
            sxy = (xv * yv).sum()
            denom = N * sxx - sx * sx
            if abs(denom) < 1e-12:
                continue
            sl = (N * sxy - sx * sy) / denom
            intercept = (sy - sl * sx) / N
            slope[i] = sl
            # R²
            y_pred = sl * xv + intercept
            ss_res = ((yv - y_pred) ** 2).sum()
            ss_tot = ((yv - yv.mean()) ** 2).sum()
            rsqr[i] = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            # Residual = last value - predicted last value
            resi[i] = yv[-1] - (sl * window + intercept)

        return (
            pd.Series(slope, index=series.index),
            pd.Series(rsqr, index=series.index),
            pd.Series(resi, index=series.index),
        )

    # ═══════════════════════════════════════════════════════════
    # Rank helper
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _rolling_rank(series: pd.Series, window: int):
        """滚动百分位排名: 当前值在窗口内的分位数 (0~1)"""
        from scipy.stats import percentileofscore

        n = len(series)
        result = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i - window + 1)
            y = series.iloc[start:i + 1].values
            mask = ~np.isnan(y)
            if mask.sum() == 0 or np.isnan(y[-1]):
                continue
            result[i] = percentileofscore(y[mask], y[-1]) / 100.0
        return pd.Series(result, index=series.index)
