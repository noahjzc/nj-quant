"""因子筛选与正交化 — 从 Alpha158 因子集中筛选有效因子"""
import json
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from strategy.factors.alpha158 import Alpha158Calculator

logger = logging.getLogger(__name__)


class FactorScreener:
    """从 Alpha158 因子集中筛选有效因子。"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.daily_dir = self.cache_dir / 'daily'
        self.calc = Alpha158Calculator()
        # 预计算因子列名（避免每次循环创建 dummy DataFrame）
        import pandas as pd
        dummy = pd.DataFrame({
            'open': [10.0]*10, 'high': [10.0]*10, 'low': [10.0]*10,
            'close': [10.0]*10, 'volume': [1e6]*10,
        })
        self._alpha_cols = list(self.calc.compute(dummy).columns)

    def _compute_cross_sectional_ic(
        self, factor_df: pd.DataFrame, forward_ret: pd.Series
    ) -> Dict[str, float]:
        """计算单日截面 Rank IC (Spearman)。"""
        common_idx = factor_df.index.intersection(forward_ret.index)
        if len(common_idx) < 30:
            return {}
        ic = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='An input array is constant')
            for col in factor_df.columns:
                valid = common_idx[
                    factor_df[col].loc[common_idx].notna() & forward_ret.loc[common_idx].notna()
                ]
                if len(valid) < 30:
                    continue
                r, _ = spearmanr(factor_df[col].loc[valid], forward_ret.loc[valid])
                ic[col] = 0.0 if np.isnan(r) else r
        return ic

    def compute_factor_ic(self, start: str, end: str, forward_days: int = 5) -> pd.DataFrame:
        """逐日计算所有因子的截面 Rank IC，返回汇总统计 DataFrame。"""
        all_dates = sorted([f.stem for f in self.daily_dir.glob('*.parquet')])
        if not all_dates:
            raise ValueError(f"缓存目录无数据: {self.daily_dir}")
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        dates = [d for d in all_dates if start <= d <= end]
        if len(dates) < 30:
            raise ValueError(f"日期范围不足: {start}~{end}, 仅{len(dates)}天")

        all_ic_records: Dict[str, list] = {}

        for i, date_str in enumerate(dates):
            daily_path = self.daily_dir / f'{date_str}.parquet'
            df = pd.read_parquet(daily_path)
            target_idx = date_to_idx[date_str] + forward_days
            if target_idx >= len(all_dates):
                continue
            target_date = all_dates[target_idx]
            target_path = self.daily_dir / f'{target_date}.parquet'
            if not target_path.exists():
                continue
            target_df = pd.read_parquet(target_path)
            merged = df.merge(
                target_df[['stock_code', 'close']],
                on='stock_code', suffixes=('', '_target'), how='inner'
            )
            if len(merged) < 100:
                continue
            forward_ret = pd.Series(
                (merged['close_target'].values - merged['close'].values) / merged['close'].values,
                index=merged['stock_code'].values
            )
            alpha_cols = [c for c in self._alpha_cols if c in df.columns]
            if not alpha_cols:
                continue
            factor_df = df.set_index('stock_code')[alpha_cols]
            day_ic = self._compute_cross_sectional_ic(factor_df, forward_ret)
            for factor, ic_val in day_ic.items():
                if factor not in all_ic_records:
                    all_ic_records[factor] = []
                all_ic_records[factor].append(ic_val)

            if (i + 1) % 50 == 0:
                logger.info(f"  IC计算进度: {i+1}/{len(dates)}")

        summary = []
        for factor, ic_list in all_ic_records.items():
            if len(ic_list) < 10:
                continue
            ic_arr = np.array(ic_list)
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr, ddof=1))
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            ic_pos = float(np.mean(ic_arr > 0))
            summary.append({
                'factor': factor, 'ic_mean': ic_mean, 'ic_std': ic_std,
                'icir': icir, 'ic_positive_ratio': ic_pos, 'n_obs': len(ic_list),
            })
        result = pd.DataFrame(summary).set_index('factor')
        result = result.sort_values('ic_mean', key=abs, ascending=False)
        logger.info(f"IC计算完成: {len(result)} 因子, "
                    f"|IC|>0.02 的有 {int((abs(result['ic_mean'])>0.02).sum())}")
        return result

    def screen_factors(
        self, ic_df: pd.DataFrame,
        min_abs_ic: float = 0.015, min_icir: float = 0.3, top_n_raw: int = 30,
    ) -> Tuple[List[str], List[str]]:
        """筛选因子，返回（原始精选, 正交化）。"""
        passed = ic_df[
            (abs(ic_df['ic_mean']) > min_abs_ic) & (ic_df['icir'] > min_icir)
        ]
        logger.info(f"IC筛选: {len(ic_df)} → {len(passed)} "
                    f"(|IC|>{min_abs_ic}, ICIR>{min_icir})")
        if len(passed) < 5:
            logger.warning("筛选后因子过少，放宽阈值")
            passed = ic_df.sort_values('ic_mean', key=abs, ascending=False).head(top_n_raw)
        raw_factors = passed.index[:top_n_raw].tolist()
        orth_factors = self._gram_schmidt_select(ic_df, raw_factors, min_abs_ic)
        logger.info(f"正交化后: {len(orth_factors)} 个因子")
        return raw_factors, orth_factors

    def _gram_schmidt_select(
        self, ic_df: pd.DataFrame, candidates: List[str], min_abs_ic: float
    ) -> List[str]:
        """Gram-Schmidt 正交化因子选择：已选因子越多，新因子需承担的独立信息门槛越高。"""
        if not candidates:
            return []
        sorted_factors = sorted(
            candidates, key=lambda f: abs(ic_df.loc[f, 'ic_mean']), reverse=True
        )
        selected = [sorted_factors[0]]
        for factor in sorted_factors[1:]:
            residual_ic = self._compute_residual_ic(factor, selected, ic_df)
            if abs(residual_ic) > min_abs_ic:
                selected.append(factor)
        return selected

    def _compute_residual_ic(
        self, factor: str, selected: List[str], ic_df: pd.DataFrame
    ) -> float:
        """估算因子对已选因子正交化后的残差 IC（近似）。"""
        ic_new = ic_df.loc[factor, 'ic_mean']
        penalty = max(0.3, 1.0 - 0.05 * len(selected))
        return ic_new * penalty

    def save_results(
        self, raw_factors: List[str], orth_factors: List[str],
        ic_df: pd.DataFrame, output_dir: str
    ) -> Tuple[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        factors_path = output / 'selected_factors.json'
        with open(factors_path, 'w', encoding='utf-8') as f:
            json.dump({
                'raw': raw_factors, 'orthogonal': orth_factors,
                'n_raw': len(raw_factors), 'n_orthogonal': len(orth_factors),
            }, f, indent=2, ensure_ascii=False)
        ic_path = output / 'factor_ic_report.csv'
        ic_df.to_csv(ic_path, encoding='utf-8-sig')
        logger.info(f"因子列表已保存: {factors_path}")
        logger.info(f"IC报告已保存: {ic_path}")
        return str(factors_path), str(ic_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='因子筛选与正交化')
    parser.add_argument('--start', required=True, help='分析开始日期')
    parser.add_argument('--end', required=True, help='分析结束日期')
    parser.add_argument('--cache-dir', default='cache/daily_rotation', help='Parquet缓存目录')
    parser.add_argument('--min-abs-ic', type=float, default=0.015)
    parser.add_argument('--min-icir', type=float, default=0.3)
    parser.add_argument('--top-n', type=int, default=30, help='原始精选因子数')
    parser.add_argument('--output', default='.', help='输出目录')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    screener = FactorScreener(args.cache_dir)
    ic_df = screener.compute_factor_ic(args.start, args.end)
    raw, orth = screener.screen_factors(ic_df, args.min_abs_ic, args.min_icir, args.top_n)
    screener.save_results(raw, orth, ic_df, args.output)
    print(f"\n原始精选 ({len(raw)}): {raw[:10]}...")
    print(f"正交化 ({len(orth)}): {orth[:10]}...")

if __name__ == '__main__':
    main()
