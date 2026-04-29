"""每日全市场轮动回测核心引擎"""
import time
import pandas as pd
import numpy as np
import logging
import optuna
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from back_testing.data.data_provider import DataProvider

logger = logging.getLogger(__name__)
from back_testing.rotation.config import RotationConfig


def compute_overheat(
    rsi: float,
    ret5: float,
    rsi_threshold: float = 75.0,
    ret5_threshold: float = 0.15
) -> float:
    """计算过热度（0~1）。仅当 RSI 和短期涨幅双高时返回正值。"""
    if not (rsi > rsi_threshold and ret5 > ret5_threshold):
        return 0.0

    rsi_component = max(0.0, (rsi - rsi_threshold) / (100 - rsi_threshold))
    ret_component = min(1.0, max(0.0, (ret5 - ret5_threshold) / 0.35))
    return (rsi_component + ret_component) / 2.0
from back_testing.rotation.signal_engine.signal_filter import SignalFilter
from back_testing.rotation.signal_engine.signal_ranker import SignalRanker
from back_testing.rotation.market_regime import MarketRegime
from back_testing.rotation.position_manager import RotationPositionManager
from back_testing.rotation.trade_executor import TradeExecutor, TradeRecord
from back_testing.risk.stop_loss_strategies import StopLossStrategies


@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    shares: int
    buy_price: float
    buy_date: str
    highest_price: float = 0.0  # 持仓期间最高价（用于移动止损）


@dataclass
class DailyResult:
    """每日结果"""
    date: str
    total_asset: float
    cash: float
    positions: Dict[str, Position]
    trades: List[TradeRecord]
    market_regime: str
    portfolio_value: float  # = total_asset


class DailyRotationEngine:
    """
    每日全市场轮动回测引擎

    每日流程：
    1. 获取当日全市场日线数据
    2. 过滤股票池（ST、涨跌停）
    3. 检查持仓 → 卖出信号 → 卖出
    4. 扫描全市场 → 买入信号 → 候选股
    5. 多因子排序 → 买入 TOP X
    6. 记录每日净值和交易
    """

    def __init__(self, config: RotationConfig, start_date: str, end_date: str,
                 data_provider=None):
        """Initialize engine with rolling pointer data model.

        Args:
            config: Strategy configuration.
            start_date / end_date: Backtest date range (YYYY-MM-DD).
            data_provider: Data source (CachedProvider with precomputed columns required).
        """
        self.config = config
        self.start_date = start_date
        self.end_date = end_date

        # ── Data source ──
        self.data_provider = data_provider or DataProvider()

        # ── Subsystems ──
        self.position_manager = RotationPositionManager(
            total_capital=config.initial_capital,
            max_total_pct=config.max_total_pct,
            max_position_pct=config.max_position_pct
        )
        self.trade_executor = TradeExecutor()
        self.buy_filter = SignalFilter(config.buy_signal_types, mode=config.buy_signal_mode,
                                        kdj_low_threshold=config.kdj_low_threshold)
        self.sell_filter = SignalFilter(config.sell_signal_types)
        self.ranker = SignalRanker(config.rank_factor_weights, config.rank_factor_directions)
        self.market_regime = MarketRegime(config.market_regime, self.data_provider)

        # ── ATR stop parameters ──
        self.atr_period = config.atr_period
        self.stop_loss_mult = config.stop_loss_mult
        self.take_profit_mult = config.take_profit_mult
        self.trailing_pct = config.trailing_pct
        self.trailing_start = config.trailing_start

        # ── Runtime state ──
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.daily_results: List[DailyResult] = []
        self.trade_history: List[TradeRecord] = []

        # ── Rolling pointer: prev/current day DataFrames (~4760 rows each) ──
        self._prev_df: pd.DataFrame = pd.DataFrame()
        self._today_df: pd.DataFrame = pd.DataFrame()

        # ── Diagnostic counters ──
        self._n_stock_df_calls = 0
        self._fallback_count = 0

    def run(self, trial=None) -> List[DailyResult]:
        """Run the backtest main loop.

        Flow:
        1. Get trading dates from benchmark index.
        2. Initialize _prev_df from the trading day before first_date.
        3. For each date: advance data pointer → run single day logic → record results.

        Args:
            trial: Optional Optuna Trial for early pruning. When provided and
                   total_asset falls below min_asset_ratio, raises TrialPruned.
        """
        dates = self._get_trading_dates()
        n_dates = len(dates)
        if n_dates < 2:
            return []

        # Initialize _prev_df: load trading day just before first_date
        self._init_prev_cache(dates[0])

        now = datetime.now
        print(f"{now():%H:%M:%S} [DailyRotation] {self.start_date} ~ {self.end_date}, {n_dates}天")

        # ── Timing diagnostics ──
        t_total = 0.0
        t_advance = 0.0
        t_regime = 0.0
        t_filter = 0.0
        t_get_prices = 0.0
        t_check_sell = 0.0
        t_scan_buy = 0.0
        t_execute_buy = 0.0
        t_stock_df = 0.0
        n_stock_df_calls = 0

        # ── Day-by-day loop ──
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            if i == 0 or (i + 1) % 10 == 0:
                prev_asset = self.daily_results[-1].total_asset if self.daily_results else self.config.initial_capital
                if i > 0:
                    avg_day = t_total / i * 1000
                    fb = f" FB:{self._fallback_count}" if self._fallback_count else ""
                    print(f"{now():%H:%M:%S}   [{i+1}/{n_dates}] {date_str} | 持仓:{len(self.positions)} | 资产:{prev_asset:,.0f} | "
                          f"avg:{avg_day:.0f}ms/d | adv:{t_advance*1000:.0f}ms reg:{t_regime*1000:.0f}ms "
                          f"filt:{t_filter*1000:.0f}ms prc:{t_get_prices*1000:.0f}ms "
                          f"sell:{t_check_sell*1000:.0f}ms scan:{t_scan_buy*1000:.0f}ms buy:{t_execute_buy*1000:.0f}ms "
                          f"stk:{t_stock_df*1000:.0f}ms({n_stock_df_calls}){fb}")
                else:
                    print(f"{now():%H:%M:%S}   [{i+1}/{n_dates}] {date_str} | 持仓:{len(self.positions)} | 资产:{prev_asset:,.0f}")

            t0 = time.perf_counter()
            self._advance_to_date(date)
            t_advance += time.perf_counter() - t0

            t0 = time.perf_counter()
            result, day_stats = self._run_single_day(date)
            t_total += time.perf_counter() - t0

            t_regime += day_stats.get('regime', 0)
            t_filter += day_stats.get('filter', 0)
            t_get_prices += day_stats.get('get_prices', 0)
            t_check_sell += day_stats.get('check_sell', 0)
            t_scan_buy += day_stats.get('scan_buy', 0)
            t_execute_buy += day_stats.get('execute_buy', 0)
            t_stock_df += day_stats.get('stock_df', 0)
            n_stock_df_calls += day_stats.get('n_stock_df', 0)

            self.daily_results.append(result)

            # Early termination: total_asset below min_asset_ratio * initial_capital
            if result.total_asset < self.config.initial_capital * self.config.min_asset_ratio:
                if trial is not None:
                    trial.report(result.total_asset / self.config.initial_capital, i)
                    raise optuna.TrialPruned(
                        f"第{i+1}天资产 {result.total_asset:,.0f} < "
                        f"初始 {self.config.initial_capital * self.config.min_asset_ratio:,.0f}"
                    )
                else:
                    return self.daily_results

        final_asset = self.daily_results[-1].total_asset if self.daily_results else self.current_capital
        avg_ms = t_total / n_dates * 1000
        fb = f" | 降级:{self._fallback_count}次" if self._fallback_count else ""
        print(f"{datetime.now():%H:%M:%S} [DailyRotation] 回测完成，最终资产: {final_asset:,.0f} | {avg_ms:.0f}ms/天 | "
              f"stk_df:{n_stock_df_calls}次{t_stock_df*1000:.0f}ms{fb}")
        return self.daily_results

    def _log_daily_summary(
        self,
        date_str: str,
        regime_name: str,
        total_asset: float,
        cash: float,
        n_positions: int,
        buy_candidates: List[str],
        top_stocks_info: List[Dict],
        all_trades: List[TradeRecord]
    ):
        """每日汇总日志"""
        position_value = total_asset - cash

        logger.info(
            f"[DAY] {date_str} | 市场:{regime_name} | 候选:{len(buy_candidates)}只 "
            f"| 买信号:{len(top_stocks_info)}只 | 卖信号:{sum(1 for t in all_trades if t.action=='SELL')}笔 "
            f"| 资产:{total_asset:,.0f} | 现金:{cash:,.0f} | 持仓:{position_value:,.0f} | {n_positions}只持仓"
        )

        if top_stocks_info:
            top_codes = [s['stock_code'] for s in top_stocks_info]
            logger.info(f"[TOP] 买入候选排名: {top_codes}")

    def _run_single_day(self, date: pd.Timestamp):
        """每日流程。返回 (DailyResult, day_stats_dict)。"""
        date_str = date.strftime('%Y-%m-%d')
        stats = {}  # 累积当日各子操作耗时

        # Step 0: 获取大盘状态，动态调整参数
        t0 = time.perf_counter()
        regime_name, regime_params = self.market_regime.get_regime(date)
        stats['regime'] = time.perf_counter() - t0

        self.position_manager.max_total_pct = regime_params.max_total_pct
        self.position_manager.max_position_pct = regime_params.max_position_pct
        max_positions = regime_params.max_positions

        # 当日无数据 → 返回空结果
        if self._today_df.empty:
            stats.update({'filter': 0, 'get_prices': 0, 'check_sell': 0,
                          'scan_buy': 0, 'execute_buy': 0, 'stock_df': 0, 'n_stock_df': 0})
            return DailyResult(date_str, self.current_capital, self.current_capital, self.positions, [], regime_name, self.current_capital), stats

        # 过滤股票池（向量化在 _today_df 上）
        t0 = time.perf_counter()
        valid_codes = self._filter_stock_pool()
        stats['filter'] = time.perf_counter() - t0

        # 获取持仓快照（代码→当前价）— 从 _today_df 直接读取，零 groupby
        t0 = time.perf_counter()
        current_prices = self._get_current_prices(valid_codes)
        stats['get_prices'] = time.perf_counter() - t0

        # Step 1: 更新持仓期间最高价
        for stock_code, position in self.positions.items():
            current_price = current_prices.get(stock_code, 0.0)
            if current_price > position.highest_price:
                position.highest_price = current_price

        # Step 2: 检查持仓卖出信号
        t_stock_start = time.perf_counter()
        n_stock_before = self._n_stock_df_calls if hasattr(self, '_n_stock_df_calls') else 0
        t0 = time.perf_counter()
        sell_trades = self._check_and_sell(date_str, valid_codes, current_prices)
        stats['check_sell'] = time.perf_counter() - t0

        # 更新现金（卖出）
        for trade in sell_trades:
            self.current_capital += trade.shares * trade.price - trade.cost

        # Step 2: 扫描买入信号（排除当日已卖出的股票，防止同日来回交易）
        sold_today = [t.stock_code for t in sell_trades]
        t0 = time.perf_counter()
        buy_candidates = self._scan_buy_candidates(valid_codes, exclude_codes=sold_today)
        stats['scan_buy'] = time.perf_counter() - t0

        # Step 3: 重新计算 total_asset（此时包含卖出后的现金更新）
        total_asset = self.current_capital + self.position_manager.get_position_value(
            {p.stock_code: p.shares for p in self.positions.values()},
            current_prices
        )
        self.position_manager.update_capital(total_asset)

        # Step 4: 多因子排序，买入 TOP X
        t0 = time.perf_counter()
        buy_trades, top_stocks_info = self._execute_buy(date_str, valid_codes, buy_candidates, max_positions, current_prices, total_asset)
        stats['execute_buy'] = time.perf_counter() - t0

        # 统计当日 _get_stock_df 调用次数与耗时
        n_stock_after = self._n_stock_df_calls if hasattr(self, '_n_stock_df_calls') else 0
        stats['n_stock_df'] = n_stock_after - n_stock_before
        stats['stock_df'] = time.perf_counter() - t_stock_start - stats['check_sell'] - stats['scan_buy'] - stats['execute_buy']

        all_trades = sell_trades + buy_trades

        # 重新计算 total_asset，反映买入后的实际资产（含手续费扣减）
        total_asset = self.current_capital + self.position_manager.get_position_value(
            {p.stock_code: p.shares for p in self.positions.values()},
            current_prices
        )

        # 每日日志
        self._log_daily_summary(date_str, regime_name, total_asset, self.current_capital,
                                 len(self.positions), buy_candidates, top_stocks_info, all_trades)

        return DailyResult(
            date=date_str,
            total_asset=total_asset,
            cash=self.current_capital,
            positions={p.stock_code: p for p in self.positions.values()},
            trades=all_trades,
            market_regime=regime_name,
            portfolio_value=total_asset
        ), stats

    def _check_and_sell(
        self,
        date_str: str,
        valid_codes: set,
        current_prices: Dict[str, float]
    ) -> List[TradeRecord]:
        """检查持仓是否有卖出信号 — 仅在持仓股上按需构建 DataFrame"""
        sell_trades = []
        positions_to_close = []

        for stock_code, position in list(self.positions.items()):
            df = self._get_stock_df(stock_code)
            if stock_code not in valid_codes:
                # 停牌/退市：从 prev 缓存取最后收盘价平仓
                price = 0.0
                if not df.empty:
                    price = float(df['close'].iloc[-1])
                if price > 0:
                    shares, cost = self.trade_executor.execute_sell(stock_code, price, position.shares)
                    if shares > 0:
                        buy_price = position.buy_price
                        holding_days = (pd.Timestamp(date_str) - pd.Timestamp(position.buy_date)).days
                        return_pct = (price - buy_price) / buy_price * 100 if buy_price > 0 else 0
                        pnl = (price - buy_price) * shares - cost
                        capital_before_sell = self.current_capital

                        trade = TradeRecord(
                            date=date_str,
                            stock_code=stock_code,
                            action='SELL',
                            price=price,
                            shares=shares,
                            cost=cost,
                            capital_before=capital_before_sell
                        )
                        sell_trades.append(trade)
                        self.trade_history.append(trade)
                        del self.positions[stock_code]

                        logger.info(
                            f"[SELL/SUSPENDED] {date_str} {stock_code} @ {price:.3f} x {shares}股 "
                            f"买价:{buy_price:.3f} 持有:{holding_days}天 收益:{return_pct:+.2f}% "
                            f"PnL:{pnl:+,.0f} (卖前现金:{capital_before_sell:,.0f})"
                        )
                continue

            if df.empty or len(df) < 2:
                continue

            if self.sell_filter.filter_sell(df, stock_code):
                if stock_code not in positions_to_close:
                    positions_to_close.append(stock_code)

            # ATR 止损/止盈/移动止损检查
            current_price = current_prices.get(stock_code, 0.0)
            if current_price > 0:
                atr = float(df['atr_14'].iloc[-1]) if 'atr_14' in df.columns else 0.0
                if atr > 0:
                    exit_result = StopLossStrategies.check_exit(
                        position={'buy_price': position.buy_price},
                        current_price=current_price,
                        atr=atr,
                        highest_price=position.highest_price,
                        stop_loss_mult=self.stop_loss_mult,
                        take_profit_mult=self.take_profit_mult,
                        trailing_pct=self.trailing_pct,
                        trailing_start=self.trailing_start,
                    )
                    if exit_result['action'] in ('stop_loss', 'trailing_stop'):
                        if stock_code not in positions_to_close:
                            positions_to_close.append(stock_code)
                        logger.info(
                            f"[EXIT] {date_str} {stock_code} @ {current_price:.3f} "
                            f"原因:{exit_result['reason']}"
                        )

        for stock_code in positions_to_close:
            position = self.positions[stock_code]
            price = current_prices.get(stock_code, 0.0)
            if price <= 0:
                continue

            shares, cost = self.trade_executor.execute_sell(stock_code, price, position.shares)
            if shares > 0:
                # 计算持有收益
                buy_price = position.buy_price
                holding_days = (pd.Timestamp(date_str) - pd.Timestamp(position.buy_date)).days
                return_pct = (price - buy_price) / buy_price * 100 if buy_price > 0 else 0
                pnl = (price - buy_price) * shares - cost
                capital_before_sell = self.current_capital

                trade = TradeRecord(
                    date=date_str,
                    stock_code=stock_code,
                    action='SELL',
                    price=price,
                    shares=shares,
                    cost=cost,
                    capital_before=capital_before_sell
                )
                sell_trades.append(trade)
                self.trade_history.append(trade)
                del self.positions[stock_code]

                logger.info(
                    f"[SELL] {date_str} {stock_code} @ {price:.3f} x {shares}股 "
                    f"买价:{buy_price:.3f} 持有:{holding_days}天 收益:{return_pct:+.2f}% "
                    f"PnL:{pnl:+,.0f} (卖前现金:{capital_before_sell:,.0f})"
                )

        return sell_trades

    def _scan_buy_candidates(self, valid_codes: set, exclude_codes: List[str] = None) -> List[str]:
        """扫描全市场，返回有买入信号的股票代码（向量化）"""
        exclude_set = set(exclude_codes) if exclude_codes else set()
        codes = [c for c in valid_codes if c not in self.positions and c not in exclude_set]
        if not codes:
            return []

        # 构建特征矩阵（每行一只股票，列为信号检测所需指标）
        try:
            features = self._build_signal_features(codes)
        except Exception:
            # 回退到逐股检测 — 极慢（每只股票 ~0.1ms × N）
            self._fallback_count = getattr(self, '_fallback_count', 0) + 1
            if self._fallback_count <= 3:
                logger.warning(f"[SLOW-PATH] _build_signal_features 失败，降级到逐股检测 ({len(codes)} 只股票)")
            candidates = []
            for stock_code in codes:
                df = self._get_stock_df(stock_code)
                if df.empty or len(df) < 2:
                    continue
                if self.buy_filter.filter_buy(df, stock_code):
                    candidates.append(stock_code)
            return candidates

        if features.empty:
            return []

        # 向量化信号检测
        active_signals = set(self.config.buy_signal_types)
        mode = self.config.buy_signal_mode
        masks = {}
        f = features

        if 'KDJ_GOLD' in active_signals:
            masks['KDJ_GOLD'] = (f['kdj_k'] > f['kdj_d']) & (f['kdj_k_p'] <= f['kdj_d_p'])
        if 'MACD_GOLD' in active_signals:
            masks['MACD_GOLD'] = (f['macd_dif'] > f['macd_dea']) & (f['macd_dif_p'] <= f['macd_dea_p'])
        if 'MA_GOLD' in active_signals:
            masks['MA_GOLD'] = (f['ma_5'] > f['ma_20']) & (f['ma_5_p'] <= f['ma_20_p'])
        if 'VOL_GOLD' in active_signals:
            masks['VOL_GOLD'] = (f['vol_ma5'] > f['vol_ma20']) & (f['vol_ma5_p'] <= f['vol_ma20_p'])
        if 'BOLL_BREAK' in active_signals:
            boll_upper = f['boll_mid'] + 2 * f['close_std_20']
            masks['BOLL_BREAK'] = f['close'] > boll_upper
        if 'HIGH_BREAK' in active_signals:
            masks['HIGH_BREAK'] = f['close'] >= f['high_20_max']
        if 'KDJ_GOLD_LOW' in active_signals:
            k_thresh = self.config.kdj_low_threshold
            masks['KDJ_GOLD_LOW'] = (
                (f['kdj_k'] > f['kdj_d']) & (f['kdj_k_p'] <= f['kdj_d_p']) & (f['kdj_k'] < k_thresh)
            )
        if 'PSY_BUY' in active_signals:
            masks['PSY_BUY'] = (f['psy'] < 25) & (f['psy'] > f['psyma'])

        if not masks:
            return []

        if mode == 'OR':
            combined = pd.Series(False, index=f.index)
            for m in masks.values():
                combined = combined | m.fillna(False)
        else:
            combined = pd.Series(True, index=f.index)
            for m in masks.values():
                combined = combined & m.fillna(False)

        return combined[combined].index.tolist()

    def _build_signal_features(self, stock_codes: List[str]) -> pd.DataFrame:
        """从 _today_df / _prev_df 直接提取预计算列，组装特征矩阵。

        零 groupby，零 rolling，零 sort_values。
        所有技术指标已在缓存构建时预计算好。
        """
        if self._today_df.empty:
            return pd.DataFrame()

        # Filter to relevant stocks
        today = self._today_df[self._today_df['stock_code'].isin(stock_codes)]
        prev = self._prev_df[self._prev_df['stock_code'].isin(stock_codes)]

        if today.empty:
            return pd.DataFrame()

        today = today.set_index('stock_code')
        prev = prev.set_index('stock_code')

        # Only keep stocks present in both days (needed for cross detection)
        common = today.index.intersection(prev.index)
        if common.empty:
            return pd.DataFrame()

        t = today.loc[common]
        p = prev.loc[common]

        return pd.DataFrame({
            'kdj_k': t['kdj_k'], 'kdj_d': t['kdj_d'],
            'kdj_k_p': p['kdj_k'], 'kdj_d_p': p['kdj_d'],
            'macd_dif': t['macd_dif'], 'macd_dea': t['macd_dea'],
            'macd_dif_p': p['macd_dif'], 'macd_dea_p': p['macd_dea'],
            'ma_5': t['ma_5'], 'ma_20': t['ma_20'],
            'ma_5_p': p['ma_5'], 'ma_20_p': p['ma_20'],
            'vol_ma5': t['vol_ma5'], 'vol_ma20': t['vol_ma20'],
            'vol_ma5_p': p['vol_ma5'], 'vol_ma20_p': p['vol_ma20'],
            'close': t['close'], 'close_std_20': t['close_std_20'],
            'boll_mid': t['boll_mid'], 'high_20_max': t['high_20_max'],
            'psy': t['psy'], 'psyma': t['psyma'],
        }, index=common)

    def _execute_buy(
        self,
        date_str: str,
        valid_codes: set,
        candidates: List[str],
        max_positions: int,
        current_prices: Dict[str, float],
        total_asset: float
    ) -> Tuple[List[TradeRecord], List[Dict]]:
        """对候选股排序，买入 TOP X（两阶段资金分配：预计算 + 执行）"""
        buy_trades = []
        top_stocks_info = []
        x = max_positions - len(self.positions)
        if x <= 0 or not candidates:
            return buy_trades, top_stocks_info

        # 向量化提取候选股因子数据 — 从 _today_df 直接构建因子矩阵
        candidates_df = self._today_df[self._today_df['stock_code'].isin(candidates)]
        if candidates_df.empty:
            return buy_trades, top_stocks_info
        cdf = candidates_df.set_index('stock_code')
        factor_df = pd.DataFrame(index=cdf.index)

        for factor in self.ranker.factor_weights.keys():
            if factor == 'RET_20':
                factor_df[factor] = cdf['ret_20'].fillna(0.0).astype(float)
            elif factor == 'OVERHEAT':
                rsi = cdf['rsi_1']
                ret5 = cdf['ret_5'].fillna(0.0).astype(float)
                oh = pd.Series(0.0, index=cdf.index)
                rsi_t = self.config.overheat_rsi_threshold
                ret5_t = self.config.overheat_ret5_threshold
                mask = (rsi > rsi_t) & (ret5 > ret5_t)
                rsi_c = np.maximum(0.0, (rsi[mask] - rsi_t) / (100 - rsi_t))
                ret_c = np.minimum(1.0, np.maximum(0.0, (ret5[mask] - ret5_t) / 0.35))
                oh[mask] = (rsi_c + ret_c) / 2.0
                factor_df[factor] = oh
            elif factor == 'circulating_mv':
                mv = cdf['circulating_mv']
                factor_df[factor] = np.where(mv > 0, np.log(mv), np.nan)
            elif factor == 'WR_10':
                factor_df[factor] = cdf['wr_10'].fillna(0.0).astype(float)
            elif factor == 'WR_14':
                factor_df[factor] = cdf['wr_14'].fillna(0.0).astype(float)
            elif factor in cdf.columns:
                factor_df[factor] = cdf[factor]
            else:
                factor_df[factor] = np.nan

        factor_df = factor_df.fillna(0)
        ranked = self.ranker.rank(factor_df, top_n=x)

        existing_positions = {p.stock_code: p.shares for p in self.positions.values()}

        # 阶段一：预计算每个候选股的可用性，确定最终能买哪些
        capital_remaining = self.current_capital
        selected = []  # (stock_code, price, shares, cost, capital_needed)

        for stock_code in ranked:
            price = current_prices.get(stock_code, 0.0)
            if price <= 0:
                continue
            if not self.position_manager.can_buy(stock_code, price, existing_positions, current_prices):
                continue

            # 计算可买股数（取资金和仓位的交集）
            max_shares_by_capital = int(capital_remaining / price)
            max_shares_by_position = int(total_asset * self.position_manager.max_position_pct / price)
            shares = min(max_shares_by_capital, max_shares_by_position)
            if shares == 0:
                continue

            # 计算成本（过户费 + 券商佣金，与 execute_buy 保持一致）
            buy_value = shares * price
            transfer_fee = buy_value * self.trade_executor.TRANSFER_FEE
            brokerage = max(buy_value * self.trade_executor.BROKERAGE, self.trade_executor.MIN_BROKERAGE)
            cost = transfer_fee + brokerage
            capital_needed = shares * price + cost
            if capital_needed > capital_remaining:
                continue

            selected.append((stock_code, price, shares, cost, capital_needed))
            capital_remaining -= capital_needed
            existing_positions[stock_code] = shares

        # 阶段二：执行选中股票的买入（从阶段一结果中扣除现金）
        for stock_code, price, shares, cost, capital_needed in selected:
            capital_before_this = self.current_capital
            self.current_capital -= capital_needed

            trade = TradeRecord(
                date=date_str,
                stock_code=stock_code,
                action='BUY',
                price=price,
                shares=shares,
                cost=cost,
                capital_before=capital_before_this
            )
            buy_trades.append(trade)
            self.trade_history.append(trade)

            self.positions[stock_code] = Position(
                stock_code=stock_code,
                shares=shares,
                buy_price=price,
                buy_date=date_str,
                highest_price=price,
            )

            logger.info(
                f"[BUY] {date_str} {stock_code} @ {price:.3f} x {shares}股 "
                f"资金:{capital_needed:,.0f} (剩余现金:{self.current_capital:,.0f})"
            )

            top_stocks_info.append({
                'stock_code': stock_code,
                'price': price,
                'shares': shares,
                'capital_used': capital_needed,
            })

        return buy_trades, top_stocks_info

    def _get_trading_dates(self) -> List[pd.Timestamp]:
        """获取回测区间内的交易日列表（使用基准指数确保完整性）"""
        index_df = self.data_provider.get_index_data(
            self.config.benchmark_index,
            start_date=self.start_date,
            end_date=self.end_date
        )
        if index_df is None or index_df.empty:
            return []

        dates = sorted(index_df.index.unique())
        return [pd.Timestamp(d) for d in dates]

    def _init_prev_cache(self, first_date: pd.Timestamp):
        """加载 first_date 前一个交易日的数据作为 _prev_df。

        向后搜索最多 15 个日历日（覆盖春节等长假），找到第一个有 Parquet 文件的交易日。
        如果找不到（极罕见），_prev_df 保持为空，Day-1 信号检测自动跳过（影响可忽略）。
        """
        for offset in range(1, 16):
            candidate = first_date - pd.Timedelta(days=offset)
            date_str = candidate.strftime('%Y-%m-%d')
            df = self.data_provider.get_daily_dataframe(date_str)
            if df is not None and not df.empty:
                df = df.copy()
                df['trade_date'] = candidate
                self._prev_df = df.set_index('trade_date')
                return

    def _advance_to_date(self, date: pd.Timestamp):
        """滚动指针: 将前一日的 _today_df 变为 _prev_df，读入当日新数据。

        不再累积历史数据，不再 concat。每天只读一个 Parquet 文件。
        """
        if not self._today_df.empty:
            self._prev_df = self._today_df

        date_str = date.strftime('%Y-%m-%d')
        day_df = self.data_provider.get_daily_dataframe(date_str)
        if day_df is None or day_df.empty:
            self._today_df = pd.DataFrame()
            return

        day_df = day_df.copy()
        day_df['trade_date'] = pd.Timestamp(date_str)
        self._today_df = day_df.set_index('trade_date')

    def _get_stock_df(self, stock_code: str) -> pd.DataFrame:
        """按需构建单只股票的 1-2 行 DataFrame（前日+当日）。

        仅在需要时（持仓卖出检查、候选买入因子提取）调用，
        每次 ~0.1ms（boolean mask on 4760 rows），替代全市场 groupby (~250ms)。
        """
        self._n_stock_df_calls = getattr(self, '_n_stock_df_calls', 0) + 1
        rows = []
        if not self._prev_df.empty:
            p = self._prev_df[self._prev_df['stock_code'] == stock_code]
            if not p.empty:
                rows.append(p)
        if not self._today_df.empty:
            t = self._today_df[self._today_df['stock_code'] == stock_code]
            if not t.empty:
                rows.append(t)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows)

    def _get_current_prices(self, valid_codes: set) -> Dict[str, float]:
        """从 _today_df 直接提取有效股票的收盘价字典 — 零 groupby。

        使用 set_index 向量化（~5ms），替代逐股 iloc[-1] 迭代 (~250ms)。
        """
        today = self._today_df[self._today_df['stock_code'].isin(valid_codes)]
        if today.empty:
            return {}
        indexed = today.set_index('stock_code')
        return indexed['close'].to_dict()

    def _filter_stock_pool(self) -> set:
        """过滤股票池（ST、涨跌停、停牌）— 向量化在 _today_df 上。

        返回有效股票代码集合。零 groupby，零逐股迭代。
        """
        if self._today_df.empty:
            return set()

        today = self._today_df
        mask = pd.Series(True, index=today.index)

        if self.config.exclude_st and 'stock_name' in today.columns:
            mask &= ~today['stock_name'].astype(str).str.contains(
                r'ST|\*ST', regex=True, na=False
            )

        if self.config.exclude_limit_up:
            chg = today['change_pct'].fillna(0.0)
            mask &= chg < 9.9

        if self.config.exclude_limit_down:
            chg = today['change_pct'].fillna(0.0)
            mask &= chg > -9.9

        if self.config.exclude_suspended:
            mask &= today['volume'].fillna(0.0) > 0

        return set(today.loc[mask, 'stock_code'])
