"""每日全市场轮动回测核心引擎"""
import pandas as pd
import numpy as np
import logging
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

    PRELOAD_DAYS = 30
    MIN_TRADING_DAYS = 20

    def __init__(self, config: RotationConfig, start_date: str, end_date: str):
        self.config = config
        self.start_date = start_date
        self.end_date = end_date

        self.data_provider = DataProvider()
        self.position_manager = RotationPositionManager(
            total_capital=config.initial_capital,
            max_total_pct=config.max_total_pct,
            max_position_pct=config.max_position_pct
        )
        self.trade_executor = TradeExecutor()
        self.buy_filter = SignalFilter(config.buy_signal_types, mode=config.buy_signal_mode)
        self.sell_filter = SignalFilter(config.sell_signal_types)
        # ATR 止损止盈参数
        self.atr_period = config.atr_period
        self.stop_loss_mult = config.stop_loss_mult
        self.take_profit_mult = config.take_profit_mult
        self.trailing_pct = config.trailing_pct
        self.trailing_start = config.trailing_start
        self.ranker = SignalRanker(config.rank_factor_weights, config.rank_factor_directions)
        self.market_regime = MarketRegime(config.market_regime, self.data_provider)

        # 缓存全市场股票列表（避免每日重复查询）
        self._all_codes = self.data_provider.get_all_stock_codes()

        # 状态
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}  # stock_code -> Position
        self.daily_results: List[DailyResult] = []
        self.trade_history: List[TradeRecord] = []
        self._stock_cache: Dict[str, pd.DataFrame] = {}  # code -> 滚动20日缓存

    def run(self) -> List[DailyResult]:
        """运行回测"""
        dates = self._get_trading_dates()
        print(f"[DailyRotation] 回测区间: {self.start_date} ~ {self.end_date}, 共 {len(dates)} 个交易日")

        # 一次性加载全市场历史数据（避免每日N+1查询）
        print("[DailyRotation] 预加载全量历史数据...")
        self._preload_histories(dates[0])
        print(f"[DailyRotation] 预加载完成，{len(self._stock_cache)} 只股票已缓存")

        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            if (i + 1) % 20 == 0:
                prev_asset = self.daily_results[-1].total_asset if self.daily_results else self.config.initial_capital
                print(f"  [{i+1}/{len(dates)}] {date_str} | 持仓:{len(self.positions)} | 资产:{prev_asset:,.0f}")

            # 推进到当日：加载当日数据到滚动缓存
            self._advance_to_date(date)
            result = self._run_single_day(date)
            self.daily_results.append(result)

        final_asset = self.daily_results[-1].total_asset if self.daily_results else self.current_capital
        print(f"[DailyRotation] 回测完成，最终资产: {final_asset:,.0f}")
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

    def _run_single_day(self, date: pd.Timestamp) -> DailyResult:
        """每日流程"""
        date_str = date.strftime('%Y-%m-%d')

        # Step 0: 获取大盘状态，动态调整参数
        regime_name, regime_params = self.market_regime.get_regime(date)
        self.position_manager.max_total_pct = regime_params.max_total_pct
        self.position_manager.max_position_pct = regime_params.max_position_pct
        max_positions = regime_params.max_positions

        # 获取当日全市场日线
        stock_data = self._get_daily_stock_data(date)
        if not stock_data:
            return DailyResult(date_str, self.current_capital, self.current_capital, self.positions, [], regime_name, self.current_capital)

        # 过滤股票池
        filtered_data = self._filter_stock_pool(stock_data)

        # 获取持仓快照（代码→当前价）
        current_prices = {code: df['close'].iloc[-1] for code, df in filtered_data.items() if not df.empty}

        # Step 1: 更新持仓期间最高价
        for stock_code, position in self.positions.items():
            current_price = current_prices.get(stock_code, 0.0)
            if current_price > position.highest_price:
                position.highest_price = current_price

        # Step 2: 检查持仓卖出信号
        sell_trades = self._check_and_sell(date_str, filtered_data, current_prices)

        # 更新现金（卖出）
        for trade in sell_trades:
            self.current_capital += trade.shares * trade.price - trade.cost

        # Step 2: 扫描买入信号（排除当日已卖出的股票，防止同日来回交易）
        sold_today = [t.stock_code for t in sell_trades]
        buy_candidates = self._scan_buy_candidates(filtered_data, exclude_codes=sold_today)

        # Step 3: 重新计算 total_asset（此时包含卖出后的现金更新）
        total_asset = self.current_capital + self.position_manager.get_position_value(
            {p.stock_code: p.shares for p in self.positions.values()},
            current_prices
        )
        self.position_manager.update_capital(total_asset)

        # Step 4: 多因子排序，买入 TOP X
        buy_trades, top_stocks_info = self._execute_buy(date_str, filtered_data, buy_candidates, max_positions, current_prices, total_asset)

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
        )

    def _check_and_sell(
        self,
        date_str: str,
        stock_data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float]
    ) -> List[TradeRecord]:
        """检查持仓是否有卖出信号"""
        sell_trades = []
        positions_to_close = []

        for stock_code, position in list(self.positions.items()):
            if stock_code not in stock_data:
                # 停牌股：无法获取当日价格，用缓存中最后一日收盘价平仓
                if stock_code in self._stock_cache and not self._stock_cache[stock_code].empty:
                    df_cached = self._stock_cache[stock_code]
                    price = df_cached['close'].iloc[-1]
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
            df = stock_data[stock_code]
            if df.empty or len(df) < 2:
                continue

            if self.sell_filter.filter_sell(df, stock_code):
                if stock_code not in positions_to_close:
                    positions_to_close.append(stock_code)

            # ATR 止损/止盈/移动止损检查
            current_price = current_prices.get(stock_code, 0.0)
            if current_price > 0:
                try:
                    atr = StopLossStrategies.calculate_atr(df, period=self.atr_period)
                except Exception:
                    atr = 0.0
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

    def _scan_buy_candidates(self, stock_data: Dict[str, pd.DataFrame], exclude_codes: List[str] = None) -> List[str]:
        """扫描全市场，返回有买入信号的股票代码"""
        exclude_set = set(exclude_codes) if exclude_codes else set()
        candidates = []
        for stock_code, df in stock_data.items():
            if stock_code in self.positions:
                continue
            if stock_code in exclude_set:
                continue
            if df.empty or len(df) < 2:
                continue
            if self.buy_filter.filter_buy(df, stock_code):
                candidates.append(stock_code)
        return candidates

    def _execute_buy(
        self,
        date_str: str,
        stock_data: Dict[str, pd.DataFrame],
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

        # 提取候选股因子数据
        factor_data_dict = {}
        for stock_code in candidates:
            df = stock_data.get(stock_code)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            factor_row = {}

            # 计算 RET_5（OVERHEAT 计算需要）
            if len(df) >= 5 and 'close' in df.columns:
                ret5 = row['close'] / df['close'].iloc[-5] - 1
            else:
                ret5 = 0.0

            for factor in self.ranker.factor_weights.keys():
                if factor == 'RET_20':
                    # 20日收益率 = 当日收盘 / 20日前收盘 - 1
                    if len(df) >= 20 and 'close' in df.columns:
                        factor_row[factor] = row['close'] / df['close'].iloc[-20] - 1
                    else:
                        factor_row[factor] = np.nan
                elif factor == 'OVERHEAT':
                    rsi_val = row.get('rsi_1', np.nan)
                    if pd.notna(rsi_val):
                        factor_row[factor] = compute_overheat(
                            float(rsi_val), ret5,
                            self.config.overheat_rsi_threshold,
                            self.config.overheat_ret5_threshold
                        )
                    else:
                        factor_row[factor] = 0.0
                elif factor in row.index:
                    val = row[factor]
                    factor_row[factor] = val if val == val else np.nan  # NaN check
                else:
                    factor_row[factor] = np.nan
            factor_data_dict[stock_code] = factor_row

        factor_df = pd.DataFrame(factor_data_dict).T
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

    def _preload_histories(self, first_date: pd.Timestamp):
        """预加载初始窗口：回测首日前30个日历日的数据"""
        if not self._all_codes:
            return

        start = (first_date - pd.Timedelta(days=self.PRELOAD_DAYS)).strftime('%Y-%m-%d')
        end = first_date.strftime('%Y-%m-%d')

        histories = self.data_provider.get_batch_histories(
            self._all_codes, end_date=end, start_date=start
        )

        self._stock_cache: Dict[str, pd.DataFrame] = {}
        for code, df in histories.items():
            if not df.empty:
                self._stock_cache[code] = df.copy()

    def _advance_to_date(self, date: pd.Timestamp):
        """推进到指定日期：加载当日数据，追加到滚动缓存"""
        date_str = date.strftime('%Y-%m-%d')

        day_data = self.data_provider.get_stocks_for_date(self._all_codes, date_str)
        if not day_data:
            return

        # 检测停牌：当日无交易的股票清空缓存（退市同理）
        # 但保留仍持仓的股票缓存，以便 _check_and_sell 能强制卖出停牌股
        trading_codes = set(day_data.keys())
        for code in list(self._stock_cache.keys()):
            if code not in trading_codes and code not in self.positions:
                del self._stock_cache[code]

        # 按股票代码分组，一次性 concat
        from collections import defaultdict
        rows_by_code: Dict[str, list] = defaultdict(list)
        for stock_code, row_data in day_data.items():
            new_row = pd.DataFrame([row_data]).set_index('trade_date')
            rows_by_code[stock_code].append(new_row)

        for stock_code, new_rows in rows_by_code.items():
            if stock_code in self._stock_cache:
                cache = self._stock_cache[stock_code]
                combined = pd.concat(new_rows, sort=False)
                if not combined.empty:
                    combined = combined[~combined.index.isin(cache.index)]
                    if not combined.empty:
                        combined = combined.dropna(axis=1, how='all')
                        if not combined.empty:
                            cache = pd.concat([cache, combined], sort=False)
                self._stock_cache[stock_code] = cache
            else:
                combined = pd.concat(new_rows, sort=False)
                self._stock_cache[stock_code] = combined.dropna(axis=1, how='all') if not combined.empty else combined

    def _get_daily_stock_data(self, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """获取当日全市场日线数据（仅返回缓存中≥20个交易日的成熟股）"""
        result = {}
        for code, df in self._stock_cache.items():
            if df.empty:
                continue
            if len(df) < self.MIN_TRADING_DAYS:
                continue
            if date in df.index:
                result[code] = df
        return result

    def _filter_stock_pool(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """过滤股票池（ST、涨跌停、停牌）"""
        filtered = {}
        for stock_code, df in stock_data.items():
            if df.empty:
                continue
            latest = df.iloc[-1]

            if self.config.exclude_st:
                name = str(latest.get('stock_name', ''))
                if 'ST' in name or '*ST' in name:
                    continue

            if self.config.exclude_limit_up:
                # 用 change_pct 近似判断涨停（A股涨跌停板 ≈ ±10%）
                change_pct = latest.get('change_pct', 0.0) or 0.0
                if change_pct >= 9.9:
                    continue

            if self.config.exclude_limit_down:
                change_pct = latest.get('change_pct', 0.0) or 0.0
                if change_pct <= -9.9:
                    continue

            if self.config.exclude_suspended:
                if latest.get('volume', 0) == 0:
                    continue

            filtered[stock_code] = df

        return filtered
