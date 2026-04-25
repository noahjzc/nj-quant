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
from back_testing.rotation.signal_engine.signal_filter import SignalFilter
from back_testing.rotation.signal_engine.signal_ranker import SignalRanker
from back_testing.rotation.market_regime import MarketRegime
from back_testing.rotation.position_manager import RotationPositionManager
from back_testing.rotation.trade_executor import TradeExecutor, TradeRecord


@dataclass
class Position:
    """持仓信息"""
    stock_code: str
    shares: int
    buy_price: float
    buy_date: str


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
        self.buy_filter = SignalFilter(config.buy_signal_types)
        self.sell_filter = SignalFilter(config.sell_signal_types)
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
                print(f"  [{i+1}/{len(dates)}] {date_str} | 持仓:{len(self.positions)} | 资产:{self.current_capital:,.0f}")

            # 推进到当日：加载当日数据到滚动缓存
            self._advance_to_date(date)
            result = self._run_single_day(date)
            self.daily_results.append(result)

        print(f"[DailyRotation] 回测完成，最终资产: {self.current_capital:,.0f}")
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
        # 持仓明细
        positions_detail = []
        for code, pos in self.positions.items():
            pvalue = pos.shares * pos.shares  # placeholder, will be computed below
            positions_detail.append(f"{code}:{pos.shares}股")

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
            return DailyResult(date_str, self.current_capital, self.current_capital, self.positions, [], regime_name)

        # 过滤股票池
        filtered_data = self._filter_stock_pool(stock_data)

        # 获取持仓快照（代码→当前价）
        current_prices = {code: df['close'].iloc[-1] for code, df in filtered_data.items() if not df.empty}
        total_asset = self.current_capital + self.position_manager.get_position_value(
            {p.stock_code: p.shares for p in self.positions.values()},
            current_prices
        )
        self.position_manager.update_capital(total_asset)

        # Step 1: 检查持仓卖出信号
        sell_trades = self._check_and_sell(date_str, filtered_data, current_prices)

        # Step 2: 扫描买入信号
        buy_candidates = self._scan_buy_candidates(filtered_data)

        # Step 3: 多因子排序，买入 TOP X
        buy_trades, top_stocks_info = self._execute_buy(date_str, filtered_data, buy_candidates, max_positions, current_prices)

        # 更新现金
        for trade in sell_trades:
            self.current_capital += trade.shares * trade.price - trade.cost
        for trade in buy_trades:
            self.current_capital -= trade.shares * trade.price + trade.cost

        all_trades = sell_trades + buy_trades

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

        for stock_code, position in self.positions.items():
            if stock_code not in stock_data:
                continue
            df = stock_data[stock_code]
            if df.empty or len(df) < 2:
                continue

            if self.sell_filter.filter_sell(df, stock_code):
                positions_to_close.append(stock_code)

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

    def _scan_buy_candidates(self, stock_data: Dict[str, pd.DataFrame]) -> List[str]:
        """扫描全市场，返回有买入信号的股票代码"""
        candidates = []
        for stock_code, df in stock_data.items():
            if stock_code in self.positions:
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
        current_prices: Dict[str, float]
    ) -> Tuple[List[TradeRecord], List[Dict]]:
        """对候选股排序，买入 TOP X"""
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
            for factor in self.ranker.factor_weights.keys():
                if factor == 'RET_20':
                    # 20日收益率 = 当日收盘 / 20日前收盘 - 1
                    if len(df) >= 20 and 'close' in df.columns:
                        ret = row['close'] / df['close'].iloc[-20] - 1
                        factor_row[factor] = ret
                elif factor in row.index:
                    factor_row[factor] = row[factor]
            if factor_row:
                factor_data_dict[stock_code] = factor_row

        factor_df = pd.DataFrame(factor_data_dict).T
        ranked = self.ranker.rank(factor_df, top_n=x)

        existing_positions = {p.stock_code: p.shares for p in self.positions.values()}
        capital_before_buy = self.current_capital

        for stock_code in ranked:
            price = current_prices.get(stock_code, 0.0)
            if price <= 0:
                continue
            if not self.position_manager.can_buy(stock_code, price, existing_positions, current_prices):
                continue

            shares, cost = self.trade_executor.execute_buy(stock_code, price, self.current_capital)
            if shares == 0:
                continue

            trade = TradeRecord(
                date=date_str,
                stock_code=stock_code,
                action='BUY',
                price=price,
                shares=shares,
                cost=cost,
                capital_before=capital_before_buy
            )
            buy_trades.append(trade)
            self.trade_history.append(trade)

            self.positions[stock_code] = Position(
                stock_code=stock_code,
                shares=shares,
                buy_price=price,
                buy_date=date_str
            )
            existing_positions[stock_code] = shares
            capital_used = shares * price + cost

            logger.info(
                f"[BUY] {date_str} {stock_code} @ {price:.3f} x {shares}股 "
                f"资金:{capital_used:,.0f} (剩余现金:{self.current_capital:,.0f})"
            )

            top_stocks_info.append({
                'stock_code': stock_code,
                'price': price,
                'shares': shares,
                'capital_used': capital_used,
            })

        return buy_trades, top_stocks_info

    def _get_trading_dates(self) -> List[pd.Timestamp]:
        """获取回测区间内的交易日列表"""
        if not self._all_codes:
            return []

        df = self.data_provider.get_stock_data(
            self._all_codes[0],
            start_date=self.start_date,
            end_date=self.end_date
        )
        if df is None or df.empty:
            return []

        dates = sorted(df.index.unique())
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
        trading_codes = set(day_data.keys())
        for code in list(self._stock_cache.keys()):
            if code not in trading_codes:
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
                combined = combined[~combined.index.isin(cache.index)]
                if not combined.empty:
                    cache = pd.concat([cache, combined], sort=False)
                self._stock_cache[stock_code] = cache
            else:
                self._stock_cache[stock_code] = pd.concat(new_rows, sort=False)

    def _get_daily_stock_data(self, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """获取当日全市场日线数据（从滚动缓存返回，date 当日已由 _advance_to_date 预加载）"""
        result = {}
        for code, df in self._stock_cache.items():
            if df.empty:
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
