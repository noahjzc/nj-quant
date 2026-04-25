"""每日全市场轮动回测核心引擎"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from back_testing.data.data_provider import DataProvider
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

        # 状态
        self.current_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}  # stock_code -> Position
        self.daily_results: List[DailyResult] = []
        self.trade_history: List[TradeRecord] = []

    def run(self) -> List[DailyResult]:
        """运行回测"""
        dates = self._get_trading_dates()
        print(f"[DailyRotation] 回测区间: {self.start_date} ~ {self.end_date}, 共 {len(dates)} 个交易日")

        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(dates)}] {date_str} | 持仓:{len(self.positions)} | 资产:{self.current_capital:,.0f}")

            result = self._run_single_day(date)
            self.daily_results.append(result)

        print(f"[DailyRotation] 回测完成，最终资产: {self.current_capital:,.0f}")
        return self.daily_results

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
        buy_trades = self._execute_buy(date_str, filtered_data, buy_candidates, max_positions, current_prices)

        # 更新现金
        for trade in sell_trades:
            self.current_capital += trade.shares * trade.price - trade.cost
        for trade in buy_trades:
            self.current_capital -= trade.shares * trade.price + trade.cost

        all_trades = sell_trades + buy_trades

        return DailyResult(
            date=date_str,
            total_asset=total_asset,
            cash=self.current_capital,
            positions={p.stock_code: p for p in self.positions.values()},
            trades=all_trades,
            market_regime=regime_name
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
                trade = TradeRecord(
                    date=date_str,
                    stock_code=stock_code,
                    action='SELL',
                    price=price,
                    shares=shares,
                    cost=cost,
                    capital_before=self.current_capital
                )
                sell_trades.append(trade)
                self.trade_history.append(trade)
                del self.positions[stock_code]

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
    ) -> List[TradeRecord]:
        """对候选股排序，买入 TOP X"""
        buy_trades = []
        x = max_positions - len(self.positions)
        if x <= 0 or not candidates:
            return buy_trades

        # 提取候选股因子数据
        factor_data_dict = {}
        for stock_code in candidates:
            df = stock_data.get(stock_code)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            factor_row = {}
            for factor in self.ranker.factor_weights.keys():
                if factor in row.index:
                    factor_row[factor] = row[factor]
            if factor_row:
                factor_data_dict[stock_code] = factor_row

        factor_df = pd.DataFrame(factor_data_dict).T
        top_stocks = self.ranker.rank(factor_df, top_n=x)

        existing_positions = {p.stock_code: p.shares for p in self.positions.values()}

        for stock_code in top_stocks:
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
                capital_before=self.current_capital
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

        return buy_trades

    def _get_trading_dates(self) -> List[pd.Timestamp]:
        """获取回测区间内的交易日列表"""
        all_codes = self.data_provider.get_all_stock_codes()
        if not all_codes:
            return []

        df = self.data_provider.get_stock_data(
            all_codes[0],
            start_date=self.start_date,
            end_date=self.end_date
        )
        if df is None or df.empty:
            return []

        dates = sorted(df.index.unique())
        return [pd.Timestamp(d) for d in dates]

    def _get_daily_stock_data(self, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """获取当日全市场日线数据"""
        date_str = date.strftime('%Y-%m-%d')
        all_codes = self.data_provider.get_all_stock_codes()
        if not all_codes:
            return {}

        result = {}
        batch_data = self.data_provider.get_batch_latest(
            all_codes, date_str, lookback_days=30
        )

        for stock_code, row_data in batch_data.items():
            try:
                # 获取该股票历史数据（含历史指标）
                hist_df = self.data_provider.get_stock_data(
                    stock_code,
                    end_date=date_str
                )
                if hist_df is not None and len(hist_df) > 0:
                    result[stock_code] = hist_df
            except Exception:
                continue

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
