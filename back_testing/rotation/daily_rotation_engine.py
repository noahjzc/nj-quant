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
    """
    计算过热度（0~1），用于惩罚短期过热股票，避免追高。

    设计原理：
    - 仅当 RSI 和 5日涨幅 同时超过阈值时才计算过热度
    - RSI 分量：线性映射 RSI 从阈值到 100 → 0~1
    - 涨幅分量：线性映射 5日涨幅从阈值到 0.50 → 0~1（上限 1.0）
    - 最终过热度 = 两个分量的均值，用作买入排序的减分项

    返回:
        0.0 ~ 1.0，0 表示不过热
    """
    # 双条件均不满足 → 不过热，直接返回 0
    if not (rsi > rsi_threshold and ret5 > ret5_threshold):
        return 0.0

    # RSI 分量：RSI 越接近 100，过热越严重
    rsi_component = max(0.0, (rsi - rsi_threshold) / (100 - rsi_threshold))
    # 涨幅分量：5日涨幅越接近 50%，过热越严重（上限 1.0 防止极端值）
    ret_component = min(1.0, max(0.0, (ret5 - ret5_threshold) / 0.35))
    # 取均值作为综合过热度
    return (rsi_component + ret_component) / 2.0
from back_testing.rotation.signal_engine.signal_filter import SignalFilter
from back_testing.rotation.signal_engine.signal_ranker import SignalRanker
from back_testing.factors.factor_utils import FactorProcessor
from back_testing.rotation.market_regime import MarketRegime
from back_testing.rotation.position_manager import RotationPositionManager
from back_testing.rotation.trade_executor import TradeExecutor, TradeRecord
from back_testing.risk.stop_loss_strategies import StopLossStrategies


@dataclass
class Position:
    """
    持仓信息，记录单只股票的持有状态。

    关键字段：
    - highest_price: 持仓期间达到的最高价，用于移动止损（trailing stop）。
      每次检查卖出时会更新，确保止损线只上移不下移。
    """
    stock_code: str
    shares: int               # 持有股数
    buy_price: float          # 买入均价
    buy_date: str             # 买入日期 (YYYY-MM-DD)
    highest_price: float = 0.0  # 持仓期间最高价（用于移动止损）


@dataclass
class DailyResult:
    """
    单个交易日的回测结果快照，用于后续绩效分析和可视化。

    portfolio_value 与 total_asset 等价，保留双字段是为了兼容不同的分析模块。
    """
    date: str
    total_asset: float        # 当日总资产（现金 + 持仓市值）
    cash: float               # 可用现金
    positions: Dict[str, Position]  # 当日收盘时的持仓
    trades: List[TradeRecord]       # 当日所有成交（买入+卖出）
    market_regime: str              # 大盘状态：strong / neutral / weak
    portfolio_value: float          # = total_asset，兼容字段


class DailyRotationEngine:
    """
    每日全市场轮动回测引擎

    核心设计：
    - 以交易日为单位，逐日推进，模拟"每日选股、轮动换仓"策略
    - 使用 Master DataFrame（_cache_df）缓存全市场历史数据，避免重复 I/O
    - 向量化信号检测：一次性为所有候选股构建特征矩阵，批量判断买卖信号
    - 两阶段资金分配：先预计算可用性，再执行买入，确保不超资金/仓位上限

    每日流程（_run_single_day）：
    1. 获取大盘状态，动态调整仓位参数
    2. 获取当日全市场日线数据
    3. 过滤股票池（ST、涨跌停、停牌）
    4. 检查持仓 → 卖出信号 + ATR止损止盈 + 移动止损 → 卖出
    5. 扫描全市场 → 买入信号 → 候选股
    6. 多因子排序 → 买入 TOP X（两阶段资金分配）
    7. 记录每日净值和交易
    """

    # 预加载窗口：回测开始前需要 30 个日历日的历史数据，
    # 确保首日就能计算出 20 日均线等趋势指标
    PRELOAD_DAYS = 30
    # 最小交易天数：一只股票必须有至少 20 个交易日的历史才参与筛选，
    # 过滤掉刚上市的新股和长期停牌后复牌的股票
    MIN_TRADING_DAYS = 20

    def __init__(self, config: RotationConfig, start_date: str, end_date: str,
                 data_provider=None, preloaded_cache=None):
        """
        初始化引擎。

        参数:
            config: 策略配置（包含仓位、信号、因子权重等所有参数）
            start_date / end_date: 回测区间 (YYYY-MM-DD)
            data_provider: 数据源，默认 DataProvider（直连数据库）。
                          传入 CachedProvider（Parquet缓存）可加速 Optuna 优化。
            preloaded_cache: 可选的历史数据预加载。
                            - pd.DataFrame: 直接作为 Master 缓存使用
                            - Dict[str, DataFrame]: 按股票代码分组的历史数据，会自动拼接
        """
        self.config = config
        self.start_date = start_date
        self.end_date = end_date

        # ── 数据源 ──
        self.data_provider = data_provider or DataProvider()
        self._preloaded_cache = preloaded_cache
        # 检测是否有快速日级数据接口（CachedProvider 特有）
        self._has_fast_daily = hasattr(self.data_provider, 'get_daily_dataframe')

        # ── 子系统初始化 ──
        # 仓位管理：控制单只股票和总仓位的上限，随大盘状态动态调整
        self.position_manager = RotationPositionManager(
            total_capital=config.initial_capital,
            max_total_pct=config.max_total_pct,
            max_position_pct=config.max_position_pct
        )
        # 交易执行：计算可买股数、手续费（过户费+券商佣金）
        self.trade_executor = TradeExecutor()
        # 买入信号过滤器：检测股票是否满足买入条件（支持 AND/OR 模式）
        self.buy_filter = SignalFilter(config.buy_signal_types, mode=config.buy_signal_mode,
                                        kdj_low_threshold=config.kdj_low_threshold)
        # 卖出信号过滤器：检测持仓是否触发卖出条件
        self.sell_filter = SignalFilter(config.sell_signal_types)
        # 多因子排序器：对候选股按因子权重打分排名
        self.ranker = SignalRanker(config.rank_factor_weights, config.rank_factor_directions)
        # 大盘状态检测器：判断当前市场是强/中/弱，影响仓位上限
        self.market_regime = MarketRegime(config.market_regime, self.data_provider)

        # ── ATR 止损止盈参数 ──
        self.atr_period = config.atr_period              # ATR 计算周期（默认14）
        self.stop_loss_mult = config.stop_loss_mult       # 止损倍数：买入价 - N*ATR
        self.take_profit_mult = config.take_profit_mult   # 止盈倍数：买入价 + N*ATR
        self.trailing_pct = config.trailing_pct           # 移动止损回撤比例
        self.trailing_start = config.trailing_start       # 移动止损启动盈利比例

        # ── 缓存全市场股票列表（避免每日重复查询数据库）──
        self._all_codes = self.data_provider.get_all_stock_codes()

        # ── 运行时状态 ──
        self.current_capital = config.initial_capital  # 当前可用现金
        self.positions: Dict[str, Position] = {}        # 当前持仓 stock_code → Position
        self.daily_results: List[DailyResult] = []       # 每日结果记录
        self.trade_history: List[TradeRecord] = []       # 完整交易历史
        # Master DataFrame: 全市场历史数据缓存，index=trade_date，含 stock_code 列
        # 每日只追加当日数据（1次 concat），避免重复查询
        self._cache_df: pd.DataFrame = pd.DataFrame()

    def run(self) -> List[DailyResult]:
        """
        运行回测主循环。

        流程:
        1. 获取交易日列表（基于基准指数）
        2. 一次性加载历史数据到 Master DataFrame（避免每日 N+1 查询）
        3. 逐日推进：追加当日数据 → 运行单日策略 → 记录结果
        4. 返回完整的每日结果列表

        返回:
            List[DailyResult]，每个元素包含当日资产、持仓、交易等信息
        """
        dates = self._get_trading_dates()
        n_dates = len(dates)
        if n_dates < 2:
            return []

        # ── 初始化 Master 缓存 ──
        # 两种来源：外部传入的预加载缓存（Optuna优化场景），或自行查询数据库（单次回测场景）
        if self._preloaded_cache is not None:
            if isinstance(self._preloaded_cache, pd.DataFrame):
                # 直接使用预拼接好的 Master DataFrame
                self._cache_df = self._preloaded_cache
            else:
                # dict 格式 {stock_code: DataFrame} → 拼接为 Master DataFrame
                frames = [df for df in self._preloaded_cache.values() if not df.empty]
                self._cache_df = pd.concat(frames) if frames else pd.DataFrame()
            self._preloaded_cache = None  # 释放引用，节省内存
        else:
            self._preload_histories(dates[0])

        now = datetime.now
        n_codes = self._cache_df['stock_code'].nunique() if not self._cache_df.empty else 0
        print(f"{now():%H:%M:%S} [DailyRotation] {self.start_date} ~ {self.end_date}, {n_dates}天, {n_codes}只")

        # ── 逐日推进 ──
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            # 首日和每10天输出一次进度
            if i == 0 or (i + 1) % 10 == 0:
                prev_asset = self.daily_results[-1].total_asset if self.daily_results else self.config.initial_capital
                print(f"{now():%H:%M:%S}   [{i+1}/{n_dates}] {date_str} | 持仓:{len(self.positions)} | 资产:{prev_asset:,.0f}")

            # 追加当日全市场数据到 Master DataFrame（1 次 concat 操作）
            self._advance_to_date(date)
            # 运行当日策略：检测信号 → 交易执行 → 记录结果
            result = self._run_single_day(date)
            self.daily_results.append(result)

        final_asset = self.daily_results[-1].total_asset if self.daily_results else self.current_capital
        print(f"{datetime.now():%H:%M:%S} [DailyRotation] 回测完成，最终资产: {final_asset:,.0f}")
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
        """
        输出每日汇总日志。

        日志分两级：
        - [DAY]: 当日市场状态、候选股数量、买卖信号数、资产概况
        - [TOP]: 最终买入的股票排名（按因子得分从高到低）
        """
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
        """
        执行单个交易日的完整流程。

        流程拆解（按顺序）:
        Step 0: 检测大盘状态 → 动态调整仓位上限
        Step 1: 获取当日全市场日线数据（从 Master 缓存中提取）
        Step 2: 过滤股票池（ST / 涨跌停 / 停牌）
        Step 3: 更新持仓期间最高价（用于移动止损判断）
        Step 4: 检查卖出信号 + ATR止损止盈 + 移动止损 → 执行卖出
        Step 5: 扫描全市场买入信号 → 候选股列表
        Step 6: 多因子排序 → 两阶段资金分配 → 买入 TOP X
        Step 7: 汇总当日结果

        关键设计：
        - 卖在买先：先释放现金，再分配买入，确保资金可用
        - 同日不来回：当日卖出的股票不会在同日被买回
        - total_asset 在卖出后和买入后各重算一次，确保资金和持仓市值准确
        """
        date_str = date.strftime('%Y-%m-%d')

        # ═══════════════════════════════════════════
        # Step 0: 大盘状态检测，动态调整仓位参数
        # ═══════════════════════════════════════════
        # 强势市场 → 高仓位；弱势市场 → 低仓位
        regime_name, regime_params = self.market_regime.get_regime(date)
        self.position_manager.max_total_pct = regime_params.max_total_pct
        self.position_manager.max_position_pct = regime_params.max_position_pct
        max_positions = regime_params.max_positions

        # ═══════════════════════════════════════════
        # Step 1: 获取当日全市场日线数据
        # ═══════════════════════════════════════════
        # 从 Master 缓存中提取：每只股票取其历史数据（至少 MIN_TRADING_DAYS 天）
        stock_data = self._get_daily_stock_data(date)
        if not stock_data:
            return DailyResult(date_str, self.current_capital, self.current_capital,
                               self.positions, [], regime_name, self.current_capital)

        # ═══════════════════════════════════════════
        # Step 2: 过滤股票池
        # ═══════════════════════════════════════════
        # 排除 ST、涨跌停、停牌股票，确保只交易正常可交易的标的
        filtered_data = self._filter_stock_pool(stock_data)

        # 提取当日收盘价快照：{stock_code: close_price}
        current_prices = {code: df['close'].iloc[-1] for code, df in filtered_data.items() if not df.empty}

        # ═══════════════════════════════════════════
        # Step 3: 更新持仓期间最高价
        # ═══════════════════════════════════════════
        # 最高价只上移不下移，用于计算移动止损触发线
        for stock_code, position in self.positions.items():
            current_price = current_prices.get(stock_code, 0.0)
            if current_price > position.highest_price:
                position.highest_price = current_price

        # ═══════════════════════════════════════════
        # Step 4: 检查持仓卖出信号 → 执行卖出
        # ═══════════════════════════════════════════
        # 三种卖出触发条件（任一满足即卖出）：
        #   1. 技术卖出信号（KDJ死叉、MACD死叉等）
        #   2. ATR 止损/止盈（固定比例止损止盈）
        #   3. 移动止损（从最高点回撤超过阈值）
        sell_trades = self._check_and_sell(date_str, filtered_data, current_prices)

        # 卖出后更新现金（现金增加 = 卖出金额 - 手续费）
        for trade in sell_trades:
            self.current_capital += trade.shares * trade.price - trade.cost

        # ═══════════════════════════════════════════
        # Step 5: 扫描全市场买入信号
        # ═══════════════════════════════════════════
        # 排除已持仓的股票 + 当日已卖出的股票（防止同日来回交易）
        sold_today = [t.stock_code for t in sell_trades]
        buy_candidates = self._scan_buy_candidates(filtered_data, exclude_codes=sold_today)

        # ═══════════════════════════════════════════
        # Step 6: 重新计算总资产（反映卖出后的现金更新）
        # ═══════════════════════════════════════════
        total_asset = self.current_capital + self.position_manager.get_position_value(
            {p.stock_code: p.shares for p in self.positions.values()},
            current_prices
        )
        self.position_manager.update_capital(total_asset)

        # ═══════════════════════════════════════════
        # Step 7: 多因子排序 + 两阶段资金分配 → 买入
        # ═══════════════════════════════════════════
        buy_trades, top_stocks_info = self._execute_buy(
            date_str, filtered_data, buy_candidates, max_positions, current_prices, total_asset
        )

        all_trades = sell_trades + buy_trades

        # ═══════════════════════════════════════════
        # Step 8: 最终重算总资产（反映买入后的手续费扣减）
        # ═══════════════════════════════════════════
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
        """
        检查所有持仓是否需要卖出。

        两阶段设计（收集 → 执行）保证先判断后操作，避免在遍历期间修改 positions 导致不一致。

        触发卖出的三种条件（任一满足）:
        1. 技术卖出信号：sell_filter 检测到死叉等形态
        2. ATR 止损/止盈：价格跌破买入价-N*ATR 或突破买入价+N*ATR
        3. 移动止损：价格从持仓期间最高点回撤超过 trailing_pct

        特殊处理：
        - 停牌股：股票不在当日数据中时，从 Master 缓存取最后收盘价强行卖出
        """
        sell_trades = []
        positions_to_close = []  # 阶段一：收集待卖出股票

        for stock_code, position in list(self.positions.items()):
            # ── 停牌股处理 ──
            # 股票不在当日数据中 = 停牌或退市，从 Master 缓存中取最后一日收盘价强行平仓
            if stock_code not in stock_data:
                if stock_code in self._cache_df['stock_code'].values:
                    df_cached = self._cache_df[self._cache_df['stock_code'] == stock_code].sort_index()
                    if not df_cached.empty:
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

            # ── 条件1: 技术卖出信号 ──
            if self.sell_filter.filter_sell(df, stock_code):
                if stock_code not in positions_to_close:
                    positions_to_close.append(stock_code)

            # ── 条件2 & 3: ATR 止损/止盈 + 移动止损 ──
            current_price = current_prices.get(stock_code, 0.0)
            if current_price > 0:
                try:
                    atr = StopLossStrategies.calculate_atr(df, period=self.atr_period)
                except Exception:
                    atr = 0.0  # 计算失败视为波动率未知，跳过 ATR 检查
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
                    # 触发 ATR 止损或移动止损
                    if exit_result['action'] in ('stop_loss', 'trailing_stop'):
                        if stock_code not in positions_to_close:
                            positions_to_close.append(stock_code)
                        logger.info(
                            f"[EXIT] {date_str} {stock_code} @ {current_price:.3f} "
                            f"原因:{exit_result['reason']}"
                        )

        # ── 阶段二：执行卖出 ──
        # 统一处理所有待卖出股票，计算持有收益并记录交易
        for stock_code in positions_to_close:
            position = self.positions[stock_code]
            price = current_prices.get(stock_code, 0.0)
            if price <= 0:
                continue

            shares, cost = self.trade_executor.execute_sell(stock_code, price, position.shares)
            if shares > 0:
                # 计算持有期收益（PnL = 价差收益 - 手续费）
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
                del self.positions[stock_code]  # 从持仓中移除

                logger.info(
                    f"[SELL] {date_str} {stock_code} @ {price:.3f} x {shares}股 "
                    f"买价:{buy_price:.3f} 持有:{holding_days}天 收益:{return_pct:+.2f}% "
                    f"PnL:{pnl:+,.0f} (卖前现金:{capital_before_sell:,.0f})"
                )

        return sell_trades

    def _scan_buy_candidates(self, stock_data: Dict[str, pd.DataFrame], exclude_codes: List[str] = None) -> List[str]:
        """
        扫描全市场，返回有买入信号的股票代码列表。

        采用向量化检测：一次性为所有候选股构建特征矩阵，批量判断信号条件，
        比逐股循环检测快一个数量级。

        信号模式:
        - AND: 所有活跃信号类型必须同时满足
        - OR:  任一活跃信号类型满足即可

        回退机制: 特征构建失败时回退到逐股检测（兼容数据不完整的情况）
        """
        exclude_set = set(exclude_codes) if exclude_codes else set()
        # 排除已持仓和当日已卖出的股票
        codes = [c for c in stock_data if c not in self.positions and c not in exclude_set]
        if not codes:
            return []

        # 构建特征矩阵（每行一只股票，列为信号检测所需指标）
        try:
            features = self._build_signal_features(codes)
        except Exception:
            # 回退到逐股检测：遍历每只股票，用 SignalFilter 逐一判断
            candidates = []
            for stock_code in codes:
                df = stock_data[stock_code]
                if df.empty or len(df) < 2:
                    continue
                if self.buy_filter.filter_buy(df, stock_code):
                    candidates.append(stock_code)
            return candidates

        if features.empty:
            return []

        # ═══════════════════════════════════════════
        # 向量化信号条件检测
        # ═══════════════════════════════════════════
        # 金叉信号通用模式：(今日快线 > 慢线) AND (昨日快线 <= 慢线)
        # 即：快慢线关系从"死叉"变为"金叉"的那一天触发信号
        active_signals = set(self.config.buy_signal_types)
        mode = self.config.buy_signal_mode
        masks = {}
        f = features

        # KDJ 金叉: K线上穿D线
        if 'KDJ_GOLD' in active_signals:
            masks['KDJ_GOLD'] = (f['kdj_k'] > f['kdj_d']) & (f['kdj_k_p'] <= f['kdj_d_p'])
        # MACD 金叉: DIF线上穿DEA线
        if 'MACD_GOLD' in active_signals:
            masks['MACD_GOLD'] = (f['macd_dif'] > f['macd_dea']) & (f['macd_dif_p'] <= f['macd_dea_p'])
        # 均线金叉: 5日均线上穿20日均线
        if 'MA_GOLD' in active_signals:
            masks['MA_GOLD'] = (f['ma_5'] > f['ma_20']) & (f['ma_5_p'] <= f['ma_20_p'])
        # 量能金叉: 5日均量上穿20日均量
        if 'VOL_GOLD' in active_signals:
            masks['VOL_GOLD'] = (f['vol_ma5'] > f['vol_ma20']) & (f['vol_ma5_p'] <= f['vol_ma20_p'])
        # 布林带突破: 收盘价突破上轨（中轨 + 2倍标准差）
        if 'BOLL_BREAK' in active_signals:
            boll_upper = f['boll_mid'] + 2 * f['close_std_20']
            masks['BOLL_BREAK'] = f['close'] > boll_upper
        # 20日新高突破: 收盘价 >= 前20日最高价
        if 'HIGH_BREAK' in active_signals:
            masks['HIGH_BREAK'] = f['close'] >= f['high_20_max']
        # KDJ 低位金叉: 金叉 + K值低于阈值（避免追高）
        if 'KDJ_GOLD_LOW' in active_signals:
            k_thresh = self.config.kdj_low_threshold
            masks['KDJ_GOLD_LOW'] = (
                (f['kdj_k'] > f['kdj_d']) & (f['kdj_k_p'] <= f['kdj_d_p']) & (f['kdj_k'] < k_thresh)
            )
        # PSY 买入: 心理线 < 25（超卖）且上穿 PSY 均线
        if 'PSY_BUY' in active_signals:
            masks['PSY_BUY'] = (f['psy'] < 25) & (f['psy'] > f['psyma'])

        if not masks:
            return []

        # 根据模式组合信号掩码
        if mode == 'OR':
            # OR 模式：任一信号满足即可
            combined = pd.Series(False, index=f.index)
            for m in masks.values():
                combined = combined | m.fillna(False)
        else:
            # AND 模式：所有信号必须同时满足
            combined = pd.Series(True, index=f.index)
            for m in masks.values():
                combined = combined & m.fillna(False)

        # 返回满足组合条件的股票代码列表
        return combined[combined].index.tolist()

    def _build_signal_features(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        从 Master 缓存构建信号特征矩阵（向量化核心）。

        返回 DataFrame，每行一只股票，列为信号检测所需的最新值(t)和前一日值(t-1)。
        之所以需要 t-1 值，是为了检测"金叉"——需要比较今日和昨日的快慢线关系变化。

        性能优化:
        - 每只股票只取最近 21 行（20日均线窗口 + 1行prev），而非全量历史
        - 所有滚动计算在压缩后的数据上一次性完成，避免重复 groupby
        - 索引对齐：groupby.last() 返回 stock_code 索引，groupby.nth(-2) 需手动对齐
        """
        mask = self._cache_df['stock_code'].isin(stock_codes)
        hist = self._cache_df[mask]

        if hist.empty:
            return pd.DataFrame()

        # 按 (stock_code, trade_date) 排序，保证后续 last/nth 取到正确顺序
        hist = hist.sort_values(['stock_code', self._cache_df.index.name])

        # 每只股票只保留最近 21 行（20 日窗口 + 1 行 prev），大幅压缩 transform 数据量
        hist = hist.groupby('stock_code', sort=False).tail(21)

        # ── 在压缩后的副本上一次性计算所有滚动指标 ──
        hist = hist.copy()
        g = hist.groupby('stock_code', sort=False)
        # 5日/20日均量
        hist['vol_ma5'] = g['volume'].rolling(5, min_periods=1).mean().values
        hist['vol_ma20'] = g['volume'].rolling(20, min_periods=5).mean().values
        # 20日收盘价标准差（布林带宽度）
        hist['close_std_20'] = g['close'].rolling(20, min_periods=5).std().values
        # 前20日最高价（不含当日，用 shift(1) 排除当日）
        hist['high_20_max'] = g['high'].shift(1).rolling(20, min_periods=1).max().values

        # ── 提取每只股票的最新行(t)和上一行(t-1) ──
        g2 = hist.groupby('stock_code', sort=False)
        latest = g2.last().copy()           # index = stock_code（每组最后一行）
        prev = g2.nth(-2).copy()            # index = trade_date（倒数第二行），需对齐
        # 对齐 prev 索引到 stock_code，使 latest 和 prev 可以按位置合并
        prev.index = prev['stock_code']

        # 填充缺失值（新股历史不足导致 NaN → 0）
        cols = ['vol_ma5', 'vol_ma20', 'close_std_20', 'high_20_max']
        for c in cols:
            if c in latest.columns:
                latest[c] = latest[c].fillna(0)
            if c in prev.columns:
                prev[c] = prev[c].fillna(0)

        # ── 组装特征矩阵 ──
        # _p 后缀 = 前一日值，用于检测金叉（今日满足 + 昨日不满足 = 刚触发）
        return pd.DataFrame({
            'kdj_k': latest['kdj_k'], 'kdj_d': latest['kdj_d'],
            'kdj_k_p': prev['kdj_k'], 'kdj_d_p': prev['kdj_d'],
            'macd_dif': latest['macd_dif'], 'macd_dea': latest['macd_dea'],
            'macd_dif_p': prev['macd_dif'], 'macd_dea_p': prev['macd_dea'],
            'ma_5': latest['ma_5'], 'ma_20': latest['ma_20'],
            'ma_5_p': prev['ma_5'], 'ma_20_p': prev['ma_20'],
            'vol_ma5': latest['vol_ma5'], 'vol_ma20': latest['vol_ma20'],
            'vol_ma5_p': prev['vol_ma5'], 'vol_ma20_p': prev['vol_ma20'],
            'close': latest['close'], 'close_std_20': latest['close_std_20'],
            'boll_mid': latest['boll_mid'], 'high_20_max': latest['high_20_max'],
            'psy': latest['psy'], 'psyma': latest['psyma'],
        }, index=latest.index)

    def _execute_buy(
        self,
        date_str: str,
        stock_data: Dict[str, pd.DataFrame],
        candidates: List[str],
        max_positions: int,
        current_prices: Dict[str, float],
        total_asset: float
    ) -> Tuple[List[TradeRecord], List[Dict]]:
        """
        对候选股进行多因子打分排序，买入排名靠前的股票。

        两阶段资金分配:
        阶段一（预计算）：按排名顺序遍历，模拟资金和仓位消耗，
                       确定哪些股票可以买、买多少股。
                       这个阶段不修改 self.current_capital，保证原子性。
        阶段二（执行）：从阶段一结果中逐笔扣除现金，创建 Position 记录。

        两阶段设计的原因：如果阶段一中某只股票因资金不足被跳过，
        后续排名更低的股票仍有机会。同时保证状态修改的一致性。

        OVERHEAT 因子：用 compute_overheat 计算过热度，作为排序的减分项，
        防止买入短期涨幅过大、RSI 过高的股票。
        """
        buy_trades = []
        top_stocks_info = []
        # 剩余可买数量 = 上限 - 当前持仓数
        x = max_positions - len(self.positions)
        if x <= 0 or not candidates:
            return buy_trades, top_stocks_info

        # ═══════════════════════════════════════════
        # 步骤1: 提取候选股因子数据
        # ═══════════════════════════════════════════
        # 遍历每只候选股，从日线数据和因子配置中提取排序所需的因子值
        factor_data_dict = {}
        for stock_code in candidates:
            df = stock_data.get(stock_code)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]  # 当日最新数据行
            factor_row = {}

            # 5日收益率（OVERHEAT 计算的前置条件）
            if len(df) >= 5 and 'close' in df.columns:
                ret5 = row['close'] / df['close'].iloc[-5] - 1
            else:
                ret5 = 0.0

            # 逐因子提取值
            for factor in self.ranker.factor_weights.keys():
                if factor == 'RET_20':
                    # 20日收益率 = 当日收盘 / 20日前收盘 - 1
                    if len(df) >= 20 and 'close' in df.columns:
                        factor_row[factor] = row['close'] / df['close'].iloc[-20] - 1
                    else:
                        factor_row[factor] = np.nan
                elif factor == 'OVERHEAT':
                    # 过热度：RSI + 5日涨幅双高时返回正值，作为减分项
                    rsi_val = row.get('rsi_1', np.nan)
                    if pd.notna(rsi_val):
                        factor_row[factor] = compute_overheat(
                            float(rsi_val), ret5,
                            self.config.overheat_rsi_threshold,
                            self.config.overheat_ret5_threshold
                        )
                    else:
                        factor_row[factor] = 0.0
                elif factor == 'circulating_mv':
                    # 流通市值取对数，减少量级差异
                    val = row.get('circulating_mv', np.nan)
                    factor_row[factor] = np.log(val) if val > 0 else np.nan
                elif factor in ('WR_10', 'WR_14'):
                    # 威廉指标：需要从历史数据中动态计算
                    period = 10 if factor == 'WR_10' else 14
                    factor_row[factor] = FactorProcessor.williams_r(df, period)
                elif factor in row.index:
                    # 直接从数据行中取值（如 rsi_1, volume 等）
                    val = row[factor]
                    factor_row[factor] = val if val == val else np.nan  # NaN check
                else:
                    factor_row[factor] = np.nan
            factor_data_dict[stock_code] = factor_row

        # 转为 DataFrame，NaN 填 0，交给 SignalRanker 打分
        factor_df = pd.DataFrame(factor_data_dict).T
        factor_df = factor_df.fillna(0)
        # ranked: 按因子得分降序排列的股票代码列表
        ranked = self.ranker.rank(factor_df, top_n=x)

        # 当前持仓股数快照（用于仓位上限检查）
        existing_positions = {p.stock_code: p.shares for p in self.positions.values()}

        # ═══════════════════════════════════════════
        # 阶段一: 预计算可用性
        # ═══════════════════════════════════════════
        # 按排名顺序遍历，模拟资金消耗，但不实际扣减现金
        capital_remaining = self.current_capital  # 模拟可用现金
        selected = []  # (stock_code, price, shares, cost, capital_needed)

        for stock_code in ranked:
            price = current_prices.get(stock_code, 0.0)
            if price <= 0:
                continue
            # 检查仓位上限（单只股票不能超过 max_position_pct）
            if not self.position_manager.can_buy(stock_code, price, existing_positions, current_prices):
                continue

            # 可买股数 = min(资金允许, 仓位允许)
            max_shares_by_capital = int(capital_remaining / price)
            max_shares_by_position = int(total_asset * self.position_manager.max_position_pct / price)
            shares = min(max_shares_by_capital, max_shares_by_position)
            if shares == 0:
                continue

            # 手续费：过户费 + 券商佣金（与 TradeExecutor.execute_buy 保持一致）
            buy_value = shares * price
            transfer_fee = buy_value * self.trade_executor.TRANSFER_FEE
            brokerage = max(buy_value * self.trade_executor.BROKERAGE, self.trade_executor.MIN_BROKERAGE)
            cost = transfer_fee + brokerage
            capital_needed = shares * price + cost  # 买入所需总资金
            if capital_needed > capital_remaining:
                continue  # 资金不足，跳过此股，尝试下一只

            selected.append((stock_code, price, shares, cost, capital_needed))
            capital_remaining -= capital_needed  # 模拟扣除
            existing_positions[stock_code] = shares  # 更新虚拟持仓，影响后续仓位检查

        # ═══════════════════════════════════════════
        # 阶段二: 执行买入（实际扣减现金 + 创建持仓）
        # ═══════════════════════════════════════════
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

            # 创建持仓记录，初始化 highest_price 为买入价
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
        """
        获取回测区间内的交易日列表。

        使用基准指数（如 000001.SH）的交易日历，而非简单日历日期，
        确保回测只在真实交易日执行，避免节假日/周末的空数据集。
        """
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
        """
        预加载初始窗口的历史数据到 Master DataFrame。

        窗口范围：回测首日前 PRELOAD_DAYS(30) 个日历日至首日。
        这 30 天的历史确保首日就能计算出 20 日均线、ATR 等趋势指标。
        数据以 Master DataFrame 格式存储：index=trade_date，包含 stock_code 列。
        """
        if not self._all_codes:
            return

        start = (first_date - pd.Timedelta(days=self.PRELOAD_DAYS)).strftime('%Y-%m-%d')
        end = first_date.strftime('%Y-%m-%d')

        histories = self.data_provider.get_batch_histories(
            self._all_codes, end_date=end, start_date=start
        )

        # 过滤空 DataFrame（没有交易数据的股票），拼接为单一 Master DataFrame
        frames = [df for df in histories.values() if not df.empty]
        self._cache_df = pd.concat(frames) if frames else pd.DataFrame()

    def _advance_to_date(self, date: pd.Timestamp):
        """
        推进缓存到指定交易日。

        将当日全市场数据追加到 Master DataFrame（1 次 pd.concat），同时清理
        已停牌/退市的股票数据以控制内存增长。

        两种数据路径:
        - 快速路径（CachedProvider）: get_daily_dataframe 直接返回 DataFrame
        - 原始路径（DataProvider）: get_stocks_for_date 返回 dict，需手动补 stock_code

        清理逻辑: 当日不交易 + 不在持仓中 = 可安全从缓存中移除（已停牌/退市）
        """
        date_str = date.strftime('%Y-%m-%d')

        if self._has_fast_daily:
            # ── 快速路径: CachedProvider.get_daily_dataframe ──
            day_df = self.data_provider.get_daily_dataframe(date_str)
            if day_df is None or day_df.empty:
                return

            day_df = day_df.copy()
            day_df['trade_date'] = pd.Timestamp(date_str)
            day_df = day_df.set_index('trade_date')

            # 清理停牌/退市股票：不在当日交易 + 不在持仓中的股票从缓存移除
            trading_codes = set(day_df['stock_code'].unique())
            if not self._cache_df.empty:
                positions_set = {p.stock_code for p in self.positions.values()}
                cached_codes = set(self._cache_df['stock_code'].unique())
                # stale = 缓存中有但当日不交易且未持仓的股票
                stale = cached_codes - trading_codes - positions_set
                if stale:
                    self._cache_df = self._cache_df[~self._cache_df['stock_code'].isin(stale)]

            # 一次 concat 追加所有股票（替代逐股 concat，性能从 O(n*m) 降为 O(m)）
            if self._cache_df.empty:
                self._cache_df = day_df
            else:
                self._cache_df = pd.concat([self._cache_df, day_df], sort=False)
        else:
            # ── 原始路径: DataProvider.get_stocks_for_date ──
            # 返回 {stock_code: {field: value}} dict，不包含 stock_code 字段
            day_data = self.data_provider.get_stocks_for_date(self._all_codes, date_str)
            if not day_data:
                return

            # 将 dict 转为 DataFrame 行列表
            rows: list = []
            for stock_code, row_data in day_data.items():
                # 关键修复: get_stocks_for_date 返回的 row dict 不含 stock_code，
                # 必须手动添加，否则后续 stock_code-based 过滤会丢失该行
                row_data['trade_date'] = pd.Timestamp(date_str)
                row_data['stock_code'] = stock_code
                rows.append(row_data)

            if not rows:
                return

            day_df = pd.DataFrame(rows).set_index('trade_date')

            trading_codes = set(day_data.keys())
            if not self._cache_df.empty:
                positions_set = {p.stock_code for p in self.positions.values()}
                cached_codes = set(self._cache_df['stock_code'].unique())
                stale = cached_codes - trading_codes - positions_set
                if stale:
                    self._cache_df = self._cache_df[~self._cache_df['stock_code'].isin(stale)]

            if self._cache_df.empty:
                self._cache_df = day_df
            else:
                self._cache_df = pd.concat([self._cache_df, day_df], sort=False)

    def _get_daily_stock_data(self, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """
        从 Master DataFrame 中提取当日活跃股票的滚动数据。

        裁剪条件:
        1. 数据截止到当前日期（<= date），不包含未来数据
        2. 每只股票至少有 MIN_TRADING_DAYS(20) 个交易日的历史
        3. 当日必须在股票的交易日期内（过滤当日停牌的股票）

        返回 {stock_code: DataFrame}，每只股票包含从缓存首日到当日的历史切片。
        """
        if self._cache_df.empty:
            return {}

        # 裁剪到当前日期，避免未来数据泄露
        window = self._cache_df[self._cache_df.index <= date]
        if window.empty:
            return {}

        result = {}
        for code, group in window.groupby('stock_code', sort=False):
            # 双条件：历史足够长 + 当日有交易
            if len(group) >= self.MIN_TRADING_DAYS and date in group.index:
                result[code] = group
        return result

    def _filter_stock_pool(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        过滤股票池，排除不可交易或不应交易的标的。

        过滤条件（由 config 控制开关）:
        - exclude_st: 过滤 ST / *ST 股票（股票名称包含 "ST"）
        - exclude_limit_up: 过滤涨停股（change_pct >= 9.9%，无法买入）
        - exclude_limit_down: 过滤跌停股（change_pct <= -9.9%，无法卖出）
        - exclude_suspended: 过滤停牌股（成交量为 0）

        注意: 涨停阈值设为 9.9% 而非 10%，因为实际涨停价可能因四舍五入
        略低于 10%，用 9.9% 确保覆盖所有涨停情形。
        """
        filtered = {}
        for stock_code, df in stock_data.items():
            if df.empty:
                continue
            latest = df.iloc[-1]  # 当日最新数据行

            # ST 过滤：股票名称包含 "ST" 或 "*ST"
            if self.config.exclude_st:
                name = str(latest.get('stock_name', ''))
                if 'ST' in name or '*ST' in name:
                    continue

            # 涨停过滤：涨幅 >= 9.9% 视为涨停，无法买入
            if self.config.exclude_limit_up:
                change_pct = latest.get('change_pct', 0.0) or 0.0
                if change_pct >= 9.9:
                    continue

            # 跌停过滤：跌幅 <= -9.9% 视为跌停，持有的话也无法卖出
            if self.config.exclude_limit_down:
                change_pct = latest.get('change_pct', 0.0) or 0.0
                if change_pct <= -9.9:
                    continue

            # 停牌过滤：成交量为 0 的股票视为停牌
            if self.config.exclude_suspended:
                if latest.get('volume', 0) == 0:
                    continue

            filtered[stock_code] = df

        return filtered
