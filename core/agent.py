"""
TradeAgent Core Orchestrator.

The Agent brain that ties everything together. Runs the main trading loop,
coordinating data ingestion, analysis, risk management, and execution.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import settings, trading_params
from core.event_bus import Event, EventType, event_bus
from core.logger import get_logger
from core.models import AccountInfo, AssetClass, Order, OrderSide, OrderStatus, Signal, SignalType, Timeframe, OrderType, OrderExecutionAlgo

# Implemented components
from data.providers.demo_provider import DemoDataProvider
from data.providers.oanda_provider import OandaDataProvider
from data.storage.storage_manager import StorageManager
from execution.brokers.paper_executor import PaperExecutor
from fundamentals.economic_calendar import EconomicCalendar
from fundamentals.macro_analyzer import MacroAnalyzer
from fundamentals.session_analyzer import SessionAnalyzer
from risk.risk_manager import RiskManager
from risk.stop_loss import StopLossManager
from sentiment.news_fetcher import NewsFetcher
from sentiment.sentiment_analyzer import SentimentAnalyzer
from signals.ml_models.ensemble import SignalEnsemble
from signals.ta_engine import TAEngine
from signals.ta_signals import TASignalGenerator
from signals.limit_agent import LimitOrderAgent
from signals.context_agent import ContextAgent, MarketContext
from signals.llm_validator import LLMTradeValidator
from data.tradingview_provider import TradingViewProvider
from signals.tradingview_agent import TradingViewAgent
from notifications.telegram_bot import TelegramBot

logger = get_logger("agent")


class TradeAgent:
    def __init__(self):
        self.is_running = False
        
        # Determine Data & Execution Engine
        if settings.trade_mode == "paper":
            self.data_provider = DemoDataProvider()
            self.executor = PaperExecutor()
        else:
            if settings.default_market == "forex" and settings.has_oanda:
                self.data_provider = OandaDataProvider()
                from execution.brokers.oanda_executor import OandaExecutor
                self.executor = OandaExecutor()
            elif settings.has_mt5:
                from data.providers.mt5_provider import MT5DataProvider
                from execution.brokers.mt5_executor import MT5Executor
                self.data_provider = MT5DataProvider()
                self.executor = MT5Executor()
                logger.info("Using MetaTrader 5 Data Provider & Executor")
            else:
                logger.warning("Falling back to Demo Provider for Data")
                self.data_provider = DemoDataProvider()
                self.executor = PaperExecutor()
                
        # Sub-systems
        self.storage = StorageManager()
        self.ta_engine = TAEngine()
        self.ta_signals = TASignalGenerator()
        self.limit_agent = LimitOrderAgent()
        self.context_agent = ContextAgent()
        self.tv_provider = TradingViewProvider()
        self.tv_agent = TradingViewAgent(self.tv_provider)
        self.llm_validator = LLMTradeValidator()
        self.risk_manager = RiskManager()
        self.stop_loss_manager = StopLossManager()
        
        # Optional components
        self.econ_calendar = EconomicCalendar()
        self.macro_analyzer = MacroAnalyzer()
        self.session_analyzer = SessionAnalyzer()
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.telegram = TelegramBot()
        
        # Configuration
        self.watchlist = []
        for market, assets in trading_params.watchlist.items():
            if isinstance(assets, list):
                self.watchlist.extend(assets)
        self.active_tfs = [
            Timeframe.from_string(tf) 
            for tf in trading_params.general.get("timeframes", {}).get("active", ["H1"])
        ]
        self.scan_interval = trading_params.scan_interval

    async def _setup_telegram(self):
        if settings.has_telegram:
            self.telegram.on_pause_req = self._pause_trading
            self.telegram.on_resume_req = self._resume_trading
            self.telegram.on_kill_req = self._kill_switch
            self.telegram.get_account_info = self.executor.get_account_info
            self.telegram.get_positions = self.executor.get_open_positions
            await self.telegram.start()

    async def _pause_trading(self):
        self.risk_manager.is_paused = True
        event_bus.pause()
        
    async def _resume_trading(self):
        self.risk_manager.is_paused = False
        event_bus.resume()
        
    async def _kill_switch(self):
        self.risk_manager.is_paused = True
        logger.error("KILL SWITCH ACTIVATED! Attempting to close all positions.")
        positions = await self.executor.get_open_positions()
        for p in positions:
            # We don't have current price immediately, fetch from data provider if needed
            # For simplicity, pass 0.0, executors should use market orders
            await self.executor.close_position(p, 0.0)

    async def start(self):
        logger.info("Initializing TradeAgent...")
        
        await self.data_provider.connect()
        await self.executor.connect()
        self.storage.connect()
        
        self.econ_calendar.connect()
        self.macro_analyzer.connect()
        self.news_fetcher.connect()
        self.sentiment_analyzer.load_model()
        
        await self._setup_telegram()
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._tick_stream_loop())
        asyncio.create_task(self._macro_update_loop())
        asyncio.create_task(self._news_update_loop())
        asyncio.create_task(self._optimization_loop())
        
        # Main trading loop
        await self._main_trading_loop()

    async def stop(self):
        logger.info("Stopping TradeAgent...")
        self.is_running = False
        await self.data_provider.disconnect()
        await self.executor.disconnect()
        self.storage.close()
        if settings.has_telegram:
            await self.telegram.stop()

    async def _news_update_loop(self):
        """Continuously fetch and analyze fundamental news."""
        self.current_sentiment_score = 0.0
        while self.is_running:
            try:
                news = await self.news_fetcher.fetch_forex_news()
                if news:
                    headlines = [n["headline"] for n in news[:15]]
                    results = await self.sentiment_analyzer.analyze_batch(headlines)
                    self.current_sentiment_score = self.sentiment_analyzer.aggregate_score(results)
                    logger.debug(f"[SENTIMENT] Market Score Updated: {self.current_sentiment_score:+.2f}")
            except Exception as e:
                logger.error(f"News update loop error: {e}")
            
            # Fetch news every 15 minutes
            await asyncio.sleep(900)

    async def _macro_update_loop(self):
        """Update slow-moving fundamentals occasionally."""
        while self.is_running:
            await self.macro_analyzer.update_rates()
            # Fetch events for the next 7 days
            now = datetime.utcnow()
            from datetime import timedelta
            await self.econ_calendar.fetch_events(now, now + timedelta(days=7))
            
            # Sleep 12 hours
            await asyncio.sleep(12 * 3600)

    async def _optimization_loop(self):
        """Walk-Forward Optimization Engine. Runs backtests periodically to dynamically adapt parameters."""
        while self.is_running:
            # Run optimization every 7 days (604800 seconds)
            # We'll sleep first to avoid running it immediately on startup since we already have a brain
            await asyncio.sleep(7 * 24 * 3600)
            
            logger.info("[WALK-FORWARD] Starting background strategy optimization...")
            try:
                import subprocess
                # Run the backtest script which updates the JSON files
                process = await asyncio.create_subprocess_exec(
                    "python", "run_backtest.py",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"[WALK-FORWARD] Optimization complete! Brain updated.")
                else:
                    logger.error(f"[WALK-FORWARD] Optimization failed. Sticking with current brain. Error: {stderr.decode()}")
            except Exception as e:
                logger.error(f"[WALK-FORWARD] Optimization loop error: {e}")

    async def _tick_stream_loop(self):
        """Consume live ticks to feed the PaperExecutor and trailing stops."""
        try:
            async for tick in self.data_provider.stream_ticks(self.watchlist):
                if not self.is_running: break
                
                # If paper trading, update simulator prices
                if settings.trade_mode == "paper":
                    await self.executor.update_simulated_price(tick)
                    
                # In live mode, we might want to check trailing stops here
                # (OANDA/MT5 can handle Trailing SL natively, but if manual, check here)
        except Exception as e:
            logger.error(f"Tick stream failed: {e}")

    async def _main_trading_loop(self):
        """Core analysis cycle running every scan_interval."""
        while self.is_running:
            start_time = datetime.utcnow()
            logger.info("Starting analysis cycle.")
            
            # Reload dynamic parameters if the optimizer updated them
            self.ta_signals.reload_brain()
            
            try:
                account = await self.executor.get_account_info()
                open_positions = await self.executor.get_open_positions()
                
                # ── 0. ACTIVE POSITION MONITOR (50% SCALE-OUT + 1R BREAKEVEN) ───
                for pos in open_positions:
                    if pos.metadata.get("is_pending", False):
                        continue
                    try:
                        # Fetch recent M5 data for current ATR / context
                        df_pos = await self.data_provider.get_historical_dataframe(pos.symbol, Timeframe.M5, limit=100)
                        if df_pos is not None and len(df_pos) >= 30:
                            df_pos_ta = self.ta_engine.add_all_indicators(df_pos)
                            if df_pos_ta is None or len(df_pos_ta) == 0:
                                continue
                            latest_pos_row = df_pos_ta.iloc[-1]
                            
                            # Check 1R Profit for Partial Scale-Out (50%)
                            risk_dist = abs(pos.entry_price - pos.stop_loss)
                            current_gain = (pos.current_price - pos.entry_price) if pos.side == OrderSide.BUY else (pos.entry_price - pos.current_price)
                            
                            # If gain >= 1.0R and position has >= 0.02 lots and not yet partially closed
                            if risk_dist > 0 and current_gain >= (risk_dist * 1.0):
                                if not pos.metadata.get("partial_closed", False) and pos.quantity >= 0.02:
                                    half_vol = round(pos.quantity / 2.0, 2)
                                    logger.info(f"[{pos.symbol}] 💰 1R PROFIT REACHED: Taking 50% Partial Profit ({half_vol} lots)!")
                                    closed_part = await self.executor.close_position(pos, pos.current_price, volume=half_vol)
                                    if closed_part:
                                        pos.metadata["partial_closed"] = True
                                        pos.quantity -= half_vol
                            
                            new_sl = self.stop_loss_manager.calculate_trailing_stop(
                                current_price=pos.current_price,
                                current_sl=pos.stop_loss,
                                side=pos.side,
                                row=latest_pos_row,
                                entry_price=pos.entry_price,
                                symbol=pos.symbol
                            )
                            
                            # If new_sl is tighter/better than current_sl, modify on broker
                            if new_sl != pos.stop_loss and new_sl > 0:
                                logger.info(f"[{pos.symbol}] 🎯 BREAKEVEN/TRAILING SL TRIGGERED: Moving SL from {pos.stop_loss:.5f} to {new_sl:.5f}")
                                modified = await self.executor.modify_position(pos, new_sl=new_sl, new_tp=pos.take_profit)
                                if modified:
                                    pos.stop_loss = new_sl
                    except Exception as ex:
                        logger.error(f"Error checking breakeven/partial close for {pos.symbol}: {ex}")

                # ── 0.5. DAILY ROLLOVER GUARD (21:50 - 22:15 UTC) ─────────────
                now_utc = datetime.utcnow()
                if now_utc.hour == 21 and now_utc.minute >= 50:
                    logger.debug("Daily Forex Rollover in progress (21:50-22:15 UTC). Pausing new entries.")
                    continue
                if now_utc.hour == 22 and now_utc.minute <= 15:
                    logger.debug("Daily Forex Rollover in progress (21:50-22:15 UTC). Pausing new entries.")
                    continue

                for symbol in self.watchlist:
                    # 1. Fundamental Kill-Switch Check
                    if not self.econ_calendar.is_safe_to_trade(symbol):
                        # Force close any existing positions for this symbol
                        for p in open_positions:
                            if p.symbol == symbol:
                                logger.warning(f"[KILL-SWITCH] High impact news approaching. Liquidating {symbol} position {p.id}.")
                                await self.executor.close_position(p, 0.0)
                        continue
                        
                    session = self.session_analyzer.get_current_session()
                    session_str = session.value if hasattr(session, 'value') else str(session)
                    if not self.session_analyzer.is_optimal_for_pair(symbol, session):
                        logger.debug(f"{symbol} skipped (sub-optimal session: {session_str})")
                        continue

                    # ── 1.5. Higher-Timeframe Macro Bias (H1 Context) ────────────
                    df_h1 = await self.data_provider.get_historical_dataframe(symbol, Timeframe.H1, limit=120)
                    macro_dir, macro_desc = (0, "MACRO_NEUTRAL")
                    h1_summary = {}
                    if df_h1 is not None and len(df_h1) >= 30:
                        df_h1_ta = self.ta_engine.add_all_indicators(df_h1)
                        if df_h1_ta is not None and len(df_h1_ta) > 0:
                            macro_dir, macro_desc = self.context_agent.evaluate_macro_bias(df_h1_ta)
                            h1_latest = df_h1_ta.iloc[-1]
                            h1_summary = {
                                "h1_close": float(h1_latest.get("close", 0)),
                                "h1_ema99": float(h1_latest.get("EMA_99", h1_latest.get("EMA_50", 0))),
                                "h1_rsi": float(h1_latest.get("RSI", 50)),
                                "h1_adx": float(h1_latest.get("ADX", 20))
                            }

                    # 2. Fetch Data (Multi-Timeframe Loop)
                    for tf in self.active_tfs:
                        candles = await self.data_provider.get_historical_candles(
                            symbol, tf, limit=300
                        )
                        if not candles or len(candles) < 50: continue
                        
                        df = await self.data_provider.get_historical_dataframe(
                            symbol, tf, limit=300
                        )
                    
                        # Determine Asset Class early
                        asset_class = AssetClass.FOREX
                        if symbol in trading_params.watchlist.get("crypto", []):
                            asset_class = AssetClass.CRYPTO
                        elif symbol in trading_params.watchlist.get("equity", []):
                            asset_class = AssetClass.EQUITY
                        elif "XAU" in symbol or "GOLD" in symbol:
                            asset_class = AssetClass.COMMODITY
                            
                        # Heartbeat / Stale Data Check
                        if df is None or len(df) < 50:
                            continue
                        last_tick_time = df.index[-1]
                        
                        # Different stale thresholds: 15 mins for Crypto/Forex, 72 hours for Equity (weekends)
                        stale_threshold = 3 * 24 * 3600 if asset_class == AssetClass.EQUITY else 900
                        
                        if isinstance(last_tick_time, datetime) and (datetime.utcnow() - last_tick_time).total_seconds() > stale_threshold:
                            logger.error(f"[{symbol}] STALE DATA DETECTED. Last tick was at {last_tick_time}. Freezing trades.")
                            continue
                    
                        # 3. Technical Analysis
                        df_ta = self.ta_engine.add_all_indicators(df)
                        if df_ta is None or len(df_ta) < 10:
                            continue
                        
                        market_context = self.context_agent.evaluate_context(symbol, df_ta)
                            
                        signals = []
                        ta_sig = self.ta_signals.generate_signal(
                            symbol, df_ta, tf, asset_class
                        )
                        if ta_sig:
                            signals.append(ta_sig)
                        else:
                            lim_sigs = self.limit_agent.generate_signal(
                                symbol, df_ta, tf, asset_class
                            )
                            if lim_sigs:
                                if isinstance(lim_sigs, list):
                                    signals.extend(lim_sigs)
                                else:
                                    signals.append(lim_sigs)
                            else:
                                tv_sig = await self.tv_agent.generate_signal(symbol, tf)
                                if tv_sig:
                                    signals.append(tv_sig)
                        
                        for signal in signals:
                            # ── MULTI-TIMEFRAME (MTF) CONFIRMATION GATE ──────────
                            # Never trade against H1 Macro Bias on M5/M15 timeframes
                            is_micro_tf = tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]
                            if is_micro_tf and macro_dir != 0:
                                if macro_dir == -1 and signal.signal_type in [SignalType.BUY, "buy"]:
                                    logger.warning(f"[{symbol}] MTF GATE VETO: Rejecting {tf.value} BUY against {macro_desc}")
                                    continue
                                elif macro_dir == 1 and signal.signal_type in [SignalType.SELL, "sell"]:
                                    logger.warning(f"[{symbol}] MTF GATE VETO: Rejecting {tf.value} SELL against {macro_desc}")
                                    continue

                            # Context Filtering (Soften veto to size reduction during strong trends)
                            if market_context == MarketContext.STRONG_BULLISH and signal.signal_type == SignalType.SELL:
                                logger.debug(f"{symbol} SELL is counter-trend (Market {market_context.name}). Reducing confidence/size by 50%.")
                                signal.confidence *= 0.5
                            elif market_context == MarketContext.STRONG_BEARISH and signal.signal_type == SignalType.BUY:
                                logger.debug(f"{symbol} BUY is counter-trend (Market {market_context.name}). Reducing confidence/size by 50%.")
                                signal.confidence *= 0.5
                                signal.win_rate *= 0.85  # Mild reduction, not full 30% penalty

                            # Fundamental Sentiment Veto — ONLY acts on strongly directional sentiment
                            if hasattr(self, 'current_sentiment_score') and self.current_sentiment_score != 0.0:
                                score = self.current_sentiment_score
                                if score > 0.4 and signal.signal_type == SignalType.SELL:
                                    logger.warning(f"Rejecting SELL on {symbol}: News sentiment strongly Bullish ({score:+.2f}).")
                                    continue
                                if score < -0.4 and signal.signal_type == SignalType.BUY:
                                    logger.warning(f"Rejecting BUY on {symbol}: News sentiment strongly Bearish ({score:+.2f}).")
                                    continue
                                # Mild penalty only when sentiment is moderately against trade
                                if score > 0.2 and signal.signal_type == SignalType.SELL:
                                    signal.win_rate *= 0.90
                                elif score < -0.2 and signal.signal_type == SignalType.BUY:
                                    signal.win_rate *= 0.90

                            # 4. Sentiment & ML Ensemble (Optional augmentation)
                            ensemble = SignalEnsemble(symbol)
                            ta_dir = 1 if signal.signal_type in [SignalType.BUY, "buy"] else -1
                            ens_dir, ens_conf, _ = ensemble.evaluate(df_ta, ta_dir, signal.confidence)
                            
                            if ens_dir == 0:
                                logger.debug(f"Ensemble vetoed {symbol} signal.")
                                continue
                                
                            # 4.5 LLM Trade Validation (Kimi) with Multi-Timeframe Context
                            if hasattr(self.llm_validator, 'enabled') and self.llm_validator.enabled:
                                llm_decision = await self.llm_validator.validate_signal(
                                    signal, df_ta, market_context, macro_bias=macro_desc, h1_summary=h1_summary
                                )
                                
                                if llm_decision == "invalid":
                                    if self.llm_validator.mode == "strict":
                                        logger.warning(f"[{symbol}] Trade REJECTED by Kimi LLM Validation.")
                                        continue
                                    else:
                                        signal.confidence *= 0.3 # Heavily penalize
                                        
                                elif llm_decision == "watch":
                                    signal.confidence *= 0.7 # Slight penalty
                                    
                            # Override TA signal type if ensemble agrees
                            signal.signal_type = SignalType.BUY if ens_dir == 1 else SignalType.SELL
                            
                            # 5. Risk Management & Stop Loss
                            signal = self.stop_loss_manager.apply_risk_parameters(signal, df_ta)
                            
                            if self.risk_manager.evaluate_signal(signal, account, open_positions, df_ta):
                                # 6. Execute Order
                                size = self.risk_manager.calculate_position_size(signal, account, df_ta)
                                if size > 0:
                                    # Decide execution algo
                                    from core.models import OrderType, OrderExecutionAlgo
                                    exec_algo = OrderExecutionAlgo.DEFAULT
                                    if signal.order_type == OrderType.LIMIT:
                                        exec_algo = OrderExecutionAlgo.TRAILING_LIMIT
                                    elif size >= 5.0:  # Huge order -> TWAP
                                        exec_algo = OrderExecutionAlgo.TWAP
                                        
                                    order = Order(
                                        signal_id=signal.id,
                                        symbol=symbol,
                                        asset_class=asset_class,
                                        order_type=signal.order_type,
                                        side=OrderSide.BUY if signal.signal_type in [SignalType.BUY, "buy"] else OrderSide.SELL,
                                        quantity=size,
                                        price=signal.limit_price if signal.limit_price > 0 else None,
                                        stop_loss=signal.stop_loss,
                                        take_profit=signal.take_profit,
                                        execution_algo=exec_algo
                                    )
                                    
                                    res_order = await self.executor.open_order(order)
                                    if res_order.status == OrderStatus.FILLED or res_order.status == OrderStatus.PENDING:
                                        # PENDING is also a valid state for limit orders
                                        await event_bus.publish(Event(EventType.ORDER_FILLED, res_order))
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                
            # Sleep until next scan interval
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            sleep_time = max(1.0, self.scan_interval - elapsed)
            await asyncio.sleep(sleep_time)
