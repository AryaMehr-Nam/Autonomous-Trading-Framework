"""
TradeAgent Technical Analysis Signals.

Evaluates the computed DataFrame from TAEngine to generate actionable trading signals
using multiple algorithmic trading strategies (Trend, Mean Reversion, Breakout).
Dynamically injects machine-learned parameters from the Strategy Discoverer if available.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import pandas_ta as ta

from config.settings import DATA_DIR, trading_params
from core.logger import get_logger
from core.models import AssetClass, Signal, SignalType, Timeframe, OrderType
from signals.smart_money_concepts import detect_smc_signal
from signals.regime_detector import RegimeDetector, MarketRegime

logger = get_logger("ta_signals")


class TASignalGenerator:
    """Generates signals using multiple distinct strategies with self-learning parameters."""
    
    def __init__(self):
        self.config = trading_params.technical
        self.min_score = self.config.get("min_confluence_score", 0.6)
        self.regime_detector = RegimeDetector()
        
        # Load the "Brain" (Optimized Strategies)
        self.brain = self._load_brain()

    def reload_brain(self):
        """Dynamically reload the learned parameters if the JSON files were updated by the Walk-Forward Optimizer."""
        self.brain = self._load_brain()
        logger.debug("Strategy brain reloaded from JSON files.")

    def _load_brain(self) -> dict:
        """Load the dynamically learned parameters from all Strategy Agents."""
        brain = {}
        strategy_types = ["trend", "mean_reversion", "breakout", "smc"]
        
        for strat in strategy_types:
            path = DATA_DIR / "strategies" / f"optimized_{strat}.json"
            if path.exists():
                try:
                    with open(path, "r") as f:
                        brain[strat] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading {strat} brain: {e}")
        return brain

    def get_learned_data(self, symbol: str, timeframe: Timeframe, strategy: str) -> Optional[dict]:
        """Fetch full optimized data dict for a specific asset/timeframe/strategy."""
        if not self.brain or strategy not in self.brain:
            return None
        
        strat_brain = self.brain[strategy]
        sym = symbol.replace("/", "")
        tf = timeframe.value
        
        # Support both multi-asset nested format and simple single-asset root format produced by Optuna
        if sym in strat_brain and tf in strat_brain[sym]:
            return strat_brain[sym][tf]
        elif "parameters" in strat_brain or "best_params" in strat_brain:
            return strat_brain
        return None

    def generate_signal(self, symbol: str, df: pd.DataFrame, timeframe: Timeframe, asset_class: AssetClass) -> Optional[Signal]:
        """
        Evaluate the latest row in the DataFrame across multiple strategies.
        Returns the strongest Signal object, or None if hold.
        """
        if df is None or df.empty or len(df) < 50:
            return None
            
        latest = df.iloc[-1]
        
        # Load parameters specific to each agent's strategy
        trend_data = self.get_learned_data(symbol, timeframe, "trend")
        mr_data = self.get_learned_data(symbol, timeframe, "mean_reversion")
        brk_data = self.get_learned_data(symbol, timeframe, "breakout")
        
        trend_params = trend_data.get("parameters", trend_data.get("best_params")) if trend_data else None
        mr_params = mr_data.get("parameters", mr_data.get("best_params")) if mr_data else None
        brk_params = brk_data.get("parameters", brk_data.get("best_params")) if brk_data else None
        
        # Determine Timeframe Horizon
        is_short_term = timeframe in [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30]
        
        # ── ASSET-SPECIFIC STRATEGY ROUTING ──────────────────────────────
        # Check if the asset has a specific preferred strategy from Optuna backtests
        asset_strat_map = getattr(trading_params, "asset_strategies", {}) or {}
        preferred_strat = asset_strat_map.get(symbol, None)
        
        # ── REGIME-BASED STRATEGY SELECTION ──────────────────────────────
        regime = self.regime_detector.detect(df)
        active_strategy = preferred_strat if preferred_strat else self.regime_detector.best_strategy_for(regime)
        
        # Always give SMC a chance in volatile regimes or when preferred
        run_trend = (active_strategy == "trend")
        run_mr    = (active_strategy == "mean_reversion")
        run_brk   = (active_strategy == "breakout")
        run_smc   = (active_strategy == "smc") or (regime == MarketRegime.VOLATILE)
        
        # Fallback: if none selected, run regime default
        if not (run_trend or run_mr or run_brk or run_smc):
            run_trend = not is_short_term
            run_smc = True
        
        if run_trend:
            trend_dir, trend_conf, trend_ot, trend_pr = self._strategy_trend_following(latest, df, trend_params)
        else:
            trend_dir, trend_conf, trend_ot, trend_pr = (0, 0.0, OrderType.MARKET, 0.0)
            
        if run_mr:
            mr_dir, mr_conf, mr_ot, mr_pr = self._strategy_mean_reversion(latest, df, mr_params)
        else:
            mr_dir, mr_conf, mr_ot, mr_pr = (0, 0.0, OrderType.MARKET, 0.0)
            
        if run_brk:
            brk_dir, brk_conf, brk_ot, brk_pr = self._strategy_breakout(latest)
        else:
            brk_dir, brk_conf, brk_ot, brk_pr = (0, 0.0, OrderType.MARKET, 0.0)
            
        if run_smc:
            smc_data = self.get_learned_data(symbol, timeframe, "smc")
            smc_params = smc_data.get("parameters", smc_data.get("best_params")) if smc_data else None
            smc_dir, smc_conf, smc_ot, smc_pr = detect_smc_signal(df, smc_params)
        else:
            smc_dir, smc_conf, smc_ot, smc_pr = (0, 0.0, OrderType.MARKET, 0.0)
            smc_data = None
        
        # Find the strategy with the highest confidence above min_score
        strategies = [
            ("Trend Following (Long Term)", trend_dir, trend_conf, trend_ot, trend_pr),
            ("Mean Reversion (Short Term)", mr_dir, mr_conf, mr_ot, mr_pr),
            ("Volatility Breakout (Long Term)", brk_dir, brk_conf, brk_ot, brk_pr),
            ("Smart Money Concepts (SMC)", smc_dir, smc_conf, smc_ot, smc_pr)
        ]
        
        best_strat = "None"
        best_dir = 0
        best_conf = 0.0
        best_ot = OrderType.MARKET
        best_pr = 0.0
        
        for name, direction, conf, ot, pr in strategies:
            if conf > best_conf and conf >= self.min_score:
                best_conf = conf
                best_dir = direction
                best_strat = name
                best_ot = ot
                best_pr = pr
                
        if best_dir == 0:
            return None
            
        signal_type = SignalType.BUY if best_dir == 1 else SignalType.SELL
            
        # Determine which strategy won to extract its learned data
        best_strat_data = None
        if "Trend Following" in best_strat: best_strat_data = trend_data
        elif "Mean Reversion" in best_strat: best_strat_data = mr_data
        elif "Volatility Breakout" in best_strat: best_strat_data = brk_data
        elif "SMC" in best_strat: best_strat_data = smc_data

        win_rate = 0.50
        risk_reward = 2.0

        if best_strat_data:
            raw_wr = best_strat_data.get("win_rate_pct", 50.0) / 100.0
            # For SMC: the strategy has a HIGH RR (1:5+) which compensates for low WR.
            # Kelly formula: f* = W - (1-W)/R  → needs R to be set correctly.
            if "SMC" in best_strat:
                risk_reward = 5.0  # SMC targets 1:5 RR
                # Minimum WR floor for positive Kelly: W > 1/(R+1) = 1/6 = 0.167
                # We add a small buffer: 0.20 to ensure positive expectancy
                win_rate = max(raw_wr, 0.20)
            elif "Mean Reversion" in best_strat:
                risk_reward = 2.0  # MR targets 1:2 RR
                win_rate = max(raw_wr, 0.40)  # Backtest shows 44.4%
            elif "Trend Following" in best_strat:
                risk_reward = 3.0  # Trend trades target 1:3 RR
                win_rate = max(raw_wr, 0.35)
            else:
                win_rate = max(raw_wr, 0.35)

        # Compile indicators for metadata
        indicators = {
            "rsi": round(latest.get("RSI", 0), 2),
            "atr": round(latest.get("ATR", 0), 5),
            "adx": round(latest.get("ADX", 0), 2)
        }
        
        # Tag whether it used master params
        source_tag = f"ta_{best_strat.lower().replace(' ', '_')}"
        if best_strat_data:
            source_tag += "_ml_optimized"
            
        reasoning = f"Strategy: {best_strat} | Confidence: {round(best_conf, 2)}"
        if best_strat_data:
            reasoning += f" | *Master Learned Params (WR: {win_rate*100:.1f}%)*"
        
        return Signal(
            symbol=symbol,
            asset_class=asset_class,
            signal_type=signal_type,
            confidence=best_conf,
            timeframe=timeframe,
            order_type=best_ot,
            limit_price=best_pr,
            entry_price=latest['close'],
            technical_score=best_conf * best_dir,
            source=source_tag,
            reasoning=reasoning,
            indicators=indicators,
            win_rate=win_rate,
            risk_reward=risk_reward
        )

    def _strategy_trend_following(self, row: pd.Series, df: pd.DataFrame, params: Optional[dict]) -> Tuple[int, float, OrderType, float]:
        """
        Trend Following Strategy with TradingView UT Bot & Lorentzian ML Enhancements.
        Returns (direction (-1, 0, 1), confidence, order_type, limit_price).
        """
        score = 0.0
        max_score = 4.0
        direction = 0
        
        # Dynamically calculate EMAs if we have master parameters
        if params and "ema_fast" in params and "ema_slow" in params:
            fast_ma = ta.ema(df['close'], length=params['ema_fast']).iloc[-1]
            slow_ma = ta.ema(df['close'], length=params['ema_slow']).iloc[-1]
        else:
            fast_ma = ta.ema(df['close'], length=13).iloc[-1]
            slow_ma = ta.ema(df['close'], length=99).iloc[-1]
            
        # 1. EMA Alignment
        if fast_ma is not None and slow_ma is not None and not pd.isna(fast_ma) and not pd.isna(slow_ma):
            if fast_ma > slow_ma:
                score += 1.0
                direction = 1
            elif fast_ma < slow_ma:
                score += 1.0
                direction = -1
                
        # 2. MACD Agreement
        if "MACD_HIST" in row and direction != 0:
            if direction == 1 and row["MACD_HIST"] > 0: score += 0.5
            elif direction == -1 and row["MACD_HIST"] < 0: score += 0.5
            
        # 3. UT Bot Alerts (QuantNomad Trailing Confirmation)
        if "UT_POSITION" in row and direction != 0:
            if direction == 1 and row.get("UT_POSITION") == 1: score += 1.0
            elif direction == -1 and row.get("UT_POSITION") == -1: score += 1.0
            
        # 4. Lorentzian ML Classification (jdehorty Non-Euclidean k-NN)
        if "LORENTZIAN_PRED" in row and direction != 0:
            l_pred = row.get("LORENTZIAN_PRED", 0)
            l_conf = row.get("LORENTZIAN_CONF", 0.5)
            if l_pred == direction and l_conf >= 0.6:
                score += 1.0 * l_conf
                
        # 5. Trend Strength (ADX)
        if "ADX" in row:
            if row["ADX"] > 25:
                score += 0.5  # Strong trend
            elif row["ADX"] < 18:
                score -= 0.5  # Weak trend / chop
                
        confidence = max(0.0, min(1.0, score / max_score))
        return direction, confidence, OrderType.MARKET, 0.0

    def _strategy_mean_reversion(self, row: pd.Series, df: pd.DataFrame, params: Optional[dict]) -> Tuple[int, float, OrderType, float]:
        """
        Mean Reversion Strategy with TradingView WaveTrend & Nadaraya-Watson Envelope Enhancements.
        Returns (direction (-1, 0, 1), confidence, order_type, limit_price).
        """
        score = 0.0
        max_score = 4.0
        direction = 0
        limit_price = 0.0
        
        if params and "rsi_period" in params:
            rsi = ta.rsi(df['close'], length=params['rsi_period']).iloc[-1]
            rsi_buy = params.get('rsi_buy', 30)
            rsi_sell = params.get('rsi_sell', 70)
        else:
            rsi = row.get("RSI")
            rsi_buy = 30
            rsi_sell = 70
        
        # 1. RSI Extremes
        if rsi is not None and not pd.isna(rsi):
            if rsi < rsi_buy:
                direction = 1
                score += 1.0 + min(0.5, (rsi_buy - rsi) / 30)  
            elif rsi > rsi_sell:
                direction = -1
                score += 1.0 + min(0.5, (rsi - rsi_sell) / 30)
                
        # 2. WaveTrend Oscillator (Cipher B Crossovers at Extremes)
        if "WT1" in row and "WT2" in row:
            wt1 = row.get("WT1", 0)
            wt2 = row.get("WT2", 0)
            wt_bull = row.get("WT_BULL_CROSS", False) or (wt1 < -50 and wt1 > wt2)
            wt_bear = row.get("WT_BEAR_CROSS", False) or (wt1 > 50 and wt1 < wt2)
            
            if direction == 1 and wt_bull:
                score += 1.0
            elif direction == -1 and wt_bear:
                score += 1.0
            elif direction == 0:
                if wt_bull:
                    direction = 1
                    score += 1.0
                elif wt_bear:
                    direction = -1
                    score += 1.0

        # 3. Nadaraya-Watson Envelope (LuxAlgo Rational Quadratic Reversal)
        if "NW_LOWER" in row and "NW_UPPER" in row and direction != 0:
            close = row.get("close", 0)
            nw_lower = row.get("NW_LOWER", 0)
            nw_upper = row.get("NW_UPPER", 0)
            
            if direction == 1 and (close <= nw_lower or row.get("NW_BULL_REV", False)):
                score += 1.0
                limit_price = nw_lower
            elif direction == -1 and (close >= nw_upper or row.get("NW_BEAR_REV", False)):
                score += 1.0
                limit_price = nw_upper
                
        # 4. Bollinger Bands Extremes
        if "BB_PCT" in row and direction != 0:
            bbp = row["BB_PCT"]
            if direction == 1 and bbp < 0.1:
                score += 0.5
                if limit_price == 0.0:
                    limit_price = row.get("BBL_20_2.0", row["close"] * 0.999)
            elif direction == -1 and bbp > 0.9:
                score += 0.5
                if limit_price == 0.0:
                    limit_price = row.get("BBU_20_2.0", row["close"] * 1.001)
                
        # Avoid fading strong trends (ADX > 32)
        if "ADX" in row and row["ADX"] > 32:
            score -= 1.0
            
        confidence = max(0.0, min(1.0, score / max_score))
        ot = OrderType.LIMIT if limit_price > 0.0 else OrderType.MARKET
        return direction, confidence, ot, limit_price

    def _strategy_breakout(self, row: pd.Series) -> Tuple[int, float, OrderType, float]:
        """
        Volatility Breakout Strategy.
        Returns (direction (-1, 0, 1), confidence, order_type, limit_price).
        """
        score = 0.0
        max_score = 2.0
        direction = 0
        
        if "BB_PCT" in row and "ADX" in row:
            bbp = row["BB_PCT"]
            adx = row["ADX"]
            
            # Upside Breakout
            if bbp > 1.0 and adx > 25:
                direction = 1
                score += 1.0 + min(1.0, (adx - 25) / 25)
            # Downside Breakout
            elif bbp < 0.0 and adx > 25:
                direction = -1
                score += 1.0 + min(1.0, (adx - 25) / 25)
                
        # Confirm with RSI
        if "RSI" in row and direction != 0:
            rsi = row["RSI"]
            if direction == 1 and rsi > 50: score += 1.0
            elif direction == -1 and rsi < 50: score += 1.0
            
        confidence = max(0.0, min(1.0, score / max_score))
        return direction, confidence, OrderType.MARKET, 0.0
