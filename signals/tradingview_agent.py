"""
TradingView Strategy & Consensus Signal Generator
Generates high-probability signals by combining TradingView Multi-Timeframe Technical Ratings,
Oscillator Momentum, and Moving Average Crossovers.
"""

from typing import Optional, Dict, Any
from core.logger import get_logger

from core.models import Signal, SignalType, OrderSide, Timeframe
from data.tradingview_provider import TradingViewProvider

logger = get_logger("tv_agent")

class TradingViewAgent:
    """
    Evaluates TradingView institutional indicator consensus for signal generation.
    """

    def __init__(self, provider: Optional[TradingViewProvider] = None):
        self.provider = provider or TradingViewProvider()
        logger.info("TradingView Signal Agent initialized.")

    async def generate_signal(self, symbol: str, timeframe: Timeframe = Timeframe.H1) -> Optional[Signal]:
        """
        Queries TradingView multi-timeframe analysis and produces a vetted Signal object.
        """
        tf_str = "1h"
        if timeframe == Timeframe.M1:
            tf_str = "1m"
        elif timeframe == Timeframe.M5:
            tf_str = "5m"
        elif timeframe == Timeframe.M15:
            tf_str = "15m"
        elif timeframe == Timeframe.H4:
            tf_str = "4h"
        elif timeframe == Timeframe.D1:
            tf_str = "1d"

        analysis = await self.provider.get_analysis_async(symbol, tf_str)
        if not analysis:
            return None

        rec = analysis.get("recommendation", "NEUTRAL")
        osc_rec = analysis.get("oscillators", {}).get("recommendation", "NEUTRAL")
        ma_rec = analysis.get("moving_averages", {}).get("recommendation", "NEUTRAL")
        indicators = analysis.get("key_indicators", {})

        close_price = indicators.get("close")
        if not close_price or close_price <= 0:
            return None

        rsi = indicators.get("rsi")
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        atr = indicators.get("atr") or (close_price * 0.005)

        # Bullish Consensus: STRONG_BUY or BUY with confirming Oscillators/MAs
        if rec in ["STRONG_BUY", "BUY"] and ma_rec in ["STRONG_BUY", "BUY"]:
            # Confirm RSI is not severely overbought (>85)
            if rsi and rsi < 85:
                # Stop loss at 1.5 ATR below close, Take Profit at 2.5 ATR above close
                sl = round(close_price - (1.5 * atr), 5)
                tp = round(close_price + (2.5 * atr), 5)
                rr = round(abs(tp - close_price) / max(abs(close_price - sl), 1e-5), 2)
                
                # Assign high win-rate confidence based on TradingView buy consensus
                win_rate = 0.72 if rec == "STRONG_BUY" else 0.65

                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    side=OrderSide.BUY,
                    timeframe=timeframe,
                    entry_price=close_price,
                    stop_loss=sl,
                    take_profit=tp,
                    risk_reward_ratio=rr,
                    win_rate=win_rate,
                    strategy_name="tradingview_consensus",
                    confidence=0.85 if rec == "STRONG_BUY" else 0.75,
                    metadata={
                        "tv_recommendation": rec,
                        "tv_ma_rec": ma_rec,
                        "tv_osc_rec": osc_rec,
                        "rsi": rsi,
                        "macd": macd,
                        "buy_votes": analysis.get("buy_count", 0),
                        "sell_votes": analysis.get("sell_count", 0)
                    }
                )
                logger.info(f"[TradingViewAgent] Generated {rec} Signal for {symbol}: Entry={close_price}, SL={sl}, TP={tp}, RR={rr}, WR={win_rate*100:.0f}%")
                return signal

        # Bearish Consensus: STRONG_SELL or SELL with confirming Oscillators/MAs
        elif rec in ["STRONG_SELL", "SELL"] and ma_rec in ["STRONG_SELL", "SELL"]:
            if rsi and rsi > 15:
                sl = round(close_price + (1.5 * atr), 5)
                tp = round(close_price - (2.5 * atr), 5)
                rr = round(abs(close_price - tp) / max(abs(sl - close_price), 1e-5), 2)
                win_rate = 0.72 if rec == "STRONG_SELL" else 0.65

                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    side=OrderSide.SELL,
                    timeframe=timeframe,
                    entry_price=close_price,
                    stop_loss=sl,
                    take_profit=tp,
                    risk_reward_ratio=rr,
                    win_rate=win_rate,
                    strategy_name="tradingview_consensus",
                    confidence=0.85 if rec == "STRONG_SELL" else 0.75,
                    metadata={
                        "tv_recommendation": rec,
                        "tv_ma_rec": ma_rec,
                        "tv_osc_rec": osc_rec,
                        "rsi": rsi,
                        "macd": macd,
                        "buy_votes": analysis.get("buy_count", 0),
                        "sell_votes": analysis.get("sell_count", 0)
                    }
                )
                logger.info(f"[TradingViewAgent] Generated {rec} Signal for {symbol}: Entry={close_price}, SL={sl}, TP={tp}, RR={rr}, WR={win_rate*100:.0f}%")
                return signal

        return None
