"""
TradingView Data & Technical Analysis Provider
Fetches live multi-timeframe indicator analysis, oscillator states, moving average ratings,
and consensus recommendations directly from TradingView.
"""

import asyncio
from typing import Dict, Any, Optional
from core.logger import get_logger
from tradingview_ta import TA_Handler, Interval

logger = get_logger("tv_provider")

class TradingViewProvider:
    """Provides real-time technical analysis and indicators from TradingView."""

    SYMBOL_MAP = {
        "XAU/USD": {"symbol": "GOLD", "screener": "cfd", "exchange": "TVC"},
        "XAUUSD": {"symbol": "GOLD", "screener": "cfd", "exchange": "TVC"},
        "GOLD": {"symbol": "GOLD", "screener": "cfd", "exchange": "TVC"},
        "US30": {"symbol": "DJI", "screener": "america", "exchange": "DJ"},
        "DJI": {"symbol": "DJI", "screener": "america", "exchange": "DJ"},
        "EUR/USD": {"symbol": "EURUSD", "screener": "forex", "exchange": "FX_IDC"},
        "EURUSD": {"symbol": "EURUSD", "screener": "forex", "exchange": "FX_IDC"},
        "GBP/USD": {"symbol": "GBPUSD", "screener": "forex", "exchange": "FX_IDC"},
        "GBPUSD": {"symbol": "GBPUSD", "screener": "forex", "exchange": "FX_IDC"},
        "USD/JPY": {"symbol": "USDJPY", "screener": "forex", "exchange": "FX_IDC"},
        "USDJPY": {"symbol": "USDJPY", "screener": "forex", "exchange": "FX_IDC"},
        "BTC/USDT": {"symbol": "BTCUSDT", "screener": "crypto", "exchange": "BINANCE"},
        "BTCUSD": {"symbol": "BTCUSDT", "screener": "crypto", "exchange": "BINANCE"},
        "ETH/USDT": {"symbol": "ETHUSDT", "screener": "crypto", "exchange": "BINANCE"},
        "ETHUSD": {"symbol": "ETHUSDT", "screener": "crypto", "exchange": "BINANCE"},
    }

    INTERVAL_MAP = {
        "1m": Interval.INTERVAL_1_MINUTE,
        "5m": Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "1h": Interval.INTERVAL_1_HOUR,
        "4h": Interval.INTERVAL_4_HOURS,
        "1d": Interval.INTERVAL_1_DAY,
        "1W": Interval.INTERVAL_1_WEEK,
        "1M": Interval.INTERVAL_1_MONTH,
    }

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info("TradingView Technical Analysis Provider initialized.")

    def get_analysis(self, symbol: str, timeframe: str = "1h") -> Optional[Dict[str, Any]]:
        """
        Synchronously fetch TradingView technical analysis.
        """
        mapping = self.SYMBOL_MAP.get(symbol.upper())
        if not mapping:
            # Fallback guessing
            clean = symbol.replace("/", "").replace("_", "").upper()
            mapping = {"symbol": clean, "screener": "forex", "exchange": "FX_IDC"}

        tv_interval = self.INTERVAL_MAP.get(timeframe.lower(), Interval.INTERVAL_1_HOUR)

        try:
            handler = TA_Handler(
                symbol=mapping["symbol"],
                screener=mapping["screener"],
                exchange=mapping["exchange"],
                interval=tv_interval
            )
            analysis = handler.get_analysis()
            
            result = {
                "symbol": symbol,
                "tv_symbol": mapping["symbol"],
                "exchange": mapping["exchange"],
                "timeframe": timeframe,
                "recommendation": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
                "buy_count": analysis.summary.get("BUY", 0),
                "sell_count": analysis.summary.get("SELL", 0),
                "neutral_count": analysis.summary.get("NEUTRAL", 0),
                "oscillators": {
                    "recommendation": analysis.oscillators.get("RECOMMENDATION", "NEUTRAL"),
                    "buy": analysis.oscillators.get("BUY", 0),
                    "sell": analysis.oscillators.get("SELL", 0),
                    "neutral": analysis.oscillators.get("NEUTRAL", 0),
                },
                "moving_averages": {
                    "recommendation": analysis.moving_averages.get("RECOMMENDATION", "NEUTRAL"),
                    "buy": analysis.moving_averages.get("BUY", 0),
                    "sell": analysis.moving_averages.get("SELL", 0),
                    "neutral": analysis.moving_averages.get("NEUTRAL", 0),
                },
                "key_indicators": {
                    "close": analysis.indicators.get("close"),
                    "rsi": analysis.indicators.get("RSI"),
                    "macd": analysis.indicators.get("MACD.macd"),
                    "macd_signal": analysis.indicators.get("MACD.signal"),
                    "adx": analysis.indicators.get("ADX"),
                    "ema20": analysis.indicators.get("EMA20"),
                    "ema50": analysis.indicators.get("EMA50"),
                    "ema200": analysis.indicators.get("EMA200"),
                    "sma50": analysis.indicators.get("SMA50"),
                    "sma200": analysis.indicators.get("SMA200"),
                    "stoch_k": analysis.indicators.get("Stoch.K"),
                    "stoch_d": analysis.indicators.get("Stoch.D"),
                    "cci": analysis.indicators.get("CCI20"),
                    "atr": analysis.indicators.get("ATR"),
                },
                "raw_indicators": analysis.indicators
            }
            return result
        except Exception as e:
            logger.error(f"[TradingViewProvider] Error fetching analysis for {symbol} ({timeframe}): {e}")
            return None

    async def get_analysis_async(self, symbol: str, timeframe: str = "1h") -> Optional[Dict[str, Any]]:
        """
        Asynchronously fetch TradingView analysis without blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_analysis, symbol, timeframe)
