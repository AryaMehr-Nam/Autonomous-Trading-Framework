"""
Market Context Agent.
Evaluates macro trend indicators to establish a definitive market bias.
"""
import pandas as pd
from typing import Tuple
from enum import Enum
from core.logger import get_logger

logger = get_logger("context_agent")

class MarketContext(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    RANGING_BULLISH = "RANGING_BULLISH"
    RANGING_BEARISH = "RANGING_BEARISH"
    NEUTRAL = "NEUTRAL"

class ContextAgent:
    def __init__(self):
        pass
        
    def evaluate_context(self, symbol: str, df: pd.DataFrame) -> MarketContext:
        """
        Determines the current market context based on moving averages and ADX.
        Returns MarketContext Enum.
        """
        if df is None or df.empty or len(df) < 200:
            return MarketContext.NEUTRAL
            
        latest = df.iloc[-1]
        close = latest.get("close", 0)
        
        # Fetch key context indicators (Use EMA_200 since SMA is not configured)
        ema_200 = latest.get("EMA_200")
        ema_50 = latest.get("EMA_50")
        adx = latest.get("ADX")
        
        if pd.isna(ema_200) or pd.isna(ema_50) or pd.isna(adx):
            return MarketContext.NEUTRAL
            
        bullish_bias = close > ema_200 and close > ema_50
        bearish_bias = close < ema_200 and close < ema_50
        
        is_trending = adx > 25
        
        if is_trending:
            if bullish_bias:
                return MarketContext.STRONG_BULLISH
            elif bearish_bias:
                return MarketContext.STRONG_BEARISH
        else:
            if bullish_bias or close > ema_200:
                return MarketContext.RANGING_BULLISH
            elif bearish_bias or close < ema_200:
                return MarketContext.RANGING_BEARISH
                
        return MarketContext.NEUTRAL

    def evaluate_macro_bias(self, df_h1: pd.DataFrame) -> Tuple[int, str]:
        """
        Calculates Higher-Timeframe (H1/H4) macro trend direction.
        Returns:
            Tuple[int, str]: (1 for Bullish, -1 for Bearish, 0 for Neutral, description)
        """
        if df_h1 is None or df_h1.empty or len(df_h1) < 50:
            return 0, "INSUFFICIENT_MACRO_DATA"
            
        latest = df_h1.iloc[-1]
        close = latest.get("close", 0)
        
        # Check EMA 99 / EMA 50 / EMA 200
        ema_99 = latest.get("EMA_99")
        if ema_99 is None or pd.isna(ema_99):
            # Compute on the fly if not in columns
            ema_99 = df_h1['close'].ewm(span=99).mean().iloc[-1]
            
        ema_13 = latest.get("EMA_13")
        if ema_13 is None or pd.isna(ema_13):
            ema_13 = df_h1['close'].ewm(span=13).mean().iloc[-1]
            
        adx = latest.get("ADX", 20)
        
        if close > ema_99 and ema_13 > ema_99:
            return 1, f"MACRO_BULLISH (Close {close:.5f} > EMA99 {ema_99:.5f})"
        elif close < ema_99 and ema_13 < ema_99:
            return -1, f"MACRO_BEARISH (Close {close:.5f} < EMA99 {ema_99:.5f})"
        elif close > ema_99:
            return 1, f"MACRO_LEAN_BULLISH (Close > EMA99)"
        elif close < ema_99:
            return -1, f"MACRO_LEAN_BEARISH (Close < EMA99)"
            
        return 0, "MACRO_NEUTRAL"

