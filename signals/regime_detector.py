"""
TradeAgent Market Regime Detector.

Dynamically identifies the current market regime (Trending, Ranging, Volatile)
and returns the best strategy to use in that regime.

Regimes:
  TRENDING  -> Use Trend Following (EMA crossovers, MACD)
  RANGING   -> Use Mean Reversion (RSI extremes, Bollinger Bands)
  VOLATILE  -> Use SMC Order Blocks (institutional footprints on impulse moves)
"""
import pandas as pd
from enum import Enum
from core.logger import get_logger

logger = get_logger("regime_detector")


class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class RegimeDetector:
    """
    Classifies current market conditions using ADX, ATR, and Bollinger Band Width.
    This allows the bot to dynamically select the best-performing strategy for the
    current environment rather than running all strategies blindly.
    """

    def __init__(self):
        self.adx_trend_threshold = 28.0    # ADX > 28 = strong trend
        self.adx_range_threshold = 18.0    # ADX < 18 = ranging/consolidation
        self.atr_volatile_pct = 0.45       # ATR/Close > 0.45% = volatile (Gold-tuned)
        self.bbw_squeeze_threshold = 0.02  # BB Width < 2% = compression = imminent breakout

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect the market regime from the last 20 bars.
        Returns a MarketRegime enum value.
        """
        if df is None or df.empty or len(df) < 20:
            return MarketRegime.UNKNOWN

        row = df.iloc[-1]

        adx = row.get("ADX", None)
        atr = row.get("ATR", None)
        close = row.get("close", None)

        # --- Volatile Check (takes priority) ---
        if atr is not None and close is not None and close > 0:
            atr_pct = (atr / close) * 100
            if atr_pct > self.atr_volatile_pct:
                logger.debug(f"[REGIME] VOLATILE detected (ATR%={atr_pct:.3f}%)")
                return MarketRegime.VOLATILE

        # --- Trend/Range Check via ADX ---
        if adx is not None:
            if adx >= self.adx_trend_threshold:
                logger.debug(f"[REGIME] TRENDING detected (ADX={adx:.1f})")
                return MarketRegime.TRENDING
            elif adx <= self.adx_range_threshold:
                logger.debug(f"[REGIME] RANGING detected (ADX={adx:.1f})")
                return MarketRegime.RANGING

        # --- Fallback: check BB width for squeeze (pre-breakout ranging) ---
        bbu = row.get("BBU_20_2.0", None)
        bbl = row.get("BBL_20_2.0", None)
        bbm = row.get("BBM_20_2.0", None)
        if bbu is not None and bbl is not None and bbm is not None and bbm > 0:
            bbw = (bbu - bbl) / bbm
            if bbw < self.bbw_squeeze_threshold:
                logger.debug(f"[REGIME] RANGING/SQUEEZE detected (BBW={bbw:.4f})")
                return MarketRegime.RANGING

        return MarketRegime.RANGING  # Default to ranging for Gold (most common state)

    def best_strategy_for(self, regime: MarketRegime) -> str:
        """
        Returns the name of the best strategy to use given the detected regime.
        These are based on the backtest optimization results:
          - Mean Reversion: 44.4% WR (best for ranging)
          - SMC Order Blocks: 16.7% WR but high RR (best for volatile/impulsive)
          - Trend Following: 0% WR on recent H1 data (only use in strong trends)
        """
        mapping = {
            MarketRegime.TRENDING: "trend",
            MarketRegime.RANGING: "mean_reversion",   # 44.4% win rate
            MarketRegime.VOLATILE: "smc",              # Catches institutional moves
            MarketRegime.UNKNOWN: "mean_reversion",    # Safe default
        }
        strategy = mapping[regime]
        logger.info(f"[REGIME] {regime.value.upper()} market -> Using '{strategy}' strategy")
        return strategy
