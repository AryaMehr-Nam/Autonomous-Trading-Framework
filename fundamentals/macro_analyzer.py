"""
TradeAgent Macro Analyzer.

Fetches macroeconomic data (rates, GDP, inflation) from FRED API
to compute fundamental scores and interest rate differentials.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

from config.settings import settings, trading_params
from core.logger import get_logger

logger = get_logger("macro_analyzer")


class MacroAnalyzer:
    def __init__(self):
        self.api_key = settings.fred_api_key
        self.fred = None
        # Map currencies to FRED Series IDs for Interest Rates
        self.rate_series_map = {
            "USD": "FEDFUNDS",   # US Fed Funds Rate
            "EUR": "ECBMRRFR",   # ECB Main Refinancing Rate
            "GBP": "IUDSOIA",    # BOE Official Bank Rate
            "JPY": "IRSTCI01JPM156N", # BOJ Rate (approx via short term rate)
            "CAD": "IRSTCI01CAM156N",
            "AUD": "IRSTCI01AUM156N",
            "CHF": "IRSTCI01CHM156N",
            "NZD": "IRSTCI01NZM156N"
        }
        self.latest_rates: Dict[str, float] = {}
        self.last_update: Optional[datetime] = None

    def connect(self) -> bool:
        if not HAS_FRED or not self.api_key:
            logger.warning("fredapi not installed or API key missing. Macro disabled.")
            return False
            
        try:
            self.fred = Fred(api_key=self.api_key)
            return True
        except Exception as e:
            logger.error(f"FRED connection error: {e}")
            return False

    async def update_rates(self):
        """Fetch latest interest rates for all supported currencies."""
        if not self.fred: return
        
        try:
            for ccy, series_id in self.rate_series_map.items():
                def _fetch():
                    # Fetch last 30 days to guarantee latest point
                    return self.fred.get_series(series_id, observation_start=datetime.utcnow() - timedelta(days=60))
                
                s = await asyncio.to_thread(_fetch)
                if not s.empty:
                    self.latest_rates[ccy] = float(s.iloc[-1])
                    
            self.last_update = datetime.utcnow()
            logger.info("Macro rates updated successfully.")
        except Exception as e:
            logger.error(f"Failed to update macro rates: {e}")

    def get_rate_differential(self, symbol: str) -> float:
        """
        Calculate interest rate differential for a pair (e.g. EUR/USD).
        Positive means base currency yields more (bullish for pair in carry trade).
        """
        if not self.latest_rates:
            return 0.0
            
        parts = symbol.replace("_", "/").replace("-", "/").split("/")
        if len(parts) != 2:
            return 0.0
            
        base_ccy, quote_ccy = parts[0], parts[1]
        
        base_rate = self.latest_rates.get(base_ccy, 0.0)
        quote_rate = self.latest_rates.get(quote_ccy, 0.0)
        
        return base_rate - quote_rate

    def get_fundamental_score(self, symbol: str) -> float:
        """
        Returns a score from -1.0 to 1.0 based on macro fundamentals.
        Currently primarily driven by interest rate differential (carry trade logic).
        """
        diff = self.get_rate_differential(symbol)
        threshold = trading_params.fundamentals.get("carry_trade_differential_threshold", 1.5)
        
        if abs(diff) < 0.5:
            return 0.0
            
        # Cap score at 1.0 (reached when diff >= threshold)
        score = diff / threshold
        return max(-1.0, min(1.0, score))
