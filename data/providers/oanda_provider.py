"""
TradeAgent OANDA Data Provider.

Integration with OANDA v20 REST API.
"""

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional

import pandas as pd
try:
    from oandapyV20 import API
    import oandapyV20.endpoints.instruments as instruments
    import oandapyV20.endpoints.pricing as pricing
    HAS_OANDA = True
except ImportError:
    HAS_OANDA = False

from config.settings import settings
from core.logger import get_logger
from core.models import AssetClass, Candle, Tick, Timeframe
from data.base_provider import BaseDataProvider

logger = get_logger("oanda_provider")


class OandaDataProvider(BaseDataProvider):
    
    def __init__(self):
        super().__init__(name="oanda")
        self.supported_assets = [AssetClass.FOREX]
        self.api = None
        self.account_id = settings.oanda_account_id

    async def connect(self) -> bool:
        if not HAS_OANDA:
            logger.error("oandapyV20 is not installed.")
            return False
        if not settings.has_oanda:
            logger.warning("OANDA API keys not configured. Use DemoProvider instead.")
            return False
            
        try:
            self.api = API(
                access_token=settings.oanda_api_key,
                environment=settings.oanda_environment
            )
            self.is_connected = True
            logger.info("OANDA Data Provider connected.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")
            return False

    async def disconnect(self) -> None:
        self.is_connected = False
        self.api = None
        logger.info("OANDA Data Provider disconnected.")

    async def get_symbols(self) -> List[str]:
        # Typically you'd fetch from an account endpoint. Returning a hardcoded list for brevity.
        return ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"]

    def normalize_symbol(self, symbol: str) -> str:
        """Standard 'EUR/USD' -> 'EUR_USD'"""
        return symbol.replace("/", "_")

    def standardize_symbol(self, broker_symbol: str) -> str:
        """'EUR_USD' -> 'EUR/USD'"""
        return broker_symbol.replace("_", "/")

    def _map_timeframe(self, tf: Timeframe) -> str:
        mapping = {
            Timeframe.M1: "M1", Timeframe.M5: "M5",
            Timeframe.M15: "M15", Timeframe.M30: "M30",
            Timeframe.H1: "H1", Timeframe.H4: "H4",
            Timeframe.D1: "D", Timeframe.W1: "W", Timeframe.MN1: "M"
        }
        return mapping.get(tf, "H1")

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Candle]:
        if not self.is_connected or not self.api:
            return []

        oanda_symbol = self.normalize_symbol(symbol)
        granularity = self._map_timeframe(timeframe)
        
        params = {"granularity": granularity, "count": min(limit, 5000)}
        if since:
            # Format to RFC3339
            params["from"] = since.replace(tzinfo=timezone.utc).isoformat()
            # If using 'from', OANDA requires dropping 'count' or keeping it <= 5000 depending on the endpoint.
        
        try:
            req = instruments.InstrumentsCandles(instrument=oanda_symbol, params=params)
            
            def _fetch():
                return self.api.request(req)
                
            resp = await asyncio.to_thread(_fetch)
            
            candles = []
            for c in resp.get("candles", []):
                if not c.get("complete", False):
                    continue
                # Parsing UTC time string
                dt = pd.to_datetime(c["time"]).to_pydatetime().replace(tzinfo=None)
                candles.append(Candle(
                    symbol=symbol,
                    asset_class=AssetClass.FOREX,
                    timeframe=timeframe,
                    timestamp=dt,
                    open=float(c["mid"]["o"]),
                    high=float(c["mid"]["h"]),
                    low=float(c["mid"]["l"]),
                    close=float(c["mid"]["c"]),
                    volume=float(c["volume"])
                ))
            return candles
        except Exception as e:
            logger.error(f"OANDA API error fetching candles for {symbol}: {e}")
            return []

    async def stream_ticks(self, symbols: List[str]) -> AsyncGenerator[Tick, None]:
        # Uses pricing stream endpoint in a real scenario.
        # For simplicity in this plan, polling pricing endpoint.
        if not self.is_connected or not self.api:
            return
            
        oanda_symbols = [self.normalize_symbol(s) for s in symbols]
        
        while self.is_connected:
            try:
                params = {"instruments": ",".join(oanda_symbols)}
                req = pricing.PricingInfo(accountID=self.account_id, params=params)
                
                def _fetch():
                    return self.api.request(req)
                    
                resp = await asyncio.to_thread(_fetch)
                
                for p in resp.get("prices", []):
                    inst = self.standardize_symbol(p["instrument"])
                    # Simple average if multiple bids/asks are provided
                    bid = float(p["bids"][0]["price"]) if p.get("bids") else 0.0
                    ask = float(p["asks"][0]["price"]) if p.get("asks") else 0.0
                    
                    if bid > 0 and ask > 0:
                        yield Tick(
                            symbol=inst,
                            asset_class=AssetClass.FOREX,
                            timestamp=datetime.utcnow(),
                            bid=bid,
                            ask=ask
                        )
                
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"OANDA Tick stream error: {e}")
                await asyncio.sleep(5.0)

    async def stream_candles(self, symbol: str, timeframe: Timeframe) -> AsyncGenerator[Candle, None]:
        yield Candle("", AssetClass.FOREX, timeframe, datetime.now(), 0.0, 0.0, 0.0, 0.0, 0.0)
