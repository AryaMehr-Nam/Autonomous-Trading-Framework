"""
TradeAgent CCXT Data Provider.

Integration with Crypto exchanges via CCXT.
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

try:
    import ccxt.async_support as ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

from config.settings import settings
from core.logger import get_logger
from core.models import AssetClass, Candle, Tick, Timeframe
from data.base_provider import BaseDataProvider

logger = get_logger("ccxt_provider")


class CCXTDataProvider(BaseDataProvider):
    
    def __init__(self, exchange_id: str = "binance"):
        super().__init__(name=f"ccxt_{exchange_id}")
        self.supported_assets = [AssetClass.CRYPTO]
        self.exchange_id = exchange_id
        self.exchange = None

    async def connect(self) -> bool:
        if not HAS_CCXT:
            logger.error("ccxt is not installed.")
            return False
            
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Use credentials if available, otherwise public only
            config = {"enableRateLimit": True}
            if self.exchange_id == "binance" and settings.has_binance:
                config["apiKey"] = settings.binance_api_key
                config["secret"] = settings.binance_secret
                
            self.exchange = exchange_class(config)
            await self.exchange.load_markets()
            self.is_connected = True
            logger.info(f"CCXT Provider connected to {self.exchange_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to CCXT {self.exchange_id}: {e}")
            return False

    async def disconnect(self) -> None:
        if self.exchange:
            await self.exchange.close()
        self.is_connected = False
        logger.info(f"CCXT Provider {self.exchange_id} disconnected.")

    async def get_symbols(self) -> List[str]:
        if not self.is_connected or not self.exchange:
            return []
        return list(self.exchange.symbols)

    def normalize_symbol(self, symbol: str) -> str:
        # ccxt uses standard 'BTC/USDT' natively in most cases
        return symbol

    def _map_timeframe(self, tf: Timeframe) -> str:
        mapping = {
            Timeframe.M1: "1m", Timeframe.M5: "5m",
            Timeframe.M15: "15m", Timeframe.M30: "30m",
            Timeframe.H1: "1h", Timeframe.H4: "4h",
            Timeframe.D1: "1d", Timeframe.W1: "1w", Timeframe.MN1: "1M"
        }
        return mapping.get(tf, "1h")

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Candle]:
        if not self.is_connected or not self.exchange:
            return []

        ccxt_symbol = self.normalize_symbol(symbol)
        tf_str = self._map_timeframe(timeframe)
        since_ms = int(since.timestamp() * 1000) if since else None
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                ccxt_symbol, timeframe=tf_str, since=since_ms, limit=limit
            )
            
            candles = []
            for row in ohlcv:
                # row: [timestamp, open, high, low, close, volume]
                dt = datetime.fromtimestamp(row[0] / 1000.0)
                candles.append(Candle(
                    symbol=symbol,
                    asset_class=AssetClass.CRYPTO,
                    timeframe=timeframe,
                    timestamp=dt,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5])
                ))
            return candles
        except Exception as e:
            logger.error(f"CCXT API error fetching candles for {symbol}: {e}")
            return []

    async def stream_ticks(self, symbols: List[str]) -> AsyncGenerator[Tick, None]:
        # Simple REST polling for demo
        # A real implementation uses ccxt.pro for WebSockets
        if not self.is_connected or not self.exchange:
            return
            
        while self.is_connected:
            try:
                tickers = await self.exchange.fetch_tickers(symbols)
                for sym, t in tickers.items():
                    if t.get("bid") and t.get("ask"):
                        yield Tick(
                            symbol=sym,
                            asset_class=AssetClass.CRYPTO,
                            timestamp=datetime.utcnow(),
                            bid=float(t["bid"]),
                            ask=float(t["ask"])
                        )
                await asyncio.sleep(5.0)
            except Exception as e:
                logger.error(f"CCXT Tick stream error: {e}")
                await asyncio.sleep(10.0)

    async def stream_candles(self, symbol: str, timeframe: Timeframe) -> AsyncGenerator[Candle, None]:
        yield Candle("", AssetClass.CRYPTO, timeframe, datetime.now(), 0.0, 0.0, 0.0, 0.0, 0.0)
