"""
TradeAgent Abstract Data Provider.

Defines the interface that all market data providers (OANDA, CCXT, Demo)
must implement to ensure a unified data layer for the rest of the system.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional
from datetime import datetime

import pandas as pd

from core.models import AssetClass, Candle, Tick, Timeframe


class BaseDataProvider(ABC):
    """
    Abstract base class for all market data providers.
    Standardizes how data is fetched and streamed regardless of the broker.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self.supported_assets: List[AssetClass] = []

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the provider."""
        pass

    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """Get a list of all supported symbols."""
        pass

    @abstractmethod
    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Candle]:
        """
        Fetch historical OHLCV data.
        Returns a list of standardized Candle objects.
        """
        pass

    async def get_historical_dataframe(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical data and return as a Pandas DataFrame.
        This is a helper method that uses `get_historical_candles`.
        """
        candles = await self.get_historical_candles(symbol, timeframe, since, limit)
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            }
            for c in candles
        ])
        df.set_index("timestamp", inplace=True)
        return df

    @abstractmethod
    async def stream_ticks(self, symbols: List[str]) -> AsyncGenerator[Tick, None]:
        """
        Subscribe to live tick data.
        Yields normalized Tick objects.
        """
        # Python requires an empty yield for async generators in abstract methods
        yield Tick("", AssetClass.FOREX, datetime.now(), 0.0, 0.0)

    @abstractmethod
    async def stream_candles(self, symbol: str, timeframe: Timeframe) -> AsyncGenerator[Candle, None]:
        """
        Subscribe to live closed candles.
        Yields normalized Candle objects when they close.
        """
        yield Candle("", AssetClass.FOREX, timeframe, datetime.now(), 0.0, 0.0, 0.0, 0.0, 0.0)

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize standard symbols (e.g., 'EUR/USD') to broker-specific format (e.g., 'EUR_USD').
        Can be overridden by subclasses.
        """
        return symbol

    def standardize_symbol(self, broker_symbol: str) -> str:
        """
        Standardize broker-specific symbol to standard format.
        Can be overridden by subclasses.
        """
        return broker_symbol
