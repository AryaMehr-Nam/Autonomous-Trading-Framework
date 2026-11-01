"""
TradeAgent Demo Data Provider.

A free, API-key-less data provider using yfinance.
Used for paper trading and backtesting when no broker API keys are available.
Supports stocks, crypto, and basic forex pairs (EOD/delayed).
"""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, List, Optional

import pandas as pd
import yfinance as yf

from core.logger import get_logger
from core.models import AssetClass, Candle, Tick, Timeframe
from data.base_provider import BaseDataProvider

logger = get_logger("demo_provider")


class DemoDataProvider(BaseDataProvider):
    """
    Downloads historical data from Yahoo Finance and simulates live streaming.
    Perfect for running the bot without any API keys.
    """

    def __init__(self):
        super().__init__(name="yfinance_demo")
        self.supported_assets = [AssetClass.EQUITY, AssetClass.CRYPTO, AssetClass.FOREX]
        self.is_connected = False

    async def connect(self) -> bool:
        """No actual connection needed for yfinance."""
        self.is_connected = True
        logger.info("Demo Data Provider 'connected' (yfinance ready)")
        return True

    async def disconnect(self) -> None:
        self.is_connected = False
        logger.info("Demo Data Provider disconnected")

    async def get_symbols(self) -> List[str]:
        """Return a basic list of popular symbols as a demo."""
        return [
            # Forex (yfinance uses =X suffix)
            "EURUSD=X", "GBPUSD=X", "JPY=X",
            # Crypto (yfinance uses -USD suffix)
            "BTC-USD", "ETH-USD", "SOL-USD",
            # Equity
            "AAPL", "MSFT", "TSLA", "SPY"
        ]

    def normalize_symbol(self, symbol: str) -> str:
        """Convert standard 'EUR/USD' or 'BTC/USDT' to yfinance format."""
        if symbol in ["XAU/USD", "GOLD"]: return "GC=F"
        if symbol in ["US30", "DAWJONZ", "DOW", "DOWJONES", "US30USD", "US30/USD"]: return "YM=F"
        if symbol in ["WTI/USD", "OIL", "CL/USD"]: return "CL=F"
        if symbol == "BTC/USDT": return "BTC-USD"
        if symbol == "ETH/USDT": return "ETH-USD"
        if symbol == "USD/JPY": return "JPY=X"
        
        if "/" in symbol:
            base, quote = symbol.split("/")
            forex_ccys = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "NZD", "CHF"}
            if base in forex_ccys and quote in forex_ccys:
                return f"{base}{quote}=X"
            elif quote == "USDT":
                return f"{base}-USD"
                
        return symbol.replace("/", "-")

    def standardize_symbol(self, broker_symbol: str) -> str:
        """Convert yfinance format back to standard."""
        if broker_symbol == "GC=F": return "XAU/USD"
        if broker_symbol in ["YM=F", "DIA", "^DJI"]: return "US30"
        if broker_symbol == "CL=F": return "WTI/USD"
        if broker_symbol.endswith("=X"):
            s = broker_symbol.replace("=X", "")
            if len(s) == 6:
                return f"{s[:3]}/{s[3:]}"
            return f"USD/{s}"  # Fallback for JPY=X
        if broker_symbol.endswith("-USD"):
            return broker_symbol.replace("-USD", "/USDT")
        return broker_symbol

    def _determine_asset_class(self, symbol: str) -> AssetClass:
        if "=" in symbol:
            return AssetClass.FOREX
        if "-" in symbol:
            return AssetClass.CRYPTO
        return AssetClass.EQUITY

    def _map_timeframe(self, tf: Timeframe) -> str:
        """Map generic timeframe to yfinance interval."""
        mapping = {
            Timeframe.M1: "1m",
            Timeframe.M5: "5m",
            Timeframe.M15: "15m",
            Timeframe.M30: "30m",
            Timeframe.H1: "1h",
            Timeframe.H4: "1h",
            Timeframe.D1: "1d",
            Timeframe.W1: "1wk",
            Timeframe.MN1: "1mo",
        }
        return mapping.get(tf, "1d")

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Candle]:
        """Fetch historical data from Yahoo Finance."""
        yf_symbol = self.normalize_symbol(symbol)
        interval = self._map_timeframe(timeframe)
        asset_class = self._determine_asset_class(yf_symbol)
        std_symbol = self.standardize_symbol(yf_symbol)

        logger.info(f"Fetching yfinance data for {std_symbol} ({interval if timeframe != Timeframe.H4 else '4h resampled'})")
        
        # yfinance can be blocking, so we run it in a thread
        def _fetch():
            ticker = yf.Ticker(yf_symbol)
            # Calculate start date based on limit and timeframe if since is not provided
            if since is None:
                period = "1y"
                if timeframe in [Timeframe.M1, Timeframe.M5]: period = "7d"
                elif timeframe in [Timeframe.M15, Timeframe.M30, Timeframe.H1]: period = "60d"
                elif timeframe == Timeframe.H4: period = "730d"
                
                df = ticker.history(period=period, interval=interval)
                if df.empty and period == "60d":
                    df = ticker.history(period="1mo", interval=interval)
            else:
                df = ticker.history(start=since, interval=interval)
            
            if df.empty:
                # One retry in case Yahoo Finance flickers
                import time
                time.sleep(1)
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period="1mo" if timeframe == Timeframe.H1 else "1y", interval=interval)
            
            if not df.empty and timeframe == Timeframe.H4:
                agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                if 'Volume' in df.columns:
                    agg_dict['Volume'] = 'sum'
                df = df.resample('4h').agg(agg_dict).dropna()
            return df

        try:
            df = await asyncio.to_thread(_fetch)
            if df.empty:
                logger.warning(f"No data returned for {yf_symbol}")
                return []

            candles = []
            # Keep only the last 'limit' rows
            df = df.tail(limit)

            for timestamp, row in df.iterrows():
                # Make naive for consistency if needed, but standardizing to UTC is better
                dt = timestamp.to_pydatetime()
                if dt.tzinfo is not None:
                    dt = dt.astimezone(tz=None).replace(tzinfo=None)

                candles.append(Candle(
                    symbol=std_symbol,
                    asset_class=asset_class,
                    timeframe=timeframe,
                    timestamp=dt,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0))
                ))
            return candles

        except Exception as e:
            logger.error(f"Error fetching yfinance data for {yf_symbol}: {e}")
            return []

    async def stream_ticks(self, symbols: List[str]) -> AsyncGenerator[Tick, None]:
        """
        Simulate tick stream by polling yfinance.
        Warning: yfinance is delayed and rate-limited. This is just for demo.
        """
        logger.info(f"Starting simulated tick stream for {symbols}")
        
        while self.is_connected:
            for symbol in symbols:
                yf_symbol = self.normalize_symbol(symbol)
                std_symbol = self.standardize_symbol(yf_symbol)
                asset_class = self._determine_asset_class(yf_symbol)
                
                def _fetch_price():
                    ticker = yf.Ticker(yf_symbol)
                    todays_data = ticker.history(period='1d')
                    if not todays_data.empty:
                        return float(todays_data['Close'].iloc[-1])
                    return None
                
                try:
                    price = await asyncio.to_thread(_fetch_price)
                    if price is not None:
                        # Simulate a tiny spread (1 pip)
                        spread = price * 0.0001
                        tick = Tick(
                            symbol=std_symbol,
                            asset_class=asset_class,
                            timestamp=datetime.utcnow(),
                            bid=price - (spread/2),
                            ask=price + (spread/2)
                        )
                        yield tick
                except Exception as e:
                    logger.debug(f"Simulated tick fetch error for {symbol}: {e}")
            
            # Sleep to avoid rate limiting
            await asyncio.sleep(5)

    async def stream_candles(self, symbol: str, timeframe: Timeframe) -> AsyncGenerator[Candle, None]:
        """Simulate candle stream by fetching recent history periodically."""
        # Polling frequency based on timeframe (approx)
        sleep_time = 60 if timeframe in [Timeframe.M1, Timeframe.M5] else 300
        
        last_timestamp = None
        
        while self.is_connected:
            candles = await self.get_historical_candles(symbol, timeframe, limit=2)
            if candles:
                latest = candles[-1]
                # Only yield if it's a new candle (or if we want to yield closed only, we look at -2)
                if last_timestamp is None or latest.timestamp > last_timestamp:
                    last_timestamp = latest.timestamp
                    yield latest
                    
            await asyncio.sleep(sleep_time)
