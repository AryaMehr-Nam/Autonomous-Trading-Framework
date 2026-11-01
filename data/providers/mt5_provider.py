"""
TradeAgent MT5 Data Provider.

Integration with MetaTrader 5 (Windows only).
"""

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional
import sys

try:
    if sys.platform == "win32":
        import MetaTrader5 as mt5
        HAS_MT5 = True
    else:
        HAS_MT5 = False
except ImportError:
    HAS_MT5 = False

from config.settings import settings
from core.logger import get_logger
from core.models import AssetClass, Candle, Tick, Timeframe
from data.base_provider import BaseDataProvider

logger = get_logger("mt5_provider")


class MT5DataProvider(BaseDataProvider):
    
    def __init__(self):
        super().__init__(name="mt5")
        self.supported_assets = [AssetClass.FOREX, AssetClass.EQUITY]

    async def connect(self) -> bool:
        if not HAS_MT5:
            logger.warning("MetaTrader5 package not installed or not on Windows.")
            return False
            
        if not settings.has_mt5:
            logger.warning("MT5 credentials not set in environment.")
            return False
            
        try:
            if settings.mt5_login and settings.mt5_password and settings.mt5_server:
                init_res = mt5.initialize(
                    path=settings.mt5_path or r"C:\Program Files\MetaTrader 5\terminal64.exe",
                    login=int(settings.mt5_login),
                    server=settings.mt5_server,
                    password=settings.mt5_password
                )
            else:
                init_res = mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe") or mt5.initialize()
                
            if not init_res:
                logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
                return False
                
            self.is_connected = True
            logger.info("MT5 Data Provider connected.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            return False

    async def disconnect(self) -> None:
        if HAS_MT5 and self.is_connected:
            mt5.shutdown()
        self.is_connected = False
        logger.info("MT5 Data Provider disconnected.")

    async def get_symbols(self) -> List[str]:
        if not self.is_connected: return []
        syms = mt5.symbols_get()
        return [s.name for s in syms] if syms else []

    def normalize_symbol(self, symbol: str) -> str:
        # Standard 'EUR/USD' -> 'EURUSD'
        return symbol.replace("/", "")

    def standardize_symbol(self, broker_symbol: str) -> str:
        # 'EURUSD' -> 'EUR/USD'
        if len(broker_symbol) == 6:
            return f"{broker_symbol[:3]}/{broker_symbol[3:]}"
        return broker_symbol

    def _map_timeframe(self, tf: Timeframe) -> int:
        if not HAS_MT5: return 0
        mapping = {
            Timeframe.M1: mt5.TIMEFRAME_M1,
            Timeframe.M5: mt5.TIMEFRAME_M5,
            Timeframe.M15: mt5.TIMEFRAME_M15,
            Timeframe.M30: mt5.TIMEFRAME_M30,
            Timeframe.H1: mt5.TIMEFRAME_H1,
            Timeframe.H4: mt5.TIMEFRAME_H4,
            Timeframe.D1: mt5.TIMEFRAME_D1,
            Timeframe.W1: mt5.TIMEFRAME_W1,
            Timeframe.MN1: mt5.TIMEFRAME_MN1,
        }
        return mapping.get(tf, mt5.TIMEFRAME_H1)

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Candle]:
        if not self.is_connected: return []

        mt5_symbol = self.normalize_symbol(symbol)
        mt5_tf = self._map_timeframe(timeframe)
        
        def _fetch():
            if since:
                rates = mt5.copy_rates_from(mt5_symbol, mt5_tf, since, limit)
            else:
                rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, limit)
            return rates
            
        try:
            rates = await asyncio.to_thread(_fetch)
            if rates is None or len(rates) == 0:
                return []
                
            candles = []
            for r in rates:
                # r: time, open, high, low, close, tick_volume, spread, real_volume
                dt = datetime.fromtimestamp(r['time'])
                candles.append(Candle(
                    symbol=symbol,
                    asset_class=AssetClass.FOREX,
                    timeframe=timeframe,
                    timestamp=dt,
                    open=float(r['open']),
                    high=float(r['high']),
                    low=float(r['low']),
                    close=float(r['close']),
                    volume=float(r['tick_volume'])
                ))
            return candles
        except Exception as e:
            logger.error(f"MT5 error fetching candles for {symbol}: {e}")
            return []

    async def stream_ticks(self, symbols: List[str]) -> AsyncGenerator[Tick, None]:
        if not self.is_connected: return
            
        mt5_symbols = [self.normalize_symbol(s) for s in symbols]
        # Enable symbols in market watch
        for s in mt5_symbols:
            mt5.symbol_select(s, True)
            
        while self.is_connected:
            try:
                for idx, sym in enumerate(symbols):
                    mt5_sym = mt5_symbols[idx]
                    tick_info = mt5.symbol_info_tick(mt5_sym)
                    if tick_info:
                        yield Tick(
                            symbol=sym,
                            asset_class=AssetClass.FOREX,
                            timestamp=datetime.fromtimestamp(tick_info.time),
                            bid=tick_info.bid,
                            ask=tick_info.ask
                        )
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"MT5 Tick stream error: {e}")
                await asyncio.sleep(5.0)

    async def stream_candles(self, symbol: str, timeframe: Timeframe) -> AsyncGenerator[Candle, None]:
        yield Candle("", AssetClass.FOREX, timeframe, datetime.now(), 0.0, 0.0, 0.0, 0.0, 0.0)
