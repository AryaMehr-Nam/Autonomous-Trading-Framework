"""
TradeAgent Storage Manager.

Caches historical data using DuckDB for blazing fast vectorized queries.
"""

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

from config.settings import settings, DATA_DIR
from core.logger import get_logger
from core.models import AssetClass, Candle, Timeframe

logger = get_logger("storage")


class StorageManager:
    """Manages historical data persistence."""
    
    def __init__(self):
        self.db_path = settings.duckdb_path
        self.conn = None

    def connect(self):
        if not HAS_DUCKDB:
            logger.warning("duckdb not installed. Caching disabled.")
            return False
            
        try:
            self.conn = duckdb.connect(self.db_path)
            self._init_schema()
            logger.info("Storage Manager connected to DuckDB.")
            return True
        except Exception as e:
            logger.error(f"DuckDB connection failed: {e}")
            return False

    def _init_schema(self):
        """Create tables if they don't exist."""
        if not self.conn: return
        query = """
        CREATE TABLE IF NOT EXISTS candles (
            symbol VARCHAR,
            asset_class VARCHAR,
            timeframe VARCHAR,
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
        """
        self.conn.execute(query)

    def close(self):
        if self.conn:
            self.conn.close()

    def store_candles(self, candles: List[Candle]):
        """Bulk insert candles into DB."""
        if not self.conn or not candles: return
        
        try:
            df = pd.DataFrame([
                {
                    "symbol": c.symbol,
                    "asset_class": c.asset_class.value,
                    "timeframe": c.timeframe.value,
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                }
                for c in candles
            ])
            # Use DuckDB appender for fast inserts, ignoring conflicts
            self.conn.execute("INSERT OR IGNORE INTO candles SELECT * FROM df")
            logger.debug(f"Stored {len(candles)} candles in DB.")
        except Exception as e:
            logger.error(f"Failed to store candles: {e}")

    def get_candles(self, symbol: str, timeframe: Timeframe, since: Optional[datetime] = None, limit: int = 1000) -> List[Candle]:
        """Retrieve candles from DB."""
        if not self.conn: return []
        
        try:
            query = f"""
            SELECT * FROM candles 
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe.value}'
            """
            if since:
                query += f" AND timestamp >= '{since.isoformat()}'"
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            df = self.conn.execute(query).df()
            if df.empty: return []
            
            # Reverse to chronological
            df = df.iloc[::-1]
            
            candles = []
            for _, row in df.iterrows():
                candles.append(Candle(
                    symbol=row["symbol"],
                    asset_class=AssetClass(row["asset_class"]),
                    timeframe=Timeframe(row["timeframe"]),
                    timestamp=row["timestamp"].to_pydatetime(),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                ))
            return candles
        except Exception as e:
            logger.error(f"Failed to fetch candles from DB: {e}")
            return []
