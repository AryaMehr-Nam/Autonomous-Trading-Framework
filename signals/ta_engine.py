"""
TradeAgent Technical Analysis Engine.

Calculates technical indicators for given OHLCV DataFrames using pandas-ta.
"""

import pandas as pd
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from config.settings import trading_params
from core.logger import get_logger
from signals.tradingview_indicators import add_tradingview_indicators

logger = get_logger("ta_engine")


class TAEngine:
    """Computes technical indicators based on configuration."""
    
    def __init__(self):
        self.config = trading_params.technical
        
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all configured indicators to the DataFrame.
        Expected input df format: index = datetime, columns = [open, high, low, close, volume]
        """
        if not HAS_PANDAS_TA:
            logger.error("pandas-ta is not installed.")
            return df
            
        if df.empty or len(df) < 50:
            logger.warning("Not enough data to compute indicators.")
            return df
            
        df = df.copy()
        # lowercase columns for pandas-ta
        df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
        
        # 1. Trend Indicators
        df = self._add_trend(df)
        
        # 2. Momentum Indicators
        df = self._add_momentum(df)
        
        # 3. Volatility Indicators
        df = self._add_volatility(df)
        
        # 4. Volume Indicators
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df = self._add_volume(df)
            
        # 5. TradingView Elite Indicators (WaveTrend, Nadaraya-Watson, UT Bot, Lorentzian)
        df = add_tradingview_indicators(df)
            
        # Clean up NaNs from indicator lookback periods
        df.dropna(inplace=True)
        return df
        
    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        trend_cfg = self.config.get("indicators", {}).get("trend", [])
        for cfg in trend_cfg:
            name = cfg.get("name")
            if name == "ema":
                for p in cfg.get("periods", [9, 21, 50, 200]):
                    df[f"EMA_{p}"] = ta.ema(df['close'], length=p)
            elif name == "sma":
                for p in cfg.get("periods", [50, 200]):
                    df[f"SMA_{p}"] = ta.sma(df['close'], length=p)
            elif name == "supertrend":
                p = cfg.get("period", 10)
                m = cfg.get("multiplier", 3.0)
                st = ta.supertrend(df['high'], df['low'], df['close'], length=p, multiplier=m)
                if st is not None:
                    # supertrend returns multiple columns (trend direction, upper, lower, etc.)
                    # We just take the direction column (SUPERTd_10_3.0)
                    dir_col = [c for c in st.columns if "SUPERTd" in c]
                    if dir_col:
                        df["SUPERTREND_DIR"] = st[dir_col[0]]
            elif name == "adx":
                p = cfg.get("period", 14)
                adx = ta.adx(df['high'], df['low'], df['close'], length=p)
                if adx is not None:
                    adx_col = [c for c in adx.columns if "ADX" in c]
                    if adx_col:
                        df["ADX"] = adx[adx_col[0]]
        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        mom_cfg = self.config.get("indicators", {}).get("momentum", [])
        for cfg in mom_cfg:
            name = cfg.get("name")
            if name == "rsi":
                p = cfg.get("period", 14)
                df["RSI"] = ta.rsi(df['close'], length=p)
            elif name == "macd":
                fast = cfg.get("fast", 12)
                slow = cfg.get("slow", 26)
                sig = cfg.get("signal", 9)
                macd = ta.macd(df['close'], fast=fast, slow=slow, signal=sig)
                if macd is not None:
                    hist_col = [c for c in macd.columns if "MACDh" in c]
                    line_col = [c for c in macd.columns if "MACD_" in c]
                    if hist_col: df["MACD_HIST"] = macd[hist_col[0]]
                    if line_col: df["MACD_LINE"] = macd[line_col[0]]
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        vol_cfg = self.config.get("indicators", {}).get("volatility", [])
        for cfg in vol_cfg:
            name = cfg.get("name")
            if name == "atr":
                p = cfg.get("period", 14)
                df["ATR"] = ta.atr(df['high'], df['low'], df['close'], length=p)
            elif name == "bollinger_bands":
                p = cfg.get("period", 20)
                std = cfg.get("std_dev", 2.0)
                bb = ta.bbands(df['close'], length=p, std=std)
                if bb is not None:
                    pct_col = [c for c in bb.columns if "BBP_" in c]
                    if pct_col: df["BB_PCT"] = bb[pct_col[0]]
        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        # Some forex providers don't give reliable volume, handled safely here
        vol_cfg = self.config.get("indicators", {}).get("volume", [])
        
        # Add Volume Moving Average (VMA) and RVOL (Relative Volume) for Order Flow / VPA analysis
        df["VMA_20"] = ta.sma(df['volume'], length=20)
        df["RVOL"] = df['volume'] / df["VMA_20"].replace(0, 1)
        
        for cfg in vol_cfg:
            name = cfg.get("name")
            if name == "obv":
                df["OBV"] = ta.obv(df['close'], df['volume'])
            elif name == "mfi":
                p = cfg.get("period", 14)
                df["MFI"] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=p)
        return df
