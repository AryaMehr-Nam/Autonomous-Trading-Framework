"""
Pour Samadi Price Action & Market Mapping Toolkit
=================================================
Technical Implementation of Mohammad Pour Samadi's Price Action Indicators:
1. Weekly & Daily Map (PDH, PDL, PWH, PWL, Daily Open, Round Psychological Levels)
2. ADR (Average Daily Range) & Exhaustion Engine
3. Session Windows Pro (Tokyo, London, New York Killzones & Overlaps)
4. Candle Pressure & Close Location Value (CLV)
5. Pro BTB (Back To Breakeven / Level Retest) Detector
6. SP2L (Spike - 2 Leg Flag Pullback) Detector
7. MicroMap Compression Engine (Tight-range absorption before explosive breakouts)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class KeyLevelsMap:
    pdh: float  # Previous Day High
    pdl: float  # Previous Day Low
    daily_open: float
    pwh: Optional[float] = None
    pwl: Optional[float] = None
    weekly_open: Optional[float] = None
    round_levels: List[float] = None
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    dist_to_support: float = 0.0
    dist_to_resistance: float = 0.0

@dataclass
class ADRMetrics:
    adr_value: float
    today_range: float
    adr_used_pct: float
    adr_high_target: float
    adr_low_target: float
    is_exhausted: bool  # True if > 85% of ADR consumed
    room_to_high: float
    room_to_low: float

class PourSamadiToolkit:
    """
    Comprehensive implementation of Pour Samadi's Price Action Indicators and Filters.
    """

    @staticmethod
    def calculate_candle_pressure(open_: float, high: float, low: float, close: float) -> float:
        """
        Close Location Value (CLV) / Candle Pressure:
        Range: 0.0 (closed at dead low) to 1.0 (closed at dead high).
        Strong Bullish: >= 0.70 (close in upper 30%)
        Strong Bearish: <= 0.30 (close in lower 30%)
        """
        bar_range = max(high - low, 1e-6)
        clv = (close - low) / bar_range
        return float(np.clip(clv, 0.0, 1.0))

    @staticmethod
    def detect_micromap_compression(highs: np.ndarray, lows: np.ndarray, atr: float, ratio: float = 0.85) -> bool:
        """
        MicroMap Compression:
        Checks if the previous 2 consecutive candles have compressed ranges (narrow range absorption).
        """
        if len(highs) < 3 or atr <= 0:
            return False
        prev1_range = highs[-2] - lows[-2]
        prev2_range = highs[-3] - lows[-3]
        return (prev1_range < atr * ratio) and (prev2_range < atr * (ratio + 0.05))

    @staticmethod
    def calculate_round_levels(price: float, symbol: str, num_levels: int = 4) -> List[float]:
        """
        Calculates institutional round psychological numbers (.000, .500, .200, .800).
        """
        levels = []
        if "JPY" in symbol:
            base = round(price)
            step = 0.50
            for i in range(-num_levels, num_levels + 1):
                levels.append(round(base + i * step, 3))
        elif "XAU" in symbol or "GOLD" in symbol:
            base = round(price / 5.0) * 5.0
            step = 5.0
            for i in range(-num_levels, num_levels + 1):
                levels.append(round(base + i * step, 2))
        elif "30" in symbol or "TEC" in symbol or "DAX" in symbol:
            base = round(price / 50.0) * 50.0
            step = 50.0
            for i in range(-num_levels, num_levels + 1):
                levels.append(round(base + i * step, 1))
        else: # Standard FX (EURUSD, etc.)
            base = round(price * 100) / 100.0
            step = 0.0050
            for i in range(-num_levels, num_levels + 1):
                levels.append(round(base + i * step, 5))
        return sorted(list(set(levels)))

    @classmethod
    def calculate_daily_map(cls, df_daily: pd.DataFrame, current_price: float, symbol: str) -> Optional[KeyLevelsMap]:
        """
        Extracts Previous Day High (PDH), Previous Day Low (PDL), Daily Open (DO)
        and computes nearest round numbers and key S/R boundaries.
        """
        if df_daily is None or len(df_daily) < 2:
            return None

        prev_day = df_daily.iloc[-2]
        today = df_daily.iloc[-1]

        pdh = float(prev_day['high'])
        pdl = float(prev_day['low'])
        daily_open = float(today['open'])

        rounds = cls.calculate_round_levels(current_price, symbol)

        resistances = [l for l in [pdh] + rounds if l > current_price]
        nearest_res = min(resistances) if resistances else current_price * 1.05

        supports = [l for l in [pdl] + rounds if l < current_price]
        nearest_sup = max(supports) if supports else current_price * 0.95

        return KeyLevelsMap(
            pdh=pdh,
            pdl=pdl,
            daily_open=daily_open,
            round_levels=rounds,
            nearest_support=nearest_sup,
            nearest_resistance=nearest_res,
            dist_to_support=current_price - nearest_sup,
            dist_to_resistance=nearest_res - current_price
        )

    @staticmethod
    def calculate_adr(df_daily: pd.DataFrame, today_high: float, today_low: float, today_open: float, current_price: float, period: int = 14) -> Optional[ADRMetrics]:
        """
        Average Daily Range (ADR) & Exhaustion Filter:
        Computes the average movement over the last 14 days and measures how much of the range is already spent.
        """
        if df_daily is None or len(df_daily) < period + 1:
            return None

        completed_days = df_daily.iloc[-(period+1):-1]
        ranges = completed_days['high'] - completed_days['low']
        adr = float(ranges.mean())

        today_range = max(today_high - today_low, 1e-6)
        adr_used_pct = (today_range / adr) * 100.0

        adr_high = today_open + adr
        adr_low = today_open - adr

        is_exhausted = adr_used_pct >= 85.0

        room_to_high = max(0.0, adr_high - current_price)
        room_to_low = max(0.0, current_price - adr_low)

        return ADRMetrics(
            adr_value=adr,
            today_range=today_range,
            adr_used_pct=adr_used_pct,
            adr_high_target=adr_high,
            adr_low_target=adr_low,
            is_exhausted=is_exhausted,
            room_to_high=room_to_high,
            room_to_low=room_to_low
        )

    @staticmethod
    def check_session_window(dt: datetime) -> Dict[str, bool]:
        """
        Pour Samadi Session Windows Pro:
        Identifies high-liquidity Killzones (London Open, NY Open, Asian/Tokyo Open)
        and flags the toxic Rollover spread widening window.
        """
        h = dt.hour
        m = dt.minute
        minute_of_day = h * 60 + m

        in_london_kz = (7 * 60 <= minute_of_day < 10 * 60)
        in_london = (7 * 60 <= minute_of_day < 16 * 60)
        in_ny_kz = (12 * 60 + 30 <= minute_of_day < 15 * 60 + 30)
        in_ny = (12 * 60 <= minute_of_day < 21 * 60)
        in_overlap = in_london and in_ny
        in_tokyo = (0 <= minute_of_day < 5 * 60)
        in_rollover = (minute_of_day >= 21 * 60 + 45) or (minute_of_day <= 1 * 60 + 15)
        is_prime_execution = (in_london_kz or in_ny_kz or in_overlap or in_tokyo) and not in_rollover

        return {
            "in_london_kz": in_london_kz,
            "in_ny_kz": in_ny_kz,
            "in_overlap": in_overlap,
            "in_tokyo": in_tokyo,
            "in_rollover": in_rollover,
            "is_prime_execution": is_prime_execution
        }