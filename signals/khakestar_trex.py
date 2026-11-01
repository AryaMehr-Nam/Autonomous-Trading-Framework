"""
Saeid Khakestar Trigger Price Action (TPA / TRex) Engine
=========================================================
Technical Implementation of Saeid Khakestar's Price Action:
1. ATR Movement Steps (گام های حرکتی با ATR26)
2. Mathematical Pivot Identification (پیوت شناسی ریاضی با پوشش کلوز)
3. FTR (Failure To Return) & FL (Flag Limit) Trading Nodes (گره های معاملاتی)
4. TPA Trigger Entries (ورود بر اساس برخورد به گره FTR با تاییدیه کندلی)
5. Multi-Timeframe Fractal Mapping (H1 Structure + M15 Trigger)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

@dataclass
class Pivot:
    index: int
    price: float
    type: str  # 'HIGH' or 'LOW'
    is_major: bool
    step_size: float

@dataclass
class FTRNode:
    level_broken: float
    node_high: float
    node_low: float
    direction: str  # 'BULL' (Demand Node) or 'BEAR' (Supply Node)
    created_idx: int
    tested: bool = False

class KhakestarTrexEngine:
    def __init__(self, atr_period: int = 26):
        self.atr_period = atr_period

    @staticmethod
    def calculate_atr_steps(df: pd.DataFrame, period: int = 26) -> np.ndarray:
        """
        Calculates Saeid Khakestar's ATR Movement Step line.
        """
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd.Series(tr).rolling(period).mean().bfill().values
        return atr

    @classmethod
    def identify_pivots(cls, df: pd.DataFrame, atr: np.ndarray) -> List[Pivot]:
        """
        Mathematical Pivot Identification:
        Requires price movement of at least 1 ATR step, followed by reversal
        where the opposing candle body covers the previous candle's close.
        """
        pivots = []
        n = len(df)
        c = df['close'].values
        h = df['high'].values
        l = df['low'].values

        for i in range(5, n - 2):
            step = atr[i]
            # High Pivot
            if h[i] == max(h[i-3:i+4]):
                # Prior move up >= 1 step
                if h[i] - min(l[i-5:i]) >= step * 0.8:
                    # Confirmation: subsequent candle close covers previous
                    if c[i+1] < c[i] or c[i+2] < l[i]:
                        is_maj = (h[i] - min(l[i-15:i])) >= step * 2.0 if i >= 15 else False
                        pivots.append(Pivot(index=i, price=h[i], type='HIGH', is_major=is_maj, step_size=step))

            # Low Pivot
            elif l[i] == min(l[i-3:i+4]):
                # Prior move down >= 1 step
                if max(h[i-5:i]) - l[i] >= step * 0.8:
                    # Confirmation
                    if c[i+1] > c[i] or c[i+2] > h[i]:
                        is_maj = (max(h[i-15:i]) - l[i]) >= step * 2.0 if i >= 15 else False
                        pivots.append(Pivot(index=i, price=l[i], type='LOW', is_major=is_maj, step_size=step))

        return pivots

    @classmethod
    def identify_ftr_nodes(cls, df: pd.DataFrame, pivots: List[Pivot], atr: np.ndarray) -> List[FTRNode]:
        """
        Identifies FTR (Failure To Return) Trading Nodes formed upon level breakout.
        """
        nodes = []
        c = df['close'].values
        h = df['high'].values
        l = df['low'].values
        n = len(df)

        for p in pivots:
            level = p.price
            p_idx = p.index

            # Look forward for breakout
            for j in range(p_idx + 1, min(p_idx + 60, n - 5)):
                # Bullish Breakout above High Pivot
                if p.type == 'HIGH' and c[j] > level and c[j-1] <= level:
                    # Search for base/pause within 1-3 bars of breakout
                    node_h = max(h[j:j+3])
                    node_l = min(l[j:j+3])
                    nodes.append(FTRNode(
                        level_broken=level,
                        node_high=node_h,
                        node_low=node_l,
                        direction='BULL',
                        created_idx=j
                    ))
                    break

                # Bearish Breakdown below Low Pivot
                elif p.type == 'LOW' and c[j] < level and c[j-1] >= level:
                    node_h = max(h[j:j+3])
                    node_l = min(l[j:j+3])
                    nodes.append(FTRNode(
                        level_broken=level,
                        node_high=node_h,
                        node_low=node_l,
                        direction='BEAR',
                        created_idx=j
                    ))
                    break

        return nodes