"""
Smart Money Concepts (SMC) - Institutional Order Block Detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from core.models import OrderType

def detect_smc_signal(df: pd.DataFrame, params: Optional[dict] = None) -> Tuple[int, float, OrderType, float]:
    """
    Detects Order Blocks and Fair Value Gaps to generate Institutional SMC signals.
    Returns (direction (-1, 0, 1), confidence, order_type, limit_price).
    """
    if df is None or len(df) < 20:
        return 0, 0.0, OrderType.MARKET, 0.0
        
    # Parameters that can be optimized
    displacement_threshold = params.get('displacement_pct', 0.15) if params else 0.15
    fvg_threshold = params.get('fvg_pct', 0.05) if params else 0.05
    lookback = params.get('ob_lookback', 10) if params else 10
    
    score = 0.0
    direction = 0
    limit_price = 0.0
    max_score = 3.0  # Increased max score to account for Order Flow points
    
    # 1. Identify displacement (strong impulsive moves) in the recent bars
    latest = df.iloc[-1]
    
    # Look for a recent strong move
    recent_bars = df.iloc[-lookback:]
    
    bullish_displacement = False
    bearish_displacement = False
    fvg_found = False
    order_flow_confirmed = False
    order_flow_weak = False
    
    for i in range(2, len(recent_bars)):
        c1 = recent_bars.iloc[i-2]
        c2 = recent_bars.iloc[i-1]
        c3 = recent_bars.iloc[i]
        
        # Bullish Displacement + FVG
        move_up = (c3['close'] - c1['open']) / c1['open'] * 100
        if move_up > displacement_threshold:
            bullish_displacement = True
            # FVG check: candle 1 high < candle 3 low
            if c1['high'] < c3['low']:
                gap = (c3['low'] - c1['high']) / c1['high'] * 100
                if gap > fvg_threshold:
                    fvg_found = True
                    direction = 1
                    # The limit price is the top of the bullish order block (candle 1 high)
                    limit_price = c1['high']
                    
                    # Order Flow check (VPA RVOL) on the displacement candle (c2)
                    if 'RVOL' in c2 and not pd.isna(c2['RVOL']):
                        if c2['RVOL'] > 1.5:
                            order_flow_confirmed = True
                        elif c2['RVOL'] < 0.8:
                            order_flow_weak = True
                    break
                    
        # Bearish Displacement + FVG
        move_down = (c1['open'] - c3['close']) / c1['open'] * 100
        if move_down > displacement_threshold:
            bearish_displacement = True
            # FVG check: candle 1 low > candle 3 high
            if c1['low'] > c3['high']:
                gap = (c1['low'] - c3['high']) / c1['low'] * 100
                if gap > fvg_threshold:
                    fvg_found = True
                    direction = -1
                    # The limit price is the bottom of the bearish order block (candle 1 low)
                    limit_price = c1['low']
                    
                    # Order Flow check (VPA RVOL) on the displacement candle (c2)
                    if 'RVOL' in c2 and not pd.isna(c2['RVOL']):
                        if c2['RVOL'] > 1.5:
                            order_flow_confirmed = True
                        elif c2['RVOL'] < 0.8:
                            order_flow_weak = True
                    break

    if direction != 0:
        score += 1.0
        if fvg_found:
            score += 1.0
            
        # Adjust score based on Volume Price Analysis
        if order_flow_confirmed:
            score += 1.0  # Big confidence boost for institutional climax volume
        elif order_flow_weak:
            score -= 1.0  # Heavy penalty for retail fakeout (low volume displacement)
            
    confidence = max(0.0, min(1.0, score / max_score))
    
    # If we found an OB with a limit price, but price is already far away, we place a limit order
    # If price is already AT the limit price, we place a market order
    ot = OrderType.LIMIT if limit_price > 0.0 else OrderType.MARKET
    
    # Optional: ensure price hasn't already broken the OB limit
    if direction == 1 and latest['close'] < limit_price * 0.999:
        # OB violated
        direction = 0
        confidence = 0.0
    elif direction == -1 and latest['close'] > limit_price * 1.001:
        # OB violated
        direction = 0
        confidence = 0.0
        
    return direction, confidence, ot, limit_price
