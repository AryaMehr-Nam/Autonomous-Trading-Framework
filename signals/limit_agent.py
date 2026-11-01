"""
Dedicated Limit Order Agent.
Scans for Support and Resistance zones using local minimums and maximums,
and emits LIMIT orders slightly ahead of the zones to catch wicks.
"""
import pandas as pd
from typing import Optional
from core.models import AssetClass, Signal, SignalType, Timeframe, OrderType
from core.logger import get_logger

logger = get_logger("limit_agent")

class LimitOrderAgent:
    def __init__(self, lookback_period: int = 50, buffer_pct: float = 0.0005):
        self.lookback_period = lookback_period
        self.buffer_pct = buffer_pct # 0.05% buffer to front-run the level
        
    def generate_signal(self, symbol: str, df: pd.DataFrame, timeframe: Timeframe, asset_class: AssetClass) -> list:
        if df is None or df.empty or len(df) < self.lookback_period:
            return []
            
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # Calculate recent Support and Resistance
        recent_df = df.iloc[-self.lookback_period:-1] # exclude current incomplete candle
        
        support_level = recent_df['low'].min()
        resistance_level = recent_df['high'].max()
        
        # Evaluate which level we are closer to. 
        # Actually, we will place both if they are valid.
        
        signals = []
        
        # Buy Limit at Support
        buy_limit_price = support_level * (1 + self.buffer_pct)
        if (current_price - buy_limit_price) / current_price > 0.0002: 
            signals.append(Signal(
                symbol=symbol,
                asset_class=asset_class,
                signal_type=SignalType.BUY,
                confidence=0.8,
                timeframe=timeframe,
                order_type=OrderType.LIMIT,
                limit_price=buy_limit_price,
                entry_price=current_price,
                technical_score=1.0,
                source="limit_agent_support",
                reasoning=f"Buy Limit at Support {support_level:.2f}",
                indicators={},
                win_rate=0.6,
                risk_reward=1.5
            ))
            
        # Sell Limit at Resistance
        sell_limit_price = resistance_level * (1 - self.buffer_pct)
        if (sell_limit_price - current_price) / current_price > 0.0002:
            signals.append(Signal(
                symbol=symbol,
                asset_class=asset_class,
                signal_type=SignalType.SELL,
                confidence=0.8,
                timeframe=timeframe,
                order_type=OrderType.LIMIT,
                limit_price=sell_limit_price,
                entry_price=current_price,
                technical_score=-1.0,
                source="limit_agent_resistance",
                reasoning=f"Sell Limit at Resistance {resistance_level:.2f}",
                indicators={},
                win_rate=0.6,
                risk_reward=1.5
            ))
                
        return signals
