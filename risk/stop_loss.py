"""
TradeAgent Stop Loss & Take Profit Management.

Calculates optimal entry sizing, dynamic stop losses (ATR based), and 
take profits for new signals.
"""

from typing import Dict

import pandas as pd

from config.settings import trading_params
from core.models import OrderSide, Signal


class StopLossManager:
    """Manages SL/TP logic and calculations."""
    
    def __init__(self):
        self.risk_cfg = trading_params.risk
        self.sl_cfg = self.risk_cfg.get("stop_loss", {})
        self.tp_cfg = self.risk_cfg.get("take_profit", {})

    def apply_risk_parameters(self, signal: Signal, df: pd.DataFrame) -> Signal:
        """
        Calculates and applies Stop Loss and Take Profit to the signal.
        """
        if df is None or df.empty or signal.entry_price == 0:
            return signal

        latest = df.iloc[-1]
        
        # Determine Pip Value (rough approximation for calculation)
        # Forex typically uses 0.0001, JPY uses 0.01. Crypto uses 1.0.
        pip = 0.0001
        asset_cls = signal.asset_class.value if hasattr(signal.asset_class, 'value') else str(signal.asset_class)
        if "JPY" in signal.symbol:
            pip = 0.01
        elif asset_cls == "crypto":
            pip = 1.0
        elif asset_cls == "equity":
            pip = 0.01

        sl_dist = self._calculate_sl_distance(latest, pip, signal.timeframe)
        tp_dist = self._calculate_tp_distance(latest, sl_dist, pip, signal.timeframe)

        sig_type = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)

        # For Limit Orders, SL/TP must be calculated from the Limit Price, not the current spot price
        from core.models import OrderType
        base_price = signal.limit_price if signal.order_type == OrderType.LIMIT and signal.limit_price > 0 else signal.entry_price

        if sig_type == "buy":
            signal.stop_loss = base_price - sl_dist
            signal.take_profit = base_price + tp_dist
        else: # sell
            signal.stop_loss = base_price + sl_dist
            signal.take_profit = base_price - tp_dist

        # Calculate optional partial Take Profit (TP1)
        partial_pct = self.tp_cfg.get("partial_close_pct", 0)
        if partial_pct > 0:
            # TP1 is halfway to TP
            tp1_dist = tp_dist * 0.5
            if sig_type == "buy":
                signal.take_profit_2 = base_price + tp1_dist
            else:
                signal.take_profit_2 = base_price - tp1_dist

        # If it is a limit order, update the entry_price to the limit_price so R/R calculations are accurate
        if signal.order_type == OrderType.LIMIT and signal.limit_price > 0:
            signal.entry_price = signal.limit_price

        return signal

    def _calculate_sl_distance(self, row: pd.Series, pip_value: float, timeframe=None) -> float:
        """Calculate Stop Loss distance from entry, dynamically scaled by timeframe."""
        sl_type = self.sl_cfg.get("type", "atr")
        
        # Scale multiplier based on timeframe horizon
        tf_scale = 1.0
        if timeframe:
            tf_val = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            if tf_val in ["1m", "5m", "15m", "30m"]:
                tf_scale = 0.75  # Tighter stops for short term
            elif tf_val in ["1h", "4h", "1d"]:
                tf_scale = 1.5   # Wider stops for long term
        
        if sl_type == "atr" and "ATR" in row and row["ATR"] > 0:
            multiplier = self.sl_cfg.get("atr_multiplier", 2.0) * tf_scale
            return row["ATR"] * multiplier
            
        elif sl_type == "fixed":
            pips = self.sl_cfg.get("fixed_pips", 30) * tf_scale
            return pips * pip_value
            
        # Fallback
        return 30 * pip_value * tf_scale

    def _calculate_tp_distance(self, row: pd.Series, sl_dist: float, pip_value: float, timeframe=None) -> float:
        """Calculate Take Profit distance from entry."""
        tp_type = self.tp_cfg.get("type", "rr_ratio")
        
        tf_scale = 1.0
        if timeframe:
            tf_val = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            if tf_val in ["1m", "5m", "15m", "30m"]:
                tf_scale = 0.75
            elif tf_val in ["1h", "4h", "1d"]:
                tf_scale = 1.5
        
        if tp_type == "rr_ratio":
            min_rr = self.risk_cfg.get("min_risk_reward_ratio", 1.5)
            return sl_dist * min_rr
            
        elif tp_type == "atr" and "ATR" in row and row["ATR"] > 0:
            multiplier = self.tp_cfg.get("atr_multiplier", 3.0) * tf_scale
            return row["ATR"] * multiplier
            
        elif tp_type == "fixed":
            pips = self.tp_cfg.get("fixed_pips", 60)
            return pips * pip_value
            
        # Fallback
        return sl_dist * 1.5

    def calculate_trailing_stop(self, current_price: float, current_sl: float, side: OrderSide, row: pd.Series, entry_price: float, symbol: str = "EUR/USD") -> float:
        """
        Calculate new trailing stop level.
        Enforces 1.0R Breakeven lock + Spread buffer to guarantee zero-risk trades.
        """
        trail_cfg = self.risk_cfg.get("trailing_stop", {})
        if not trail_cfg.get("enabled", False):
            return current_sl

        trail_type = trail_cfg.get("type", "breakeven")
        new_sl = current_sl

        # Determine Pip Value
        pip = 0.0001
        if "JPY" in symbol: pip = 0.01
        elif "BTC" in symbol or "ETH" in symbol: pip = 1.0
        
        spread_buffer = trail_cfg.get("spread_buffer_pips", 2) * pip
        trigger_r = trail_cfg.get("breakeven_trigger_r", 1.0)
        
        # Calculate initial risk distance (1R)
        initial_risk = abs(entry_price - current_sl) if current_sl > 0 else 30 * pip
        
        if trail_type == "breakeven":
            trigger_pips = trail_cfg.get("breakeven_trigger_pips", 15) * pip
            profit_dist = (current_price - entry_price) if side == OrderSide.BUY else (entry_price - current_price)
            
            # Check 1R trigger OR fixed pips trigger
            is_1r_reached = (profit_dist >= trigger_r * initial_risk) and (initial_risk > 0)
            is_pips_reached = profit_dist >= trigger_pips
            
            if is_1r_reached or is_pips_reached:
                if side == OrderSide.BUY:
                    be_level = entry_price + spread_buffer
                    if be_level > current_sl and current_price > be_level:
                        new_sl = be_level
                elif side == OrderSide.SELL:
                    be_level = entry_price - spread_buffer
                    if (current_sl == 0 or be_level < current_sl) and current_price < be_level:
                        new_sl = be_level

        elif trail_type == "atr" and "ATR" in row:
            multiplier = trail_cfg.get("atr_multiplier", 1.5)
            trail_dist = row["ATR"] * multiplier
            
            if side == OrderSide.BUY:
                candidate = current_price - trail_dist
                if candidate > current_sl:
                    new_sl = candidate
            else:
                candidate = current_price + trail_dist
                if candidate < current_sl or current_sl == 0:
                    new_sl = candidate

        return new_sl
