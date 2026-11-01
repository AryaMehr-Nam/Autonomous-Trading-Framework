"""
Smart Money Concepts (SMC) Liquidity Sweep & Confirm Strategy for Gold (XAUUSD).
================================================================================
Institutional Mechanics:
1. Identifies significant swing highs and swing lows (liquidity pools).
2. Detects False Breakout / Liquidity Sweep: price pierces the swing level to trigger retail stops.
3. Confirms Market Structure Shift (MSS) / Displacement via Pour Samadi CLV rejection pressure.
4. Enters on the retest / close with tight stop behind the sweep wick.
5. Dynamic 2-stage banking: 80% at TP1 (1.8R), runner at TP2 (3.5R).
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from signals.strategies.base_strategy import BaseStrategy, StrategySignal


class SMCLiquiditySweepStrategy(BaseStrategy):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("smc_liquidity_sweep", params)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "lookback_bars": 24,           # Number of M15 bars to locate swing highs/lows
            "min_sweep_dist": 0.20,         # Minimum wick pierce in USD
            "max_sweep_dist": 2.20,         # Maximum wick pierce before considered true breakout
            "clv_rejection_bull": 0.58,     # CLV >= 0.58 confirms buyers absorption
            "clv_rejection_bear": 0.42,     # CLV <= 0.42 confirms sellers absorption
            "sl_buffer": 0.35,              # Buffer behind the sweep wick extreme
            "min_risk": 1.50,               # Minimum risk distance ($1.50)
            "max_risk": 4.20,               # Maximum allowed risk ($4.20)
            "tp1_r_multiple": 1.8,          # Partial TP1 target
            "tp2_r_multiple": 3.5,          # Runner TP2 target
            "adr_cap": 0.85,                # Max daily ADR ratio allowed
        }

    def _find_swings(self, highs: np.ndarray, lows: np.ndarray, lookback: int) -> Tuple[float, float, int, int]:
        """Finds highest high and lowest low within lookback window (excluding current bar)."""
        if len(highs) <= lookback + 1:
            return float(np.max(highs[:-1])), float(np.min(lows[:-1])), 0, 0
        window_h = highs[-(lookback + 1):-1]
        window_l = lows[-(lookback + 1):-1]
        swing_h = float(np.max(window_h))
        swing_l = float(np.min(window_l))
        h_idx = int(np.argmax(window_h))
        l_idx = int(np.argmin(window_l))
        return swing_h, swing_l, h_idx, l_idx

    def evaluate(
        self,
        market_data: Dict[str, Any],
        spot_price: float,
        equity: float = 500.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[StrategySignal]:
        params = dict(self.params)
        if config:
            params.update(config)

        if not params.get("enabled", True):
            return None

        m15 = market_data.get("m15")
        if not m15 or "df" not in m15 or len(m15["df"]) < params["lookback_bars"] + 5:
            return None

        # ADR usage filter
        adr_data = market_data.get("adr", (0.5, 10.0, 25.0))
        adr_ratio = adr_data[0]
        if adr_ratio > params["adr_cap"]:
            return None

        # Session filter
        sess_data = market_data.get("session", (True, "ACTIVE"))
        is_prime = sess_data[0]
        sess_name = sess_data[1]
        if not is_prime and sess_name == "OUT_OF_SESSION":
            return None

        h15 = m15["high"]
        l15 = m15["low"]
        c15 = m15["close"]
        o15 = m15["df"]["open"].values if "open" in m15["df"] else c15
        clv = m15["clv"][-1]
        cur_atr = m15.get("atr14", [2.5])[-1]

        lookback = params["lookback_bars"]
        swing_h, swing_l, _, _ = self._find_swings(h15, l15, lookback)

        cur_h = h15[-1]
        cur_l = l15[-1]
        cur_c = c15[-1]
        prev_h = h15[-2]
        prev_l = l15[-2]
        prev_c = c15[-2]

        signal_dir = None
        sl_price = 0.0
        risk_dist = 0.0
        reason = ""

        # Check Higher-Timeframe Trend alignment if H4 is present
        h4 = market_data.get("h4")
        h4_bear = False
        h4_bull = False
        if h4 and "ema50" in h4 and "ema200" in h4:
            ema50 = h4["ema50"][-1]
            ema200 = h4["ema200"][-1]
            h4_bull = spot_price > ema50 and ema50 > ema200 * 0.998
            h4_bear = spot_price < ema50 and ema50 < ema200 * 1.002

        # -------------------------------------------------------------
        # 1. BULLISH LIQUIDITY SWEEP (Sell-Side Liquidity Swept)
        # -------------------------------------------------------------
        # Conditions:
        # A) Either current bar or previous bar pierced below swing_l
        # B) Piercing distance is between min_sweep_dist and max_sweep_dist
        # C) Current bar closes BACK ABOVE swing_l (liquidity reclaim / fakeout)
        # D) CLV shows strong buyer absorption (>= clv_rejection_bull)
        # E) Not trading counter to strong H4 bear trend
        # -------------------------------------------------------------
        swept_low = False
        lowest_sweep = cur_l
        if (cur_l < swing_l) and (cur_c > swing_l):
            sweep_depth = swing_l - cur_l
            if params["min_sweep_dist"] <= sweep_depth <= params["max_sweep_dist"]:
                swept_low = True
                lowest_sweep = cur_l
        elif (prev_l < swing_l) and (prev_c < swing_l or prev_c > swing_l) and (cur_c > swing_l):
            sweep_depth = swing_l - prev_l
            if params["min_sweep_dist"] <= sweep_depth <= params["max_sweep_dist"]:
                swept_low = True
                lowest_sweep = min(cur_l, prev_l)

        if swept_low and not h4_bear and clv >= params["clv_rejection_bull"]:
            candidate_sl = lowest_sweep - params["sl_buffer"]
            risk = spot_price - candidate_sl
            if params["min_risk"] <= risk <= params["max_risk"]:
                signal_dir = "BUY"
                sl_price = candidate_sl
                risk_dist = risk
                reason = f"[{sess_name}] SMC Sell-Side Liquidity Swept @ {swing_l:.2f} (Low: {lowest_sweep:.2f}) | CLV: {clv:.2f}"

        # -------------------------------------------------------------
        # 2. BEARISH LIQUIDITY SWEEP (Buy-Side Liquidity Swept)
        # -------------------------------------------------------------
        swept_high = False
        highest_sweep = cur_h
        if (cur_h > swing_h) and (cur_c < swing_h):
            sweep_depth = cur_h - swing_h
            if params["min_sweep_dist"] <= sweep_depth <= params["max_sweep_dist"]:
                swept_high = True
                highest_sweep = cur_h
        elif (prev_h > swing_h) and (cur_c < swing_h):
            sweep_depth = prev_h - swing_h
            if params["min_sweep_dist"] <= sweep_depth <= params["max_sweep_dist"]:
                swept_high = True
                highest_sweep = max(cur_h, prev_h)

        if not signal_dir and swept_high and not h4_bull and clv <= params["clv_rejection_bear"]:
            candidate_sl = highest_sweep + params["sl_buffer"]
            risk = candidate_sl - spot_price
            if params["min_risk"] <= risk <= params["max_risk"]:
                signal_dir = "SELL"
                sl_price = candidate_sl
                risk_dist = risk
                reason = f"[{sess_name}] SMC Buy-Side Liquidity Swept @ {swing_h:.2f} (High: {highest_sweep:.2f}) | CLV: {clv:.2f}"

        if not signal_dir:
            return None

        tp1_r = params["tp1_r_multiple"]
        tp2_r = params["tp2_r_multiple"]

        if signal_dir == "BUY":
            tp1_price = spot_price + (risk_dist * tp1_r)
            tp2_price = spot_price + (risk_dist * tp2_r)
        else:
            tp1_price = spot_price - (risk_dist * tp1_r)
            tp2_price = spot_price - (risk_dist * tp2_r)

        return StrategySignal(
            strategy_name=self.name,
            lane="SMC-LIQUIDITY-SWEEP M15",
            symbol="XAUUSD",
            signal=signal_dir,
            entry=round(spot_price, 2),
            sl=round(sl_price, 2),
            tp1=round(tp1_price, 2),
            tp2=round(tp2_price, 2),
            risk_dist=round(risk_dist, 2),
            reason=reason,
            confidence=0.82,
            parameters=params,
        )
