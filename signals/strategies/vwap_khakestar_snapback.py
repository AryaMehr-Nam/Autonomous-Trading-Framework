"""
Micro-VWAP 2.2-Sigma Mean Reversion + Khakestar FTR Node Strategy for Gold (XAUUSD).
===================================================================================
Institutional Mechanics:
1. Calculates rolling session VWAP and Standard Deviation bands (2.0σ - 2.5σ).
2. Detects extreme price exhaustion beyond bands in confluence with Khakestar FTR nodes.
3. Confirms institutional absorption via CLV candlestick triggers and RSI extremes.
4. Executes mean reversion snapback targeting VWAP equilibrium (TP1) and runner (TP2).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from signals.strategies.base_strategy import BaseStrategy, StrategySignal


class VWAPKhakestarSnapbackStrategy(BaseStrategy):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("vwap_khakestar_snapback", params)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "vwap_window": 28,             # Number of M15 bars for rolling VWAP & sigma
            "sigma_multiplier": 2.1,       # Standard deviation multiplier threshold (2.1σ)
            "clv_rejection_bull": 0.58,    # Candle pressure for snapback buy
            "clv_rejection_bear": 0.42,    # Candle pressure for snapback sell
            "rsi_oversold": 35.0,
            "rsi_overbought": 65.0,
            "sl_buffer": 0.40,
            "min_risk": 1.50,
            "max_risk": 4.00,
            "tp1_r_multiple": 1.8,
            "tp2_r_multiple": 3.2,
            "adr_cap": 0.85,
        }

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
        if not m15 or "df" not in m15 or len(m15["df"]) < params["vwap_window"] + 2:
            return None

        # ADR usage filter
        adr_data = market_data.get("adr", (0.5, 10.0, 25.0))
        if adr_data[0] > params["adr_cap"]:
            return None

        # Session filter
        sess_data = market_data.get("session", (True, "ACTIVE"))
        if not sess_data[0] and sess_data[1] == "OUT_OF_SESSION":
            return None
        sess_name = sess_data[1]

        c = m15["close"]
        h = m15["high"]
        l = m15["low"]
        v = m15["df"]["tick_volume"].values.astype(float) if "tick_volume" in m15["df"] else np.ones(len(c))
        clv = m15["clv"][-1]
        rsi = m15.get("rsi", [50.0])[-1]

        w = params["vwap_window"]
        recent_c = c[-w:]
        recent_h = h[-w:]
        recent_l = l[-w:]
        recent_v = v[-w:]

        hlc3 = (recent_c + recent_h + recent_l) / 3.0
        sum_v = np.sum(recent_v)
        if sum_v <= 0:
            return None
        vwap_val = float(np.sum(hlc3 * recent_v) / sum_v)
        std_val = float(np.std(recent_c))
        if std_val < 0.30:
            std_val = 0.30

        mult = params["sigma_multiplier"]
        upper_band = vwap_val + (mult * std_val)
        lower_band = vwap_val - (mult * std_val)

        cur_h = h[-1]
        cur_l = l[-1]
        cur_c = c[-1]

        # Khakestar FTR nodes touch check if present
        ftr_nodes = m15.get("ftr_nodes", [])
        untested_nodes = [node for node in ftr_nodes if not getattr(node, "tested", False)]

        signal_dir = None
        sl_price = 0.0
        risk_dist = 0.0
        reason = ""

        # HTF Trend check
        h4 = market_data.get("h4")
        h4_bear = False
        h4_bull = False
        if h4 and "ema50" in h4 and "ema200" in h4:
            ema50 = h4["ema50"][-1]
            ema200 = h4["ema200"][-1]
            h4_bull = spot_price > ema50 and ema50 > ema200 * 0.998
            h4_bear = spot_price < ema50 and ema50 < ema200 * 1.002

        # -------------------------------------------------------------
        # 1. OVERSOLD SNAPBACK BUY (Price stretched below Lower Band)
        # -------------------------------------------------------------
        if (cur_l <= lower_band or spot_price <= lower_band) and clv >= params["clv_rejection_bull"] and rsi <= params["rsi_oversold"] + 15:
            if not h4_bear:
                # Confluence: touches an FTR demand node
                node_confluence = any(
                    node.direction == 'BULL' and (node.node_low * 0.998 <= cur_l <= node.node_high * 1.002)
                    for node in untested_nodes
                ) if untested_nodes else True

                if node_confluence:
                    candidate_sl = cur_l - params["sl_buffer"]
                    risk = spot_price - candidate_sl
                    if params["min_risk"] <= risk <= params["max_risk"]:
                        signal_dir = "BUY"
                        sl_price = candidate_sl
                        risk_dist = risk
                        reason = f"[{sess_name}] VWAP {mult}σ Oversold Snapback @ {spot_price:.2f} (VWAP: {vwap_val:.2f}, Lower: {lower_band:.2f}) | CLV: {clv:.2f}"

        # -------------------------------------------------------------
        # 2. OVERBOUGHT SNAPBACK SELL (Price stretched above Upper Band)
        # -------------------------------------------------------------
        if not signal_dir and (cur_h >= upper_band or spot_price >= upper_band) and clv <= params["clv_rejection_bear"] and rsi >= params["rsi_overbought"] - 15:
            if not h4_bull:
                node_confluence = any(
                    node.direction == 'BEAR' and (node.node_low * 0.998 <= cur_h <= node.node_high * 1.002)
                    for node in untested_nodes
                ) if untested_nodes else True

                if node_confluence:
                    candidate_sl = cur_h + params["sl_buffer"]
                    risk = candidate_sl - spot_price
                    if params["min_risk"] <= risk <= params["max_risk"]:
                        signal_dir = "SELL"
                        sl_price = candidate_sl
                        risk_dist = risk
                        reason = f"[{sess_name}] VWAP {mult}σ Overbought Snapback @ {spot_price:.2f} (VWAP: {vwap_val:.2f}, Upper: {upper_band:.2f}) | CLV: {clv:.2f}"

        if not signal_dir:
            return None

        tp1_r = params["tp1_r_multiple"]
        tp2_r = params["tp2_r_multiple"]

        if signal_dir == "BUY":
            # TP1 targets VWAP with minimum 1.2R to prevent colliding with breakeven buffer
            min_tp1 = spot_price + (risk_dist * 1.2)
            max_tp1 = spot_price + (risk_dist * tp1_r)
            tp1_price = max(min_tp1, min(max_tp1, vwap_val)) if vwap_val > spot_price else max_tp1
            tp2_price = spot_price + (risk_dist * tp2_r)
        else:
            min_tp1 = spot_price - (risk_dist * 1.2)
            max_tp1 = spot_price - (risk_dist * tp1_r)
            tp1_price = min(min_tp1, max(max_tp1, vwap_val)) if vwap_val < spot_price else max_tp1
            tp2_price = spot_price - (risk_dist * tp2_r)

        return StrategySignal(
            strategy_name=self.name,
            lane="VWAP-SNAPBACK M15",
            symbol="XAUUSD",
            signal=signal_dir,
            entry=round(spot_price, 2),
            sl=round(sl_price, 2),
            tp1=round(tp1_price, 2),
            tp2=round(tp2_price, 2),
            risk_dist=round(risk_dist, 2),
            reason=reason,
            confidence=0.80,
            parameters=params,
        )
