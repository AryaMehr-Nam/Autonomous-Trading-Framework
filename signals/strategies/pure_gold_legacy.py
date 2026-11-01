"""
Modularized Pure Gold Sniper Production Strategies (Macro-Swing & Prime Scalp).
==============================================================================
Encapsulates Khakestar T-Rex FTR + Pour Samadi Killzone Scalper into BaseStrategy.
"""

from typing import Dict, Any, Optional
import numpy as np
from signals.strategies.base_strategy import BaseStrategy, StrategySignal


class MacroSwingStrategy(BaseStrategy):
    """
    Lane 1: H4 Macro-Swing (Khakestar FTR Origin Nodes + HTF Trend + 1.8R/3.5R Runner).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("h4_macro_swing", params)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "min_score": 75,
            "min_risk": 3.00,
            "max_risk": 7.50,
            "tp1_r_multiple": 1.5,
            "tp2_r_multiple": 3.0,
            "max_spread_pts": 45,
            "atr_shock_multiplier": 2.8,
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

        h4 = market_data.get("h4")
        if not h4 or "df" not in h4 or len(h4["df"]) < 30:
            return None

        cur_atr = h4["atr26"][-1]
        if np.isnan(cur_atr) or cur_atr < 2.0:
            return None

        # Volatility shock candle filter
        last_bar_size = h4["high"][-1] - h4["low"][-1]
        if last_bar_size > cur_atr * params["atr_shock_multiplier"]:
            return None

        ema50 = h4["ema50"][-1]
        ema200 = h4["ema200"][-1]
        clv = h4["clv"][-1]
        rsi = h4["rsi"][-1]

        macro_bull = spot_price > ema50 and ema50 > (ema200 * 0.998)
        macro_bear = spot_price < ema50 and ema50 < (ema200 * 1.002)

        untested_nodes = [
            node for node in h4.get("ftr_nodes", [])
            if not getattr(node, "tested", False) and (len(h4["df"]) - getattr(node, "created_idx", 0)) <= 70
        ]

        signal = None
        sl_price = 0.0
        risk_dist = 0.0
        reason = ""

        # BUY SCENARIO
        if macro_bull and clv >= 0.55:
            for node in untested_nodes:
                if getattr(node, "direction", "") == 'BULL':
                    if (node.node_low - cur_atr * 0.15 <= spot_price <= node.node_high + cur_atr * 0.20):
                        score = 40
                        if clv >= 0.65: score += 20
                        if rsi <= 55: score += 20
                        if score >= params["min_score"]:
                            sl_cand = node.node_low - (cur_atr * 0.20)
                            risk = spot_price - sl_cand
                            if params["min_risk"] <= risk <= params["max_risk"]:
                                signal = "BUY"
                                sl_price = sl_cand
                                risk_dist = risk
                                reason = f"H4 FTR Demand [{node.node_low:.2f}-{node.node_high:.2f}] | Score: {score}/100"
                                break

        # SELL SCENARIO
        elif macro_bear and clv <= 0.45:
            for node in untested_nodes:
                if getattr(node, "direction", "") == 'BEAR':
                    if (node.node_low - cur_atr * 0.20 <= spot_price <= node.node_high + cur_atr * 0.15):
                        score = 40
                        if clv <= 0.35: score += 20
                        if rsi >= 45: score += 20
                        if score >= params["min_score"]:
                            sl_cand = node.node_high + (cur_atr * 0.20)
                            risk = sl_cand - spot_price
                            if params["min_risk"] <= risk <= params["max_risk"]:
                                signal = "SELL"
                                sl_price = sl_cand
                                risk_dist = risk
                                reason = f"H4 FTR Supply [{node.node_low:.2f}-{node.node_high:.2f}] | Score: {score}/100"
                                break

        if not signal:
            return None

        tp1_r = params["tp1_r_multiple"]
        tp2_r = params["tp2_r_multiple"]
        tp1_price = spot_price + (risk_dist * tp1_r) if signal == "BUY" else spot_price - (risk_dist * tp1_r)
        tp2_price = spot_price + (risk_dist * tp2_r) if signal == "BUY" else spot_price - (risk_dist * tp2_r)

        return StrategySignal(
            strategy_name=self.name,
            lane="MACRO-SWING H4",
            symbol="XAUUSD",
            signal=signal,
            entry=round(spot_price, 2),
            sl=round(sl_price, 2),
            tp1=round(tp1_price, 2),
            tp2=round(tp2_price, 2),
            risk_dist=round(risk_dist, 2),
            reason=reason,
            confidence=0.85,
            parameters=params,
        )


class PrimeScalpStrategy(BaseStrategy):
    """
    Lane 2: M5/M15 Prime Scalper (Pour Samadi London/NY Killzones + ADR + CLV + Khakestar FTR).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("prime_m15_scalp", params)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "adr_cap": 0.85,
            "min_risk": 1.50,
            "max_risk": 3.80,
            "tp1_r_multiple": 1.8,
            "tp2_r_multiple": 3.5,
            "clv_buy_trigger": 0.58,
            "clv_sell_trigger": 0.42,
            "sl_buffer": 0.40,
            "rsi_buy_max": 72.0,
            "rsi_sell_min": 28.0,
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

        # ADR Filter
        adr_data = market_data.get("adr", (0.5, 10.0, 25.0))
        if adr_data[0] > params["adr_cap"]:
            return None

        # Session filter
        sess_data = market_data.get("session", (True, "PRIME"))
        if not sess_data[0] and sess_data[1] == "OUT_OF_SESSION":
            return None
        sess_name = sess_data[1]

        m15 = market_data.get("m15")
        m5 = market_data.get("m5")
        if not m15 or "df" not in m15 or len(m15["df"]) < 20:
            return None

        cur_atr = m15.get("atr14", [2.0])[-1]
        if np.isnan(cur_atr) or cur_atr < 0.70:
            return None

        # Volatility spike filter
        last_bar_size = m15["high"][-1] - m15["low"][-1]
        if last_bar_size > cur_atr * 2.6:
            return None

        # H4 Macro Alignment
        h4 = market_data.get("h4")
        h4_bear = False
        h4_bull = False
        if h4 and "ema50" in h4 and "ema200" in h4:
            ema50 = h4["ema50"][-1]
            ema200 = h4["ema200"][-1]
            h4_bull = spot_price > ema50 and ema50 > ema200 * 0.998
            h4_bear = spot_price < ema50 and ema50 < ema200 * 1.002

        allow_buy = not h4_bear
        allow_sell = not h4_bull

        clv = m15["clv"][-1]
        rsi = m15.get("rsi", [50.0])[-1]
        m5_clv = m5["clv"][-1] if m5 and "clv" in m5 else clv

        m15_nodes = [node for node in m15.get("ftr_nodes", []) if not getattr(node, "tested", False) and (len(m15["df"]) - getattr(node, "created_idx", 0)) <= 50]
        m5_nodes = [node for node in m5.get("ftr_nodes", []) if not getattr(node, "tested", False) and (len(m5["df"]) - getattr(node, "created_idx", 0)) <= 60] if m5 else []
        untested_nodes = m15_nodes + m5_nodes

        signal = None
        sl_price = 0.0
        risk_dist = 0.0
        reason = ""

        # BUY SCENARIO
        if allow_buy and (clv >= params["clv_buy_trigger"] or m5_clv >= params["clv_buy_trigger"]) and rsi <= params["rsi_buy_max"]:
            for node in untested_nodes:
                if getattr(node, "direction", "") == 'BULL':
                    low_test = min(m15["low"][-1], m5["low"][-1] if m5 else m15["low"][-1])
                    if (node.node_low * 0.999 <= low_test <= node.node_high * 1.0015):
                        local_low = min(m15["low"][-1], m5["low"][-1] if m5 else m15["low"][-1])
                        sl_cand = local_low - params["sl_buffer"]
                        risk = spot_price - sl_cand
                        if params["min_risk"] <= risk <= params["max_risk"]:
                            signal = "BUY"
                            sl_price = sl_cand
                            risk_dist = risk
                            reason = f"[{sess_name}] FTR Demand [{node.node_low:.2f}-{node.node_high:.2f}] | CLV {max(clv, m5_clv):.2f}"
                            break

        # SELL SCENARIO
        elif allow_sell and (clv <= params["clv_sell_trigger"] or m5_clv <= params["clv_sell_trigger"]) and rsi >= params["rsi_sell_min"]:
            for node in untested_nodes:
                if getattr(node, "direction", "") == 'BEAR':
                    high_test = max(m15["high"][-1], m5["high"][-1] if m5 else m15["high"][-1])
                    if (node.node_low * 0.9985 <= high_test <= node.node_high * 1.001):
                        local_high = max(m15["high"][-1], m5["high"][-1] if m5 else m15["high"][-1])
                        sl_cand = local_high + params["sl_buffer"]
                        risk = sl_cand - spot_price
                        if params["min_risk"] <= risk <= params["max_risk"]:
                            signal = "SELL"
                            sl_price = sl_cand
                            risk_dist = risk
                            reason = f"[{sess_name}] FTR Supply [{node.node_low:.2f}-{node.node_high:.2f}] | CLV {min(clv, m5_clv):.2f}"
                            break

        if not signal:
            return None

        tp1_r = params["tp1_r_multiple"]
        tp2_r = params["tp2_r_multiple"]
        tp1_price = spot_price + (risk_dist * tp1_r) if signal == "BUY" else spot_price - (risk_dist * tp1_r)
        tp2_price = spot_price + (risk_dist * tp2_r) if signal == "BUY" else spot_price - (risk_dist * tp2_r)

        return StrategySignal(
            strategy_name=self.name,
            lane="PRIME-SCALP M5/M15",
            symbol="XAUUSD",
            signal=signal,
            entry=round(spot_price, 2),
            sl=round(sl_price, 2),
            tp1=round(tp1_price, 2),
            tp2=round(tp2_price, 2),
            risk_dist=round(risk_dist, 2),
            reason=reason,
            confidence=0.85,
            parameters=params,
        )
