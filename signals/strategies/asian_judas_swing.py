"""
Asian Range Judas Swing Strategy for Gold (XAUUSD).
===================================================
Institutional Mechanics:
1. Measures Asian Range (00:00 - 06:30 UTC) High, Low, and Midpoint.
2. In London Open Killzone (07:00 - 09:30 UTC), identifies the false directional move ("Judas Swing").
3. Detects institutional stop run beyond the Asian Range boundary.
4. Enters upon immediate rejection / absorption back into the range (CLV trigger).
5. Targets the opposing Asian extreme (TP1) and daily trend expansion (TP2).
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from signals.strategies.base_strategy import BaseStrategy, StrategySignal


class AsianJudasSwingStrategy(BaseStrategy):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__("asian_judas_swing", params)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "asian_start_hour": 0,          # 00:00 UTC
            "asian_end_hour": 6,            # 06:30 UTC
            "judas_start_hour": 7,          # 07:00 UTC (London Open)
            "judas_end_hour": 10,           # 10:00 UTC
            "min_asian_range": 3.0,         # Minimum Asian range in USD
            "max_asian_range": 18.0,        # Maximum Asian range in USD (skip if already massive)
            "min_judas_sweep": 0.30,        # Min USD sweep beyond Asian range
            "max_judas_sweep": 3.50,        # Max USD sweep beyond Asian range
            "clv_rejection_bull": 0.60,     # Candle pressure on reclaim
            "clv_rejection_bear": 0.40,     # Candle pressure on reclaim
            "sl_buffer": 0.40,              # Buffer past the Judas spike extreme
            "min_risk": 1.50,
            "max_risk": 4.50,
            "tp1_r_multiple": 1.8,
            "tp2_r_multiple": 3.5,
            "adr_cap": 0.85,
        }

    def _calculate_asian_range(self, df: pd.DataFrame, params: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        """
        Calculates the Asian Range High and Low for the current trading day.
        Returns: (asian_high, asian_low, asian_range)
        """
        if "time" not in df.columns and "timestamp" not in df.columns:
            return None

        time_col = "time" if "time" in df.columns else "timestamp"
        times = df[time_col]
        if len(times) == 0:
            return None

        last_t = times.iloc[-1]
        if not isinstance(last_t, (pd.Timestamp, datetime)):
            times = pd.to_datetime(times, utc=True)
            last_t = times.iloc[-1]

        cur_date = last_t.date()
        start_h = params.get("asian_start_hour", 0)
        end_h = params.get("asian_end_hour", 6)

        if hasattr(times, "dt"):
            asian_mask = (
                (times.dt.date == cur_date)
                & (times.dt.hour >= start_h)
                & ((times.dt.hour < end_h) | ((times.dt.hour == end_h) & (times.dt.minute <= 30)))
            )
        else:
            asian_mask = [
                (t.date() == cur_date and t.hour >= start_h and (t.hour < end_h or (t.hour == end_h and t.minute <= 30)))
                for t in times
            ]

        asian_df = df[asian_mask]
        if len(asian_df) < 4:
            return None

        asian_h = float(asian_df["high"].max())
        asian_l = float(asian_df["low"].min())
        rng = asian_h - asian_l
        return asian_h, asian_l, rng

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
        if not m15 or "df" not in m15 or len(m15["df"]) < 20:
            return None

        # Check ADR
        adr_data = market_data.get("adr", (0.5, 10.0, 25.0))
        if adr_data[0] > params["adr_cap"]:
            return None

        # Check current time is within Judas Open Window (07:00 - 10:00 UTC)
        cur_dt = market_data.get("time")
        if cur_dt is None:
            # Fallback to last candle time in m15
            df = m15["df"]
            t_col = "time" if "time" in df.columns else "timestamp"
            if t_col in df.columns:
                cur_dt = pd.to_datetime(df[t_col].iloc[-1], utc=True)
            else:
                cur_dt = datetime.now(timezone.utc)
        elif not isinstance(cur_dt, datetime):
            cur_dt = pd.to_datetime(cur_dt, utc=True)

        hour = cur_dt.hour
        if not (params["judas_start_hour"] <= hour < params["judas_end_hour"]):
            return None

        asian_metrics = self._calculate_asian_range(m15["df"], params)
        if not asian_metrics:
            return None

        asian_h, asian_l, asian_range = asian_metrics
        if not (params["min_asian_range"] <= asian_range <= params["max_asian_range"]):
            return None

        cur_h = m15["high"][-1]
        cur_l = m15["low"][-1]
        cur_c = m15["close"][-1]
        prev_h = m15["high"][-2]
        prev_l = m15["low"][-2]
        prev_c = m15["close"][-2]
        clv = m15["clv"][-1]

        signal_dir = None
        sl_price = 0.0
        risk_dist = 0.0
        reason = ""

        # HTF Trend check if available
        h4 = market_data.get("h4")
        h4_bear = False
        h4_bull = False
        if h4 and "ema50" in h4 and "ema200" in h4:
            ema50 = h4["ema50"][-1]
            ema200 = h4["ema200"][-1]
            h4_bull = spot_price > ema50 and ema50 > ema200 * 0.998
            h4_bear = spot_price < ema50 and ema50 < ema200 * 1.002

        # -------------------------------------------------------------
        # 1. BULLISH JUDAS SWING (Asian Low Swept during London Open)
        # -------------------------------------------------------------
        judas_low_swept = False
        lowest_sweep = cur_l
        if (cur_l < asian_l) and (cur_c > asian_l * 0.9995):
            depth = asian_l - cur_l
            if params["min_judas_sweep"] <= depth <= params["max_judas_sweep"]:
                judas_low_swept = True
                lowest_sweep = cur_l
        elif (prev_l < asian_l) and (cur_c >= asian_l):
            depth = asian_l - prev_l
            if params["min_judas_sweep"] <= depth <= params["max_judas_sweep"]:
                judas_low_swept = True
                lowest_sweep = min(cur_l, prev_l)

        if judas_low_swept and not h4_bear and clv >= params["clv_rejection_bull"]:
            candidate_sl = lowest_sweep - params["sl_buffer"]
            risk = spot_price - candidate_sl
            if params["min_risk"] <= risk <= params["max_risk"]:
                signal_dir = "BUY"
                sl_price = candidate_sl
                risk_dist = risk
                reason = f"[LONDON OPEN] Judas Swing Low Swept @ {asian_l:.2f} (Spike: {lowest_sweep:.2f}) | Asian Range: ${asian_range:.2f} | CLV: {clv:.2f}"

        # -------------------------------------------------------------
        # 2. BEARISH JUDAS SWING (Asian High Swept during London Open)
        # -------------------------------------------------------------
        judas_high_swept = False
        highest_sweep = cur_h
        if (cur_h > asian_h) and (cur_c < asian_h * 1.0005):
            depth = cur_h - asian_h
            if params["min_judas_sweep"] <= depth <= params["max_judas_sweep"]:
                judas_high_swept = True
                highest_sweep = cur_h
        elif (prev_h > asian_h) and (cur_c <= asian_h):
            depth = prev_h - asian_h
            if params["min_judas_sweep"] <= depth <= params["max_judas_sweep"]:
                judas_high_swept = True
                highest_sweep = max(cur_h, prev_h)

        if not signal_dir and judas_high_swept and not h4_bull and clv <= params["clv_rejection_bear"]:
            candidate_sl = highest_sweep + params["sl_buffer"]
            risk = candidate_sl - spot_price
            if params["min_risk"] <= risk <= params["max_risk"]:
                signal_dir = "SELL"
                sl_price = candidate_sl
                risk_dist = risk
                reason = f"[LONDON OPEN] Judas Swing High Swept @ {asian_h:.2f} (Spike: {highest_sweep:.2f}) | Asian Range: ${asian_range:.2f} | CLV: {clv:.2f}"

        if not signal_dir:
            return None

        tp1_r = params["tp1_r_multiple"]
        tp2_r = params["tp2_r_multiple"]

        if signal_dir == "BUY":
            tp1_price = spot_price + (risk_dist * tp1_r)
            tp2_price = max(spot_price + (risk_dist * tp2_r), asian_h)
        else:
            tp1_price = spot_price - (risk_dist * tp1_r)
            tp2_price = min(spot_price - (risk_dist * tp2_r), asian_l)

        return StrategySignal(
            strategy_name=self.name,
            lane="ASIAN-JUDAS-SWING M15",
            symbol="XAUUSD",
            signal=signal_dir,
            entry=round(spot_price, 2),
            sl=round(sl_price, 2),
            tp1=round(tp1_price, 2),
            tp2=round(tp2_price, 2),
            risk_dist=round(risk_dist, 2),
            reason=reason,
            confidence=0.84,
            parameters=params,
        )
