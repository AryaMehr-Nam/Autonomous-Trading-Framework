"""
Hypothesis Generation & Strategy Mutation Engine for Autonomous Strategy Improver.
===================================================================================
Formulates novel trading hypotheses and generates exploratory parameter sets
grounded in institutional market mechanics (SMC, Judas Swings, VWAP, FTR Nodes).
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone


@dataclass
class TradingHypothesis:
    hypothesis_id: str
    strategy_name: str
    title: str
    market_mechanism: str
    target_regime: str  # "ALL", "TRENDING", "VOLATILE_CHOP", "RANGING"
    timeframes: List[str]
    parameters: Dict[str, Any]
    status: str = "PROPOSED"  # "PROPOSED", "TESTING", "VETTED", "REJECTED"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evaluation_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HypothesisEngine:
    """
    Maintains a curated knowledge base of institutional market hypotheses and
    generates parametric mutations to find higher-expectancy configurations.
    """

    CORE_HYPOTHESIS_TEMPLATES = [
        {
            "strategy_name": "h4_macro_swing",
            "title": "H4 Khakestar FTR Origin Node Trend Continuation",
            "market_mechanism": "H4 institutional trend pullbacks into fresh Khakestar FTR demand/supply nodes with CLV rejection confirmation.",
            "target_regime": "TRENDING",
            "timeframes": ["H4", "M15"],
            "param_ranges": {
                "min_score": [65, 70, 75, 80],
                "min_risk": [2.50, 3.00, 3.50],
                "max_risk": [6.50, 7.50, 8.50],
                "tp1_r_multiple": [1.5, 1.8, 2.0],
                "tp2_r_multiple": [2.8, 3.2, 3.5],
                "max_spread_pts": [40, 45, 50],
                "atr_shock_multiplier": [2.5, 2.8, 3.2],
            },
        },
        {
            "strategy_name": "smc_liquidity_sweep",
            "title": "SMC Liquidity Sweep & Displacement Reclaim",
            "market_mechanism": "Institutional stop hunts above swing highs / below swing lows followed by aggressive displacement and rejection candle (CLV trigger).",
            "target_regime": "ALL",
            "timeframes": ["M15", "M5"],
            "param_ranges": {
                "lookback_bars": [16, 20, 24, 32],
                "min_sweep_dist": [0.15, 0.25, 0.35],
                "max_sweep_dist": [1.80, 2.50, 3.50],
                "clv_rejection_bull": [0.52, 0.55, 0.58],
                "clv_rejection_bear": [0.42, 0.45, 0.48],
                "sl_buffer": [0.25, 0.35, 0.45],
                "min_risk": [1.50, 1.80],
                "max_risk": [4.50, 5.50, 6.50],
                "tp1_r_multiple": [1.5, 1.8, 2.0],
                "tp2_r_multiple": [2.8, 3.2, 3.5],
                "adr_cap": [0.80, 0.85, 0.90],
            },
        },
        {
            "strategy_name": "asian_judas_swing",
            "title": "Asian Range Judas Swing Manipulation Breakout",
            "market_mechanism": "False morning directional expansion out of Asian consolidation (07:00-09:30 UTC) trapping breakout retail traders before reversing into the true daily institutional trend.",
            "target_regime": "TRENDING",
            "timeframes": ["M15", "M5"],
            "param_ranges": {
                "judas_start_hour": [7],
                "judas_end_hour": [9, 10],
                "min_judas_sweep": [0.15, 0.25, 0.35],
                "max_judas_sweep": [3.50, 4.50, 6.00],
                "clv_rejection_bull": [0.52, 0.55, 0.58],
                "clv_rejection_bear": [0.42, 0.45, 0.48],
                "sl_buffer": [0.35, 0.45, 0.60],
                "min_risk": [1.50, 1.80],
                "max_risk": [5.50, 6.50, 7.50],
                "tp1_r_multiple": [1.5, 1.8, 2.0],
                "tp2_r_multiple": [2.8, 3.2, 3.5],
                "adr_cap": [0.80, 0.85, 0.90],
            },
        },
        {
            "strategy_name": "vwap_khakestar_snapback",
            "title": "Micro-VWAP 2.2-Sigma Stretch & Khakestar FTR Node Touch",
            "market_mechanism": "Extreme standard deviation exhaustion (>=2.1σ from VWAP) colliding with untested Khakestar FTR order blocks producing high-probability mean-reversion snapbacks.",
            "target_regime": "RANGING",
            "timeframes": ["M15"],
            "param_ranges": {
                "vwap_window": [16, 20, 24, 28],
                "sigma_multiplier": [1.8, 2.0, 2.2],
                "clv_rejection_bull": [0.52, 0.55, 0.58],
                "clv_rejection_bear": [0.42, 0.45, 0.48],
                "rsi_oversold": [32.0, 36.0, 40.0],
                "rsi_overbought": [60.0, 64.0, 68.0],
                "sl_buffer": [0.30, 0.40, 0.50],
                "min_risk": [1.50, 1.80],
                "max_risk": [4.00, 5.00, 6.00],
                "tp1_r_multiple": [1.4, 1.6, 1.8],
                "tp2_r_multiple": [2.6, 3.0, 3.4],
                "adr_cap": [0.80, 0.85, 0.90],
            },
        },
        {
            "strategy_name": "prime_m15_scalp",
            "title": "Pour Samadi Prime Killzone Scalper + Asymmetric 1.8R/3.5R Banking",
            "market_mechanism": "London and NY Prime Killzone FTR node reactions filtered by ADR exhaustion and local trigger pin-bar stops.",
            "target_regime": "ALL",
            "timeframes": ["M15", "M5"],
            "param_ranges": {
                "adr_cap": [0.80, 0.85, 0.90],
                "min_risk": [1.20, 1.50],
                "max_risk": [3.50, 3.80, 4.20],
                "tp1_r_multiple": [1.6, 1.8, 2.0],
                "tp2_r_multiple": [3.0, 3.5, 4.0],
                "clv_buy_trigger": [0.55, 0.58, 0.62],
                "clv_sell_trigger": [0.38, 0.42, 0.45],
                "sl_buffer": [0.30, 0.40, 0.50],
                "rsi_buy_max": [70.0, 72.0, 75.0],
                "rsi_sell_min": [25.0, 28.0, 30.0],
            },
        },
    ]

    def __init__(self):
        self.hypothesis_counter = 0

    def generate_candidate_hypotheses(self, count_per_strategy: int = 3, target_strategy: Optional[str] = None) -> List[TradingHypothesis]:
        """
        Generates varied hypotheses across the templates.
        """
        hypotheses = []
        for template in self.CORE_HYPOTHESIS_TEMPLATES:
            strat_name = template["strategy_name"]
            if target_strategy and target_strategy != "all" and target_strategy != strat_name:
                continue

            for _ in range(count_per_strategy):
                self.hypothesis_counter += 1
                h_id = f"HYP-{strat_name.upper()[:4]}-{self.hypothesis_counter:04d}"

                # Sample parameters randomly within institutional ranges
                params = {}
                for k, v_list in template["param_ranges"].items():
                    params[k] = random.choice(v_list)

                # Mutate slightly to explore continuous space
                if "min_risk" in params and "max_risk" in params:
                    if params["min_risk"] >= params["max_risk"]:
                        params["max_risk"] = round(params["min_risk"] + 2.0, 2)

                hypo = TradingHypothesis(
                    hypothesis_id=h_id,
                    strategy_name=strat_name,
                    title=f"{template['title']} (Variant #{self.hypothesis_counter})",
                    market_mechanism=template["market_mechanism"],
                    target_regime=template["target_regime"],
                    timeframes=template["timeframes"],
                    parameters=params,
                )
                hypotheses.append(hypo)

        return hypotheses
