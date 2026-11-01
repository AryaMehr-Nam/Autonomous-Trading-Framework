"""
TradeAgent ML Ensemble.

Combines signals from TA Engine, LightGBM, and RL Agent into a final 
consensus signal using confidence-weighted voting.
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from config.settings import trading_params
from core.logger import get_logger
from signals.ml_models.feature_engineer import FeatureEngineer
from signals.ml_models.lgbm_classifier import LGBMClassifier
from signals.ml_models.rl_agent import RLAgent
from signals.ml_models.drl_agent import DRLAgent
from signals.ml_models.xgboost_forecaster import XGBoostForecaster
from signals.ml_models.transformer_predictor import TransformerPredictor

logger = get_logger("ensemble")


class SignalEnsemble:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.config = trading_params.ml.get("ensemble", {})
        
        self.feat_eng = FeatureEngineer()
        self.lgbm = LGBMClassifier(symbol)
        self.rl = RLAgent(symbol)
        self.drl = DRLAgent()
        self.xgb = XGBoostForecaster(symbol)
        self.transformer = TransformerPredictor()
        
        self.enabled = trading_params.ml.get("enabled", True)

    def evaluate(self, df_ta: pd.DataFrame, ta_direction: int, ta_confidence: float) -> Tuple[int, float, Dict]:
        """
        Evaluate all models and return consensus.
        Returns (direction, confidence, metadata)
        Direction: 1 (Buy), -1 (Sell), 0 (Hold)
        """
        if not self.enabled or df_ta.empty:
            return ta_direction, ta_confidence, {"source": "ta_only"}
            
        features = self.feat_eng.create_features(df_ta)
        
        lgb_dir, lgb_conf = self.lgbm.predict(features)
        rl_dir, rl_conf = self.rl.predict(features)
        xgb_dir, xgb_conf = self.xgb.predict(df_ta)
        drl_dir, drl_conf = self.drl.predict(df_ta)
        trans_dir, trans_conf = self.transformer.predict(df_ta)
        
        # Load adaptive weights from model_performance.json (fallback to defaults if missing)
        import os
        import json
        weight_path = "data_store/ml_models/model_performance.json"
        
        adaptive_weights = {
            "TA": 0.2, "LGBM": 0.1, "XGBoost": 0.2, "DRL": 0.2, "Transformer": 0.3
        }
        
        if os.path.exists(weight_path):
            try:
                with open(weight_path, "r") as f:
                    perf = json.load(f)
                # Recalculate weights based on rolling accuracy
                total_acc = sum(perf.values())
                if total_acc > 0:
                    adaptive_weights = {k: v / total_acc for k, v in perf.items()}
            except Exception as e:
                logger.error(f"Failed to load adaptive weights: {e}")
                
        # Collect votes with Adaptive Weights
        votes = [
            {"dir": ta_direction, "conf": ta_confidence, "weight": adaptive_weights.get("TA", 0.2), "name": "TA"},
            {"dir": lgb_dir, "conf": lgb_conf, "weight": adaptive_weights.get("LGBM", 0.1), "name": "LGBM"},
            {"dir": xgb_dir, "conf": xgb_conf, "weight": adaptive_weights.get("XGBoost", 0.2), "name": "XGBoost"},
            {"dir": drl_dir, "conf": drl_conf, "weight": adaptive_weights.get("DRL", 0.2), "name": "DRL"},
            {"dir": trans_dir, "conf": trans_conf, "weight": adaptive_weights.get("Transformer", 0.3), "name": "Transformer"},
        ]
        
        # Tally weighted score
        total_score = 0.0
        total_weight = 0.0
        
        for v in votes:
            if v["dir"] != 0 and v["conf"] > 0:
                total_score += (v["dir"] * v["conf"] * v["weight"])
                total_weight += v["weight"]
                
        if total_weight == 0:
            return 0, 0.0, {"source": "none"}
            
        consensus_score = total_score / total_weight
        
        direction = 0
        min_agreement = self.config.get("min_agreement", 0.6)
        
        if consensus_score >= min_agreement:
            direction = 1
        elif consensus_score <= -min_agreement:
            direction = -1
            
        metadata = {
            "source": "ensemble",
            "ta_vote": ta_direction,
            "lgbm_vote": lgb_dir,
            "rl_vote": rl_dir,
            "xgb_vote": xgb_dir,
            "consensus_score": consensus_score
        }
        
        return direction, abs(consensus_score), metadata
