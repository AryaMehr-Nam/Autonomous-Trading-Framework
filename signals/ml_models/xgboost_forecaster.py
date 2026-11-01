import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from core.logger import get_logger

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = get_logger("xgboost_forecaster")

class XGBoostForecaster:
    def __init__(self, symbol: str):
        self.symbol = symbol.replace("/", "_")
        self.model_path = f"data_store/ml_models/xgb_{self.symbol}.json"
        self.model = None
        self.features = ['RSI', 'MACD_LINE', 'MACD_HIST', 'ATR', 'BB_PCT', 'ADX']
        self._load_model()

    def _load_model(self):
        if not HAS_XGB:
            logger.warning("XGBoost not installed. Forecaster disabled.")
            return
            
        if os.path.exists(self.model_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                logger.info(f"Loaded XGBoost model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model: {e}")
                self.model = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and clean features from TA dataframe."""
        X = pd.DataFrame(index=df.index)
        for col in self.features:
            if col in df.columns:
                X[col] = df[col]
            else:
                X[col] = 0.0
        X = X.fillna(0)
        return X

    def train(self, df: pd.DataFrame, target_col: str = 'close') -> bool:
        """
        Train the model on historical data. 
        Target is 1 if next candle closes higher, 0 if lower.
        """
        if not HAS_XGB: return False
        
        logger.info(f"Training XGBoost model on {len(df)} samples...")
        
        df = df.copy()
        df['target'] = (df[target_col].shift(-1) > df[target_col]).astype(int)
        
        df = df.iloc[:-1] # Drop last row (unknown future)
        
        X = self._prepare_features(df)
        y = df['target']
        
        try:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.model.fit(X, y)
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_model(self.model_path)
            logger.info("XGBoost model trained and saved successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to train XGBoost model: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict the direction of the next candle.
        Returns:
            Tuple[direction, probability]
            direction: 1 (Up), -1 (Down), 0 (Unknown)
        """
        if self.model is None or not HAS_XGB or df.empty:
            return 0, 0.0
            
        try:
            X = self._prepare_features(df.tail(1))
            
            prob = self.model.predict_proba(X)[0]
            pred = self.model.predict(X)[0]
            
            confidence = float(prob[pred])
            direction = 1 if pred == 1 else -1
            
            return direction, confidence
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return 0, 0.0
