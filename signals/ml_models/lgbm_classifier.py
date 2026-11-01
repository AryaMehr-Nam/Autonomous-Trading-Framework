"""
TradeAgent LightGBM Classifier.

Trains and infers market direction using LightGBM gradient boosting.
"""

import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from config.settings import DATA_DIR, trading_params
from core.logger import get_logger

logger = get_logger("lgbm_classifier")


class LGBMClassifier:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.config = trading_params.ml.get("lightgbm", {})
        self.model = None
        self.is_trained = False
        self.model_path = DATA_DIR / f"lgbm_{symbol.replace('/', '_')}.pkl"
        
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.debug(f"Loaded LGBM model for {self.symbol}")
            except Exception as e:
                logger.error(f"Failed to load LGBM model for {self.symbol}: {e}")

    def save_model(self):
        if self.model:
            try:
                with open(self.model_path, "wb") as f:
                    pickle.dump(self.model, f)
            except Exception as e:
                logger.error(f"Failed to save LGBM model for {self.symbol}: {e}")

    def train(self, features: pd.DataFrame, labels: pd.Series) -> bool:
        if not HAS_LGB:
            logger.warning("lightgbm not installed.")
            return False
            
        if features.empty or labels.empty:
            return False
            
        # Align indices and drop NaNs
        df = features.join(labels.rename("label")).dropna()
        if len(df) < 100:
            logger.warning(f"Not enough data to train LGBM for {self.symbol}")
            return False
            
        X = df.drop(columns=["label"])
        y = df["label"]
        
        # Shift labels from -1,0,1 to 0,1,2 for LGBM multiclass
        y_mapped = y + 1 
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1,
            'random_state': 42
        }
        
        try:
            # TimeSeries split for walk forward validation (simplified to 1 split for speed here)
            tscv = TimeSeriesSplit(n_splits=2)
            train_idx, val_idx = list(tscv.split(X))[-1]
            
            X_train, y_train = X.iloc[train_idx], y_mapped.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y_mapped.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Using standard dict format for early stopping in modern lightgbm
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
            
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data, val_data],
                callbacks=callbacks
            )
            
            self.is_trained = True
            self.save_model()
            logger.info(f"Successfully trained LGBM for {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"LGBM training failed for {self.symbol}: {e}")
            return False

    def predict(self, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict direction for the latest row.
        Returns (direction, confidence)
        Direction: 1 (Up), -1 (Down), 0 (Flat)
        """
        if not self.is_trained or not self.model or features.empty:
            return 0, 0.0
            
        try:
            # Get latest feature row
            latest_X = features.iloc[[-1]]
            
            # Returns probabilities for classes [0, 1, 2] (which map to [-1, 0, 1])
            probs = self.model.predict(latest_X)[0]
            
            class_idx = np.argmax(probs)
            confidence = probs[class_idx]
            
            direction = class_idx - 1
            
            min_conf = self.config.get("min_confidence", 0.65)
            if confidence < min_conf:
                direction = 0 # Revert to flat if confidence is low
                
            return direction, confidence
            
        except Exception as e:
            logger.error(f"LGBM prediction failed for {self.symbol}: {e}")
            return 0, 0.0
