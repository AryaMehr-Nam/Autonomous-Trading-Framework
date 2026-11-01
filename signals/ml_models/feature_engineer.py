"""
TradeAgent ML Feature Engineer.

Transforms raw OHLCV and TA data into stationary, normalized features
suitable for Machine Learning models (LightGBM, RL).
"""

import pandas as pd
import numpy as np
from core.logger import get_logger

logger = get_logger("feature_engineer")


class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates ML-ready features from a DataFrame that already has TA indicators.
        Returns a new DataFrame with only the feature columns.
        """
        if df.empty: return pd.DataFrame()
        
        feat = pd.DataFrame(index=df.index)
        
        try:
            # 1. Price Returns (Stationary)
            feat['ret_1'] = df['close'].pct_change(1)
            feat['ret_3'] = df['close'].pct_change(3)
            feat['ret_5'] = df['close'].pct_change(5)
            
            # 2. Volatility normalized (ATR ratio)
            if 'ATR' in df.columns:
                feat['atr_ratio'] = df['ATR'] / df['close']
                
            # 3. Distance to Moving Averages (Stationary)
            if 'EMA_50' in df.columns:
                feat['dist_ema50'] = (df['close'] - df['EMA_50']) / df['close']
            if 'EMA_200' in df.columns:
                feat['dist_ema200'] = (df['close'] - df['EMA_200']) / df['close']
                
            # 4. Momentum Oscillators (Already stationary 0-100 or bounded)
            if 'RSI' in df.columns:
                feat['rsi'] = df['RSI'] / 100.0  # Normalize 0-1
                
            if 'MACD_HIST' in df.columns and 'close' in df.columns:
                feat['macd_hist_norm'] = df['MACD_HIST'] / df['close']
                
            # 5. Time features (Cyclical encoding)
            feat['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
            feat['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
            feat['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
            feat['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
            
            # Drop NaNs created by pct_change
            feat.dropna(inplace=True)
            
            # Clip extreme outliers (3 std dev)
            # In a real system, you'd fit a robust scaler, but this is a quick proxy
            for col in feat.columns:
                if 'sin' not in col and 'cos' not in col:
                    mean = feat[col].mean()
                    std = feat[col].std()
                    feat[col] = feat[col].clip(mean - 3*std, mean + 3*std)
                    
            return feat
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return pd.DataFrame()

    def create_labels(self, df: pd.DataFrame, horizon: int = 12, threshold: float = 0.001) -> pd.Series:
        """
        Creates classification labels for training.
        1 = Up, -1 = Down, 0 = Flat
        """
        if df.empty: return pd.Series(dtype=int)
        
        # Future return over horizon
        future_ret = df['close'].shift(-horizon) / df['close'] - 1.0
        
        labels = pd.Series(0, index=df.index)
        labels[future_ret > threshold] = 1
        labels[future_ret < -threshold] = -1
        
        # Drop the last 'horizon' rows as they have no future data
        labels.iloc[-horizon:] = np.nan
        
        return labels
