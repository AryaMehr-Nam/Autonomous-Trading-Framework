"""
Deep Reinforcement Learning Agent using Stable-Baselines3 (PPO).
"""
import pandas as pd
from typing import Tuple
from core.logger import get_logger

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

logger = get_logger("drl_agent")

class DRLAgent:
    def __init__(self):
        self.is_trained = False
        self.model = None
        if HAS_SB3:
            logger.info("DRL Agent initialized (PPO Architecture Ready).")
        else:
            logger.error("stable-baselines3 not installed.")
            
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        if not HAS_SB3 or not self.is_trained or self.model is None:
            return 0, 0.0
            
        try:
            # We would normally format a 'state' observation here
            obs = df['close'].iloc[-10:].values # dummy observation
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Assuming action space is 0 (Sell), 1 (Hold), 2 (Buy)
            if action == 2: return 1, 0.6
            if action == 0: return -1, 0.6
            return 0, 0.0
            
        except Exception as e:
            logger.error(f"DRL Agent prediction error: {e}")
            return 0, 0.0
