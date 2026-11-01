"""
TradeAgent Reinforcement Learning Agent.

Uses stable-baselines3 to train an RL agent on a custom trading gym environment.
"""

import os
from typing import Tuple

import pandas as pd
import numpy as np

try:
    from stable_baselines3 import PPO
    import gymnasium as gym
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from config.settings import DATA_DIR, trading_params
from core.logger import get_logger

logger = get_logger("rl_agent")


class RLAgent:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.config = trading_params.ml.get("reinforcement_learning", {})
        self.model = None
        self.is_trained = False
        self.model_path = DATA_DIR / f"ppo_{symbol.replace('/', '_')}.zip"
        
        self._load_model()

    def _load_model(self):
        if not HAS_SB3: return
        if self.model_path.exists():
            try:
                self.model = PPO.load(str(self.model_path))
                self.is_trained = True
                logger.debug(f"Loaded PPO model for {self.symbol}")
            except Exception as e:
                logger.error(f"Failed to load PPO model for {self.symbol}: {e}")

    def train(self, df: pd.DataFrame) -> bool:
        """
        Placeholder for RL training.
        Requires a fully defined Gym environment (e.g. gym-anytrading) 
        which is beyond the scope of this single file but easily integrated.
        """
        if not HAS_SB3:
            logger.warning("stable-baselines3 not installed.")
            return False
            
        if not self.config.get("enabled", False):
            return False
            
        logger.info(f"RL training is stubbed out for {self.symbol}. Requires Gym env implementation.")
        return False

    def predict(self, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict action from observation.
        Returns (direction, confidence)
        Direction: 1 (Buy), -1 (Sell), 0 (Hold)
        """
        if not self.is_trained or not self.model or features.empty:
            return 0, 0.0
            
        try:
            # Flatten latest row to match observation space
            obs = features.iloc[-1].values.astype(np.float32)
            
            # Predict action (Assuming action space is Discrete(3): 0=Sell, 1=Hold, 2=Buy)
            action, _states = self.model.predict(obs, deterministic=True)
            
            direction = int(action) - 1
            confidence = 0.5 # PPO doesn't output confidence directly like LGBM
            
            return direction, confidence
        except Exception as e:
            logger.error(f"PPO prediction error: {e}")
            return 0, 0.0
