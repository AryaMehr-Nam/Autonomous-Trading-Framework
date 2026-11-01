"""
Base Strategy Interface for TradeAgent Pure Gold Sniper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class StrategySignal:
    strategy_name: str
    lane: str
    symbol: str
    signal: str  # "BUY" or "SELL"
    entry: float
    sl: float
    tp1: float
    tp2: float
    risk_dist: float
    reason: str
    confidence: float = 0.80
    lot: float = 0.0
    tp1_lot: float = 0.0
    runner_lot: float = 0.0
    tp: float = field(init=False)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.tp = self.tp2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane": self.lane,
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "signal": self.signal,
            "entry": self.entry,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp": self.tp,
            "risk_dist": self.risk_dist,
            "reason": self.reason,
            "confidence": self.confidence,
            "lot": self.lot,
            "tp1_lot": self.tp1_lot,
            "runner_lot": self.runner_lot,
            "parameters": self.parameters,
        }


class BaseStrategy(ABC):
    """
    Abstract Base Class for modular trading strategies.
    Ensures unified evaluation across both live MT5 sniper and historical backtester.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = self.get_default_params()
        if params:
            self.params.update(params)

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Returns default parameters for the strategy."""
        pass

    @abstractmethod
    def evaluate(
        self,
        market_data: Dict[str, Any],
        spot_price: float,
        equity: float = 500.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[StrategySignal]:
        """
        Evaluates current market data and generates a StrategySignal if rules are met.
        market_data contains:
          - "m15": dict of M15 indicators and candles
          - "m5": dict of M5 indicators and candles (optional)
          - "h4": dict of H4 indicators and candles (optional)
          - "adr": (consumed_ratio, today_range, adr14)
          - "session": (is_prime, session_name)
          - "time": current timestamp or bar time
        """
        pass

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """Dynamically update strategy parameters."""
        self.params.update(new_params)
