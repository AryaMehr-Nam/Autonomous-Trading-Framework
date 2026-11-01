"""
Strategy Registry for TradeAgent Pure Gold Sniper.
==================================================
Manages all available strategies, configuration binding, and strategy lifecycle.
"""

from typing import Dict, List, Optional, Type
from signals.strategies.base_strategy import BaseStrategy
from signals.strategies.pure_gold_legacy import MacroSwingStrategy, PrimeScalpStrategy
from signals.strategies.smc_liquidity_sweep import SMCLiquiditySweepStrategy
from signals.strategies.asian_judas_swing import AsianJudasSwingStrategy
from signals.strategies.vwap_khakestar_snapback import VWAPKhakestarSnapbackStrategy


class StrategyRegistry:
    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(MacroSwingStrategy())
        self.register(PrimeScalpStrategy())
        self.register(SMCLiquiditySweepStrategy())
        self.register(AsianJudasSwingStrategy())
        self.register(VWAPKhakestarSnapbackStrategy())

    def register(self, strategy: BaseStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Optional[BaseStrategy]:
        return self._strategies.get(name)

    def get_all(self) -> Dict[str, BaseStrategy]:
        return dict(self._strategies)

    def get_active(self, active_names: Optional[List[str]] = None) -> List[BaseStrategy]:
        if active_names is None:
            return [s for s in self._strategies.values() if s.params.get("enabled", True)]
        return [self._strategies[name] for name in active_names if name in self._strategies]

    def update_strategy_params(self, strategy_name: str, params: dict) -> bool:
        if strategy_name in self._strategies:
            self._strategies[strategy_name].update_params(params)
            return True
        return False


# Global Singleton Registry
strategy_registry = StrategyRegistry()
