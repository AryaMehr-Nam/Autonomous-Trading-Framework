"""
Strategies Module for TradeAgent Pure Gold Sniper.
"""
from signals.strategies.base_strategy import BaseStrategy, StrategySignal
from signals.strategies.pure_gold_legacy import MacroSwingStrategy, PrimeScalpStrategy
from signals.strategies.smc_liquidity_sweep import SMCLiquiditySweepStrategy
from signals.strategies.asian_judas_swing import AsianJudasSwingStrategy
from signals.strategies.vwap_khakestar_snapback import VWAPKhakestarSnapbackStrategy
from signals.strategies.strategy_registry import StrategyRegistry, strategy_registry

__all__ = [
    "BaseStrategy",
    "StrategySignal",
    "MacroSwingStrategy",
    "PrimeScalpStrategy",
    "SMCLiquiditySweepStrategy",
    "AsianJudasSwingStrategy",
    "VWAPKhakestarSnapbackStrategy",
    "StrategyRegistry",
    "strategy_registry",
]
