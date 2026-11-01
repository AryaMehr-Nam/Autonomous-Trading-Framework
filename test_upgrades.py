import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import asyncio
import pandas as pd
import numpy as np
from core.models import Timeframe, AssetClass, Signal, SignalType, OrderSide
from signals.context_agent import ContextAgent, MarketContext
from signals.ta_signals import TASignalGenerator
from signals.ta_engine import TAEngine
from risk.stop_loss import StopLossManager
from config.settings import trading_params

def test_breakeven():
    sl_mgr = StopLossManager()
    row = pd.Series({"ATR": 0.0015})
    
    # 1. Test Buy Trade moving to Breakeven at 1R profit
    entry = 1.1000
    initial_sl = 1.0970  # Risk = 30 pips
    current_price_1r = 1.1035  # Profit = 35 pips (> 1R)
    
    new_sl = sl_mgr.calculate_trailing_stop(
        current_price=current_price_1r,
        current_sl=initial_sl,
        side=OrderSide.BUY,
        row=row,
        entry_price=entry,
        symbol="EUR/USD"
    )
    print(f"Breakeven Test BUY: Entry={entry}, Initial SL={initial_sl}, Current={current_price_1r} -> New SL={new_sl:.5f}")
    assert new_sl > initial_sl, "SL should have been moved to breakeven!"
    print("✅ Breakeven SL calculation passed!")

def test_context_macro():
    ctx_agent = ContextAgent()
    
    # Fake H1 DataFrame with clear uptrend (300 bars)
    dates = pd.date_range("2026-01-01", periods=300, freq="1h")
    prices = np.linspace(1.0500, 1.1500, 300)
    df_h1 = pd.DataFrame({
        "open": prices, "high": prices + 0.0010, "low": prices - 0.0010, "close": prices, "volume": 1000
    }, index=dates)
    
    ta_eng = TAEngine()
    df_h1_ta = ta_eng.add_all_indicators(df_h1)
    
    macro_dir, macro_desc = ctx_agent.evaluate_macro_bias(df_h1_ta)
    print(f"Macro Context Test: Direction={macro_dir}, Description={macro_desc}")
    assert macro_dir == 1, "Macro bias should be Bullish (1)!"
    print("✅ Macro Context evaluation passed!")

def test_asset_strategy_routing():
    print(f"Asset Strategies configured: {getattr(trading_params, 'asset_strategies', {})}")
    assert "EUR/USD" in trading_params.asset_strategies
    assert trading_params.asset_strategies["EUR/USD"] in ["smc", "mean_reversion"]
    assert trading_params.asset_strategies["GBP/USD"] in ["trend", "mean_reversion"]
    print("✅ Asset-to-Strategy routing configuration verified!")

if __name__ == "__main__":
    print("Running Institutional Upgrade verification tests...")
    test_breakeven()
    test_context_macro()
    test_asset_strategy_routing()
    print("\n🎉 ALL INSTITUTIONAL UPGRADES VERIFIED!")
