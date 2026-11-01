import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import asyncio
import pandas as pd
from core.models import Signal, SignalType, OrderSide, OrderType, Timeframe, AssetClass, AccountInfo, Position
from risk.risk_manager import RiskManager
from config.settings import trading_params

def test_risk_manager_min_win_rate():
    rm = RiskManager()
    acc = AccountInfo(balance=200.0, equity=200.0, leverage=100.0)
    
    # 1. Test Low Win-Rate Signal (< 40%) -> Must be rejected!
    bad_sig = Signal(
        symbol="EUR/USD",
        signal_type=SignalType.BUY,
        entry_price=1.1500,
        stop_loss=1.1470,
        take_profit=1.1560,
        confidence=0.8,
        win_rate=0.25, # 25% win rate < 40% threshold
        risk_reward=2.0,
        timeframe=Timeframe.H1,
        asset_class=AssetClass.FOREX
    )
    
    approved = rm.evaluate_signal(bad_sig, acc, [])
    print(f"Low Win-Rate Signal (25%) Evaluation: Approved={approved}")
    assert not approved, "Low win-rate signal should have been rejected by RiskManager!"
    print("✅ Strict Min Win-Rate (>=40%) Filter passed!")
    
    # 2. Test High Win-Rate Signal (>= 40%) -> Must be approved!
    good_sig = Signal(
        symbol="EUR/USD",
        signal_type=SignalType.BUY,
        entry_price=1.1500,
        stop_loss=1.1470,
        take_profit=1.1560,
        confidence=0.8,
        win_rate=0.57, # 57% win rate >= 40% threshold
        risk_reward=2.0,
        timeframe=Timeframe.H1,
        asset_class=AssetClass.FOREX
    )
    approved_good = rm.evaluate_signal(good_sig, acc, [])
    print(f"High Win-Rate Signal (57%) Evaluation: Approved={approved_good}")
    assert approved_good, "High win-rate signal should have been approved by RiskManager!"
    print("✅ High Win-Rate Approval passed!")

if __name__ == "__main__":
    print("Testing Institutional Engine upgrades...")
    test_risk_manager_min_win_rate()
    print("\n🎉 ALL INSTITUTIONAL ENGINE CHECKS PASSED!")
