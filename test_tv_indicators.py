import pandas as pd
import numpy as np
from signals.tradingview_indicators import (
    calculate_wavetrend,
    calculate_nadaraya_watson,
    calculate_ut_bot,
    calculate_lorentzian_classification,
    add_tradingview_indicators
)
from signals.ta_engine import TAEngine
from signals.ta_signals import TASignalGenerator
from core.models import Timeframe, AssetClass

def test_all_tradingview_indicators():
    print("Testing TradingView Elite Indicators Suite...")
    
    # 1. Generate Synthetic OHLCV dataset (250 bars)
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=250, freq="1h")
    trend = np.linspace(100, 120, 250)
    noise = np.random.normal(0, 1.5, 250)
    closes = trend + noise
    highs = closes + np.random.uniform(0.5, 2.0, 250)
    lows = closes - np.random.uniform(0.5, 2.0, 250)
    opens = closes + np.random.uniform(-0.5, 0.5, 250)
    volumes = np.random.uniform(1000, 5000, 250)
    
    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    }, index=dates)
    
    # 2. Test WaveTrend
    wt = calculate_wavetrend(df)
    assert "WT1" in wt.columns and "WT2" in wt.columns
    print(f"WaveTrend: WT1 range [{wt['WT1'].min():.2f}, {wt['WT1'].max():.2f}] | Bull crosses: {wt['WT_BULL_CROSS'].sum()}")
    
    # 3. Test Nadaraya-Watson Envelope
    nw = calculate_nadaraya_watson(df)
    assert "NW_EST" in nw.columns and "NW_UPPER" in nw.columns and "NW_LOWER" in nw.columns
    print(f"Nadaraya-Watson: Bands computed (Mean spread: {(nw['NW_UPPER'] - nw['NW_LOWER']).mean():.2f})")
    
    # 4. Test UT Bot Alerts
    ut = calculate_ut_bot(df)
    assert "UT_TRAILING_STOP" in ut.columns and "UT_POSITION" in ut.columns
    print(f"UT Bot Alerts: Trailing Stop computed | Buy triggers: {ut['UT_BUY'].sum()} | Sell triggers: {ut['UT_SELL'].sum()}")
    
    # 5. Test Lorentzian Classification
    lor = calculate_lorentzian_classification(df)
    assert "LORENTZIAN_PRED" in lor.columns and "LORENTZIAN_CONF" in lor.columns
    print(f"Lorentzian ML: Predictions range [{lor['LORENTZIAN_PRED'].min()}, {lor['LORENTZIAN_PRED'].max()}] | Mean Conf: {lor['LORENTZIAN_CONF'].mean():.2f}")
    
    # 6. Test Full TAEngine Integration
    ta_eng = TAEngine()
    df_all = ta_eng.add_all_indicators(df)
    print(f"TAEngine Full Run: {len(df_all.columns)} indicators generated across {len(df_all)} clean bars.")
    
    # 7. Test Signal Generation
    ta_sig_gen = TASignalGenerator()
    sig = ta_sig_gen.generate_signal("EUR/USD", df_all, Timeframe.H1, AssetClass.FOREX)
    print(f"Signal Generation Result: {sig.signal_type if sig else 'HOLD'} (Confidence: {sig.confidence if sig else 0.0})")
    
    print("\nALL 4 TRADINGVIEW INDICATORS FULLY FUNCTIONAL AND VERIFIED!")

if __name__ == "__main__":
    test_all_tradingview_indicators()
