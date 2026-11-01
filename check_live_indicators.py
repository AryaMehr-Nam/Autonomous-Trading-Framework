import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.providers.mt5_provider import MT5DataProvider
from signals.ta_engine import TAEngine
from core.models import Timeframe

async def main():
    provider = MT5DataProvider()
    connected = await provider.connect()
    if not connected:
        print("Failed to connect to MT5 Data Provider.")
        return
        
    engine = TAEngine()
    symbol = "XAU/USD"
    
    print(f"=== REAL-TIME GOLD ({symbol}) MARKET SCAN (WEEKEND / FRIDAY CLOSE) ===")
    
    for tf in [Timeframe.M5, Timeframe.H1, Timeframe.H4]:
        print(f"\n--- TIMEFRAME: {tf.value} ---")
        df = await provider.get_historical_dataframe(symbol, tf, limit=500)
        if df is None or df.empty:
            print(f"Failed to fetch data for {tf.value}. Total rows: 0")
            continue
        print(f"Raw candles fetched: {len(df)} | Last candle timestamp: {df.index[-1]}")
        
        df_ta = engine.add_all_indicators(df)
        if df_ta is None or df_ta.empty:
            print("Not enough candles to compute all indicators after dropping NAs.")
            continue
            
        latest = df_ta.iloc[-1]
        price = latest['close']
        rsi = latest.get('RSI', 0)
        atr = latest.get('ATR', 0)
        adx = latest.get('ADX', 0)
        
        print(f"Last Traded Price: ${price:.2f}")
        print(f"RSI: {rsi:.2f} (Target Buy: <=37 | Target Sell: >=67)")
        print(f"ADX (Trend Strength): {adx:.2f} (Needs > 25 for Trend/Breakout)")
        print(f"ATR (Volatility): {atr:.2f}")
        if 'BBL_20_2.0' in latest and 'BBU_20_2.0' in latest:
            print(f"Bollinger Band Lower (Limit Buy Support): ${latest['BBL_20_2.0']:.2f}")
            print(f"Bollinger Band Upper (Limit Sell Resistance): ${latest['BBU_20_2.0']:.2f}")
            print(f"BB %: {latest.get('BB_PCT', 0):.2f}")
            
    await provider.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
