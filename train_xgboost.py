import asyncio
import pandas as pd
from datetime import datetime, timedelta
from data.providers.demo_provider import DemoDataProvider
from signals.ta_engine import TAEngine
from signals.ml_models.xgboost_forecaster import XGBoostForecaster
from config.settings import trading_params

async def main():
    print("Fetching historical data for training all symbols...")
    provider = DemoDataProvider()
    await provider.connect()
    
    symbols = []
    if hasattr(trading_params, 'general') and 'watchlist' in trading_params.general:
        wl = trading_params.general['watchlist']
        symbols.extend(wl.get('forex', []))
        symbols.extend(wl.get('crypto', []))
        symbols.extend(wl.get('equity', []))
    else:
        symbols = ["XAU/USD", "EUR/USD", "US30", "GBP/USD", "BTC/USDT", "ETH/USDT"]
        
    engine = TAEngine()
    
    for sym in symbols:
        print(f"\n--- Training {sym} ---")
        try:
            df = await provider.get_historical_dataframe(sym, "H1", limit=1000)
            if df.empty:
                print(f"No data for {sym}. Skipping.")
                continue
                
            df_ta = engine.add_all_indicators(df)
            
            forecaster = XGBoostForecaster(sym)
            success = forecaster.train(df_ta)
            
            if success:
                print(f"Model for {sym} trained successfully!")
            else:
                print(f"Failed to train model for {sym}.")
        except Exception as e:
            print(f"Error training {sym}: {e}")
            
    await provider.disconnect()

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
