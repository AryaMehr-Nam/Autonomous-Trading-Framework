import asyncio
import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")

from config.settings import trading_params, settings
from core.models import Timeframe, AssetClass, SignalType, OrderSide
from data.providers.mt5_provider import MT5DataProvider
from signals.ta_engine import TAEngine
from signals.context_agent import ContextAgent, MarketContext
from signals.ta_signals import TASignalGenerator
from signals.smart_money_concepts import detect_smc_signal
from risk.stop_loss import StopLossManager

async def run_live_market_analysis():
    print("================================================================================")
    print("                 TradeAgent V2 - LIVE META TRADER 5 MARKET ANALYSIS              ")
    print("================================================================================")
    
    provider = MT5DataProvider()
    connected = await provider.connect()
    
    if not connected:
        print("[!] MT5 connection not available. Falling back to Live Demo Data Provider...")
        from data.providers.demo_provider import DemoDataProvider
        provider = DemoDataProvider()
        await provider.connect()
        
    ta_engine = TAEngine()
    context_agent = ContextAgent()
    ta_signals = TASignalGenerator()
    sl_manager = StopLossManager()
    
    watchlist = ["EUR/USD", "GBP/USD", "BTC/USDT", "ETH/USDT"]
    
    analysis_reports = []
    
    for symbol in watchlist:
        # 1. Fetch H1 Macro Data (300 bars)
        df_h1 = await provider.get_historical_dataframe(symbol, Timeframe.H1, limit=300)
        # 2. Fetch M15 Micro Data (300 bars)
        df_m15 = await provider.get_historical_dataframe(symbol, Timeframe.M15, limit=300)
        
        if df_m15 is None or df_m15.empty or len(df_m15) < 50:
            continue
            
        df_m15_ta = ta_engine.add_all_indicators(df_m15)
        latest_m15 = df_m15_ta.iloc[-1]
        
        # Macro H1 Context
        macro_dir = 0
        macro_desc = "MACRO_NEUTRAL"
        h1_close = 0
        h1_ema99 = 0
        if df_h1 is not None and not df_h1.empty:
            df_h1_ta = ta_engine.add_all_indicators(df_h1)
            macro_dir, macro_desc = context_agent.evaluate_macro_bias(df_h1_ta)
            h1_close = df_h1_ta.iloc[-1]['close']
            h1_ema99 = df_h1_ta.iloc[-1].get('EMA_99', df_h1_ta.iloc[-1].get('EMA_50', 0))
            
        # Micro Indicators
        current_price = latest_m15['close']
        rsi = latest_m15.get('RSI', 50.0)
        wt1 = latest_m15.get('WT1', 0.0)
        wt2 = latest_m15.get('WT2', 0.0)
        nw_upper = latest_m15.get('NW_UPPER', current_price * 1.01)
        nw_lower = latest_m15.get('NW_LOWER', current_price * 0.99)
        ut_pos = latest_m15.get('UT_POSITION', 0)
        lor_pred = latest_m15.get('LORENTZIAN_PRED', 0)
        lor_conf = latest_m15.get('LORENTZIAN_CONF', 0.5)
        atr = latest_m15.get('ATR', 0.0010)
        
        # Determine Preferred Strategy
        asset_strat = getattr(trading_params, "asset_strategies", {}).get(symbol, "smc")
        
        # SMC Signals
        smc_dir, smc_conf, smc_ot, smc_pr = detect_smc_signal(df_m15_ta)
        
        # TA Signals
        asset_class = AssetClass.CRYPTO if "USDT" in symbol else AssetClass.FOREX
        ta_sig = ta_signals.generate_signal(symbol, df_m15_ta, Timeframe.M15, asset_class)
        
        # Synthesize Trade Recommendation
        rec = "WATCH / WAIT"
        action = "NEUTRAL"
        setup_sl = 0.0
        setup_tp1 = 0.0
        setup_tp2 = 0.0
        risk_dist = atr * 1.5
        
        pip = 0.0001 if "USD" in symbol and "JPY" not in symbol else (1.0 if "USDT" in symbol else 0.01)
        
        if macro_dir == 1 and (lor_pred == 1 or wt1 < -40 or ut_pos == 1 or smc_dir == 1):
            if rsi < 68 and current_price <= nw_upper:
                rec = "POTENTIAL BUY (Dip into Support)"
                action = "BUY"
                setup_sl = current_price - risk_dist
                setup_tp1 = current_price + risk_dist # 1R Breakeven target
                setup_tp2 = current_price + (risk_dist * 2.5) # Extended TP
        elif macro_dir == -1 and (lor_pred == -1 or wt1 > 40 or ut_pos == -1 or smc_dir == -1):
            if rsi > 32 and current_price >= nw_lower:
                rec = "POTENTIAL SELL (Rally into Resistance)"
                action = "SELL"
                setup_sl = current_price + risk_dist
                setup_tp1 = current_price - risk_dist # 1R Breakeven target
                setup_tp2 = current_price - (risk_dist * 2.5) # Extended TP
        else:
            if wt1 <= -50 or current_price <= nw_lower:
                rec = "MEAN REVERSION BUY (Oversold Snapback)"
                action = "BUY"
                setup_sl = current_price - risk_dist
                setup_tp1 = current_price + risk_dist
                setup_tp2 = current_price + (risk_dist * 2.0)
            elif wt1 >= 50 or current_price >= nw_upper:
                rec = "MEAN REVERSION SELL (Overbought Snapback)"
                action = "SELL"
                setup_sl = current_price + risk_dist
                setup_tp1 = current_price - risk_dist
                setup_tp2 = current_price - (risk_dist * 2.0)

        analysis_reports.append({
            "symbol": symbol,
            "price": current_price,
            "macro_desc": macro_desc,
            "h1_ema99": h1_ema99,
            "rsi": rsi,
            "wt": f"WT1: {wt1:.1f} / WT2: {wt2:.1f}",
            "nw_bands": f"[{nw_lower:.5f} - {nw_upper:.5f}]",
            "ut_bot": "BULLISH" if ut_pos == 1 else ("BEARISH" if ut_pos == -1 else "FLAT"),
            "lorentzian": f"{'BULL' if lor_pred == 1 else ('BEAR' if lor_pred == -1 else 'FLAT')} ({lor_conf*100:.0f}%)",
            "assigned_strategy": asset_strat.upper(),
            "recommendation": rec,
            "action": action,
            "sl": setup_sl,
            "tp1": setup_tp1,
            "tp2": setup_tp2,
            "lot": 0.01
        })
        
    await provider.disconnect()
    
    # Print clean formatted summary
    for r in analysis_reports:
        print(f"\n────────────────────────────────────────────────────────────────────────")
        print(f" Asset: {r['symbol']:<10} | Current MT5 Price: {r['price']}")
        print(f" Assigned Edge: {r['assigned_strategy']} | Macro H1 Trend: {r['macro_desc']}")
        print(f" Technicals: RSI={r['rsi']:.1f} | {r['wt']} | UT Bot: {r['ut_bot']}")
        print(f" Nadaraya-Watson Envelope: {r['nw_bands']}")
        print(f" Lorentzian Machine Learning: {r['lorentzian']}")
        print(f" Plan Recommendation: >>> {r['recommendation']} <<<")
        if r['action'] != "NEUTRAL" and r['sl'] > 0:
            print(f"   • Order Type: {r['action']} @ Market / Limit ({r['price']})")
            print(f"   • Suggested Volume: {r['lot']} Lots")
            print(f"   • Stop Loss (SL): {r['sl']:.5f}")
            print(f"   • Take Profit 1 (1R Breakeven Lock): {r['tp1']:.5f}")
            print(f"   • Take Profit 2 (Target / Liquidity): {r['tp2']:.5f}")
        else:
            print("   • Status: Consolidating inside envelope. Waiting for key level liquidity tap.")
            
    print("────────────────────────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_live_market_analysis())
