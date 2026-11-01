"""
Ultra-Short Sub-Minute (30-Second) Scalper Backtest Simulator.
Simulates real tick feeds, tick velocity acceleration, micro-VWAP bands, and time-stop exits.
"""

import sys
import asyncio
import numpy as np
import pandas as pd

from signals.micro_scalper import MicroTickScalper
from data.providers.mt5_provider import MT5DataProvider
from data.providers.demo_provider import DemoDataProvider
from core.models import Timeframe, SignalType

async def run_30s_scalper_simulation():
    print("================================================================================")
    print("        ULTRA-SHORT 30-SECOND MICRO-TICK SCALPING BACKTEST SIMULATOR            ")
    print("================================================================================")
    
    # 1. Fetch granular M1 candle data to synthesize realistic tick streams
    provider = MT5DataProvider()
    await provider.connect()
    df_m1 = await provider.get_historical_dataframe("EUR/USD", Timeframe.M1, limit=500)
    await provider.disconnect()
    
    if df_m1 is None or len(df_m1) < 50:
        demo = DemoDataProvider()
        await demo.connect()
        df_m1 = await demo.get_historical_dataframe("EUR/USD", Timeframe.M1, limit=500)
        await demo.disconnect()
        
    print(f"Loaded {len(df_m1)} M1 bars for EUR/USD. Synthesizing sub-minute tick stream...")
    
    # Synthesize realistic sub-minute tick series (10 ticks per M1 bar = 5,000 sub-minute ticks)
    np.random.seed(42)
    ticks = []
    base_time = 1770000000.0
    pip = 0.0001
    spread = 0.00005 # 0.5 pips raw spread
    
    for idx, row in df_m1.iterrows():
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        vol = int(max(5, row.get('volume', 10)))
        
        # Intra-bar trajectory with realistic volatility bursts
        micro_path = np.linspace(o, c, vol) + np.random.normal(0, pip * 0.4, vol)
        if len(micro_path) > 4:
            micro_path[2] = min(l, micro_path[2])
            micro_path[-3] = max(h, micro_path[-3])
        
        # Micro burst: rapid ticks during volatile wicks, slower during consolidation
        is_burst = abs(c - o) > (pip * 2.0)
        
        for p in micro_path:
            tick_dt = np.random.uniform(0.1, 0.8) if is_burst else np.random.uniform(1.5, 4.0)
            base_time += tick_dt
            ticks.append({
                "timestamp": base_time,
                "bid": p - (spread / 2.0),
                "ask": p + (spread / 2.0),
                "volume": float(np.random.randint(2, 12)) if is_burst else 1.0
            })
            
    print(f"Generated {len(ticks)} live micro-ticks. Running 30-second Micro-Tick Scalper...")
    
    scalper = MicroTickScalper(
        symbol="EUR/USD",
        window_seconds=30,
        velocity_threshold=1.8,
        micro_band_std=2.0,
        target_pips=4.0,   # 4 pips target
        stop_pips=3.0,     # 3 pips stop
        max_allowed_spread_pips=0.8,
        time_stop_seconds=45
    )
    
    trades = []
    in_pos = 0 # 1 = BUY, -1 = SELL
    entry_price = 0.0
    entry_time = 0.0
    sl_price = 0.0
    tp_price = 0.0
    time_stop_sec = 45
    trade_durations = []
    
    for i, t in enumerate(ticks):
        scalper.add_tick(bid=t['bid'], ask=t['ask'], timestamp=t['timestamp'], volume=t['volume'])
        
        curr_bid = t['bid']
        curr_ask = t['ask']
        curr_ts = t['timestamp']
        
        # Position Exit Check
        if in_pos == 1:
            duration = curr_ts - entry_time
            if curr_bid <= sl_price: # SL hit
                pnl = (sl_price - entry_price) / pip
                trades.append(pnl)
                trade_durations.append(duration)
                in_pos = 0
            elif curr_bid >= tp_price: # TP hit
                pnl = (tp_price - entry_price) / pip
                trades.append(pnl)
                trade_durations.append(duration)
                in_pos = 0
            elif duration >= time_stop_sec: # Time-Stop Invalidation (45s timeout)
                pnl = (curr_bid - entry_price) / pip
                trades.append(pnl)
                trade_durations.append(duration)
                in_pos = 0
        elif in_pos == -1:
            duration = curr_ts - entry_time
            if curr_ask >= sl_price: # SL hit
                pnl = (entry_price - sl_price) / pip
                trades.append(pnl)
                trade_durations.append(duration)
                in_pos = 0
            elif curr_ask <= tp_price: # TP hit
                pnl = (entry_price - tp_price) / pip
                trades.append(pnl)
                trade_durations.append(duration)
                in_pos = 0
            elif duration >= time_stop_sec: # Time-Stop Invalidation
                pnl = (entry_price - curr_ask) / pip
                trades.append(pnl)
                trade_durations.append(duration)
                in_pos = 0
                
        if in_pos != 0:
            continue
            
        # Entry Evaluation with Macro Bias Filter
        sig = scalper.evaluate_micro_signal(macro_bias=1)
        if sig:
            if sig.signal_type == SignalType.BUY:
                in_pos = 1
                entry_price = curr_ask
                entry_time = curr_ts
                sl_price = sig.stop_loss
                tp_price = sig.take_profit
                time_stop_sec = sig.indicators.get("time_stop_sec", 45)
            elif sig.signal_type == SignalType.SELL:
                in_pos = -1
                entry_price = curr_bid
                entry_time = curr_ts
                sl_price = sig.stop_loss
                tp_price = sig.take_profit
                time_stop_sec = sig.indicators.get("time_stop_sec", 45)
                
    if not trades:
        print("No trades triggered.")
        return
        
    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    flats = trades[trades == 0]
    
    win_rate = (len(wins) / len(trades)) * 100.0
    net_pips = np.sum(trades)
    profit_factor = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and abs(np.sum(losses)) > 0 else 5.0
    avg_dur = np.mean(trade_durations)
    
    print("\n================================================================================")
    print("                     30-SECOND SCALPER PERFORMANCE REPORT                       ")
    print("================================================================================")
    print(f" Total Scalp Trades Taken:     {len(trades)}")
    print(f" Winning Trades:               {len(wins)} ({win_rate:.1f}% Win Rate)")
    print(f" Losing Trades:                {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
    print(f" Scratch / Flat Trades:        {len(flats)}")
    print(f" Net Pips Captured:            +{net_pips:.1f} Pips")
    print(f" Profit Factor:                {profit_factor:.2f}")
    print(f" Average Time in Trade:        {avg_dur:.1f} Seconds")
    print(f" Fastest Exit:                 {min(trade_durations):.1f} Seconds")
    print(f" Target Holding Window:        15s to 45s (Micro-VWAP Snapback)")
    print("================================================================================\n")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_30s_scalper_simulation())
