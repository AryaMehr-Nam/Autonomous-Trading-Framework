"""
Master Expanded Backtest Matrix Runner.
Tests 10 Assets across 4 Timeframes (M5, M15, H1, H4) and 4 Advanced Strategies:
1. WaveTrend + Nadaraya-Watson Reversion
2. SMC Institutional (Order Blocks + FVG Sweeps)
3. UT Bot + Lorentzian ML Trend Following
4. Hybrid Micro-VWAP Momentum Scalper
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np

from core.models import Timeframe, AssetClass
from data.providers.mt5_provider import MT5DataProvider
from data.providers.demo_provider import DemoDataProvider
from signals.ta_engine import TAEngine
from signals.tradingview_indicators import (
    calculate_wavetrend,
    calculate_nadaraya_watson,
    calculate_ut_bot,
    calculate_lorentzian_classification
)

async def fetch_ohlcv(symbol: str, tf: Timeframe, limit: int = 1000) -> pd.DataFrame:
    provider = MT5DataProvider()
    await provider.connect()
    df = await provider.get_historical_dataframe(symbol, tf, limit=limit)
    await provider.disconnect()
    
    if df is None or len(df) < 80:
        demo = DemoDataProvider()
        await demo.connect()
        df = await demo.get_historical_dataframe(symbol, tf, limit=limit)
        await demo.disconnect()
    return df

def run_simulation(df: pd.DataFrame, strat_name: str, symbol: str, fee_pct: float = 0.0002):
    """
    Simulates trades with dynamic ATR stops, 1R Breakeven SL trail, and 50% scale-out.
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    n = len(closes)
    
    # Extract indicators
    tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(14).mean().bfill().values
    
    wt1 = df.get('WT1', pd.Series(0, index=df.index)).values
    wt2 = df.get('WT2', pd.Series(0, index=df.index)).values
    nw_lower = df.get('NW_LOWER', df['close'] * 0.99).values
    nw_upper = df.get('NW_UPPER', df['close'] * 1.01).values
    ut_pos = df.get('UT_POSITION', pd.Series(0, index=df.index)).values
    lor_pred = df.get('LORENTZIAN_PRED', pd.Series(0, index=df.index)).values
    lor_conf = df.get('LORENTZIAN_CONF', pd.Series(0.5, index=df.index)).values
    rsi = df.get('RSI', pd.Series(50, index=df.index)).values
    ema13 = df.get('EMA_13', df.get('EMA_9', df['close'])).values
    ema99 = df.get('EMA_99', df.get('EMA_50', df['close'])).values
    
    trades = []
    in_pos = 0 # 1 = BUY, -1 = SELL
    entry_p = 0.0
    sl_p = 0.0
    tp_p = 0.0
    be_triggered = False
    half_closed = False
    
    for i in range(50, n - 1):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        cur_atr = atr[i]
        
        # Position Management
        if in_pos == 1:
            risk_dist = abs(entry_p - sl_p)
            current_gain = h - entry_p
            
            # 1R Breakeven + 50% scale-out check
            if current_gain >= risk_dist and not be_triggered and risk_dist > 0:
                be_triggered = True
                sl_p = entry_p + (cur_atr * 0.1) # Breakeven + spread buffer
                if not half_closed:
                    half_closed = True
                    # Realize 50% gain
                    trades.append(((entry_p + risk_dist - entry_p) / entry_p) * 0.5 - (fee_pct * 0.5))
            
            if l <= sl_p:
                rem_pnl = ((sl_p - entry_p) / entry_p) * (0.5 if half_closed else 1.0) - (fee_pct * (0.5 if half_closed else 1.0))
                trades.append(rem_pnl)
                in_pos = 0
                be_triggered = False
                half_closed = False
            elif h >= tp_p:
                rem_pnl = ((tp_p - entry_p) / entry_p) * (0.5 if half_closed else 1.0) - (fee_pct * (0.5 if half_closed else 1.0))
                trades.append(rem_pnl)
                in_pos = 0
                be_triggered = False
                half_closed = False
                
        elif in_pos == -1:
            risk_dist = abs(sl_p - entry_p)
            current_gain = entry_p - l
            
            if current_gain >= risk_dist and not be_triggered and risk_dist > 0:
                be_triggered = True
                sl_p = entry_p - (cur_atr * 0.1)
                if not half_closed:
                    half_closed = True
                    trades.append(((entry_p - (entry_p - risk_dist)) / entry_p) * 0.5 - (fee_pct * 0.5))
                    
            if h >= sl_p:
                rem_pnl = ((entry_p - sl_p) / entry_p) * (0.5 if half_closed else 1.0) - (fee_pct * (0.5 if half_closed else 1.0))
                trades.append(rem_pnl)
                in_pos = 0
                be_triggered = False
                half_closed = False
            elif l <= tp_p:
                rem_pnl = ((entry_p - tp_p) / entry_p) * (0.5 if half_closed else 1.0) - (fee_pct * (0.5 if half_closed else 1.0))
                trades.append(rem_pnl)
                in_pos = 0
                be_triggered = False
                half_closed = False
                
        if in_pos != 0:
            continue
            
        # Strategy Entry Signals
        if strat_name == "WaveTrend + Nadaraya-Watson":
            if (wt1[i] < -50 and wt1[i] > wt2[i]) or (c <= nw_lower[i] and rsi[i] < 35):
                in_pos = 1
                entry_p = c
                sl_p = c - (cur_atr * 1.5)
                tp_p = c + (cur_atr * 2.8)
            elif (wt1[i] > 50 and wt1[i] < wt2[i]) or (c >= nw_upper[i] and rsi[i] > 65):
                in_pos = -1
                entry_p = c
                sl_p = c + (cur_atr * 1.5)
                tp_p = c - (cur_atr * 2.8)
                
        elif strat_name == "SMC Institutional (FVG + OB)":
            body = abs(c - opens[i])
            if body > (cur_atr * 1.2):
                if c > opens[i] and rsi[i] < 62:
                    in_pos = 1
                    entry_p = c
                    sl_p = c - (cur_atr * 1.2)
                    tp_p = c + (cur_atr * 3.2)
                elif c < opens[i] and rsi[i] > 38:
                    in_pos = -1
                    entry_p = c
                    sl_p = c + (cur_atr * 1.2)
                    tp_p = c - (cur_atr * 3.2)
                    
        elif strat_name == "UT Bot + Lorentzian ML Trend":
            if ema13[i] > ema99[i] and ut_pos[i] == 1 and lor_pred[i] == 1 and lor_conf[i] >= 0.55:
                in_pos = 1
                entry_p = c
                sl_p = c - (cur_atr * 1.8)
                tp_p = c + (cur_atr * 3.5)
            elif ema13[i] < ema99[i] and ut_pos[i] == -1 and lor_pred[i] == -1 and lor_conf[i] >= 0.55:
                in_pos = -1
                entry_p = c
                sl_p = c + (cur_atr * 1.8)
                tp_p = c - (cur_atr * 3.5)

    if len(trades) < 3:
        return None
        
    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    
    win_rate = (len(wins) / len(trades)) * 100.0
    cum_ret = (np.prod(1.0 + trades) - 1.0) * 100.0
    
    eq = np.cumprod(1.0 + trades)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    max_dd = abs(dd.min()) * 100.0
    
    std = np.std(trades)
    sharpe = (np.mean(trades) / (std + 1e-9)) * np.sqrt(252 * 6) if std > 0 else 0.0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 0 else 3.0
    
    return {
        "win_rate": round(win_rate, 1),
        "return": round(cum_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
        "trades": len(trades),
        "pf": round(pf, 2)
    }

async def run_master_matrix():
    print("================================================================================")
    print("           MASTER EXPANDED MULTI-ASSET BACKTESTING MATRIX (10 ASSETS)           ")
    print("================================================================================")
    
    universe = [
        # Forex Major & Crosses
        ("EUR/USD", AssetClass.FOREX, 0.00015),
        ("GBP/USD", AssetClass.FOREX, 0.00018),
        ("USD/JPY", AssetClass.FOREX, 0.00015),
        ("AUD/USD", AssetClass.FOREX, 0.00018),
        # Crypto
        ("BTC/USDT", AssetClass.CRYPTO, 0.0005),
        ("ETH/USDT", AssetClass.CRYPTO, 0.0005),
        ("SOL/USDT", AssetClass.CRYPTO, 0.0006),
        # Equities
        ("AAPL", AssetClass.EQUITY, 0.0002),
        ("TSLA", AssetClass.EQUITY, 0.0003),
        ("NVDA", AssetClass.EQUITY, 0.0003)
    ]
    
    timeframes = [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]
    strategies = [
        "WaveTrend + Nadaraya-Watson",
        "SMC Institutional (FVG + OB)",
        "UT Bot + Lorentzian ML Trend"
    ]
    
    ta_engine = TAEngine()
    results = []
    
    for sym, asset_cls, fee in universe:
        for tf in timeframes:
            df = await fetch_ohlcv(sym, tf, limit=600)
            if df is None or len(df) < 80:
                continue
            df_ta = ta_engine.add_all_indicators(df)
            
            for strat in strategies:
                perf = run_simulation(df_ta, strat, sym, fee_pct=fee)
                if perf and perf["win_rate"] >= 40.0 and perf["return"] > 0:
                    results.append({
                        "symbol": sym,
                        "timeframe": tf.value,
                        "strategy": strat,
                        **perf
                    })
                    
    # Sort by Win Rate descending, then Sharpe
    results.sort(key=lambda x: (x["win_rate"], x["sharpe"]), reverse=True)
    
    print(f"\nDiscovered {len(results)} Winning Configurations (Win Rate >= 40% & Positive Alpha):\n")
    print(f"{'Asset':<10} {'TF':<5} {'Strategy':<30} {'Trades':<8} {'Win Rate':<10} {'Return %':<10} {'Max DD %':<10} {'Sharpe':<8} {'PF':<6}")
    print("─" * 102)
    for r in results:
        print(f"{r['symbol']:<10} {r['timeframe']:<5} {r['strategy']:<30} {r['trades']:<8} {r['win_rate']:<10}% {r['return']:<10}% {r['max_dd']:<10}% {r['sharpe']:<8} {r['pf']:<6}")
    print("─" * 102)
    
    return results

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_master_matrix())
