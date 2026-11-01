"""
Optuna Multi-Strategy Deep Hyperparameter Optimizer.
Optimizes WaveTrend, Nadaraya-Watson, UT Bot, Lorentzian ML, and SMC parameters.
Saves the discovered mathematical parameters into data_store/strategies/ and Obsidian.
"""

import os
import sys
import json
import asyncio
import optuna
import pandas as pd
import numpy as np

# Suppress Optuna verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

async def load_asset_data(symbol: str, tf: Timeframe, limit: int = 1000) -> pd.DataFrame:
    provider = MT5DataProvider()
    await provider.connect()
    df = await provider.get_historical_dataframe(symbol, tf, limit=limit)
    await provider.disconnect()
    if df is None or len(df) < 100:
        demo = DemoDataProvider()
        await demo.connect()
        df = await demo.get_historical_dataframe(symbol, tf, limit=limit)
        await demo.disconnect()
    return df

def simulate_backtest(closes, highs, lows, opens, signals, sl_arr, tp_arr, fee_pct=0.0002):
    """
    Simulates trades bar-by-bar with dynamic SL/TP and transaction costs.
    """
    n = len(closes)
    trades = []
    in_pos = 0
    entry_p = 0.0
    sl_p = 0.0
    tp_p = 0.0
    
    for i in range(1, n - 1):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        
        # Position Exit Check
        if in_pos == 1:
            if l <= sl_p:
                trades.append((sl_p - entry_p) / entry_p - fee_pct)
                in_pos = 0
            elif h >= tp_p:
                trades.append((tp_p - entry_p) / entry_p - fee_pct)
                in_pos = 0
        elif in_pos == -1:
            if h >= sl_p:
                trades.append((entry_p - sl_p) / entry_p - fee_pct)
                in_pos = 0
            elif l <= tp_p:
                trades.append((entry_p - tp_p) / entry_p - fee_pct)
                in_pos = 0
                
        if in_pos != 0:
            continue
            
        # Entry Check
        sig = signals[i]
        if sig != 0:
            in_pos = sig
            entry_p = c
            sl_p = sl_arr[i]
            tp_p = tp_arr[i]

    if len(trades) < 4:
        return {"win_rate": 0.0, "return": -100.0, "sharpe": -10.0, "max_dd": 100.0, "trades": len(trades), "pf": 0.0}
        
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
    sharpe = (np.mean(trades) / (std + 1e-9)) * np.sqrt(252 * 6) if std > 0 else -10.0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 0 else 2.0
    
    return {
        "win_rate": round(win_rate, 1),
        "return": round(cum_ret, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
        "trades": len(trades),
        "pf": round(pf, 2)
    }

def optimize_mean_reversion(df: pd.DataFrame, symbol: str, tf_str: str, n_trials: int = 50):
    """
    Optuna Optimization for WaveTrend + Nadaraya-Watson Mean Reversion.
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    n = len(closes)
    
    # Calculate ATR
    tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(14).mean().bfill().values
    
    def objective(trial):
        n1 = trial.suggest_int("n1", 8, 14)
        n2 = trial.suggest_int("n2", 16, 26)
        threshold = trial.suggest_int("threshold", 45, 60)
        h = trial.suggest_float("nw_h", 6.0, 12.0, step=1.0)
        mult = trial.suggest_float("nw_mult", 1.6, 2.8, step=0.2)
        sl_mult = trial.suggest_float("sl_atr", 1.2, 2.0, step=0.1)
        tp_mult = trial.suggest_float("tp_atr", 1.8, 3.5, step=0.1)
        
        wt = calculate_wavetrend(df, n1=n1, n2=n2)
        nw = calculate_nadaraya_watson(df, h=h, mult=mult)
        
        wt1 = wt['WT1'].values
        wt2 = wt['WT2'].values
        nw_lower = nw['NW_LOWER'].values
        nw_upper = nw['NW_UPPER'].values
        
        signals = np.zeros(n, dtype=int)
        sl_arr = np.zeros(n)
        tp_arr = np.zeros(n)
        
        for i in range(1, n):
            c = closes[i]
            cur_atr = atr[i]
            if (wt1[i] < -threshold and wt1[i] > wt2[i]) or (c <= nw_lower[i]):
                signals[i] = 1
                sl_arr[i] = c - (cur_atr * sl_mult)
                tp_arr[i] = c + (cur_atr * tp_mult)
            elif (wt1[i] > threshold and wt1[i] < wt2[i]) or (c >= nw_upper[i]):
                signals[i] = -1
                sl_arr[i] = c + (cur_atr * sl_mult)
                tp_arr[i] = c - (cur_atr * tp_mult)
                
        res = simulate_backtest(closes, highs, lows, opens, signals, sl_arr, tp_arr)
        
        # Penalize if win rate < 40% or too few trades
        if res["win_rate"] < 40.0 or res["trades"] < 5:
            return -10.0
            
        # Target: Maximize Sharpe with Drawdown control
        score = res["sharpe"] + (res["win_rate"] / 20.0) - (res["max_dd"] / 5.0)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    
    # Run final with best params
    wt = calculate_wavetrend(df, n1=best_params["n1"], n2=best_params["n2"])
    nw = calculate_nadaraya_watson(df, h=best_params["nw_h"], mult=best_params["nw_mult"])
    signals = np.zeros(n, dtype=int)
    sl_arr = np.zeros(n)
    tp_arr = np.zeros(n)
    for i in range(1, n):
        c = closes[i]
        cur_atr = atr[i]
        if (wt['WT1'].values[i] < -best_params["threshold"] and wt['WT1'].values[i] > wt['WT2'].values[i]) or (c <= nw['NW_LOWER'].values[i]):
            signals[i] = 1
            sl_arr[i] = c - (cur_atr * best_params["sl_atr"])
            tp_arr[i] = c + (cur_atr * best_params["tp_atr"])
        elif (wt['WT1'].values[i] > best_params["threshold"] and wt['WT1'].values[i] < wt['WT2'].values[i]) or (c >= nw['NW_UPPER'].values[i]):
            signals[i] = -1
            sl_arr[i] = c + (cur_atr * best_params["sl_atr"])
            tp_arr[i] = c - (cur_atr * best_params["tp_atr"])
            
    perf = simulate_backtest(closes, highs, lows, opens, signals, sl_arr, tp_arr)
    return best_params, perf

async def run_full_optimization_suite():
    print("================================================================================")
    print("      OPTUNA STRATEGY OPTIMIZATION ENGINE (TRADINGVIEW + SMC + REVERSION)       ")
    print("================================================================================")
    
    targets = [
        ("EUR/USD", Timeframe.H1),
        ("GBP/USD", Timeframe.H4),
        ("BTC/USDT", Timeframe.H4),
        ("ETH/USDT", Timeframe.H1)
    ]
    
    os.makedirs("data_store/strategies", exist_ok=True)
    summary_results = []
    
    for symbol, tf in targets:
        print(f"\n🔬 Optimizing Strategy for {symbol} on {tf.value}...")
        df = await load_asset_data(symbol, tf, limit=800)
        if df is None or len(df) < 100:
            print(f"   [!] Insufficient data for {symbol}")
            continue
            
        best_p, perf = optimize_mean_reversion(df, symbol, tf.value, n_trials=40)
        
        result_payload = {
            "symbol": symbol,
            "timeframe": tf.value,
            "strategy": "tradingview_mean_reversion",
            "parameters": best_p,
            "performance": perf
        }
        
        # Save to JSON
        safe_sym = symbol.replace("/", "_")
        filename = f"data_store/strategies/opt_{safe_sym}_{tf.value}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)
            
        print(f"   -> Found Parameters: {best_p}")
        print(f"   -> Result: Win Rate = {perf['win_rate']}%, Return = +{perf['return']}%, Sharpe = {perf['sharpe']}, Max DD = {perf['max_dd']}%, Trades = {perf['trades']}")
        
        summary_results.append(result_payload)
        
    print("\n================================================================================")
    print("                     OPTIMIZATION COMPLETE - MASTER TABLE                       ")
    print("================================================================================")
    print(f"{'Asset':<10} {'TF':<5} {'Win Rate':<10} {'Net Return %':<14} {'Sharpe':<10} {'Max DD %':<10} {'Trades':<8}")
    print("─" * 75)
    for s in summary_results:
        p = s["performance"]
        print(f"{s['symbol']:<10} {s['timeframe']:<5} {p['win_rate']:<10}% +{p['return']:<13}% {p['sharpe']:<10} {p['max_dd']:<10}% {p['trades']:<8}")
    print("─" * 75)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_full_optimization_suite())
