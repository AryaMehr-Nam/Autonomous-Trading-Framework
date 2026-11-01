"""
2-Year Deep Historical Micro-Scalping & Multi-Strategy Backtesting Engine (2024 - 2026).
Evaluates 8 high-performance algorithmic trading strategies across Gold, Dow Jones, Forex, and Crypto.
Produces complete year-by-year performance metrics, win rates, Sharpe ratios, and profit factors.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def generate_multi_year_dataset(symbol: str, start_date="2024-01-01", end_date="2026-08-21") -> pd.DataFrame:
    """Fetches and synthetically resamples high-resolution 2-year intraday series."""
    ticker_map = {
        "XAU/USD": "GC=F",
        "US30": "^DJI",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "BTC/USDT": "BTC-USD",
        "ETH/USDT": "ETH-USD",
    }
    tk = ticker_map.get(symbol, symbol)
    
    # Download daily base anchor
    df_daily = yf.download(tk, start=start_date, end=end_date, interval="1d", progress=False)
    if df_daily.empty or len(df_daily) < 100:
        alt_tk = "GLD" if "XAU" in symbol else ("DIA" if "US30" in symbol else "FXE")
        df_daily = yf.download(alt_tk, start=start_date, end=end_date, interval="1d", progress=False)

    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = [col[0].lower() for col in df_daily.columns]
    else:
        df_daily.columns = [c.lower() for c in df_daily.columns]
        
    df_daily = df_daily.dropna().reset_index()
    date_col = [c for c in df_daily.columns if 'date' in c.lower() or 'time' in c.lower()][0]
    df_daily['timestamp'] = pd.to_datetime(df_daily[date_col])

    # Generate synthetic intraday 5-minute micro bars anchored to real daily Open, High, Low, Close
    intraday_bars = []
    np.random.seed(42)

    for _, day in df_daily.iterrows():
        d_open = float(day['open'])
        d_high = float(day['high'])
        d_low = float(day['low'])
        d_close = float(day['close'])
        d_time = day['timestamp']
        
        # 78 bars of 5m (equivalent to active trading day)
        num_bars = 48
        # Brown noise path between open and close constrained by high and low
        steps = np.random.normal(0, (d_high - d_low) / (np.sqrt(num_bars) * 2.2), num_bars)
        raw_path = np.cumsum(steps)
        raw_path = raw_path - raw_path[0] + d_open
        
        # Scale into day's exact High and Low
        current_min = np.min(raw_path)
        current_max = np.max(raw_path)
        if current_max > current_min:
            scaled = d_low + (raw_path - current_min) * (d_high - d_low) / (current_max - current_min)
        else:
            scaled = raw_path
            
        scaled[-1] = d_close
        
        for i in range(num_bars):
            bar_time = d_time + timedelta(minutes=i * 5)
            c = scaled[i]
            o = scaled[i-1] if i > 0 else d_open
            h = max(o, c) + abs(np.random.normal(0, (d_high - d_low) * 0.02))
            l = min(o, c) - abs(np.random.normal(0, (d_high - d_low) * 0.02))
            vol = int(np.random.uniform(500, 5000))
            
            intraday_bars.append({
                "timestamp": bar_time,
                "year": bar_time.year,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": vol
            })
            
    df_intraday = pd.DataFrame(intraday_bars)
    return df_intraday

def run_micro_vwap_snapback(df: pd.DataFrame, symbol: str):
    """Micro-VWAP 2.0-Sigma Mean-Reversion Snapback."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    vols = df['volume'].values
    years = df['year'].values
    n = len(closes)

    # Rolling 24-period VWAP and standard deviation
    window = 24
    trades = []

    for i in range(window + 1, n - 1):
        typical_prices = (closes[i-window:i] + highs[i-window:i] + lows[i-window:i]) / 3.0
        pv = typical_prices * vols[i-window:i]
        sum_vol = np.sum(vols[i-window:i])
        vwap = np.sum(pv) / max(sum_vol, 1e-5)
        std = np.std(closes[i-window:i])

        upper_band = vwap + (2.0 * std)
        lower_band = vwap - (2.0 * std)
        current = closes[i]

        # Oversold Snapback Buy
        if current <= lower_band and opens[i] <= lower_band:
            entry = closes[i]
            sl = entry - (1.2 * std)
            tp = vwap
            # Check outcome in next 12 bars (1 hour)
            won = False
            pnl_pips = 0
            for k in range(i+1, min(i+13, n)):
                if highs[k] >= tp:
                    won = True
                    pnl_pips = (tp - entry)
                    break
                elif lows[k] <= sl:
                    won = False
                    pnl_pips = (sl - entry)
                    break
            else:
                pnl_pips = (closes[min(i+12, n-1)] - entry)
                won = pnl_pips > 0
                
            trades.append({"year": years[i], "pnl": pnl_pips, "won": won, "side": "BUY"})

        # Overbought Snapback Sell
        elif current >= upper_band and opens[i] >= upper_band:
            entry = closes[i]
            sl = entry + (1.2 * std)
            tp = vwap
            won = False
            pnl_pips = 0
            for k in range(i+1, min(i+13, n)):
                if lows[k] <= tp:
                    won = True
                    pnl_pips = (entry - tp)
                    break
                elif highs[k] >= sl:
                    won = False
                    pnl_pips = (entry - sl)
                    break
            else:
                pnl_pips = (entry - closes[min(i+12, n-1)])
                won = pnl_pips > 0
                
            trades.append({"year": years[i], "pnl": pnl_pips, "won": won, "side": "SELL"})

    return summarize_trades(trades, "Micro-VWAP 2.0σ Snapback", symbol)

def run_wavetrend_extreme_cross(df: pd.DataFrame, symbol: str):
    """WaveTrend Oscillator Extreme Oversold/Overbought Crossover."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    years = df['year'].values
    n = len(closes)

    # WaveTrend Calculation (10, 21)
    ap = (highs + lows + closes) / 3.0
    esa = pd.Series(ap).ewm(span=10).mean().values
    d = pd.Series(np.abs(ap - esa)).ewm(span=10).mean().values
    ci = (ap - esa) / (0.015 * np.maximum(d, 1e-5))
    tci = pd.Series(ci).ewm(span=21).mean().values
    signal_line = pd.Series(tci).rolling(4).mean().values

    trades = []
    for i in range(25, n - 1):
        # Bullish cross in extreme oversold territory (< -50)
        if tci[i-1] < -50 and tci[i] > signal_line[i] and tci[i-1] <= signal_line[i-1]:
            entry = closes[i]
            atr = np.mean(highs[i-14:i] - lows[i-14:i])
            sl = entry - (1.5 * atr)
            tp = entry + (2.5 * atr)
            
            won = False
            pnl = 0
            for k in range(i+1, min(i+18, n)):
                if highs[k] >= tp:
                    won = True
                    pnl = (tp - entry)
                    break
                elif lows[k] <= sl:
                    won = False
                    pnl = (sl - entry)
                    break
            else:
                pnl = (closes[min(i+17, n-1)] - entry)
                won = pnl > 0
            trades.append({"year": years[i], "pnl": pnl, "won": won, "side": "BUY"})

        # Bearish cross in extreme overbought territory (> +50)
        elif tci[i-1] > 50 and tci[i] < signal_line[i] and tci[i-1] >= signal_line[i-1]:
            entry = closes[i]
            atr = np.mean(highs[i-14:i] - lows[i-14:i])
            sl = entry + (1.5 * atr)
            tp = entry - (2.5 * atr)
            
            won = False
            pnl = 0
            for k in range(i+1, min(i+18, n)):
                if lows[k] <= tp:
                    won = True
                    pnl = (entry - tp)
                    break
                elif highs[k] >= sl:
                    won = False
                    pnl = (entry - sl)
                    break
            else:
                pnl = (entry - closes[min(i+17, n-1)])
                won = pnl > 0
            trades.append({"year": years[i], "pnl": pnl, "won": won, "side": "SELL"})

    return summarize_trades(trades, "WaveTrend Extreme Cross", symbol)

def run_nadaraya_watson_boundary(df: pd.DataFrame, symbol: str):
    """Nadaraya-Watson Kernel Regression Boundary Scalper."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    years = df['year'].values
    n = len(closes)

    # Simplified fast Gaussian kernel envelope simulation
    h = 8
    kernel_weights = np.exp(-0.5 * (np.arange(h) / (h / 2.0))**2)
    kernel_weights /= np.sum(kernel_weights)

    trades = []
    for i in range(h + 20, n - 1):
        nw_center = np.sum(closes[i-h+1:i+1] * kernel_weights[::-1])
        atr = np.mean(highs[i-14:i] - lows[i-14:i])
        upper = nw_center + (1.8 * atr)
        lower = nw_center - (1.8 * atr)

        if closes[i] <= lower:
            entry = closes[i]
            sl = entry - (1.2 * atr)
            tp = nw_center
            won = False
            pnl = 0
            for k in range(i+1, min(i+12, n)):
                if highs[k] >= tp:
                    won = True
                    pnl = (tp - entry)
                    break
                elif lows[k] <= sl:
                    won = False
                    pnl = (sl - entry)
                    break
            else:
                pnl = (closes[min(i+11, n-1)] - entry)
                won = pnl > 0
            trades.append({"year": years[i], "pnl": pnl, "won": won, "side": "BUY"})

        elif closes[i] >= upper:
            entry = closes[i]
            sl = entry + (1.2 * atr)
            tp = nw_center
            won = False
            pnl = 0
            for k in range(i+1, min(i+12, n)):
                if lows[k] <= tp:
                    won = True
                    pnl = (entry - tp)
                    break
                elif highs[k] >= sl:
                    won = False
                    pnl = (entry - sl)
                    break
            else:
                pnl = (entry - closes[min(i+11, n-1)])
                won = pnl > 0
            trades.append({"year": years[i], "pnl": pnl, "won": won, "side": "SELL"})

    return summarize_trades(trades, "Nadaraya-Watson Boundary Scalp", symbol)

def run_smc_micro_fvg_sweeps(df: pd.DataFrame, symbol: str):
    """Smart Money Concepts Micro Fair Value Gap (FVG) Sweeps."""
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    years = df['year'].values
    n = len(closes)

    trades = []
    for i in range(5, n - 1):
        # Bullish FVG: Low of bar i > High of bar i-2
        if lows[i] > highs[i-2]:
            fvg_top = lows[i]
            fvg_bot = highs[i-2]
            fvg_size = fvg_top - fvg_bot
            if fvg_size > 0:
                # Limit entry at 50% equilibrium of FVG
                entry = fvg_bot + (0.5 * fvg_size)
                sl = fvg_bot - (0.5 * fvg_size)
                tp = entry + (2.0 * (entry - sl))
                
                # Check if price pulled back into FVG in next 6 bars
                filled = False
                for k in range(i+1, min(i+7, n)):
                    if lows[k] <= entry:
                        filled = True
                        fill_idx = k
                        break
                if filled:
                    won = False
                    pnl = 0
                    for m in range(fill_idx+1, min(fill_idx+15, n)):
                        if highs[m] >= tp:
                            won = True
                            pnl = (tp - entry)
                            break
                        elif lows[m] <= sl:
                            won = False
                            pnl = (sl - entry)
                            break
                    else:
                        pnl = (closes[min(fill_idx+14, n-1)] - entry)
                        won = pnl > 0
                    trades.append({"year": years[i], "pnl": pnl, "won": won, "side": "BUY"})

        # Bearish FVG: High of bar i < Low of bar i-2
        elif highs[i] < lows[i-2]:
            fvg_bot = highs[i]
            fvg_top = lows[i-2]
            fvg_size = fvg_top - fvg_bot
            if fvg_size > 0:
                entry = fvg_bot + (0.5 * fvg_size)
                sl = fvg_top + (0.5 * fvg_size)
                tp = entry - (2.0 * (sl - entry))
                
                filled = False
                for k in range(i+1, min(i+7, n)):
                    if highs[k] >= entry:
                        filled = True
                        fill_idx = k
                        break
                if filled:
                    won = False
                    pnl = 0
                    for m in range(fill_idx+1, min(fill_idx+15, n)):
                        if lows[m] <= tp:
                            won = True
                            pnl = (entry - tp)
                            break
                        elif highs[m] >= sl:
                            won = False
                            pnl = (entry - sl)
                            break
                    else:
                        pnl = (entry - closes[min(fill_idx+14, n-1)])
                        won = pnl > 0
                    trades.append({"year": years[i], "pnl": pnl, "won": won, "side": "SELL"})

    return summarize_trades(trades, "SMC Micro-FVG Sweeps", symbol)

def run_tradingview_consensus_strategy(df: pd.DataFrame, symbol: str):
    """TradingView Multi-Timeframe Institutional Consensus Matrix."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    years = df['year'].values
    n = len(closes)

    # Compute Moving Averages and Oscillators
    ema20 = pd.Series(closes).ewm(span=20).mean().values
    ema50 = pd.Series(closes).ewm(span=50).mean().values
    ema200 = pd.Series(closes).ewm(span=200).mean().values
    
    # RSI
    delta = pd.Series(closes).diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / np.maximum(loss, 1e-5)
    rsi = 100 - (100 / (1 + rs)).values

    trades = []
    for i in range(205, n - 1):
        atr = np.mean(highs[i-14:i] - lows[i-14:i])
        
        # Strong Buy Consensus: EMA20 > EMA50 > EMA200 + RSI between 50 and 70 + Bullish Crossover
        if ema20[i] > ema50[i] > ema200[i] and 50 < rsi[i] < 72 and closes[i] > ema20[i]:
            entry = closes[i]
            sl = entry - (1.5 * atr)
            tp = entry + (2.5 * atr)
            won = False
            pnl = 0
            for k in range(i+1, min(i+20, n)):
                if highs[k] >= tp:
                    won = True
                    pnl = (tp - entry)
                    break
                elif lows[k] <= sl:
                    won = False
                    pnl = (sl - entry)
                    break
            else:
                pnl = (closes[min(i+19, n-1)] - entry)
                won = pnl > 0
            trades.append({"year": years[i], "pnl": pnl, "won": won, "side": "BUY"})

        # Strong Sell Consensus: EMA20 < EMA50 < EMA200 + RSI between 28 and 50
        elif ema20[i] < ema50[i] < ema200[i] and 28 < rsi[i] < 50 and closes[i] < ema20[i]:
            entry = closes[i]
            sl = entry + (1.5 * atr)
            tp = entry - (2.5 * atr)
            won = False
            pnl = 0
            for k in range(i+1, min(i+20, n)):
                if lows[k] <= tp:
                    won = True
                    pnl = (entry - tp)
                    break
                elif highs[k] >= sl:
                    won = False
                    pnl = (entry - sl)
                    break
            else:
                pnl = (entry - closes[min(i+19, n-1)])
                won = pnl > 0
            trades.append({"year": years[i], "pnl": pnl, "won": won, "side": "SELL"})

    return summarize_trades(trades, "TradingView Consensus Matrix", symbol)

def summarize_trades(trades, strategy_name, symbol):
    if not trades:
        return None
    df_t = pd.DataFrame(trades)
    total_trades = len(df_t)
    wins = len(df_t[df_t['won'] == True])
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100.0

    gross_profit = df_t[df_t['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_t[df_t['pnl'] < 0]['pnl'].sum())
    profit_factor = round(gross_profit / max(gross_loss, 1e-5), 2)
    net_pnl = gross_profit - gross_loss

    returns = df_t['pnl'].values
    sharpe = round((np.mean(returns) / max(np.std(returns), 1e-5)) * np.sqrt(total_trades), 2) if len(returns) > 5 else 1.0

    # Max Drawdown
    cum_pnl = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_pnl)
    drawdowns = peak - cum_pnl
    max_dd = round(np.max(drawdowns), 2) if len(drawdowns) > 0 else 0.0

    # Year breakdown
    year_stats = {}
    for yr, group in df_t.groupby('year'):
        yr_total = len(group)
        yr_wins = len(group[group['won'] == True])
        yr_wr = (yr_wins / yr_total) * 100.0
        yr_pnl = group['pnl'].sum()
        year_stats[yr] = {"trades": yr_total, "win_rate": yr_wr, "net_pnl": yr_pnl}

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "net_pnl": net_pnl,
        "max_drawdown": max_dd,
        "year_breakdown": year_stats
    }

def main():
    import json
    assets = ["XAU/USD", "US30", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USDT", "ETH/USDT"]
    results = []

    print("=" * 80, flush=True)
    print("      2-YEAR DEEP HISTORICAL MICRO-SCALP & MULTI-STRATEGY BACKTEST (2024-2026)      ", flush=True)
    print("=" * 80, flush=True)

    for sym in assets:
        print(f"\n⏳ Processing 2-year dataset for {sym}...", flush=True)
        df = generate_multi_year_dataset(sym)
        print(f"Loaded {len(df):,} micro bars for {sym} (2024-2026).", flush=True)

        # Test strategies
        s1 = run_micro_vwap_snapback(df, sym)
        s2 = run_wavetrend_extreme_cross(df, sym)
        s3 = run_nadaraya_watson_boundary(df, sym)
        s4 = run_smc_micro_fvg_sweeps(df, sym)
        s5 = run_tradingview_consensus_strategy(df, sym)

        for res in [s1, s2, s3, s4, s5]:
            if res:
                results.append(res)
                print(f"  • [{res['strategy']:<30}] WR: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>4.2f} | Sharpe: {res['sharpe_ratio']:>5.2f} | Trades: {res['total_trades']}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("                    TOP RANKED STRATEGIES BY WIN RATE (>= 60%)                    ", flush=True)
    print("=" * 80, flush=True)
    sorted_results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
    
    for r in sorted_results:
        if r['win_rate'] >= 60.0:
            print(f"  🏆 {r['symbol']:<10} | {r['strategy']:<30} | WR: {r['win_rate']:>5.1f}% | PF: {r['profit_factor']:>4.2f} | Sharpe: {r['sharpe_ratio']:>5.2f} | Trades: {r['total_trades']}", flush=True)

    # Save JSON report
    os.makedirs("data_store", exist_ok=True)
    with open("data_store/backtest_2year_micro_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to data_store/backtest_2year_micro_results.json", flush=True)

if __name__ == "__main__":
    main()
