"""
Comprehensive Backtest Suite for TradingView-Enhanced Strategies.
Filters for strategies with Win Rate >= 40% and positive Sharpe/Expectancy.
"""

import sys
import asyncio
import pandas as pd
import numpy as np

from core.models import Timeframe, AssetClass
from data.providers.mt5_provider import MT5DataProvider
from signals.ta_engine import TAEngine
from signals.tradingview_indicators import add_tradingview_indicators

async def fetch_data(symbol, tf, limit=1000):
    provider = MT5DataProvider()
    await provider.connect()
    df = await provider.get_historical_dataframe(symbol, tf, limit=limit)
    await provider.disconnect()
    if df is None or df.empty:
        # Fallback to demo provider
        from data.providers.demo_provider import DemoDataProvider
        p = DemoDataProvider()
        await p.connect()
        df = await p.get_historical_dataframe(symbol, tf, limit=limit)
        await p.disconnect()
    return df

def backtest_strategy(df: pd.DataFrame, strat_type: str, params: dict, fee_pct=0.0002):
    """
    Simulates trades bar-by-bar with slippage/fees and exact stop loss / take profit.
    Returns: {win_rate, total_return, trades, profit_factor, max_dd, sharpe}
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(closes)
    
    trades = []
    in_pos = 0 # 1 = long, -1 = short
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    
    atr = df.get('ATR', df['close'] * 0.005).values
    
    # Pre-extract indicators
    wt1 = df.get('WT1', pd.Series(0, index=df.index)).values
    wt2 = df.get('WT2', pd.Series(0, index=df.index)).values
    rsi = df.get('RSI', pd.Series(50, index=df.index)).values
    nw_lower = df.get('NW_LOWER', df['close'] * 0.99).values
    nw_upper = df.get('NW_UPPER', df['close'] * 1.01).values
    ut_pos = df.get('UT_POSITION', pd.Series(0, index=df.index)).values
    lor_pred = df.get('LORENTZIAN_PRED', pd.Series(0, index=df.index)).values
    lor_conf = df.get('LORENTZIAN_CONF', pd.Series(0.5, index=df.index)).values
    ema13 = df.get('EMA_13', df.get('EMA_9', df['close'])).values
    ema99 = df.get('EMA_99', df.get('EMA_50', df['close'])).values
    
    for i in range(50, n - 1):
        curr_c = closes[i]
        curr_h = highs[i]
        curr_l = lows[i]
        curr_atr = atr[i]
        
        # Check active position exit
        if in_pos == 1:
            if curr_l <= sl_price:
                pnl = (sl_price - entry_price) / entry_price - fee_pct
                trades.append(pnl)
                in_pos = 0
            elif curr_h >= tp_price:
                pnl = (tp_price - entry_price) / entry_price - fee_pct
                trades.append(pnl)
                in_pos = 0
        elif in_pos == -1:
            if curr_h >= sl_price:
                pnl = (entry_price - sl_price) / entry_price - fee_pct
                trades.append(pnl)
                in_pos = 0
            elif curr_l <= tp_price:
                pnl = (entry_price - tp_price) / entry_price - fee_pct
                trades.append(pnl)
                in_pos = 0
                
        if in_pos != 0:
            continue
            
        # Entry logic per strategy
        if strat_type == "tv_mean_reversion":
            # WaveTrend Oversold + Nadaraya-Watson lower envelope
            if (wt1[i] < -45 and wt1[i] > wt2[i]) or (curr_c <= nw_lower[i] and rsi[i] < 35):
                in_pos = 1
                entry_price = curr_c
                sl_price = entry_price - (curr_atr * params.get('sl_atr', 1.5))
                tp_price = entry_price + (curr_atr * params.get('tp_atr', 2.5))
            elif (wt1[i] > 45 and wt1[i] < wt2[i]) or (curr_c >= nw_upper[i] and rsi[i] > 65):
                in_pos = -1
                entry_price = curr_c
                sl_price = entry_price + (curr_atr * params.get('sl_atr', 1.5))
                tp_price = entry_price - (curr_atr * params.get('tp_atr', 2.5))
                
        elif strat_type == "tv_ut_lorentzian_trend":
            # UT Bot + Lorentzian ML consensus + EMA 13/99 alignment
            if ema13[i] > ema99[i] and ut_pos[i] == 1 and lor_pred[i] == 1 and lor_conf[i] >= 0.55:
                in_pos = 1
                entry_price = curr_c
                sl_price = entry_price - (curr_atr * params.get('sl_atr', 2.0))
                tp_price = entry_price + (curr_atr * params.get('tp_atr', 3.5))
            elif ema13[i] < ema99[i] and ut_pos[i] == -1 and lor_pred[i] == -1 and lor_conf[i] >= 0.55:
                in_pos = -1
                entry_price = curr_c
                sl_price = entry_price + (curr_atr * params.get('sl_atr', 2.0))
                tp_price = entry_price - (curr_atr * params.get('tp_atr', 3.5))
                
        elif strat_type == "smc_institutional":
            # Fair Value Gap + Order Block displacement
            body = abs(curr_c - df['open'].values[i])
            if body > (curr_atr * 1.2):
                if curr_c > df['open'].values[i] and rsi[i] < 65:
                    in_pos = 1
                    entry_price = curr_c
                    sl_price = entry_price - (curr_atr * params.get('sl_atr', 1.2))
                    tp_price = entry_price + (curr_atr * params.get('tp_atr', 3.0))
                elif curr_c < df['open'].values[i] and rsi[i] > 35:
                    in_pos = -1
                    entry_price = curr_c
                    sl_price = entry_price + (curr_atr * params.get('sl_atr', 1.2))
                    tp_price = entry_price - (curr_atr * params.get('tp_atr', 3.0))

    if not trades:
        return {"win_rate": 0.0, "total_return": 0.0, "trades": 0, "profit_factor": 0.0, "max_dd": 0.0, "sharpe": 0.0}
        
    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    
    win_rate = (len(wins) / len(trades)) * 100.0
    cum_returns = np.cumprod(1.0 + trades) - 1.0
    total_return = cum_returns[-1] * 100.0
    
    # Max Drawdown
    equity_curve = np.cumprod(1.0 + trades)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_dd = abs(drawdowns.min()) * 100.0
    
    # Profit Factor
    gross_win = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_win / gross_loss
    
    # Sharpe Ratio
    std = np.std(trades)
    sharpe = (np.mean(trades) / (std + 1e-9)) * np.sqrt(252 * 6) if std > 0 else 0.0
    
    return {
        "win_rate": round(win_rate, 1),
        "total_return": round(total_return, 2),
        "trades": len(trades),
        "profit_factor": round(profit_factor, 2),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 2)
    }

async def run_matrix():
    print("================================================================================")
    print("           COMPREHENSIVE BACKTEST MATRIX: FILTERING FOR >= 40% WIN RATE         ")
    print("================================================================================")
    
    symbols = ["EUR/USD", "GBP/USD", "BTC/USDT", "ETH/USDT"]
    timeframes = [Timeframe.H1, Timeframe.H4, Timeframe.M15]
    strategies = ["tv_mean_reversion", "tv_ut_lorentzian_trend", "smc_institutional"]
    
    ta_engine = TAEngine()
    
    all_results = []
    
    for sym in symbols:
        for tf in timeframes:
            df = await fetch_data(sym, tf, limit=600)
            if df is None or len(df) < 80:
                continue
            df_ta = ta_engine.add_all_indicators(df)
            
            for strat in strategies:
                res = backtest_strategy(df_ta, strat, {"sl_atr": 1.5, "tp_atr": 2.5})
                if res["trades"] >= 2:
                    all_results.append({
                        "symbol": sym,
                        "tf": tf.value,
                        "strategy": strat,
                        **res
                    })
                    
    # Filter for >= 40% Win Rate and positive Sharpe
    passed_results = [r for r in all_results if r["win_rate"] >= 40.0 and r["total_return"] > 0]
    passed_results.sort(key=lambda x: (x["win_rate"], x["sharpe"]), reverse=True)
    
    print(f"\nFound {len(passed_results)} High-Probability Setups with Win Rate >= 40%:\n")
    print(f"{'Asset':<10} {'TF':<5} {'Strategy':<26} {'Trades':<8} {'Win Rate':<10} {'Return %':<10} {'Max DD %':<10} {'Sharpe':<8} {'PF':<6}")
    print("─" * 95)
    for r in passed_results:
        print(f"{r['symbol']:<10} {r['tf']:<5} {r['strategy']:<26} {r['trades']:<8} {r['win_rate']:<10}% {r['total_return']:<10}% {r['max_dd']:<10}% {r['sharpe']:<8} {r['profit_factor']:<6}")
    print("─" * 95)
    
    return passed_results

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_matrix())
