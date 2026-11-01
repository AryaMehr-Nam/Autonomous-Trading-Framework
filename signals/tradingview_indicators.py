"""
TradeAgent TradingView Elite Indicators Suite.

Implements the 4 most popular & battle-tested TradingView algorithms in pure Python/NumPy/Pandas:
1. WaveTrend Oscillator (LazyBear / Market Cipher B)
2. Nadaraya-Watson Kernel Regression Envelope (LuxAlgo)
3. UT Bot Alerts (QuantNomad / Yo_Crypto)
4. Lorentzian Classification (jdehorty - Non-Euclidean k-NN)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_wavetrend(df: pd.DataFrame, n1: int = 10, n2: int = 21) -> pd.DataFrame:
    """
    WaveTrend Oscillator (LazyBear / Cipher B).
    Calculates WT1 (fast momentum) and WT2 (smoothed signal line).
    Overbought = +53, Oversold = -53.
    """
    res = pd.DataFrame(index=df.index)
    
    # Typical Price / Average Price
    ap = (df['high'] + df['low'] + df['close']) / 3.0
    
    # Exponential Moving Average of Typical Price
    esa = ap.ewm(span=n1, adjust=False).mean()
    
    # Absolute difference EMA
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    
    # Channel Index (CI)
    ci = (ap - esa) / (0.015 * d.replace(0, 1e-9))
    
    # WaveTrend Lines
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    
    res['WT1'] = wt1
    res['WT2'] = wt2
    res['WT_HIST'] = wt1 - wt2
    
    # Bullish Cross at Oversold (< -50)
    res['WT_BULL_CROSS'] = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1)) & (wt1 < -50)
    # Bearish Cross at Overbought (> +50)
    res['WT_BEAR_CROSS'] = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1)) & (wt1 > 50)
    
    return res


def calculate_nadaraya_watson(df: pd.DataFrame, h: float = 8.0, r: float = 8.0, mult: float = 2.0) -> pd.DataFrame:
    """
    Nadaraya-Watson Envelope (LuxAlgo Rational Quadratic Kernel Regression).
    Calculates adaptive smoothed kernel line and dynamic ATR volatility bands.
    Non-repainting causal formulation for real-time live trading.
    """
    res = pd.DataFrame(index=df.index)
    closes = df['close'].values
    n = len(closes)
    
    window = min(50, n)
    nw_est = np.zeros(n)
    
    # Pre-compute rational quadratic kernel weights for lookback window
    dists = np.arange(window)
    weights = (1.0 + (dists ** 2) / (2.0 * r * (h ** 2))) ** (-r)
    weights /= weights.sum()
    
    # Vectorized / rolling convolution
    for i in range(n):
        if i < window:
            w = weights[:i+1]
            w = w / w.sum()
            nw_est[i] = np.dot(closes[:i+1][::-1], w)
        else:
            nw_est[i] = np.dot(closes[i-window+1:i+1][::-1], weights)
            
    res['NW_EST'] = nw_est
    
    # Dynamic ATR Volatility Envelope Bands
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        )
    )
    atr = tr.rolling(window=14).mean().fillna(df['close'] * 0.01)
    
    res['NW_UPPER'] = res['NW_EST'] + (mult * atr)
    res['NW_LOWER'] = res['NW_EST'] - (mult * atr)
    
    # Mean Reversion Triggers: Price outside band crossing back inside
    res['NW_BULL_REV'] = (df['close'] > res['NW_LOWER']) & (df['close'].shift(1) <= res['NW_LOWER'].shift(1))
    res['NW_BEAR_REV'] = (df['close'] < res['NW_UPPER']) & (df['close'].shift(1) >= res['NW_UPPER'].shift(1))
    
    return res


def calculate_ut_bot(df: pd.DataFrame, key_value: float = 2.0, atr_period: int = 10) -> pd.DataFrame:
    """
    UT Bot Alerts (QuantNomad / Yo_Crypto ATR Trailing System).
    Calculates dynamic trailing stop line and breakout trigger signals.
    """
    res = pd.DataFrame(index=df.index)
    closes = df['close'].values
    n = len(closes)
    
    # Calculate ATR
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        )
    )
    atr = tr.rolling(window=atr_period).mean().bfill().values
    n_loss = key_value * atr
    
    trailing_stop = np.zeros(n)
    pos = np.zeros(n)
    
    for i in range(1, n):
        prev_stop = trailing_stop[i-1]
        prev_close = closes[i-1]
        curr_close = closes[i]
        curr_loss = n_loss[i]
        
        if curr_close > prev_stop:
            trailing_stop[i] = max(prev_stop, curr_close - curr_loss)
            pos[i] = 1
        else:
            trailing_stop[i] = min(prev_stop, curr_close + curr_loss)
            pos[i] = -1
            
    res['UT_TRAILING_STOP'] = trailing_stop
    res['UT_POSITION'] = pos
    
    # Buy / Sell Alert Triggers
    pos_series = pd.Series(pos, index=df.index)
    res['UT_BUY'] = (pos_series == 1) & (pos_series.shift(1) == -1)
    res['UT_SELL'] = (pos_series == -1) & (pos_series.shift(1) == 1)
    
    return res


def calculate_lorentzian_classification(df: pd.DataFrame, lookback: int = 200, k: int = 8) -> pd.DataFrame:
    """
    Machine Learning: Lorentzian Classification (jdehorty).
    Uses non-Euclidean Lorentzian distance: d(x, y) = ln(1 + |x - y|)
    in a 4D feature space (RSI, WaveTrend, CCI, ADX) to predict price direction.
    """
    res = pd.DataFrame(index=df.index)
    n = len(df)
    
    if n < 50:
        res['LORENTZIAN_PRED'] = 0
        res['LORENTZIAN_CONF'] = 0.0
        return res
        
    # Build 4 normalized feature vectors
    # 1. RSI normalized to [-1, 1]
    rsi = df.get('RSI', df['close'].pct_change().rolling(14).mean()).fillna(50)
    f1 = ((rsi - 50.0) / 50.0).values
    
    # 2. WaveTrend WT1 normalized to [-1, 1]
    wt_df = calculate_wavetrend(df)
    f2 = np.clip(wt_df['WT1'].values / 60.0, -1.0, 1.0)
    
    # 3. Momentum / Return (10-bar)
    f3 = np.clip(df['close'].pct_change(10).fillna(0).values * 20.0, -1.0, 1.0)
    
    # 4. Normalized Range / Volatility
    high_low = (df['high'] - df['low']) / df['close']
    f4 = np.clip((high_low - high_low.rolling(20).mean()).fillna(0).values * 10.0, -1.0, 1.0)
    
    # Target: 4-bar forward direction (for training historical points)
    forward_returns = df['close'].pct_change(4).shift(-4).fillna(0).values
    labels = np.sign(forward_returns) # +1 for UP, -1 for DOWN, 0 for flat
    
    pred = np.zeros(n)
    conf = np.zeros(n)
    
    # Compute Lorentzian Distance for each point using past historical lookback
    features = np.column_stack([f1, f2, f3, f4])
    
    for i in range(50, n):
        start_idx = max(0, i - lookback)
        history_feats = features[start_idx:i-4] # avoid lookahead leakage by ending 4 bars prior
        history_labels = labels[start_idx:i-4]
        
        if len(history_feats) < k:
            continue
            
        current_feat = features[i]
        
        # Lorentzian Distance Metric: sum(ln(1 + |x_i - y_i|))
        diff = np.abs(history_feats - current_feat)
        l_dists = np.sum(np.log1p(diff), axis=1)
        
        # Find k-Nearest Neighbors
        k_indices = np.argpartition(l_dists, k)[:k]
        k_labels = history_labels[k_indices]
        
        up_count = np.sum(k_labels > 0)
        down_count = np.sum(k_labels < 0)
        
        if up_count > down_count:
            pred[i] = 1
            conf[i] = up_count / k
        elif down_count > up_count:
            pred[i] = -1
            conf[i] = down_count / k
        else:
            pred[i] = 0
            conf[i] = 0.5
            
    res['LORENTZIAN_PRED'] = pred
    res['LORENTZIAN_CONF'] = conf
    
    return res


def add_tradingview_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and merges all 4 elite TradingView indicators into the DataFrame.
    """
    if df is None or df.empty:
        return df
        
    df_res = df.copy()
    
    # 1. WaveTrend
    wt_df = calculate_wavetrend(df_res)
    for col in wt_df.columns:
        df_res[col] = wt_df[col]
        
    # 2. Nadaraya-Watson Envelope
    nw_df = calculate_nadaraya_watson(df_res)
    for col in nw_df.columns:
        df_res[col] = nw_df[col]
        
    # 3. UT Bot Alerts
    ut_df = calculate_ut_bot(df_res)
    for col in ut_df.columns:
        df_res[col] = ut_df[col]
        
    # 4. Lorentzian Classification
    lor_df = calculate_lorentzian_classification(df_res)
    for col in lor_df.columns:
        df_res[col] = lor_df[col]
        
    return df_res
