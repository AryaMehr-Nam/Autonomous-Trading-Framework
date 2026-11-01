"""
TradeAgent Multi-Timeframe Strategy Matrix Backtest Runner.
Evaluates Trend Following, Mean Reversion, and Smart Money Concepts (SMC)
across M15, H1, and H4 timeframes for EUR/USD, GBP/USD, BTC/USDT, ETH/USDT.
"""

import asyncio
import os
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import optuna
from datetime import datetime

# Suppress optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

from core.models import Timeframe, AssetClass
from data.providers.demo_provider import DemoDataProvider


FREQ_MAP = {
    "M15": "15m",
    "H1": "1h",
    "H4": "4h"
}


def optimize_trend(df: pd.DataFrame, freq: str = "1h", n_trials: int = 25):
    """Optimize Fast and Slow EMA periods for Trend Following."""
    if len(df) < 50:
        return {"fast_ema": 12, "slow_ema": 26}

    def objective(trial):
        fast = trial.suggest_int("fast_ema", 5, 25)
        slow = trial.suggest_int("slow_ema", 26, 100)
        if fast >= slow:
            raise optuna.TrialPruned()

        fast_series = df["close"].ewm(span=fast).mean()
        slow_series = df["close"].ewm(span=slow).mean()

        entries = (fast_series > slow_series) & (fast_series.shift(1) <= slow_series.shift(1))
        exits = (fast_series < slow_series) & (fast_series.shift(1) >= slow_series.shift(1))

        if entries.sum() < 2:
            return -10.0

        try:
            pf = vbt.Portfolio.from_signals(df["close"], entries, exits, init_cash=10000, fees=0.001, freq=freq)
            sharpe = pf.sharpe_ratio()
            if np.isnan(sharpe):
                return -10.0
            return sharpe
        except Exception:
            return -10.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params if study.best_value > -9.0 else {"fast_ema": 12, "slow_ema": 26}


def optimize_mean_reversion(df: pd.DataFrame, freq: str = "1h", n_trials: int = 25):
    """Optimize RSI length and threshold levels for Mean Reversion."""
    if len(df) < 50:
        return {"rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70}

    def objective(trial):
        period = trial.suggest_int("rsi_period", 5, 25)
        buy = trial.suggest_int("rsi_buy", 20, 45)
        sell = trial.suggest_int("rsi_sell", 55, 80)

        rsi = ta.rsi(df["close"], length=period)
        if rsi is None or rsi.empty:
            return -10.0

        entries = (rsi < buy) & (rsi.shift(1) >= buy)
        exits = (rsi > sell) & (rsi.shift(1) <= sell)

        if entries.sum() < 2:
            return -10.0

        try:
            pf = vbt.Portfolio.from_signals(df["close"], entries, exits, init_cash=10000, fees=0.001, freq=freq)
            sharpe = pf.sharpe_ratio()
            if np.isnan(sharpe):
                return -10.0
            return sharpe
        except Exception:
            return -10.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params if study.best_value > -9.0 else {"rsi_period": 14, "rsi_buy": 30, "rsi_sell": 70}


def optimize_smc(df: pd.DataFrame, freq: str = "1h", is_crypto: bool = False, n_trials: int = 25):
    """Optimize Displacement %, FVG Gap %, and Lookback for SMC."""
    if len(df) < 50:
        return {"displacement_pct": 0.5 if is_crypto else 0.15, "fvg_pct": 0.1 if is_crypto else 0.05, "ob_lookback": 10}

    min_disp = 0.2 if is_crypto else 0.05
    max_disp = 2.0 if is_crypto else 0.40
    min_fvg = 0.05 if is_crypto else 0.01
    max_fvg = 0.8 if is_crypto else 0.15

    def objective(trial):
        disp = trial.suggest_float("displacement_pct", min_disp, max_disp)
        fvg = trial.suggest_float("fvg_pct", min_fvg, max_fvg)
        lb = trial.suggest_int("ob_lookback", 5, 30)

        move_up = (df["close"] - df["open"].shift(2)) / df["open"].shift(2) * 100
        fvg_gap = (df["low"] - df["high"].shift(2)) / df["high"].shift(2) * 100
        entries = (move_up > disp) & (fvg_gap > fvg)

        move_down = (df["open"].shift(2) - df["close"]) / df["open"].shift(2) * 100
        fvg_gap_down = (df["low"].shift(2) - df["high"]) / df["low"].shift(2) * 100
        exits = (move_down > disp) & (fvg_gap_down > fvg)

        if entries.sum() < 2:
            return -10.0

        try:
            pf = vbt.Portfolio.from_signals(df["close"], entries, exits, init_cash=10000, fees=0.001, freq=freq)
            sharpe = pf.sharpe_ratio()
            if np.isnan(sharpe):
                return -10.0
            return sharpe
        except Exception:
            return -10.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params if study.best_value > -9.0 else {
        "displacement_pct": 0.5 if is_crypto else 0.15,
        "fvg_pct": 0.1 if is_crypto else 0.05,
        "ob_lookback": 10
    }


def run_signals_backtest(df: pd.DataFrame, strategy: str, params: dict, freq: str = "1h", fee: float = 0.001):
    """Run VectorBT backtest and compute metrics."""
    if strategy == "Trend Following":
        fast = params.get("fast_ema", 12)
        slow = params.get("slow_ema", 26)
        fast_series = df["close"].ewm(span=fast).mean()
        slow_series = df["close"].ewm(span=slow).mean()
        entries = (fast_series > slow_series) & (fast_series.shift(1) <= slow_series.shift(1))
        exits = (fast_series < slow_series) & (fast_series.shift(1) >= slow_series.shift(1))

    elif strategy == "Mean Reversion":
        period = params.get("rsi_period", 14)
        buy = params.get("rsi_buy", 30)
        sell = params.get("rsi_sell", 70)
        rsi = ta.rsi(df["close"], length=period)
        if rsi is None or rsi.empty:
            return None
        entries = (rsi < buy) & (rsi.shift(1) >= buy)
        exits = (rsi > sell) & (rsi.shift(1) <= sell)

    elif strategy == "Smart Money Concepts (SMC)":
        disp = params.get("displacement_pct", 0.15)
        fvg = params.get("fvg_pct", 0.05)
        move_up = (df["close"] - df["open"].shift(2)) / df["open"].shift(2) * 100
        fvg_gap = (df["low"] - df["high"].shift(2)) / df["high"].shift(2) * 100
        entries = (move_up > disp) & (fvg_gap > fvg)

        move_down = (df["open"].shift(2) - df["close"]) / df["open"].shift(2) * 100
        fvg_gap_down = (df["low"].shift(2) - df["high"]) / df["low"].shift(2) * 100
        exits = (move_down > disp) & (fvg_gap_down > fvg)
    else:
        return None

    # Run vectorbt portfolio
    portfolio = vbt.Portfolio.from_signals(
        df["close"],
        entries,
        exits,
        init_cash=10000.0,
        fees=fee,
        freq=freq
    )

    stats = portfolio.stats()
    trades = portfolio.trades.records_readable

    total_trades = int(stats.get("Total Trades", 0))
    win_rate = float(stats.get("Win Rate [%]", 0.0))
    total_return = float(stats.get("Total Return [%]", 0.0))
    benchmark_return = float(stats.get("Benchmark Return [%]", 0.0))
    max_dd = float(stats.get("Max Drawdown [%]", 0.0))
    sharpe = float(stats.get("Sharpe Ratio", 0.0))
    profit_factor = float(stats.get("Profit Factor", 0.0))

    if np.isnan(sharpe):
        sharpe = 0.0
    if np.isnan(win_rate):
        win_rate = 0.0
    if np.isnan(profit_factor):
        profit_factor = 0.0
    if np.isnan(max_dd):
        max_dd = 0.0

    # Monte Carlo simulation
    mc_95_dd = max_dd
    mc_ruin = 0.0
    if len(trades) >= 5:
        try:
            trade_returns = trades["Return"].values
            n_trades = len(trade_returns)
            mc_sims = 1000
            mc_drawdowns = []
            ruined = 0
            for _ in range(mc_sims):
                sampled = np.random.choice(trade_returns, size=n_trades, replace=True)
                eq = 10000.0 * np.cumprod(1 + sampled)
                peak = np.maximum.accumulate(eq)
                dd = (peak - eq) / peak
                mc_drawdowns.append(np.max(dd) * 100 if len(dd) > 0 else 0)
                if np.min(eq) < 2000.0:
                    ruined += 1
            mc_95_dd = float(np.percentile(mc_drawdowns, 95))
            mc_ruin = float((ruined / mc_sims) * 100)
        except Exception:
            pass

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "mc_95_dd": mc_95_dd,
        "mc_ruin": mc_ruin,
        "start_date": str(df.index[0]),
        "end_date": str(df.index[-1]),
        "candle_count": len(df),
        "params": params
    }


async def main():
    print("================================================================")
    print("Executing Multi-Timeframe Matrix Backtest for TradeAgent")
    print("Symbols: EUR/USD, GBP/USD, BTC/USDT, ETH/USDT")
    print("Timeframes: M15, H1, H4")
    print("Strategies: Trend Following, Mean Reversion, SMC")
    print("================================================================")

    provider = DemoDataProvider()
    await provider.connect()

    symbols = ["EUR/USD", "GBP/USD", "BTC/USDT", "ETH/USDT"]
    timeframes = [("M15", Timeframe.M15), ("H1", Timeframe.H1), ("H4", Timeframe.H4)]
    strategies = ["Trend Following", "Mean Reversion", "Smart Money Concepts (SMC)"]

    all_results = []

    for sym in symbols:
        is_crypto = "BTC" in sym or "ETH" in sym
        fee = 0.001 if is_crypto else 0.0005  # 0.1% for crypto, 0.05% for forex

        for tf_name, tf_enum in timeframes:
            freq_str = FREQ_MAP.get(tf_name, "1h")
            print(f"\n>>> Fetching data for {sym} [{tf_name}]...")
            candles = await provider.get_historical_candles(sym, tf_enum, limit=1000)

            if not candles or len(candles) < 30:
                print(f"Failed to fetch sufficient candles for {sym} {tf_name}.")
                continue

            df = pd.DataFrame([c.__dict__ for c in candles])
            df.set_index("timestamp", inplace=True)
            print(f"Loaded {len(df)} candles for {sym} [{tf_name}] ({df.index[0]} to {df.index[-1]})")

            for strat in strategies:
                print(f"  Optimizing & Backtesting: {sym} | {tf_name} | {strat}...")

                if strat == "Trend Following":
                    params = optimize_trend(df, freq=freq_str, n_trials=30)
                elif strat == "Mean Reversion":
                    params = optimize_mean_reversion(df, freq=freq_str, n_trials=30)
                else:
                    params = optimize_smc(df, freq=freq_str, is_crypto=is_crypto, n_trials=30)

                metrics = run_signals_backtest(df, strat, params, freq=freq_str, fee=fee)

                if metrics:
                    res_item = {
                        "symbol": sym,
                        "timeframe": tf_name,
                        "strategy": strat,
                        "is_crypto": is_crypto,
                        **metrics
                    }
                    all_results.append(res_item)
                    print(f"    -> Trades: {metrics['total_trades']:2d} | WinRate: {metrics['win_rate']:5.1f}% | "
                          f"Return: {metrics['total_return']:+6.2f}% | MaxDD: {metrics['max_drawdown']:5.2f}% | "
                          f"Sharpe: {metrics['sharpe_ratio']:+5.2f} | Params: {metrics['params']}")

    # Save JSON results locally
    results_json_path = "data_store/mtf_backtest_matrix_results.json"
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n================================================================")
    print(f"Completed {len(all_results)} backtests. Saved to {results_json_path}")
    print("================================================================")

    return all_results

if __name__ == "__main__":
    asyncio.run(main())
