"""
TradeAgent Backtest & Optimization Runner.

Run backtests on historical market data using VectorBT and Optuna.
"""

import argparse
import asyncio
import pandas as pd

from backtesting.backtester import Backtester
from backtesting.optimizer import Optimizer
from data.providers.demo_provider import DemoDataProvider
from signals.ta_engine import TAEngine
from signals.ta_signals import TASignalGenerator
from core.models import Timeframe, AssetClass


async def run_standalone_backtest(symbol: str = "EUR/USD", timeframe_str: str = "H1", optimize: bool = False, strategy: str = "trend", timeout: int = None):
    print("==================================================")
    print(f"Starting Backtest for {symbol} ({timeframe_str})")
    print("==================================================")
    
    provider = DemoDataProvider()
    await provider.connect()
    
    tf = Timeframe.from_string(timeframe_str)
    print("Fetching historical data from yfinance...")
    candles = await provider.get_historical_candles(symbol, tf, limit=1000)
    
    if not candles:
        print(f"Failed to fetch data for {symbol}.")
        return

    # Convert to DataFrame
    df = pd.DataFrame([c.__dict__ for c in candles])
    df.set_index("timestamp", inplace=True)
    
    print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}.")
    
    ta_engine = TAEngine()
    df_ta = ta_engine.add_all_indicators(df)
    
    if optimize:
        print(f"\nRunning Optuna Parameter Optimization for {strategy}...")
        optimizer = Optimizer()
        
        if strategy == "mean_reversion":
            best_params = optimizer.optimize_mean_reversion_params(df_ta, n_trials=50, timeout=timeout)
            print(f"Optimization finished! Best MR Params: {best_params}")
            
            import json
            import os
            strategy_path = "data_store/strategies/optimized_mean_reversion.json"
            os.makedirs(os.path.dirname(strategy_path), exist_ok=True)
            
            try:
                with open(strategy_path, "w") as f:
                    strategy_payload = {
                        "strategy_name": "MeanReversion",
                        "win_rate_pct": best_params.pop("win_rate_pct", 50.0),
                        "parameters": best_params
                    }
                    json.dump(strategy_payload, f, indent=4)
                print(f"Saved optimized parameters to {strategy_path}")
            except Exception as e:
                print(f"Failed to save strategy JSON: {e}")
                
        elif strategy == "smc":
            best_params = optimizer.optimize_smc_params(df_ta, n_trials=50, timeout=timeout)
            print(f"Optimization finished! Best SMC Params: {best_params}")
            
            import json
            import os
            strategy_path = "data_store/strategies/optimized_smc.json"
            os.makedirs(os.path.dirname(strategy_path), exist_ok=True)
            
            try:
                with open(strategy_path, "w") as f:
                    strategy_payload = {
                        "strategy_name": "SmartMoneyConcepts",
                        "win_rate_pct": best_params.pop("win_rate_pct", 50.0),
                        "parameters": best_params
                    }
                    json.dump(strategy_payload, f, indent=4)
                print(f"Saved optimized parameters to {strategy_path}")
            except Exception as e:
                print(f"Failed to save strategy JSON: {e}")
                
        else:
            best_params = optimizer.optimize_ta_params(df_ta, n_trials=50, timeout=timeout)
            print(f"Optimization finished! Best Params: {best_params}")
            
            import json
            import os
            strategy_path = "data_store/strategies/optimized_trend.json"
            os.makedirs(os.path.dirname(strategy_path), exist_ok=True)
            
            try:
                with open(strategy_path, "w") as f:
                    strategy_payload = {
                        "strategy_name": "TrendFollow",
                        "parameters": {
                            "ema_fast": best_params.get("fast_ma", 7),
                            "ema_slow": best_params.get("slow_ma", 36)
                        }
                    }
                    json.dump(strategy_payload, f, indent=4)
                print(f"Saved optimized parameters to {strategy_path}")
            except Exception as e:
                print(f"Failed to save strategy JSON: {e}")
    else:
        # Generate simple signals for VectorBT
        print(f"\nCalculating TA Signals & Entry/Exit vectors for {strategy}...")
        
        import json
        import os
        import pandas_ta as ta
        
        if strategy == "mean_reversion":
            strategy_path = "data_store/strategies/optimized_mean_reversion.json"
            rsi_period, rsi_buy, rsi_sell = 14, 30, 70
            
            if os.path.exists(strategy_path):
                with open(strategy_path, "r") as f:
                    data = json.load(f)
                    params = data.get("parameters", {})
                    rsi_period = params.get("rsi_period", 14)
                    rsi_buy = params.get("rsi_buy", 30)
                    rsi_sell = params.get("rsi_sell", 70)
            
            print(f"Using Optimized MR Parameters: RSI Period = {rsi_period}, Buy = {rsi_buy}, Sell = {rsi_sell}")
            
            rsi_series = ta.rsi(df_ta['close'], length=rsi_period)
            entries = (rsi_series < rsi_buy) & (rsi_series.shift(1) >= rsi_buy)
            exits = (rsi_series > rsi_sell) & (rsi_series.shift(1) <= rsi_sell)
            
        elif strategy == "smc":
            strategy_path = "data_store/strategies/optimized_smc.json"
            displacement_pct, fvg_pct, ob_lookback = 0.15, 0.05, 10
            
            if os.path.exists(strategy_path):
                with open(strategy_path, "r") as f:
                    data = json.load(f)
                    params = data.get("parameters", {})
                    displacement_pct = params.get("displacement_pct", 0.15)
                    fvg_pct = params.get("fvg_pct", 0.05)
                    ob_lookback = params.get("ob_lookback", 10)
            
            print(f"Using Optimized SMC Parameters: Displacement > {displacement_pct}%, FVG > {fvg_pct}%, Lookback = {ob_lookback}")
            
            # Reconstruct the boolean arrays using the same logic as the optimizer
            move_up = (df_ta['close'] - df_ta['open'].shift(2)) / df_ta['open'].shift(2) * 100
            fvg_gap = (df_ta['low'] - df_ta['high'].shift(2)) / df_ta['high'].shift(2) * 100
            entries = (move_up > displacement_pct) & (fvg_gap > fvg_pct)
            
            move_down = (df_ta['open'].shift(2) - df_ta['close']) / df_ta['open'].shift(2) * 100
            fvg_gap_down = (df_ta['low'].shift(2) - df_ta['high']) / df_ta['low'].shift(2) * 100
            exits = (move_down > displacement_pct) & (fvg_gap_down > fvg_pct)
            
        else:
            strategy_path = "data_store/strategies/optimized_trend.json"
            fast_period, slow_period = 12, 26
            
            if os.path.exists(strategy_path):
                with open(strategy_path, "r") as f:
                    data = json.load(f)
                    params = data.get("parameters", {})
                    fast_period = params.get("ema_fast", 12)
                    slow_period = params.get("ema_slow", 26)
                    
            print(f"Using Optimized Trend Parameters: Fast EMA = {fast_period}, Slow EMA = {slow_period}")
            df_ta['EMA_FAST'] = df_ta['close'].ewm(span=fast_period).mean()
            df_ta['EMA_SLOW'] = df_ta['close'].ewm(span=slow_period).mean()
            
            entries = (df_ta['EMA_FAST'] > df_ta['EMA_SLOW']) & (df_ta['EMA_FAST'].shift(1) <= df_ta['EMA_SLOW'].shift(1))
            exits = (df_ta['EMA_FAST'] < df_ta['EMA_SLOW']) & (df_ta['EMA_FAST'].shift(1) >= df_ta['EMA_SLOW'].shift(1))
        
        backtester = Backtester()
        stats = backtester.run_backtest(df_ta, entries, exits)
        
        print("\n================ Backtest Results ================")
        if not stats.empty:
            for metric in ['Start', 'End', 'Total Trades', 'Total Return [%]', 'Benchmark Return [%]', 'Max Drawdown [%]', 'Sharpe Ratio', 'Win Rate [%]', 'MC 95% Max DD [%]', 'MC Prob of Ruin [%]']:
                if metric in stats:
                    print(f"{metric:25s}: {stats[metric]}")
            print("==================================================")
            print("Report saved to backtest_results.html")
        else:
            print("Could not calculate backtest stats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeAgent Backtesting Tool")
    parser.add_argument("--symbol", type=str, default="EUR/USD", help="Symbol to backtest (default: EUR/USD)")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe (default: H1)")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna parameter optimization")
    parser.add_argument("--strategy", type=str, default="trend", help="Strategy to optimize (trend | mean_reversion)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for optimization (Deep Learning mode)")
    
    args = parser.parse_args()
    
    asyncio.run(run_standalone_backtest(args.symbol, args.timeframe, args.optimize, args.strategy, args.timeout))
