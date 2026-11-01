"""
TradeAgent Strategy Discoverer Agent.

Standalone agent that backtests a specific strategy across all assets
and writes the learned parameters to JSON and Obsidian.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from config.settings import DATA_DIR, trading_params
from core.logger import get_logger
from core.models import Timeframe
from data.providers.demo_provider import DemoDataProvider

logger = get_logger("strategy_agent")


class StrategyDiscovererAgent:
    def __init__(self, strategy_type: str, timeframes: list = None):
        self.strategy_type = strategy_type.lower()
        self.symbols = []
        for market, assets in trading_params.watchlist.items():
            if isinstance(assets, list):
                self.symbols.extend(assets)
        self.timeframes = timeframes or ["1m", "5m", "15m"]
        self.provider = DemoDataProvider()
        self.results = {}

    async def run(self, trials: int = 20):
        print(f"================================================================")
        print(f"Agent [{self.strategy_type.upper()}] Starting Discovery")
        print(f"================================================================")
        
        await self.provider.connect()
        
        for symbol in self.symbols:
            self.results[symbol] = {}
            for tf_str in self.timeframes:
                print(f"[{self.strategy_type}] Analyzing {symbol} on {tf_str}...")
                tf = Timeframe.from_string(tf_str)
                
                limit = 2000 if tf_str in ["1m", "5m"] else 1000
                candles = await self.provider.get_historical_candles(symbol, tf, limit=limit)
                
                if not candles or len(candles) < 100:
                    continue
                    
                df = pd.DataFrame([c.__dict__ for c in candles])
                df.set_index("timestamp", inplace=True)
                
                best_strat = self._optimize(df, symbol, tf_str, trials)
                if best_strat:
                    self.results[symbol][tf_str] = best_strat
                
        await self.provider.disconnect()
        
        # Save JSON
        strat_dir = DATA_DIR / "strategies"
        strat_dir.mkdir(parents=True, exist_ok=True)
        json_path = strat_dir / f"optimized_{self.strategy_type}.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\n[{self.strategy_type}] Discovery Complete! Saved to {json_path}")
        return self.results

    def _optimize(self, df: pd.DataFrame, symbol: str, tf_str: str, trials: int) -> dict:
        if not HAS_OPTUNA or not HAS_VBT: return {}
        
        def objective_trend(trial):
            ema_f = trial.suggest_int("ema_fast", 5, 20)
            ema_s = trial.suggest_int("ema_slow", 21, 60)
            rsi_buy = trial.suggest_int("rsi_buy", 40, 55)
            if ema_f >= ema_s: raise optuna.TrialPruned()
            
            fast = ta.ema(df['close'], length=ema_f)
            slow = ta.ema(df['close'], length=ema_s)
            rsi = ta.rsi(df['close'], length=14)
            if fast is None or slow is None or rsi is None: return -100.0
            
            entries = (fast > slow) & (rsi > rsi_buy)
            exits = (fast < slow)
            
            pf = vbt.Portfolio.from_signals(df['close'], entries, exits, init_cash=200, freq=tf_str)
            stats = pf.stats()
            sharpe = stats.get('Sharpe Ratio', -99.0)
            return sharpe if not pd.isna(sharpe) and not np.isinf(sharpe) else -100.0

        def objective_mean_reversion(trial):
            rsi_p = trial.suggest_int("rsi_period", 5, 21)
            rsi_buy = trial.suggest_int("rsi_buy", 20, 40)
            rsi_sell = trial.suggest_int("rsi_sell", 60, 80)
            
            rsi = ta.rsi(df['close'], length=rsi_p)
            if rsi is None: return -100.0
            
            entries = rsi < rsi_buy
            exits = rsi > rsi_sell
            
            pf = vbt.Portfolio.from_signals(df['close'], entries, exits, init_cash=200, freq=tf_str)
            stats = pf.stats()
            sharpe = stats.get('Sharpe Ratio', -99.0)
            return sharpe if not pd.isna(sharpe) and not np.isinf(sharpe) else -100.0

        def objective_breakout(trial):
            bb_p = trial.suggest_int("bb_period", 10, 40)
            bb_std = trial.suggest_float("bb_std", 1.5, 3.0)
            
            bb = ta.bbands(df['close'], length=bb_p, std=bb_std)
            if bb is None: return -100.0
            pct_col = [c for c in bb.columns if "BBP_" in c]
            if not pct_col: return -100.0
            
            bbp = bb[pct_col[0]]
            entries = bbp > 1.0  # Upside breakout
            exits = bbp < 0.5    # Return to mean
            
            pf = vbt.Portfolio.from_signals(df['close'], entries, exits, init_cash=200, freq=tf_str)
            stats = pf.stats()
            sharpe = stats.get('Sharpe Ratio', -99.0)
            return sharpe if not pd.isna(sharpe) and not np.isinf(sharpe) else -100.0

        study = optuna.create_study(direction="maximize")
        
        if self.strategy_type == "trend":
            study.optimize(objective_trend, n_trials=trials)
            best = study.best_params
            fast = ta.ema(df['close'], length=best['ema_fast'])
            slow = ta.ema(df['close'], length=best['ema_slow'])
            rsi = ta.rsi(df['close'], length=14)
            entries = (fast > slow) & (rsi > best['rsi_buy'])
            exits = (fast < slow)
        elif self.strategy_type == "mean_reversion":
            study.optimize(objective_mean_reversion, n_trials=trials)
            best = study.best_params
            rsi = ta.rsi(df['close'], length=best['rsi_period'])
            entries = rsi < best['rsi_buy']
            exits = rsi > best['rsi_sell']
        elif self.strategy_type == "breakout":
            study.optimize(objective_breakout, n_trials=trials)
            best = study.best_params
            bb = ta.bbands(df['close'], length=best['bb_period'], std=best['bb_std'])
            pct_col = [c for c in bb.columns if "BBP_" in c]
            bbp = bb[pct_col[0]]
            entries = bbp > 1.0
            exits = bbp < 0.5
        else:
            return {}

        pf = vbt.Portfolio.from_signals(df['close'], entries, exits, init_cash=200, freq=tf_str)
        stats = pf.stats()
        
        return {
            "symbol": symbol,
            "timeframe": tf_str,
            "best_params": best,
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0.0) or 0.0),
            "total_return_pct": float(stats.get("Total Return [%]", 0.0) or 0.0),
            "win_rate_pct": float(stats.get("Win Rate [%]", 0.0) or 0.0),
            "total_trades": int(stats.get("Total Trades", 0) or 0)
        }

def generate_markdown(results: dict, strategy_type: str) -> str:
    md = f"# 🧠 Agent [{strategy_type.upper()}] Memory Bank\n\n"
    md += f"This note contains the optimized hyper-parameters for {strategy_type.capitalize()} trading.\n"
    md += "Connects to: [[TradeAgent_Hub]]\n\n"
    for symbol, tf_data in results.items():
        md += f"## `{symbol}`\n"
        for tf, strat in tf_data.items():
            if not strat: continue
            md += f"- **{tf}**: Sharpe `{strat['sharpe_ratio']:.2f}` | Win `{strat['win_rate_pct']:.1f}%`\n"
            md += f"  - Params: {strat['best_params']}\n"
    return md

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True, choices=["trend", "mean_reversion", "breakout"])
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()
    
    agent = StrategyDiscovererAgent(args.strategy)
    results = asyncio.run(agent.run(trials=args.trials))
    
    md_content = generate_markdown(results, args.strategy)
    note_path = DATA_DIR / f"Agent_{args.strategy.capitalize()}_Brain.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(md_content)
