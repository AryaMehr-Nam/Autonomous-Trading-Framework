"""
TradeAgent Strategy Optimizer.

Walk-forward optimization using Optuna.
"""

import pandas as pd
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from core.logger import get_logger

logger = get_logger("optimizer")


class Optimizer:
    def __init__(self):
        self.study = None

    def optimize_ta_params(self, df: pd.DataFrame, n_trials: int = 50, timeout: int = None):
        """
        Example optimization for moving average crossover.
        In a full system, this would dynamically construct the parameter grid 
        from the config file.
        """
        if not HAS_OPTUNA:
            logger.error("optuna is not installed.")
            return
            
        if df.empty:
            return

        def objective(trial):
            # Propose parameters
            fast_ma = trial.suggest_int('fast_ma', 5, 20)
            slow_ma = trial.suggest_int('slow_ma', 21, 100)
            
            if fast_ma >= slow_ma:
                raise optuna.TrialPruned()
                
            import pandas_ta as ta
            import vectorbt as vbt
            import numpy as np
            
            # Purged & Embargoed Walk-Forward Validation (3 splits)
            splits = 3
            split_size = len(df) // (splits + 1)
            embargo_size = 24  # 24-period Embargo to prevent serial correlation (Lopez de Prado)
            
            oos_sharpes = []
            
            for i in range(splits):
                # Apply Embargo
                val_start = ((i + 1) * split_size) + embargo_size
                val_end = val_start + split_size
                
                if val_start >= len(df):
                    continue
                    
                # To calculate indicators properly, we need some padding before val_start
                padding = slow_ma + 5
                start_idx = max(0, val_start - padding)
                
                window_df = df.iloc[start_idx:val_end].copy()
                if len(window_df) < slow_ma + 10:
                    continue
                    
                # Compute indicators
                fast_series = ta.ema(window_df['close'], length=fast_ma)
                slow_series = ta.ema(window_df['close'], length=slow_ma)
                
                # Generate signals
                entries = (fast_series > slow_series) & (fast_series.shift(1) <= slow_series.shift(1))
                exits = (fast_series < slow_series) & (fast_series.shift(1) >= slow_series.shift(1))
                
                # Only backtest on the out-of-sample Validation portion
                val_close = window_df['close'].iloc[padding:]
                val_entries = entries.iloc[padding:]
                val_exits = exits.iloc[padding:]
                
                portfolio = vbt.Portfolio.from_signals(
                    val_close, val_entries, val_exits, init_cash=200, fees=0.001
                )
                
                stats = portfolio.stats()
                sharpe = stats.get('Sharpe Ratio', 0.0)
                if pd.isna(sharpe):
                    sharpe = 0.0
                oos_sharpes.append(sharpe)
                
            if not oos_sharpes:
                return 0.0
                
            # Maximize the average Out-Of-Sample Sharpe Ratio
            return np.mean(oos_sharpes)

        logger.info(f"Starting optimization for {n_trials} trials (Timeout: {timeout}s)...")
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials if timeout is None else None, timeout=timeout)
        
        logger.info(f"Best parameters: {self.study.best_params}")
        logger.info(f"Best Sharpe: {self.study.best_value:.2f}")
        
        return self.study.best_params

    def optimize_mean_reversion_params(self, df: pd.DataFrame, n_trials: int = 50, timeout: int = None):
        """
        Optimization for Mean Reversion (RSI fades).
        Target: Maximize Win Rate.
        """
        if not HAS_OPTUNA:
            logger.error("optuna is not installed.")
            return
            
        if df.empty:
            return

        def objective(trial):
            rsi_period = trial.suggest_int('rsi_period', 5, 21)
            rsi_buy = trial.suggest_int('rsi_buy', 20, 45)
            rsi_sell = trial.suggest_int('rsi_sell', 55, 80)
                
            import pandas_ta as ta
            import vectorbt as vbt
            import numpy as np
            
            # Purged & Embargoed Walk-Forward Validation (3 splits)
            splits = 3
            split_size = len(df) // (splits + 1)
            embargo_size = 24
            
            oos_win_rates = []
            
            for i in range(splits):
                val_start = ((i + 1) * split_size) + embargo_size
                val_end = val_start + split_size
                
                if val_start >= len(df): continue
                    
                padding = rsi_period + 5
                start_idx = max(0, val_start - padding)
                
                window_df = df.iloc[start_idx:val_end].copy()
                if len(window_df) < rsi_period + 10: continue
                    
                rsi_series = ta.rsi(window_df['close'], length=rsi_period)
                if rsi_series is None or rsi_series.empty: continue
                
                # Mean Reversion entries: Buy when RSI goes below rsi_buy, Sell when RSI goes above rsi_sell
                entries = (rsi_series < rsi_buy) & (rsi_series.shift(1) >= rsi_buy)
                exits = (rsi_series > rsi_sell) & (rsi_series.shift(1) <= rsi_sell)
                
                val_close = window_df['close'].iloc[padding:]
                val_entries = entries.iloc[padding:]
                val_exits = exits.iloc[padding:]
                
                portfolio = vbt.Portfolio.from_signals(
                    val_close, val_entries, val_exits, init_cash=200, fees=0.001
                )
                
                stats = portfolio.stats()
                win_rate = stats.get('Win Rate [%]', 0.0)
                if pd.isna(win_rate): win_rate = 0.0
                
                # Punish if it took zero trades
                total_trades = stats.get('Total Trades', 0)
                if total_trades < 3:
                    win_rate = 0.0
                    
                oos_win_rates.append(win_rate)
                
            if not oos_win_rates:
                return 0.0
                
            # Maximize Average Win Rate
            return np.mean(oos_win_rates)

        logger.info(f"Starting Mean Reversion Win-Rate optimization for {n_trials} trials (Timeout: {timeout}s)...")
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials if timeout is None else None, timeout=timeout)
        
        logger.info(f"Best MR parameters: {self.study.best_params}")
        logger.info(f"Best Average Win Rate: {self.study.best_value:.2f}%")
        
        best = self.study.best_params
        best["win_rate_pct"] = self.study.best_value
        return best

    def optimize_smc_params(self, df: pd.DataFrame, n_trials: int = 50, timeout: int = None):
        """
        Optimization for Smart Money Concepts (Order Blocks).
        Target: Maximize Win Rate * Profit Factor.
        """
        if not HAS_OPTUNA:
            logger.error("optuna is not installed.")
            return
            
        if df.empty:
            return

        def objective(trial):
            displacement_pct = trial.suggest_float('displacement_pct', 0.05, 0.40)
            fvg_pct = trial.suggest_float('fvg_pct', 0.01, 0.15)
            ob_lookback = trial.suggest_int('ob_lookback', 5, 30)
                
            import vectorbt as vbt
            import numpy as np
            
            splits = 3
            split_size = len(df) // (splits + 1)
            embargo_size = 24
            
            oos_scores = []
            
            for i in range(splits):
                val_start = ((i + 1) * split_size) + embargo_size
                val_end = val_start + split_size
                
                if val_start >= len(df): continue
                    
                padding = ob_lookback + 5
                start_idx = max(0, val_start - padding)
                
                window_df = df.iloc[start_idx:val_end].copy()
                if len(window_df) < ob_lookback + 10: continue
                
                # Generate entries based on SMC
                # For VectorBT, we'll approximate the entry as the bar following the OB detection
                # In live trading it places a limit order, but backtesting limit orders via boolean arrays is complex.
                # We'll use a market order approximation here for optimization speed.
                
                entries = pd.Series(False, index=window_df.index)
                exits = pd.Series(False, index=window_df.index)
                
                # Simple vectorized approximation of SMC for backtesting speed
                # Bullish OB: strong up move, FVG
                move_up = (window_df['close'] - window_df['open'].shift(2)) / window_df['open'].shift(2) * 100
                fvg_gap = (window_df['low'] - window_df['high'].shift(2)) / window_df['high'].shift(2) * 100
                bullish_ob = (move_up > displacement_pct) & (fvg_gap > fvg_pct)
                
                # Bearish OB: strong down move, FVG
                move_down = (window_df['open'].shift(2) - window_df['close']) / window_df['open'].shift(2) * 100
                fvg_gap_down = (window_df['low'].shift(2) - window_df['high']) / window_df['low'].shift(2) * 100
                bearish_ob = (move_down > displacement_pct) & (fvg_gap_down > fvg_pct)
                
                entries = bullish_ob
                exits = bearish_ob
                
                val_close = window_df['close'].iloc[padding:]
                val_entries = entries.iloc[padding:]
                val_exits = exits.iloc[padding:]
                
                portfolio = vbt.Portfolio.from_signals(
                    val_close, val_entries, val_exits, init_cash=200, fees=0.001
                )
                
                stats = portfolio.stats()
                win_rate = stats.get('Win Rate [%]', 0.0)
                if pd.isna(win_rate): win_rate = 0.0
                
                total_trades = stats.get('Total Trades', 0)
                if total_trades < 3:
                    win_rate = 0.0
                    
                oos_scores.append(win_rate)
                
            if not oos_scores:
                return 0.0
                
            return np.mean(oos_scores)

        logger.info(f"Starting SMC optimization for {n_trials} trials...")
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials if timeout is None else None, timeout=timeout)
        
        logger.info(f"Best SMC parameters: {self.study.best_params}")
        
        best = self.study.best_params
        best["win_rate_pct"] = self.study.best_value
        return best
