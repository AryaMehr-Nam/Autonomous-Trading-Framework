"""
TradeAgent Backtester.

Uses vectorbt to run extremely fast vectorized backtests on historical data.
"""

import pandas as pd
try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

from config.settings import trading_params
from core.logger import get_logger

logger = get_logger("backtester")


class Backtester:
    def __init__(self):
        self.config = trading_params.backtesting_config
        self.init_capital = self.config.get("initial_capital", 10000.0)
        self.fees = self.config.get("commission_pct", 0.001)

    def run_backtest(self, df: pd.DataFrame, entries: pd.Series, exits: pd.Series) -> pd.DataFrame:
        """
        Run a vectorized backtest given boolean entry and exit signals.
        Returns a summary of performance metrics.
        """
        if not HAS_VBT:
            logger.error("vectorbt is not installed.")
            return pd.DataFrame()
            
        if df.empty or entries.empty or exits.empty:
            logger.warning("Empty data provided to backtester.")
            return pd.DataFrame()

        try:
            # Create a portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                df['close'],
                entries,
                exits,
                init_cash=self.init_capital,
                fees=self.fees,
                freq='h' # Assuming hourly data for now, should be dynamic
            )
            
            stats = portfolio.stats()
            
            # --- Monte Carlo Trade Bootstrapping (Stress Testing) ---
            try:
                trades = portfolio.trades.records_readable
                if len(trades) >= 10:
                    import numpy as np
                    trade_returns = trades['Return'].values
                    n_trades = len(trade_returns)
                    mc_sims = 10000
                    
                    mc_drawdowns = []
                    mc_ruined = 0
                    
                    for _ in range(mc_sims):
                        # Reshuffle trades with replacement
                        sampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)
                        # Reconstruct synthetic equity curve
                        equity_curve = self.init_capital * np.cumprod(1 + sampled_returns)
                        
                        # Calculate path max drawdown
                        peak = np.maximum.accumulate(equity_curve)
                        drawdown = (peak - equity_curve) / peak
                        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
                        mc_drawdowns.append(max_dd * 100)
                        
                        # Ruin condition: lose 80% of account
                        if np.min(equity_curve) < self.init_capital * 0.2:
                            mc_ruined += 1
                            
                    p95_dd = np.percentile(mc_drawdowns, 95)
                    ruin_prob = (mc_ruined / mc_sims) * 100
                    
                    # Add to stats dictionary
                    # Note: portfolio.stats() returns a pd.Series
                    stats['MC 95% Max DD [%]'] = p95_dd
                    stats['MC Prob of Ruin [%]'] = ruin_prob
                    logger.info(f"Monte Carlo complete. 95% Worst DD: {p95_dd:.2f}%, Ruin Prob: {ruin_prob:.2f}%")
            except Exception as mc_e:
                logger.warning(f"Monte Carlo simulation failed: {mc_e}")
            # --------------------------------------------------------
            
            # Save equity curve plot locally
            fig = portfolio.plot()
            fig.write_html("backtest_results.html")
            
            logger.info(f"Backtest complete. Total Return: {stats.get('Total Return [%]', 0):.2f}%")
            return stats
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return pd.DataFrame()
