"""
TradeAgent Risk Manager.

Enforces hard-coded safety guardrails, position sizing, and drawdown limits.
This component is the final gatekeeper before any trade is executed.
"""

from typing import List

from config.settings import trading_params
from core.logger import get_logger
from core.models import AccountInfo, Position, Signal
from risk.stop_loss import StopLossManager

logger = get_logger("risk_manager")


import pandas as pd

class RiskManager:
    """Gatekeeper for all trade executions based on risk rules."""
    
    def __init__(self):
        self.config = trading_params.risk
        self.sl_manager = StopLossManager()
        self.is_paused = False
        
        # Track for multi-horizon drawdowns
        self.peak_equity_all_time = 0.0
        self.start_of_day_equity = 0.0
        self.start_of_week_equity = 0.0
        self.last_update_date = None
        self.last_update_week = None
        
    def evaluate_signal(self, signal: Signal, account: AccountInfo, open_positions: List[Position], df: pd.DataFrame = None) -> bool:
        """
        Evaluate if a signal should be traded based on risk rules.
        Modifies the signal in-place with optimal position size if approved.
        """
        if self.is_paused:
            logger.warning(f"RiskManager is PAUSED. Rejecting signal {signal.id}.")
            return False
            
        # 1. Max Open Positions Check
        max_pos = self.config.get("max_open_positions", 5)
        if len(open_positions) >= max_pos:
            logger.info(f"Max open positions ({max_pos}) reached. Rejecting signal.")
            return False
            
        # 2. Correlation Check (Don't open too many trades on same base/quote currency)
        if not self._check_correlation(signal.symbol, open_positions):
            return False
            
        # 3. Drawdown Checks
        if not self.check_drawdown(account):
            return False
            
        # 3.5 Strict Minimum Win-Rate Filter (>= 60% rule)
        min_win_rate = self.config.get("min_win_rate", 0.60)
        if signal.win_rate < min_win_rate:
            logger.warning(f"[{signal.symbol}] Strategy win rate {signal.win_rate*100:.1f}% below minimum required {min_win_rate*100:.0f}%. Rejecting signal.")
            return False

        # 3.6 Small Account Dollar Risk Cap (Protect against oversized commodity swings)
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        if "XAU" in signal.symbol or "GOLD" in signal.symbol:
            dollar_risk_min_lot = sl_dist * 1.0  # 0.01 lot = 1 oz
            max_allowed_dollar_risk = self.config.get("max_dollar_risk", 3.00)
            if dollar_risk_min_lot > max_allowed_dollar_risk:
                logger.warning(f"[{signal.symbol}] Stop Loss distance ${sl_dist:.2f} would risk ${dollar_risk_min_lot:.2f} on minimum lot (exceeds ${max_allowed_dollar_risk:.2f} limit). Rejecting signal to protect capital.")
                return False
            
        # 4. Risk/Reward Ratio Check
        min_rr = self.config.get("min_risk_reward_ratio", 1.5)
        if signal.risk_reward_ratio < min_rr - 0.01:  # Allow signals that meet the minimum (avoid float precision issues)
            logger.info(f"R/R ratio {signal.risk_reward_ratio:.2f} below minimum {min_rr:.2f}. Rejecting.")
            return False
            
        # 5. Position Sizing (with Volatility Regime filtering)
        size = self.calculate_position_size(signal, account, df)
        if size <= 0:
            logger.warning("Calculated position size is 0 or negative. Rejecting.")
            return False
            
        # All checks passed, update signal quantity (to be used by executor)
        # Note: quantity field is not in Signal dataclass by default, but we can pass it along
        # or the Executor will call calculate_position_size directly. 
        # For this design, we assume the executor asks the RiskManager for the size.
        
        logger.info(f"Signal {signal.id} ({signal.symbol} {signal.signal_type.value}) APPROVED by RiskManager.")
        return True

    def calculate_position_size(self, signal: Signal, account: AccountInfo, df: pd.DataFrame = None) -> float:
        """
        Calculate the optimal position size using the Fractional Kelly Criterion.
        Formula: f* = W - ((1 - W) / R)
        """
        if signal.stop_loss == 0 or signal.entry_price == 0:
            return 0.0
            
        W = signal.win_rate
        R = signal.risk_reward
        
        # 1. Calculate Full Kelly Fraction
        kelly_fraction = W - ((1.0 - W) / R)
        
        if kelly_fraction <= 0:
            logger.warning(f"Negative or zero Kelly Edge ({kelly_fraction:.4f}). Trade expectancy is negative. Rejecting.")
            return 0.0
            
        # 2. Apply Fractional Kelly (Safety)
        fractional_multiplier = self.config.get("kelly_fraction", 0.25)
        # HARD CAP fractional multiplier at 0.5 max per PDF rules
        fractional_multiplier = min(fractional_multiplier, 0.5)
        
        # DYNAMIC FRACTIONAL KELLY (Drawdown scaling)
        if self.config.get("dynamic_kelly_drawdown_scaling", True) and getattr(self, 'peak_equity_all_time', 0.0) > 0:
            current_drawdown = (self.peak_equity_all_time - account.equity) / self.peak_equity_all_time
            if current_drawdown > 0:
                max_rolling = self.config.get("max_rolling_drawdown_pct", 20.0) / 100.0
                # Linear penalty factor: approaches 0.1 as drawdown approaches max_rolling limit
                penalty_factor = max(0.1, 1.0 - (current_drawdown / max_rolling))
                fractional_multiplier *= penalty_factor
                logger.info(f"Dynamic Kelly: Account in {current_drawdown*100:.1f}% drawdown. Scaling Kelly fraction by {penalty_factor:.2f}x to {fractional_multiplier:.2f}")
        
        # VOLATILITY REGIME FILTER:
        if df is not None and not df.empty and "ATR" in df.columns:
            atr = df["ATR"].iloc[-1]
            close = df["close"].iloc[-1]
            atr_pct = (atr / close) * 100
            if atr_pct > 0.4:
                logger.warning(f"CHAOS REGIME: High ATR ({atr_pct:.2f}%). Slashing Kelly fraction by 50%!")
                fractional_multiplier *= 0.5
                
        safe_kelly_fraction = kelly_fraction * fractional_multiplier
        
        # 3. Cap max risk per trade to absolute hard limit
        max_risk_pct = self.config.get("risk_per_trade_pct", 1.0) / 100.0
        final_risk_pct = min(safe_kelly_fraction, max_risk_pct)
        
        risk_amount_fiat = account.equity * final_risk_pct
        
        logger.info(f"Kelly sizing: W={W:.2f}, R={R:.2f} -> Risking {final_risk_pct*100:.2f}% of equity.")
        
        # Distance to Stop Loss
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        if sl_dist == 0:
            return 0.0
            
        # Simplified formula: (Risk Amount) / (Stop Loss Distance)
        raw_size_units = risk_amount_fiat / sl_dist
        
        # Apply leverage constraint
        max_notional = account.equity * account.leverage
        current_notional = account.margin_used * account.leverage
        available_notional = max_notional - current_notional
        
        trade_notional = raw_size_units * signal.entry_price
        
        if trade_notional > available_notional:
            logger.warning(f"Trade notional ({trade_notional}) exceeds available ({available_notional}). Scaling down.")
            raw_size_units = available_notional / signal.entry_price
            
        # Convert units to standard contract lots
        lot_size = 100000.0
        if "XAU" in signal.symbol:
            lot_size = 100.0
        elif "US30" in signal.symbol or "DJI" in signal.symbol or "WS30" in signal.symbol:
            lot_size = 1.0
        elif "BTC" in signal.symbol or "ETH" in signal.symbol or "SOL" in signal.symbol:
            lot_size = 1.0
            
        raw_size_lots = raw_size_units / lot_size
        
        # 4. HARD CAP Max Position Size (Regardless of Kelly)
        max_absolute_lots = self.config.get("max_absolute_lots", 2.0)
        final_lots = min(raw_size_lots, max_absolute_lots)
        
        # Round to 2 decimal places
        rounded = round(final_lots, 2)
        
        # For small accounts: force minimum micro-lot of 0.01 if Kelly produces a positive edge
        # This allows accounts under $500 to participate with controlled risk
        if rounded < 0.01:
            if kelly_fraction > 0:
                rounded = 0.01
                logger.info(f"Small account override: forcing minimum lot 0.01 (Kelly edge: {kelly_fraction:.4f}).")
            else:
                logger.warning(f"Calculated lot size {final_lots:.4f} rounds to 0 and no Kelly edge. Rejecting.")
                return 0.0
            
        return rounded

    def check_drawdown(self, account: AccountInfo) -> bool:
        """
        Check if daily/weekly drawdown limits have been breached.
        Evaluates against rolling equity peaks.
        """
        from datetime import datetime
        now = datetime.utcnow()
        today = now.date()
        this_week = now.isocalendar()[1]
        
        # Initialize trackers if first run
        if self.last_update_date != today:
            self.start_of_day_equity = account.equity
            self.last_update_date = today
            
        if self.last_update_week != this_week:
            self.start_of_week_equity = account.equity
            self.last_update_week = this_week
            
        # Update rolling all-time peak
        if account.equity > self.peak_equity_all_time:
            self.peak_equity_all_time = account.equity
            
        # Check Daily Drawdown
        if self.start_of_day_equity > 0:
            daily_dd = (self.start_of_day_equity - account.equity) / self.start_of_day_equity
            max_daily = self.config.get("max_daily_drawdown_pct", 3.0) / 100.0
            if daily_dd > max_daily:
                logger.error(f"DAILY DRAWDOWN BREACHED: {daily_dd*100:.2f}% > {max_daily*100:.2f}%. Halting!")
                self.is_paused = True
                return False
                
        # Check Weekly Drawdown
        if self.start_of_week_equity > 0:
            weekly_dd = (self.start_of_week_equity - account.equity) / self.start_of_week_equity
            max_weekly = self.config.get("max_weekly_drawdown_pct", 5.0) / 100.0
            if weekly_dd > max_weekly:
                logger.error(f"WEEKLY DRAWDOWN BREACHED: {weekly_dd*100:.2f}% > {max_weekly*100:.2f}%. Halting!")
                self.is_paused = True
                return False
                
        # Check Rolling Peak Drawdown (Equity Curve Breaker)
        if self.peak_equity_all_time > 0:
            rolling_dd = (self.peak_equity_all_time - account.equity) / self.peak_equity_all_time
            max_rolling = self.config.get("max_rolling_drawdown_pct", 10.0) / 100.0
            if rolling_dd > max_rolling:
                logger.error(f"EQUITY CURVE BREAKER BREACHED: Down {rolling_dd*100:.2f}% from peak. Halting!")
                self.is_paused = True
                return False
                
        # Margin check fallback
        if account.margin_level_pct < 50.0 and account.margin_level_pct > 0:
            logger.warning(f"Margin level critical ({account.margin_level_pct:.1f}%). Halting trading.")
            self.is_paused = True
            return False
            
        return True

    def _check_correlation(self, symbol: str, open_positions: List[Position]) -> bool:
        """
        Prevent opening too many trades with correlated currencies.
        E.g., if we are long EUR/USD and long GBP/USD, we are heavily short USD.
        """
        max_corr = self.config.get("max_correlated_positions", 2)
        
        # Basic check: split "EUR/USD" into ["EUR", "USD"]
        parts = symbol.replace("_", "/").replace("-", "/").split("/")
        if len(parts) != 2:
            return True
            
        base, quote = parts[0], parts[1]
        
        base_count = sum(1 for p in open_positions if p.symbol.startswith(base))
        quote_count = sum(1 for p in open_positions if p.symbol.endswith(quote))
        
        if base_count >= max_corr:
            logger.info(f"Max correlation reached for base {base}. Rejecting {symbol}.")
            return False
            
        if quote_count >= max_corr:
            logger.info(f"Max correlation reached for quote {quote}. Rejecting {symbol}.")
            return False
            
        return True
