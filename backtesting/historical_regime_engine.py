"""
Multi-Year Multi-Regime Backtesting Engine for Pure Gold (XAUUSD).
==================================================================
Simulates institutional execution dynamics across multiple market regimes:
1. Bull Trend (Institutional Long Flow)
2. Bear Trend (Institutional Short Flow)
3. Volatile Chop / News Spikes
4. Range Consolidation / Low Volatility

Models realistic spreads, dynamic compounding, 2-stage partial profit banking,
and structural trailing stops across 2024 - 2026 data.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from signals.strategies.base_strategy import BaseStrategy, StrategySignal
from signals.khakestar_trex import KhakestarTrexEngine, FTRNode, Pivot
from signals.poursamadi_indicators import PourSamadiToolkit


DATA_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_store"


class HistoricalRegimeEngine:
    def __init__(self, symbol: str = "XAUUSD", initial_balance: float = 500.0):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.df_m15: Optional[pd.DataFrame] = None
        self.df_h4: Optional[pd.DataFrame] = None
        self._all_nodes_by_bar: Dict[int, List[FTRNode]] = {}

    def _init_node_index(self, df: pd.DataFrame) -> None:
        if not self._all_nodes_by_bar:
            try:
                atr_vals = df["atr26"].values
                pivots = KhakestarTrexEngine.identify_pivots(df, atr_vals)
                nodes = KhakestarTrexEngine.identify_ftr_nodes(df, pivots, atr_vals)
                from collections import defaultdict
                node_map = defaultdict(list)
                for node in nodes:
                    node_map[node.created_idx].append(node)
                self._all_nodes_by_bar = dict(node_map)
            except Exception:
                self._all_nodes_by_bar = {}

    def load_or_generate_dataset(self, start_date="2024-01-01", end_date="2026-09-01", n_days: int = 600) -> pd.DataFrame:
        """
        Loads cached multi-year historical dataset or generates high-fidelity intraday bars.
        """
        cache_file = DATA_CACHE_DIR / "gold_m15_multiyear.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 5000:
                    self.df_m15 = df
                    self._init_node_index(df)
                    return df
            except Exception:
                pass

        # Generate realistic multi-year intraday M15 series anchored to gold prices
        print(f"  ⏳ Synthesizing high-resolution multi-year intraday dataset for {self.symbol} ({start_date} to {end_date})...", flush=True)
        np.random.seed(42)

        # Base gold trajectory: 2024 (~$2,050) -> 2025 (~$2,450) -> 2026 (~$4,350)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days

        # Daily anchor points
        dates = [start_dt + timedelta(days=d) for d in range(total_days) if (start_dt + timedelta(days=d)).weekday() < 5]
        n_days_actual = len(dates)

        # Gold price curve simulation
        t = np.linspace(0, 1, n_days_actual)
        trend = 2050.0 + (1300.0 * (t ** 1.3)) + (1000.0 * (t ** 2.2))
        volatility = 18.0 + (35.0 * t)  # Volatility expands as gold surges
        daily_noise = np.random.normal(0, volatility, n_days_actual)
        daily_close = trend + np.cumsum(daily_noise * 0.15)

        bars = []
        for i, dt in enumerate(dates):
            d_close = daily_close[i]
            d_open = daily_close[i-1] if i > 0 else d_close - 5.0
            daily_rng = max(12.0, volatility[i] * np.random.uniform(0.7, 1.6))
            d_high = max(d_open, d_close) + (daily_rng * np.random.uniform(0.2, 0.6))
            d_low = min(d_open, d_close) - (daily_rng * np.random.uniform(0.2, 0.6))

            # 32 M15 bars per trading day (covers Asian, London, and NY Prime)
            n_bars = 32
            steps = np.random.normal(0, daily_rng / np.sqrt(n_bars), n_bars)
            raw = np.cumsum(steps)
            raw = raw - raw[0] + d_open
            r_min, r_max = np.min(raw), np.max(raw)
            if r_max > r_min:
                scaled = d_low + (raw - r_min) * (d_high - d_low) / (r_max - r_min)
            else:
                scaled = raw
            scaled[-1] = d_close

            for b in range(n_bars):
                b_time = dt + timedelta(hours=b * 0.5 + 2) # Start 02:00 UTC
                c = float(scaled[b])
                o = float(scaled[b-1]) if b > 0 else float(d_open)
                noise = abs(np.random.normal(0, 0.45))
                h = max(o, c) + noise
                l = min(o, c) - noise
                vol = int(np.random.uniform(800, 7500))

                bars.append({
                    "time": b_time,
                    "timestamp": b_time,
                    "year": b_time.year,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "tick_volume": vol,
                })

        df_bars = pd.DataFrame(bars)
        # Compute indicators
        c_series = pd.Series(df_bars["close"])
        df_bars["ema9"] = c_series.ewm(span=9).mean()
        df_bars["ema21"] = c_series.ewm(span=21).mean()
        df_bars["ema50"] = c_series.ewm(span=50).mean()
        df_bars["ema200"] = c_series.ewm(span=200).mean()
        df_bars["atr14"] = (df_bars["high"] - df_bars["low"]).rolling(14).mean().bfill()
        df_bars["atr26"] = KhakestarTrexEngine.calculate_atr_steps(df_bars, period=26)

        hlc3 = (df_bars["high"] + df_bars["low"] + df_bars["close"]) / 3.0
        cum_v = np.cumsum(df_bars["tick_volume"])
        cum_pv = np.cumsum(hlc3 * df_bars["tick_volume"])
        df_bars["vwap"] = cum_pv / np.maximum(cum_v, 1e-5)

        rng = np.maximum(df_bars["high"] - df_bars["low"], 1e-5)
        df_bars["clv"] = (df_bars["close"] - df_bars["low"]) / rng

        delta = c_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / np.maximum(loss, 1e-8)
        df_bars["rsi"] = (100 - (100 / (1 + rs))).bfill()

        # Tag Market Regimes
        regimes = []
        for _, row in df_bars.iterrows():
            atr = row["atr14"]
            close = row["close"]
            e50 = row["ema50"]
            e200 = row["ema200"]

            if atr > 4.5:
                regimes.append("VOLATILE_CHOP")
            elif close > e50 and e50 > e200:
                regimes.append("BULL_TREND")
            elif close < e50 and e50 < e200:
                regimes.append("BEAR_TREND")
            else:
                regimes.append("RANGE_CONSOLIDATION")

        df_bars["regime"] = regimes

        try:
            DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df_bars.to_parquet(cache_file)
        except Exception:
            pass

        self.df_m15 = df_bars
        self._init_node_index(df_bars)
        return df_bars

    def run_strategy_backtest(
        self,
        strategy: BaseStrategy,
        params: Optional[Dict[str, Any]] = None,
        spread_usd: float = 0.35,
        risk_pct: float = 0.025,
    ) -> Dict[str, Any]:
        """
        Simulates execution of strategy across the multi-year dataset.
        """
        if self.df_m15 is None:
            self.load_or_generate_dataset()

        df = self.df_m15
        strat_params = dict(strategy.params)
        if params:
            strat_params.update(params)

        tp1_r = strat_params.get("tp1_r_multiple", 1.8)
        tp2_r = strat_params.get("tp2_r_multiple", 3.5)
        bank_frac = strat_params.get("tp1_bank_fraction", 0.80)

        n = len(df)
        trades = []
        equity = self.initial_balance
        peak_equity = equity
        max_dd_pct = 0.0

        i = 50
        while i < n - 35:
            # Build rolling window for market_data
            sub_df = df.iloc[max(0, i - 60):i + 1]
            cur_bar = df.iloc[i]
            cur_price = cur_bar["close"]
            cur_dt = cur_bar["time"]

            # Session simulation
            hour = cur_dt.hour
            is_prime = (7 <= hour <= 11) or (12 <= hour <= 17)
            sess_name = "LONDON_PRIME" if 7 <= hour <= 11 else ("NY_PRIME" if 12 <= hour <= 17 else "OTHER")

            # Mock ADR
            cur_atr = cur_bar["atr14"]
            adr_data = (0.55, 18.0, 28.0)

            # Khakestar nodes calculation for the sub-window
            atr_steps = sub_df["atr26"].values
            ftr_nodes = []
            if self._all_nodes_by_bar:
                for idx in range(max(0, i - 60), i + 1):
                    if idx in self._all_nodes_by_bar:
                        ftr_nodes.extend(self._all_nodes_by_bar[idx])
            else:
                pivots = KhakestarTrexEngine.identify_pivots(sub_df, atr_steps)
                ftr_nodes = KhakestarTrexEngine.identify_ftr_nodes(sub_df, pivots, atr_steps)

            market_data = {
                "m15": {
                    "df": sub_df,
                    "close": sub_df["close"].values,
                    "high": sub_df["high"].values,
                    "low": sub_df["low"].values,
                    "clv": sub_df["clv"].values,
                    "rsi": sub_df["rsi"].values,
                    "atr14": sub_df["atr14"].values,
                    "atr26": atr_steps,
                    "ftr_nodes": ftr_nodes,
                },
                "m5": None,
                "h4": {
                    "df": sub_df,
                    "high": sub_df["high"].values,
                    "low": sub_df["low"].values,
                    "close": sub_df["close"].values,
                    "ema50": sub_df["ema50"].values,
                    "ema200": sub_df["ema200"].values,
                    "atr26": atr_steps,
                    "clv": sub_df["clv"].values,
                    "rsi": sub_df["rsi"].values,
                    "ftr_nodes": ftr_nodes,
                },
                "adr": adr_data,
                "session": (is_prime, sess_name),
                "time": cur_dt,
            }

            sig: Optional[StrategySignal] = strategy.evaluate(
                market_data=market_data,
                spot_price=cur_price,
                equity=equity,
                config=strat_params,
            )

            if sig:
                # Trade Entry
                entry = sig.entry + (spread_usd if sig.signal == "BUY" else -spread_usd)
                sl = sig.sl
                risk_dist = abs(entry - sl)
                if risk_dist <= 0:
                    risk_dist = 2.50

                target_tp1 = entry + (risk_dist * tp1_r) if sig.signal == "BUY" else entry - (risk_dist * tp1_r)
                target_tp2 = entry + (risk_dist * tp2_r) if sig.signal == "BUY" else entry - (risk_dist * tp2_r)

                # Compounded lot calculation
                risk_usd = equity * risk_pct
                lot = round(risk_usd / (risk_dist * 100.0), 2)
                lot = max(0.01, min(lot, 1.00))
                lot_tp1 = round(lot * bank_frac, 2)
                lot_runner = round(lot - lot_tp1, 2)
                if lot_runner <= 0:
                    lot_runner = 0.01

                # Simulate forward bars until exit (max 28 bars = 7 hours)
                won = False
                tp1_hit = False
                pnl_dollars = 0.0
                exit_idx = i + 1

                for k in range(i + 1, min(i + 29, n)):
                    k_high = df.iloc[k]["high"]
                    k_low = df.iloc[k]["low"]

                    if sig.signal == "BUY":
                        if not tp1_hit:
                            # Check SL before TP1
                            if k_low <= sl:
                                pnl_dollars = -risk_usd
                                won = False
                                exit_idx = k
                                break
                            # Check TP1
                            if k_high >= target_tp1:
                                tp1_hit = True
                                sl = entry + 0.30 # Move SL to risk-free
                                banked = lot_tp1 * 100.0 * (target_tp1 - entry)
                                pnl_dollars += banked
                        else:
                            # Check TP2 runner
                            if k_high >= target_tp2:
                                runner_pnl = lot_runner * 100.0 * (target_tp2 - entry)
                                pnl_dollars += runner_pnl
                                won = True
                                exit_idx = k
                                break
                            # Check Risk-Free / Breakeven SL
                            if k_low <= sl:
                                runner_pnl = lot_runner * 100.0 * (sl - entry)
                                pnl_dollars += runner_pnl
                                won = pnl_dollars > 0
                                exit_idx = k
                                break
                    else: # SELL
                        if not tp1_hit:
                            # Check SL before TP1
                            if k_high >= sl:
                                pnl_dollars = -risk_usd
                                won = False
                                exit_idx = k
                                break
                            if k_low <= target_tp1:
                                tp1_hit = True
                                sl = entry - 0.30 # Move SL to risk-free
                                banked = lot_tp1 * 100.0 * (entry - target_tp1)
                                pnl_dollars += banked
                        else:
                            # Check TP2 runner
                            if k_low <= target_tp2:
                                runner_pnl = lot_runner * 100.0 * (entry - target_tp2)
                                pnl_dollars += runner_pnl
                                won = True
                                exit_idx = k
                                break
                            # Check Risk-Free / Breakeven SL
                            if k_high >= sl:
                                runner_pnl = lot_runner * 100.0 * (entry - sl)
                                pnl_dollars += runner_pnl
                                won = pnl_dollars > 0
                                exit_idx = k
                                break
                else:
                    # Timeout exit at bar close
                    last_c = df.iloc[min(i + 28, n - 1)]["close"]
                    remaining_lot = lot_runner if tp1_hit else lot
                    diff = (last_c - entry) if sig.signal == "BUY" else (entry - last_c)
                    pnl_dollars += remaining_lot * 100.0 * diff
                    won = pnl_dollars > 0
                    exit_idx = min(i + 28, n - 1)

                equity += pnl_dollars
                peak_equity = max(peak_equity, equity)
                cur_dd = (peak_equity - equity) / peak_equity * 100.0
                max_dd_pct = max(max_dd_pct, cur_dd)

                trades.append({
                    "entry_time": cur_dt,
                    "year": cur_dt.year,
                    "regime": cur_bar.get("regime", "ALL"),
                    "signal": sig.signal,
                    "entry": entry,
                    "pnl": round(pnl_dollars, 2),
                    "won": won,
                    "tp1_hit": tp1_hit,
                    "equity": round(equity, 2),
                })

                # Anti-chop jump
                i = exit_idx + 4
                continue

            i += 1

        return self._compute_performance_metrics(trades, strategy.name, strat_params, max_dd_pct, equity)

    def _compute_performance_metrics(
        self,
        trades: List[Dict[str, Any]],
        strategy_name: str,
        params: Dict[str, Any],
        max_dd_pct: float,
        final_equity: float,
    ) -> Dict[str, Any]:
        if not trades:
            return {
                "strategy": strategy_name,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "expected_value_usd": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "net_profit_usd": 0.0,
                "final_equity": self.initial_balance,
                "parameters": params,
                "year_breakdown": {},
                "regime_breakdown": {},
            }

        df_t = pd.DataFrame(trades)
        total_trades = len(df_t)
        wins = len(df_t[df_t["won"] == True])
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100.0

        pnl_vals = df_t["pnl"].values
        gross_profit = float(np.sum(pnl_vals[pnl_vals > 0]))
        gross_loss = float(abs(np.sum(pnl_vals[pnl_vals < 0])))
        profit_factor = round(gross_profit / max(gross_loss, 1e-4), 2)
        net_profit = round(float(np.sum(pnl_vals)), 2)
        expected_val = round(net_profit / total_trades, 2)

        sharpe = 0.0
        if len(pnl_vals) > 5 and np.std(pnl_vals) > 0:
            sharpe = round(float(np.mean(pnl_vals) / np.std(pnl_vals) * np.sqrt(min(total_trades, 150))), 2)

        # Year breakdown
        year_breakdown = {}
        for yr, group in df_t.groupby("year"):
            yr_total = len(group)
            yr_wins = len(group[group["won"] == True])
            yr_wr = round((yr_wins / yr_total) * 100.0, 1)
            yr_pnl = round(float(group["pnl"].sum()), 2)
            year_breakdown[int(yr)] = {"trades": yr_total, "win_rate": yr_wr, "net_profit": yr_pnl}

        # Regime breakdown
        regime_breakdown = {}
        for rg, group in df_t.groupby("regime"):
            rg_total = len(group)
            rg_wins = len(group[group["won"] == True])
            rg_wr = round((rg_wins / rg_total) * 100.0, 1)
            rg_pnl = round(float(group["pnl"].sum()), 2)
            regime_breakdown[str(rg)] = {"trades": rg_total, "win_rate": rg_wr, "net_profit": rg_pnl}

        return {
            "strategy": strategy_name,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": profit_factor,
            "expected_value_usd": expected_val,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": round(max_dd_pct, 2),
            "net_profit_usd": net_profit,
            "final_equity": round(final_equity, 2),
            "parameters": params,
            "year_breakdown": year_breakdown,
            "regime_breakdown": regime_breakdown,
        }
