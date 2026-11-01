"""
Comprehensive Test Suite for Autonomous Self-Improvement System and Modular Sniper Engine.
========================================================================================
Covers:
1. Config Manager & Hard Safety Clamping
2. State Manager Persistence & Broker Reconciliation
3. Strategy Modules (SMC Sweep, Judas Swing, VWAP Snapback, Khakestar FTR)
4. Multi-Year Multi-Regime Historical Backtesting Engine
5. Hypothesis Engine & Optimizer Quality Gates
6. Continuous Learning Journal Logging & Promotion
7. End-to-End Improvement Cycle Execution
"""

import os
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.sniper_config_manager import SniperConfigManager
from core.sniper_state_manager import SniperStateManager
from core.continuous_learning_logger import ContinuousLearningLogger
from signals.strategies import (
    strategy_registry,
    StrategySignal,
    SMCLiquiditySweepStrategy,
    AsianJudasSwingStrategy,
    VWAPKhakestarSnapbackStrategy,
    MacroSwingStrategy,
    PrimeScalpStrategy,
)
from agents.hypothesis_engine import HypothesisEngine, TradingHypothesis
from backtesting.historical_regime_engine import HistoricalRegimeEngine
from agents.strategy_optimizer_evaluator import StrategyOptimizerEvaluator
from autonomous_strategy_improver import run_improvement_cycle


class TestSniperConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fallback_and_safety_clamping(self):
        mgr = SniperConfigManager(config_path=self.config_path)
        self.assertTrue(self.config_path.exists())
        self.assertEqual(mgr.get("symbol"), "XAUUSD")

        # Test safety clamping: attempt to inject dangerous 50% risk
        unsafe_cfg = {
            "symbol": "XAUUSD",
            "risk": {
                "risk_percent": 0.50, # 50% risk -> must be clamped to 5.0%
                "max_lot": 25.0,      # 25 lots -> must be clamped to 2.0
            }
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(unsafe_cfg, f)

        mgr.load(force=True)
        self.assertLessEqual(mgr.risk["risk_percent"], 0.05)
        self.assertLessEqual(mgr.risk["max_lot"], 2.00)

    def test_update_and_hot_reload(self):
        mgr = SniperConfigManager(config_path=self.config_path)
        mgr.update_strategy_config("smc_liquidity_sweep", {"min_sweep_dist": 0.45})
        self.assertEqual(mgr.get_strategy_config("smc_liquidity_sweep")["min_sweep_dist"], 0.45)


class TestSniperStateManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.state_path = Path(self.temp_dir) / "test_state.json"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_and_reconciliation(self):
        state = SniperStateManager(state_path=self.state_path)
        ticket = 999123
        state.register_open_trade(ticket, {
            "lane": "SCALP",
            "entry": 4350.0,
            "sl": 4347.5,
            "tp1": 4354.5,
            "tp2": 4359.0,
            "risk_dist": 2.5,
        })
        self.assertIn(ticket, state.active_tracked_tickets)

        # Reload from disk
        state2 = SniperStateManager(state_path=self.state_path)
        self.assertIn(ticket, state2.active_tracked_tickets)
        self.assertEqual(state2.active_tracked_tickets[ticket]["entry"], 4350.0)

        # Reconcile: broker now has empty positions -> ticket must be closed
        closed = state2.reconcile_with_live_positions([])
        self.assertIn(ticket, closed)
        self.assertNotIn(ticket, state2.active_tracked_tickets)
        self.assertGreater(state2.last_micro_exit_time, 0.0)


class TestStrategyModules(unittest.TestCase):
    def _create_mock_m15_data(self, n_bars=40, base_price=4350.0):
        dates = [datetime.now(timezone.utc) - timedelta(minutes=15 * (n_bars - i)) for i in range(n_bars)]
        closes = [base_price + (np.sin(i / 3.0) * 5.0) for i in range(n_bars)]
        highs = [c + 1.5 for c in closes]
        lows = [c - 1.5 for c in closes]
        opens = [c - 0.2 for c in closes]

        df = pd.DataFrame({
            "time": dates,
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [2000] * n_bars,
        })
        df["clv"] = 0.65
        df["rsi"] = 45.0
        df["atr14"] = 2.5
        df["atr26"] = 2.8

        return {
            "df": df,
            "close": df["close"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "clv": df["clv"].values,
            "rsi": df["rsi"].values,
            "atr14": df["atr14"].values,
            "atr26": df["atr26"].values,
            "ftr_nodes": [],
        }

    def test_smc_liquidity_sweep_evaluation(self):
        m15 = self._create_mock_m15_data(n_bars=35, base_price=4350.0)
        # Create an artificial liquidity sweep:
        # Swing low at bar 10 is 4340.0. Current bar dips to 4339.2 (sweep of 0.8) and closes back at 4341.0 with high CLV (0.75)
        m15["low"][10] = 4340.0
        m15["low"][-1] = 4339.2
        m15["close"][-1] = 4341.2
        m15["clv"][-1] = 0.75

        strat = SMCLiquiditySweepStrategy()
        market_data = {
            "m15": m15,
            "adr": (0.50, 15.0, 30.0),
            "session": (True, "LONDON_PRIME"),
            "time": datetime.now(timezone.utc),
        }

        signal = strat.evaluate(market_data, spot_price=4341.2)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal, "BUY")
        self.assertGreater(signal.tp1, signal.entry)
        self.assertLess(signal.sl, signal.entry)

    def test_asian_judas_swing_evaluation(self):
        # Create dataset starting at 00:00 UTC up to 08:00 UTC (33 bars: index 0 to 32)
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        n_bars = 33
        dates = [today + timedelta(minutes=15 * i) for i in range(n_bars)] # 00:00 to 08:00
        closes = [4350.0 + (i * 0.1) for i in range(n_bars)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]

        df = pd.DataFrame({
            "time": dates,
            "timestamp": dates,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [1500] * n_bars,
        })
        df["clv"] = 0.50
        df["atr14"] = 2.0
        df["atr26"] = 2.5

        # Asian low (00:00-06:30) is 4349.0.
        # At 08:00 (last bar, index 32), price sweeps down to 4347.8 (1.2 USD sweep) and reclaims at 4349.6 with CLV 0.72
        df.loc[32, "low"] = 4347.8
        df.loc[32, "close"] = 4349.6
        df.loc[32, "clv"] = 0.72

        m15 = {
            "df": df,
            "close": df["close"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "clv": df["clv"].values,
            "atr14": df["atr14"].values,
            "atr26": df["atr26"].values,
        }

        strat = AsianJudasSwingStrategy()
        market_data = {
            "m15": m15,
            "adr": (0.45, 12.0, 28.0),
            "session": (True, "LONDON_PRIME"),
            "time": df["time"].iloc[-1],
        }

        signal = strat.evaluate(market_data, spot_price=4349.6)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal, "BUY")
        self.assertGreater(signal.tp1, signal.entry)
        self.assertLess(signal.sl, signal.entry)

    def test_vwap_snapback_evaluation(self):
        m15 = self._create_mock_m15_data(n_bars=35, base_price=4350.0)
        strat = VWAPKhakestarSnapbackStrategy()
        # Set current price stretched below lower VWAP band with bullish CLV
        m15["low"][-1] = 4336.0
        m15["close"][-1] = 4338.0
        m15["clv"][-1] = 0.70
        m15["rsi"][-1] = 28.0
        spot_price = 4338.0  # Deep below mock VWAP (~4350)
        market_data = {
            "m15": m15,
            "adr": (0.45, 12.0, 28.0),
            "session": (True, "LONDON_PRIME"),
            "time": datetime.now(timezone.utc),
        }
        sig = strat.evaluate(market_data, spot_price=spot_price)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.signal, "BUY")
        self.assertGreater(sig.tp1, sig.entry)
        self.assertLess(sig.sl, sig.entry)

    def test_macro_swing_evaluation(self):
        m15 = self._create_mock_m15_data(n_bars=45, base_price=4350.0)
        strat = MacroSwingStrategy()
        # Mock H4 with EMA50 > EMA200 and bullish CLV
        h4_df = m15["df"].copy()
        h4_df["ema50"] = 4345.0
        h4_df["ema200"] = 4320.0
        from signals.khakestar_trex import FTRNode
        node = FTRNode(level_broken=4348.0, node_high=4352.0, node_low=4346.0, direction="BULL", created_idx=20)
        market_data = {
            "h4": {
                "df": h4_df,
                "high": h4_df["high"].values,
                "low": h4_df["low"].values,
                "close": h4_df["close"].values,
                "ema50": np.full(len(h4_df), 4345.0),
                "ema200": np.full(len(h4_df), 4320.0),
                "atr26": np.full(len(h4_df), 3.5),
                "clv": np.full(len(h4_df), 0.70),
                "rsi": np.full(len(h4_df), 50.0),
                "ftr_nodes": [node],
            },
            "adr": (0.50, 15.0, 30.0),
            "session": (True, "LONDON_PRIME"),
            "time": datetime.now(timezone.utc),
        }
        sig = strat.evaluate(market_data, spot_price=4350.0)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.signal, "BUY")
        self.assertGreater(sig.tp1, sig.entry)
        self.assertLess(sig.sl, sig.entry)

    def test_prime_scalp_evaluation(self):
        m15 = self._create_mock_m15_data(n_bars=30, base_price=4350.0)
        strat = PrimeScalpStrategy()
        from signals.khakestar_trex import FTRNode
        node = FTRNode(level_broken=4348.0, node_high=4352.0, node_low=4347.0, direction="BULL", created_idx=15)
        m15["ftr_nodes"] = [node]
        m15["clv"][-1] = 0.68
        m15["rsi"][-1] = 48.0
        market_data = {
            "m15": m15,
            "m5": None,
            "h4": None,
            "adr": (0.40, 10.0, 25.0),
            "session": (True, "LONDON_PRIME"),
            "time": datetime.now(timezone.utc),
        }
        sig = strat.evaluate(market_data, spot_price=4350.0)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.signal, "BUY")


class TestHistoricalRegimeEngine(unittest.TestCase):
    def test_backtest_execution(self):
        engine = HistoricalRegimeEngine(initial_balance=500.0)
        strat = strategy_registry.get("smc_liquidity_sweep")
        self.assertIsNotNone(strat)

        res = engine.run_strategy_backtest(strat)
        self.assertIn("total_trades", res)
        self.assertIn("win_rate_pct", res)
        self.assertIn("profit_factor", res)
        self.assertIn("expected_value_usd", res)
        self.assertIn("max_drawdown_pct", res)
        self.assertGreater(res["total_trades"], 0)
        self.assertGreater(res["final_equity"], 0)

    def test_macro_swing_backtest_no_key_error(self):
        engine = HistoricalRegimeEngine(initial_balance=500.0)
        strat = strategy_registry.get("h4_macro_swing")
        self.assertIsNotNone(strat)
        # Verifies that KeyError: 'high' is completely eliminated
        res = engine.run_strategy_backtest(strat)
        self.assertGreater(res["total_trades"], 0)

    def test_two_stage_profit_banking_breakeven_pnl(self):
        engine = HistoricalRegimeEngine(initial_balance=500.0)
        # Verify that TP1 banking followed by breakeven stop out does not overwrite PnL as -risk_usd
        strat = strategy_registry.get("smc_liquidity_sweep")
        res = engine.run_strategy_backtest(strat)
        # With the fix, EV and PF should be positive and robust
        self.assertGreater(res["profit_factor"], 1.0)
        self.assertGreater(res["expected_value_usd"], 0.0)


class TestHypothesisAndEvaluator(unittest.TestCase):
    def test_hypothesis_generation(self):
        engine = HypothesisEngine()
        hypos = engine.generate_candidate_hypotheses(count_per_strategy=2, target_strategy="smc_liquidity_sweep")
        self.assertEqual(len(hypos), 2)
        self.assertEqual(hypos[0].strategy_name, "smc_liquidity_sweep")

    def test_quality_gate_evaluation(self):
        evaluator = StrategyOptimizerEvaluator()
        good_metrics = {
            "win_rate_pct": 62.5,
            "profit_factor": 1.85,
            "expected_value_usd": 3.40,
            "max_drawdown_pct": 8.5,
            "total_trades": 35,
            "sharpe_ratio": 1.65,
            "year_breakdown": {
                2024: {"net_profit": 55.0},
                2025: {"net_profit": 80.0},
                2026: {"net_profit": 40.0},
            }
        }
        self.assertTrue(evaluator.check_quality_gates(good_metrics))

        bad_metrics = {
            "win_rate_pct": 35.0,
            "profit_factor": 0.85,
            "expected_value_usd": -2.10,
            "max_drawdown_pct": 28.0,
            "total_trades": 15,
            "sharpe_ratio": 0.40,
        }
        self.assertFalse(evaluator.check_quality_gates(bad_metrics))


class TestContinuousLearningLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = Path(self.temp_dir) / "test_Continuous_Learning_Log.md"
        self.journal_path = Path(self.temp_dir) / "test_journal.json"
        self.config_path = Path(self.temp_dir) / "test_config.json"

        # Initialize mock log with Case Study 8
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# Learning Log\n\n## 🔬 9. Case Study 8: The 1-Year Optimization Lab\n\n")

        self.cfg_mgr = SniperConfigManager(config_path=self.config_path)
        self.logger = ContinuousLearningLogger(
            log_path=self.log_path,
            journal_path=self.journal_path,
            config_mgr=self.cfg_mgr,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_promotion_and_case_study_parsing(self):
        next_case = self.logger.get_next_case_study_number()
        self.assertEqual(next_case, 9)

        champion_data = {
            "strategy_name": "smc_liquidity_sweep",
            "hypothesis_id": "HYP-SMC-9999",
            "title": "SMC Liquidity Sweep Optimization",
            "parameters": {
                "lookback_bars": 24,
                "min_sweep_dist": 0.25,
                "max_sweep_dist": 2.20,
                "tp1_r_multiple": 1.8,
                "tp2_r_multiple": 3.5,
            },
            "metrics": {
                "win_rate_pct": 60.5,
                "profit_factor": 1.92,
                "expected_value_usd": 4.10,
                "sharpe_ratio": 1.70,
                "max_drawdown_pct": 9.2,
                "total_trades": 42,
                "net_profit_usd": 172.20,
                "year_breakdown": {
                    2024: {"trades": 14, "win_rate": 57.1, "net_profit": 48.0},
                    2025: {"trades": 18, "win_rate": 61.1, "net_profit": 84.0},
                    2026: {"trades": 10, "win_rate": 60.0, "net_profit": 40.2},
                }
            }
        }

        success = self.logger.promote_vetted_strategy(champion_data)
        self.assertTrue(success)

        # Verify log has Case Study 9
        with open(self.log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Case Study 9", content)
        self.assertIn("HYP-SMC-9999", content)

        # Verify config updated
        self.assertEqual(self.cfg_mgr.get_strategy_config("smc_liquidity_sweep")["lookback_bars"], 24)
        self.assertTrue(self.cfg_mgr.get_strategy_config("smc_liquidity_sweep")["enabled"])

        # Verify journal JSON
        self.assertTrue(self.journal_path.exists())
        with open(self.journal_path, "r", encoding="utf-8") as f:
            journal = json.load(f)
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["hypothesis_id"], "HYP-SMC-9999")


class TestAutonomousImproverCycle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = Path(self.temp_dir) / "test_Continuous_Learning_Log.md"
        self.journal_path = Path(self.temp_dir) / "test_journal.json"
        self.config_path = Path(self.temp_dir) / "test_config.json"

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# Learning Log\n\n## 🔬 9. Case Study 8: Baseline\n\n")

        self.cfg_mgr = SniperConfigManager(config_path=self.config_path)
        self.logger = ContinuousLearningLogger(
            log_path=self.log_path,
            journal_path=self.journal_path,
            config_mgr=self.cfg_mgr,
        )
        self.engine = HistoricalRegimeEngine(initial_balance=500.0)
        self.evaluator = StrategyOptimizerEvaluator(backtest_engine=self.engine)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_improvement_cycle_end_to_end(self):
        summary = run_improvement_cycle(
            evaluator=self.evaluator,
            logger=self.logger,
            target_strategies=["smc_liquidity_sweep"],
            trials=2,
            auto_promote=False,
        )
        self.assertIn("timestamp", summary)
        self.assertEqual(summary["strategies_tested"], 1)
        self.assertIn("smc_liquidity_sweep", summary["cycle_results"])
        report = summary["cycle_results"]["smc_liquidity_sweep"]
        self.assertEqual(report["trials_run"], 2)


if __name__ == "__main__":
    unittest.main()
