"""
Autonomous Strategy Self-Improvement Daemon for TradeAgent.
===========================================================
Continuously formulates new institutional trading hypotheses, executes multi-year
walk-forward backtests across market regimes, optimizes hyperparameters, verifies
strict quality gates, and safely promotes vetted models to the live production sniper.

Usage:
  python autonomous_strategy_improver.py --once
  python autonomous_strategy_improver.py --strategy smc_liquidity_sweep --trials 15 --promote
  python autonomous_strategy_improver.py --daemon --interval 3600
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
import argparse
import signal
from datetime import datetime, timezone
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.strategy_optimizer_evaluator import StrategyOptimizerEvaluator
from backtesting.historical_regime_engine import HistoricalRegimeEngine
from core.continuous_learning_logger import ContinuousLearningLogger
from core.sniper_config_manager import SniperConfigManager

running = True

def handle_sigint(sig, frame):
    global running
    print("\n🛑 [SHUTDOWN] Autonomous Strategy Improver gracefully exiting...", flush=True)
    running = False

signal.signal(signal.SIGINT, handle_sigint)


def run_improvement_cycle(
    evaluator: StrategyOptimizerEvaluator,
    logger: ContinuousLearningLogger,
    target_strategies: List[str],
    trials: int = 12,
    auto_promote: bool = True,
) -> Dict[str, Any]:
    """
    Executes one complete continuous self-improvement cycle across target strategies.
    """
    cycle_start = time.time()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("\n" + "=" * 80, flush=True)
    print(f"🚀 [CYCLE START] Autonomous Strategy Improver at {now_str}", flush=True)
    print(f"   Target Strategies: {target_strategies} | Trials/Strategy: {trials}", flush=True)
    print("=" * 80, flush=True)

    summary = {
        "timestamp": now_str,
        "strategies_tested": len(target_strategies),
        "promoted_champions": [],
        "cycle_results": {},
    }

    for strat_name in target_strategies:
        try:
            report = evaluator.optimize_strategy(strategy_name=strat_name, trials=trials)
            summary["cycle_results"][strat_name] = report

            champion = report.get("champion")
            if champion and champion.get("is_vetted"):
                print(f"\n🎉 [QUALITY GATE PASSED] Champion for '{strat_name}': {champion['hypothesis_id']}", flush=True)
                metrics = champion["metrics"]
                print(
                    f"   WR: {metrics['win_rate_pct']:.1f}% | PF: {metrics['profit_factor']:.2f} | "
                    f"EV: +${metrics['expected_value_usd']:.2f} | DD: {metrics['max_drawdown_pct']:.1f}% | "
                    f"Net PnL: +${metrics['net_profit_usd']:.2f}",
                    flush=True
                )

                if auto_promote:
                    print(f"   📢 Promoting champion model to live production engine and logging to journal...", flush=True)
                    success = logger.promote_vetted_strategy(champion)
                    if success:
                        print(f"   ✅ [PROMOTION SUCCESSFUL] Updated pure_gold_config.json & Continuous_Learning_Log.md", flush=True)
                        summary["promoted_champions"].append(champion["hypothesis_id"])
                    else:
                        print(f"   ⚠️ Promotion write encountered an issue.", flush=True)
            else:
                if champion:
                    print(f"  ℹ️ Champion for '{strat_name}' did not meet all strict quality gates (Fit: {champion['fitness']:.2f}). Holding in candidate pool.", flush=True)
                else:
                    print(f"  ℹ️ No viable candidate found for '{strat_name}' in this iteration.", flush=True)

        except Exception as e:
            print(f"❌ Error optimizing strategy {strat_name}: {e}", flush=True)

    duration = round(time.time() - cycle_start, 1)
    print("\n" + "=" * 80, flush=True)
    print(f"🏁 [CYCLE COMPLETE] Finished in {duration}s | Promoted {len(summary['promoted_champions'])} champion models", flush=True)
    print("=" * 80 + "\n", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="TradeAgent Autonomous Strategy Improver Daemon")
    parser.add_argument("--daemon", action="store_true", help="Run indefinitely as a persistent background daemon")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between daemon improvement cycles (default: 3600)")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation/optimization cycle and exit")
    parser.add_argument("--strategy", type=str, default="all", choices=["all", "h4_macro_swing", "smc_liquidity_sweep", "asian_judas_swing", "vwap_khakestar_snapback", "prime_m15_scalp"], help="Target strategy (default: all)")
    parser.add_argument("--trials", type=int, default=10, help="Number of hypothesis mutation trials per strategy (default: 10)")
    parser.add_argument("--no-promote", action="store_true", help="Do not automatically promote vetted models to config")
    args = parser.parse_args()

    print("================================================================================", flush=True)
    print("       🧠 TRADEAGENT AUTONOMOUS STRATEGY SELF-IMPROVEMENT ENGINE 🧠            ", flush=True)
    print("   Persistent Institutional Intelligence, Hypothesis Generation & Promotion    ", flush=True)
    print("   Guarantees: Strict Quality Gates (WR >= 50%, PF >= 1.45, DD <= 15%, EV > 0) ", flush=True)
    print("================================================================================", flush=True)

    config_mgr = SniperConfigManager()
    backtest_engine = HistoricalRegimeEngine()
    evaluator = StrategyOptimizerEvaluator(backtest_engine=backtest_engine)
    logger = ContinuousLearningLogger(config_mgr=config_mgr)

    if args.strategy == "all":
        strategies_to_test = [
            "h4_macro_swing",
            "smc_liquidity_sweep",
            "asian_judas_swing",
            "vwap_khakestar_snapback",
            "prime_m15_scalp",
        ]
    else:
        strategies_to_test = [args.strategy]

    auto_promote = not args.no_promote
    cycle_count = 0

    if args.once or not args.daemon:
        run_improvement_cycle(
            evaluator=evaluator,
            logger=logger,
            target_strategies=strategies_to_test,
            trials=args.trials,
            auto_promote=auto_promote,
        )
        return

    # Daemon Mode
    print(f"🔄 Daemon active. Interval: {args.interval}s. Press Ctrl+C to terminate.", flush=True)
    while running:
        cycle_count += 1
        print(f"\n[DAEMON] Running improvement cycle #{cycle_count}...", flush=True)
        run_improvement_cycle(
            evaluator=evaluator,
            logger=logger,
            target_strategies=strategies_to_test,
            trials=args.trials,
            auto_promote=auto_promote,
        )

        # Sleep in small slices to respond promptly to SIGINT
        sleep_elapsed = 0
        while running and sleep_elapsed < args.interval:
            time.sleep(1.0)
            sleep_elapsed += 1


if __name__ == "__main__":
    main()
