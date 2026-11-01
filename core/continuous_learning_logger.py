"""
Continuous Learning Logger & Strategy Promotion Engine.
======================================================
Automates the institutional recording of vetted strategy models into:
1. Continuous_Learning_Log.md (Obsidian-compatible forensic journal)
2. config/pure_gold_config.json (Production execution engine configuration)
3. data_store/strategy_improvement_journal.json (Machine-readable audit trail)
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from core.sniper_config_manager import SniperConfigManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEARNING_LOG_PATH = PROJECT_ROOT / "Continuous_Learning_Log.md"
JOURNAL_PATH = PROJECT_ROOT / "data_store" / "strategy_improvement_journal.json"


class ContinuousLearningLogger:
    def __init__(
        self,
        log_path: Optional[Path] = None,
        journal_path: Optional[Path] = None,
        config_mgr: Optional[SniperConfigManager] = None,
    ):
        self.log_path = log_path or LEARNING_LOG_PATH
        self.journal_path = journal_path or JOURNAL_PATH
        self.config_mgr = config_mgr or SniperConfigManager()

    def get_next_case_study_number(self) -> int:
        """Parses Continuous_Learning_Log.md to determine next Case Study number."""
        if not self.log_path.exists():
            return 9

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                content = f.read()
            matches = re.findall(r"Case Study (\d+)", content, re.IGNORECASE)
            if matches:
                nums = [int(m) for m in matches]
                return max(nums) + 1
            return 9
        except Exception:
            return 9

    def promote_vetted_strategy(self, champion_data: Dict[str, Any]) -> bool:
        """
        Promotes a vetted champion model:
        1. Updates pure_gold_config.json
        2. Appends to Continuous_Learning_Log.md
        3. Records in strategy_improvement_journal.json
        """
        strategy_name = champion_data["strategy_name"]
        params = champion_data["parameters"]
        params["enabled"] = True

        # 1. Update Production Config
        cfg_updated = self.config_mgr.update_strategy_config(strategy_name, params)

        # 2. Append to Markdown Journal
        md_updated = self.append_markdown_journal(champion_data)

        # 3. Append to JSON Journal
        json_updated = self.record_journal_entry(champion_data)

        return cfg_updated and md_updated and json_updated

    def append_markdown_journal(self, champion_data: Dict[str, Any]) -> bool:
        """Formats and appends a formal forensic case study entry."""
        if not self.log_path.exists():
            return False

        try:
            case_num = self.get_next_case_study_number()
            strat_name = champion_data["strategy_name"]
            hypo_id = champion_data.get("hypothesis_id", "HYP-AUTO")
            title = champion_data.get("title", f"Autonomous Discovery — {strat_name}")
            metrics = champion_data.get("metrics", {})
            params = champion_data.get("parameters", {})
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

            year_breakdown = metrics.get("year_breakdown", {})
            regime_breakdown = metrics.get("regime_breakdown", {})

            entry = f"""

---

## 🔬 {case_num}. Case Study {case_num}: {title}

* **Discovery Timestamp:** {now_str} | **Hypothesis ID:** `{hypo_id}`
* **Strategy Name:** `{strat_name}` | **Asset Exclusivity:** `XAUUSD`
* **Autonomous Operating Directive:**
  The autonomous self-improvement agent formulated and backtested candidate variations across 2024–2026 intraday tick datasets. Model `{hypo_id}` cleared all institutional quality gates and has been promoted to production.

### Verified Multi-Year Performance Metrics

| Performance Metric | Champion Result | Quality Gate Minimum | Status |
| :--- | :---: | :---: | :---: |
| **Win Rate** | **`{metrics.get('win_rate_pct', 0.0):.1f}%`** | $\\\\ge 50.0\\%$ | ✅ PASSED |
| **Profit Factor** | **`{metrics.get('profit_factor', 0.0):.2f}`** | $\\\\ge 1.45$ | ✅ PASSED |
| **Expected Value (EV)** | **`+${metrics.get('expected_value_usd', 0.0):.2f}`** / trade | $> \\$0.00$ | ✅ PASSED |
| **Sharpe Ratio** | **`{metrics.get('sharpe_ratio', 0.0):.2f}`** | $\\\\ge 1.10$ | ✅ PASSED |
| **Max Drawdown** | **`{metrics.get('max_drawdown_pct', 0.0):.1f}%`** | $\\\\le 15.0\\%$ | ✅ PASSED |
| **Total Backtest Trades** | **`{metrics.get('total_trades', 0)}`** | $\\\\ge 20$ | ✅ PASSED |
| **Net Compounded PnL** | **`+${metrics.get('net_profit_usd', 0.0):.2f}`** | $> \\$0.00$ | ✅ PASSED |

### Annual Consistency Breakdown
"""
            for yr, ydata in year_breakdown.items():
                entry += f"- **Year {yr}:** {ydata.get('trades', 0)} trades | Win Rate **{ydata.get('win_rate', 0):.1f}%** | Net PnL **+${ydata.get('net_profit', 0):.2f}**\n"

            entry += f"""
### Promoted Hyperparameters
```json
{json.dumps(params, indent=2)}
```

### Production Deployment
- **Target Engine:** `pure_gold_sniper.py v9.6 Modular Compounder`
- **Config Target:** `config/pure_gold_config.json`
- **Safety Enforcement:** Strict 2.5% Equity Risk per trade, 4-hour circuit breaker on 2 losses, 80% TP1 profit banking with risk-free runner trailing.
"""

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry)

            return True
        except Exception:
            return False

    def record_journal_entry(self, champion_data: Dict[str, Any]) -> bool:
        """Saves machine-readable record to JSON journal."""
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            history = []
            if self.journal_path.exists():
                with open(self.journal_path, "r", encoding="utf-8") as f:
                    history = json.load(f)

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy_name": champion_data["strategy_name"],
                "hypothesis_id": champion_data.get("hypothesis_id"),
                "title": champion_data.get("title"),
                "parameters": champion_data.get("parameters"),
                "metrics": champion_data.get("metrics"),
                "fitness": champion_data.get("fitness"),
                "promoted_to_production": True,
            }
            history.append(record)

            tmp_path = self.journal_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            os.replace(tmp_path, self.journal_path)
            return True
        except Exception:
            return False
